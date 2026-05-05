# import json
# import torch
# import torch.nn.functional as F
# from typing import Optional, Dict, List, Tuple
# from train.Trainer import Trainer


# def build_relation_type_mapping(
#     kg_gdp,
#     onto_json_path: str,
#     onto_gdp
# ) -> Dict[int, List[Tuple[int, int]]]:
#     """
#     Build mapping:
#         relation_id (KG) -> [(src_type_idx, dst_type_idx), ...]

#     Only relations whose predicate name appears in BOTH the KG and the ontology JSON
#     are included.
#     """
#     with open(onto_json_path, "r") as f:
#         onto_json = json.load(f)

#     valid_pairs_per_type: Dict[int, List[Tuple[int, int]]] = {}
#     skipped_relations = []
#     skipped_types = []

#     for predicate_str, onto_rel_data in onto_json.items():
#         if predicate_str not in kg_gdp.predicate_to_id:
#             skipped_relations.append(predicate_str)
#             continue

#         relation_id = kg_gdp.predicate_to_id[predicate_str]
#         pairs: List[Tuple[int, int]] = []

#         for src_type_name, _, dst_type_name in onto_rel_data["triplets"]:
#             if src_type_name not in onto_gdp.nodes_index:
#                 skipped_types.append(src_type_name)
#                 continue
#             if dst_type_name not in onto_gdp.nodes_index:
#                 skipped_types.append(dst_type_name)
#                 continue

#             src_type_idx = onto_gdp.nodes_index[src_type_name]
#             dst_type_idx = onto_gdp.nodes_index[dst_type_name]
#             pairs.append((src_type_idx, dst_type_idx))

#         if pairs:
#             valid_pairs_per_type[relation_id] = pairs

#     mapped = len(valid_pairs_per_type)
#     total_kg = len(kg_gdp.predicate_to_id)
#     print(
#         f"[Mapping] {mapped}/{total_kg} KG relations mapped to ontology "
#         f"({100 * mapped / total_kg:.1f}%)"
#     )

#     if skipped_relations:
#         print(
#             f"[Mapping] {len(skipped_relations)} relations not found in onto: "
#             f"{skipped_relations[:5]}{'...' if len(skipped_relations) > 5 else ''}"
#         )

#     if skipped_types:
#         unique_skipped = list(set(skipped_types))
#         print(
#             f"[Mapping] {len(unique_skipped)} type nodes not found in onto graph: "
#             f"{unique_skipped[:5]}{'...' if len(unique_skipped) > 5 else ''}"
#         )

#     return valid_pairs_per_type


# class StableOntologyTrainer(Trainer):
#     def __init__(
#         self,
#         ontology_data,
#         lambda_type: float = 0.5,
#         valid_pairs_per_type: Optional[Dict[int, List[Tuple[int, int]]]] = None,
#         **kwargs
#     ):
#         """
#         Stable version of OntologyTrainer.

#         Same signature as OntologyTrainer:
#         - ontology_data
#         - lambda_type
#         - valid_pairs_per_type
#         - **kwargs passed to Trainer
#         """
#         super().__init__(**kwargs)

#         self.lambda_type = lambda_type
#         self.valid_pairs_per_type = valid_pairs_per_type or {}
#         self.ontology_data = ontology_data.to(self.device)

#         # Stability parameters
#         self.temperature = 0.7
#         self.eps = 1e-8
#         self.min_direction_norm = 1e-6
#         self.warmup_epochs = 30
#         self.grad_clip_norm = 2.0
#         self.current_epoch_idx = 0

#         # NEW: loss history
#         self.loss_history = {
#             "task_loss": [],
#             "onto_loss": [],
#             "total_loss": [],
#             "lambda": []
#         }

#         # Cache tensors for faster indexing
#         self.valid_pairs_tensor: Dict[int, torch.Tensor] = {}
#         for rel, pairs in self.valid_pairs_per_type.items():
#             if len(pairs) > 0:
#                 self.valid_pairs_tensor[rel] = torch.tensor(
#                     pairs, dtype=torch.long, device=self.device
#                 )

#     def _get_current_lambda(self) -> float:
#         if self.warmup_epochs <= 0:
#             return self.lambda_type
#         alpha = min(1.0, (self.current_epoch_idx + 1) / self.warmup_epochs)
#         return self.lambda_type * alpha

#     def _compute_type_constraint_loss(
#         self,
#         kg_out: torch.Tensor,
#         onto_out: torch.Tensor,
#         batch
#     ) -> torch.Tensor:
#         """
#         Stable ontology constraint loss.
#         """
#         if not self.valid_pairs_tensor:
#             return torch.tensor(0.0, device=self.device)

#         edge_index = getattr(batch, "edge_index", None)
#         edge_type = getattr(batch, "edge_type", None)

#         if edge_index is None or edge_type is None:
#             return torch.tensor(0.0, device=self.device)

#         relation_losses = []

#         unique_relations = torch.unique(edge_type)

#         for rel_tensor in unique_relations:
#             rel = rel_tensor.item()

#             if rel not in self.valid_pairs_tensor:
#                 continue

#             rel_mask = (edge_type == rel)
#             if rel_mask.sum() == 0:
#                 continue

#             rel_edges = edge_index[:, rel_mask]
#             src_idx = rel_edges[0]
#             dst_idx = rel_edges[1]

#             # KG directions
#             kg_dirs = kg_out[dst_idx] - kg_out[src_idx]
#             kg_norms = torch.norm(kg_dirs, dim=1)
#             valid_kg = kg_norms > self.min_direction_norm

#             if valid_kg.sum() == 0:
#                 continue

#             kg_dirs = kg_dirs[valid_kg]
#             kg_dirs = F.normalize(kg_dirs, p=2, dim=1, eps=self.eps)

#             # Ontology directions
#             pairs = self.valid_pairs_tensor[rel]
#             src_type_idx = pairs[:, 0]
#             dst_type_idx = pairs[:, 1]

#             onto_dirs = onto_out[dst_type_idx] - onto_out[src_type_idx]
#             onto_norms = torch.norm(onto_dirs, dim=1)
#             valid_onto = onto_norms > self.min_direction_norm

#             if valid_onto.sum() == 0:
#                 continue

#             onto_dirs = onto_dirs[valid_onto]
#             onto_dirs = F.normalize(onto_dirs, p=2, dim=1, eps=self.eps)

#             # Similarity matrix
#             sims = kg_dirs @ onto_dirs.T

#             # Soft matching
#             weights = F.softmax(sims / self.temperature, dim=1)
#             soft_targets = weights @ onto_dirs
#             soft_targets = F.normalize(soft_targets, p=2, dim=1, eps=self.eps)

#             cos = F.cosine_similarity(kg_dirs, soft_targets, dim=1, eps=self.eps)
#             rel_loss = 1.0 - cos

#             relation_losses.append(rel_loss.mean())

#         if not relation_losses:
#             return torch.tensor(0.0, device=self.device)

#         return torch.stack(relation_losses).mean()

#     def _train_epoch(self, train_loader) -> float:
#         """
#         Train for one epoch:
#             loss = task_loss + lambda(t) * onto_loss
#         """
#         self.model.train()

#         total_loss_epoch = 0.0
#         total_task_loss_epoch = 0.0
#         total_onto_loss_epoch = 0.0
#         num_batches = 0

#         current_lambda = self._get_current_lambda()

#         for batch in train_loader:
#             batch = batch.to(self.device)
#             self.optimizer.zero_grad()

#             # KG forward
#             kg_out = self._forward_pass(batch)

#             # Ontology forward
#             onto_out = self.model(
#                 x=self.ontology_data.x,
#                 edge_index=self.ontology_data.edge_index,
#                 edge_type=getattr(self.ontology_data, "edge_type", None),
#                 edge_weight=getattr(self.ontology_data, "edge_weight", None),
#                 edge_attr=getattr(self.ontology_data, "edge_attr", None)
#             )

#             # Task loss
#             mask_input_id = torch.isin(batch.n_id, batch.input_id)
#             mask = mask_input_id & batch.train_mask & (batch.y >= 0)

#             if mask.sum() == 0:
#                 continue

#             task_loss = self.criterion(kg_out[mask], batch.y[mask])

#             # Ontology loss
#             onto_loss = self._compute_type_constraint_loss(
#                 kg_out,
#                 onto_out,
#                 batch
#             )

#             loss = task_loss + current_lambda * onto_loss
#             loss.backward()

#             torch.nn.utils.clip_grad_norm_(
#                 self.model.parameters(),
#                 self.grad_clip_norm
#             )
#             self.optimizer.step()

#             total_task_loss_epoch += task_loss.item()
#             total_onto_loss_epoch += onto_loss.item()
#             total_loss_epoch += loss.item()
#             num_batches += 1

#         self.current_epoch_idx += 1

#         if num_batches == 0:
#             avg_task_loss = 0.0
#             avg_onto_loss = 0.0
#             avg_total_loss = 0.0
#         else:
#             avg_task_loss = total_task_loss_epoch / num_batches
#             avg_onto_loss = total_onto_loss_epoch / num_batches
#             avg_total_loss = total_loss_epoch / num_batches

#         # Save history
#         self.loss_history["task_loss"].append(avg_task_loss)
#         self.loss_history["onto_loss"].append(avg_onto_loss)
#         self.loss_history["total_loss"].append(avg_total_loss)
#         self.loss_history["lambda"].append(current_lambda)

#         return avg_total_loss

#     def train_with_display_matrix(
#         self,
#         train_loader,
#         val_loader,
#         test_loader,
#         onto_type_names: List[str],
#         gdp,
#         epochs: int = 100,
#         patience: int = 100,
#         verbose: bool = True,
#         eval_every: int = 1,
#         save_best_model: bool = True,
#         onto_sim_save_path: Optional[str] = "onto_type_similarity.png",
#         loss_plot_path: Optional[str] = "training_losses.png"
#     ) -> Dict:
#         """
#         Same public API as OntologyTrainer.
#         """
#         results = self.train(
#             train_loader=train_loader,
#             val_loader=val_loader,
#             test_loader=test_loader,
#             epochs=epochs,
#             patience=patience,
#             verbose=verbose,
#             eval_every=eval_every,
#             save_best_model=save_best_model
#         )

#         import matplotlib.pyplot as plt
#         import seaborn as sns

#         # ==================================================
#         # Plot losses
#         # ==================================================
#         fig, ax = plt.subplots(figsize=(10, 6))

#         epochs_range = range(1, len(self.loss_history["total_loss"]) + 1)

#         ax.plot(
#             epochs_range,
#             self.loss_history["task_loss"],
#             label="Task loss",
#             linewidth=2
#         )
#         ax.plot(
#             epochs_range,
#             self.loss_history["onto_loss"],
#             label="Ontology loss",
#             linewidth=2
#         )
#         ax.plot(
#             epochs_range,
#             self.loss_history["total_loss"],
#             label="Total loss",
#             linewidth=2
#         )

#         ax2 = ax.twinx()
#         ax2.plot(
#             epochs_range,
#             self.loss_history["lambda"],
#             label="Lambda",
#             linestyle="--",
#             linewidth=2
#         )
#         ax2.set_ylabel("Lambda")

#         lines1, labels1 = ax.get_legend_handles_labels()
#         lines2, labels2 = ax2.get_legend_handles_labels()
#         ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

#         ax.set_xlabel("Epoch")
#         ax.set_ylabel("Loss")
#         ax.set_title("Training losses and lambda over epochs")
#         ax.grid(True, alpha=0.3)

#         plt.tight_layout()

#         if loss_plot_path:
#             plt.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
#             if verbose:
#                 print(f"  Loss curves saved -> {loss_plot_path}")

#         plt.show()

#         # ==================================================
#         # Ontology similarity matrix
#         # ==================================================
#         if verbose:
#             print("\nComputing ontology type similarity matrix...")

#         onto_type_indices = [
#             gdp.nodes_index[name]
#             for name in onto_type_names
#             if name in gdp.nodes_index
#         ]
#         type_labels = [
#             name for name in onto_type_names
#             if name in gdp.nodes_index
#         ]

#         missing = [n for n in onto_type_names if n not in gdp.nodes_index]
#         if missing and verbose:
#             print(f"  [WARN] Types not found in ontology graph: {missing}")

#         if not onto_type_indices:
#             print("  [WARN] No valid type indices found, skipping similarity matrix.")
#             return results

#         self.model.eval()
#         with torch.no_grad():
#             onto_out = self.model(
#                 x=self.ontology_data.x,
#                 edge_index=self.ontology_data.edge_index,
#                 edge_type=getattr(self.ontology_data, "edge_type", None),
#                 edge_weight=getattr(self.ontology_data, "edge_weight", None),
#                 edge_attr=getattr(self.ontology_data, "edge_attr", None)
#             )

#         type_embs = onto_out[onto_type_indices]
#         normed = F.normalize(type_embs, p=2, dim=1, eps=self.eps)
#         sim_matrix = (normed @ normed.T).cpu().numpy()

#         k = len(onto_type_indices)
#         fig, ax = plt.subplots(figsize=(max(8, k), max(6, k - 2)))

#         sns.heatmap(
#             sim_matrix,
#             xticklabels=type_labels,
#             yticklabels=type_labels,
#             cmap="coolwarm",
#             center=0,
#             vmin=-1,
#             vmax=1,
#             annot=k <= 20,
#             fmt=".2f",
#             linewidths=0.5,
#             ax=ax
#         )

#         ax.set_title(
#             "Cosine similarity — ontology type embeddings (stable trainer)",
#             fontsize=13
#         )

#         plt.xticks(rotation=45, ha="right", fontsize=9)
#         plt.yticks(rotation=0, fontsize=9)
#         plt.tight_layout()

#         if onto_sim_save_path:
#             plt.savefig(onto_sim_save_path, dpi=150, bbox_inches="tight")
#             if verbose:
#                 print(f"  Similarity matrix saved -> {onto_sim_save_path}")

#         plt.show()

#         print("\n===== Best typing results =====")
#         print(results["final_test"])

#         return results



"""
ImprovedOntologyTrainer — v3
─────────────────────────────
Problème v2 : InfoNCE bloquée à ~4.5 (modèle fait du hasard sur les négatifs)

CAUSE   : Trop de négatifs → InfoNCE trop difficile pour apprendre
          log(N_neg) ≈ 4.5 → N_neg ≈ 90, le modèle ne peut pas discriminer

FIX 1 — InfoNCE avec négatifs bornés (max_neg=20)
  On tire au sort max 20 négatifs par relation au lieu de toutes les directions.
  Le problème devient tractable → loss devrait descendre sous 2.0

FIX 2 — Soft cosine alignment loss (signal plus doux en parallèle)
  Pour chaque direction KG, on minimise (1 - cos(kg_dir, mean_onto_dir)).
  Converge facilement, guide le modèle avant que l'InfoNCE soit utile.

FIX 3 — ReduceLROnPlateau
  Le LR est réduit quand le total loss stagne → élimine les oscillations
  résiduelles de la task loss.

FIX 4 — Curriculum sur la difficulté InfoNCE
  Pendant le warmup : seulement le soft alignment
  Après warmup     : InfoNCE prend le relais (signal plus discriminant)
"""

import json
import torch
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from train.Trainer import Trainer


# ─── Mapping helper (inchangé) ────────────────────────────────────────────────

def build_relation_type_mapping(
    kg_gdp,
    onto_json_path: str,
    onto_gdp,
) -> Dict[int, List[Tuple[int, int]]]:
    with open(onto_json_path, "r") as f:
        onto_json = json.load(f)

    valid_pairs_per_type: Dict[int, List[Tuple[int, int]]] = {}
    skipped_relations, skipped_types = [], []

    for predicate_str, onto_rel_data in onto_json.items():
        if predicate_str not in kg_gdp.predicate_to_id:
            skipped_relations.append(predicate_str)
            continue
        relation_id = kg_gdp.predicate_to_id[predicate_str]
        pairs = []
        for src_type_name, _, dst_type_name in onto_rel_data["triplets"]:
            if src_type_name not in onto_gdp.nodes_index:
                skipped_types.append(src_type_name); continue
            if dst_type_name not in onto_gdp.nodes_index:
                skipped_types.append(dst_type_name); continue
            pairs.append((onto_gdp.nodes_index[src_type_name],
                          onto_gdp.nodes_index[dst_type_name]))
        if pairs:
            valid_pairs_per_type[relation_id] = pairs

    mapped   = len(valid_pairs_per_type)
    total_kg = len(kg_gdp.predicate_to_id)
    print(f"[Mapping] {mapped}/{total_kg} KG relations mapped "
          f"({100*mapped/total_kg:.1f}%)")
    return valid_pairs_per_type


# ─── Trainer v3 ───────────────────────────────────────────────────────────────

class ImprovedOntologyTrainer(Trainer):

    def __init__(
        self,
        ontology_data,
        # Lambda
        lambda_type: float = 0.15,
        warmup_epochs: int = 20,
        # InfoNCE (FIX: négatifs bornés)
        infonce_temperature: float = 0.5,
        max_neg_per_rel: int = 20,          # FIX: cap sur les négatifs
        infonce_weight: float = 1.0,
        # Soft cosine alignment (FIX: signal auxiliaire plus doux)
        soft_align_weight: float = 0.5,     # actif dès epoch 1
        # Type-separation (inchangé, fonctionne)
        type_sep_margin: float = 0.5,
        type_sep_weight: float = 0.3,
        # Paires négatives explicites (inchangé, fonctionne)
        explicit_neg_type_pairs: Optional[List[Tuple[str, str]]] = None,
        explicit_neg_weight: float = 0.5,
        # LR scheduler (FIX)
        use_lr_scheduler: bool = True,
        lr_patience: int = 10,
        lr_factor: float = 0.5,
        lr_min: float = 1e-5,
        # Infra
        valid_pairs_per_type: Optional[Dict[int, List[Tuple[int, int]]]] = None,
        grad_clip_norm: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.lambda_type          = lambda_type
        self.warmup_epochs        = warmup_epochs
        self.infonce_temp         = infonce_temperature
        self.max_neg_per_rel      = max_neg_per_rel
        self.infonce_weight       = infonce_weight
        self.soft_align_weight    = soft_align_weight
        self.type_sep_margin      = type_sep_margin
        self.type_sep_weight      = type_sep_weight
        self.explicit_neg_weight  = explicit_neg_weight
        self.grad_clip_norm       = grad_clip_norm
        self.valid_pairs_per_type = valid_pairs_per_type or {}
        self.ontology_data        = ontology_data.to(self.device)
        self.eps                  = 1e-8
        self.min_direction_norm   = 1e-6
        self.current_epoch_idx    = 0

        # LR scheduler (ReduceLROnPlateau)
        self.use_lr_scheduler = use_lr_scheduler
        self._lr_scheduler    = None   # initialisé dans _init_scheduler()
        self._lr_patience     = lr_patience
        self._lr_factor       = lr_factor
        self._lr_min          = lr_min

        self.loss_history = {
            "task_loss":       [],
            "onto_infonce":    [],
            "onto_soft":       [],
            "sep_loss":        [],
            "neg_loss":        [],
            "total_loss":      [],
            "lambda":          [],
            "lr":              [],
        }

        # Tenseurs paires de types
        self.valid_pairs_tensor: Dict[int, torch.Tensor] = {}
        for rel, pairs in self.valid_pairs_per_type.items():
            if pairs:
                self.valid_pairs_tensor[rel] = torch.tensor(
                    pairs, dtype=torch.long, device=self.device)

        # Paires négatives explicites
        self._explicit_neg_pairs_names = explicit_neg_type_pairs or []
        self._explicit_neg_tensor: Optional[torch.Tensor] = None

    # ── Initialisation du LR scheduler après super().__init__() ──────────────

    def _init_lr_scheduler(self):
        """Appeler après que self.optimizer est construit par Trainer.__init__."""
        if self.use_lr_scheduler and self.optimizer is not None:
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            self._lr_scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode      = "min",
                patience  = self._lr_patience,
                factor    = self._lr_factor,
                min_lr    = self._lr_min,
            )

    def resolve_explicit_neg_pairs(self, onto_gdp):
        """Appeler UNE FOIS avant train(), après construction."""
        pairs = []
        for (a, b) in self._explicit_neg_pairs_names:
            if a in onto_gdp.nodes_index and b in onto_gdp.nodes_index:
                pairs.append((onto_gdp.nodes_index[a], onto_gdp.nodes_index[b]))
            else:
                print(f"  [WARN] explicit neg pair not found: ({a}, {b})")
        if pairs:
            self._explicit_neg_tensor = torch.tensor(
                pairs, dtype=torch.long, device=self.device)
            print(f"  [ExplicitNeg] {len(pairs)} paires négatives enregistrées.")

        # Initialiser le LR scheduler ici (optimizer déjà prêt)
        self._init_lr_scheduler()

    # ── Lambda warmup ─────────────────────────────────────────────────────────

    def _get_lambda(self) -> float:
        if self.warmup_epochs <= 0:
            return self.lambda_type
        alpha = min(1.0, (self.current_epoch_idx + 1) / self.warmup_epochs)
        return self.lambda_type * alpha

    def _infonce_active(self) -> bool:
        """InfoNCE actif seulement APRÈS le warmup (curriculum)."""
        return self.current_epoch_idx >= self.warmup_epochs

    # ── ① Soft cosine alignment (signal doux, toujours actif) ─────────────────

    def _compute_soft_alignment_loss(
        self,
        kg_out: torch.Tensor,
        onto_out: torch.Tensor,
        batch,
    ) -> torch.Tensor:
        """
        Pour chaque relation r :
          - Calculer la direction KG moyenne normalisée
          - Calculer la direction ontologie moyenne normalisée
          - Loss = 1 - cosine(kg_mean_dir, onto_mean_dir)

        Signal doux et stable → converge facilement → guide le modèle
        pendant et après le warmup.
        """
        if not self.valid_pairs_tensor:
            return torch.tensor(0.0, device=self.device)

        edge_index = getattr(batch, "edge_index", None)
        edge_type  = getattr(batch, "edge_type",  None)
        if edge_index is None or edge_type is None:
            return torch.tensor(0.0, device=self.device)

        relation_losses = []

        for rel_tensor in torch.unique(edge_type):
            rel = rel_tensor.item()
            if rel not in self.valid_pairs_tensor:
                continue

            # Direction KG moyenne
            rel_mask  = (edge_type == rel)
            rel_edges = edge_index[:, rel_mask]
            kg_dirs   = kg_out[rel_edges[1]] - kg_out[rel_edges[0]]
            kg_norms  = torch.norm(kg_dirs, dim=1)
            valid_kg  = kg_norms > self.min_direction_norm
            if valid_kg.sum() == 0:
                continue
            kg_dirs    = F.normalize(kg_dirs[valid_kg], p=2, dim=1, eps=self.eps)
            kg_mean    = F.normalize(kg_dirs.mean(dim=0, keepdim=True),
                                     p=2, dim=1, eps=self.eps)

            # Direction ontologie moyenne
            pairs      = self.valid_pairs_tensor[rel]
            onto_dirs  = onto_out[pairs[:, 1]] - onto_out[pairs[:, 0]]
            onto_norms = torch.norm(onto_dirs, dim=1)
            valid_onto = onto_norms > self.min_direction_norm
            if valid_onto.sum() == 0:
                continue
            onto_dirs  = F.normalize(onto_dirs[valid_onto], p=2, dim=1,
                                     eps=self.eps)
            onto_mean  = F.normalize(onto_dirs.mean(dim=0, keepdim=True),
                                     p=2, dim=1, eps=self.eps)

            cos_sim = F.cosine_similarity(kg_mean, onto_mean, dim=1, eps=self.eps)
            relation_losses.append((1.0 - cos_sim).mean())

        if not relation_losses:
            return torch.tensor(0.0, device=self.device)

        return torch.stack(relation_losses).mean()

    # ── ② InfoNCE avec négatifs bornés ───────────────────────────────────────

    def _compute_infonce_loss(
        self,
        kg_out: torch.Tensor,
        onto_out: torch.Tensor,
        batch,
    ) -> torch.Tensor:
        """
        InfoNCE avec max_neg_per_rel négatifs tirés aléatoirement.
        Rend le problème tractable → loss devrait descendre sous 2.0.
        """
        if not self.valid_pairs_tensor:
            return torch.tensor(0.0, device=self.device)

        edge_index = getattr(batch, "edge_index", None)
        edge_type  = getattr(batch, "edge_type",  None)
        if edge_index is None or edge_type is None:
            return torch.tensor(0.0, device=self.device)

        # Précalcul des directions ontologie par relation
        onto_dirs_per_rel: Dict[int, torch.Tensor] = {}
        for rel, pairs in self.valid_pairs_tensor.items():
            dirs  = onto_out[pairs[:, 1]] - onto_out[pairs[:, 0]]
            norms = torch.norm(dirs, dim=1)
            valid = norms > self.min_direction_norm
            if valid.sum() == 0:
                continue
            onto_dirs_per_rel[rel] = F.normalize(
                dirs[valid], p=2, dim=1, eps=self.eps)

        if not onto_dirs_per_rel:
            return torch.tensor(0.0, device=self.device)

        relation_losses = []

        for rel_tensor in torch.unique(edge_type):
            rel = rel_tensor.item()
            if rel not in onto_dirs_per_rel:
                continue

            # Directions KG
            rel_mask  = (edge_type == rel)
            rel_edges = edge_index[:, rel_mask]
            kg_dirs   = kg_out[rel_edges[1]] - kg_out[rel_edges[0]]
            kg_norms  = torch.norm(kg_dirs, dim=1)
            valid_kg  = kg_norms > self.min_direction_norm
            if valid_kg.sum() == 0:
                continue
            kg_dirs = F.normalize(kg_dirs[valid_kg], p=2, dim=1, eps=self.eps)

            pos_dirs = onto_dirs_per_rel[rel]   # (P, D)

            # Négatifs : autres relations, BORNÉS à max_neg_per_rel
            neg_list = []
            for other_rel, other_dirs in onto_dirs_per_rel.items():
                if other_rel != rel:
                    neg_list.append(other_dirs)
            if not neg_list:
                continue

            neg_all = torch.cat(neg_list, dim=0)   # (N_all, D)

            # Sous-échantillonnage des négatifs
            if neg_all.size(0) > self.max_neg_per_rel:
                perm    = torch.randperm(neg_all.size(0), device=self.device)
                neg_all = neg_all[perm[:self.max_neg_per_rel]]

            T = self.infonce_temp
            sim_pos = (kg_dirs @ pos_dirs.T) / T    # (K, P)
            sim_neg = (kg_dirs @ neg_all.T) / T     # (K, max_neg)

            best_pos  = sim_pos.max(dim=1).values
            log_denom = torch.logsumexp(
                torch.cat([sim_pos, sim_neg], dim=1), dim=1)
            relation_losses.append((-(best_pos - log_denom)).mean())

        if not relation_losses:
            return torch.tensor(0.0, device=self.device)

        return torch.stack(relation_losses).mean()

    # ── ③ Type-separation (inchangé, fonctionne) ─────────────────────────────

    def _compute_type_separation_loss(self, onto_out: torch.Tensor) -> torch.Tensor:
        type_indices = set()
        for pairs in self.valid_pairs_tensor.values():
            type_indices.update(pairs[:, 0].tolist())
            type_indices.update(pairs[:, 1].tolist())

        if len(type_indices) < 2:
            return torch.tensor(0.0, device=self.device)

        idx_list   = sorted(type_indices)
        normed     = F.normalize(onto_out[idx_list], p=2, dim=1, eps=self.eps)
        sim_matrix = normed @ normed.T
        T          = sim_matrix.size(0)
        mask       = ~torch.eye(T, dtype=torch.bool, device=self.device)
        return F.relu(sim_matrix[mask] - self.type_sep_margin).mean()

    # ── ④ Explicit negative pairs (inchangé, fonctionne) ─────────────────────

    def _compute_explicit_neg_loss(self, onto_out: torch.Tensor) -> torch.Tensor:
        if self._explicit_neg_tensor is None:
            return torch.tensor(0.0, device=self.device)
        pairs = self._explicit_neg_tensor
        emb_a = F.normalize(onto_out[pairs[:, 0]], p=2, dim=1, eps=self.eps)
        emb_b = F.normalize(onto_out[pairs[:, 1]], p=2, dim=1, eps=self.eps)
        sims  = (emb_a * emb_b).sum(dim=1)
        return F.relu(sims - 2 * self.type_sep_margin).mean()

    # ── Epoch ─────────────────────────────────────────────────────────────────

    def _train_epoch(self, train_loader) -> float:
        self.model.train()

        total_task = total_infonce = total_soft = 0.0
        total_sep  = total_neg    = total_loss  = 0.0
        n_batches  = 0
        lam        = self._get_lambda()

        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            kg_out   = self._forward_pass(batch)
            onto_out = self.model(
                x          = self.ontology_data.x,
                edge_index = self.ontology_data.edge_index,
                edge_type  = getattr(self.ontology_data, "edge_type",   None),
                edge_weight= getattr(self.ontology_data, "edge_weight", None),
                edge_attr  = getattr(self.ontology_data, "edge_attr",   None),
            )

            mask_input = torch.isin(batch.n_id, batch.input_id)
            mask       = mask_input & batch.train_mask & (batch.y >= 0)
            if mask.sum() == 0:
                continue

            task_loss  = self.criterion(kg_out[mask], batch.y[mask])

            # Signal doux : toujours actif
            soft_loss  = self._compute_soft_alignment_loss(kg_out, onto_out, batch)

            # InfoNCE : actif seulement après warmup (curriculum)
            if self._infonce_active():
                infonce_loss = self._compute_infonce_loss(kg_out, onto_out, batch)
            else:
                infonce_loss = torch.tensor(0.0, device=self.device)

            sep_loss  = self._compute_type_separation_loss(onto_out)
            neg_loss  = self._compute_explicit_neg_loss(onto_out)

            loss = (
                task_loss
                + lam * (
                    self.soft_align_weight  * soft_loss
                    + self.infonce_weight   * infonce_loss
                )
                + self.type_sep_weight   * sep_loss
                + self.explicit_neg_weight * neg_loss
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.grad_clip_norm)
            self.optimizer.step()

            total_task    += task_loss.item()
            total_infonce += infonce_loss.item()
            total_soft    += soft_loss.item()
            total_sep     += sep_loss.item()
            total_neg     += neg_loss.item()
            total_loss    += loss.item()
            n_batches     += 1

        self.current_epoch_idx += 1

        if n_batches == 0:
            avg = dict(task=0, infonce=0, soft=0, sep=0, neg=0, total=0)
        else:
            avg = dict(
                task    = total_task    / n_batches,
                infonce = total_infonce / n_batches,
                soft    = total_soft    / n_batches,
                sep     = total_sep     / n_batches,
                neg     = total_neg     / n_batches,
                total   = total_loss    / n_batches,
            )

        # LR scheduler step
        current_lr = self.optimizer.param_groups[0]["lr"]
        if self._lr_scheduler is not None:
            self._lr_scheduler.step(avg["total"])

        self.loss_history["task_loss"].append(avg["task"])
        self.loss_history["onto_infonce"].append(avg["infonce"])
        self.loss_history["onto_soft"].append(avg["soft"])
        self.loss_history["sep_loss"].append(avg["sep"])
        self.loss_history["neg_loss"].append(avg["neg"])
        self.loss_history["total_loss"].append(avg["total"])
        self.loss_history["lambda"].append(lam)
        self.loss_history["lr"].append(current_lr)

        return avg["total"]

    # ── Visualisation ─────────────────────────────────────────────────────────

    def train_with_display_matrix(
        self,
        train_loader,
        val_loader,
        test_loader,
        onto_type_names: List[str],
        gdp,
        epochs: int = 100,
        patience: int = 100,
        verbose: bool = True,
        eval_every: int = 1,
        save_best_model: bool = True,
        onto_sim_save_path: Optional[str] = "onto_type_similarity.png",
        loss_plot_path: Optional[str] = "training_losses.png",
    ) -> Dict:

        results = self.train(
            train_loader    = train_loader,
            val_loader      = val_loader,
            test_loader     = test_loader,
            epochs          = epochs,
            patience        = patience,
            verbose         = verbose,
            eval_every      = eval_every,
            save_best_model = save_best_model,
        )

        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(2, 1, figsize=(13, 10))
        ep = range(1, len(self.loss_history["total_loss"]) + 1)

        # ── Plot 1 : losses ───────────────────────────────────────────────────
        ax = axes[0]
        ax.plot(ep, self.loss_history["task_loss"],
                label="Task loss", linewidth=2, color="#2196F3")
        ax.plot(ep, self.loss_history["onto_soft"],
                label="Soft align loss", linewidth=2, color="#00BCD4")
        ax.plot(ep, self.loss_history["onto_infonce"],
                label="InfoNCE loss", linewidth=2, color="#FF9800")
        ax.plot(ep, self.loss_history["sep_loss"],
                label="Type-sep loss", linewidth=1.5, color="#9C27B0",
                linestyle=":")
        ax.plot(ep, self.loss_history["neg_loss"],
                label="Explicit neg", linewidth=1.5, color="#795548",
                linestyle="-.")
        ax.plot(ep, self.loss_history["total_loss"],
                label="Total loss", linewidth=2.5, color="#4CAF50")

        ax2 = ax.twinx()
        ax2.plot(ep, self.loss_history["lambda"],
                 label="Lambda", linestyle="--", linewidth=2, color="#F44336")
        ax2.set_ylabel("Lambda", color="#F44336")

        # Marquer le début d'InfoNCE
        if self.warmup_epochs < len(list(ep)):
            ax.axvline(x=self.warmup_epochs, color="gray",
                       linestyle="--", alpha=0.5, label=f"InfoNCE start (ep {self.warmup_epochs})")

        l1, lb1 = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, lb1 + lb2, loc="upper right", fontsize=8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training losses — ImprovedOntologyTrainer v3")
        ax.grid(True, alpha=0.3)

        # ── Plot 2 : learning rate ────────────────────────────────────────────
        axes[1].plot(ep, self.loss_history["lr"],
                     color="#607D8B", linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Learning rate")
        axes[1].set_title("Learning rate (ReduceLROnPlateau)")
        axes[1].set_yscale("log")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if loss_plot_path:
            plt.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
        plt.show()

        # ── Matrice de similarité ─────────────────────────────────────────────
        onto_type_indices = [gdp.nodes_index[n] for n in onto_type_names
                             if n in gdp.nodes_index]
        type_labels = [n for n in onto_type_names if n in gdp.nodes_index]

        if not onto_type_indices:
            return results

        self.model.eval()
        with torch.no_grad():
            onto_out = self.model(
                x          = self.ontology_data.x,
                edge_index = self.ontology_data.edge_index,
                edge_type  = getattr(self.ontology_data, "edge_type",   None),
                edge_weight= getattr(self.ontology_data, "edge_weight", None),
                edge_attr  = getattr(self.ontology_data, "edge_attr",   None),
            )

        type_embs  = onto_out[onto_type_indices]
        normed     = F.normalize(type_embs, p=2, dim=1, eps=self.eps)
        sim_matrix = (normed @ normed.T).cpu().numpy()

        k   = len(onto_type_indices)
        fig, ax = plt.subplots(figsize=(max(8, k), max(6, k - 2)))
        sns.heatmap(
            sim_matrix,
            xticklabels=type_labels, yticklabels=type_labels,
            cmap="coolwarm", center=0, vmin=-1, vmax=1,
            annot=(k <= 20), fmt=".2f", linewidths=0.5, ax=ax,
        )
        ax.set_title(
            "Cosine similarity — ontology type embeddings (v3)", fontsize=13)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        if onto_sim_save_path:
            plt.savefig(onto_sim_save_path, dpi=150, bbox_inches="tight")
        plt.show()

        if verbose:
            print("\n===== Best typing results =====")
            print(results["final_test"])

        return results