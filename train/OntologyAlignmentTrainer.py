# import torch
# import torch.nn.functional as F
# from typing import List, Tuple
# from torch_geometric.loader import NeighborLoader
# from train.Trainer import Trainer


# # ──────────────────────────────────────────────────────────────────────────────
# # Helper : résoudre les noms de types en indices
# # ──────────────────────────────────────────────────────────────────────────────


# def build_shared_type_indices(
#     type_names: List[str],
#     kg_gdp,
#     onto_gdp,
# ) -> List[Tuple[int, int]]:
#     """
#     Résout une liste de noms de types en paires :
#         (kg_node_idx, onto_node_idx)
#     """
#     pairs, skipped = [], []

#     for name in type_names:
#         if name not in kg_gdp.nodes_index:
#             skipped.append(f"{name} (absent du KG)")
#             continue

#         if name not in onto_gdp.nodes_index:
#             skipped.append(f"{name} (absent de l'onto)")
#             continue

#         pairs.append((kg_gdp.nodes_index[name], onto_gdp.nodes_index[name]))

#     print(f"[SharedTypes] {len(pairs)}/{len(type_names)} types résolus.")

#     if skipped:
#         print(f"[SharedTypes] Ignorés : {skipped}")

#     return pairs


# # ──────────────────────────────────────────────────────────────────────────────
# # Trainer
# # ──────────────────────────────────────────────────────────────────────────────

# class OntologyAlignmentTrainer(Trainer):
#     """
#     Entraînement alterné par epoch :

#       1. Classification :
#          plusieurs mini-batchs du KG via train_loader

#       2. Alignement :
#          un batch KG_types via NeighborLoader
#          un batch ontology_types via NeighborLoader

#     Objectif :
#         rapprocher l'embedding du même type dans le KG et dans l'ontologie.
#     """

#     def __init__(
#         self,
#         ontology_data,
#         kg_data,
#         shared_type_pairs: List[Tuple[int, int]],
#         lambda_align: float = 0.01,
#         align_batch_size: int = None,
#         align_num_neighbors: List[int] = [20, 10],
#         **kwargs,
#     ):
#         """
#         Paramètres
#         ----------
#         ontology_data       : PyG Data — graphe de l'ontologie
#         kg_data             : PyG Data — graphe complet du KG
#         shared_type_pairs   : liste de (kg_type_idx, onto_type_idx)
#         lambda_align        : poids de la loss d'alignement
#         align_batch_size    : taille du batch d'alignement.
#                               Si None, tous les types partagés sont utilisés.
#         align_num_neighbors : voisinage utilisé par NeighborLoader
#         **kwargs            : paramètres du Trainer parent
#                               ex: model, device, lr, weight_decay...
#         """
#         super().__init__(**kwargs)

#         self.ontology_data = ontology_data
#         self.kg_data = kg_data
#         self.lambda_align = lambda_align

#         if shared_type_pairs:
#             kg_idxs, onto_idxs = zip(*shared_type_pairs)

#             self.kg_type_idx = torch.tensor(kg_idxs, dtype=torch.long)
#             self.onto_type_idx = torch.tensor(onto_idxs, dtype=torch.long)

#             if align_batch_size is None:
#                 align_batch_size = len(shared_type_pairs)

#             self.kg_align_loader = NeighborLoader(
#                 self.kg_data,
#                 input_nodes=self.kg_type_idx,
#                 num_neighbors=align_num_neighbors,
#                 batch_size=align_batch_size,
#                 shuffle=False,
#             )

#             self.onto_align_loader = NeighborLoader(
#                 self.ontology_data,
#                 input_nodes=self.onto_type_idx,
#                 num_neighbors=align_num_neighbors,
#                 batch_size=align_batch_size,
#                 shuffle=False,
#             )

#         else:
#             self.kg_type_idx = None
#             self.onto_type_idx = None
#             self.kg_align_loader = None
#             self.onto_align_loader = None

#     # ──────────────────────────────────────────────────────────────────────────
#     # Forward générique
#     # ──────────────────────────────────────────────────────────────────────────

#     def _model_forward(self, data):
#         return self.model(
#             x=data.x,
#             edge_index=data.edge_index,
#             edge_type=getattr(data, "edge_type", None),
#             edge_weight=getattr(data, "edge_weight", None),
#             edge_attr=getattr(data, "edge_attr", None),
#         )

#     # ──────────────────────────────────────────────────────────────────────────
#     # Step 1 — classification
#     # ──────────────────────────────────────────────────────────────────────────

#     def _step_classification(self, batch) -> torch.Tensor:
#         batch = batch.to(self.device)

#         kg_out = self._forward_pass(batch)

#         # Avec NeighborLoader, les nœuds cibles sont au début du batch
#         if hasattr(batch, "batch_size"):
#             bs = batch.batch_size
#             mask = batch.train_mask[:bs] & (batch.y[:bs] >= 0)

#             return self.criterion(
#                 kg_out[:bs][mask],
#                 batch.y[:bs][mask],
#             )

#         # fallback si pas de batch_size
#         mask = batch.train_mask & (batch.y >= 0)

#         return self.criterion(
#             kg_out[mask],
#             batch.y[mask],
#         )

#     # ──────────────────────────────────────────────────────────────────────────
#     # Step 2 — alignement
#     # ──────────────────────────────────────────────────────────────────────────
#     def _unwrap_embeddings(self, out):
#         """
#         Certains encoders retournent (node_emb, rel_emb) ou plusieurs sorties.
#         Pour l'alignement, on garde seulement les embeddings des nœuds.
#         """
#         if isinstance(out, tuple):
#             return out[0]
#         return out
#     def _step_alignment(self) -> torch.Tensor:
#         if self.kg_align_loader is None or self.onto_align_loader is None:
#             return torch.tensor(0.0, device=self.device)

#         total_align_loss = 0.0
#         n_batches = 0

#         for kg_batch, onto_batch in zip(self.kg_align_loader, self.onto_align_loader):
#             kg_batch = kg_batch.to(self.device)
#             onto_batch = onto_batch.to(self.device)

#             # kg_out = self._model_forward(kg_batch)
#             # onto_out = self._model_forward(onto_batch)
#             kg_out = self._encode_pass(kg_batch)
#             onto_out = self._encode_pass(onto_batch)
#             kg_out = self._unwrap_embeddings(kg_out)
#             onto_out = self._unwrap_embeddings(onto_out)
#             # Les seed/input nodes sont au début du batch
#             kg_bs = kg_batch.batch_size
#             onto_bs = onto_batch.batch_size

#             # Normalement égaux si les deux loaders ont le même batch_size
#             bs = min(kg_bs, onto_bs)

#             kg_seed_embs = kg_out[:bs]
#             onto_seed_embs = onto_out[:bs]

#             loss = 1.0 - F.cosine_similarity(
#                 kg_seed_embs,
#                 onto_seed_embs,
#                 dim=1,
#             )

#             total_align_loss += loss.mean()
#             n_batches += 1

#         return total_align_loss / max(n_batches, 1)

#     # ──────────────────────────────────────────────────────────────────────────
#     # Epoch complète
#     # ──────────────────────────────────────────────────────────────────────────

#     def _train_epoch(self, train_loader) -> float:
#         self.model.train()

#         total_cls_loss = 0.0

#         # 1. Phase classification
#         for batch in train_loader:
#             self.optimizer.zero_grad()

#             cls_loss = self._step_classification(batch)

#             cls_loss.backward()
#             self.optimizer.step()

#             total_cls_loss += cls_loss.item()

#         avg_cls_loss = total_cls_loss / max(len(train_loader), 1)

#         # 2. Phase alignement
#         self.optimizer.zero_grad()

#         align_loss = self.lambda_align * self._step_alignment()

#         align_loss.backward()
#         self.optimizer.step()

#         total_loss = avg_cls_loss + align_loss.item()

#         return total_loss

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
from torch_geometric.loader import NeighborLoader
from train.Trainer import Trainer


# ──────────────────────────────────────────────────────────────────────────────
# Helper : résoudre les noms de types en indices
# ──────────────────────────────────────────────────────────────────────────────

def build_shared_type_indices(
    type_names: List[str],
    kg_gdp,
    onto_gdp,
) -> List[Tuple[int, int]]:
    """
    Résout une liste de noms de types en paires :
        (kg_node_idx, onto_node_idx)
    """
    pairs, skipped = [], []

    for name in type_names:
        if name not in kg_gdp.nodes_index:
            skipped.append(f"{name} (absent du KG)")
            continue

        if name not in onto_gdp.nodes_index:
            skipped.append(f"{name} (absent de l'onto)")
            continue

        pairs.append((kg_gdp.nodes_index[name], onto_gdp.nodes_index[name]))

    print(f"[SharedTypes] {len(pairs)}/{len(type_names)} types résolus.")

    if skipped:
        print(f"[SharedTypes] Ignorés : {skipped}")

    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class OntologyAlignmentTrainer(Trainer):
    """
    Entraînement alterné par epoch :

      1. Classification :
         plusieurs mini-batchs du KG via train_loader

      2. Alignement :
         un batch KG_types via NeighborLoader
         un batch ontology_types via NeighborLoader

    Modes d'alignement :
      - "cosine"      : rapproche seulement les mêmes types
      - "contrastive" : InfoNCE, rapproche les mêmes types et éloigne les autres
    """

    def __init__(
        self,
        ontology_data,
        kg_data,
        shared_type_pairs: List[Tuple[int, int]],
        lambda_align: float = 0.01,
        alignment_mode: str = "cosine",
        temperature: float = 0.2,
        align_batch_size: Optional[int] = None,
        align_num_neighbors: List[int] = [20, 10],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.ontology_data = ontology_data
        self.kg_data = kg_data

        self.lambda_align = lambda_align
        self.alignment_mode = alignment_mode
        self.temperature = temperature

        if self.alignment_mode not in ["cosine", "contrastive"]:
            raise ValueError(
                f"alignment_mode={self.alignment_mode} invalide. "
                "Utilise 'cosine' ou 'contrastive'."
            )

        if shared_type_pairs:
            kg_idxs, onto_idxs = zip(*shared_type_pairs)

            self.kg_type_idx = torch.tensor(kg_idxs, dtype=torch.long)
            self.onto_type_idx = torch.tensor(onto_idxs, dtype=torch.long)

            if align_batch_size is None:
                align_batch_size = len(shared_type_pairs)

            self.kg_align_loader = NeighborLoader(
                self.kg_data,
                input_nodes=self.kg_type_idx,
                num_neighbors=align_num_neighbors,
                batch_size=align_batch_size,
                shuffle=False,
            )

            self.onto_align_loader = NeighborLoader(
                self.ontology_data,
                input_nodes=self.onto_type_idx,
                num_neighbors=align_num_neighbors,
                batch_size=align_batch_size,
                shuffle=False,
            )

            print(
                f"[AlignmentTrainer] mode={self.alignment_mode} | "
                f"lambda_align={self.lambda_align} | "
                f"temperature={self.temperature} | "
                f"shared_types={len(shared_type_pairs)} | "
                f"align_batch_size={align_batch_size} | "
                f"neighbors={align_num_neighbors}"
            )

        else:
            self.kg_type_idx = None
            self.onto_type_idx = None
            self.kg_align_loader = None
            self.onto_align_loader = None

            print("[AlignmentTrainer] Aucun type partagé trouvé.")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1 — classification
    # ──────────────────────────────────────────────────────────────────────────

    def _step_classification(self, batch) -> torch.Tensor:
        batch = batch.to(self.device)

        kg_out = self._forward_pass(batch)

        # Avec NeighborLoader, les nœuds cibles sont au début du batch
        if hasattr(batch, "batch_size"):
            bs = batch.batch_size
            mask = batch.train_mask[:bs] & (batch.y[:bs] >= 0)

            return self.criterion(
                kg_out[:bs][mask],
                batch.y[:bs][mask],
            )

        # fallback si pas de batch_size
        mask = batch.train_mask & (batch.y >= 0)

        return self.criterion(
            kg_out[mask],
            batch.y[mask],
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers alignement
    # ──────────────────────────────────────────────────────────────────────────

    def _unwrap_embeddings(self, out):
        """
        Certains encoders retournent (node_emb, rel_emb) ou plusieurs sorties.
        Pour l'alignement, on garde seulement les embeddings des nœuds.
        """
        if isinstance(out, tuple):
            return out[0]
        return out

    def _info_nce_loss(
        self,
        kg_embs: torch.Tensor,
        onto_embs: torch.Tensor,
    ) -> torch.Tensor:
        """
        InfoNCE cross-view.

        Positif :
            kg_embs[i] ↔ onto_embs[i]

        Négatifs :
            kg_embs[i] ↔ onto_embs[j], j != i
        """
        kg_embs = F.normalize(kg_embs, p=2, dim=1)
        onto_embs = F.normalize(onto_embs, p=2, dim=1)

        logits = kg_embs @ onto_embs.t()
        logits = logits / self.temperature

        labels = torch.arange(logits.size(0), device=logits.device)

        loss_kg_to_onto = F.cross_entropy(logits, labels)
        loss_onto_to_kg = F.cross_entropy(logits.t(), labels)

        return 0.5 * (loss_kg_to_onto + loss_onto_to_kg)

    def _compute_pair_loss(
        self,
        kg_seed_embs: torch.Tensor,
        onto_seed_embs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calcule la loss selon self.alignment_mode.
        """
        if self.alignment_mode == "cosine":
            loss = 1.0 - F.cosine_similarity(
                kg_seed_embs,
                onto_seed_embs,
                dim=1,
            )
            return loss.mean()

        if self.alignment_mode == "contrastive":
            if kg_seed_embs.size(0) < 2:
                return torch.tensor(0.0, device=kg_seed_embs.device)

            return self._info_nce_loss(
                kg_seed_embs,
                onto_seed_embs,
            )

        raise ValueError(f"Unknown alignment_mode={self.alignment_mode}")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2 — alignement
    # ──────────────────────────────────────────────────────────────────────────

    def _step_alignment(self) -> torch.Tensor:
        if self.kg_align_loader is None or self.onto_align_loader is None:
            return torch.tensor(0.0, device=self.device)

        total_align_loss = torch.tensor(0.0, device=self.device)
        n_batches = 0

        for kg_batch, onto_batch in zip(self.kg_align_loader, self.onto_align_loader):
            kg_batch = kg_batch.to(self.device)
            onto_batch = onto_batch.to(self.device)

            # embeddings, pas logits de classification
            kg_out = self._encode_pass(kg_batch)
            onto_out = self._encode_pass(onto_batch)

            kg_out = self._unwrap_embeddings(kg_out)
            onto_out = self._unwrap_embeddings(onto_out)

            kg_bs = kg_batch.batch_size
            onto_bs = onto_batch.batch_size
            bs = min(kg_bs, onto_bs)

            kg_seed_embs = kg_out[:bs]
            onto_seed_embs = onto_out[:bs]

            loss = self._compute_pair_loss(
                kg_seed_embs,
                onto_seed_embs,
            )

            total_align_loss = total_align_loss + loss
            n_batches += 1

        return total_align_loss / max(n_batches, 1)

    # ──────────────────────────────────────────────────────────────────────────
    # Epoch complète
    # ──────────────────────────────────────────────────────────────────────────

    # def _train_epoch(self, train_loader) -> float:
    #     self.model.train()

    #     total_cls_loss = 0.0

    #     # 1. Phase classification
    #     for batch in train_loader:
    #         self.optimizer.zero_grad()

    #         cls_loss = self._step_classification(batch)

    #         cls_loss.backward()
    #         self.optimizer.step()

    #         total_cls_loss += cls_loss.item()

    #     avg_cls_loss = total_cls_loss / max(len(train_loader), 1)

    #     # 2. Phase alignement
    #     self.optimizer.zero_grad()

    #     raw_align_loss = self._step_alignment()
    #     align_loss = self.lambda_align * raw_align_loss

    #     align_loss.backward()
    #     self.optimizer.step()

    #     total_loss = avg_cls_loss + align_loss.item()

    #     return total_loss
    def _train_epoch(self, train_loader) -> float:
        self.model.train()

        total_cls_loss = 0.0
        total_align_loss = 0.0
        align_steps = 0

        align_iter = iter(self.kg_align_loader) if self.kg_align_loader else None
        onto_iter = iter(self.onto_align_loader) if self.onto_align_loader else None

        def next_align_batch():
            nonlocal align_iter, onto_iter

            try:
                kg_b = next(align_iter)
                onto_b = next(onto_iter)
            except StopIteration:
                align_iter = iter(self.kg_align_loader)
                onto_iter = iter(self.onto_align_loader)
                kg_b = next(align_iter)
                onto_b = next(onto_iter)

            return kg_b.to(self.device), onto_b.to(self.device)

        for batch in train_loader:
            # 1. Classification / intra-view
            self.optimizer.zero_grad()

            cls_loss = self._step_classification(batch)

            cls_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_cls_loss += cls_loss.item()

            # 2. Cross-view alignment
            if align_iter is not None and self.lambda_align > 0:
                kg_batch, onto_batch = next_align_batch()

                kg_out = self._unwrap_embeddings(self._encode_pass(kg_batch))
                onto_out = self._unwrap_embeddings(self._encode_pass(onto_batch))

                bs = min(kg_batch.batch_size, onto_batch.batch_size)

                if bs >= 1:
                    align_loss = self._compute_pair_loss(
                        kg_out[:bs],
                        onto_out[:bs],
                    )

                    self.optimizer.zero_grad()
                    align_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    # JOIE-style: θ ← θ - (ω · η) ∇J_cross
                    self._scaled_step(self.lambda_align)

                    total_align_loss += align_loss.item()
                    align_steps += 1

        n_cls = max(len(train_loader), 1)
        n_align = max(align_steps, 1)

        avg_cls_loss = total_cls_loss / n_cls
        avg_align_loss = total_align_loss / n_align

        # pour logging: équivalent lisible de J = J_cls + ω J_align
        return avg_cls_loss + self.lambda_align * avg_align_loss

    
    def _train_epoch_v2(self, train_loader) -> float:
        self.model.train()

        total_cls_loss = 0.0
        total_align_loss = 0.0
        align_steps = 0

        align_iter = iter(self.kg_align_loader) if self.kg_align_loader else None
        onto_iter = iter(self.onto_align_loader) if self.onto_align_loader else None

        def next_align_batch():
            nonlocal align_iter, onto_iter

            try:
                kg_b = next(align_iter)
                onto_b = next(onto_iter)
            except StopIteration:
                align_iter = iter(self.kg_align_loader)
                onto_iter = iter(self.onto_align_loader)
                kg_b = next(align_iter)
                onto_b = next(onto_iter)

            return kg_b.to(self.device), onto_b.to(self.device)

        for batch in train_loader:
            self.optimizer.zero_grad()

            # 1. Classification loss
            cls_loss = self._step_classification(batch)

            # 2. Alignment loss
            align_loss = torch.tensor(0.0, device=self.device)

            if align_iter is not None and self.lambda_align > 0:
                kg_batch, onto_batch = next_align_batch()

                kg_out = self._unwrap_embeddings(self._encode_pass(kg_batch))
                onto_out = self._unwrap_embeddings(self._encode_pass(onto_batch))

                bs = min(kg_batch.batch_size, onto_batch.batch_size)

                if bs >= 1:
                    align_loss = self._compute_pair_loss(
                        kg_out[:bs],
                        onto_out[:bs],
                    )

                    total_align_loss += align_loss.item()
                    align_steps += 1

            # 3. Backward combiné — Adam voit un seul gradient cohérent
            total_loss = cls_loss + self.lambda_align * align_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_cls_loss += cls_loss.item()

        n_cls = max(len(train_loader), 1)
        n_align = max(align_steps, 1)

        avg_cls_loss = total_cls_loss / n_cls
        avg_align_loss = total_align_loss / n_align

        return avg_cls_loss + self.lambda_align * avg_align_loss

    def _scaled_step(self, omega: float):
        """
        Applique une update avec lr effectif = omega * lr.
        """
        if omega <= 0:
            return

        old_lrs = []

        for group in self.optimizer.param_groups:
            old_lrs.append(group["lr"])
            group["lr"] = group["lr"] * omega

        self.optimizer.step()

        for group, old_lr in zip(self.optimizer.param_groups, old_lrs):
            group["lr"] = old_lr