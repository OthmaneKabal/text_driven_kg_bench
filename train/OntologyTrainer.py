import json
import torch
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from train.Trainer import Trainer


# ----------------------------------------------------------------------
# Standalone helper — build the mapping dict
# ----------------------------------------------------------------------

def build_relation_type_mapping(
    kg_gdp,
    onto_json_path: str,
    onto_gdp
) -> Dict[int, List[Tuple[int, int]]]:
    """
    Build the mapping:
        relation_id (int from KG) -> [(src_type_idx, dst_type_idx), ...]

    Only relations whose predicate name appears in BOTH the KG and the
    ontology JSON are included (the 58% mapped relations).

    Parameters:
    - kg_gdp   : GraphDataPreparation of the KG  (has predicate_to_id, nodes_index)
    - onto_json_path: path to the ontology JSON file
    - onto_gdp : GraphDataPreparation of the ontology (has nodes_index)

    Returns:
    - valid_pairs_per_type: Dict[int, List[Tuple[int, int]]]
    """
    with open(onto_json_path, "r") as f:
        onto_json = json.load(f)

    valid_pairs_per_type: Dict[int, List[Tuple[int, int]]] = {}
    skipped_relations = []
    skipped_types     = []

    for predicate_str, onto_rel_data in onto_json.items():

        # 1. Is this predicate present in the KG?
        if predicate_str not in kg_gdp.predicate_to_id:
            skipped_relations.append(predicate_str)
            continue

        relation_id = kg_gdp.predicate_to_id[predicate_str]
        pairs: List[Tuple[int, int]] = []

        # 2. For each (src_type, predicate, dst_type) triplet in the onto
        for src_type_name, _, dst_type_name in onto_rel_data["triplets"]:

            # Both type nodes must exist in the ontology graph
            if src_type_name not in onto_gdp.nodes_index:
                skipped_types.append(src_type_name)
                continue
            if dst_type_name not in onto_gdp.nodes_index:
                skipped_types.append(dst_type_name)
                continue

            src_type_idx = onto_gdp.nodes_index[src_type_name]
            dst_type_idx = onto_gdp.nodes_index[dst_type_name]
            pairs.append((src_type_idx, dst_type_idx))

        if pairs:
            valid_pairs_per_type[relation_id] = pairs

    # Summary
    mapped   = len(valid_pairs_per_type)
    total_kg = len(kg_gdp.predicate_to_id)
    print(f"[Mapping] {mapped}/{total_kg} KG relations mapped to ontology "
          f"({100 * mapped / total_kg:.1f}%)")
    if skipped_relations:
        print(f"[Mapping] {len(skipped_relations)} relations not found in onto: "
              f"{skipped_relations[:5]}{'...' if len(skipped_relations) > 5 else ''}")
    if skipped_types:
        unique_skipped = list(set(skipped_types))
        print(f"[Mapping] {len(unique_skipped)} type nodes not found in onto graph: "
              f"{unique_skipped[:5]}{'...' if len(unique_skipped) > 5 else ''}")

    return valid_pairs_per_type


# ----------------------------------------------------------------------
# OntologyTrainer
# ----------------------------------------------------------------------

class OntologyTrainer(Trainer):
    def __init__(
        self,
        ontology_data,
        lambda_type: float = 0.5,
        valid_pairs_per_type: Optional[Dict[int, List[Tuple[int, int]]]] = None,
        **kwargs
    ):
        """
        Trainer with ontological type-direction constraint.

        For each mapped triplet (src, rel, dst) in a batch:
          1. Compute direction_kg  = dst_emb  - src_emb
          2. Compute direction_onto = type_dst_emb - type_src_emb  for all allowed pairs
          3. Select best matching onto direction (hard, no gradient through selection)
          4. loss = 1 - cosine(direction_kg, best_onto_direction)

        Parameters:
        - ontology_data       : PyG Data object of the ontology graph
        - lambda_type         : weight of onto loss vs task loss
        - valid_pairs_per_type: output of build_relation_type_mapping()
        - **kwargs            : Trainer parameters (model, device, lr, ...)
        """
        super().__init__(**kwargs)

        self.lambda_type          = lambda_type
        self.valid_pairs_per_type = valid_pairs_per_type or {}
        self.ontology_data        = ontology_data.to(self.device)

    # ------------------------------------------------------------------
    # Type-direction constraint loss
    # ------------------------------------------------------------------

    def _compute_type_constraint_loss(
        self,
        kg_out:   torch.Tensor,   # (N_batch, d)
        onto_out: torch.Tensor,   # (N_onto,  d)
        batch
    ) -> torch.Tensor:
        """
        Compute the type-direction constraint loss over all mapped
        triplets in the current batch.

        For each triplet (src, rel, dst) where rel is mapped:
          - direction_kg   = dst_emb - src_emb
          - best_direction = argmax cosine over all allowed onto directions
          - loss           = 1 - cosine(direction_kg, best_direction)

        Returns scalar loss (0.0 if no mapped triplet found).
        """
        if not self.valid_pairs_per_type:
            return torch.tensor(0.0, device=self.device)

        edge_index = batch.edge_index                          # (2, E)
        edge_type  = getattr(batch, "edge_type", None)         # (E,)

        if edge_type is None:
            return torch.tensor(0.0, device=self.device)

        total_loss = torch.tensor(0.0, device=self.device)
        count      = 0

        for e_idx in range(edge_index.size(1)):

            rel = edge_type[e_idx].item()
            if rel not in self.valid_pairs_per_type:
                continue

            src_local = edge_index[0, e_idx].item()
            dst_local = edge_index[1, e_idx].item()

            # Direction in KG embedding space
            direction_kg = kg_out[dst_local] - kg_out[src_local]   # (d,)

            # All allowed onto directions for this relation
            pairs = self.valid_pairs_per_type[rel]                  # [(si, di), ...]

            src_type_embs = onto_out[[p[0] for p in pairs]]         # (K, d)
            dst_type_embs = onto_out[[p[1] for p in pairs]]         # (K, d)
            onto_directions = dst_type_embs - src_type_embs         # (K, d)

            # Cosine similarities between direction_kg and each onto direction
            sims = F.cosine_similarity(
                direction_kg.unsqueeze(0),                          # (1, d)
                onto_directions,                                    # (K, d)
                dim=1
            )                                                       # (K,)

            # Hard selection — no gradient through argmax
            with torch.no_grad():
                best_i = sims.argmax().item()

            best_direction = onto_directions[best_i]                # (d,)

            # Loss: push direction_kg toward best onto direction
            loss = 1.0 - F.cosine_similarity(
                direction_kg.unsqueeze(0),
                best_direction.unsqueeze(0)
            )
            total_loss = total_loss + loss.squeeze()
            count += 1

        return total_loss / max(count, 1)

    # ------------------------------------------------------------------
    # Override _train_epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, train_loader) -> float:
        """
        Train for one epoch: task loss + lambda_type * onto direction loss.
        Onto embeddings evolve with the model (no detach).
        """
        self.model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            # KG forward
            kg_out = self._forward_pass(batch)

            # Onto forward — evolves with the model, no detach
            onto_out = self.model(
                x=self.ontology_data.x,
                edge_index=self.ontology_data.edge_index,
                edge_type=getattr(self.ontology_data, "edge_type",   None),
                edge_weight=getattr(self.ontology_data, "edge_weight", None),
                edge_attr=getattr(self.ontology_data, "edge_attr",   None)
            )

            # Task loss
            mask_input_id = torch.isin(batch.n_id, batch.input_id)
            mask          = mask_input_id & batch.train_mask & (batch.y >= 0)
            task_loss     = self.criterion(kg_out[mask], batch.y[mask])

            # Onto type-direction constraint loss
            onto_loss = self._compute_type_constraint_loss(kg_out, onto_out, batch)

            loss = task_loss + self.lambda_type * onto_loss

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    # ------------------------------------------------------------------
    # train_with_display_matrix (diagnostic)
    # ------------------------------------------------------------------

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
        onto_sim_save_path: Optional[str] = "onto_type_similarity.png"
    ) -> Dict:
        """
        Full training loop + ontology type similarity matrix visualization
        at the end on the best model.

        Parameters:
        - train_loader      : DataLoader for training
        - val_loader        : DataLoader for validation
        - test_loader       : DataLoader for testing
        - onto_type_names   : List of type node names (text) to visualize
        - gdp               : GraphDataPreparation of the ontology
        - epochs            : Maximum number of epochs
        - patience          : Early stopping patience
        - verbose           : Print training progress
        - eval_every        : Evaluate every N epochs
        - save_best_model   : Keep best model state
        - onto_sim_save_path: Path to save the similarity matrix image

        Returns:
        - Dictionary with best validation and final test metrics
        """
        results = self.train(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=epochs,
            patience=patience,
            verbose=verbose,
            eval_every=eval_every,
            save_best_model=save_best_model
        )

        # ------------------------------------------------------------------
        # Similarity matrix (post-training, best model)
        # ------------------------------------------------------------------
        import matplotlib.pyplot as plt
        import seaborn as sns

        if verbose:
            print("\nComputing ontology type similarity matrix...")

        onto_type_indices = [
            gdp.nodes_index[name]
            for name in onto_type_names
            if name in gdp.nodes_index
        ]
        type_labels = [
            name for name in onto_type_names
            if name in gdp.nodes_index
        ]
        missing = [n for n in onto_type_names if n not in gdp.nodes_index]
        if missing and verbose:
            print(f"  [WARN] Types not found in ontology graph: {missing}")

        if not onto_type_indices:
            print("  [WARN] No valid type indices found, skipping similarity matrix.")
            return results

        self.model.eval()
        with torch.no_grad():
            onto_out = self.model(
                x=self.ontology_data.x,
                edge_index=self.ontology_data.edge_index,
                edge_type=getattr(self.ontology_data, "edge_type",   None),
                edge_weight=getattr(self.ontology_data, "edge_weight", None),
                edge_attr=getattr(self.ontology_data, "edge_attr",   None)
            )

        type_embs  = onto_out[onto_type_indices]
        normed     = F.normalize(type_embs, dim=1)
        sim_matrix = (normed @ normed.T).cpu().numpy()

        k = len(onto_type_indices)
        fig, ax = plt.subplots(figsize=(max(8, k), max(6, k - 2)))
        sns.heatmap(
            sim_matrix,
            xticklabels=type_labels,
            yticklabels=type_labels,
            cmap="coolwarm",
            center=0,
            vmin=-1, vmax=1,
            annot=k <= 20,
            fmt=".2f",
            linewidths=0.5,
            ax=ax
        )
        ax.set_title(
            "Cosine similarity — ontology type embeddings (post-training)",
            fontsize=13
        )
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()

        if onto_sim_save_path:
            plt.savefig(onto_sim_save_path, dpi=150, bbox_inches="tight")
            if verbose:
                print(f"  Similarity matrix saved -> {onto_sim_save_path}")

        plt.show()
        # ------------------------------------------------------------------
        
        print("\n===== Best typing results =====")
        print(results["final_test"])
        return results