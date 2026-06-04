# from typing import Optional
# from data_preprocessing.GraphDataPreparation import GraphDataPreparation

# import pandas as pd
# import torch
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from typing import Dict, List, Optional, Callable
# import json
# from data_preprocessing.data_manager import get_data_and_loaders
# from data_preprocessing.splits import generate_and_save_splits
# from models.StandardClassifier import StandardClassifier
# from train.Trainer import Trainer
# from train.OntologyTrainer import OntologyTrainer
# from utilities.utilities import load_config, seed_everything
# from train.OntologyTrainer import build_relation_type_mapping
# from train.StableOntologyTrainer import ImprovedOntologyTrainer
# class TDGBench:
#     def __init__(self, use_classifier=True, config_path="config.yml"):
#         self.config = load_config(config_path)
#         self.use_classifier = use_classifier
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     def get_data(
#             self,
#             kg_name="GT2KG_kg",
#             init_embd="sentence-transformers/all-MiniLM-L6-v2",
#             split_path="datasets/split/split_42.json",
#             entities_embd_path=None,
#             edges_embd_path=None,
#             random_embd_dim=256,
#             use_cache=False
#     ):
#         return get_data_and_loaders(
#             kg_name=kg_name,
#             model_name_init=init_embd,
#             common_nodes_path=self.config["common_nodes_path"],
#             entities_embd_path=entities_embd_path,
#             edges_embd_path=edges_embd_path,
#             split_file=split_path,
#             random_embd_dim=random_embd_dim,
#             use_cache=use_cache
#         )

#     def generate_splits(self):
#         generate_and_save_splits(
#             self.config["common_nodes_path"],
#             self.config["default_splits_dir"],
#             self.config["n_splits"],
#             self.config["seeds"],
#             self.config["train_ratio"],
#             self.config["val_ratio"],
#             self.config["test_ratio"],
#             self.config["stratify"],
#         )

#     def prepare_model(self, model_or_encoder):
#         """
#         Prepare the model for evaluation.

#         Parameters:
#         - model_or_encoder:
#             - If use_classifier=True: expects an encoder (GNN backbone)
#             - If use_classifier=False: expects a complete model

#         Returns:
#         - model: Ready-to-use model
#         """
#         if self.use_classifier:
#             model = StandardClassifier(
#                 encoder=model_or_encoder,
#                 num_classes=self.config["num_classes"],
#                 dropout=self.config["classifier_dropout"]
#             )
#         else:
#             model = model_or_encoder
#         return model.to(self.device)

#     def evaluate(
#             self,
#             kg_name: str,
#             model_factory: Callable,
#             init_embd: str,
#             split_path: str,
#             entities_embd_path: Optional[str] = None,
#             edges_embd_path: Optional[str] = None,
#             epochs: int = 100,
#             patience: int = 100,
#             lr: float = 0.01,
#             weight_decay: float = 5e-4,
#             verbose: bool = True
#     ) -> Dict:
#         """
#         Evaluate a model on a single split.

#         Parameters:
#         - kg_name: Knowledge graph name
#         - model_factory: Function that returns a fresh model instance
#         - init_embd: Embedding initialization method
#         - split_path: Path to split file
#         - entities_embd_path: Optional path to entity embeddings
#         - edges_embd_path: Optional path to edge embeddings
#         - epochs: Number of training epochs
#         - patience: Early stopping patience
#         - lr: Learning rate
#         - weight_decay: Weight decay
#         - verbose: Print training progress

#         Returns:
#         - Dictionary with training results
#         """
#         annotated_graph, train_loader, val_loader, test_loader, gdp = self.get_data(
#             kg_name=kg_name,
#             init_embd=init_embd,
#             split_path=split_path,
#             entities_embd_path=entities_embd_path,
#             edges_embd_path=edges_embd_path
#         )

#         if verbose:
#             print(f"\n{'=' * 70}")
#             print(f"Evaluating on split: {split_path}")
#             print(f"Graph: {annotated_graph}")
#             print(f"{'=' * 70}\n")

#         model = model_factory()
#         prepared_model = self.prepare_model(model)

#         trainer = Trainer(
#             model=prepared_model,
#             device=self.device,
#             lr=lr,
#             weight_decay=weight_decay,
#             optimizer_type='adam'
#         )

#         results = trainer.train(
#             train_loader=train_loader,
#             val_loader=val_loader,
#             test_loader=test_loader,
#             epochs=epochs,
#             patience=patience,
#             verbose=verbose
#         )

#         return results

#     def evaluate_with_onto(
#             self,
#             kg_name: str,
#             onto_name: str,
#             model_factory: Callable,
#             init_embd: str,
#             split_path: str,
#             kg_entities_embd_path: Optional[str] = None,
#             kg_edges_embd_path: Optional[str] = None,
#             onto_entities_embd_path: Optional[str] = None,
#             onto_edges_embd_path: Optional[str] = None,
#             epochs: int = 100,
#             patience: int = 100,
#             lr: float = 0.01,
#             weight_decay: float = 5e-4,
#             verbose: bool = True,
#             # --- Ontology visualization ---
#             onto_type_names: Optional[List[str]] = None,
#             onto_sim_save_path: Optional[str] = "onto_type_similarity.png",
#             seed = 42
#     ) -> Dict:
#         """
#         Train a model normally (base Trainer, no onto loss), then after training
#         run a forward pass on the ontology and export the type similarity matrix.

#         Parameters:
#         - kg_name: Knowledge graph name
#         - onto_name: Ontology graph name
#         - model_factory: Function that returns a fresh model instance
#         - init_embd: Embedding initialization method
#         - split_path: Path to split file
#         - kg_entities_embd_path: Optional path to KG entity embeddings
#         - kg_edges_embd_path: Optional path to KG edge embeddings
#         - onto_entities_embd_path: Optional path to ontology entity embeddings
#         - onto_edges_embd_path: Optional path to ontology edge embeddings
#         - epochs: Number of training epochs
#         - patience: Early stopping patience
#         - lr: Learning rate
#         - weight_decay: Weight decay
#         - verbose: Print training progress
#         - onto_type_names: List of type node names (text) to include in the similarity matrix
#         - onto_sim_save_path: Path to save the similarity matrix image

#         Returns:
#         - Dictionary with training results
#         """
#         # Load KG data
#         seed_everything(seed, deterministic=True)
#         annotated_graph, train_loader, val_loader, test_loader, gdp = self.get_data(
#             kg_name=kg_name,
#             init_embd=init_embd,
#             split_path=split_path,
#             entities_embd_path=kg_entities_embd_path,
#             edges_embd_path=kg_edges_embd_path
#         )

#         # Load ontology graph
#         gdp_onto = GraphDataPreparation(
#             kg_name=onto_name,
#             model_name_init=init_embd,
#             entities_embd_path=onto_entities_embd_path,
#             edges_embd_path=onto_edges_embd_path,
#             is_directed=True,
#             with_self_loop=False
#         )
#         onto_data = gdp_onto.prepare_graph_with_type()

#         if verbose:
#             print(f"\n{'=' * 70}")
#             print(f"Evaluating on split : {split_path}")
#             print(f"KG graph            : {annotated_graph}")
#             print(f"Ontology nodes      : {onto_data.num_nodes} | Edges: {onto_data.num_edges}")
#             print(f"{'=' * 70}\n")

#         # Standard training — base Trainer, pas d'onto loss
#         model = model_factory()
#         prepared_model = self.prepare_model(model)

#         # trainer = OntologyTrainer(
#         #                             model=prepared_model,
#         #                             ontology_data=onto_data,
#         #                             device=self.device,
#         #                             lr=lr,
#         #                             weight_decay=weight_decay,
#         #                         )
#         valid_pairs = build_relation_type_mapping(gdp, "datasets/onto_rel.json", gdp_onto)

#         # trainer = OntologyTrainer(
#         #     model=prepared_model,
#         #     ontology_data=onto_data,
#         #     lambda_type=0.25,              # commencer petit
#         #     valid_pairs_per_type=valid_pairs,
#         #     device=self.device,
#         #     lr=lr,
#         #     weight_decay=weight_decay,
#         # )
#         # trainer = StableOntologyTrainer(
#         #                                 model=prepared_model,
#         #                                 ontology_data=onto_data,
#         #                                 lambda_type=0.2,#,0.02,
#         #                                 valid_pairs_per_type=valid_pairs,
#         #                                 device=self.device,
#         #                                 lr=lr,
#         #                                 weight_decay=weight_decay,
#         #                                 optimizer_type='adam'
#         #                             )

#         # trainer = ImprovedOntologyTrainer(
#         #     model                = prepared_model,
#         #     device               = self.device,
#         #     lr                   = lr,
#         #     weight_decay         = weight_decay,
#         #     optimizer_type       = 'adam',
#         #     ontology_data        = onto_data,
#         #     valid_pairs_per_type = valid_pairs,
#         #     lambda_type          = 0.15,
#         #     warmup_epochs        = 20,
#         #     infonce_temperature  = 0.3,
#         #     max_neg_per_rel      = 20,       # ← clé
#         #     soft_align_weight    = 0.5,      # ← nouveau
#         #     type_sep_margin      = 0.3,
#         #     type_sep_weight      = 0.4,
#         #     explicit_neg_type_pairs = [
#         #         ("Organic Chemical", "Pharmacologic Substance"),
#         #         ("Finding", "Intellectual Product"),
#         #         ("Finding", "Laboratory Procedure"),
#         #         ("Pharmacologic Substance", "Laboratory Procedure"),
                
#         #     ],
#         #     explicit_neg_weight  = 0.5,
#         #     use_lr_scheduler     = True,
#         #     lr_patience          = 10,
#         # )

#         # # Obligatoire avant train()
#         # trainer.resolve_explicit_neg_pairs(gdp_onto)
                                                
#         # results = trainer.train_with_display_matrix(
#         #     train_loader=train_loader,
#         #     val_loader=val_loader,

#         #     test_loader=test_loader,
#         #     onto_type_names=[
#         #                     "Body Part, Organ, or Organ Component",
#         #                     "Disease or Syndrome",
#         #                     "Finding",
#         #                     "Intellectual Product",
#         #                     "Laboratory Procedure",
#         #                     "Organic Chemical",
#         #                     "Pharmacologic Substance",
#         #                     "Therapeutic or Preventive Procedure"
#         #                 ],
#         #     gdp=gdp_onto,
#         #     epochs=100,
#         #     onto_sim_save_path="results/onto_similarity.png"
#         # )

#         # return results
#         type_names = onto_type_names or [
#             "Body Part, Organ, or Organ Component",
#             "Disease or Syndrome",
#             "Finding",
#             "Intellectual Product",
#             "Laboratory Procedure",
#             "Organic Chemical",
#             "Pharmacologic Substance",
#             "Therapeutic or Preventive Procedure",
#         ]

#         shared_type_pairs = build_shared_type_indices(
#             type_names=type_names,
#             kg_gdp=gdp,
#             onto_gdp=gdp_onto,
#         )

#         trainer = OntologyAlignmentTrainer(
#             model=prepared_model,
#             device=self.device,
#             lr=lr,
#             weight_decay=weight_decay,
#             optimizer_type="adam",

#             kg_data=annotated_graph,
#             ontology_data=onto_data,
#             shared_type_pairs=shared_type_pairs,

#             lambda_align=0.01,
#             align_batch_size=None,
#             align_num_neighbors=[20, 10],
#         )

#         results = trainer.train(
#             train_loader=train_loader,
#             val_loader=val_loader,
#             test_loader=test_loader,
#             epochs=epochs,
#             patience=patience,
#             verbose=verbose,
#         )

#         return results

#     def evaluate_all(
#             self,
#             kg_name: str,
#             model_factory: Callable,
#             init_embd: str,
#             seeds: List[int],
#             splits_dir: str = "datasets/split",
#             entities_embd_path: Optional[str] = None,
#             edges_embd_path: Optional[str] = None,
#             epochs: int = 100,
#             patience: int = 100,
#             lr: float = 0.01,
#             weight_decay: float = 5e-4,
#             verbose: bool = True,
#             save_results: bool = True,
#             results_dir: str = "results",
#             run_id: Optional[str] = None
#     ) -> Dict:
#         """
#         Evaluate a model across multiple random seeds.

#         Parameters:
#         - kg_name: Knowledge graph name
#         - model_factory: Function that returns a fresh model instance
#         - init_embd: Embedding initialization method
#         - seeds: List of random seeds to evaluate on
#         - splits_dir: Directory containing split files
#         - entities_embd_path: Optional path to entity embeddings
#         - edges_embd_path: Optional path to edge embeddings
#         - epochs: Number of training epochs
#         - patience: Early stopping patience
#         - lr: Learning rate
#         - weight_decay: Weight decay
#         - verbose: Print training progress
#         - save_results: Whether to save results to file
#         - results_dir: Directory to save results
#         - run_id: Unique identifier for this run (used in filenames)

#         Returns:
#         - Dictionary with aggregated results across all seeds
#         """
#         all_results = {
#             'seeds': seeds,
#             'per_seed': [],
#             'aggregated': {}
#         }

#         metrics_per_seed = {
#             'train_acc': [],
#             'train_f1': [],
#             'val_acc': [],
#             'val_f1': [],
#             'test_acc': [],
#             'test_f1': [],
#             'best_epoch': []
#         }

#         print(f"\n{'#' * 70}")
#         print(f"# Evaluating model on {len(seeds)} seeds: {seeds}")
#         print(f"# Knowledge Graph: {kg_name}")
#         print(f"# Embedding: {init_embd}")
#         if run_id:
#             print(f"# Run ID: {run_id}")
#         print(f"{'#' * 70}\n")

#         for i, seed in enumerate(seeds, 1):
#             seed_everything(seed, deterministic=True)
#             print(f"\n{'=' * 70}")
#             print(f"SEED {i}/{len(seeds)}: {seed}")
#             print(f"{'=' * 70}")

#             split_path = f"{splits_dir}/split_{seed}.json"

#             try:
#                 results = self.evaluate(
#                     kg_name=kg_name,
#                     model_factory=model_factory,
#                     init_embd=init_embd,
#                     split_path=split_path,
#                     entities_embd_path=entities_embd_path,
#                     edges_embd_path=edges_embd_path,
#                     epochs=epochs,
#                     patience=patience,
#                     lr=lr,
#                     weight_decay=weight_decay,
#                     verbose=verbose
#                 )

#                 seed_result = {
#                     'seed': seed,
#                     'best_val_f1': results['best_val']['f1'],
#                     'best_epoch': results['best_val']['epoch'],
#                     'final_test': results['final_test'],
#                     'history': results['history']
#                 }
#                 all_results['per_seed'].append(seed_result)

#                 train_acc  = results['history']['train_acc'][-1]
#                 train_f1   = results['history']['train_f1'][-1]
#                 val_acc    = results['history']['val_acc'][-1]
#                 val_f1     = results['history']['val_f1'][-1]
#                 test_acc   = results['final_test']['accuracy']
#                 test_f1    = results['final_test']['f1']

#                 metrics_per_seed['train_acc'].append(train_acc)
#                 metrics_per_seed['train_f1'].append(train_f1)
#                 metrics_per_seed['val_acc'].append(val_acc)
#                 metrics_per_seed['val_f1'].append(val_f1)
#                 metrics_per_seed['test_acc'].append(test_acc)
#                 metrics_per_seed['test_f1'].append(test_f1)
#                 metrics_per_seed['best_epoch'].append(results['best_val']['epoch'])

#                 print(f"\n✓ Seed {seed} completed:")
#                 print(f"  Train: Acc={train_acc:.4f}, F1={train_f1:.4f}")
#                 print(f"  Val:   Acc={val_acc:.4f}, F1={val_f1:.4f}")
#                 print(f"  Test:  Acc={test_acc:.4f}, F1={test_f1:.4f}")

#             except Exception as e:
#                 print(f"\n✗ Error on seed {seed}: {str(e)}")
#                 import traceback
#                 traceback.print_exc()
#                 continue

#         print(f"\n{'#' * 70}")
#         print("# AGGREGATED RESULTS")
#         print(f"{'#' * 70}\n")

#         for metric_name, values in metrics_per_seed.items():
#             if len(values) > 0:
#                 mean = np.mean(values)
#                 std  = np.std(values)
#                 all_results['aggregated'][metric_name] = {
#                     'mean': float(mean),
#                     'std': float(std),
#                     'values': values
#                 }
#                 print(f"{metric_name:15s}: {mean:.4f} ± {std:.4f}")

#         if save_results:
#             results_path = Path(results_dir)
#             results_path.mkdir(parents=True, exist_ok=True)

#             base_name = run_id if run_id else f"{kg_name}_{init_embd.replace('/', '_')}"

#             json_path = results_path / f"results_{base_name}.json"
#             with open(json_path, 'w') as f:
#                 json.dump(all_results, f, indent=2)
#             print(f"\n✓ Detailed results saved to: {json_path}")

#             summary_data = {'metric': [], 'mean': [], 'std': []}
#             for metric_name, stats in all_results['aggregated'].items():
#                 summary_data['metric'].append(metric_name)
#                 summary_data['mean'].append(stats['mean'])
#                 summary_data['std'].append(stats['std'])

#             df_summary = pd.DataFrame(summary_data)
#             csv_path = results_path / f"summary_{base_name}.csv"
#             df_summary.to_csv(csv_path, index=False)
#             print(f"✓ Summary saved to: {csv_path}")

#             per_seed_data = []
#             for seed_result in all_results['per_seed']:
#                 per_seed_data.append({
#                     'seed':       seed_result['seed'],
#                     'train_acc':  seed_result['history']['train_acc'][-1],
#                     'train_f1':   seed_result['history']['train_f1'][-1],
#                     'val_acc':    seed_result['history']['val_acc'][-1],
#                     'val_f1':     seed_result['history']['val_f1'][-1],
#                     'test_acc':   seed_result['final_test']['accuracy'],
#                     'test_f1':    seed_result['final_test']['f1'],
#                     'best_epoch': seed_result['best_epoch']
#                 })

#             df_per_seed = pd.DataFrame(per_seed_data)
#             per_seed_csv = results_path / f"per_seed_{base_name}.csv"
#             df_per_seed.to_csv(per_seed_csv, index=False)
#             print(f"✓ Per-seed results saved to: {per_seed_csv}")

#         return all_results


###################################

#######################################

# from typing import Optional, Dict, List, Callable
# from pathlib import Path
# import json

# import pandas as pd
# import torch
# import numpy as np

# from data_preprocessing.GraphDataPreparation import GraphDataPreparation
# from data_preprocessing.data_manager import get_data_and_loaders
# from data_preprocessing.splits import generate_and_save_splits

# from models.StandardClassifier import StandardClassifier

# from train.Trainer import Trainer
# from train.OntologyAlignmentTrainer import (
#     OntologyAlignmentTrainer,
#     build_shared_type_indices,
# )


# from utilities.utilities import load_config, seed_everything


# class TDGBench:
#     def __init__(self, use_classifier=True, config_path="config.yml"):
#         self.config = load_config(config_path)
#         self.use_classifier = use_classifier
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # ------------------------------------------------------------------
#     # Data
#     # ------------------------------------------------------------------

#     def get_data(
#         self,
#         kg_name="GT2KG_kg",
#         init_embd="sentence-transformers/all-MiniLM-L6-v2",
#         split_path="datasets/split/split_42.json",
#         entities_embd_path=None,
#         edges_embd_path=None,
#         random_embd_dim=256,
#         use_cache=False,
#     ):
#         return get_data_and_loaders(
#             kg_name=kg_name,
#             model_name_init=init_embd,
#             common_nodes_path=self.config["common_nodes_path"],
#             entities_embd_path=entities_embd_path,
#             edges_embd_path=edges_embd_path,
#             split_file=split_path,
#             random_embd_dim=random_embd_dim,
#             use_cache=use_cache,
#         )

#     def generate_splits(self):
#         generate_and_save_splits(
#             self.config["common_nodes_path"],
#             self.config["default_splits_dir"],
#             self.config["n_splits"],
#             self.config["seeds"],
#             self.config["train_ratio"],
#             self.config["val_ratio"],
#             self.config["test_ratio"],
#             self.config["stratify"],
#         )

#     # ------------------------------------------------------------------
#     # Model
#     # ------------------------------------------------------------------

#     def prepare_model(self, model_or_encoder):
#         if self.use_classifier:
#             model = StandardClassifier(
#                 encoder=model_or_encoder,
#                 num_classes=self.config["num_classes"],
#                 dropout=self.config["classifier_dropout"],
#             )
#         else:
#             model = model_or_encoder

#         return model.to(self.device)

#     # ------------------------------------------------------------------
#     # Normal evaluation
#     # ------------------------------------------------------------------

#     def evaluate(
#         self,
#         kg_name: str,
#         model_factory: Callable,
#         init_embd: str,
#         split_path: str,
#         entities_embd_path: Optional[str] = None,
#         edges_embd_path: Optional[str] = None,
#         epochs: int = 100,
#         patience: int = 100,
#         lr: float = 0.01,
#         weight_decay: float = 5e-4,
#         verbose: bool = True,
#     ) -> Dict:

#         annotated_graph, train_loader, val_loader, test_loader, gdp = self.get_data(
#             kg_name=kg_name,
#             init_embd=init_embd,
#             split_path=split_path,
#             entities_embd_path=entities_embd_path,
#             edges_embd_path=edges_embd_path,
#         )

#         if verbose:
#             print(f"\n{'=' * 70}")
#             print(f"Normal evaluation")
#             print(f"Split : {split_path}")
#             print(f"KG    : {kg_name}")
#             print(f"Graph : {annotated_graph}")
#             print(f"{'=' * 70}\n")

#         model = model_factory()
#         prepared_model = self.prepare_model(model)

#         trainer = Trainer(
#             model=prepared_model,
#             device=self.device,
#             lr=lr,
#             weight_decay=weight_decay,
#             optimizer_type="adam",
#         )

#         results = trainer.train(
#             train_loader=train_loader,
#             val_loader=val_loader,
#             test_loader=test_loader,
#             epochs=epochs,
#             patience=patience,
#             verbose=verbose,
#         )

#         return results

#     # ------------------------------------------------------------------
#     # Ontology alignment evaluation
#     # ------------------------------------------------------------------

#     def evaluate_with_onto(
#         self,
#         kg_name: str,
#         onto_name: str,
#         model_factory: Callable,
#         init_embd: str,
#         split_path: str,
#         kg_entities_embd_path: Optional[str] = None,
#         kg_edges_embd_path: Optional[str] = None,
#         onto_entities_embd_path: Optional[str] = None,
#         onto_edges_embd_path: Optional[str] = None,
#         epochs: int = 100,
#         patience: int = 100,
#         lr: float = 0.01,
#         weight_decay: float = 5e-4,
#         verbose: bool = True,
#         onto_type_names: Optional[List[str]] = None,
#         lambda_align: float = 0.01,
#         align_batch_size: Optional[int] = None,
#         align_num_neighbors: Optional[List[int]] = None,
#         seed: int = 42,
#     ) -> Dict:

#         if align_num_neighbors is None:
#             align_num_neighbors = [200,200]

#         seed_everything(seed, deterministic=True)

#         annotated_graph, train_loader, val_loader, test_loader, gdp = self.get_data(
#             kg_name=kg_name,
#             init_embd=init_embd,
#             split_path=split_path,
#             entities_embd_path=kg_entities_embd_path,
#             edges_embd_path=kg_edges_embd_path,
#         )

#         gdp_onto = GraphDataPreparation(
#             kg_name=onto_name,
#             model_name_init=init_embd,
#             entities_embd_path=onto_entities_embd_path,
#             edges_embd_path=onto_edges_embd_path,
#             is_directed=True,
#             with_self_loop=False,
#         )

#         onto_data = gdp_onto.prepare_graph_with_type()

#         type_names = onto_type_names or [
#             "Body Part, Organ, or Organ Component",
#             "Disease or Syndrome",
#             "Finding",
#             "Intellectual Product",
#             "Laboratory Procedure",
#             "Organic Chemical",
#             "Pharmacologic Substance",
#             "Therapeutic or Preventive Procedure",
#         ]

#         shared_type_pairs = build_shared_type_indices(
#             type_names=type_names,
#             kg_gdp=gdp,
#             onto_gdp=gdp_onto,
#         )

#         if verbose:
#             print(f"\n{'=' * 70}")
#             print(f"Ontology alignment evaluation")
#             print(f"Split          : {split_path}")
#             print(f"KG             : {kg_name}")
#             print(f"Ontology       : {onto_name}")
#             print(f"KG graph       : {annotated_graph}")
#             print(f"Ontology nodes : {onto_data.num_nodes} | Edges: {onto_data.num_edges}")
#             print(f"Shared types   : {len(shared_type_pairs)}")
#             print(f"lambda_align   : {lambda_align}")
#             print(f"{'=' * 70}\n")

#         model = model_factory()
#         prepared_model = self.prepare_model(model)

#         trainer = OntologyAlignmentTrainer(
#             model=prepared_model,
#             device=self.device,
#             lr=lr,
#             weight_decay=weight_decay,
#             optimizer_type="adam",
#             kg_data=annotated_graph,
#             ontology_data=onto_data,
#             shared_type_pairs=shared_type_pairs,
#             lambda_align=lambda_align,
#             align_batch_size=align_batch_size,
#             align_num_neighbors=align_num_neighbors,
#         )

#         results = trainer.train(
#             train_loader=train_loader,
#             val_loader=val_loader,
#             test_loader=test_loader,
#             epochs=epochs,
#             patience=patience,
#             verbose=verbose,
#         )

#         return results

#     # ------------------------------------------------------------------
#     # Multi-seed evaluation with aggregation
#     # ------------------------------------------------------------------

#     def evaluate_all(
#         self,
#         kg_name: str,
#         model_factory: Callable,
#         init_embd: str,
#         seeds: List[int],
#         splits_dir: str = "datasets/split",
#         entities_embd_path: Optional[str] = None,
#         edges_embd_path: Optional[str] = None,
#         epochs: int = 100,
#         patience: int = 100,
#         lr: float = 0.01,
#         weight_decay: float = 5e-4,
#         verbose: bool = True,
#         save_results: bool = True,
#         results_dir: str = "results",
#         run_id: Optional[str] = None,
#         onto_incorporation: Optional[str] = None,
#         onto_name: str = "UMLS_NCI",
#         onto_entities_embd_path: Optional[str] = None,
#         onto_edges_embd_path: Optional[str] = None,
#         onto_type_names: Optional[List[str]] = None,
#         lambda_align: float = 0.01,
#         align_batch_size: Optional[int] = None,
#         align_num_neighbors: Optional[List[int]] = None,
#     ) -> Dict:

#         if onto_incorporation not in [None, "align"]:
#             raise ValueError(
#                 f"Unknown onto_incorporation={onto_incorporation}. "
#                 "Supported values are None or 'align'."
#             )

#         all_results = {
#             "seeds": seeds,
#             "kg_name": kg_name,
#             "init_embd": init_embd,
#             "onto_incorporation": onto_incorporation,
#             "onto_name": onto_name if onto_incorporation == "align" else None,
#             "per_seed": [],
#             "aggregated": {},
#         }

#         metrics_per_seed = {
#             "train_acc": [],
#             "train_f1": [],
#             "val_acc": [],
#             "val_f1": [],
#             "test_acc": [],
#             "test_f1": [],
#             "best_epoch": [],
#         }

#         print(f"\n{'#' * 70}")
#         print(f"# Evaluating model on {len(seeds)} seeds: {seeds}")
#         print(f"# Knowledge Graph      : {kg_name}")
#         print(f"# Embedding            : {init_embd}")
#         print(f"# Onto incorporation   : {onto_incorporation}")
#         if onto_incorporation == "align":
#             print(f"# Ontology             : {onto_name}")
#             print(f"# lambda_align         : {lambda_align}")
#         if run_id:
#             print(f"# Run ID               : {run_id}")
#         print(f"{'#' * 70}\n")

#         for i, seed in enumerate(seeds, 1):
#             seed_everything(seed, deterministic=True)

#             print(f"\n{'=' * 70}")
#             print(f"SEED {i}/{len(seeds)}: {seed}")
#             print(f"{'=' * 70}")

#             split_path = f"{splits_dir}/split_{seed}.json"

#             try:
#                 if onto_incorporation == "align":
#                     results = self.evaluate_with_onto(
#                         kg_name=kg_name,
#                         onto_name=onto_name,
#                         model_factory=model_factory,
#                         init_embd=init_embd,
#                         split_path=split_path,
#                         kg_entities_embd_path=entities_embd_path,
#                         kg_edges_embd_path=edges_embd_path,
#                         onto_entities_embd_path=onto_entities_embd_path,
#                         onto_edges_embd_path=onto_edges_embd_path,
#                         epochs=epochs,
#                         patience=patience,
#                         lr=lr,
#                         weight_decay=weight_decay,
#                         verbose=verbose,
#                         onto_type_names=onto_type_names,
#                         lambda_align=lambda_align,
#                         align_batch_size=align_batch_size,
#                         align_num_neighbors=align_num_neighbors,
#                         seed=seed,
#                     )
#                 else:
#                     results = self.evaluate(
#                         kg_name=kg_name,
#                         model_factory=model_factory,
#                         init_embd=init_embd,
#                         split_path=split_path,
#                         entities_embd_path=entities_embd_path,
#                         edges_embd_path=edges_embd_path,
#                         epochs=epochs,
#                         patience=patience,
#                         lr=lr,
#                         weight_decay=weight_decay,
#                         verbose=verbose,
#                     )

#                 seed_result = {
#                     "seed": seed,
#                     "best_val_f1": results["best_val"]["f1"],
#                     "best_epoch": results["best_val"]["epoch"],
#                     "final_test": results["final_test"],
#                     "history": results["history"],
#                 }

#                 all_results["per_seed"].append(seed_result)

#                 train_acc = results["history"]["train_acc"][-1]
#                 train_f1 = results["history"]["train_f1"][-1]
#                 val_acc = results["history"]["val_acc"][-1]
#                 val_f1 = results["history"]["val_f1"][-1]
#                 test_acc = results["final_test"]["accuracy"]
#                 test_f1 = results["final_test"]["f1"]
#                 best_epoch = results["best_val"]["epoch"]

#                 metrics_per_seed["train_acc"].append(train_acc)
#                 metrics_per_seed["train_f1"].append(train_f1)
#                 metrics_per_seed["val_acc"].append(val_acc)
#                 metrics_per_seed["val_f1"].append(val_f1)
#                 metrics_per_seed["test_acc"].append(test_acc)
#                 metrics_per_seed["test_f1"].append(test_f1)
#                 metrics_per_seed["best_epoch"].append(best_epoch)

#                 print(f"\n✓ Seed {seed} completed:")
#                 print(f"  Train: Acc={train_acc:.4f}, F1={train_f1:.4f}")
#                 print(f"  Val:   Acc={val_acc:.4f}, F1={val_f1:.4f}")
#                 print(f"  Test:  Acc={test_acc:.4f}, F1={test_f1:.4f}")
#                 print(f"  Best epoch: {best_epoch}")

#             except Exception as e:
#                 print(f"\n✗ Error on seed {seed}: {str(e)}")
#                 import traceback

#                 traceback.print_exc()
#                 continue

#         print(f"\n{'#' * 70}")
#         print("# AGGREGATED RESULTS")
#         print(f"{'#' * 70}\n")

#         for metric_name, values in metrics_per_seed.items():
#             if len(values) > 0:
#                 mean = np.mean(values)
#                 std = np.std(values)

#                 all_results["aggregated"][metric_name] = {
#                     "mean": float(mean),
#                     "std": float(std),
#                     "values": values,
#                 }

#                 print(f"{metric_name:15s}: {mean:.4f} ± {std:.4f}")

#         if save_results:
#             results_path = Path(results_dir)
#             results_path.mkdir(parents=True, exist_ok=True)

#             if run_id:
#                 base_name = run_id
#             else:
#                 embd_name = init_embd.replace("/", "_")
#                 suffix = f"_onto_{onto_incorporation}" if onto_incorporation else ""
#                 base_name = f"{kg_name}_{embd_name}{suffix}"

#             json_path = results_path / f"results_{base_name}.json"
#             with open(json_path, "w") as f:
#                 json.dump(all_results, f, indent=2)

#             print(f"\n✓ Detailed results saved to: {json_path}")

#             summary_data = {
#                 "metric": [],
#                 "mean": [],
#                 "std": [],
#             }

#             for metric_name, stats in all_results["aggregated"].items():
#                 summary_data["metric"].append(metric_name)
#                 summary_data["mean"].append(stats["mean"])
#                 summary_data["std"].append(stats["std"])

#             df_summary = pd.DataFrame(summary_data)
#             csv_path = results_path / f"summary_{base_name}.csv"
#             df_summary.to_csv(csv_path, index=False)

#             print(f"✓ Summary saved to: {csv_path}")

#             per_seed_data = []

#             for seed_result in all_results["per_seed"]:
#                 per_seed_data.append(
#                     {
#                         "seed": seed_result["seed"],
#                         "onto_incorporation": onto_incorporation,
#                         "train_acc": seed_result["history"]["train_acc"][-1],
#                         "train_f1": seed_result["history"]["train_f1"][-1],
#                         "val_acc": seed_result["history"]["val_acc"][-1],
#                         "val_f1": seed_result["history"]["val_f1"][-1],
#                         "test_acc": seed_result["final_test"]["accuracy"],
#                         "test_f1": seed_result["final_test"]["f1"],
#                         "best_epoch": seed_result["best_epoch"],
#                     }
#                 )

#             df_per_seed = pd.DataFrame(per_seed_data)
#             per_seed_csv = results_path / f"per_seed_{base_name}.csv"
#             df_per_seed.to_csv(per_seed_csv, index=False)

#             print(f"✓ Per-seed results saved to: {per_seed_csv}")

#         return all_results

################################################### V3 ######################################################
#############################################################################################################

from typing import Optional, Dict, List, Callable
from pathlib import Path
import json

import pandas as pd
import torch
import numpy as np

from data_preprocessing.GraphDataPreparation import GraphDataPreparation
from data_preprocessing.data_manager import get_data_and_loaders
from data_preprocessing.splits import generate_and_save_splits

from models.StandardClassifier import StandardClassifier

from train.Trainer import Trainer
from train.OntologyAlignmentTrainer import (
    OntologyAlignmentTrainer,
    build_shared_type_indices,
)

from utilities.utilities import load_config, seed_everything

class TDGBench:
    def __init__(self, use_classifier=True, config_path="config.yml"):
        self.config = load_config(config_path)
        self.use_classifier = use_classifier
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def get_data(
        self,
        kg_name="GT2KG_kg",
        init_embd="sentence-transformers/all-MiniLM-L6-v2",
        split_path="datasets/split/split_42.json",
        entities_embd_path=None,
        edges_embd_path=None,
        random_embd_dim=256,
        use_cache=False,
    ):
        return get_data_and_loaders(
            kg_name=kg_name,
            model_name_init=init_embd,
            common_nodes_path=self.config["common_nodes_path"],
            entities_embd_path=entities_embd_path,
            edges_embd_path=edges_embd_path,
            split_file=split_path,
            random_embd_dim=random_embd_dim,
            use_cache=use_cache,
        )

    def generate_splits(self):
        generate_and_save_splits(
            self.config["common_nodes_path"],
            self.config["default_splits_dir"],
            self.config["n_splits"],
            self.config["seeds"],
            self.config["train_ratio"],
            self.config["val_ratio"],
            self.config["test_ratio"],
            self.config["stratify"],
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def prepare_model(self, model_or_encoder):
        if self.use_classifier:
            model = StandardClassifier(
                encoder=model_or_encoder,
                num_classes=self.config["num_classes"],
                dropout=self.config["classifier_dropout"],
            )
        else:
            model = model_or_encoder

        return model.to(self.device)

    # ------------------------------------------------------------------
    # Normal evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        kg_name: str,
        model_factory: Callable,
        init_embd: str,
        split_path: str,
        entities_embd_path: Optional[str] = None,
        edges_embd_path: Optional[str] = None,
        epochs: int = 100,
        patience: int = 100,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        verbose: bool = True,
        artifacts_dir: Optional[str] = None,
        artifact_prefix: str = "run",
        save_prediction_splits: Optional[List[str]] = None,
    ) -> Dict:

        annotated_graph, train_loader, val_loader, test_loader, gdp = self.get_data(
            kg_name=kg_name,
            init_embd=init_embd,
            split_path=split_path,
            entities_embd_path=entities_embd_path,
            edges_embd_path=edges_embd_path,
        )
        if hasattr(annotated_graph, "num_classes"):
            self.config["num_classes"] = int(annotated_graph.num_classes)

        if verbose:
            print(f"\n{'=' * 70}")
            print("Normal evaluation")
            print(f"Split : {split_path}")
            print(f"KG    : {kg_name}")
            print(f"Classes: {self.config['num_classes']}")
            print(f"Graph : {annotated_graph}")
            print(f"{'=' * 70}\n")

        model = model_factory()
        prepared_model = self.prepare_model(model)

        trainer = Trainer(
            model=prepared_model,
            device=self.device,
            lr=lr,
            weight_decay=weight_decay,
            optimizer_type="adam",
        )

        index_to_term = gdp.decode_indexes()
        decode_indexes_fn = lambda node_id: index_to_term.get(int(node_id), int(node_id))

        return trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=epochs,
            patience=patience,
            verbose=verbose,
            artifacts_dir=artifacts_dir,
            artifact_prefix=artifact_prefix,
            label_encoder=gdp.label_encoder,
            decode_indexes_fn=decode_indexes_fn,
            save_prediction_splits=save_prediction_splits,
        )

    # ------------------------------------------------------------------
    # Ontology alignment evaluation
    # ------------------------------------------------------------------

    def evaluate_with_onto(
        self,
        kg_name: str,
        onto_name: str,
        model_factory: Callable,
        init_embd: str,
        split_path: str,
        kg_entities_embd_path: Optional[str] = None,
        kg_edges_embd_path: Optional[str] = None,
        onto_entities_embd_path: Optional[str] = None,
        onto_edges_embd_path: Optional[str] = None,
        epochs: int = 100,
        patience: int = 100,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        verbose: bool = True,
        onto_type_names: Optional[List[str]] = None,
        lambda_align: float = 0.01,
        alignment_mode: str = "cosine",      # "cosine" ou "contrastive"
        temperature: float = 0.2,            # utilisé seulement avec contrastive
        align_batch_size: Optional[int] = None,
        align_num_neighbors: Optional[List[int]] = None,
        seed: int = 42,
    ) -> Dict:

        if align_num_neighbors is None:
            align_num_neighbors = [200, 200]

        seed_everything(seed, deterministic=True)

        annotated_graph, train_loader, val_loader, test_loader, gdp = self.get_data(
            kg_name=kg_name,
            init_embd=init_embd,
            split_path=split_path,
            entities_embd_path=kg_entities_embd_path,
            edges_embd_path=kg_edges_embd_path,
        )
        if hasattr(annotated_graph, "num_classes"):
            self.config["num_classes"] = int(annotated_graph.num_classes)

        gdp_onto = GraphDataPreparation(
            kg_name=onto_name,
            model_name_init=init_embd,
            entities_embd_path=onto_entities_embd_path,
            edges_embd_path=onto_edges_embd_path,
            is_directed=True,
            with_self_loop=False,
        )

        onto_data = gdp_onto.prepare_graph_with_type()

        type_names = onto_type_names or [
            "Body Part, Organ, or Organ Component",
            "Disease or Syndrome",
            "Finding",
            "Intellectual Product",
            "Laboratory Procedure",
            "Organic Chemical",
            "Pharmacologic Substance",
            "Therapeutic or Preventive Procedure",
        ]

        shared_type_pairs = build_shared_type_indices(
            type_names=type_names,
            kg_gdp=gdp,
            onto_gdp=gdp_onto,
        )

        if verbose:
            print(f"\n{'=' * 70}")
            print("Ontology alignment evaluation")
            print(f"Split          : {split_path}")
            print(f"KG             : {kg_name}")
            print(f"Ontology       : {onto_name}")
            print(f"KG graph       : {annotated_graph}")
            print(f"Ontology nodes : {onto_data.num_nodes} | Edges: {onto_data.num_edges}")
            print(f"Shared types   : {len(shared_type_pairs)}")
            print(f"lambda_align   : {lambda_align}")
            print(f"alignment_mode : {alignment_mode}")
            print(f"temperature    : {temperature}")
            print(f"neighbors      : {align_num_neighbors}")
            print(f"{'=' * 70}\n")

        model = model_factory()
        prepared_model = self.prepare_model(model)

        trainer = OntologyAlignmentTrainer(
            model=prepared_model,
            device=self.device,
            lr=lr,
            weight_decay=weight_decay,
            optimizer_type="adam",
            kg_data=annotated_graph,
            ontology_data=onto_data,
            shared_type_pairs=shared_type_pairs,
            lambda_align=lambda_align,
            alignment_mode=alignment_mode,
            temperature=temperature,
            align_batch_size=align_batch_size,
            align_num_neighbors=align_num_neighbors,
        )

        return trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=epochs,
            patience=patience,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Multi-seed evaluation with aggregation
    # ------------------------------------------------------------------

    def evaluate_all(
        self,
        kg_name: str,
        model_factory: Callable,
        init_embd: str,
        seeds: List[int],
        splits_dir: str = "datasets/split",
        entities_embd_path: Optional[str] = None,
        edges_embd_path: Optional[str] = None,
        epochs: int = 100,
        patience: int = 100,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        verbose: bool = True,
        save_results: bool = True,
        results_dir: str = "results",
        run_id: Optional[str] = None,
        onto_incorporation: Optional[str] = None,
        onto_name: str = "UMLS_NCI",
        onto_entities_embd_path: Optional[str] = None,
        onto_edges_embd_path: Optional[str] = None,
        onto_type_names: Optional[List[str]] = None,
        lambda_align: float = 0.01,
        alignment_mode: str = "cosine",      # "cosine" ou "contrastive"
        temperature: float = 0.2,
        align_batch_size: Optional[int] = None,
        align_num_neighbors: Optional[List[int]] = None,
    ) -> Dict:

        if onto_incorporation not in [None, "align"]:
            raise ValueError(
                f"Unknown onto_incorporation={onto_incorporation}. "
                "Supported values are None or 'align'."
            )

        if alignment_mode not in ["cosine", "contrastive"]:
            raise ValueError(
                f"Unknown alignment_mode={alignment_mode}. "
                "Supported values are 'cosine' or 'contrastive'."
            )

        all_results = {
            "seeds": seeds,
            "kg_name": kg_name,
            "init_embd": init_embd,
            "onto_incorporation": onto_incorporation,
            "onto_name": onto_name if onto_incorporation == "align" else None,
            "lambda_align": lambda_align if onto_incorporation == "align" else None,
            "alignment_mode": alignment_mode if onto_incorporation == "align" else None,
            "temperature": temperature if onto_incorporation == "align" else None,
            "per_seed": [],
            "aggregated": {},
        }

        metrics_per_seed = {
            "train_acc": [],
            "train_f1": [],
            "val_acc": [],
            "val_f1": [],
            "test_acc": [],
            "test_f1": [],
            "best_epoch": [],
        }

        print(f"\n{'#' * 70}")
        print(f"# Evaluating model on {len(seeds)} seeds: {seeds}")
        print(f"# Knowledge Graph      : {kg_name}")
        print(f"# Embedding            : {init_embd}")
        print(f"# Onto incorporation   : {onto_incorporation}")

        if onto_incorporation == "align":
            print(f"# Ontology             : {onto_name}")
            print(f"# lambda_align         : {lambda_align}")
            print(f"# alignment_mode       : {alignment_mode}")
            print(f"# temperature          : {temperature}")

        if run_id:
            print(f"# Run ID               : {run_id}")

        print(f"{'#' * 70}\n")

        for i, seed in enumerate(seeds, 1):
            seed_everything(seed, deterministic=True)

            print(f"\n{'=' * 70}")
            print(f"SEED {i}/{len(seeds)}: {seed}")
            print(f"{'=' * 70}")

            split_path = f"{splits_dir}/split_{seed}.json"
            seed_artifacts_dir = str(Path(results_dir) / "artifacts" / f"seed_{seed}")
            seed_artifact_prefix = run_id if run_id else f"{kg_name}_{init_embd.replace('/', '_')}"
            seed_artifact_prefix = f"{seed_artifact_prefix}_seed{seed}"

            try:
                if onto_incorporation == "align":
                    results = self.evaluate_with_onto(
                        kg_name=kg_name,
                        onto_name=onto_name,
                        model_factory=model_factory,
                        init_embd=init_embd,
                        split_path=split_path,
                        kg_entities_embd_path=entities_embd_path,
                        kg_edges_embd_path=edges_embd_path,
                        onto_entities_embd_path=onto_entities_embd_path,
                        onto_edges_embd_path=onto_edges_embd_path,
                        epochs=epochs,
                        patience=patience,
                        lr=lr,
                        weight_decay=weight_decay,
                        verbose=verbose,
                        onto_type_names=onto_type_names,
                        lambda_align=lambda_align,
                        alignment_mode=alignment_mode,
                        temperature=temperature,
                        align_batch_size=align_batch_size,
                        align_num_neighbors=align_num_neighbors,
                        seed=seed,
                    )
                else:
                    results = self.evaluate(
                        kg_name=kg_name,
                        model_factory=model_factory,
                        init_embd=init_embd,
                        split_path=split_path,
                        entities_embd_path=entities_embd_path,
                        edges_embd_path=edges_embd_path,
                        epochs=epochs,
                        patience=patience,
                        lr=lr,
                        weight_decay=weight_decay,
                        verbose=verbose,
                        artifacts_dir=seed_artifacts_dir,
                        artifact_prefix=seed_artifact_prefix,
                        save_prediction_splits=["test"],
                    )

                seed_result = {
                    "seed": seed,
                    "best_val_f1": results["best_val"]["f1"],
                    "best_epoch": results["best_val"]["epoch"],
                    "final_test": results["final_test"],
                    "history": results["history"],
                    "onto_incorporation": onto_incorporation,
                    "onto_name": onto_name if onto_incorporation == "align" else None,
                    "lambda_align": lambda_align if onto_incorporation == "align" else None,
                    "alignment_mode": alignment_mode if onto_incorporation == "align" else None,
                    "temperature": temperature if onto_incorporation == "align" else None,
                    "artifacts": results.get("artifacts", {}),
                }

                all_results["per_seed"].append(seed_result)

                train_acc = results["history"]["train_acc"][-1]
                train_f1 = results["history"]["train_f1"][-1]
                val_acc = results["history"]["val_acc"][-1]
                val_f1 = results["history"]["val_f1"][-1]
                test_acc = results["final_test"]["accuracy"]
                test_f1 = results["final_test"]["f1"]
                best_epoch = results["best_val"]["epoch"]

                metrics_per_seed["train_acc"].append(train_acc)
                metrics_per_seed["train_f1"].append(train_f1)
                metrics_per_seed["val_acc"].append(val_acc)
                metrics_per_seed["val_f1"].append(val_f1)
                metrics_per_seed["test_acc"].append(test_acc)
                metrics_per_seed["test_f1"].append(test_f1)
                metrics_per_seed["best_epoch"].append(best_epoch)

                print(f"\n✓ Seed {seed} completed:")
                print(f"  Train: Acc={train_acc:.4f}, F1={train_f1:.4f}")
                print(f"  Val:   Acc={val_acc:.4f}, F1={val_f1:.4f}")
                print(f"  Test:  Acc={test_acc:.4f}, F1={test_f1:.4f}")
                print(f"  Best epoch: {best_epoch}")

            except Exception as e:
                print(f"\n✗ Error on seed {seed}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\n{'#' * 70}")
        print("# AGGREGATED RESULTS")
        print(f"{'#' * 70}\n")

        for metric_name, values in metrics_per_seed.items():
            if len(values) > 0:
                mean = np.mean(values)
                std = np.std(values)

                all_results["aggregated"][metric_name] = {
                    "mean": float(mean),
                    "std": float(std),
                    "values": values,
                }

                print(f"{metric_name:15s}: {mean:.4f} ± {std:.4f}")

        if save_results:
            results_path = Path(results_dir)
            results_path.mkdir(parents=True, exist_ok=True)

            if run_id:
                base_name = run_id
            else:
                embd_name = init_embd.replace("/", "_")
                suffix = f"_onto_{onto_incorporation}" if onto_incorporation else ""
                mode_suffix = (
                    f"_{alignment_mode}_temp{temperature}"
                    if onto_incorporation == "align"
                    else ""
                )
                base_name = f"{kg_name}_{embd_name}{suffix}{mode_suffix}"

            json_path = results_path / f"results_{base_name}.json"

            with open(json_path, "w") as f:
                json.dump(all_results, f, indent=2)

            print(f"\n✓ Detailed results saved to: {json_path}")

            summary_data = {
                "metric": [],
                "mean": [],
                "std": [],
            }

            for metric_name, stats in all_results["aggregated"].items():
                summary_data["metric"].append(metric_name)
                summary_data["mean"].append(stats["mean"])
                summary_data["std"].append(stats["std"])

            df_summary = pd.DataFrame(summary_data)
            csv_path = results_path / f"summary_{base_name}.csv"
            df_summary.to_csv(csv_path, index=False)

            print(f"✓ Summary saved to: {csv_path}")

            per_seed_data = []

            for seed_result in all_results["per_seed"]:
                per_seed_data.append(
                    {
                        "seed": seed_result["seed"],
                        "onto_incorporation": onto_incorporation,
                        "onto_name": seed_result["onto_name"],
                        "lambda_align": seed_result["lambda_align"],
                        "alignment_mode": seed_result["alignment_mode"],
                        "temperature": seed_result["temperature"],
                        "train_acc": seed_result["history"]["train_acc"][-1],
                        "train_f1": seed_result["history"]["train_f1"][-1],
                        "val_acc": seed_result["history"]["val_acc"][-1],
                        "val_f1": seed_result["history"]["val_f1"][-1],
                        "test_acc": seed_result["final_test"]["accuracy"],
                        "test_f1": seed_result["final_test"]["f1"],
                        "best_epoch": seed_result["best_epoch"],
                        "best_model_path": seed_result["artifacts"].get("best_model"),
                        "test_predictions_path": seed_result["artifacts"]
                        .get("predictions", {})
                        .get("test"),
                    }
                )

            df_per_seed = pd.DataFrame(per_seed_data)
            per_seed_csv = results_path / f"per_seed_{base_name}.csv"
            df_per_seed.to_csv(per_seed_csv, index=False)

            print(f"✓ Per-seed results saved to: {per_seed_csv}")

        return all_results
