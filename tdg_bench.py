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

from typing import Any, Iterable, Mapping, Optional, Dict, List, Callable
from pathlib import Path
import json
import warnings
from itertools import product

import pandas as pd
import torch
import numpy as np

from data_preprocessing.GraphDataPreparation import GraphDataPreparation
from data_preprocessing.data_manager import get_data_and_loaders
from data_preprocessing.benchmark_registry import resolve_benchmark_protocol
from data_preprocessing.no_graph_data import (
    build_no_graph_features,
    load_no_graph_dataset,
    load_split_indices,
)
from evaluate.evaluation_statistics import aggregate_split_and_randomness
from data_preprocessing.graph_catalog import available_graph_names, validate_graph_name
from data_preprocessing.graph_variants import get_graph, graph_variant_name
from build_models import available_model_names, build_encoder, get_default_model_kwargs
from data_preprocessing.splits import PREGENERATED_SPLIT_SEEDS, generate_and_save_splits

from models.StandardClassifier import StandardClassifier
from models.NoGraphMLP import NoGraphMLP
from train.NoGraphTrainer import NoGraphTrainer

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

    @staticmethod
    def available_graph_names() -> tuple[str, ...]:
        """Return every graph name available as ``datasets/<name>.json``.

        The catalogue is discovered from the repository rather than maintained
        manually.  Consequently, newly added graphs and graph variants become
        available to :meth:`get_data` as soon as their JSON file is placed in
        ``datasets``.  This includes the scale-matched clean references.
        """
        return available_graph_names()

    @classmethod
    def _validate_graph_name(cls, kg_name: str) -> str:
        """Validate a graph name before embeddings or loaders are built."""
        return validate_graph_name(kg_name)

    def get_data(
        self,
        kg_name="GT2KG_kg",
        init_embd="sentence-transformers/all-MiniLM-L6-v2",
        split_path="datasets/split/split_42.json",
        entities_embd_path=None,
        edges_embd_path=None,
        random_embd_dim=256,
        use_cache=False,
        options: Iterable[str] | None = None,
        save_variant: bool = False,
    ):
        """Build data/loaders for a named graph and optional preprocessing variant.

        Examples of registered scale-matched references are
        ``UMLS_nci_kg_scale_matched_GT2KG_kg`` and
        ``UMLS_nci_kg_scale_matched_KG_GEN_kg``.

        ``options`` is passed to :func:`data_preprocessing.graph_variants.get_graph`.
        Valid values are ``raw``, ``no_smnt``, ``no_freq_filter`` and, only for
        ``UMLS_nci_kg``, ``with_inverse``.  With no options, the stored default
        graph is used.  Variants are prepared in memory; set ``save_variant``
        to persist a generated JSON variant.
        """
        kg_name = self._validate_graph_name(kg_name)
        resolved_options = tuple(options or ())
        graph_records = get_graph(
            kg_name=kg_name,
            options=resolved_options,
            save=save_variant,
        )
        graph_identity = graph_variant_name(kg_name, resolved_options)
        return get_data_and_loaders(
            kg_name=graph_identity,
            model_name_init=init_embd,
            common_nodes_path=self.config["common_nodes_path"],
            entities_embd_path=entities_embd_path,
            edges_embd_path=edges_embd_path,
            split_file=split_path,
            random_embd_dim=random_embd_dim,
            use_cache=use_cache,
            graph_records=graph_records,
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
        random_embd_dim: int = 256,
        epochs: int = 100,
        patience: int = 100,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        verbose: bool = True,
        artifacts_dir: Optional[str] = None,
        artifact_prefix: str = "run",
        save_prediction_splits: Optional[List[str]] = None,
        save_models: bool = True,
        options: Iterable[str] | None = None,
        save_variant: bool = False,
    ) -> Dict:
        options = tuple(options or ())
        annotated_graph, train_loader, val_loader, test_loader, gdp = self.get_data(
            kg_name=kg_name,
            init_embd=init_embd,
            split_path=split_path,
            entities_embd_path=entities_embd_path,
            edges_embd_path=edges_embd_path,
            random_embd_dim=random_embd_dim,
            options=options,
            save_variant=save_variant,
        )
        if hasattr(annotated_graph, "num_classes"):
            self.config["num_classes"] = int(annotated_graph.num_classes)

        if verbose:
            print(f"\n{'=' * 70}")
            print("Normal evaluation")
            print(f"Split : {split_path}")
            print(f"KG    : {kg_name}")
            print(f"Options: {list(options)}")
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
            save_model_checkpoint=save_models,
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
        options: Iterable[str] | None = None,
        save_variant: bool = False,
    ) -> Dict:
        options = tuple(options or ())
        if align_num_neighbors is None:
            align_num_neighbors = [200, 200]

        seed_everything(seed, deterministic=True)

        annotated_graph, train_loader, val_loader, test_loader, gdp = self.get_data(
            kg_name=kg_name,
            init_embd=init_embd,
            split_path=split_path,
            entities_embd_path=kg_entities_embd_path,
            edges_embd_path=kg_edges_embd_path,
            options=options,
            save_variant=save_variant,
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
            print(f"Options        : {list(options)}")
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
        seeds: Optional[List[int]] = None,
        splits_dir: str = "datasets/split",
        entities_embd_path: Optional[str] = None,
        edges_embd_path: Optional[str] = None,
        random_embd_dim: int = 256,
        epochs: int = 100,
        patience: int = 100,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        verbose: bool = True,
        save_results: bool = True,
        save_models: bool = True,
        save_predictions: bool = True,
        resume: bool = True,
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
        options: Iterable[str] | None = None,
        save_variant: bool = False,
        split_seeds: Optional[List[int]] = None,
        random_seeds: Optional[List[int]] = None,
    ) -> Dict:
        options = tuple(options or ())
        legacy_coupled_seeds = seeds is not None
        if seeds is not None and (split_seeds is not None or random_seeds is not None):
            raise ValueError("Use either legacy 'seeds' or both 'split_seeds' and 'random_seeds', not both.")
        if seeds is not None:
            warnings.warn(
                "'seeds' couples split selection and training randomness. "
                "Use 'split_seeds' and 'random_seeds' to keep both variance sources separate.",
                DeprecationWarning,
                stacklevel=2,
            )
            split_seeds = list(seeds)
            random_seeds = list(seeds)
        if split_seeds is None or random_seeds is None:
            raise ValueError("evaluate_all requires both split_seeds and random_seeds")
        split_seeds = list(split_seeds)
        random_seeds = list(random_seeds)
        if not split_seeds or not random_seeds:
            raise ValueError("split_seeds and random_seeds must both be non-empty")
        if random_embd_dim < 1:
            raise ValueError("random_embd_dim must be >= 1")
        run_pairs = (
            list(zip(split_seeds, random_seeds))
            if legacy_coupled_seeds
            else list(product(split_seeds, random_seeds))
        )
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

        graph_identity = graph_variant_name(kg_name, options)
        all_results = {
            "kg_name": kg_name,
            "graph_variant": graph_identity,
            "options": list(options),
            "init_embd": init_embd,
            "random_embd_dim": random_embd_dim,
            "split_seeds": split_seeds,
            "random_seeds": random_seeds,
            "total_runs_requested": len(run_pairs),
            "epochs": epochs,
            "patience": patience,
            "lr": lr,
            "weight_decay": weight_decay,
            "save_models": save_models,
            "save_predictions": save_predictions,
            "onto_incorporation": onto_incorporation,
            "onto_name": onto_name if onto_incorporation == "align" else None,
            "lambda_align": lambda_align if onto_incorporation == "align" else None,
            "alignment_mode": alignment_mode if onto_incorporation == "align" else None,
            "temperature": temperature if onto_incorporation == "align" else None,
            "per_run": [],
            "failed_runs": [],
            "aggregated": {},
        }

        json_path: Optional[Path] = None
        output_file = None
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
                base_name = f"{graph_identity}_{embd_name}{suffix}{mode_suffix}"

            def output_file(prefix: str, suffix: str) -> Path:
                candidate = results_path / f"{prefix}{base_name}{suffix}"
                if len(str(candidate.resolve())) < 240:
                    return candidate
                return results_path / f"{prefix}run{suffix}"

            json_path = output_file("results_", ".json")
            if resume and json_path.exists():
                with json_path.open(encoding="utf-8") as handle:
                    previous = json.load(handle)
                identity = {
                    "kg_name": kg_name,
                    "graph_variant": graph_identity,
                    "options": list(options),
                    "init_embd": init_embd,
                    "random_embd_dim": random_embd_dim,
                    "split_seeds": split_seeds,
                    "random_seeds": random_seeds,
                    "total_runs_requested": len(run_pairs),
                    "epochs": epochs,
                    "patience": patience,
                    "lr": lr,
                    "weight_decay": weight_decay,
                }
                mismatched = [
                    key for key, value in identity.items()
                    if previous.get(key) != value
                ]
                if mismatched:
                    raise ValueError(
                        f"Cannot resume {json_path}: configuration differs in {mismatched}. "
                        "Use a different results_dir or set resume=False."
                    )
                all_results = previous
                all_results.setdefault("failed_runs", [])
                all_results.setdefault("aggregated", {})
                print(f"[INFO] Resuming {json_path}: {len(all_results['per_run'])} completed run(s) found.")

        def save_progress() -> None:
            if json_path is None:
                return
            temporary = json_path.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(all_results, handle, indent=2)
            temporary.replace(json_path)

        metrics_by_split = {
            split_seed: {
                "train_acc": [], "train_f1": [], "val_acc": [], "val_f1": [],
                "test_acc": [], "test_recall": [], "test_precision": [],
                "test_f1": [], "best_epoch": [],
            }
            for split_seed in split_seeds
        }

        def add_metrics(seed_result: Mapping[str, Any]) -> None:
            split_seed = int(seed_result["split_seed"])
            history = seed_result["history"]
            final_test = seed_result["final_test"]
            metrics_by_split[split_seed]["train_acc"].append(history["train_acc"][-1])
            metrics_by_split[split_seed]["train_f1"].append(history["train_f1"][-1])
            metrics_by_split[split_seed]["val_acc"].append(history["val_acc"][-1])
            metrics_by_split[split_seed]["val_f1"].append(history["val_f1"][-1])
            metrics_by_split[split_seed]["test_acc"].append(final_test["accuracy"])
            metrics_by_split[split_seed]["test_recall"].append(final_test["recall"])
            metrics_by_split[split_seed]["test_precision"].append(final_test["precision"])
            metrics_by_split[split_seed]["test_f1"].append(final_test["f1"])
            metrics_by_split[split_seed]["best_epoch"].append(seed_result["best_epoch"])

        valid_pairs = set(run_pairs)
        completed_pairs: set[tuple[int, int]] = set()
        for completed in all_results["per_run"]:
            pair = (int(completed["split_seed"]), int(completed["random_seed"]))
            if pair not in valid_pairs:
                raise ValueError(f"Stored result contains an unexpected run pair: {pair}")
            if pair in completed_pairs:
                raise ValueError(f"Stored result contains a duplicate run pair: {pair}")
            completed_pairs.add(pair)
            add_metrics(completed)

        print(f"\n{'#' * 70}")
        print(f"# Split seeds          : {split_seeds}")
        print(f"# Random seeds         : {random_seeds}")
        print(f"# Total runs           : {len(run_pairs)}")
        print(f"# Knowledge Graph      : {kg_name}")
        print(f"# Options              : {list(options)}")
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

        for i, (split_seed, random_seed) in enumerate(run_pairs, 1):
            run_pair = (split_seed, random_seed)
            if run_pair in completed_pairs:
                print(
                    f"[INFO] Skipping completed run {i}/{len(run_pairs)}: "
                    f"split_seed={split_seed}, random_seed={random_seed}"
                )
                continue
            seed_everything(random_seed, deterministic=True)
            # ``seed`` remains a local alias for legacy log messages below.
            seed = random_seed

            print(f"\n{'=' * 70}")
            print(
                f"RUN {i}/{len(run_pairs)}: "
                f"split_seed={split_seed}, random_seed={random_seed}"
            )
            print(f"{'=' * 70}")

            split_path = f"{splits_dir}/split_{split_seed}.json"
            # Result directories are already unique per graph/model/width.
            # Keep artifact paths deliberately short for Windows' traditional
            # 260-character path limit.
            seed_artifacts_dir = (
                str(Path(results_dir) / "artifacts" / f"s{split_seed}" / f"r{random_seed}")
                if save_results and (save_models or save_predictions)
                else None
            )
            seed_artifact_prefix = f"run_s{split_seed}_r{random_seed}"

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
                        seed=random_seed,
                        options=options,
                        save_variant=save_variant,
                    )
                else:
                    results = self.evaluate(
                        kg_name=kg_name,
                        model_factory=model_factory,
                        init_embd=init_embd,
                        split_path=split_path,
                        entities_embd_path=entities_embd_path,
                        edges_embd_path=edges_embd_path,
                        random_embd_dim=random_embd_dim,
                        epochs=epochs,
                        patience=patience,
                        lr=lr,
                        weight_decay=weight_decay,
                        verbose=verbose,
                        artifacts_dir=seed_artifacts_dir,
                        artifact_prefix=seed_artifact_prefix,
                        save_prediction_splits=["test"] if save_predictions else [],
                        save_models=save_models,
                        options=options,
                        save_variant=save_variant,
                    )

                seed_result = {
                    "split_seed": split_seed,
                    "random_seed": random_seed,
                    "options": list(options),
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

                all_results["per_run"].append(seed_result)
                completed_pairs.add(run_pair)
                all_results["failed_runs"] = [
                    failed for failed in all_results["failed_runs"]
                    if (int(failed["split_seed"]), int(failed["random_seed"])) != run_pair
                ]
                add_metrics(seed_result)
                save_progress()

                train_acc = results["history"]["train_acc"][-1]
                train_f1 = results["history"]["train_f1"][-1]
                val_acc = results["history"]["val_acc"][-1]
                val_f1 = results["history"]["val_f1"][-1]
                test_acc = results["final_test"]["accuracy"]
                test_f1 = results["final_test"]["f1"]
                best_epoch = results["best_val"]["epoch"]

                print(f"\n✓ Seed {seed} completed:")
                print(f"  Train: Acc={train_acc:.4f}, F1={train_f1:.4f}")
                print(f"  Val:   Acc={val_acc:.4f}, F1={val_f1:.4f}")
                print(f"  Test:  Acc={test_acc:.4f}, F1={test_f1:.4f}")
                print(f"  Best epoch: {best_epoch}")

            except Exception as e:
                all_results["failed_runs"] = [
                    failed for failed in all_results["failed_runs"]
                    if (int(failed["split_seed"]), int(failed["random_seed"])) != run_pair
                ]
                all_results["failed_runs"].append(
                    {
                        "split_seed": split_seed,
                        "random_seed": random_seed,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
                save_progress()
                print(
                    f"\nError on split_seed={split_seed}, "
                    f"random_seed={random_seed}: {e}"
                )
                import traceback
                traceback.print_exc()
                continue

        print(f"\n{'#' * 70}")
        print("# AGGREGATED RESULTS")
        print(f"{'#' * 70}\n")

        all_results["aggregated"] = {}
        all_results["runs_completed"] = len(all_results["per_run"])
        all_results["runs_failed"] = len(all_results["failed_runs"])
        metric_names = next(iter(metrics_by_split.values())).keys()
        for metric_name in metric_names:
            scores_by_split = {
                split_seed: split_metrics[metric_name]
                for split_seed, split_metrics in metrics_by_split.items()
                if split_metrics[metric_name]
            }
            if scores_by_split:
                stats = aggregate_split_and_randomness(scores_by_split)
                mean = stats["mean"]
                std = stats["split_std"]

                all_results["aggregated"][metric_name] = stats
                print(f"  between-split SD={stats['split_std']:.4f} | mean within-split random SD={stats['randomness_std']:.4f}")

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
                graph_identity = graph_variant_name(kg_name, options)
                base_name = f"{graph_identity}_{embd_name}{suffix}{mode_suffix}"

            def output_file(prefix: str, suffix: str) -> Path:
                """Keep benchmark output names below the Windows path limit.

                ``evaluate_benchmark`` already encodes graph/model/width in
                its directory hierarchy. Repeating that identity in the file
                name can exceed MAX_PATH for long persisted graph names.
                """
                candidate = results_path / f"{prefix}{base_name}{suffix}"
                # ``candidate`` is normally relative to the repository, but
                # Windows applies MAX_PATH to its absolute path.
                if len(str(candidate.resolve())) < 240:
                    return candidate
                return results_path / f"{prefix}run{suffix}"

            json_path = output_file("results_", ".json")

            with open(json_path, "w") as f:
                json.dump(all_results, f, indent=2)

            print(f"\n✓ Detailed results saved to: {json_path}")

            summary_data = {
                "metric": [],
                "mean": [],
                "split_std": [],
                "randomness_std": [],
                "overall_std": [],
                "n_scores": [],
            }

            for metric_name, stats in all_results["aggregated"].items():
                summary_data["metric"].append(metric_name)
                summary_data["mean"].append(stats["mean"])
                summary_data["split_std"].append(stats["split_std"])
                summary_data["randomness_std"].append(stats["randomness_std"])
                summary_data["overall_std"].append(stats["overall_std"])
                summary_data["n_scores"].append(stats["n_scores"])

            df_summary = pd.DataFrame(summary_data)
            csv_path = output_file("summary_", ".csv")
            df_summary.to_csv(csv_path, index=False)

            print(f"✓ Summary saved to: {csv_path}")

            per_run_data = []

            for seed_result in all_results["per_run"]:
                per_run_data.append(
                    {
                        "split_seed": seed_result["split_seed"],
                        "random_seed": seed_result["random_seed"],
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

            df_per_run = pd.DataFrame(per_run_data)
            per_run_csv = output_file("per_run_", ".csv")
            df_per_run.to_csv(per_run_csv, index=False)
            # Alias retained only for the legacy log message immediately below.
            per_seed_csv = per_run_csv

            print(f"✓ Per-seed results saved to: {per_seed_csv}")

        return all_results

    # ------------------------------------------------------------------
    # Multi-model evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_hidden_channels(hidden_channels: int | Iterable[int]) -> List[int]:
        """Validate a hidden-dimension grid while accepting one legacy int."""
        if isinstance(hidden_channels, int):
            dimensions = [hidden_channels]
        else:
            dimensions = list(hidden_channels)
        if not dimensions:
            raise ValueError("hidden_channels must contain at least one dimension")
        if any(not isinstance(dimension, int) or dimension < 1 for dimension in dimensions):
            raise ValueError("Each hidden_channels value must be an integer >= 1")
        if len(set(dimensions)) != len(dimensions):
            raise ValueError("hidden_channels must not contain duplicates")
        return dimensions

    def _resolve_benchmark_split_seeds(
        self,
        split_seeds: Optional[Iterable[int]],
        splits_dir: str,
        common_nodes_path: Optional[str] = None,
    ) -> List[int]:
        """Use the ten reference splits by default and create only missing ones.

        Extra or arbitrary seed values are legitimate reproducible splits. They
        are generated from the configured gold-standard file on first use.
        """
        requested = (
            list(PREGENERATED_SPLIT_SEEDS)
            if split_seeds is None
            else list(split_seeds)
        )
        if not requested:
            raise ValueError("split_seeds must contain at least one split seed")
        if any(not isinstance(seed, int) for seed in requested):
            raise ValueError("Each split_seeds value must be an integer")
        if len(set(requested)) != len(requested):
            raise ValueError("split_seeds must not contain duplicates")

        split_directory = Path(splits_dir)
        missing_seeds = [
            seed for seed in requested
            if not (split_directory / f"split_{seed}.json").exists()
        ]
        requests_more_than_default = len(requested) > len(PREGENERATED_SPLIT_SEEDS)

        if requests_more_than_default or missing_seeds:
            if missing_seeds:
                warnings.warn(
                    "Requested benchmark splits are not available yet. "
                    "New reproducible splits will be generated before training: "
                    f"{missing_seeds}. The standard benchmark provides "
                    f"{len(PREGENERATED_SPLIT_SEEDS)} pre-generated splits.",
                    UserWarning,
                    stacklevel=2,
                )
                generate_and_save_splits(
                    xlsx_path=common_nodes_path or self.config["common_nodes_path"],
                    save_dir=str(split_directory),
                    seeds=missing_seeds,
                    train_ratio=self.config.get("train_ratio", 0.10),
                    val_ratio=self.config.get("val_ratio", 0.10),
                    test_ratio=self.config.get("test_ratio", 0.80),
                    stratify=self.config.get("stratify", True),
                )
            else:
                warnings.warn(
                    f"{len(requested)} splits were requested, while the standard "
                    f"benchmark provides {len(PREGENERATED_SPLIT_SEEDS)} pre-generated "
                    "splits. All requested split files already exist and will be reused.",
                    UserWarning,
                    stacklevel=2,
                )

        return requested

    def evaluate_no_graph_baseline(
        self,
        kg_name: str,
        init_embds: Iterable[str] = ("sentence-transformers/all-MiniLM-L6-v2",),
        split_seeds: Optional[Iterable[int]] = None,
        random_seeds: Optional[Iterable[int]] = None,
        *,
        hidden_channels: int | Iterable[int] = (64,),
        random_embedding_dim: int = 384,
        dropout: float = 0.0,
        epochs: int = 100,
        patience: int = 100,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        verbose: bool = True,
        save_results: bool = True,
        save_models: bool = True,
        save_predictions: bool = True,
        resume: bool = True,
        results_dir: str = "results/no_graph",
    ) -> Dict[str, Any]:
        """Evaluate text-only and random-feature MLP baselines for one protocol.

        ``kg_name`` resolves the benchmark's gold-standard terms and split
        directory, but no graph JSON, edge list, predicate or graph embedding
        cache is read.  Each item in ``init_embds`` is independently evaluated
        over the full Cartesian product of split and training-randomness seeds.

        Use ``random`` for Gaussian random features of
        ``random_embedding_dim`` dimensions, or ``random_<dimension>`` to set
        the dimension directly, for example ``random_384``.  Random features
        are fixed for all splits sharing the same ``random_seed``.
        """
        protocol = resolve_benchmark_protocol(kg_name)
        feature_sources = tuple(init_embds)
        if not feature_sources or any(not isinstance(source, str) or not source.strip() for source in feature_sources):
            raise ValueError("init_embds must contain at least one non-empty embedding source")
        if len(set(feature_sources)) != len(feature_sources):
            raise ValueError("init_embds must not contain duplicates")
        hidden_dimensions = self._normalise_hidden_channels(hidden_channels)
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if random_embedding_dim < 1:
            raise ValueError("random_embedding_dim must be >= 1")
        if epochs < 1 or patience < 1:
            raise ValueError("epochs and patience must both be >= 1")

        resolved_split_seeds = self._resolve_benchmark_split_seeds(
            split_seeds=split_seeds,
            splits_dir=str(protocol.splits_dir),
            common_nodes_path=str(protocol.common_nodes_path),
        )
        resolved_random_seeds = [1, 2, 3, 4, 5] if random_seeds is None else list(random_seeds)
        if not resolved_random_seeds or any(not isinstance(seed, int) for seed in resolved_random_seeds):
            raise ValueError("random_seeds must contain at least one integer")
        if len(set(resolved_random_seeds)) != len(resolved_random_seeds):
            raise ValueError("random_seeds must not contain duplicates")

        dataset = load_no_graph_dataset(protocol.common_nodes_path)
        splits = {
            seed: load_split_indices(protocol.splits_dir / f"split_{seed}.json", len(dataset.terms))
            for seed in resolved_split_seeds
        }
        root = Path(results_dir)
        suite: Dict[str, Any] = {
            "kg_name": kg_name,
            # Keep the graph name so the existing paired-test loader can align
            # this baseline with GNN runs over the same labelled-node protocol.
            "graph_variant": kg_name,
            "benchmark_protocol": protocol.name,
            "common_nodes_path": str(protocol.common_nodes_path),
            "splits_dir": str(protocol.splits_dir),
            "baseline_type": "no_graph_mlp",
            "init_embds": list(feature_sources),
            "split_seeds": resolved_split_seeds,
            "random_seeds": resolved_random_seeds,
            "hidden_channels": hidden_dimensions,
            "random_embedding_dim": random_embedding_dim,
            "dropout": dropout,
            "save_models": save_models,
            "save_predictions": save_predictions,
            "models": {},
        }
        summary_rows: list[dict[str, Any]] = []

        for init_embd in feature_sources:
            source_id = "".join(
                character if character.isalnum() else "_" for character in init_embd.rsplit("/", 1)[-1]
            ).strip("_") or "embedding"
            model_name = f"NoGraphMLP_{source_id}"
            source_result: Dict[str, Any] = {"init_embd": init_embd, "runs": {}}
            suite["models"][model_name] = source_result

            for hidden_dimension in hidden_dimensions:
                run_key = f"hidden_{hidden_dimension}_out_{dataset.num_classes}"
                run_results: Dict[str, Any] = {
                    "kg_name": kg_name,
                    "graph_variant": kg_name,
                    "benchmark_protocol": protocol.name,
                    "baseline_type": "no_graph_mlp",
                    "model_name": model_name,
                    "init_embd": init_embd,
                    "split_seeds": resolved_split_seeds,
                    "random_seeds": resolved_random_seeds,
                    "total_runs_requested": len(resolved_split_seeds) * len(resolved_random_seeds),
                    "epochs": epochs,
                    "patience": patience,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "save_models": save_models,
                    "save_predictions": save_predictions,
                    "per_run": [],
                    "failed_runs": [],
                    "aggregated": {},
                }
                result_dir = root / "no_graph" / source_id / run_key
                result_path = result_dir / (
                    f"results_{kg_name}__{model_name}__h{hidden_dimension}__out{dataset.num_classes}.json"
                )
                if save_results and resume and result_path.exists():
                    with result_path.open(encoding="utf-8") as handle:
                        previous = json.load(handle)
                    identity = {
                        "kg_name": kg_name,
                        "graph_variant": kg_name,
                        "benchmark_protocol": protocol.name,
                        "baseline_type": "no_graph_mlp",
                        "model_name": model_name,
                        "init_embd": init_embd,
                        "split_seeds": resolved_split_seeds,
                        "random_seeds": resolved_random_seeds,
                        "total_runs_requested": len(resolved_split_seeds) * len(resolved_random_seeds),
                        "epochs": epochs,
                        "patience": patience,
                        "lr": lr,
                        "weight_decay": weight_decay,
                    }
                    mismatched = [key for key, value in identity.items() if previous.get(key) != value]
                    if mismatched:
                        raise ValueError(
                            f"Cannot resume {result_path}: configuration differs in {mismatched}. "
                            "Use a different results_dir or set resume=False."
                        )
                    run_results = previous
                    run_results.setdefault("failed_runs", [])
                    run_results.setdefault("aggregated", {})
                    print(f"[INFO] Resuming {result_path}: {len(run_results['per_run'])} completed run(s) found.")

                def save_run_progress() -> None:
                    if not save_results:
                        return
                    result_dir.mkdir(parents=True, exist_ok=True)
                    temporary = result_path.with_suffix(".tmp")
                    with temporary.open("w", encoding="utf-8") as handle:
                        json.dump(run_results, handle, indent=2)
                    temporary.replace(result_path)

                metrics_by_split = {
                    seed: {
                        "train_acc": [], "train_f1": [], "val_acc": [], "val_f1": [],
                        "test_acc": [], "test_recall": [], "test_precision": [],
                        "test_f1": [], "best_epoch": [],
                    }
                    for seed in resolved_split_seeds
                }

                def add_no_graph_metrics(per_run: Mapping[str, Any]) -> None:
                    split_seed = int(per_run["split_seed"])
                    history = per_run["history"]
                    final = per_run["final_test"]
                    metrics_by_split[split_seed]["train_acc"].append(history["train_acc"][-1])
                    metrics_by_split[split_seed]["train_f1"].append(history["train_f1"][-1])
                    metrics_by_split[split_seed]["val_acc"].append(history["val_acc"][-1])
                    metrics_by_split[split_seed]["val_f1"].append(history["val_f1"][-1])
                    metrics_by_split[split_seed]["test_acc"].append(final["accuracy"])
                    metrics_by_split[split_seed]["test_recall"].append(final["recall"])
                    metrics_by_split[split_seed]["test_precision"].append(final["precision"])
                    metrics_by_split[split_seed]["test_f1"].append(final["f1"])
                    metrics_by_split[split_seed]["best_epoch"].append(per_run["best_epoch"])

                valid_pairs = {
                    (split_seed, random_seed)
                    for split_seed in resolved_split_seeds
                    for random_seed in resolved_random_seeds
                }
                completed_pairs: set[tuple[int, int]] = set()
                for completed in run_results["per_run"]:
                    pair = (int(completed["split_seed"]), int(completed["random_seed"]))
                    if pair not in valid_pairs or pair in completed_pairs:
                        raise ValueError(f"Invalid stored no-graph run pair: {pair}")
                    completed_pairs.add(pair)
                    add_no_graph_metrics(completed)
                print(
                    f"\n{'#' * 70}\nNo-graph baseline: {model_name} "
                    f"(hidden={hidden_dimension}, input={init_embd})\n{'#' * 70}"
                )

                for split_seed in resolved_split_seeds:
                    for random_seed in resolved_random_seeds:
                        run_pair = (split_seed, random_seed)
                        if run_pair in completed_pairs:
                            print(
                                f"[INFO] Skipping completed no-graph run: "
                                f"split_seed={split_seed}, random_seed={random_seed}"
                            )
                            continue
                        try:
                            seed_everything(random_seed, deterministic=True)
                            features = build_no_graph_features(
                                dataset.terms,
                                init_embd=init_embd,
                                random_seed=random_seed,
                                random_embedding_dim=random_embedding_dim,
                            )
                            model = NoGraphMLP(
                                in_channels=int(features.shape[1]),
                                hidden_channels=hidden_dimension,
                                num_classes=dataset.num_classes,
                                dropout=dropout,
                            )
                            artifacts_dir = (
                                result_dir / "artifacts" / f"s{split_seed}" / f"r{random_seed}"
                                if save_results and (save_models or save_predictions)
                                else None
                            )
                            trainer = NoGraphTrainer(
                                model=model,
                                device=self.device,
                                lr=lr,
                                weight_decay=weight_decay,
                            )
                            result = trainer.train(
                                features=features,
                                labels=dataset.labels,
                                split_indices=splits[split_seed],
                                terms=dataset.terms,
                                label_encoder=dataset.label_encoder,
                                epochs=epochs,
                                patience=patience,
                                verbose=verbose,
                                artifacts_dir=str(artifacts_dir) if artifacts_dir is not None else None,
                                artifact_prefix=f"run_s{split_seed}_r{random_seed}",
                                save_model_checkpoint=save_models,
                                save_prediction_splits=["test"] if save_predictions else [],
                            )
                            per_run = {
                                "split_seed": split_seed,
                                "random_seed": random_seed,
                                "best_val_f1": result["best_val"]["f1"],
                                "best_epoch": result["best_val"]["epoch"],
                                "final_test": result["final_test"],
                                "history": result["history"],
                                "artifacts": result["artifacts"],
                            }
                            run_results["per_run"].append(per_run)
                            completed_pairs.add(run_pair)
                            run_results["failed_runs"] = [
                                failed for failed in run_results["failed_runs"]
                                if (int(failed["split_seed"]), int(failed["random_seed"])) != run_pair
                            ]
                            add_no_graph_metrics(per_run)
                            save_run_progress()
                        except Exception as error:
                            run_results["failed_runs"] = [
                                failed for failed in run_results["failed_runs"]
                                if (int(failed["split_seed"]), int(failed["random_seed"])) != run_pair
                            ]
                            run_results["failed_runs"].append(
                                {
                                    "split_seed": split_seed,
                                    "random_seed": random_seed,
                                    "error": f"{type(error).__name__}: {error}",
                                }
                            )
                            save_run_progress()
                            print(
                                f"Error on no-graph split_seed={split_seed}, "
                                f"random_seed={random_seed}: {error}"
                            )
                            import traceback
                            traceback.print_exc()
                            continue

                run_results["aggregated"] = {}
                run_results["runs_completed"] = len(run_results["per_run"])
                run_results["runs_failed"] = len(run_results["failed_runs"])
                metric_names = next(iter(metrics_by_split.values())).keys()
                for metric_name in metric_names:
                    non_empty = {
                        seed: split_metrics[metric_name]
                        for seed, split_metrics in metrics_by_split.items()
                        if split_metrics[metric_name]
                    }
                    if non_empty:
                        run_results["aggregated"][metric_name] = aggregate_split_and_randomness(non_empty)

                source_result["runs"][run_key] = run_results
                final_stats = run_results["aggregated"].get("test_f1", {})
                summary_rows.append(
                    {
                        "graph_name": kg_name,
                        "benchmark_protocol": protocol.name,
                        "baseline_type": "no_graph_mlp",
                        "model_name": model_name,
                        "init_embd": init_embd,
                        "configuration": run_key,
                        "hidden_channels": hidden_dimension,
                        "out_channels": dataset.num_classes,
                        "runs_requested": run_results["total_runs_requested"],
                        "runs_completed": len(run_results["per_run"]),
                        "test_macro_f1_mean": final_stats.get("mean"),
                        "test_macro_f1_split_std": final_stats.get("split_std"),
                        "test_macro_f1_randomness_std": final_stats.get("randomness_std"),
                    }
                )

                if save_results:
                    result_dir = root / "no_graph" / source_id / run_key
                    result_dir.mkdir(parents=True, exist_ok=True)
                    result_path = result_dir / (
                        f"results_{kg_name}__{model_name}__h{hidden_dimension}__out{dataset.num_classes}.json"
                    )
                    with result_path.open("w", encoding="utf-8") as handle:
                        json.dump(run_results, handle, indent=2)
                    pd.DataFrame(summary_rows[-1:]).to_csv(
                        result_dir / f"summary_{kg_name}__{model_name}__h{hidden_dimension}__out{dataset.num_classes}.csv",
                        index=False,
                    )
                    print(f"No-graph detailed results saved to: {result_path}")

        if save_results:
            root.mkdir(parents=True, exist_ok=True)
            suite["output_files"] = {
                "json": str(root / "no_graph_baseline_results.json"),
                "summary_csv": str(root / "no_graph_baseline_summary.csv"),
            }
            pd.DataFrame(summary_rows).to_csv(suite["output_files"]["summary_csv"], index=False)
            with Path(suite["output_files"]["json"]).open("w", encoding="utf-8") as handle:
                json.dump(suite, handle, indent=2)
            print(f"No-graph baseline summary saved to: {suite['output_files']['summary_csv']}")
        return suite

    def evaluate_models(
        self,
        kg_name: str,
        model_names: Iterable[str],
        init_embd: str,
        split_seeds: List[int],
        random_seeds: List[int],
        *,
        splits_dir: str = "datasets/split",
        hidden_channels: int | Iterable[int] = (64,),
        out_channels: Optional[int] = None,
        model_kwargs_by_name: Optional[Mapping[str, Mapping[str, Any]]] = None,
        entities_embd_path: Optional[str] = None,
        edges_embd_path: Optional[str] = None,
        random_embd_dim: int = 256,
        epochs: int = 100,
        patience: int = 100,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        verbose: bool = True,
        save_results: bool = True,
        save_models: bool = True,
        save_predictions: bool = True,
        resume: bool = True,
        results_dir: str = "results",
        options: Iterable[str] | None = None,
        save_variant: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate several registered encoders on one stored graph.

        Every model and every hidden dimension receives the same split and
        random seeds. Each model is freshly built for every run by
        :meth:`evaluate_all`. If ``out_channels`` is ``None``, it is set to the
        current hidden dimension. The graph can optionally be an in-memory
        variant for this lower-level, single-graph API.
        Model-specific keyword arguments may override the defaults through
        ``model_kwargs_by_name``, for example ``{"RGCN": {"num_bases": 25}}``.
        """
        selected_models = tuple(model_names)
        if not selected_models:
            raise ValueError("model_names must contain at least one model name")
        if len(set(selected_models)) != len(selected_models):
            raise ValueError("model_names must not contain duplicates")
        unknown_models = sorted(set(selected_models) - set(available_model_names()))
        if unknown_models:
            raise ValueError(
                f"Unknown model name(s): {unknown_models}. "
                f"Available models: {list(available_model_names())}"
            )
        if not split_seeds or not random_seeds:
            raise ValueError("split_seeds and random_seeds must both be non-empty")
        hidden_dimensions = self._normalise_hidden_channels(hidden_channels)
        if out_channels is not None and out_channels < 1:
            raise ValueError("out_channels must be >= 1")

        options = tuple(options or ())
        graph_identity = graph_variant_name(kg_name, options)
        first_split = Path(splits_dir) / f"split_{split_seeds[0]}.json"
        if not first_split.exists():
            raise FileNotFoundError(f"First requested split does not exist: {first_split}")

        data, _, _, _, graph_preparation = self.get_data(
            kg_name=kg_name,
            init_embd=init_embd,
            split_path=str(first_split),
            entities_embd_path=entities_embd_path,
            edges_embd_path=edges_embd_path,
            random_embd_dim=random_embd_dim,
            options=options,
            save_variant=save_variant,
        )
        in_channels = int(data.x.shape[1])
        num_relations = len(graph_preparation.predicate_to_id)
        overrides = model_kwargs_by_name or {}
        unused_overrides = sorted(set(overrides) - set(selected_models))
        if unused_overrides:
            raise ValueError(
                f"model_kwargs_by_name contains model(s) not selected: {unused_overrides}"
            )

        suite_results: Dict[str, Any] = {
            "kg_name": kg_name,
            "graph_variant": graph_identity,
            "options": list(options),
            "model_names": list(selected_models),
            "split_seeds": list(split_seeds),
            "random_seeds": list(random_seeds),
            "in_channels": in_channels,
            "num_relations": num_relations,
            "hidden_channels": hidden_dimensions,
            "out_channels": out_channels,
            "random_embd_dim": random_embd_dim,
            "save_models": save_models,
            "save_predictions": save_predictions,
            "models": {},
        }

        for model_name in selected_models:
            model_kwargs = {
                **get_default_model_kwargs(model_name),
                **dict(overrides.get(model_name, {})),
            }

            model_suite: Dict[str, Any] = {"runs": {}}
            suite_results["models"][model_name] = model_suite
            for hidden_dimension in hidden_dimensions:
                output_channels = (
                    out_channels if out_channels is not None else hidden_dimension
                )

                def model_factory(
                    _model_name=model_name,
                    _kwargs=model_kwargs,
                    _hidden_dimension=hidden_dimension,
                    _output_channels=output_channels,
                ):
                    return build_encoder(
                        model_name=_model_name,
                        in_channels=in_channels,
                        hidden_channels=_hidden_dimension,
                        out_channels=_output_channels,
                        num_relations=num_relations,
                        **_kwargs,
                    )

                run_key = f"hidden_{hidden_dimension}_out_{output_channels}"
                print(
                    f"\n{'=' * 70}\nEvaluating model: {model_name} "
                    f"(hidden={hidden_dimension}, out={output_channels})\n{'=' * 70}"
                )
                model_results_dir = str(
                    Path(results_dir) / graph_identity / model_name / run_key
                )
                model_suite["runs"][run_key] = self.evaluate_all(
                    kg_name=kg_name,
                    model_factory=model_factory,
                    init_embd=init_embd,
                    split_seeds=list(split_seeds),
                    random_seeds=list(random_seeds),
                    splits_dir=splits_dir,
                    entities_embd_path=entities_embd_path,
                    edges_embd_path=edges_embd_path,
                    random_embd_dim=random_embd_dim,
                    epochs=epochs,
                    patience=patience,
                    lr=lr,
                    weight_decay=weight_decay,
                    verbose=verbose,
                    save_results=save_results,
                    save_models=save_models,
                    save_predictions=save_predictions,
                    resume=resume,
                    results_dir=model_results_dir,
                    run_id=(
                        f"{graph_identity}__{model_name}__h{hidden_dimension}"
                        f"__out{output_channels}"
                    ),
                    options=options,
                    save_variant=save_variant,
                )

        return suite_results

    # ------------------------------------------------------------------
    # Benchmark: several persisted graphs x several registered models
    # ------------------------------------------------------------------

    @staticmethod
    def _benchmark_summary_rows(benchmark_results: Mapping[str, Any]) -> List[Dict[str, Any]]:
        """Flatten completed graph/model/configuration results into CSV rows."""
        metric_names = {
            "test_accuracy": "test_acc",
            "test_recall": "test_recall",
            "test_precision": "test_precision",
            "test_macro_f1": "test_f1",
        }
        rows: List[Dict[str, Any]] = []
        for graph_name, graph_results in benchmark_results["graphs"].items():
            for model_name, model_results in graph_results["models"].items():
                for run_key, evaluation in model_results["runs"].items():
                    try:
                        _, hidden_channels, _, out_channels = run_key.split("_")
                    except ValueError as exc:
                        raise ValueError(
                            f"Unexpected benchmark configuration key: {run_key!r}"
                        ) from exc

                    completed_runs = evaluation["per_run"]
                    validation_scores = [
                        float(run["best_val_f1"]) for run in completed_runs
                    ]
                    row: Dict[str, Any] = {
                        "graph_name": graph_name,
                        "graph_variant": evaluation.get("graph_variant", graph_name),
                        "model_name": model_name,
                        "configuration": run_key,
                        "hidden_channels": int(hidden_channels),
                        "out_channels": int(out_channels),
                        "init_embd": benchmark_results["init_embd"],
                        "split_seeds": ",".join(
                            map(str, benchmark_results["split_seeds"])
                        ),
                        "random_seeds": ",".join(
                            map(str, benchmark_results["random_seeds"])
                        ),
                        "runs_requested": evaluation["total_runs_requested"],
                        "runs_completed": len(completed_runs),
                        "mean_best_validation_f1": (
                            float(np.mean(validation_scores))
                            if validation_scores
                            else float("nan")
                        ),
                    }
                    for column_prefix, aggregated_name in metric_names.items():
                        statistics = evaluation["aggregated"].get(aggregated_name, {})
                        row[f"{column_prefix}_mean"] = statistics.get("mean")
                        row[f"{column_prefix}_split_std"] = statistics.get("split_std")
                        row[f"{column_prefix}_randomness_std"] = statistics.get(
                            "randomness_std"
                        )
                        row[f"{column_prefix}_overall_std"] = statistics.get(
                            "overall_std"
                        )
                    rows.append(row)
        return rows

    def evaluate_benchmark(
        self,
        graph_names: Iterable[str],
        model_names: Iterable[str],
        init_embd: str,
        split_seeds: Optional[Iterable[int]] = None,
        random_seeds: Optional[Iterable[int]] = None,
        *,
        splits_dir: str = "datasets/split",
        hidden_channels: int | Iterable[int] = (64,),
        out_channels: Optional[int] = None,
        model_kwargs_by_name: Optional[Mapping[str, Mapping[str, Any]]] = None,
        entities_embd_path: Optional[str] = None,
        edges_embd_path: Optional[str] = None,
        random_embd_dim: int = 256,
        epochs: int = 100,
        patience: int = 100,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        verbose: bool = True,
        save_results: bool = True,
        save_models: bool = True,
        save_predictions: bool = True,
        resume: bool = True,
        results_dir: str = "results",
    ) -> Dict[str, Any]:
        """Evaluate a Cartesian product of persisted graphs, models and widths.

        ``graph_names`` must be names of JSON files already present in
        ``datasets``. In particular, this high-level benchmark deliberately
        has no graph-variant ``options`` parameter: create and save a variant
        first, then pass its stored graph name here. This gives every run a
        stable, explicit graph artifact.

        The returned hierarchy is ``graphs[graph_name]["models"][model_name]``
        and then ``runs["hidden_<h>_out_<o>"]``. By default, it uses the ten
        standard split seeds and training random seeds ``[1, 2, 3, 4, 5]``.
        Missing requested split files are generated reproducibly with a clear
        warning. For each leaf, the result is
        exactly :meth:`evaluate_all`'s result, including the separate split and
        randomness standard deviations. When ``save_results=True``, the method
        additionally writes ``benchmark_results.json`` and
        ``benchmark_summary.csv`` at the root of ``results_dir``, plus
        ``best_by_graph.csv``. The summary has one row per graph, architecture
        and hidden/output configuration; the latter selects one configuration
        per graph only from mean validation Macro-F1.
        """
        selected_graphs = tuple(graph_names)
        selected_models = tuple(model_names)
        if not selected_graphs:
            raise ValueError("graph_names must contain at least one graph name")
        if not selected_models:
            raise ValueError("model_names must contain at least one model name")
        if len(set(selected_graphs)) != len(selected_graphs):
            raise ValueError("graph_names must not contain duplicates")
        if len(set(selected_models)) != len(selected_models):
            raise ValueError("model_names must not contain duplicates")

        # Validate the whole requested benchmark before training the first run.
        selected_graphs = tuple(self._validate_graph_name(name) for name in selected_graphs)
        unknown_models = sorted(set(selected_models) - set(available_model_names()))
        if unknown_models:
            raise ValueError(
                f"Unknown model name(s): {unknown_models}. "
                f"Available models: {list(available_model_names())}"
            )
        hidden_dimensions = self._normalise_hidden_channels(hidden_channels)
        if out_channels is not None and out_channels < 1:
            raise ValueError("out_channels must be >= 1")
        resolved_split_seeds = self._resolve_benchmark_split_seeds(
            split_seeds=split_seeds,
            splits_dir=splits_dir,
        )
        resolved_random_seeds = (
            [1, 2, 3, 4, 5] if random_seeds is None else list(random_seeds)
        )
        if not resolved_random_seeds:
            raise ValueError("random_seeds must contain at least one random seed")
        if any(not isinstance(seed, int) for seed in resolved_random_seeds):
            raise ValueError("Each random_seeds value must be an integer")
        if len(set(resolved_random_seeds)) != len(resolved_random_seeds):
            raise ValueError("random_seeds must not contain duplicates")
        if random_embd_dim < 1:
            raise ValueError("random_embd_dim must be >= 1")

        benchmark_results: Dict[str, Any] = {
            "graph_names": list(selected_graphs),
            "model_names": list(selected_models),
            "init_embd": init_embd,
            "split_seeds": resolved_split_seeds,
            "random_seeds": resolved_random_seeds,
            "hidden_channels": hidden_dimensions,
            "out_channels": out_channels,
            "random_embd_dim": random_embd_dim,
            "save_models": save_models,
            "save_predictions": save_predictions,
            "graphs": {},
        }

        for graph_name in selected_graphs:
            print(f"\n{'#' * 70}\nBenchmark graph: {graph_name}\n{'#' * 70}")
            benchmark_results["graphs"][graph_name] = self.evaluate_models(
                kg_name=graph_name,
                model_names=selected_models,
                init_embd=init_embd,
                split_seeds=resolved_split_seeds,
                random_seeds=resolved_random_seeds,
                splits_dir=splits_dir,
                hidden_channels=hidden_dimensions,
                out_channels=out_channels,
                model_kwargs_by_name=model_kwargs_by_name,
                entities_embd_path=entities_embd_path,
                edges_embd_path=edges_embd_path,
                random_embd_dim=random_embd_dim,
                epochs=epochs,
                patience=patience,
                lr=lr,
                weight_decay=weight_decay,
                verbose=verbose,
                save_results=save_results,
                save_models=save_models,
                save_predictions=save_predictions,
                resume=resume,
                results_dir=results_dir,
            )

        if save_results:
            benchmark_directory = Path(results_dir)
            benchmark_directory.mkdir(parents=True, exist_ok=True)
            summary_path = benchmark_directory / "benchmark_summary.csv"
            summary_rows = self._benchmark_summary_rows(benchmark_results)
            summary_table = pd.DataFrame(summary_rows)
            summary_table.to_csv(
                summary_path,
                index=False,
            )

            complete_rows = [
                row for row in summary_rows
                if row["runs_completed"] == row["runs_requested"]
            ]
            best_rows: List[Dict[str, Any]] = []
            for graph_name in selected_graphs:
                candidates = [
                    row for row in complete_rows if row["graph_name"] == graph_name
                ]
                if not candidates:
                    warnings.warn(
                        f"No complete model configuration is available for {graph_name!r}; "
                        "it is omitted from best_by_graph.csv.",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
                best_row = min(
                    candidates,
                    key=lambda row: (
                        -row["mean_best_validation_f1"],
                        row["hidden_channels"],
                        row["out_channels"],
                        row["model_name"],
                    ),
                )
                best_rows.append(
                    {
                        **best_row,
                        "selection_metric": "mean_best_validation_f1",
                    }
                )

            best_by_graph_path = benchmark_directory / "best_by_graph.csv"
            pd.DataFrame(best_rows).to_csv(best_by_graph_path, index=False)
            json_path = benchmark_directory / "benchmark_results.json"
            benchmark_results["output_files"] = {
                "json": str(json_path),
                "summary_csv": str(summary_path),
                "best_by_graph_csv": str(best_by_graph_path),
            }
            benchmark_results["best_by_graph"] = best_rows

            with json_path.open("w", encoding="utf-8") as handle:
                json.dump(benchmark_results, handle, indent=2)
            print(f"\nBenchmark JSON saved to: {json_path}")
            print(f"Benchmark summary saved to: {summary_path}")
            print(f"Best configuration by graph saved to: {best_by_graph_path}")

        return benchmark_results
