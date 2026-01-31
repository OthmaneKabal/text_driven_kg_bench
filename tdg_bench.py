from typing import Optional

import pandas as pd
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Callable
import json
from data_preprocessing.data_manager import get_data_and_loaders
from data_preprocessing.splits import generate_and_save_splits
from models.StandardClassifier import StandardClassifier
from train.Trainer import Trainer
from utilities.utilities import load_config, seed_everything


class TDGBench:
    def __init__(self, use_classifier = True, config_path="config.yml"):
        self.config = load_config(config_path)
        self.use_classifier = use_classifier
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_data(self, kg_name = "GT2KG_kg", init_embd= "sentence-transformers/all-MiniLM-L6-v2", split_path = "datasets/split/split_42.json", entities_embd_path = None, edges_embd_path = None, random_embd_dim = 256, use_cache = False ):
        return get_data_and_loaders(
        kg_name = kg_name,
        model_name_init = init_embd,
        common_nodes_path = self.config["common_nodes_path"],
        entities_embd_path = entities_embd_path,
        edges_embd_path = edges_embd_path,
        split_file = split_path,
        random_embd_dim = random_embd_dim,
        use_cache = use_cache)



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


    def prepare_model(self, model_or_encoder):
        """
        Prepare the model for evaluation.

        Parameters:
        - model_or_encoder:
            - If use_builtin_classifier=True: expects an encoder (GNN backbone)
            - If use_builtin_classifier=False: expects a complete model

        Returns:
        - model: Ready-to-use model
        """

        if self.use_classifier:
            model = StandardClassifier(
                encoder=model_or_encoder,
                num_classes=self.config["num_classes"],
                dropout=self.config["classifier_dropout"]
            )

        else:
            # L'utilisateur fournit le modèle complet
            model = model_or_encoder
        return model.to(self.device)

    def evaluate(
            self,
            kg_name: str,
            model_factory: Callable,
            init_embd: str,
            split_path: str,
            entities_embd_path: Optional[str] = None,
            edges_embd_path: Optional[str] = None,
            epochs: int = 100,
            patience: int = 20,
            lr: float = 0.01,
            weight_decay: float = 5e-4,
            verbose: bool = True
    ) -> Dict:
        """
        Evaluate a model on a single split.

        Parameters:
        - kg_name: Knowledge graph name
        - model_factory: Function that returns a fresh model instance
        - init_embd: Embedding initialization method
        - split_path: Path to split file
        - entities_embd_path: Optional path to entity embeddings
        - edges_embd_path: Optional path to edge embeddings
        - epochs: Number of training epochs
        - patience: Early stopping patience
        - lr: Learning rate
        - weight_decay: Weight decay
        - verbose: Print training progress

        Returns:
        - Dictionary with training results
        """
        # Load data with specific split
        annotated_graph, train_loader, val_loader, test_loader, gdp = self.get_data(
            kg_name=kg_name,
            init_embd=init_embd,
            split_path=split_path,
            entities_embd_path=entities_embd_path,
            edges_embd_path=edges_embd_path
        )


        if verbose:
            print(f"\n{'=' * 70}")
            print(f"Evaluating on split: {split_path}")
            print(f"Graph: {annotated_graph}")
            print(f"{'=' * 70}\n")

        # Create fresh model
        model = model_factory()
        prepared_model = self.prepare_model(model)

        # Train
        trainer = Trainer(
            model=prepared_model,
            device=self.device,
            lr=lr,
            weight_decay=weight_decay,
            optimizer_type='adam'
        )

        results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=epochs,
            patience=patience,
            verbose=verbose
        )

        return results

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
            patience: int = 50,
            lr: float = 0.01,
            weight_decay: float = 5e-4,
            verbose: bool = True,
            save_results: bool = True,
            results_dir: str = "results",
            run_id: Optional[str] = None  # ✅ NOUVEAU PARAMÈTRE
    ) -> Dict:
        """
        Evaluate a model across multiple random seeds.

        Parameters:
        - kg_name: Knowledge graph name
        - model_factory: Function that returns a fresh model instance
        - init_embd: Embedding initialization method
        - seeds: List of random seeds to evaluate on
        - splits_dir: Directory containing split files
        - entities_embd_path: Optional path to entity embeddings
        - edges_embd_path: Optional path to edge embeddings
        - epochs: Number of training epochs
        - patience: Early stopping patience
        - lr: Learning rate
        - weight_decay: Weight decay
        - verbose: Print training progress
        - save_results: Whether to save results to file
        - results_dir: Directory to save results
        - run_id: Unique identifier for this run (used in filenames)

        Returns:
        - Dictionary with aggregated results across all seeds
        """
        all_results = {
            'seeds': seeds,
            'per_seed': [],
            'aggregated': {}
        }

        metrics_per_seed = {
            'train_acc': [],
            'train_f1': [],
            'val_acc': [],
            'val_f1': [],
            'test_acc': [],
            'test_f1': [],
            'best_epoch': []
        }

        print(f"\n{'#' * 70}")
        print(f"# Evaluating model on {len(seeds)} seeds: {seeds}")
        print(f"# Knowledge Graph: {kg_name}")
        print(f"# Embedding: {init_embd}")
        if run_id:
            print(f"# Run ID: {run_id}")
        print(f"{'#' * 70}\n")

        for i, seed in enumerate(seeds, 1):
            seed_everything(seed, deterministic=True)
            print(f"\n{'=' * 70}")
            print(f"SEED {i}/{len(seeds)}: {seed}")
            print(f"{'=' * 70}")

            split_path = f"{splits_dir}/split_{seed}.json"

            try:
                # Evaluate on this seed
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
                    verbose=verbose
                )

                # Store results for this seed
                seed_result = {
                    'seed': seed,
                    'best_val_f1': results['best_val']['f1'],
                    'best_epoch': results['best_val']['epoch'],
                    'final_test': results['final_test'],
                    'history': results['history']
                }
                all_results['per_seed'].append(seed_result)

                # Extract final metrics from history
                train_acc = results['history']['train_acc'][-1]
                train_f1 = results['history']['train_f1'][-1]
                val_acc = results['history']['val_acc'][-1]
                val_f1 = results['history']['val_f1'][-1]
                test_acc = results['final_test']['accuracy']
                test_f1 = results['final_test']['f1']

                metrics_per_seed['train_acc'].append(train_acc)
                metrics_per_seed['train_f1'].append(train_f1)
                metrics_per_seed['val_acc'].append(val_acc)
                metrics_per_seed['val_f1'].append(val_f1)
                metrics_per_seed['test_acc'].append(test_acc)
                metrics_per_seed['test_f1'].append(test_f1)
                metrics_per_seed['best_epoch'].append(results['best_val']['epoch'])

                print(f"\n✓ Seed {seed} completed:")
                print(f"  Train: Acc={train_acc:.4f}, F1={train_f1:.4f}")
                print(f"  Val:   Acc={val_acc:.4f}, F1={val_f1:.4f}")
                print(f"  Test:  Acc={test_acc:.4f}, F1={test_f1:.4f}")

            except Exception as e:
                print(f"\n✗ Error on seed {seed}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        # Compute aggregated statistics
        print(f"\n{'#' * 70}")
        print("# AGGREGATED RESULTS")
        print(f"{'#' * 70}\n")

        for metric_name, values in metrics_per_seed.items():
            if len(values) > 0:
                mean = np.mean(values)
                std = np.std(values)
                all_results['aggregated'][metric_name] = {
                    'mean': float(mean),
                    'std': float(std),
                    'values': values
                }
                print(f"{metric_name:15s}: {mean:.4f} ± {std:.4f}")

        # Save results if requested
        if save_results:
            results_path = Path(results_dir)
            results_path.mkdir(parents=True, exist_ok=True)

            # ✅ UTILISER run_id dans les noms de fichiers si fourni
            if run_id:
                base_name = run_id
            else:
                base_name = f"{kg_name}_{init_embd.replace('/', '_')}"

            # Save JSON with all details
            json_path = results_path / f"results_{base_name}.json"
            with open(json_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\n✓ Detailed results saved to: {json_path}")

            # Save summary CSV
            summary_data = {
                'metric': [],
                'mean': [],
                'std': []
            }
            for metric_name, stats in all_results['aggregated'].items():
                summary_data['metric'].append(metric_name)
                summary_data['mean'].append(stats['mean'])
                summary_data['std'].append(stats['std'])

            df_summary = pd.DataFrame(summary_data)
            csv_path = results_path / f"summary_{base_name}.csv"
            df_summary.to_csv(csv_path, index=False)
            print(f"✓ Summary saved to: {csv_path}")

            # Save per-seed detailed results
            per_seed_data = []
            for seed_result in all_results['per_seed']:
                per_seed_data.append({
                    'seed': seed_result['seed'],
                    'train_acc': seed_result['history']['train_acc'][-1],
                    'train_f1': seed_result['history']['train_f1'][-1],
                    'val_acc': seed_result['history']['val_acc'][-1],
                    'val_f1': seed_result['history']['val_f1'][-1],
                    'test_acc': seed_result['final_test']['accuracy'],
                    'test_f1': seed_result['final_test']['f1'],
                    'best_epoch': seed_result['best_epoch']
                })

            df_per_seed = pd.DataFrame(per_seed_data)
            per_seed_csv = results_path / f"per_seed_{base_name}.csv"
            df_per_seed.to_csv(per_seed_csv, index=False)
            print(f"✓ Per-seed results saved to: {per_seed_csv}")

        return all_results