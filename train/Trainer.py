import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from typing import Optional, Dict
from pathlib import Path


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
            lr: float = 0.01,
            weight_decay: float = 5e-4,
            optimizer_type: str = 'adam'
    ):
        """
        Trainer for semi-supervised node classification with mini-batch training.

        Parameters:
        - model: GNN model to train
        - device: Device for computation ('cuda' or 'cpu')
        - lr: Learning rate
        - weight_decay: L2 regularization
        - optimizer_type: Optimizer type ('adam', 'sgd', etc.)
        """
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay

        # Initialize optimizer
        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Optimizer {optimizer_type} not supported")

        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'train_f1': [],
            'val_acc': [],
            'val_f1': [],
            'test_acc': [],
            'test_f1': []
        }

        # Best model tracking
        self.best_val_f1 = 0.0
        self.best_model_state = None
        self.best_epoch = 0

    # def _forward_pass(self, batch):
    #     """
    #     Handle forward pass for both Data objects and separate arguments.
    #
    #     Returns model output.
    #     """
    #     # Check if model expects Data object or (x, edge_index, ...)
    #     if hasattr(batch, 'x') and hasattr(batch, 'edge_index'):
    #         # Extract attributes from batch
    #         x = batch.x
    #         edge_index = batch.edge_index
    #         edge_type = getattr(batch, 'edge_type', None)
    #         edge_weight = getattr(batch, 'edge_attr', None)
    #
    #         # Call model with appropriate arguments
    #         if edge_type is not None:
    #             out = self.model(x, edge_index, edge_type, edge_weight)
    #         else:
    #             out = self.model(x, edge_index, edge_weight)
    #     else:
    #         # Fallback: pass batch directly (for models that expect Data objects)
    #         out = self.model(batch)
    #
    #     return out

    # def _encode_pass(self, batch):
    #     """
    #     Retourne les embeddings du GNN encoder, avant la tête de classification.
    #     """
    #     if hasattr(batch, "x") and hasattr(batch, "edge_index"):
    #         return self.model.encoder(
    #             x=batch.x,
    #             edge_index=batch.edge_index,
    #             edge_type=getattr(batch, "edge_type", None),
    #             edge_weight=getattr(batch, "edge_weight", None),
    #             edge_attr=getattr(batch, "edge_attr", None),
    #         )

    #     return self.model.encoder(batch)

    def _forward_pass(self, batch):
        """
        Handle forward pass for both Data objects and separate arguments.

        Returns model output.
        """
        if hasattr(batch, 'x') and hasattr(batch, 'edge_index'):
            x = batch.x
            edge_index = batch.edge_index
            edge_attr = getattr(batch, 'edge_attr', None)
            edge_type = getattr(batch, 'edge_type', None)
            edge_weight = getattr(batch, 'edge_weight', None)

            # Passe TOUS les arguments, le modèle décide quoi utiliser
            out = self.model(
                x=x,
                edge_index=edge_index,
                edge_type=edge_type,
                edge_weight=edge_weight,
                edge_attr=edge_attr  # ← Ajouter ce paramètre
            )
        else:
            out = self.model(batch)

        return out

    def _encode_pass(self, batch):
        """
        Forward de l'encoder seulement.
        Retourne les embeddings avant la tête de classification.
        """
        if not hasattr(self.model, "encoder"):
            raise AttributeError("Model must have .encoder attribute")

        encoder = self.model.encoder

        if hasattr(batch, "x") and hasattr(batch, "edge_index"):
            x = batch.x
            edge_index = batch.edge_index
            edge_attr = getattr(batch, "edge_attr", None)
            edge_type = getattr(batch, "edge_type", None)
            edge_weight = getattr(batch, "edge_weight", None)

            # 1) TransGCN / RotatEGCN souvent : forward(x, edge_index, edge_attr)
            if edge_attr is not None:
                try:
                    return encoder(
                        x=x,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                    )
                except TypeError:
                    pass

                try:
                    return encoder(
                        x=x,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        edge_type=edge_type,
                    )
                except TypeError:
                    pass

            # 2) RGCN : forward(x, edge_index, edge_type, edge_weight=None)
            if edge_type is not None:
                try:
                    return encoder(
                        x=x,
                        edge_index=edge_index,
                        edge_type=edge_type,
                        edge_weight=edge_weight,
                    )
                except TypeError:
                    pass

            # 3) GCN classique
            try:
                return encoder(
                    x=x,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                )
            except TypeError:
                pass

            # 4) Minimal
            return encoder(
                x=x,
                edge_index=edge_index,
            )

        return encoder(batch)
    def _train_epoch(self, train_loader) -> float:
        """
        Train for one epoch using mini-batches.

        Returns:
        - Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            out = self._forward_pass(batch)

            # Compute loss only on input nodes in training set
            mask_input_id = torch.isin(batch.n_id, batch.input_id)
            mask = mask_input_id & batch.train_mask & (batch.y >= 0)

            loss = self.criterion(out[mask], batch.y[mask])

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    @torch.no_grad()
    def _evaluate_split(
            self,
            data_loader,
            split: str,
            save_predictions: bool = False,
            save_path: Optional[str] = None,
            label_encoder=None,
            decode_indexes_fn=None
    ) -> Dict[str, float]:
        """
        Evaluate model on a specific split.

        Parameters:
        - data_loader: DataLoader for the split
        - split: 'train', 'val', or 'test'
        - save_predictions: Whether to save predictions to file
        - save_path: Path to save predictions Excel file
        - label_encoder: Optional label encoder for decoding
        - decode_indexes_fn: Optional function to decode node IDs to terms

        Returns:
        - Dictionary with metrics (accuracy, f1, recall, precision)
        """
        assert split in ["train", "val", "test"], "split must be 'train', 'val', or 'test'"

        self.model.eval()
        y_true, preds, node_ids = [], [], []

        mask_name = f"{split}_mask"

        for batch in data_loader:
            batch = batch.to(self.device)
            out = self._forward_pass(batch)

            # Keep only annotated input nodes from the correct split
            mask_input_id = torch.isin(batch.n_id, batch.input_id)
            mask = mask_input_id & getattr(batch, mask_name) & (batch.y >= 0)

            batch_y_true = batch.y[mask].cpu()
            batch_y_pred = out[mask].argmax(dim=1).cpu()
            batch_node_ids = batch.n_id[mask].cpu()

            y_true.append(batch_y_true)
            preds.append(batch_y_pred)
            node_ids.append(batch_node_ids)

        if len(y_true) == 0:
            print(f"[WARN] No annotated nodes found for split '{split}'")
            return {
                'split': split,
                'accuracy': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'precision': 0.0
            }

        # Concatenate all batches
        y_true = torch.cat(y_true, dim=0).numpy()
        preds = torch.cat(preds, dim=0).numpy()
        node_ids = torch.cat(node_ids, dim=0).numpy()

        # Decode labels if encoder provided
        if label_encoder:
            decoded_true = label_encoder.inverse_transform(y_true)
            decoded_pred = label_encoder.inverse_transform(preds)
        else:
            decoded_true, decoded_pred = y_true, preds

        # Save predictions if requested
        if save_predictions and save_path:
            terms = [decode_indexes_fn(id) for id in node_ids] if decode_indexes_fn else node_ids
            df = pd.DataFrame({
                "node_id": node_ids,
                "term": terms,
                "label": decoded_true,
                "predictions": decoded_pred
            })
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if save_path.suffix.lower() == ".csv":
                df.to_csv(save_path, index=False)
            else:
                df.to_excel(save_path, index=False)
            print(f"Predictions for '{split}' saved to {save_path}")

        # Compute metrics
        return {
            'split': split,
            'accuracy': accuracy_score(y_true, preds),
            'recall': recall_score(y_true, preds, average='macro', zero_division=0),
            'f1': f1_score(y_true, preds, average='macro', zero_division=0),
            'precision': precision_score(y_true, preds, average='macro', zero_division=0)
        }

    def train(
            self,
            train_loader,
            val_loader,
            test_loader,
            epochs: int = 100,
            patience: int = 100,
            verbose: bool = True,
            eval_every: int = 1,
            save_best_model: bool = True,
            artifacts_dir: Optional[str] = None,
            artifact_prefix: str = "run",
            label_encoder=None,
            decode_indexes_fn=None,
            save_prediction_splits: Optional[list] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Full training loop with early stopping.

        Parameters:
        - train_loader: DataLoader for training
        - val_loader: DataLoader for validation
        - test_loader: DataLoader for testing
        - epochs: Maximum number of epochs
        - patience: Early stopping patience
        - verbose: Print training progress
        - eval_every: Evaluate every N epochs
        - save_best_model: Keep best model state
        - artifacts_dir: Optional directory where the best checkpoint and predictions are saved
        - artifact_prefix: Prefix used for saved artifact filenames
        - save_prediction_splits: Splits to save predictions for after reloading the best model

        Returns:
        - Dictionary with best validation and final test metrics
        """
        patience_counter = 0
        artifact_paths = {}

        if save_prediction_splits is None:
            save_prediction_splits = ["test"]

        if verbose:
            print("Starting training...")
            print("=" * 70)

        for epoch in range(1, epochs + 1):
            # Training
            avg_loss = self._train_epoch(train_loader)
            self.history['train_loss'].append(avg_loss)

            # Evaluation
            if epoch % eval_every == 0:
                train_metrics = self._evaluate_split(train_loader, "train")
                val_metrics = self._evaluate_split(val_loader, "val")
                test_metrics = self._evaluate_split(test_loader, "test")

                # Update history
                self.history['train_acc'].append(train_metrics['accuracy'])
                self.history['train_f1'].append(train_metrics['f1'])
                self.history['val_acc'].append(val_metrics['accuracy'])
                self.history['val_f1'].append(val_metrics['f1'])
                self.history['test_acc'].append(test_metrics['accuracy'])
                self.history['test_f1'].append(test_metrics['f1'])

                # Track best model
                if val_metrics['f1'] > self.best_val_f1:
                    self.best_val_f1 = val_metrics['f1']
                    self.best_epoch = epoch
                    patience_counter = 0

                    if save_best_model:
                        self.best_model_state = {
                            k: v.cpu().clone() for k, v in self.model.state_dict().items()
                        }
                else:
                    patience_counter += 1

                # Logging
                if verbose:
                    print(f"Epoch {epoch:3d}/{epochs}")
                    print(f"  Loss:     {avg_loss:.4f}")
                    print(f"  Train  -  Acc: {train_metrics['accuracy']:.4f} | F1: {train_metrics['f1']:.4f}")
                    print(f"  Val    -  Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}")
                    print(f"  Test   -  Acc: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1']:.4f}")
                    print(f"  Best Val F1: {self.best_val_f1:.4f} (Epoch {self.best_epoch})")
                    print("-" * 70)

                # Early stopping
                if patience_counter >= patience:
                    if verbose:
                        print(f"\nEarly stopping triggered at epoch {epoch}")
                    break

        # Load best model and evaluate on test set
        if verbose:
            print(f"\nTraining completed! Best Val F1: {self.best_val_f1:.4f} at epoch {self.best_epoch}")

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            final_test_metrics = self._evaluate_split(test_loader, "test")
        else:
            final_test_metrics = test_metrics

        if artifacts_dir is not None:
            artifacts_path = Path(artifacts_dir)
            artifacts_path.mkdir(parents=True, exist_ok=True)

            model_path = artifacts_path / f"{artifact_prefix}_best_model.pt"
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "best_val_f1": self.best_val_f1,
                    "best_epoch": self.best_epoch,
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                },
                model_path,
            )
            artifact_paths["best_model"] = str(model_path)
            print(f"Best model saved to {model_path}")

            loaders_by_split = {
                "train": train_loader,
                "val": val_loader,
                "test": test_loader,
            }

            prediction_paths = {}
            for split_name in save_prediction_splits:
                if split_name not in loaders_by_split:
                    raise ValueError(
                        f"Unknown prediction split '{split_name}'. "
                        "Supported values are 'train', 'val', and 'test'."
                    )

                pred_path = artifacts_path / f"{artifact_prefix}_best_{split_name}_predictions.csv"
                split_metrics = self._evaluate_split(
                    loaders_by_split[split_name],
                    split_name,
                    save_predictions=True,
                    save_path=str(pred_path),
                    label_encoder=label_encoder,
                    decode_indexes_fn=decode_indexes_fn,
                )
                prediction_paths[split_name] = str(pred_path)

                if split_name == "test":
                    final_test_metrics = split_metrics

            artifact_paths["predictions"] = prediction_paths

        return {
            'best_val': {
                'f1': self.best_val_f1,
                'epoch': self.best_epoch
            },
            'final_test': final_test_metrics,
            'history': self.history,
            'artifacts': artifact_paths,
        }

    def evaluate(
            self,
            data_loader,
            split: str = 'test',
            save_predictions: bool = False,
            save_path: Optional[str] = None,
            label_encoder=None,
            decode_indexes_fn=None
    ) -> Dict[str, float]:
        """
        Evaluate the model on a data loader.
        """
        return self._evaluate_split(
            data_loader,
            split,
            save_predictions,
            save_path,
            label_encoder,
            decode_indexes_fn
        )

    def get_best_model(self) -> nn.Module:
        """Load and return the best model."""
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        return self.model

    def save_model(self, path: str):
        """Save the current model state."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model state from file."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")
