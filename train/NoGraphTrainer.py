"""Training loop for a feature-only MLP baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class NoGraphTrainer:
    """Full-batch trainer that never receives an edge list or relation types."""

    def __init__(self, model, device, lr: float, weight_decay: float) -> None:
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.history = {
            "train_loss": [], "train_acc": [], "train_f1": [],
            "val_acc": [], "val_f1": [], "test_acc": [], "test_f1": [],
        }
        self.best_val_f1 = float("-inf")
        self.best_epoch = 0
        self.best_model_state = None

    @torch.no_grad()
    def _evaluate(self, features, labels, indices, split: str):
        self.model.eval()
        logits = self.model(features)
        y_true = labels[indices].detach().cpu().numpy()
        predictions = logits[indices].argmax(dim=1).detach().cpu().numpy()
        return {
            "split": split,
            "accuracy": accuracy_score(y_true, predictions),
            "recall": recall_score(y_true, predictions, average="macro", zero_division=0),
            "precision": precision_score(y_true, predictions, average="macro", zero_division=0),
            "f1": f1_score(y_true, predictions, average="macro", zero_division=0),
        }

    @torch.no_grad()
    def _save_predictions(self, features, labels, indices, split, terms, label_encoder, save_path):
        self.model.eval()
        predictions = self.model(features)[indices].argmax(dim=1).detach().cpu().numpy()
        true_labels = labels[indices].detach().cpu().numpy()
        original_indices = indices.detach().cpu().tolist()
        table = pd.DataFrame(
            {
                "node_id": original_indices,
                "term": [terms[index] for index in original_indices],
                "label": label_encoder.inverse_transform(true_labels),
                "predictions": label_encoder.inverse_transform(predictions),
            }
        )
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(destination, index=False)
        print(f"Predictions for '{split}' saved to {destination}")

    def train(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        split_indices: dict[str, torch.Tensor],
        terms: list[str],
        label_encoder,
        *,
        epochs: int = 100,
        patience: int = 100,
        verbose: bool = True,
        artifacts_dir: Optional[str] = None,
        artifact_prefix: str = "run",
        save_model_checkpoint: bool = True,
        save_prediction_splits: Optional[list[str]] = None,
    ) -> dict:
        if epochs < 1 or patience < 1:
            raise ValueError("epochs and patience must both be >= 1")
        if save_prediction_splits is None:
            save_prediction_splits = ["test"]
        unknown_splits = set(save_prediction_splits) - {"train", "val", "test"}
        if unknown_splits:
            raise ValueError(f"Unknown prediction split(s): {sorted(unknown_splits)}")

        features = features.to(self.device)
        labels = labels.to(self.device)
        indices = {name: values.to(self.device) for name, values in split_indices.items()}
        if any(values.numel() == 0 for values in indices.values()):
            raise ValueError("train, val and test splits must all be non-empty")

        stale_epochs = 0
        for epoch in range(1, epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()
            logits = self.model(features)
            loss = self.criterion(logits[indices["train"]], labels[indices["train"]])
            loss.backward()
            self.optimizer.step()

            train_metrics = self._evaluate(features, labels, indices["train"], "train")
            val_metrics = self._evaluate(features, labels, indices["val"], "val")
            test_metrics = self._evaluate(features, labels, indices["test"], "test")
            self.history["train_loss"].append(float(loss.item()))
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["train_f1"].append(train_metrics["f1"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["test_acc"].append(test_metrics["accuracy"])
            self.history["test_f1"].append(test_metrics["f1"])

            if self.best_model_state is None or val_metrics["f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["f1"]
                self.best_epoch = epoch
                stale_epochs = 0
                self.best_model_state = {
                    key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()
                }
            else:
                stale_epochs += 1

            if verbose:
                print(
                    f"Epoch {epoch:3d}/{epochs} | loss={loss.item():.4f} | "
                    f"train F1={train_metrics['f1']:.4f} | val F1={val_metrics['f1']:.4f} | "
                    f"test F1={test_metrics['f1']:.4f}"
                )
            if stale_epochs >= patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch}")
                break

        self.model.load_state_dict(self.best_model_state)
        final_test_metrics = self._evaluate(features, labels, indices["test"], "test")
        artifact_paths: dict[str, object] = {}
        if artifacts_dir is not None:
            directory = Path(artifacts_dir)
            directory.mkdir(parents=True, exist_ok=True)
            if save_model_checkpoint:
                model_path = directory / f"{artifact_prefix}_best_model.pt"
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
            prediction_paths = {}
            for split in save_prediction_splits:
                prediction_path = directory / f"{artifact_prefix}_best_{split}_predictions.csv"
                self._save_predictions(
                    features, labels, indices[split], split, terms, label_encoder, prediction_path
                )
                prediction_paths[split] = str(prediction_path)
            artifact_paths["predictions"] = prediction_paths

        return {
            "best_val": {"f1": self.best_val_f1, "epoch": self.best_epoch},
            "final_test": final_test_metrics,
            "history": self.history,
            "artifacts": artifact_paths,
        }
