"""Aggregate test precision, recall and F1 per class from saved predictions.

The script is deliberately post-hoc: it never retrains a model.  For every
Sentence-BERT setting, it selects the best hidden-size configuration by mean
validation Macro-F1 for every graph/architecture reported in the main-results
table, reads its saved best-test predictions, and aggregates class-wise scores
with the benchmark's two-level protocol (between-split SD and mean
within-split random-seed SD).

Prediction CSVs are written only when an experiment was run with an artifacts
directory.  If they are absent, a missing-artifacts report is produced instead
of silently reporting incomplete class metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

# Allow ``python evaluate/per_class_performance.py`` from the repository root.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluate.aggregate_experiment_results import (
    _model_configuration,
    find_result_files,
    row_from_result,
)
from evaluate.evaluation_statistics import aggregate_split_and_randomness


METRICS = ("precision", "recall", "f1")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _best_model_files(results_dir: Path) -> list[Path]:
    """Select one fully completed width per graph and architecture by validation."""
    candidates: list[dict[str, Any]] = []
    for path in find_result_files(results_dir):
        row = row_from_result(path)
        if not row["complete"]:
            continue
        row["path"] = path
        candidates.append(row)
    if not candidates:
        raise ValueError(f"No complete detailed results found under {results_dir}")

    selected: list[Path] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        groups[(str(row["graph_name"]), str(row["architecture"]))].append(row)
    for _, rows in sorted(groups.items()):
        winner = min(
            rows,
            key=lambda row: (
                -float(row["mean_best_validation_f1"]),
                int(row["hidden_channels"] or 0),
                int(row["out_channels"] or 0),
            ),
        )
        selected.append(Path(winner["path"]))
    return selected


def _prediction_path(run: dict[str, Any]) -> str | None:
    artifacts = run.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    predictions = artifacts.get("predictions")
    if isinstance(predictions, dict) and isinstance(predictions.get("test"), str):
        return predictions["test"]
    # Compatibility with per_run CSVs created by older benchmark runs.
    direct = artifacts.get("test_predictions_path")
    return direct if isinstance(direct, str) else None


def _resolve_path(stored: str, result_path: Path) -> Path:
    """Resolve paths written on Windows or Linux from the current workspace."""
    raw = Path(stored)
    if raw.exists():
        return raw
    normalized = Path(stored.replace("\\", os.sep).replace("/", os.sep))
    if normalized.exists():
        return normalized
    if not normalized.is_absolute():
        for ancestor in (Path.cwd(), result_path.parent, result_path.parent.parent):
            candidate = ancestor / normalized
            if candidate.exists():
                return candidate
    return normalized


def _class_scores(prediction_csv: Path) -> dict[str, dict[str, float]]:
    frame = pd.read_csv(prediction_csv)
    required = {"label", "predictions"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{prediction_csv} is missing columns: {sorted(missing)}")
    labels = sorted(set(frame["label"].astype(str)).union(frame["predictions"].astype(str)))
    precision, recall, f1, support = precision_recall_fscore_support(
        frame["label"].astype(str),
        frame["predictions"].astype(str),
        labels=labels,
        zero_division=0,
    )
    return {
        label: {
            "precision": float(p),
            "recall": float(r),
            "f1": float(score),
            "support": float(n),
        }
        for label, p, r, score, n in zip(labels, precision, recall, f1, support)
    }


def process_embedding(results_dir: Path, embedding_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return class metrics, selected configurations and unavailable runs."""
    observations: dict[tuple[str, str, int, int, str, str], list[float]] = defaultdict(list)
    selected_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []

    for result_path in _best_model_files(results_dir):
        result = _load_json(result_path)
        graph_name = str(result.get("graph_variant") or result.get("kg_name"))
        architecture, hidden, out = _model_configuration(result_path, result)
        selected_rows.append(
            {
                "embedding": embedding_name,
                "graph_name": graph_name,
                "architecture": architecture,
                "hidden_channels": hidden,
                "out_channels": out,
                "mean_validation_f1": float(np.mean([
                    float(run["best_val_f1"])
                    for run in result["per_run"]
                    if run.get("best_val_f1") is not None
                ])),
                "source_result": str(result_path),
            }
        )
        for run in result.get("per_run", []):
            if not isinstance(run, dict):
                continue
            split_seed = int(run["split_seed"])
            random_seed = int(run["random_seed"])
            stored_path = _prediction_path(run)
            path = _resolve_path(stored_path, result_path) if stored_path else None
            if path is None or not path.exists():
                missing_rows.append(
                    {
                        "embedding": embedding_name,
                        "graph_name": graph_name,
                        "architecture": architecture,
                        "hidden_channels": hidden,
                        "out_channels": out,
                        "split_seed": split_seed,
                        "random_seed": random_seed,
                        "stored_prediction_path": stored_path or "",
                        "reason": "test prediction CSV was not saved or is unavailable",
                    }
                )
                continue
            for class_name, scores in _class_scores(path).items():
                for metric in METRICS:
                    observations[
                        (graph_name, architecture, hidden or 0, out or 0, class_name, metric, split_seed)
                    ].append(scores[metric])
                observations[
                    (graph_name, architecture, hidden or 0, out or 0, class_name, "support", split_seed)
                ].append(scores["support"])

    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, int, int, str], dict[str, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for (graph, model, hidden, out, class_name, metric, split), values in observations.items():
        grouped[(graph, model, hidden, out, class_name)][metric][split] = values

    for (graph, model, hidden, out, class_name), metric_values in sorted(grouped.items()):
        row: dict[str, Any] = {
            "embedding": embedding_name,
            "graph_name": graph,
            "architecture": model,
            "hidden_channels": hidden,
            "out_channels": out,
            "class": class_name,
        }
        for metric in METRICS:
            stats = aggregate_split_and_randomness(metric_values[metric])
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_split_std"] = stats["split_std"]
            row[f"{metric}_randomness_std"] = stats["randomness_std"]
            row[f"{metric}_n_runs"] = stats["n_scores"]
        supports = [score for values in metric_values["support"].values() for score in values]
        row["mean_test_support"] = float(np.mean(supports))
        rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(selected_rows), pd.DataFrame(missing_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plm-results-dir",
        default="results/experiment_1_static/sentencebert",
        help="Experiment-1 Sentence-BERT results directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/experiment_1_static/per_class_performance",
        help="Directory for CSV, Excel, and missing-artifacts reports.",
    )
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics, selected, missing = process_embedding(
        Path(args.plm_results_dir), "Sentence-BERT"
    )
    metrics.to_csv(output / "per_class_metrics.csv", index=False)
    selected.to_csv(output / "selected_configurations.csv", index=False)
    missing.to_csv(output / "missing_prediction_artifacts.csv", index=False)
    with pd.ExcelWriter(output / "per_class_metrics.xlsx") as writer:
        metrics.to_excel(writer, sheet_name="Sentence_BERT", index=False)
        selected.to_excel(writer, sheet_name="selected_configs", index=False)
        missing.to_excel(writer, sheet_name="missing_predictions", index=False)

    print(f"Class metrics: {output / 'per_class_metrics.csv'}")
    print(f"Selected configurations: {output / 'selected_configurations.csv'}")
    print(f"Missing prediction artifacts: {output / 'missing_prediction_artifacts.csv'}")
    print(f"Rows with class metrics: {len(metrics)}; missing prediction runs: {len(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
