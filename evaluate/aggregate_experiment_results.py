"""Aggregate persistent TDG-Bench detailed result JSONs into final tables.

Training writes one ``results_*.json`` file per graph, architecture and width.
This module recursively reads those durable leaf files, so it remains correct
when a launch script evaluates configurations one after another.  It produces
one Excel workbook and companion CSVs without retraining any model.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


CONFIGURATION = re.compile(
    r"^(?P<architecture>.+)__h(?P<hidden>\d+)__out(?P<out>\d+)$"
)
METRICS = {
    "test_accuracy": "test_acc",
    "test_recall": "test_recall",
    "test_precision": "test_precision",
    "test_macro_f1": "test_f1",
}


def find_result_files(results_dir: str | Path) -> list[Path]:
    root = Path(results_dir)
    files = sorted(root.rglob("results_*.json"))
    if not files:
        raise FileNotFoundError(f"No detailed results_*.json files found under {root}")
    return files


def _model_configuration(path: Path, result: dict[str, Any]) -> tuple[str, int | None, int | None]:
    """Extract architecture/width from a TDG or no-graph detailed JSON."""
    graph_name = str(result.get("graph_variant") or result.get("kg_name") or "")
    prefix = f"results_{graph_name}__"
    label = path.stem[len(prefix):] if path.stem.startswith(prefix) else ""
    match = CONFIGURATION.fullmatch(label)
    if match:
        return match.group("architecture"), int(match.group("hidden")), int(match.group("out"))

    # Long Windows paths may intentionally shorten the file name to
    # ``results_run.json``. The parent directories retain the full identity.
    directory_match = re.fullmatch(
        r"hidden_(?P<hidden>\d+)_out_(?P<out>\d+)", path.parent.name
    )
    if directory_match and path.parent.parent.name:
        return (
            path.parent.parent.name,
            int(directory_match.group("hidden")),
            int(directory_match.group("out")),
        )

    model_name = result.get("model_name")
    if isinstance(model_name, str) and model_name:
        # NoGraphMLP results use their class/source as the model label.
        hidden = path.parent.name.removeprefix("hidden_").split("_out_", 1)[0]
        out = path.parent.name.split("_out_", 1)[-1]
        return model_name, int(hidden) if hidden.isdigit() else None, int(out) if out.isdigit() else None
    raise ValueError(f"Cannot infer model configuration from {path}")


def row_from_result(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"Detailed result must be a JSON object: {path}")
    graph_name = result.get("graph_variant") or result.get("kg_name")
    if not isinstance(graph_name, str) or not graph_name:
        raise ValueError(f"Missing graph name in {path}")
    architecture, hidden, out = _model_configuration(path, result)
    per_run = result.get("per_run")
    if not isinstance(per_run, list):
        raise ValueError(f"Missing per_run list in {path}")
    validation = [
        float(run["best_val_f1"])
        for run in per_run
        if isinstance(run, dict) and run.get("best_val_f1") is not None
    ]
    requested = int(result.get("total_runs_requested", len(per_run)))
    row: dict[str, Any] = {
        "graph_name": graph_name,
        "init_embd": result.get("init_embd"),
        "architecture": architecture,
        "hidden_channels": hidden,
        "out_channels": out,
        "runs_requested": requested,
        "runs_completed": len(per_run),
        "runs_failed": len(result.get("failed_runs", [])),
        "complete": len(per_run) == requested,
        "mean_best_validation_f1": float(np.mean(validation)) if validation else float("nan"),
        "split_seeds": ",".join(map(str, result.get("split_seeds", []))),
        "random_seeds": ",".join(map(str, result.get("random_seeds", []))),
        "source_file": str(path),
    }
    aggregated = result.get("aggregated", {})
    if not isinstance(aggregated, dict):
        raise ValueError(f"Invalid aggregated section in {path}")
    for output_name, stored_name in METRICS.items():
        values = aggregated.get(stored_name, {})
        if not isinstance(values, dict):
            values = {}
        row[f"{output_name}_mean"] = values.get("mean")
        row[f"{output_name}_split_std"] = values.get("split_std")
        row[f"{output_name}_randomness_std"] = values.get("randomness_std")
        row[f"{output_name}_overall_std"] = values.get("overall_std")
    return row


def aggregate_results(result_files: Iterable[str | Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return all configurations, validation-selected architecture reps, and winners."""
    rows = [row_from_result(Path(path)) for path in result_files]
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No result rows to aggregate")
    duplicate_keys = ["graph_name", "init_embd", "architecture", "hidden_channels", "out_channels"]
    duplicates = table.duplicated(duplicate_keys, keep=False)
    if duplicates.any():
        duplicate_rows = table.loc[duplicates, duplicate_keys + ["source_file"]]
        raise ValueError(
            "Duplicate detailed configurations found. Aggregate one experiment/output root at a time:\n"
            + duplicate_rows.to_string(index=False)
        )
    table = table.sort_values(
        ["init_embd", "graph_name", "architecture", "hidden_channels", "out_channels"],
        kind="stable",
    ).reset_index(drop=True)
    complete = table.loc[table["complete"]].copy()
    group_architecture = ["init_embd", "graph_name", "architecture"]
    representatives = (
        complete.sort_values(
            group_architecture + ["mean_best_validation_f1", "hidden_channels", "out_channels"],
            ascending=[True, True, True, False, True, True],
            kind="stable",
        )
        .groupby(group_architecture, as_index=False, sort=False)
        .head(1)
        .reset_index(drop=True)
    )
    group_graph = ["init_embd", "graph_name"]
    winners = (
        complete.sort_values(
            group_graph + ["mean_best_validation_f1", "hidden_channels", "out_channels", "architecture"],
            ascending=[True, True, False, True, True, True],
            kind="stable",
        )
        .groupby(group_graph, as_index=False, sort=False)
        .head(1)
        .reset_index(drop=True)
    )
    return table, representatives, winners


def save_aggregates(
    results_dir: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    root = Path(results_dir)
    output = Path(output_dir) if output_dir is not None else root
    output.mkdir(parents=True, exist_ok=True)
    all_rows, representatives, winners = aggregate_results(find_result_files(root))
    paths = {
        "all_configurations_csv": output / "aggregated_configurations.csv",
        "representatives_csv": output / "representative_models_by_graph.csv",
        "best_by_graph_csv": output / "best_by_graph.csv",
        "excel": output / "aggregated_results.xlsx",
    }
    all_rows.to_csv(paths["all_configurations_csv"], index=False)
    representatives.to_csv(paths["representatives_csv"], index=False)
    winners.to_csv(paths["best_by_graph_csv"], index=False)
    with pd.ExcelWriter(paths["excel"]) as writer:
        all_rows.to_excel(writer, sheet_name="all_configurations", index=False)
        representatives.to_excel(writer, sheet_name="representatives", index=False)
        winners.to_excel(writer, sheet_name="best_by_graph", index=False)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, help="Experiment output root to scan recursively.")
    parser.add_argument("--output-dir", default=None, help="Directory for CSV/XLSX outputs (default: results-dir).")
    args = parser.parse_args()
    paths = save_aggregates(args.results_dir, args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
