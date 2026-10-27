"""Replay selected Experiment-1 Sentence-BERT configurations with test predictions.

The original Experiment 1 saved aggregate metrics but not per-node predictions.
This runner reads its validation-selected configurations, retrains exactly
those 21 graph/architecture/width combinations over 10 splits x 5 random
seeds, and saves only best-test prediction CSVs. It is safe to stop and rerun:
a run is skipped only when both its durable run record and prediction CSV are
present.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from itertools import product
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_preprocessing.splits import PREGENERATED_SPLIT_SEEDS
from evaluate.evaluation_statistics import aggregate_split_and_randomness


EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SELECTION = "results/experiment_1_static/per_class_sentencebert/selected_configurations.csv"
WORKERS_BY_GRAPH = {"GT2KG_kg": 30, "KG_GEN_kg": 20, "UMLS_nci_kg": 8}
WORKERS_BY_GRAPH_MODEL = {("KG_GEN_kg", "RGCN"): 5}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-file", default=DEFAULT_SELECTION)
    parser.add_argument("--results-dir", default="results/experiment_1_plm_per_class_replay")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--max-processes", type=int, default=None,
                        help="Optional cap applied to each graph/model worker limit.")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def atomic_json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
    temporary.replace(path)


def worker_initializer() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def configuration_directory(task: Mapping[str, Any]) -> Path:
    return (
        Path(str(task["results_dir"])) / str(task["graph"]) / str(task["model"])
        / f"hidden_{task['hidden']}_out_{task['out']}"
    )


def run_record_path(task: Mapping[str, Any]) -> Path:
    return (
        configuration_directory(task) / "individual_runs"
        / f"split_{task['split_seed']}__random_{task['random_seed']}.json"
    )


def prediction_exists(run: Mapping[str, Any]) -> bool:
    artifacts = run.get("artifacts")
    predictions = artifacts.get("predictions") if isinstance(artifacts, dict) else None
    path = predictions.get("test") if isinstance(predictions, dict) else None
    if not isinstance(path, str):
        return False
    return Path(path).exists()


def completed(task: Mapping[str, Any]) -> bool:
    path = run_record_path(task)
    if not path.exists():
        return False
    try:
        with path.open(encoding="utf-8") as handle:
            return prediction_exists(json.load(handle))
    except (OSError, ValueError, TypeError):
        return False


def run_one(task: Mapping[str, Any]) -> str:
    """Run a single matched split/randomness pair in a spawned process."""
    worker_initializer()
    if bool(task.get("resume", True)) and completed(task):
        return str(run_record_path(task))

    from tdg_bench import TDGBench

    output_dir = configuration_directory(task)
    suite = TDGBench(use_classifier=True).evaluate_models(
        kg_name=str(task["graph"]),
        model_names=[str(task["model"])],
        init_embd=EMBEDDING,
        split_seeds=[int(task["split_seed"])],
        random_seeds=[int(task["random_seed"])],
        hidden_channels=[int(task["hidden"])],
        out_channels=int(task["out"]),
        random_embd_dim=384,
        epochs=int(task["epochs"]),
        patience=int(task["patience"]),
        verbose=False,
        save_results=False,
        save_models=False,
        save_predictions=True,
        resume=False,
        results_dir=str(output_dir),
    )
    key = f"hidden_{task['hidden']}_out_{task['out']}"
    evaluation = suite["models"][str(task["model"])]["runs"][key]
    if evaluation["failed_runs"] or len(evaluation["per_run"]) != 1:
        raise RuntimeError(f"Unexpected single-run result: {evaluation['failed_runs']}")
    run = evaluation["per_run"][0]
    if not prediction_exists(run):
        raise RuntimeError("Training completed but no test-prediction CSV was saved")
    atomic_json_write(run_record_path(task), run)
    return str(run_record_path(task))


def selected_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    selection = pd.read_csv(args.selection_file)
    expected = {"graph_name", "architecture", "hidden_channels", "out_channels"}
    missing = expected.difference(selection.columns)
    if missing:
        raise ValueError(f"Selection file is missing columns: {sorted(missing)}")
    tasks: list[dict[str, Any]] = []
    for row in selection.sort_values(["graph_name", "architecture"], kind="stable").to_dict("records"):
        for split_seed, random_seed in product(PREGENERATED_SPLIT_SEEDS, [1, 2, 3, 4, 5]):
            tasks.append({
                "graph": str(row["graph_name"]),
                "model": str(row["architecture"]),
                "hidden": int(row["hidden_channels"]),
                "out": int(row["out_channels"]),
                "split_seed": int(split_seed),
                "random_seed": int(random_seed),
                "epochs": args.epochs,
                "patience": args.patience,
                "results_dir": args.results_dir,
            })
    return tasks


def worker_limit(task: Mapping[str, Any], maximum: int | None) -> int:
    graph, model = str(task["graph"]), str(task["model"])
    limit = WORKERS_BY_GRAPH_MODEL.get((graph, model), WORKERS_BY_GRAPH[graph])
    return min(limit, maximum) if maximum is not None else limit


def aggregate_configuration(prototype: Mapping[str, Any]) -> dict[str, Any]:
    per_run: list[dict[str, Any]] = []
    failed_runs: list[dict[str, Any]] = []
    split_seeds, random_seeds = list(PREGENERATED_SPLIT_SEEDS), [1, 2, 3, 4, 5]
    for split_seed, random_seed in product(split_seeds, random_seeds):
        task = {**prototype, "split_seed": split_seed, "random_seed": random_seed}
        path = run_record_path(task)
        if not path.exists():
            failed_runs.append({"split_seed": split_seed, "random_seed": random_seed, "error": "missing run"})
            continue
        with path.open(encoding="utf-8") as handle:
            run = json.load(handle)
        if not prediction_exists(run):
            failed_runs.append({"split_seed": split_seed, "random_seed": random_seed, "error": "missing prediction CSV"})
            continue
        per_run.append(run)
    per_run.sort(key=lambda value: (int(value["split_seed"]), int(value["random_seed"])))

    metrics_by_split: dict[int, dict[str, list[float]]] = {
        seed: {metric: [] for metric in ("train_acc", "train_f1", "val_acc", "val_f1", "test_acc", "test_recall", "test_precision", "test_f1", "best_epoch")}
        for seed in split_seeds
    }
    for run in per_run:
        values = metrics_by_split[int(run["split_seed"])]
        history, final = run["history"], run["final_test"]
        values["train_acc"].append(float(history["train_acc"][-1]))
        values["train_f1"].append(float(history["train_f1"][-1]))
        values["val_acc"].append(float(history["val_acc"][-1]))
        values["val_f1"].append(float(history["val_f1"][-1]))
        values["test_acc"].append(float(final["accuracy"]))
        values["test_recall"].append(float(final["recall"]))
        values["test_precision"].append(float(final["precision"]))
        values["test_f1"].append(float(final["f1"]))
        values["best_epoch"].append(float(run["best_epoch"]))
    aggregated = {
        metric: aggregate_split_and_randomness({seed: scores[metric] for seed, scores in metrics_by_split.items() if scores[metric]})
        for metric in next(iter(metrics_by_split.values()))
        if any(scores[metric] for scores in metrics_by_split.values())
    }
    result = {
        "kg_name": prototype["graph"], "graph_variant": prototype["graph"], "options": [],
        "init_embd": EMBEDDING, "random_embd_dim": 384, "split_seeds": split_seeds,
        "random_seeds": random_seeds, "total_runs_requested": 50,
        "epochs": prototype["epochs"], "patience": prototype["patience"],
        "lr": 0.01, "weight_decay": 5e-4, "save_models": False, "save_predictions": True,
        "per_run": per_run, "failed_runs": failed_runs, "runs_completed": len(per_run),
        "runs_failed": len(failed_runs), "aggregated": aggregated,
    }
    directory = configuration_directory(prototype)
    basename = f"{prototype['graph']}__{prototype['model']}__h{prototype['hidden']}__out{prototype['out']}"
    atomic_json_write(directory / f"results_{basename}.json", result)
    pd.DataFrame([
        {
            "split_seed": run["split_seed"], "random_seed": run["random_seed"],
            "best_val_f1": run["best_val_f1"], "best_epoch": run["best_epoch"],
            "test_acc": run["final_test"]["accuracy"], "test_f1": run["final_test"]["f1"],
            "test_predictions_path": run["artifacts"]["predictions"]["test"],
        }
        for run in per_run
    ]).to_csv(directory / f"per_run_{basename}.csv", index=False)
    return result


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("epochs and patience must be positive")
    if args.max_processes is not None and args.max_processes < 1:
        raise ValueError("max-processes must be positive")
    tasks = selected_tasks(args)
    groups: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
    for task in tasks:
        groups.setdefault((task["graph"], task["model"], task["hidden"], task["out"]), []).append(task)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_file": str(args.selection_file), "embedding": EMBEDDING,
        "epochs": args.epochs, "patience": args.patience, "n_configurations": len(groups),
        "n_requested_runs": len(tasks),
    }
    atomic_json_write(Path(args.results_dir) / "replay_manifest.json", manifest)
    all_failures: list[dict[str, Any]] = []
    for key, group in groups.items():
        graph, model, hidden, out = key
        pending = [task for task in group if args.no_resume or not completed(task)]
        workers = worker_limit(group[0], args.max_processes)
        print(f"\n{'=' * 72}\n{graph} | {model} | h={hidden} | pending={len(pending)}/50 | workers={workers}\n{'=' * 72}")
        if args.dry_run or not pending:
            continue
        for task in pending:
            task["resume"] = not args.no_resume
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"), initializer=worker_initializer) as executor:
            futures = {executor.submit(run_one, task): task for task in pending}
            for number, future in enumerate(as_completed(futures), 1):
                task = futures[future]
                try:
                    future.result()
                    print(f"  [{number}/{len(pending)}] split={task['split_seed']} random={task['random_seed']}")
                except Exception as error:
                    failure = {**{name: task[name] for name in ("graph", "model", "hidden", "out", "split_seed", "random_seed")}, "error": f"{type(error).__name__}: {error}"}
                    all_failures.append(failure)
                    print(f"  [RUN ERROR] {failure}")
        aggregate_configuration(group[0])

    # Also rebuild aggregate files for fully resumed groups.  A dry run must
    # remain read-only with respect to result content.
    if not args.dry_run:
        for group in groups.values():
            aggregate_configuration(group[0])
        atomic_json_write(Path(args.results_dir) / "replay_failures.json", {"failed_runs": all_failures})
    print(f"\nFinished. Configurations: {len(groups)}; run errors: {len(all_failures)}")
    print(f"Prediction-ready results: {args.results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
