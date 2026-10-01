"""Run Experiment 1 with a reproducible graph/model-specific worker schedule.

This runner intentionally avoids co-locating different graph/architecture
classes.  Every homogeneous group is completed before the next one begins:

* GT2KG: 30 workers for every architecture;
* KG_GEN: 20 workers, except RGCN: 5 workers;
* UMLS-NCI: 7 workers for every architecture.

The unit of parallelism is one (hidden, embedding, split seed, random seed)
run.  Each successful unit is written atomically.  Hence an interrupted job can
be resumed without rerunning completed seeds.  Once all groups have run, the
script rebuilds the normal TDGBench result JSON/CSV files and the benchmark
summary files.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from itertools import product
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_preprocessing.splits import PREGENERATED_SPLIT_SEEDS
from evaluate.evaluation_statistics import aggregate_split_and_randomness


GRAPHS = ("GT2KG_kg", "KG_GEN_kg", "UMLS_nci_kg")
MODELS = (
    "GCN", "GAT", "RGCN", "TransEGCN_conv", "RotatEGCN_conv",
    "TransEGCN_attn", "RotatEGCN_attn",
)
HIDDEN_CHANNELS = (64, 128, 256, 512)
EMBEDDINGS = (
    ("sentencebert", "sentence-transformers/all-MiniLM-L6-v2"),
    ("random384", "random_42"),
)

# The measured h=512 VRAM profiles justify these intentionally static caps.
WORKERS_BY_GRAPH = {"GT2KG_kg": 30, "KG_GEN_kg": 20, "UMLS_nci_kg": 7}
WORKERS_BY_GRAPH_MODEL = {("KG_GEN_kg", "RGCN"): 5}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/experiment_1_static")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument(
        "--max-processes", type=int, default=None,
        help="Optional hard cap applied in addition to the graph/model schedule.",
    )
    parser.add_argument("--graphs", nargs="+", choices=GRAPHS, default=GRAPHS)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--hidden-channels", nargs="+", type=int, default=HIDDEN_CHANNELS)
    parser.add_argument("--split-seeds", nargs="+", type=int, default=PREGENERATED_SPLIT_SEEDS)
    parser.add_argument("--random-seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def atomic_json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
    temporary.replace(path)


def embedding_directory(label: str, results_dir: Path) -> Path:
    return results_dir / label


def configuration_directory(task: Mapping[str, Any]) -> Path:
    return (
        Path(task["results_dir"])
        / task["embedding_label"]
        / task["graph"]
        / task["model"]
        / f"hidden_{task['hidden']}_out_{task['hidden']}"
    )


def run_path(task: Mapping[str, Any]) -> Path:
    return (
        configuration_directory(task)
        / "individual_runs"
        / f"split_{task['split_seed']}__random_{task['random_seed']}.json"
    )


def worker_initializer() -> None:
    """Prevent dozens of processes from each using all CPU threads."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def run_one(task: Mapping[str, Any]) -> str:
    """Evaluate and persist one seed pair in a fresh spawned process."""
    worker_initializer()
    from tdg_bench import TDGBench

    output = run_path(task)
    if bool(task.get("resume", True)) and output.exists():
        return str(output)

    suite = TDGBench(use_classifier=True).evaluate_models(
        kg_name=str(task["graph"]),
        model_names=[str(task["model"])],
        init_embd=str(task["embedding"]),
        split_seeds=[int(task["split_seed"])],
        random_seeds=[int(task["random_seed"])],
        hidden_channels=[int(task["hidden"])],
        random_embd_dim=384,
        epochs=int(task["epochs"]),
        patience=int(task["patience"]),
        verbose=False,
        save_results=False,
        save_models=False,
        save_predictions=False,
        resume=False,
        results_dir=str(configuration_directory(task)),
    )
    run_key = f"hidden_{task['hidden']}_out_{task['hidden']}"
    evaluation = suite["models"][str(task["model"])]["runs"][run_key]
    if evaluation["failed_runs"] or len(evaluation["per_run"]) != 1:
        raise RuntimeError(f"Unexpected single-run result: {evaluation['failed_runs']}")
    atomic_json_write(output, evaluation["per_run"][0])
    return str(output)


def worker_limit(graph: str, model: str, maximum: int | None) -> int:
    limit = WORKERS_BY_GRAPH_MODEL.get((graph, model), WORKERS_BY_GRAPH[graph])
    return min(limit, maximum) if maximum is not None else limit


def build_group_tasks(
    *, graph: str, model: str, hidden_channels: Iterable[int], embeddings: Iterable[tuple[str, str]],
    split_seeds: Iterable[int], random_seeds: Iterable[int], args: argparse.Namespace,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for (embedding_label, embedding), hidden, split_seed, random_seed in product(
        embeddings, hidden_channels, split_seeds, random_seeds
    ):
        tasks.append(
            {
                "graph": graph,
                "model": model,
                "hidden": hidden,
                "embedding_label": embedding_label,
                "embedding": embedding,
                "split_seed": split_seed,
                "random_seed": random_seed,
                "epochs": args.epochs,
                "patience": args.patience,
                "results_dir": args.results_dir,
            }
        )
    return tasks


def run_group(tasks: list[dict[str, Any]], workers: int, resume: bool) -> list[dict[str, Any]]:
    pending = [task for task in tasks if not (resume and run_path(task).exists())]
    if not pending:
        print("  All runs already completed.")
        return []
    print(f"  Launching {len(pending)} run(s) with {workers} worker(s).")
    failures: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
        initializer=worker_initializer,
    ) as executor:
        futures = {executor.submit(run_one, task): task for task in pending}
        for number, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            try:
                future.result()
                print(
                    f"  [{number}/{len(pending)}] completed: h={task['hidden']} | "
                    f"{task['embedding_label']} | split={task['split_seed']} | "
                    f"random={task['random_seed']}"
                )
            except Exception as error:
                failures.append(
                    {
                        "graph": task["graph"], "model": task["model"],
                        "hidden": task["hidden"], "embedding_label": task["embedding_label"],
                        "split_seed": task["split_seed"], "random_seed": task["random_seed"],
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                print(f"  [RUN ERROR] {failures[-1]}")
    return failures


def aggregate_configuration(
    *, graph: str, model: str, hidden: int, embedding_label: str, embedding: str,
    split_seeds: list[int], random_seeds: list[int], args: argparse.Namespace,
) -> dict[str, Any]:
    prototype = {
        "graph": graph, "model": model, "hidden": hidden,
        "embedding_label": embedding_label, "embedding": embedding,
        "split_seed": split_seeds[0], "random_seed": random_seeds[0],
        "epochs": args.epochs, "patience": args.patience, "results_dir": args.results_dir,
    }
    per_run: list[dict[str, Any]] = []
    failed_runs: list[dict[str, Any]] = []
    for split_seed, random_seed in product(split_seeds, random_seeds):
        task = {**prototype, "split_seed": split_seed, "random_seed": random_seed}
        path = run_path(task)
        if not path.exists():
            failed_runs.append({"split_seed": split_seed, "random_seed": random_seed, "error": "missing run result"})
            continue
        with path.open(encoding="utf-8") as handle:
            per_run.append(json.load(handle))
    per_run.sort(key=lambda item: (int(item["split_seed"]), int(item["random_seed"])))

    metrics_by_split = {
        split_seed: {
            "train_acc": [], "train_f1": [], "val_acc": [], "val_f1": [],
            "test_acc": [], "test_recall": [], "test_precision": [], "test_f1": [], "best_epoch": [],
        }
        for split_seed in split_seeds
    }
    for result in per_run:
        values = metrics_by_split[int(result["split_seed"])]
        history = result["history"]
        final = result["final_test"]
        values["train_acc"].append(history["train_acc"][-1])
        values["train_f1"].append(history["train_f1"][-1])
        values["val_acc"].append(history["val_acc"][-1])
        values["val_f1"].append(history["val_f1"][-1])
        values["test_acc"].append(final["accuracy"])
        values["test_recall"].append(final["recall"])
        values["test_precision"].append(final["precision"])
        values["test_f1"].append(final["f1"])
        values["best_epoch"].append(result["best_epoch"])
    aggregated = {}
    for metric in next(iter(metrics_by_split.values())):
        scores = {split: values[metric] for split, values in metrics_by_split.items() if values[metric]}
        if scores:
            aggregated[metric] = aggregate_split_and_randomness(scores)

    result = {
        "kg_name": graph, "graph_variant": graph, "options": [], "init_embd": embedding,
        "random_embd_dim": 384, "split_seeds": split_seeds, "random_seeds": random_seeds,
        "total_runs_requested": len(split_seeds) * len(random_seeds), "epochs": args.epochs,
        "patience": args.patience, "lr": 0.01, "weight_decay": 5e-4,
        "save_models": False, "save_predictions": False, "onto_incorporation": None,
        "onto_name": None, "lambda_align": None, "alignment_mode": None, "temperature": None,
        "per_run": per_run, "failed_runs": failed_runs, "runs_completed": len(per_run),
        "runs_failed": len(failed_runs), "aggregated": aggregated,
    }
    directory = configuration_directory(prototype)
    basename = f"{graph}__{model}__h{hidden}__out{hidden}"
    atomic_json_write(directory / f"results_{basename}.json", result)

    summary_rows = [
        {"metric": name, **stats}
        for name, stats in result["aggregated"].items()
    ]
    directory.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(summary_rows).to_csv(directory / f"summary_{basename}.csv", index=False)
    per_run_rows = [
        {
            "split_seed": item["split_seed"], "random_seed": item["random_seed"],
            "train_acc": item["history"]["train_acc"][-1], "train_f1": item["history"]["train_f1"][-1],
            "val_acc": item["history"]["val_acc"][-1], "val_f1": item["history"]["val_f1"][-1],
            "test_acc": item["final_test"]["accuracy"], "test_f1": item["final_test"]["f1"],
            "best_epoch": item["best_epoch"],
        }
        for item in per_run
    ]
    pd.DataFrame(per_run_rows).to_csv(directory / f"per_run_{basename}.csv", index=False)
    return result


def write_benchmark(
    *, label: str, embedding: str, results: dict[str, dict[str, dict[str, dict[str, Any]]]],
    graphs: list[str], models: list[str], split_seeds: list[int], random_seeds: list[int],
    hidden_channels: list[int], args: argparse.Namespace,
) -> None:
    from tdg_bench import TDGBench
    import pandas as pd

    root = embedding_directory(label, Path(args.results_dir))
    benchmark = {
        "graph_names": graphs, "model_names": models, "init_embd": embedding,
        "split_seeds": split_seeds, "random_seeds": random_seeds,
        "hidden_channels": hidden_channels, "out_channels": None, "random_embd_dim": 384,
        "save_models": False, "save_predictions": False, "graphs": results,
    }
    rows = TDGBench._benchmark_summary_rows(benchmark)
    pd.DataFrame(rows).to_csv(root / "benchmark_summary.csv", index=False)
    complete = [row for row in rows if row["runs_completed"] == row["runs_requested"]]
    best_rows = []
    for graph in graphs:
        choices = [row for row in complete if row["graph_name"] == graph]
        if choices:
            best_rows.append({
                **min(choices, key=lambda row: (-row["mean_best_validation_f1"], row["hidden_channels"], row["model_name"])),
                "selection_metric": "mean_best_validation_f1",
            })
    pd.DataFrame(best_rows).to_csv(root / "best_by_graph.csv", index=False)
    benchmark["best_by_graph"] = best_rows
    benchmark["output_files"] = {
        "json": str(root / "benchmark_results.json"),
        "summary_csv": str(root / "benchmark_summary.csv"),
        "best_by_graph_csv": str(root / "best_by_graph.csv"),
    }
    atomic_json_write(root / "benchmark_results.json", benchmark)


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("epochs and patience must be >= 1")
    if args.max_processes is not None and args.max_processes < 1:
        raise ValueError("max-processes must be >= 1")
    graphs, models = list(args.graphs), list(args.models)
    hidden_channels = [512] if args.smoke_test else list(args.hidden_channels)
    split_seeds = [42] if args.smoke_test else list(args.split_seeds)
    random_seeds = [1] if args.smoke_test else list(args.random_seeds)
    if args.smoke_test:
        args.epochs = 2

    atomic_json_write(
        Path(args.results_dir) / "scheduler_manifest.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "graphs": graphs, "models": models, "hidden_channels": hidden_channels,
            "split_seeds": split_seeds, "random_seeds": random_seeds,
            "epochs": args.epochs, "patience": args.patience,
            "worker_schedule": {
                f"{graph}__{model}": worker_limit(graph, model, args.max_processes)
                for graph in graphs for model in models
            },
        },
    )
    all_failures: list[dict[str, Any]] = []
    for graph in graphs:
        for model in models:
            cap = worker_limit(graph, model, args.max_processes)
            tasks = build_group_tasks(
                graph=graph, model=model, hidden_channels=hidden_channels, embeddings=EMBEDDINGS,
                split_seeds=split_seeds, random_seeds=random_seeds, args=args,
            )
            for task in tasks:
                task["resume"] = not args.no_resume
            print(f"\n{'=' * 72}\nGROUP: {graph} | {model} | workers={cap}\n{'=' * 72}")
            if args.dry_run:
                print(f"  Would schedule {len(tasks)} runs.")
            else:
                all_failures.extend(run_group(tasks, cap, resume=not args.no_resume))

    if not args.dry_run:
        for label, embedding in EMBEDDINGS:
            nested: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
            for graph in graphs:
                nested[graph] = {"kg_name": graph, "graph_variant": graph, "options": [], "models": {}}
                for model in models:
                    nested[graph]["models"][model] = {"runs": {}}
                    for hidden in hidden_channels:
                        key = f"hidden_{hidden}_out_{hidden}"
                        nested[graph]["models"][model]["runs"][key] = aggregate_configuration(
                            graph=graph, model=model, hidden=hidden, embedding_label=label, embedding=embedding,
                            split_seeds=split_seeds, random_seeds=random_seeds, args=args,
                        )
            write_benchmark(
                label=label, embedding=embedding, results=nested, graphs=graphs, models=models,
                split_seeds=split_seeds, random_seeds=random_seeds,
                hidden_channels=hidden_channels, args=args,
            )
    failure_path = Path(args.results_dir) / "scheduler_failures.json"
    atomic_json_write(failure_path, {"failed_runs": all_failures})
    print(f"\nFinished with {len(all_failures)} run error(s). Failures: {failure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
