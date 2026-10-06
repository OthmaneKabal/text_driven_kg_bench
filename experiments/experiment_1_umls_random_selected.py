"""Evaluate Random-384 on UMLS-NCI using validation-selected architecture sizes.

The hidden width of each architecture was selected beforehand from the complete
Sentence-BERT UMLS-NCI runs, using mean validation Macro-F1 only.  This runner
does *not* tune widths on the Random-384 test results.  It evaluates the seven
fixed configurations over the usual 10 data splits and 5 training seeds.

Successful seed-pair runs are stored atomically, so rerunning this command
only evaluates missing runs unless ``--no-resume`` is supplied.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPERIMENTS_DIRECTORY = ROOT / "experiments"
if str(EXPERIMENTS_DIRECTORY) not in sys.path:
    # The repository also has an experiments.py module at its root.  Importing
    # through this explicit directory avoids that module masking this folder.
    sys.path.insert(0, str(EXPERIMENTS_DIRECTORY))

from data_preprocessing.splits import PREGENERATED_SPLIT_SEEDS
from experiment_1_static_scheduler import (
    MODELS,
    aggregate_configuration,
    atomic_json_write,
    build_group_tasks,
    run_group,
    write_benchmark,
)


GRAPH_NAME = "UMLS_nci_kg"
RANDOM_EMBEDDING = ("random384", "random_42")

# Selected on mean validation Macro-F1 from complete Sentence-BERT UMLS runs.
SELECTED_HIDDEN_BY_MODEL = {
    "GCN": 64,
    "GAT": 128,
    "RGCN": 64,
    "TransEGCN_conv": 256,
    "RotatEGCN_conv": 64,
    "TransEGCN_attn": 64,
    "RotatEGCN_attn": 64,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default="results/experiment_1_umls_random_selected",
        help="Separate output directory; existing Experiment 1 files are not overwritten.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument(
        "--max-processes",
        type=int,
        default=3,
        help="Concurrent UMLS runs (default: 3, conservatively below the failed seven-worker run).",
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--split-seeds", nargs="+", type=int, default=PREGENERATED_SPLIT_SEEDS)
    parser.add_argument("--random-seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("epochs and patience must be >= 1")
    if args.max_processes < 1:
        raise ValueError("max-processes must be >= 1")

    models = list(args.models)
    split_seeds = [42] if args.smoke_test else list(args.split_seeds)
    random_seeds = [1] if args.smoke_test else list(args.random_seeds)
    if args.smoke_test:
        args.epochs = 2

    atomic_json_write(
        Path(args.results_dir) / "selected_hyperparameters.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "graph": GRAPH_NAME,
            "embedding": RANDOM_EMBEDDING[1],
            "selection_source": (
                "Complete Sentence-BERT UMLS-NCI runs; maximum mean validation Macro-F1 "
                "within each architecture."
            ),
            "selected_hidden_by_model": {model: SELECTED_HIDDEN_BY_MODEL[model] for model in models},
            "split_seeds": split_seeds,
            "random_seeds": random_seeds,
            "epochs": args.epochs,
            "patience": args.patience,
            "max_processes": args.max_processes,
        },
    )

    failures: list[dict[str, object]] = []
    for model in models:
        hidden = SELECTED_HIDDEN_BY_MODEL[model]
        tasks = build_group_tasks(
            graph=GRAPH_NAME,
            model=model,
            hidden_channels=[hidden],
            embeddings=[RANDOM_EMBEDDING],
            split_seeds=split_seeds,
            random_seeds=random_seeds,
            args=args,
        )
        for task in tasks:
            task["resume"] = not args.no_resume
        print(f"\n{'=' * 72}\nGROUP: {GRAPH_NAME} | {model} | h={hidden} | workers={args.max_processes}\n{'=' * 72}")
        if args.dry_run:
            print(f"  Would schedule {len(tasks)} run(s).")
        else:
            failures.extend(run_group(tasks, args.max_processes, resume=not args.no_resume))

    if not args.dry_run:
        label, embedding = RANDOM_EMBEDDING
        graph_results: dict[str, object] = {
            "kg_name": GRAPH_NAME,
            "graph_variant": GRAPH_NAME,
            "options": [],
            "models": {},
        }
        for model in models:
            hidden = SELECTED_HIDDEN_BY_MODEL[model]
            key = f"hidden_{hidden}_out_{hidden}"
            graph_results["models"][model] = {
                "runs": {
                    key: aggregate_configuration(
                        graph=GRAPH_NAME,
                        model=model,
                        hidden=hidden,
                        embedding_label=label,
                        embedding=embedding,
                        split_seeds=split_seeds,
                        random_seeds=random_seeds,
                        args=args,
                    )
                }
            }
        write_benchmark(
            label=label,
            embedding=embedding,
            results={GRAPH_NAME: graph_results},
            graphs=[GRAPH_NAME],
            models=models,
            split_seeds=split_seeds,
            random_seeds=random_seeds,
            hidden_channels=sorted(set(SELECTED_HIDDEN_BY_MODEL[model] for model in models)),
            args=args,
        )

    atomic_json_write(Path(args.results_dir) / "scheduler_failures.json", {"failed_runs": failures})
    print(f"\nFinished with {len(failures)} run error(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
