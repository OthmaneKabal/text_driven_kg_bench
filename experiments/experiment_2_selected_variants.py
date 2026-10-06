"""Evaluate the primary preprocessing variants with validation-selected widths.

For every architecture, its hidden width is fixed from the corresponding main
graph (GT2KG, KG-GEN, or UMLS-NCI).  Selection was done on complete
Sentence-BERT main-graph runs using mean validation Macro-F1.  The variants
are evaluated with Sentence-BERT only; no width is selected using a variant's
test scores.

The runner stores each (variant, architecture, split, training-seed) result
atomically and resumes missing runs on a later invocation.
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


SENTENCE_BERT = ("sentencebert", "sentence-transformers/all-MiniLM-L6-v2")

# Only the preprocessing variants presented in the paper.
VARIANTS_BY_PARENT = {
    "GT2KG_kg": (
        "GT2KG_kg_raw",
        "GT2KG_kg_no_freq_filter",
        "GT2KG_kg_no_smnt",
    ),
    "KG_GEN_kg": (
        "KG_GEN_kg_raw",
        "KG_GEN_kg_no_freq_filter",
        "KG_GEN_kg_no_smnt",
    ),
    "UMLS_nci_kg": (
        "UMLS_nci_kg_raw",
        "UMLS_nci_kg_no_freq_filter",
        "UMLS_nci_kg_no_smnt",
        "UMLS_nci_kg_with_inverse",
    ),
}
PARENT_BY_VARIANT = {
    variant: parent
    for parent, variants in VARIANTS_BY_PARENT.items()
    for variant in variants
}
VARIANTS = tuple(PARENT_BY_VARIANT)

# Selected from complete Sentence-BERT main-graph configurations by mean
# validation Macro-F1, separately for each (parent graph, architecture).
SELECTED_HIDDEN_BY_PARENT = {
    "GT2KG_kg": {
        "GCN": 64, "GAT": 512, "RGCN": 64,
        "TransEGCN_conv": 64, "RotatEGCN_conv": 64,
        "TransEGCN_attn": 64, "RotatEGCN_attn": 64,
    },
    "KG_GEN_kg": {
        "GCN": 256, "GAT": 64, "RGCN": 64,
        "TransEGCN_conv": 64, "RotatEGCN_conv": 64,
        "TransEGCN_attn": 64, "RotatEGCN_attn": 64,
    },
    "UMLS_nci_kg": {
        "GCN": 64, "GAT": 128, "RGCN": 64,
        "TransEGCN_conv": 256, "RotatEGCN_conv": 64,
        "TransEGCN_attn": 64, "RotatEGCN_attn": 64,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/experiment_2_selected_variants")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument(
        "--max-processes", type=int, default=None,
        help="Optional cap in addition to the conservative per-parent worker schedule.",
    )
    parser.add_argument("--graphs", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--split-seeds", nargs="+", type=int, default=PREGENERATED_SPLIT_SEEDS)
    parser.add_argument("--random-seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def worker_limit(parent: str, model: str, maximum: int | None) -> int:
    """Use measured Experiment-1 caps, with a safer UMLS cap after pool failures."""
    if parent == "GT2KG_kg":
        limit = 30
    elif parent == "KG_GEN_kg":
        limit = 5 if model == "RGCN" else 20
    else:
        limit = 3
    return min(limit, maximum) if maximum is not None else limit


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("epochs and patience must be >= 1")
    if args.max_processes is not None and args.max_processes < 1:
        raise ValueError("max-processes must be >= 1")

    variants, models = list(args.graphs), list(args.models)
    split_seeds = [42] if args.smoke_test else list(args.split_seeds)
    random_seeds = [1] if args.smoke_test else list(args.random_seeds)
    if args.smoke_test:
        args.epochs = 2

    atomic_json_write(
        Path(args.results_dir) / "selected_hyperparameters.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "embedding": SENTENCE_BERT[1],
            "selection_source": (
                "Complete Sentence-BERT main-graph runs; maximum mean validation Macro-F1 "
                "within each parent graph and architecture."
            ),
            "variants": variants,
            "parent_by_variant": {variant: PARENT_BY_VARIANT[variant] for variant in variants},
            "selected_hidden_by_parent": SELECTED_HIDDEN_BY_PARENT,
            "split_seeds": split_seeds,
            "random_seeds": random_seeds,
            "epochs": args.epochs,
            "patience": args.patience,
            "worker_schedule": {
                f"{variant}__{model}": worker_limit(PARENT_BY_VARIANT[variant], model, args.max_processes)
                for variant in variants for model in models
            },
        },
    )

    failures: list[dict[str, object]] = []
    for variant in variants:
        parent = PARENT_BY_VARIANT[variant]
        for model in models:
            hidden = SELECTED_HIDDEN_BY_PARENT[parent][model]
            workers = worker_limit(parent, model, args.max_processes)
            tasks = build_group_tasks(
                graph=variant,
                model=model,
                hidden_channels=[hidden],
                embeddings=[SENTENCE_BERT],
                split_seeds=split_seeds,
                random_seeds=random_seeds,
                args=args,
            )
            for task in tasks:
                task["resume"] = not args.no_resume
            print(f"\n{'=' * 72}\nGROUP: {variant} | {model} | h={hidden} | workers={workers}\n{'=' * 72}")
            if args.dry_run:
                print(f"  Would schedule {len(tasks)} run(s).")
            else:
                failures.extend(run_group(tasks, workers, resume=not args.no_resume))

    if not args.dry_run:
        label, embedding = SENTENCE_BERT
        results: dict[str, dict[str, object]] = {}
        for variant in variants:
            parent = PARENT_BY_VARIANT[variant]
            graph_results: dict[str, object] = {
                "kg_name": variant,
                "graph_variant": variant,
                "options": [],
                "models": {},
            }
            for model in models:
                hidden = SELECTED_HIDDEN_BY_PARENT[parent][model]
                key = f"hidden_{hidden}_out_{hidden}"
                graph_results["models"][model] = {
                    "runs": {
                        key: aggregate_configuration(
                            graph=variant,
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
            results[variant] = graph_results
        write_benchmark(
            label=label,
            embedding=embedding,
            results=results,
            graphs=variants,
            models=models,
            split_seeds=split_seeds,
            random_seeds=random_seeds,
            hidden_channels=sorted({
                SELECTED_HIDDEN_BY_PARENT[PARENT_BY_VARIANT[variant]][model]
                for variant in variants for model in models
            }),
            args=args,
        )

    atomic_json_write(Path(args.results_dir) / "scheduler_failures.json", {"failed_runs": failures})
    print(f"\nFinished with {len(failures)} run error(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
