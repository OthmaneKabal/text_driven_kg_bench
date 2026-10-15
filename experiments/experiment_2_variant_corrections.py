"""Re-run incomplete or invalid Experiment-2 preprocessing-variant results.

This runner deliberately writes to a new results directory.  It never reads
the original Experiment-2 individual-run artefacts, so copied or otherwise
invalid artefacts cannot be resumed accidentally.  Re-running this command
resumes only the verified runs created in its own output directory.

The correction plan contains:
* groups that were incomplete; and
* groups whose stored artefacts were identical to a different graph variant.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
from experiment_2_selected_variants import (
    PARENT_BY_VARIANT,
    SELECTED_HIDDEN_BY_PARENT,
    SENTENCE_BERT,
    worker_limit,
)


# Every listed group is rerun from scratch in this new results directory.
# This avoids mixing a complete but invalid artefact with newly generated runs.
CORRECTION_MODELS: dict[str, tuple[str, ...]] = {
    # Incomplete: 23/50 completed.
    "GT2KG_kg_no_smnt": ("GCN",),
    # Raw and no-frequency results shared identical individual artefacts.
    "KG_GEN_kg_raw": tuple(MODELS),
    "KG_GEN_kg_no_freq_filter": tuple(MODELS),
    # Incomplete: 38/50 completed.
    "KG_GEN_kg_no_smnt": ("GCN",),
    # Missing: 0/50 completed for every architecture.
    "UMLS_nci_kg_raw": tuple(MODELS),
    # Individual artefacts were identical to UMLS no-frequency results.
    "UMLS_nci_kg_with_inverse": tuple(MODELS),
}
CORRECTION_GRAPHS = tuple(CORRECTION_MODELS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default="results/experiment_2_variant_corrections",
        help="Fresh, correction-only output directory.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument(
        "--max-processes",
        type=int,
        default=None,
        help="Optional cap in addition to the conservative per-parent worker schedule.",
    )
    parser.add_argument("--graphs", nargs="+", choices=CORRECTION_GRAPHS, default=CORRECTION_GRAPHS)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--split-seeds", nargs="+", type=int, default=PREGENERATED_SPLIT_SEEDS)
    parser.add_argument("--random-seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def selected_models(graph: str, requested: list[str]) -> tuple[str, ...]:
    """Keep only models that need correction for ``graph``."""
    needed = CORRECTION_MODELS[graph]
    return tuple(model for model in requested if model in needed)


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("epochs and patience must be >= 1")
    if args.max_processes is not None and args.max_processes < 1:
        raise ValueError("max-processes must be >= 1")

    graphs = list(args.graphs)
    requested_models = list(args.models)
    split_seeds = [42] if args.smoke_test else list(args.split_seeds)
    random_seeds = [1] if args.smoke_test else list(args.random_seeds)
    if args.smoke_test:
        args.epochs = 2

    plan = {
        graph: list(selected_models(graph, requested_models))
        for graph in graphs
    }
    empty_graphs = [graph for graph, models in plan.items() if not models]
    if empty_graphs:
        raise ValueError(
            "No selected model requires correction for: " + ", ".join(empty_graphs)
        )

    total_runs = sum(
        len(models) * len(split_seeds) * len(random_seeds)
        for models in plan.values()
    )
    atomic_json_write(
        Path(args.results_dir) / "correction_plan.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "Fresh rerun of incomplete or invalid Experiment-2 variant results.",
            "source_results_not_used": "results/experiment_2_selected_variants",
            "graphs_and_models": plan,
            "split_seeds": split_seeds,
            "random_seeds": random_seeds,
            "epochs": args.epochs,
            "patience": args.patience,
            "runs_requested": total_runs,
            "embedding": SENTENCE_BERT[1],
            "reasons": {
                "GT2KG_kg_no_smnt/GCN": "incomplete original run",
                "KG_GEN_kg_raw/*": "invalid: artefacts duplicated with no-frequency variant",
                "KG_GEN_kg_no_freq_filter/*": "invalid: artefacts duplicated with raw variant",
                "KG_GEN_kg_no_smnt/GCN": "incomplete original run",
                "UMLS_nci_kg_raw/*": "missing original run",
                "UMLS_nci_kg_with_inverse/*": "invalid: artefacts duplicated with no-frequency variant",
            },
        },
    )

    failures: list[dict[str, object]] = []
    for graph in graphs:
        parent = PARENT_BY_VARIANT[graph]
        for model in plan[graph]:
            hidden = SELECTED_HIDDEN_BY_PARENT[parent][model]
            workers = worker_limit(parent, model, args.max_processes)
            tasks = build_group_tasks(
                graph=graph,
                model=model,
                hidden_channels=[hidden],
                embeddings=[SENTENCE_BERT],
                split_seeds=split_seeds,
                random_seeds=random_seeds,
                args=args,
            )
            for task in tasks:
                task["resume"] = not args.no_resume
            print(
                f"\n{'=' * 72}\nCORRECTION: {graph} | {model} | "
                f"h={hidden} | workers={workers}\n{'=' * 72}"
            )
            if args.dry_run:
                print(f"  Would schedule {len(tasks)} fresh run(s).")
            else:
                failures.extend(run_group(tasks, workers, resume=not args.no_resume))

    if not args.dry_run:
        label, embedding = SENTENCE_BERT
        results: dict[str, dict[str, Any]] = {}
        models_for_summary: list[str] = []
        for graph in graphs:
            parent = PARENT_BY_VARIANT[graph]
            graph_models: dict[str, Any] = {}
            for model in plan[graph]:
                hidden = SELECTED_HIDDEN_BY_PARENT[parent][model]
                key = f"hidden_{hidden}_out_{hidden}"
                graph_models[model] = {
                    "runs": {
                        key: aggregate_configuration(
                            graph=graph,
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
                if model not in models_for_summary:
                    models_for_summary.append(model)
            results[graph] = {
                "kg_name": graph,
                "graph_variant": graph,
                "options": [],
                "models": graph_models,
            }
        write_benchmark(
            label=label,
            embedding=embedding,
            results=results,
            graphs=graphs,
            models=models_for_summary,
            split_seeds=split_seeds,
            random_seeds=random_seeds,
            hidden_channels=sorted({
                SELECTED_HIDDEN_BY_PARENT[PARENT_BY_VARIANT[graph]][model]
                for graph, models in plan.items() for model in models
            }),
            args=args,
        )

    atomic_json_write(
        Path(args.results_dir) / "scheduler_failures.json",
        {"failed_runs": failures},
    )
    print(f"\nFinished with {len(failures)} run error(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
