"""Experiment 4: text-only and random-feature MLP baselines, without a graph.

``GT2KG_kg`` is used only to resolve the shared TDG labelled-node protocol;
no graph JSON is loaded.  Random features have exactly 384 dimensions.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_preprocessing.splits import PREGENERATED_SPLIT_SEEDS
from tdg_bench import TDGBench


FEATURE_SOURCES = (
    ("sentencebert", "sentence-transformers/all-MiniLM-L6-v2"),
    ("random384", "random_384"),
)
HIDDEN_CHANNELS = (64, 128, 256, 512)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/experiment_4_no_graph")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run both feature sources with h=512, split=42, random seed=1 and 2 epochs.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=1,
        help="Maximum number of independent configurations run in parallel (default: 1).",
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def run_smoke_task(task: tuple[str, str, int], output_root: str) -> None:
    label, source, hidden = task
    TDGBench(use_classifier=False).evaluate_no_graph_baseline(
        kg_name="GT2KG_kg", init_embds=[source], split_seeds=[42], random_seeds=[1],
        hidden_channels=[hidden], random_embedding_dim=384, epochs=2, patience=100,
        verbose=False, save_models=False, save_predictions=False, resume=True,
        results_dir=str(Path(output_root) / label),
    )


def run_full_task(
    task: tuple[str, str, int], output_root: str, epochs: int, patience: int,
    verbose: bool, resume: bool,
) -> None:
    """Run one complete no-graph configuration in a child process."""
    label, source, hidden = task
    TDGBench(use_classifier=False).evaluate_no_graph_baseline(
        kg_name="GT2KG_kg", init_embds=[source],
        split_seeds=PREGENERATED_SPLIT_SEEDS, random_seeds=[1, 2, 3, 4, 5],
        hidden_channels=[hidden], random_embedding_dim=384, epochs=epochs,
        patience=patience, verbose=verbose, save_models=False,
        save_predictions=False, resume=resume,
        results_dir=str(Path(output_root) / label),
    )


def main() -> int:
    args = parse_args()
    hidden_channels = (512,) if args.smoke_test else HIDDEN_CHANNELS
    split_seeds = [42] if args.smoke_test else PREGENERATED_SPLIT_SEEDS
    random_seeds = [1] if args.smoke_test else [1, 2, 3, 4, 5]
    epochs = 2 if args.smoke_test else args.epochs
    output_root = Path(args.results_dir) / "smoke_test" if args.smoke_test else Path(args.results_dir)
    tasks = [(label, source, hidden) for label, source in FEATURE_SOURCES for hidden in hidden_channels]
    print(f"{len(tasks)} configurations.")
    if args.num_workers < 1:
        raise ValueError("num-workers must be >= 1")
    if args.smoke_test and args.num_workers > 1:
        failures = 0
        with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=get_context("spawn")) as executor:
            futures = {
                executor.submit(run_smoke_task, task, str(output_root)): task
                for task in tasks
            }
            for number, future in enumerate(as_completed(futures), 1):
                task = futures[future]
                try:
                    future.result()
                    print(f"[{number}/{len(tasks)}] completed: {task[0]} | h={task[2]}")
                except Exception as error:
                    failures += 1
                    print(f"[CONFIGURATION ERROR] {task}: {type(error).__name__}: {error}")
        print(f"Smoke test finished: {len(tasks) - failures} completed, {failures} errors.")
        return 0
    if args.num_workers > 1:
        failures = 0
        with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=get_context("spawn")) as executor:
            futures = {
                executor.submit(
                    run_full_task, task, str(output_root), epochs, args.patience,
                    args.verbose, not args.no_resume,
                ): task
                for task in tasks
            }
            for number, future in enumerate(as_completed(futures), 1):
                task = futures[future]
                try:
                    future.result()
                    print(f"[{number}/{len(tasks)}] completed: {task[0]} | h={task[2]}")
                except Exception as error:
                    failures += 1
                    print(f"[CONFIGURATION ERROR] {task}: {type(error).__name__}: {error}")
        print(f"Finished: {len(tasks) - failures} configurations completed, {failures} configuration errors.")
        return 0
    bench = TDGBench(use_classifier=False)
    failures = 0
    for number, (label, source, hidden) in enumerate(tasks, 1):
        print(f"\n[{number}/{len(tasks)}] no-graph | {label} | h={hidden}")
        try:
            bench.evaluate_no_graph_baseline(
                kg_name="GT2KG_kg", init_embds=[source],
                split_seeds=split_seeds, random_seeds=random_seeds,
                hidden_channels=[hidden], random_embedding_dim=384,
                epochs=epochs, patience=args.patience, verbose=args.verbose,
                save_models=False, save_predictions=False, resume=not args.no_resume,
                results_dir=str(output_root / label),
            )
        except Exception as error:
            failures += 1
            print(f"[CONFIGURATION ERROR] {type(error).__name__}: {error}")
            traceback.print_exc()
    print(f"Finished: {len(tasks) - failures} configurations completed, {failures} configuration errors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
