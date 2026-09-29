"""Experiment 3: three scale-matched and three sparsity-matched clean controls per TDG.

Uses Sentence-BERT only. The 12 persisted UMLS-NCI control graphs are run on
one node and individual split/random runs resume.
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


TARGETS = ("GT2KG_kg", "KG_GEN_kg")
GRAPHS = tuple(
    f"UMLS_nci_kg_scale_matched_{target}_top_{rank}{suffix}"
    for target in TARGETS
    for suffix in ("", "_sparsity_matched")
    for rank in (1, 2, 3)
)
MODELS = (
    "GCN", "GAT", "RGCN", "TransEGCN_conv", "RotatEGCN_conv",
    "TransEGCN_attn", "RotatEGCN_attn",
)
HIDDEN_CHANNELS = (64, 128, 256, 512)
EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/experiment_3_scale_sparsity_controls")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run every graph/model once with h=512, split=42, random seed=1 and 2 epochs.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=1,
        help="Maximum number of independent configurations run in parallel (default: 1).",
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def run_smoke_task(task: tuple[str, str, int], output_root: str) -> None:
    graph, model, hidden = task
    TDGBench(use_classifier=True).evaluate_models(
        kg_name=graph, model_names=[model], init_embd=EMBEDDING,
        split_seeds=[42], random_seeds=[1], hidden_channels=[hidden],
        epochs=2, patience=100, verbose=False, save_models=False,
        save_predictions=False, resume=True, results_dir=output_root,
    )


def run_full_task(
    task: tuple[str, str, int], output_root: str, epochs: int, patience: int,
    verbose: bool, resume: bool,
) -> None:
    """Run one complete configuration in a child process."""
    graph, model, hidden = task
    TDGBench(use_classifier=True).evaluate_models(
        kg_name=graph, model_names=[model], init_embd=EMBEDDING,
        split_seeds=PREGENERATED_SPLIT_SEEDS, random_seeds=[1, 2, 3, 4, 5],
        hidden_channels=[hidden], epochs=epochs, patience=patience, verbose=verbose,
        save_models=False, save_predictions=False, resume=resume,
        results_dir=output_root,
    )


def main() -> int:
    args = parse_args()
    hidden_channels = (512,) if args.smoke_test else HIDDEN_CHANNELS
    split_seeds = [42] if args.smoke_test else PREGENERATED_SPLIT_SEEDS
    random_seeds = [1] if args.smoke_test else [1, 2, 3, 4, 5]
    epochs = 2 if args.smoke_test else args.epochs
    output_root = Path(args.results_dir) / "smoke_test" if args.smoke_test else Path(args.results_dir)
    tasks = [(graph, model, hidden) for graph in GRAPHS for model in MODELS for hidden in hidden_channels]
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
                    print(f"[{number}/{len(tasks)}] completed: {task[0]} | {task[1]} | h={task[2]}")
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
                    print(f"[{number}/{len(tasks)}] completed: {task[0]} | {task[1]} | h={task[2]}")
                except Exception as error:
                    failures += 1
                    print(f"[CONFIGURATION ERROR] {task}: {type(error).__name__}: {error}")
        print(f"Finished: {len(tasks) - failures} configurations completed, {failures} configuration errors.")
        return 0
    bench = TDGBench(use_classifier=True)
    failures = 0
    for number, (graph, model, hidden) in enumerate(tasks, 1):
        print(f"\n[{number}/{len(tasks)}] {graph} | {model} | h={hidden}")
        try:
            bench.evaluate_benchmark(
                graph_names=[graph], model_names=[model], init_embd=EMBEDDING,
                split_seeds=split_seeds, random_seeds=random_seeds,
                hidden_channels=[hidden], epochs=epochs, patience=args.patience,
                verbose=args.verbose, save_models=False, save_predictions=False,
                resume=not args.no_resume, results_dir=str(output_root),
            )
        except Exception as error:
            failures += 1
            print(f"[CONFIGURATION ERROR] {type(error).__name__}: {error}")
            traceback.print_exc()
    print(f"Finished: {len(tasks) - failures} configurations completed, {failures} configuration errors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
