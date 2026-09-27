"""Evaluate RotatEGCN-attn on GT2KG and its two UMLS scale-matched controls.

The experiment uses the ten pre-generated split seeds and one or more
independent training-randomness seeds.  The three graphs are persisted files,
so graph-variant options are deliberately not accepted here.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from data_preprocessing.splits import PREGENERATED_SPLIT_SEEDS
from tdg_bench import TDGBench


GRAPHS = (
    "GT2KG_kg",
    "UMLS_nci_kg_scale_matched_GT2KG_kg_top_1",
    "UMLS_nci_kg_scale_matched_GT2KG_kg_top_1_sparsity_matched",
)
MODEL = "RotatEGCN_attn"


def parse_seeds(value: str) -> list[int]:
    seeds = [int(seed.strip()) for seed in value.split(",") if seed.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("At least one random seed is required.")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--random-seeds",
        type=parse_seeds,
        default=[1],
        help="Comma-separated training-randomness seeds (default: 1).",
    )
    parser.add_argument(
        "--init-embd",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Initial node embedding model.",
    )
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--out-channels", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--splits-dir", default="datasets/split")
    parser.add_argument(
        "--results-dir",
        default="results/gt2kg_scale_matched_top1_rotate_10splits_1seed",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark = TDGBench(use_classifier=True)
    benchmark.evaluate_benchmark(
        graph_names=GRAPHS,
        model_names=[MODEL],
        init_embd=args.init_embd,
        split_seeds=PREGENERATED_SPLIT_SEEDS,
        random_seeds=args.random_seeds,
        splits_dir=args.splits_dir,
        hidden_channels=[args.hidden_channels],
        out_channels=args.out_channels,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        verbose=True,
        save_results=True,
        results_dir=args.results_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
