"""Evaluate RotatEGCN-attn on GT2KG with the supplied benchmark splits.

Run from the repository root:
    python tests/test_evaluate_gt2kg_rotategcn_attn.py

By default this evaluates the processed ``GT2KG_kg`` graph on the ten
pre-generated splits, selects the checkpoint with
validation macro-F1, and reports final test metrics for every split.
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

from build_models import build_encoder, get_default_model_kwargs
from data_preprocessing.graph_variants import graph_variant_name
from data_preprocessing.splits import PREGENERATED_SPLIT_SEEDS, generate_and_save_splits
from tdg_bench import TDGBench


MODEL_NAME = "RotatEGCN_attn"
KG_NAME = "GT2KG_kg"
EXTRA_SPLIT_SEED_START = 3000


def parse_seeds(value: str) -> list[int]:
    seeds = [int(seed.strip()) for seed in value.split(",") if seed.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required.")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-splits",
        type=int,
        default=None,
        help="Number of splits. Defaults to all 10 pre-generated benchmark splits.",
    )
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=None,
        help="Explicit comma-separated split seeds; required when --num-splits is below 10.",
    )
    parser.add_argument(
        "--confirm-generate-more",
        action="store_true",
        help="Confirm generation of splits beyond the 10 pre-generated benchmark splits.",
    )
    parser.add_argument(
        "--random-seeds",
        type=parse_seeds,
        default=[11],
        help="Independent training-randomness seeds (default: 11).",
    )
    parser.add_argument("--splits-dir", default="datasets/split")
    parser.add_argument(
        "--init-embd",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model, or random_<seed> for deterministic random features.",
    )
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument(
        "--out-channels",
        type=int,
        default=None,
        help="GNN output width; defaults to --hidden-channels.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument(
        "--options",
        nargs="*",
        default=(),
        help="GT2KG options: raw, no_smnt, no_freq_filter.",
    )
    parser.add_argument(
        "--save-variant",
        action="store_true",
        help="Save the selected graph variant if it does not already exist.",
    )
    parser.add_argument("--results-dir", default="results/test_GT2KG_RotatEGCN_attn")
    return parser.parse_args()


def resolve_seeds(args: argparse.Namespace) -> list[int]:
    """Resolve a selected subset, the standard ten splits, or confirmed extras."""
    requested = args.num_splits
    explicit_seeds = args.seeds
    if requested is None:
        requested = len(explicit_seeds) if explicit_seeds is not None else len(PREGENERATED_SPLIT_SEEDS)
    if requested < 1:
        raise ValueError("--num-splits must be >= 1")
    if explicit_seeds is not None:
        if len(explicit_seeds) != requested:
            raise ValueError("--num-splits must equal the number of comma-separated --seeds")
        if requested > len(PREGENERATED_SPLIT_SEEDS) and not args.confirm_generate_more:
            raise ValueError(
                "Only 10 benchmark splits are pre-generated. "
                "Are you sure you want to generate more? Re-run with --confirm-generate-more."
            )
        return explicit_seeds
    if requested < len(PREGENERATED_SPLIT_SEEDS):
        raise ValueError(
            "For fewer than 10 splits, specify exactly which ones with --seeds. "
            "Example: --num-splits 3 --seeds 42,123,2024"
        )
    if requested == len(PREGENERATED_SPLIT_SEEDS):
        return list(PREGENERATED_SPLIT_SEEDS)
    if not args.confirm_generate_more:
        raise ValueError(
            "Only 10 benchmark splits are pre-generated. "
            "Are you sure you want to generate more? Re-run with --confirm-generate-more."
        )

    extra_count = requested - len(PREGENERATED_SPLIT_SEEDS)
    extra_seeds = list(range(EXTRA_SPLIT_SEED_START, EXTRA_SPLIT_SEED_START + extra_count))
    generate_and_save_splits(
        xlsx_path="datasets/common_nodes.xlsx",
        save_dir=args.splits_dir,
        seeds=extra_seeds,
        train_ratio=0.10,
        val_ratio=0.10,
        test_ratio=0.80,
        stratify=True,
    )
    return [*PREGENERATED_SPLIT_SEEDS, *extra_seeds]


def main() -> int:
    args = parse_args()
    options = tuple(args.options)
    out_channels = args.out_channels or args.hidden_channels
    seeds = resolve_seeds(args)
    first_split = Path(args.splits_dir) / f"split_{seeds[0]}.json"
    if not first_split.exists():
        raise FileNotFoundError(f"First requested split does not exist: {first_split}")

    tdg = TDGBench(use_classifier=True)
    data, _, _, _, graph_preparation = tdg.get_data(
        kg_name=KG_NAME,
        init_embd=args.init_embd,
        split_path=str(first_split),
        options=options,
        save_variant=args.save_variant,
    )
    in_channels = int(data.x.shape[1])
    num_relations = len(graph_preparation.predicate_to_id)
    model_kwargs = get_default_model_kwargs(MODEL_NAME)

    def model_factory():
        return build_encoder(
            model_name=MODEL_NAME,
            in_channels=in_channels,
            hidden_channels=args.hidden_channels,
            out_channels=out_channels,
            num_relations=num_relations,
            **model_kwargs,
        )

    variant_name = graph_variant_name(KG_NAME, options)
    run_id = f"{variant_name}__{MODEL_NAME}__h{args.hidden_channels}__out{out_channels}"
    print(
        f"\nEvaluating {MODEL_NAME} on {variant_name}: "
        f"in={in_channels}, hidden={args.hidden_channels}, out={out_channels}, "
        f"relations={num_relations}, split_seeds={seeds}, random_seeds={args.random_seeds}\n"
    )
    results = tdg.evaluate_all(
        kg_name=KG_NAME,
        model_factory=model_factory,
        init_embd=args.init_embd,
        split_seeds=seeds,
        random_seeds=args.random_seeds,
        splits_dir=args.splits_dir,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        verbose=True,
        save_results=True,
        results_dir=args.results_dir,
        run_id=run_id,
        options=options,
        save_variant=args.save_variant,
    )
    test_f1 = results.get("aggregated", {}).get("test_f1", {})
    if test_f1:
        print(f"\nFinal test macro-F1: {test_f1['mean']:.4f} +/- {test_f1['std']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
