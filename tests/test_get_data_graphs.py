"""Load every supported benchmark graph variant through ``TDGBench.get_data``.

Run from the repository root:
    python tests/test_get_data_graphs.py

The default matrix contains the 19 valid option combinations of the three
source graphs, plus the two final scale-matched clean references (21 loads).
The script uses small deterministic random embeddings, so it neither downloads
an embedding model nor writes embedding caches.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from tdg_bench import TDGBench


BASE_OPTIONS = (
    (),
    ("raw",),
    ("no_smnt",),
    ("no_freq_filter",),
    ("no_smnt", "no_freq_filter"),
)
UMLS_OPTIONS = (
    *BASE_OPTIONS,
    ("with_inverse",),
    ("no_smnt", "with_inverse"),
    ("no_freq_filter", "with_inverse"),
    ("no_smnt", "no_freq_filter", "with_inverse"),
)
SCALE_MATCHED_GRAPHS = (
    "UMLS_nci_kg_scale_matched_GT2KG_kg",
    "UMLS_nci_kg_scale_matched_KG_GEN_kg",
)


def default_cases() -> list[tuple[str, tuple[str, ...]]]:
    """Return all valid variants for source graphs plus final samples."""
    cases = [
        (kg_name, options)
        for kg_name in ("GT2KG_kg", "KG_GEN_kg")
        for options in BASE_OPTIONS
    ]
    cases.extend(("UMLS_nci_kg", options) for options in UMLS_OPTIONS)
    cases.extend((kg_name, ()) for kg_name in SCALE_MATCHED_GRAPHS)
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graphs",
        nargs="+",
        help="Override the default matrix with graph names using one shared --options list.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=8,
        help="Dimension of deterministic random test embeddings (default: 8).",
    )
    parser.add_argument(
        "--options",
        nargs="*",
        default=(),
        help="Variant options: raw, no_smnt, no_freq_filter, with_inverse.",
    )
    parser.add_argument(
        "--save-variant",
        action="store_true",
        help="Persist a requested variant JSON when it does not already exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = (
        [(kg_name, tuple(args.options)) for kg_name in args.graphs]
        if args.graphs
        else default_cases()
    )
    if args.embedding_dim < 1:
        raise ValueError("--embedding-dim must be >= 1")

    bench = TDGBench()
    failures: list[str] = []
    for position, (kg_name, options) in enumerate(cases, start=1):
        try:
            data, train_loader, val_loader, test_loader, graph_preparation = bench.get_data(
                kg_name=kg_name,
                init_embd="random_42",
                split_path=None,
                random_embd_dim=args.embedding_dim,
                use_cache=False,
                options=options,
                save_variant=args.save_variant,
            )
            print(
                f"OK [{position}/{len(cases)}] | {kg_name} | options={list(options)} "
                f"| nodes={data.num_nodes:,} | edges={data.num_edges:,} "
                f"| relations={len(graph_preparation.predicate_to_id)} | classes={data.num_classes} "
                f"| loaders=({len(train_loader)}, {len(val_loader)}, {len(test_loader)})"
            )
        except Exception as error:  # Report every failing graph in one run.
            failures.append(f"{kg_name} options={list(options)}: {type(error).__name__}: {error}")
            print(f"FAILED [{position}/{len(cases)}] | {failures[-1]}", file=sys.stderr)
        finally:
            # Each graph can be large; release it before loading the next one.
            data = train_loader = val_loader = test_loader = graph_preparation = None
            gc.collect()

    if failures:
        print(f"\n{len(failures)}/{len(cases)} graph configuration(s) failed.", file=sys.stderr)
        return 1
    print(f"\nAll {len(cases)} graph configuration(s) loaded successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
