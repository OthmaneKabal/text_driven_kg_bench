"""Generate ranked scale-matched UMLS--NCI clean references.

The default command creates the three structurally closest candidates for
each of the GT2KG and KGGEN target sizes.  It deliberately writes no summary
files: the rank and structural score are printed to the console only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_preprocessing.clean_graph_sampling import (
    DEFAULT_COMMON_NODES,
    DEFAULT_SEMANTIC_TYPES,
    sample_scale_matched_clean_graphs,
)


DEFAULT_CLEAN = ROOT / "datasets" / "UMLS_nci_kg.json"
DEFAULT_TARGETS = (ROOT / "datasets" / "GT2KG_kg.json", ROOT / "datasets" / "KG_GEN_kg.json")


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    """Atomically write a graph JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the top-k anchor-guided scale-matched UMLS--NCI graphs."
    )
    parser.add_argument("--clean-graph", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--target-graphs", type=Path, nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--common-nodes", type=Path, default=DEFAULT_COMMON_NODES)
    parser.add_argument("--semantic-types", type=Path, default=DEFAULT_SEMANTIC_TYPES)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "datasets")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--walkers", type=int, default=64)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing ranked output. By default existing files stop the command.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.top_k < 1 or args.candidates < args.top_k:
        raise ValueError("Require 1 <= --top-k <= --candidates")

    output_dir = args.output_dir
    planned_outputs = [
        output_dir / f"{args.clean_graph.stem}_scale_matched_{target.stem}_top_{rank}.json"
        for target in args.target_graphs
        for rank in range(1, args.top_k + 1)
    ]
    existing = [path for path in planned_outputs if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output already exists; use --overwrite to replace it: {names}")

    for target in args.target_graphs:
        ranked = sample_scale_matched_clean_graphs(
            args.clean_graph,
            target,
            common_nodes_path=args.common_nodes,
            semantic_types_path=args.semantic_types,
            candidates=args.candidates,
            walkers=args.walkers,
            random_seed=args.random_seed,
            top_k=args.top_k,
            with_stats=True,
        )
        for graph, report in ranked:
            rank = report["selected_candidate_rank"]
            output = output_dir / f"{args.clean_graph.stem}_scale_matched_{target.stem}_top_{rank}.json"
            write_records(output, graph)
            print(
                f"Saved rank {rank}/{args.top_k}: {output} "
                f"(candidate seed={report['selected_candidate_seed']}, "
                f"distance={report['selected_candidate_score']:.6f})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
