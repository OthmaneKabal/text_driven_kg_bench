"""Materialise every supported preprocessing variant for selected TDG graphs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_preprocessing.graph_variants import (
    OPTION_NO_FREQUENCY_FILTER,
    OPTION_NO_SEMANTIC_TYPES,
    OPTION_RAW,
    OPTION_WITH_INVERSE,
    UMLS_NCI_KG_NAME,
    get_graph,
)


DEFAULT_GRAPHS = ("GT2KG_kg", "KG_GEN_kg", UMLS_NCI_KG_NAME)


def variants_for(kg_name: str) -> tuple[tuple[str, ...], ...]:
    """Return raw, default, and every supported opt-out combination."""
    common = (
        (OPTION_RAW,),
        (),
        (OPTION_NO_SEMANTIC_TYPES,),
        (OPTION_NO_FREQUENCY_FILTER,),
        (OPTION_NO_FREQUENCY_FILTER, OPTION_NO_SEMANTIC_TYPES),
    )
    if kg_name != UMLS_NCI_KG_NAME:
        return common
    return (
        *common,
        (OPTION_WITH_INVERSE,),
        (OPTION_WITH_INVERSE, OPTION_NO_SEMANTIC_TYPES),
        (OPTION_WITH_INVERSE, OPTION_NO_FREQUENCY_FILTER),
        (OPTION_WITH_INVERSE, OPTION_NO_FREQUENCY_FILTER, OPTION_NO_SEMANTIC_TYPES),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="+", default=DEFAULT_GRAPHS)
    parser.add_argument(
        "--format", choices=("json", "rdf"), default="json",
        help="Output format to materialise. RDF writes paired RDF/XML .rdf files.",
    )
    args = parser.parse_args()

    for kg_name in args.graphs:
        print(f"\nGenerating variants for {kg_name}")
        for options in variants_for(kg_name):
            label = "default" if not options else ", ".join(options)
            output = get_graph(
                kg_name=kg_name, options=options, save=True, format=args.format
            )
            if args.format == "rdf":
                print(f"  {label}: {output}")
            else:
                print(f"  {label}: {len(output):,} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
