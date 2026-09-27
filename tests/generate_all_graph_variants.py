"""Generate every supported variant for the three benchmark graphs.

The JSON variants are saved in ``datasets/`` through ``get_graph(save=True)``.
The only report written by this script is the Excel workbook in ``tests/``.

Run:
    python tests/generate_all_graph_variants.py
"""

from __future__ import annotations

import argparse
import statistics
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data_preprocessing.graph_variants import get_graph


DEFAULT_OUTPUT = Path(__file__).with_name("graph_variants_statistics.xlsx")
GRAPH_NAMES = ("UMLS_nci_kg", "GT2KG_kg", "KG_GEN_kg")


def normalise(value: object) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def all_options(kg_name: str) -> list[tuple[str, ...]]:
    """Return every valid option set, including default and raw."""
    base = [
        (),
        ("no_smnt",),
        ("no_freq_filter",),
        ("no_freq_filter", "no_smnt"),
        ("raw",),
    ]
    if kg_name != "UMLS_nci_kg":
        return base
    return [
        (),
        ("no_smnt",),
        ("no_freq_filter",),
        ("no_freq_filter", "no_smnt"),
        ("with_inverse",),
        ("no_smnt", "with_inverse"),
        ("no_freq_filter", "with_inverse"),
        ("no_freq_filter", "no_smnt", "with_inverse"),
        ("raw",),
    ]


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.size: dict[str, int] = {}

    def add(self, value: str) -> None:
        if value not in self.parent:
            self.parent[value] = value
            self.size[value] = 1

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while value != root:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


def graph_statistics(graph: Iterable[dict[str, Any]]) -> dict[str, int | float]:
    """Compute undirected degree and weakly connected-component statistics."""
    union_find = UnionFind()
    degree: Counter[str] = Counter()
    predicates: set[str] = set()
    edge_count = 0

    for record in graph:
        edge_count += 1
        predicates.add(str(record.get("predicate", "")))
        subject, obj = record.get("subject"), record.get("object")
        if not isinstance(subject, str) or not subject.strip():
            continue
        if not isinstance(obj, str) or not obj.strip():
            continue
        subject_id, object_id = normalise(subject), normalise(obj)
        union_find.add(subject_id)
        union_find.add(object_id)
        union_find.union(subject_id, object_id)
        degree[subject_id] += 1
        degree[object_id] += 1

    component_sizes: Counter[str] = Counter(
        union_find.find(node) for node in union_find.parent
    )
    degrees = list(degree.values())
    return {
        "edges": edge_count,
        "nodes": len(union_find.parent),
        "unique_relations": len(predicates),
        "mean_undirected_degree": (sum(degrees) / len(degrees)) if degrees else 0.0,
        "median_undirected_degree": statistics.median(degrees) if degrees else 0.0,
        "max_undirected_degree": max(degrees, default=0),
        "weakly_connected_components": len(component_sizes),
        "largest_component_nodes": max(component_sizes.values(), default=0),
    }


def option_label(options: tuple[str, ...]) -> str:
    return "default_all_treatments" if not options else "+".join(options)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--kg-name",
        action="append",
        choices=GRAPH_NAMES,
        dest="kg_names",
        help="Optional graph to process; repeat to process more than one. Default: all three.",
    )
    args = parser.parse_args()
    kg_names = args.kg_names or list(GRAPH_NAMES)

    rows: list[dict[str, Any]] = []
    for kg_name in kg_names:
        for options in all_options(kg_name):
            print(f"Building {kg_name}: {option_label(options)}", flush=True)
            graph = get_graph(kg_name=kg_name, options=list(options), save=True)
            rows.append(
                {
                    "kg_name": kg_name,
                    "options": option_label(options),
                    "saved_json": str(
                        REPOSITORY_ROOT
                        / "datasets"
                        / (
                            f"{kg_name}.json"
                            if not options
                            else f"{kg_name}_{'_'.join(options)}.json"
                        )
                    ),
                    **graph_statistics(graph),
                }
            )

    report = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_excel(args.output, index=False, sheet_name="graph_variants")
    print(f"Excel report written: {args.output}")
    print(report.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
