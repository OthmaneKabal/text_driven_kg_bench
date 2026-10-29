"""Verify and summarise every stored preprocessing variant.

Each graph is loaded through the public ``get_graph`` API.  No graph is
rewritten: existing variant JSON files are only read.  The CSV and Excel
reports make it easy to check that raw, no-frequency, no-semantic-type, and
UMLS inverse-relation variants contain the expected structure.

Example
-------
python utilities/verify_preprocessing_graph_statistics.py \
  --output-dir results/preprocessing_graph_check
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"

from data_preprocessing.graph_variants import (
    OPTION_NO_FREQUENCY_FILTER,
    OPTION_NO_SEMANTIC_TYPES,
    OPTION_RAW,
    OPTION_WITH_INVERSE,
    get_graph,
    graph_variant_name,
)


COMMON_VARIANTS: tuple[tuple[str, ...], ...] = (
    (OPTION_RAW,),
    (),
    (OPTION_NO_SEMANTIC_TYPES,),
    (OPTION_NO_FREQUENCY_FILTER,),
    (OPTION_NO_FREQUENCY_FILTER, OPTION_NO_SEMANTIC_TYPES),
)
UMLS_VARIANTS: tuple[tuple[str, ...], ...] = (
    *COMMON_VARIANTS,
    (OPTION_WITH_INVERSE,),
    (OPTION_WITH_INVERSE, OPTION_NO_SEMANTIC_TYPES),
    (OPTION_WITH_INVERSE, OPTION_NO_FREQUENCY_FILTER),
    (OPTION_WITH_INVERSE, OPTION_NO_FREQUENCY_FILTER, OPTION_NO_SEMANTIC_TYPES),
)


class DisjointSet:
    """Union-find used to obtain weak connected components."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.size: dict[str, int] = {}

    def add(self, node: str) -> None:
        if node not in self.parent:
            self.parent[node] = node
            self.size[node] = 1

    def find(self, node: str) -> str:
        root = node
        while root != self.parent[root]:
            root = self.parent[root]
        while node != root:
            parent = self.parent[node]
            self.parent[node] = root
            node = parent
        return root

    def union(self, left: str, right: str) -> None:
        self.add(left)
        self.add(right)
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


@dataclass(frozen=True)
class VariantStatistics:
    graph: str
    base_graph: str
    options: str
    edges: int
    nodes: int
    unique_relations: int
    weakly_connected_components: int
    largest_component_size: int
    annotated_nodes_present: int
    classes_present: int
    annotated_nodes_in_components_smaller_than_10: int
    annotated_edge_homophily_percent: float | None
    annotated_edges_used_for_homophily: int
    synthetic_semantic_edges: int
    synthetic_isa_edges: int
    inverse_relation_edges: int | None
    low_frequency_natural_relations: int
    minimum_natural_relation_frequency: int | None


def load_labels(path: Path) -> dict[str, str]:
    table = pd.read_excel(path, engine="openpyxl")
    required = {"term", "label"}
    if missing := required - set(table.columns):
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return {
        str(row.term): str(row.label)
        for row in table.loc[:, ["term", "label"]].itertuples(index=False)
        if isinstance(row.term, str) and row.term
    }


def load_umls_inverse_predicates(path: Path) -> set[str]:
    table = pd.read_excel(path, engine="openpyxl")
    if "inverse" not in table.columns:
        raise ValueError(f"{path} is missing its 'inverse' column")
    return {str(value) for value in table["inverse"].dropna() if str(value).strip()}


def calculate_statistics(
    graph: str,
    base_graph: str,
    options: tuple[str, ...],
    records: list[dict[str, Any]],
    labels: dict[str, str],
    inverse_predicates: set[str],
) -> VariantStatistics:
    components = DisjointSet()
    predicates: set[str] = set()
    natural_predicate_counts: Counter[str] = Counter()
    labelled_edges = same_label_edges = 0
    synthetic_edges = synthetic_isa_edges = inverse_edges = 0

    for index, record in enumerate(records):
        subject, predicate, obj = record.get("subject"), record.get("predicate"), record.get("object")
        if not all(isinstance(value, str) and value.strip() for value in (subject, predicate, obj)):
            raise ValueError(f"{graph}: invalid SPO record at index {index}")
        components.union(subject, obj)
        predicates.add(predicate)
        is_synthetic = record.get("synthetic") is True
        synthetic_edges += int(is_synthetic)
        synthetic_isa_edges += int(is_synthetic and predicate == "isa")
        inverse_edges += int(predicate in inverse_predicates)
        if not is_synthetic:
            natural_predicate_counts[predicate] += 1
        if subject in labels and obj in labels:
            labelled_edges += 1
            same_label_edges += int(labels[subject] == labels[obj])

    component_sizes = Counter(components.find(node) for node in components.parent)
    annotated_terms = set(labels).intersection(components.parent)
    small_component_terms = sum(
        component_sizes[components.find(term)] < 10 for term in annotated_terms
    )
    natural_counts = list(natural_predicate_counts.values())
    return VariantStatistics(
        graph=graph,
        base_graph=base_graph,
        options="default (all treatments)" if not options else ", ".join(options),
        edges=len(records),
        nodes=len(components.parent),
        unique_relations=len(predicates),
        weakly_connected_components=len(component_sizes),
        largest_component_size=max(component_sizes.values(), default=0),
        annotated_nodes_present=len(annotated_terms),
        classes_present=len({labels[term] for term in annotated_terms}),
        annotated_nodes_in_components_smaller_than_10=small_component_terms,
        annotated_edge_homophily_percent=(100 * same_label_edges / labelled_edges if labelled_edges else None),
        annotated_edges_used_for_homophily=labelled_edges,
        synthetic_semantic_edges=synthetic_edges,
        synthetic_isa_edges=synthetic_isa_edges,
        inverse_relation_edges=inverse_edges if base_graph == "UMLS_nci_kg" else None,
        low_frequency_natural_relations=sum(count < 50 for count in natural_counts),
        minimum_natural_relation_frequency=min(natural_counts, default=None),
    )


def requested_variants() -> list[tuple[str, tuple[str, ...]]]:
    variants: list[tuple[str, tuple[str, ...]]] = []
    for base_graph in ("GT2KG_kg", "KG_GEN_kg"):
        variants.extend((base_graph, options) for options in COMMON_VARIANTS)
    variants.extend(("UMLS_nci_kg", options) for options in UMLS_VARIANTS)
    return variants


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-dir", type=Path, default=DATASETS)
    parser.add_argument("--common-nodes", type=Path, default=DATASETS / "common_nodes.xlsx")
    parser.add_argument(
        "--inverse-resolver", type=Path,
        default=DATASETS / "UMLS-NCI" / "rela_inverse_resolver.xlsx",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "preprocessing_graph_check")
    args = parser.parse_args()

    labels = load_labels(args.common_nodes)
    inverse_predicates = load_umls_inverse_predicates(args.inverse_resolver)
    rows: list[VariantStatistics] = []
    failures: list[str] = []

    for base_graph, options in requested_variants():
        graph_name = graph_variant_name(base_graph, options)
        try:
            print(f"Loading {graph_name} through get_graph(...)")
            records = get_graph(
                kg_name=base_graph,
                options=options,
                save=False,
                datasets_dir=args.datasets_dir,
            )
            if not isinstance(records, list):
                raise RuntimeError("get_graph did not return JSON records")
            rows.append(calculate_statistics(
                graph_name, base_graph, options, records, labels, inverse_predicates,
            ))
        except Exception as error:  # continue to inspect the other variants
            failures.append(f"{graph_name}: {type(error).__name__}: {error}")
            print(f"FAIL: {failures[-1]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([asdict(row) for row in rows])
    csv_path = args.output_dir / "preprocessing_variant_statistics.csv"
    xlsx_path = args.output_dir / "preprocessing_variant_statistics.xlsx"
    failures_path = args.output_dir / "preprocessing_variant_failures.txt"
    frame.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="Statistics", index=False)
        pd.DataFrame({"failure": failures or ["None"]}).to_excel(writer, sheet_name="Failures", index=False)
    failures_path.write_text("\n".join(failures) + ("\n" if failures else ""), encoding="utf-8")

    print(f"\nChecked: {len(rows)}/{len(requested_variants())} variants")
    print(f"CSV:   {csv_path}")
    print(f"Excel: {xlsx_path}")
    if failures:
        print(f"Failures ({len(failures)}): {failures_path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
