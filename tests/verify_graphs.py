"""Fast integrity check for TDGBench graph JSON files.

The script loads one graph at a time (so it does not accumulate all graphs in
memory) and never modifies a dataset.  It checks JSON/SPO integrity, common
annotated-node coverage, semantic-type edges, frequency-filter invariants and
UMLS inverse-relation invariants.  A CSV and JSON report are written at the
requested output location.

Examples
--------
python tests/verify_graphs.py --set paper
python tests/verify_graphs.py --set all --output results/graph_verification
python tests/verify_graphs.py --graphs GT2KG_kg KG_GEN_kg UMLS_nci_kg
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
COMMON_NODES = DATASETS / "common_nodes.xlsx"
INVERSE_RESOLVER = DATASETS / "UMLS-NCI" / "rela_inverse_resolver.xlsx"

# Variants used in the paper and the experiment runners.
PAPER_GRAPHS = (
    "GT2KG_kg", "GT2KG_kg_raw", "GT2KG_kg_no_freq_filter", "GT2KG_kg_no_smnt",
    "KG_GEN_kg", "KG_GEN_kg_raw", "KG_GEN_kg_no_freq_filter", "KG_GEN_kg_no_smnt",
    "UMLS_nci_kg", "UMLS_nci_kg_raw", "UMLS_nci_kg_no_freq_filter",
    "UMLS_nci_kg_no_smnt", "UMLS_nci_kg_with_inverse",
    "UMLS_nci_kg_scale_matched_GT2KG_kg_top_1",
    "UMLS_nci_kg_scale_matched_GT2KG_kg_top_1_sparsity_matched",
    "UMLS_nci_kg_scale_matched_KG_GEN_kg_top_1",
    "UMLS_nci_kg_scale_matched_KG_GEN_kg_top_1_sparsity_matched",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--set", choices=("paper", "all"), default="paper",
        help="'paper' checks graphs used by the article/runners; 'all' checks every *_kg*.json in datasets.",
    )
    parser.add_argument("--graphs", nargs="+", help="Explicit graph stems; overrides --set.")
    parser.add_argument(
        "--files", nargs="+", type=Path,
        help="Explicit JSON file path(s); useful for a graph staged outside datasets/.",
    )
    parser.add_argument("--output", default="results/graph_verification")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def names_for(args: argparse.Namespace) -> list[str]:
    if args.graphs:
        return list(dict.fromkeys(args.graphs))
    if args.set == "paper":
        return list(PAPER_GRAPHS)
    return sorted(path.stem for path in DATASETS.glob("*_kg*.json"))


def expected_semantic_edges(name: str) -> bool:
    return "_raw" not in name and "_no_smnt" not in name


def expected_frequency_filter(name: str) -> bool:
    return "_raw" not in name and "_no_freq_filter" not in name


def expected_inverse_presence(name: str) -> bool | None:
    if not name.startswith("UMLS_nci_kg"):
        return None
    return name.endswith("_raw") or "_with_inverse" in name


def load_common_nodes() -> tuple[set[str], int]:
    if not COMMON_NODES.is_file():
        raise FileNotFoundError(f"Common-node file not found: {COMMON_NODES}")
    table = pd.read_excel(COMMON_NODES).dropna(subset=["term", "label"])
    return set(table["term"].astype(str)), int(table["label"].nunique())


def inverse_predicates() -> set[str]:
    if not INVERSE_RESOLVER.is_file():
        raise FileNotFoundError(f"Inverse resolver not found: {INVERSE_RESOLVER}")
    table = pd.read_excel(INVERSE_RESOLVER)
    if "inverse" not in table:
        raise ValueError(f"Column 'inverse' missing in {INVERSE_RESOLVER}")
    return {str(value) for value in table["inverse"].dropna() if str(value).strip()}


def inspect_graph(
    name: str, common_nodes: set[str], n_classes: int, inverses: set[str], path: Path | None = None,
) -> dict[str, Any]:
    path = path or DATASETS / f"{name}.json"
    row: dict[str, Any] = {"graph": name, "path": str(path), "status": "PASS"}
    if not path.is_file():
        return {**row, "status": "FAIL", "issues": "file missing"}

    try:
        with path.open(encoding="utf-8") as handle:
            records = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        return {**row, "status": "FAIL", "issues": f"invalid JSON: {error}"}
    if not isinstance(records, list):
        return {**row, "status": "FAIL", "issues": "root JSON value is not a list"}

    nodes: set[str] = set()
    predicates: Counter[str] = Counter()
    natural_predicates: Counter[str] = Counter()
    synthetic_edges = 0
    invalid_records = 0
    invalid_examples: list[dict[str, Any]] = []
    inverse_edges = 0
    triple_digest = hashlib.sha256()
    natural_triple_digest = hashlib.sha256()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            invalid_records += 1
            if len(invalid_examples) < 5:
                invalid_examples.append({"index": index, "record": record})
            continue
        subject, predicate, obj = record.get("subject"), record.get("predicate"), record.get("object")
        if not all(isinstance(value, str) and value.strip() for value in (subject, predicate, obj)):
            invalid_records += 1
            if len(invalid_examples) < 5:
                invalid_examples.append({"index": index, "record": record})
            continue
        nodes.update((subject, obj))
        predicates[predicate] += 1
        if record.get("synthetic") is True:
            synthetic_edges += 1
        else:
            natural_predicates[predicate] += 1
            natural_triple_digest.update(subject.encode("utf-8"))
            natural_triple_digest.update(b"\x1f")
            natural_triple_digest.update(predicate.encode("utf-8"))
            natural_triple_digest.update(b"\x1f")
            natural_triple_digest.update(obj.encode("utf-8"))
            natural_triple_digest.update(b"\n")
        inverse_edges += int(predicate in inverses)
        triple_digest.update(subject.encode("utf-8"))
        triple_digest.update(b"\x1f")
        triple_digest.update(predicate.encode("utf-8"))
        triple_digest.update(b"\x1f")
        triple_digest.update(obj.encode("utf-8"))
        triple_digest.update(b"\n")

    missing_common = sorted(common_nodes - nodes)
    issues: list[str] = []
    if invalid_records:
        issues.append(f"{invalid_records} invalid SPO record(s)")
    if missing_common:
        issues.append(f"{len(missing_common)} common node(s) missing")
    if expected_semantic_edges(name) and synthetic_edges == 0:
        issues.append("semantic-type edges expected but none found")
    if not expected_semantic_edges(name) and synthetic_edges:
        issues.append(f"{synthetic_edges} unexpected semantic-type edge(s)")
    if expected_frequency_filter(name):
        low_frequency = sum(count < 50 for count in natural_predicates.values())
        if low_frequency:
            issues.append(f"{low_frequency} natural predicate(s) occur fewer than 50 times")
    inverse_expected = expected_inverse_presence(name)
    if inverse_expected is True and inverse_edges == 0:
        issues.append("inverse relations expected but none found")
    if inverse_expected is False and inverse_edges:
        issues.append(f"{inverse_edges} inverse-relation edge(s) remain")

    row.update(
        {
            "records": len(records),
            "nodes": len(nodes),
            "unique_relations": len(predicates),
            "synthetic_semantic_edges": synthetic_edges,
            "inverse_edges": inverse_edges if inverse_expected is not None else "n/a",
            "common_nodes_present": len(common_nodes) - len(missing_common),
            "common_nodes_expected": len(common_nodes),
            "classes_expected": n_classes,
            "low_frequency_natural_relations": sum(count < 50 for count in natural_predicates.values()),
            "triple_sequence_sha256": triple_digest.hexdigest(),
            "natural_triple_sequence_sha256": natural_triple_digest.hexdigest(),
            "file_sha256": sha256(path),
            "file_bytes": path.stat().st_size,
            "status": "PASS" if not issues else "FAIL",
            "issues": "; ".join(issues) if issues else "",
            "invalid_record_examples": invalid_examples,
        }
    )
    del records
    return row


def add_cross_variant_checks(rows: list[dict[str, Any]]) -> list[str]:
    """Check that variants differ only by the operation named in their suffix."""
    by_name = {str(row["graph"]): row for row in rows if row.get("status") != "FAIL"}
    messages: list[str] = []

    def fail(graph: str, message: str) -> None:
        row = by_name.get(graph)
        if row is None:
            return
        row["status"] = "FAIL"
        row["issues"] = "; ".join(filter(None, (str(row.get("issues", "")), message)))
        messages.append(f"{graph}: {message}")

    def natural_equal(first: str, second: str, expectation: str) -> None:
        left, right = by_name.get(first), by_name.get(second)
        if left is None or right is None:
            return
        if left["natural_triple_sequence_sha256"] != right["natural_triple_sequence_sha256"]:
            fail(second, expectation)

    # GT2KG and KG-GEN: no-frequency differs from raw only by synthetic
    # semantic-type edges; final differs from no-semantic only by these edges.
    for prefix in ("GT2KG_kg", "KG_GEN_kg"):
        natural_equal(
            f"{prefix}_raw", f"{prefix}_no_freq_filter",
            "raw and no_freq_filter natural triples differ",
        )
        natural_equal(
            prefix, f"{prefix}_no_smnt",
            "final and no_smnt natural triples differ",
        )
        natural_equal(
            f"{prefix}_raw", f"{prefix}_no_freq_filter_no_smnt",
            "raw and no_freq_filter_no_smnt natural triples differ",
        )

    # UMLS-NCI: the final and no-STI variants have the same filtered,
    # inverse-removed natural graph.  The with-inverse graph must differ from
    # the no-frequency graph and must actually retain inverse relations.
    natural_equal(
        "UMLS_nci_kg", "UMLS_nci_kg_no_smnt",
        "final and no_smnt natural triples differ",
    )
    natural_equal(
        "UMLS_nci_kg_no_freq_filter", "UMLS_nci_kg_no_freq_filter_no_smnt",
        "no_freq_filter and no_freq_filter_no_smnt natural triples differ",
    )
    natural_equal(
        "UMLS_nci_kg_raw", "UMLS_nci_kg_no_freq_filter_no_smnt_with_inverse",
        "raw and no_freq_filter_no_smnt_with_inverse natural triples differ",
    )
    inverse = by_name.get("UMLS_nci_kg_with_inverse")
    no_frequency = by_name.get("UMLS_nci_kg_no_freq_filter")
    if inverse is not None and no_frequency is not None:
        if inverse["natural_triple_sequence_sha256"] == no_frequency["natural_triple_sequence_sha256"]:
            fail("UMLS_nci_kg_with_inverse", "with_inverse is identical to no_freq_filter")
        if int(inverse["inverse_edges"]) == 0:
            fail("UMLS_nci_kg_with_inverse", "with_inverse contains no inverse relations")
    return messages


def main() -> int:
    args = parse_args()
    common_nodes, n_classes = load_common_nodes()
    inverses = inverse_predicates()
    items = [(path.stem, path) for path in args.files] if args.files else [(name, None) for name in names_for(args)]
    print(f"Checking {len(items)} graph file(s) one at a time...")
    rows = []
    for index, (graph, path) in enumerate(items, 1):
        print(f"[{index}/{len(items)}] {graph}", flush=True)
        row = inspect_graph(graph, common_nodes, n_classes, inverses, path)
        rows.append(row)
        records = row.get("records")
        nodes = row.get("nodes")
        records_text = f"{records:,}" if isinstance(records, int) else "-"
        nodes_text = f"{nodes:,}" if isinstance(nodes, int) else "-"
        print(f"  {row['status']}: {records_text} edges, {nodes_text} nodes {row.get('issues', '')}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cross_variant_issues = add_cross_variant_checks(rows) if not args.files else []
    # Cross-variant checks may have changed statuses, so write the final rows now.
    pd.DataFrame(rows).to_csv(output / "graph_verification.csv", index=False)
    with (output / "graph_verification.json").open("w", encoding="utf-8") as handle:
        json.dump({"graphs": rows, "cross_variant_issues": cross_variant_issues}, handle, indent=2, ensure_ascii=False)
    failures = [row for row in rows if row["status"] != "PASS"]
    print(f"\nReport: {output / 'graph_verification.csv'}")
    if cross_variant_issues:
        print("Cross-variant failures:")
        for issue in cross_variant_issues:
            print(f"  - {issue}")
    print(f"Result: {len(rows) - len(failures)}/{len(rows)} PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
