"""Build the three reproducible KGGEN graph variants used in the benchmark.

``KG_GEN_kg_full`` keeps every original triple and applies the available
DeepSeek predicate mappings. ``KG_GEN_kg_no_smnt`` additionally applies the
frequency-50 filter while restoring coverage-critical mapped triples.
``KG_GEN_kg_vf`` is the latter graph with the selected semantic-type edges.
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = ROOT / "datasets" / "KG_GEN_kg.json"
DEFAULT_DECISIONS = ROOT / "datasets" / "kggen_relation_mapping" / "rare_relation_mapping_decisions.json"
DEFAULT_COMMON_NODES = ROOT / "datasets" / "common_nodes.xlsx"
DEFAULT_SEMANTIC_TYPES = ROOT / "datasets" / "semantic_type_integrations" / "umls_nci_clean_1pct_common_terms.json"


def normalize_term(value: object) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def graph_summary(records: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    nodes = {
        normalize_term(record.get(field, ""))
        for record in records
        for field in ("subject", "object")
        if isinstance(record.get(field), str) and record[field].strip()
    }
    predicates = Counter(str(record.get("predicate", "")) for record in records)
    return {
        "records": len(records),
        "unique_nodes": len(nodes),
        "unique_relations": len(predicates),
        "relation_frequencies": dict(sorted(predicates.items())),
        **extra,
    }


def mapped_copy(record: dict[str, Any], mappings: dict[str, str]) -> dict[str, Any]:
    predicate = str(record["predicate"])
    target = mappings.get(predicate)
    if target is None:
        return record
    replacement = dict(record)
    replacement["predicate"] = target
    replacement["old_predicate"] = predicate
    replacement["predicate_mapping_source"] = "deepseek_forced_frequent_choice"
    return replacement


def add_semantic_types(
    graph: list[dict[str, Any]], selected: list[dict[str, Any]], relation: str
) -> tuple[list[dict[str, Any]], int, list[str]]:
    labels: dict[str, set[str]] = {}
    for record in graph:
        for field in ("subject", "object"):
            value = record.get(field)
            if isinstance(value, str) and value.strip():
                labels.setdefault(normalize_term(value), set()).add(value)
    existing = {
        (normalize_term(record.get("subject", "")), str(record.get("predicate", "")), str(record.get("object", "")))
        for record in graph
    }
    additions: list[dict[str, Any]] = []
    missing: list[str] = []
    for item in selected:
        normalized = item.get("normalized_term")
        semantic_type = item.get("semantic_type")
        if not isinstance(normalized, str) or not isinstance(semantic_type, str):
            raise ValueError("Each semantic-type record needs normalized_term and semantic_type")
        originals = labels.get(normalized)
        if not originals:
            missing.append(normalized)
            continue
        subject = min(originals, key=lambda value: (normalize_term(value), value))
        key = (normalized, relation, semantic_type)
        if key in existing:
            continue
        additions.append({
            "subject": subject,
            "predicate": relation,
            "object": semantic_type,
            "synthetic": True,
            "source": "semantic_type_integrations",
            "anchor_id": item.get("anchor_id"),
            "normalized_term": normalized,
            "semantic_type_tui": item.get("semantic_type_tui"),
        })
        existing.add(key)
    return [*graph, *additions], len(additions), missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--common-nodes", type=Path, default=DEFAULT_COMMON_NODES)
    parser.add_argument("--semantic-types", type=Path, default=DEFAULT_SEMANTIC_TYPES)
    parser.add_argument("--threshold", type=int, default=50)
    parser.add_argument("--semantic-relation", default="isa")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "datasets")
    args = parser.parse_args()
    if args.threshold < 1 or not args.semantic_relation.strip():
        parser.error("threshold must be positive and semantic relation cannot be empty")

    graph = read_json(args.graph)
    decisions = read_json(args.decisions)
    selected = read_json(args.semantic_types)
    if not isinstance(graph, list) or not all(isinstance(row, dict) for row in graph):
        raise ValueError("Graph must be a JSON list of objects")
    if not isinstance(decisions, list) or not all(isinstance(row, dict) for row in decisions):
        raise ValueError("Decisions must be a JSON list of objects")
    if not isinstance(selected, list) or not all(isinstance(row, dict) for row in selected):
        raise ValueError("Semantic types must be a JSON list of objects")

    mappings = {
        str(row["rare_relation"]): str(row["target_relation"])
        for row in decisions
        if row.get("decision") == "map"
        and isinstance(row.get("rare_relation"), str)
        and isinstance(row.get("target_relation"), str)
    }
    if not mappings:
        raise ValueError("No accepted predicate mappings found")
    counts = Counter(str(row["predicate"]) for row in graph)
    common_nodes = pd.read_excel(args.common_nodes)
    if "term" not in common_nodes.columns:
        raise ValueError(f"{args.common_nodes} must contain a 'term' column")
    gs_terms = {normalize_term(value) for value in common_nodes["term"].dropna()}
    semantic_terms = {
        row["normalized_term"] for row in selected
        if isinstance(row.get("normalized_term"), str)
    }
    kept_nodes = {
        normalize_term(value)
        for row in graph if counts[str(row["predicate"])] >= args.threshold
        for field in ("subject", "object")
        if isinstance((value := row.get(field)), str) and value.strip()
    }
    protected_terms = (gs_terms - kept_nodes) | (semantic_terms - kept_nodes)

    def coverage_critical(row: dict[str, Any]) -> bool:
        return any(normalize_term(row.get(field, "")) in protected_terms for field in ("subject", "object"))

    full = [mapped_copy(row, mappings) for row in graph]
    no_smnt = [
        mapped_copy(row, mappings)
        for row in graph
        if counts[str(row["predicate"])] >= args.threshold
        or (coverage_critical(row) and str(row["predicate"]) in mappings)
    ]
    vf, type_edges_added, missing_types = add_semantic_types(no_smnt, selected, args.semantic_relation)
    if missing_types:
        raise ValueError(f"{len(missing_types)} selected semantic-type terms are absent after mapping: {missing_types[:10]}")

    output_dir = args.output_dir
    paths = {
        "full": output_dir / "KG_GEN_kg_full.json",
        "no_smnt": output_dir / "KG_GEN_kg_no_smnt.json",
        "vf": output_dir / "KG_GEN_kg_vf.json",
    }
    for name, records in (("full", full), ("no_smnt", no_smnt), ("vf", vf)):
        write_json(paths[name], records)
    print(json.dumps({
        name: {"path": str(paths[name]), "records": len(records)}
        for name, records in (("full", full), ("no_smnt", no_smnt), ("vf", vf))
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
