"""Build the three reproducible GT2KG graph variants used in the benchmark.

The current ``GT2KG_kg.json`` is the predicate-frequency-50 graph without
semantic-type edges. The recovered noisy triples below that threshold are
added only to the unfiltered ``full`` variant.
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTERED_GRAPH = ROOT / "datasets" / "GT2KG_kg.json"
DEFAULT_RECOVERED = ROOT / "datasets" / "get_without_50_reverse_ing" / "noisy_kg_relations_not_in_gt2kg_under_50.json"
DEFAULT_SEMANTIC_TYPES = ROOT / "datasets" / "semantic_type_integrations" / "umls_nci_clean_1pct_common_terms.json"


def normalize_term(value: object) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def read_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{path} must be a JSON list of objects")
    return records


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
        normalized, semantic_type = item.get("normalized_term"), item.get("semantic_type")
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
    parser.add_argument("--filtered-graph", type=Path, default=DEFAULT_FILTERED_GRAPH)
    parser.add_argument("--recovered-under-50", type=Path, default=DEFAULT_RECOVERED)
    parser.add_argument("--semantic-types", type=Path, default=DEFAULT_SEMANTIC_TYPES)
    parser.add_argument("--semantic-relation", default="isa")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "datasets")
    args = parser.parse_args()
    if not args.semantic_relation.strip():
        parser.error("semantic relation cannot be empty")

    filtered = read_records(args.filtered_graph)
    recovered = read_records(args.recovered_under_50)
    selected = read_records(args.semantic_types)
    if any(str(record.get("predicate", "")) == args.semantic_relation for record in filtered):
        raise ValueError("The filtered input already contains semantic-type edges")
    base_keys = {(row.get("subject"), row.get("predicate"), row.get("object")) for row in filtered}
    overlapping_recovered = sum(
        (row.get("subject"), row.get("predicate"), row.get("object")) in base_keys for row in recovered
    )
    if overlapping_recovered:
        raise ValueError(f"Recovered input has {overlapping_recovered} exact triples already in the filtered graph")

    full = [*filtered, *recovered]
    no_smnt = list(filtered)
    vf, type_edges_added, missing_types = add_semantic_types(no_smnt, selected, args.semantic_relation)
    if missing_types:
        raise ValueError(f"{len(missing_types)} selected semantic-type terms are absent: {missing_types[:10]}")

    output_dir = args.output_dir
    paths = {
        "full": output_dir / "GT2KG_kg_full.json",
        "no_smnt": output_dir / "GT2KG_kg_no_smnt.json",
        "vf": output_dir / "GT2KG_kg_vf.json",
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
