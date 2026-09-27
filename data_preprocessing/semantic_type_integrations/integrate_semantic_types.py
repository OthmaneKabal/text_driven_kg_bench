"""Append a selected semantic-type list to any JSON graph.

The relation is explicit at the command line, so the same selected terms can
be integrated with ``isa``, ``has_semantic_type`` or another chosen predicate.
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Any


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--semantic-types", type=Path, required=True)
    parser.add_argument("--relation", required=True, help="Predicate for the injected type edge.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true", help="Fail if any selected term is absent.")
    args = parser.parse_args()
    if not args.relation.strip():
        parser.error("--relation cannot be empty")

    graph = read_records(args.graph)
    selected = read_records(args.semantic_types)
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
        key = (normalized, args.relation, semantic_type)
        if key in existing:
            continue
        additions.append({
            "subject": subject,
            "predicate": args.relation,
            "object": semantic_type,
            "synthetic": True,
            "source": "semantic_type_integrations",
            "anchor_id": item.get("anchor_id"),
            "normalized_term": normalized,
            "semantic_type_tui": item.get("semantic_type_tui"),
        })
        existing.add(key)
    if args.strict and missing:
        raise ValueError(f"{len(missing)} selected terms are absent from {args.graph}")
    write_json(args.output, [*graph, *additions])
    print(json.dumps({
        "graph_records_before": len(graph),
        "selected_type_terms": len(selected),
        "type_edges_added": len(additions),
        "already_present": len(selected) - len(additions) - len(missing),
        "missing_selected_terms": len(missing),
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
