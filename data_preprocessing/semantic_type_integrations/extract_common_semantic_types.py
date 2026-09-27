"""Select a shared, type-stratified UMLS semantic-type annotation list.

The fixed annotation budget is a fraction of clean UMLS-NCI concept CUIs.
Candidates are textual concepts shared by GT2KG, KGGEN and UMLS-NCI, after
excluding the pre-existing benchmark ``common_nodes`` list.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GT2KG = ROOT / "datasets" / "GT2KG_kg.json"
DEFAULT_KGGEN = ROOT / "datasets" / "KG_GEN_kg.json"
DEFAULT_NCI = ROOT / "datasets" / "UMLS-NCI" / "KG_NCI_vf.json"
DEFAULT_COMMON_NODES = ROOT / "datasets" / "common_nodes.xlsx"
DEFAULT_OUTPUT = (
    ROOT / "datasets" / "semantic_type_integrations" / "umls_nci_clean_1pct_common_terms.json"
)


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


def concept_records(records: Iterable[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
    return (record for record in records if not record.get("synthetic", False))


def labels_by_normalized_term(records: Iterable[Mapping[str, Any]]) -> dict[str, set[str]]:
    labels: dict[str, set[str]] = defaultdict(set)
    for record in records:
        for field in ("subject", "object"):
            value = record.get(field)
            if isinstance(value, str) and value.strip():
                labels[normalize_term(value)].add(value)
    return labels


def preferred_label(labels: Iterable[str]) -> str:
    return min(labels, key=lambda label: (normalize_term(label), label))


def concept_cuis(records: Iterable[Mapping[str, Any]]) -> set[str]:
    return {
        value
        for record in records
        for field in ("subject_cui", "object_cui")
        if isinstance((value := record.get(field)), str) and value
    }


def primary_type_evidence(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Counter[tuple[str, str, str]]]:
    """Collect label -> (primary type, TUI, CUI) evidence from NCI."""
    evidence: dict[str, Counter[tuple[str, str, str]]] = defaultdict(Counter)
    for record in records:
        for label_field, type_field, types_field, cui_field in (
            ("subject", "subject_type", "subject_types", "subject_cui"),
            ("object", "object_type", "object_types", "object_cui"),
        ):
            label = record.get(label_field)
            semantic_type = record.get(type_field)
            cui = record.get(cui_field)
            if not isinstance(label, str) or not isinstance(semantic_type, str):
                continue
            if not isinstance(cui, str) or not cui:
                continue
            tui = ""
            types = record.get(types_field)
            if isinstance(types, list) and types and isinstance(types[0], dict):
                candidate_tui = types[0].get("tui")
                if isinstance(candidate_tui, str):
                    tui = candidate_tui
            evidence[normalize_term(label)][(semantic_type, tui, cui)] += 1
    return evidence


def choose_type(evidence: Counter[tuple[str, str, str]]) -> tuple[str, str, str]:
    return min(evidence, key=lambda item: (-evidence[item], item[0], item[1], item[2]))


def proportional_allocation(counts: Mapping[str, int], budget: int) -> dict[str, int]:
    total = sum(counts.values())
    if budget > total:
        raise ValueError(f"Budget {budget} exceeds candidate count {total}")
    quotas = {name: budget * count / total for name, count in counts.items()}
    allocation = {name: math.floor(quota) for name, quota in quotas.items()}
    remaining = budget - sum(allocation.values())
    for name in sorted(counts, key=lambda item: (-(quotas[item] - allocation[item]), item))[:remaining]:
        allocation[name] += 1
    return allocation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt2kg", type=Path, default=DEFAULT_GT2KG)
    parser.add_argument("--kggen", type=Path, default=DEFAULT_KGGEN)
    parser.add_argument("--nci", type=Path, default=DEFAULT_NCI)
    parser.add_argument("--common-nodes", type=Path, default=DEFAULT_COMMON_NODES)
    parser.add_argument("--fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not 0 < args.fraction <= 1:
        parser.error("--fraction must be in (0, 1]")

    gt2kg = read_records(args.gt2kg)
    kggen = read_records(args.kggen)
    nci_all = read_records(args.nci)
    nci = list(concept_records(nci_all))
    gt2kg_labels = labels_by_normalized_term(gt2kg)
    kggen_labels = labels_by_normalized_term(kggen)
    nci_labels = labels_by_normalized_term(nci)

    common_frame = pd.read_excel(args.common_nodes)
    if "term" not in common_frame.columns:
        raise ValueError(f"{args.common_nodes} must contain a 'term' column")
    excluded_common = {normalize_term(term) for term in common_frame["term"].dropna()}
    evidence = primary_type_evidence(nci)
    candidates = (
        set(gt2kg_labels) & set(kggen_labels) & set(nci_labels) & set(evidence)
    ) - excluded_common

    clean_nci_cuis = len(concept_cuis(nci))
    budget = math.floor(clean_nci_cuis * args.fraction)
    if budget > len(candidates):
        raise ValueError(f"Budget {budget} exceeds {len(candidates)} eligible shared terms")

    types = {term: choose_type(evidence[term]) for term in candidates}
    terms_by_type: dict[str, list[str]] = defaultdict(list)
    for term, (semantic_type, _tui, _cui) in types.items():
        terms_by_type[semantic_type].append(term)
    allocation = proportional_allocation(
        {semantic_type: len(terms) for semantic_type, terms in terms_by_type.items()}, budget
    )
    rng = random.Random(args.seed)
    selected: list[dict[str, Any]] = []
    for semantic_type in sorted(terms_by_type):
        for term in sorted(rng.sample(sorted(terms_by_type[semantic_type]), allocation[semantic_type])):
            selected_type, tui, cui = types[term]
            selected.append({
                "normalized_term": term,
                "umls_nci_label": preferred_label(nci_labels[term]),
                "semantic_type": selected_type,
                "semantic_type_tui": tui,
                "umls_nci_cui": cui,
            })
    selected.sort(key=lambda item: (item["semantic_type"], item["normalized_term"]))
    for index, item in enumerate(selected, start=1):
        item["anchor_id"] = f"umls-nci-clean-1pct-{index:04d}"
    write_json(args.output, selected)
    print(json.dumps({
        "clean_umls_nci_concept_cuis": clean_nci_cuis,
        "fraction": args.fraction,
        "annotation_budget": budget,
        "eligible_shared_terms": len(candidates),
        "semantic_type_strata": len(terms_by_type),
        "selected_terms": len(selected),
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
