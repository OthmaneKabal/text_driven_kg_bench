"""Rebuild the fixed STI selection without pre-existing hierarchy triples.

The selection retains the original 1,846-edge budget and semantic-type
stratification.  Selected pairs already expressed through the hierarchy
predicate of any source graph are replaced by eligible shared terms from the
same semantic-type stratum.
"""

from __future__ import annotations

import argparse
import json
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"
DEFAULT_SELECTION = DATASETS / "semantic_type_integrations" / "umls_nci_clean_1pct_common_terms.json"
DEFAULT_OUTPUT_DIR = DATASETS / "regenerated_graph_variants" / "semantic_type_integrations"
FREQUENCY_THRESHOLD = 50
HIERARCHY_PREDICATES = {
    "UMLS_nci_kg": "isa",
    "GT2KG_kg": "is",
    "KG_GEN_kg": "is a",
}
SOURCE_HIERARCHY_PREDICATE_FORMS = {
    "UMLS_nci_kg": frozenset({"isa", "is a", "is-a"}),
    # GT2KG expresses the hierarchy-like relation as the OpenIE predicate
    # ``is``; using it avoids introducing a second predicate for STI edges.
    "GT2KG_kg": frozenset({"is", "isa", "is a", "is-a"}),
    "KG_GEN_kg": frozenset({"isa", "is a", "is-a"}),
}


def normalize(value: object) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def read_json(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError(f"Expected a JSON list of objects: {path}")
    return data


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def inverse_predicates(resolver_path: Path) -> set[str]:
    table = pd.read_excel(resolver_path)
    if "inverse" not in table.columns:
        raise ValueError(f"{resolver_path} must have an 'inverse' column")
    return {str(value) for value in table["inverse"].dropna() if str(value).strip()}


def base_labels(records: list[dict[str, Any]], excluded: set[str] | None = None) -> set[str]:
    excluded = excluded or set()
    counts = Counter(str(record.get("predicate", "")) for record in records)
    labels: set[str] = set()
    for record in records:
        predicate = str(record.get("predicate", ""))
        if predicate in excluded or counts[predicate] < FREQUENCY_THRESHOLD:
            continue
        for field in ("subject", "object"):
            value = record.get(field)
            if isinstance(value, str) and value.strip():
                labels.add(normalize(value))
    return labels


def hierarchy_pairs(
    records: list[dict[str, Any]], predicate_forms: frozenset[str]
) -> set[tuple[str, str]]:
    """Return hierarchy pairs under every spelling found in source data."""
    return {
        (normalize(record.get("subject", "")), normalize(record.get("object", "")))
        for record in records
        if normalize(record.get("predicate", "")) in predicate_forms
        and isinstance(record.get("subject"), str)
        and isinstance(record.get("object"), str)
    }


def semantic_evidence(records: list[dict[str, Any]]) -> dict[str, Counter[tuple[str, str, str]]]:
    evidence: dict[str, Counter[tuple[str, str, str]]] = defaultdict(Counter)
    for record in records:
        for label_field, type_field, types_field, cui_field in (
            ("subject", "subject_type", "subject_types", "subject_cui"),
            ("object", "object_type", "object_types", "object_cui"),
        ):
            label = record.get(label_field)
            semantic_type = record.get(type_field)
            cui = record.get(cui_field)
            if not isinstance(label, str) or not isinstance(semantic_type, str) or not isinstance(cui, str) or not cui:
                continue
            tui = ""
            types = record.get(types_field)
            if isinstance(types, list) and types and isinstance(types[0], dict):
                candidate = types[0].get("tui")
                if isinstance(candidate, str):
                    tui = candidate
            evidence[normalize(label)][(semantic_type, tui, cui)] += 1
    return evidence


def normalized_labels(records: list[dict[str, Any]]) -> dict[str, set[str]]:
    """Collect display labels once, keyed by their normalized form.

    This deliberately runs in one pass over the UMLS records.  Re-scanning the
    1.2M-record graph once per candidate would make selection impractical.
    """
    labels: dict[str, set[str]] = defaultdict(set)
    for record in records:
        for field in ("subject", "object"):
            label = record.get(field)
            if isinstance(label, str) and label.strip():
                labels[normalize(label)].add(label)
    return labels


def choose_type(counter: Counter[tuple[str, str, str]]) -> tuple[str, str, str]:
    return min(counter, key=lambda item: (-counter[item], item[0], item[1], item[2]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--umls-raw", type=Path, default=DATASETS / "UMLS_nci_kg_raw.json")
    parser.add_argument("--gt2kg-raw", type=Path, default=DATASETS / "GT2KG_kg_raw.json")
    parser.add_argument("--kggen-raw", type=Path, default=DATASETS / "KG_GEN_kg_raw.json")
    parser.add_argument("--common-nodes", type=Path, default=DATASETS / "common_nodes.xlsx")
    parser.add_argument("--inverse-resolver", type=Path, default=DATASETS / "UMLS-NCI" / "rela_inverse_resolver.xlsx")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    original = read_json(args.selection)
    budget = len(original)
    if len({(item.get("normalized_term"), item.get("semantic_type")) for item in original}) != budget:
        raise ValueError("Input STI selection contains duplicate term/type pairs")

    umls = read_json(args.umls_raw)
    gt2kg = read_json(args.gt2kg_raw)
    kggen = read_json(args.kggen_raw)
    labels = {
        "UMLS_nci_kg": base_labels(umls, inverse_predicates(args.inverse_resolver)),
        "GT2KG_kg": base_labels(gt2kg),
        "KG_GEN_kg": base_labels(kggen),
    }
    evidence = semantic_evidence(umls)
    umls_labels = normalized_labels(umls)
    existing = {
        graph: hierarchy_pairs(records, SOURCE_HIERARCHY_PREDICATE_FORMS[graph])
        for graph, records in {
            "UMLS_nci_kg": umls,
            "GT2KG_kg": gt2kg,
            "KG_GEN_kg": kggen,
        }.items()
    }
    shared_labels = set.intersection(*labels.values())
    common = pd.read_excel(args.common_nodes)
    excluded_common = {normalize(term) for term in common["term"].dropna()}
    eligible_terms = shared_labels.intersection(evidence).difference(excluded_common)

    def already_expressed(term: str, semantic_type: str) -> bool:
        pair = (term, normalize(semantic_type))
        return any(pair in pairs for pairs in existing.values())

    invalid = [
        item for item in original
        if item.get("normalized_term") not in eligible_terms
        or already_expressed(str(item.get("normalized_term")), str(item.get("semantic_type")))
    ]
    retained = [item for item in original if item not in invalid]
    targets = Counter(str(item["semantic_type"]) for item in original)
    retained_by_type = Counter(str(item["semantic_type"]) for item in retained)
    replacements_needed = {semantic_type: targets[semantic_type] - retained_by_type[semantic_type] for semantic_type in targets}

    retained_pairs = {(str(item["normalized_term"]), str(item["semantic_type"])) for item in retained}
    candidates_by_type: dict[str, list[dict[str, str]]] = defaultdict(list)
    for term in sorted(eligible_terms):
        semantic_type, tui, cui = choose_type(evidence[term])
        if (term, semantic_type) in retained_pairs or already_expressed(term, semantic_type):
            continue
        candidates_by_type[semantic_type].append({
            "normalized_term": term,
            "umls_nci_label": min(umls_labels[term], key=lambda label: (normalize(label), label)),
            "semantic_type": semantic_type,
            "semantic_type_tui": tui,
            "umls_nci_cui": cui,
        })

    rng = random.Random(args.seed)
    replacements: list[dict[str, str]] = []
    unreplaced_by_type: dict[str, int] = {}
    for semantic_type, count in sorted(replacements_needed.items()):
        if not count:
            continue
        candidates = candidates_by_type[semantic_type]
        selected_count = min(len(candidates), count)
        replacements.extend(rng.sample(candidates, selected_count))
        if selected_count < count:
            # Never borrow a candidate from a different semantic-type stratum:
            # preserving the stratification is more important than preserving
            # one rounded 1% budget entry.
            unreplaced_by_type[semantic_type] = count - selected_count

    selected = [
        {key: value for key, value in item.items() if key != "anchor_id"}
        for item in [*retained, *replacements]
    ]
    selected.sort(key=lambda item: (str(item["semantic_type"]), str(item["normalized_term"])))
    expected_selected = budget - sum(unreplaced_by_type.values())
    if len(selected) != expected_selected:
        raise AssertionError(f"Expected {expected_selected} selected pairs, got {len(selected)}")
    for index, item in enumerate(selected, start=1):
        item["anchor_id"] = f"umls-nci-clean-1pct-{index:04d}"

    for item in selected:
        if already_expressed(item["normalized_term"], item["semantic_type"]):
            raise AssertionError(f"Selected an already expressed hierarchy pair: {item}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "umls_nci_clean_1pct_common_terms.json"
    excel_path = args.output_dir / "umls_nci_clean_1pct_common_terms.xlsx"
    audit_path = args.output_dir / "selection_audit.json"
    replacements_path = args.output_dir / "replaced_preexisting_pairs.json"
    write_json(json_path, selected)
    pd.DataFrame(selected).to_excel(excel_path, index=False)
    write_json(replacements_path, {
        "replaced": invalid,
        "replacement": replacements,
    })
    write_json(audit_path, {
        "budget": budget,
        "final_selection_edges": len(selected),
        "retained": len(retained),
        "replaced": len(invalid),
        "unreplaced_removed_by_semantic_type": unreplaced_by_type,
        "semantic_type_strata": len(targets),
        "hierarchy_relations": HIERARCHY_PREDICATES,
        "preexisting_selected_pairs_by_graph": {
            graph: sum(
                (str(item.get("normalized_term")), normalize(str(item.get("semantic_type")))) in pairs
                for item in original
            )
            for graph, pairs in existing.items()
        },
        "source_hierarchy_predicate_counts": {
            graph: dict(sorted(Counter(
                str(record.get("predicate", ""))
                for record in records
                if normalize(record.get("predicate", "")) in SOURCE_HIERARCHY_PREDICATE_FORMS[graph]
            ).items()))
            for graph, records in {
                "UMLS_nci_kg": umls,
                "GT2KG_kg": gt2kg,
                "KG_GEN_kg": kggen,
            }.items()
        },
        "replacement_counts_by_semantic_type": dict(sorted(replacements_needed.items())),
        "eligible_shared_terms": len(eligible_terms),
    })
    print(json.dumps({
        "selection": str(json_path), "excel": str(excel_path), "audit": str(audit_path),
        "source_budget": budget, "final_selection_edges": len(selected),
        "replaced": len(invalid), "retained": len(retained),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
