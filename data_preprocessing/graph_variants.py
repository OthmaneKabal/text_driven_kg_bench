"""Load or construct graph variants from raw JSON graphs.

The module intentionally performs only three generic transformations:
frequency filtering, semantic-type integration, and (for UMLS-NCI only)
inverse-relation removal. It never calls an LLM or runs source-specific
predicate normalization.
"""

from __future__ import annotations

import json
import unicodedata
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS_DIR = ROOT / "datasets"
DEFAULT_SEMANTIC_TYPES = (
    DEFAULT_DATASETS_DIR
    / "semantic_type_integrations"
    / "umls_nci_clean_1pct_common_terms.json"
)
DEFAULT_UMLS_INVERSE_RESOLVER = (
    DEFAULT_DATASETS_DIR / "UMLS-NCI" / "rela_inverse_resolver.xlsx"
)
FREQUENCY_THRESHOLD = 50
UMLS_NCI_KG_NAME = "UMLS_nci_kg"

OPTION_RAW = "raw"
OPTION_NO_SEMANTIC_TYPES = "no_smnt"
OPTION_NO_FREQUENCY_FILTER = "no_freq_filter"
OPTION_WITH_INVERSE = "with_inverse"
VALID_OPTIONS = frozenset(
    {
        OPTION_RAW,
        OPTION_NO_SEMANTIC_TYPES,
        OPTION_NO_FREQUENCY_FILTER,
        OPTION_WITH_INVERSE,
    }
)


def _normalize_text(value: object) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def _read_graph(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError(f"Graph file must be a JSON list of objects: {path}")
    return records


def _write_graph(path: Path, graph: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(graph, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _normalise_options(kg_name: str, options: Iterable[str] | None) -> tuple[str, ...]:
    resolved = tuple(sorted(set(options or ())))
    unknown = set(resolved) - VALID_OPTIONS
    if unknown:
        raise ValueError(f"Unknown graph option(s): {sorted(unknown)}. Supported options: {sorted(VALID_OPTIONS)}")
    if OPTION_RAW in resolved and len(resolved) != 1:
        raise ValueError("'raw' is exclusive: it already means no treatment")
    if OPTION_WITH_INVERSE in resolved and kg_name != UMLS_NCI_KG_NAME:
        raise ValueError("'with_inverse' is available only for kg_name='UMLS_nci_kg'")
    return resolved


def _variant_path(kg_name: str, options: tuple[str, ...], datasets_dir: Path) -> Path:
    if not options:
        return datasets_dir / f"{kg_name}.json"
    if options == (OPTION_RAW,):
        return datasets_dir / f"{kg_name}_raw.json"
    return datasets_dir / f"{kg_name}_{'_'.join(options)}.json"


def graph_variant_name(kg_name: str, options: Iterable[str] | None = None) -> str:
    """Return the deterministic cache/embedding name for a graph variant.

    The returned value is a file stem, e.g. ``UMLS_nci_kg_no_smnt``.  It does
    not create or require the corresponding JSON file.
    """
    if not isinstance(kg_name, str) or not kg_name.strip():
        raise ValueError("kg_name must be a non-empty string")
    kg_name = kg_name.strip()
    resolved_options = _normalise_options(kg_name, options)
    return _variant_path(kg_name, resolved_options, DEFAULT_DATASETS_DIR).stem


def _load_raw_or_warn(kg_name: str, datasets_dir: Path) -> list[dict[str, Any]]:
    raw_path = datasets_dir / f"{kg_name}_raw.json"
    if raw_path.exists():
        return _read_graph(raw_path)
    default_path = datasets_dir / f"{kg_name}.json"
    if not default_path.exists():
        raise FileNotFoundError(
            f"Neither raw nor default graph exists for '{kg_name}': {raw_path}, {default_path}"
        )
    warnings.warn(
        f"Raw graph not found for '{kg_name}'; requested treatments are applied to '{default_path.name}'.",
        RuntimeWarning,
        stacklevel=3,
    )
    return _read_graph(default_path)


def _remove_umls_inverse_relations(graph: list[dict[str, Any]], resolver_path: Path) -> list[dict[str, Any]]:
    if not resolver_path.exists():
        raise FileNotFoundError(f"UMLS inverse resolver not found: {resolver_path}")
    resolver = pd.read_excel(resolver_path)
    if "inverse" not in resolver.columns:
        raise ValueError(f"{resolver_path} must contain an 'inverse' column")
    inverse_predicates = {
        str(value)
        for value in resolver["inverse"].dropna()
        if str(value).strip()
    }
    return [record for record in graph if str(record.get("predicate", "")) not in inverse_predicates]


def _apply_frequency_filter(graph: list[dict[str, Any]], threshold: int) -> list[dict[str, Any]]:
    counts = Counter(str(record.get("predicate", "")) for record in graph)
    return [record for record in graph if counts[str(record.get("predicate", ""))] >= threshold]


def _add_semantic_type_edges(
    graph: list[dict[str, Any]], semantic_types_path: Path, relation: str = "isa"
) -> list[dict[str, Any]]:
    if not semantic_types_path.exists():
        raise FileNotFoundError(f"Semantic-type selection not found: {semantic_types_path}")
    selected = _read_graph(semantic_types_path)
    labels: dict[str, set[str]] = {}
    for record in graph:
        for field in ("subject", "object"):
            value = record.get(field)
            if isinstance(value, str) and value.strip():
                labels.setdefault(_normalize_text(value), set()).add(value)
    existing = {
        (_normalize_text(record.get("subject", "")), str(record.get("predicate", "")), str(record.get("object", "")))
        for record in graph
    }
    additions: list[dict[str, Any]] = []
    for item in selected:
        normalized_term = item.get("normalized_term")
        semantic_type = item.get("semantic_type")
        if not isinstance(normalized_term, str) or not isinstance(semantic_type, str):
            raise ValueError("Semantic-type records require normalized_term and semantic_type")
        originals = labels.get(normalized_term)
        if not originals:
            continue
        key = (normalized_term, relation, semantic_type)
        if key in existing:
            continue
        additions.append(
            {
                "subject": min(originals, key=lambda value: (_normalize_text(value), value)),
                "predicate": relation,
                "object": semantic_type,
                "synthetic": True,
                "source": "semantic_type_integrations",
                "anchor_id": item.get("anchor_id"),
                "normalized_term": normalized_term,
                "semantic_type_tui": item.get("semantic_type_tui"),
            }
        )
        existing.add(key)
    return [*graph, *additions]


def compute_graph_stats(graph: list[dict[str, Any]]) -> dict[str, Any]:
    """Return statistics in memory; this function never writes a summary file."""
    predicates = Counter(str(record.get("predicate", "")) for record in graph)
    nodes = {
        _normalize_text(record.get(field, ""))
        for record in graph
        for field in ("subject", "object")
        if isinstance(record.get(field), str) and record[field].strip()
    }
    return {
        "records": len(graph),
        "unique_nodes": len(nodes),
        "unique_relations": len(predicates),
        "relation_frequencies": dict(sorted(predicates.items())),
    }


def get_graph(
    kg_name: str,
    options: Iterable[str] | None = None,
    save: bool = False,
    with_stats: bool = False,
    datasets_dir: str | Path = DEFAULT_DATASETS_DIR,
    semantic_types_path: str | Path = DEFAULT_SEMANTIC_TYPES,
    inverse_resolver_path: str | Path = DEFAULT_UMLS_INVERSE_RESOLVER,
    frequency_threshold: int = FREQUENCY_THRESHOLD,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return one graph variant, optionally saving it and/or returning statistics.

    ``options=[]`` loads ``<kg_name>.json`` directly. Any non-raw option starts
    from ``<kg_name>_raw.json`` when available and applies all default treatments
    except those explicitly disabled by an option.
    """
    if not isinstance(kg_name, str) or not kg_name.strip():
        raise ValueError("kg_name must be a non-empty string")
    if frequency_threshold < 1:
        raise ValueError("frequency_threshold must be >= 1")
    kg_name = kg_name.strip()
    resolved_options = _normalise_options(kg_name, options)
    root = Path(datasets_dir)
    output_path = _variant_path(kg_name, resolved_options, root)

    if not resolved_options:
        if not output_path.exists():
            raise FileNotFoundError(f"Default graph does not exist: {output_path}")
        graph = _read_graph(output_path)
    elif resolved_options != (OPTION_RAW,) and output_path.exists():
        print(f"Variant already exists: {output_path}")
        graph = _read_graph(output_path)
    elif resolved_options == (OPTION_RAW,):
        graph = _load_raw_or_warn(kg_name, root)
    else:
        graph = _load_raw_or_warn(kg_name, root)
        if kg_name == UMLS_NCI_KG_NAME and OPTION_WITH_INVERSE not in resolved_options:
            graph = _remove_umls_inverse_relations(graph, Path(inverse_resolver_path))
        if OPTION_NO_FREQUENCY_FILTER not in resolved_options:
            graph = _apply_frequency_filter(graph, frequency_threshold)
        if OPTION_NO_SEMANTIC_TYPES not in resolved_options:
            graph = _add_semantic_type_edges(graph, Path(semantic_types_path))
        if save:
            _write_graph(output_path, graph)

    if with_stats:
        return graph, compute_graph_stats(graph)
    return graph
