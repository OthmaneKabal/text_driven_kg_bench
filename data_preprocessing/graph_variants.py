"""Load or construct graph variants from raw JSON graphs.

The module intentionally performs only three generic transformations:
frequency filtering, semantic-type integration, and (for UMLS-NCI only)
inverse-relation removal. It never calls an LLM or runs source-specific
predicate normalization.
"""

from __future__ import annotations

import base64
import json
import unicodedata
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote
from xml.sax.saxutils import escape, quoteattr

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
VALID_FORMATS = frozenset({"json", "rdf"})

RDF_NAMESPACE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS_NAMESPACE = "http://www.w3.org/2000/01/rdf-schema#"
RDF_ENTITY_NAMESPACE = "https://tdg-bench.org/resource/entity/"
RDF_RELATION_NAMESPACE = "https://tdg-bench.org/resource/relation/"


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


def _rdf_path(kg_name: str, options: tuple[str, ...], datasets_dir: Path) -> Path:
    """Return the RDF/XML path paired with a JSON graph variant path."""
    return _variant_path(kg_name, options, datasets_dir).with_suffix(".rdf")


def _rdf_entity_iri(value: str) -> str:
    return f"{RDF_ENTITY_NAMESPACE}{quote(value, safe='-._~')}"


def _rdf_predicate_local_name(predicate: str) -> str:
    """Produce an XML-safe, reversible local name for an arbitrary predicate."""
    token = base64.urlsafe_b64encode(predicate.encode("utf-8")).decode("ascii").rstrip("=")
    return f"p_{token or 'empty'}"


def _write_rdf_xml(path: Path, graph: list[dict[str, Any]]) -> None:
    """Write graph records as RDF/XML while preserving each original SPO label.

    Subjects and objects are RDF resources. Predicates use deterministic IRIs
    in ``RDF_RELATION_NAMESPACE``; every generated resource has an
    ``rdfs:label`` carrying its original JSON value, so source labels remain
    recoverable. RDF is a set of triples: identical JSON SPO records are
    represented by the same RDF triple.
    """
    entities: dict[str, str] = {}
    relations: dict[str, str] = {}
    edges: list[tuple[str, str, str]] = []
    for index, record in enumerate(graph):
        subject = record.get("subject")
        predicate = record.get("predicate")
        obj = record.get("object")
        if not all(isinstance(value, str) for value in (subject, predicate, obj)):
            raise ValueError(
                f"Cannot export record {index} to RDF: subject, predicate and object must be strings"
            )
        subject_iri = _rdf_entity_iri(subject)
        object_iri = _rdf_entity_iri(obj)
        predicate_local = _rdf_predicate_local_name(predicate)
        entities[subject_iri] = subject
        entities[object_iri] = obj
        relations[predicate_local] = predicate
        edges.append((subject_iri, predicate_local, object_iri))

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        handle.write(
            f'<rdf:RDF xmlns:rdf="{RDF_NAMESPACE}" '
            f'xmlns:rdfs="{RDFS_NAMESPACE}" '
            f'xmlns:tdgr="{RDF_RELATION_NAMESPACE}">\n'
        )
        for subject_iri, predicate_local, object_iri in edges:
            handle.write(f'  <rdf:Description rdf:about={quoteattr(subject_iri)}>\n')
            handle.write(
                f'    <tdgr:{predicate_local} rdf:resource={quoteattr(object_iri)}/>\n'
            )
            handle.write("  </rdf:Description>\n")
        for entity_iri, label in entities.items():
            handle.write(f'  <rdf:Description rdf:about={quoteattr(entity_iri)}>\n')
            handle.write(f"    <rdfs:label>{escape(label)}</rdfs:label>\n")
            handle.write("  </rdf:Description>\n")
        for predicate_local, label in relations.items():
            predicate_iri = f"{RDF_RELATION_NAMESPACE}{predicate_local}"
            handle.write(f'  <rdf:Description rdf:about={quoteattr(predicate_iri)}>\n')
            handle.write(f"    <rdfs:label>{escape(label)}</rdfs:label>\n")
            handle.write("  </rdf:Description>\n")
        handle.write("</rdf:RDF>\n")
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
    format: str = "json",
    with_stats: bool = False,
    datasets_dir: str | Path = DEFAULT_DATASETS_DIR,
    semantic_types_path: str | Path = DEFAULT_SEMANTIC_TYPES,
    inverse_resolver_path: str | Path = DEFAULT_UMLS_INVERSE_RESOLVER,
    frequency_threshold: int = FREQUENCY_THRESHOLD,
) -> list[dict[str, Any]] | Path | tuple[list[dict[str, Any]] | Path, dict[str, Any]]:
    """Return one graph variant in JSON records or persist it as RDF/XML.

    ``options=[]`` loads ``<kg_name>.json`` directly. Any non-raw option starts
    from ``<kg_name>_raw.json`` when available and applies all default treatments
    except those explicitly disabled by an option.

    ``format="json"`` (default) returns JSON records and, with ``save=True``,
    writes ``<variant>.json``. ``format="rdf"`` requires ``save=True`` and
    writes/returns the paired ``<variant>.rdf`` RDF/XML file. RDF resources are
    labelled with their original JSON values. Identical JSON SPO records map to
    one RDF triple because RDF graphs have set semantics.
    """
    if not isinstance(kg_name, str) or not kg_name.strip():
        raise ValueError("kg_name must be a non-empty string")
    if frequency_threshold < 1:
        raise ValueError("frequency_threshold must be >= 1")
    if format not in VALID_FORMATS:
        raise ValueError(f"Unknown graph format '{format}'. Supported formats: {sorted(VALID_FORMATS)}")
    if format == "rdf" and not save:
        raise ValueError("format='rdf' writes an RDF/XML file; set save=True")
    kg_name = kg_name.strip()
    resolved_options = _normalise_options(kg_name, options)
    root = Path(datasets_dir)
    output_path = _variant_path(kg_name, resolved_options, root)
    rdf_output_path = _rdf_path(kg_name, resolved_options, root)

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
        if save and format == "json":
            _write_graph(output_path, graph)

    result: list[dict[str, Any]] | Path
    if format == "rdf":
        if rdf_output_path.exists():
            print(f"RDF variant already exists: {rdf_output_path}")
        else:
            _write_rdf_xml(rdf_output_path, graph)
        result = rdf_output_path
    else:
        result = graph
    if with_stats:
        return result, compute_graph_stats(graph)
    return result
