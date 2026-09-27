"""Reproducible construction of configurable UMLS-NCI graph variants.

Database extraction, graph transformations, serialization, and statistics are
kept separate so the pipeline is importable, testable, and usable from a CLI.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    AbstractSet,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)


FILTER_RARE = "filter_rare"
FILTER_INVERSE = "filter_inverse"
INTEGRATE_SMT = "integrate_smt"

DEFAULT_TREATMENTS: Tuple[str, ...] = (
    FILTER_INVERSE,
    FILTER_RARE,
    INTEGRATE_SMT,
)

OPTION_KEEP_RARE = "kept_rare"
OPTION_KEEP_INVERSE = "kept_inverse"
OPTION_NO_SMT = "no_SMT"
SUPPORTED_OPTIONS = frozenset({OPTION_KEEP_RARE, OPTION_KEEP_INVERSE, OPTION_NO_SMT})


@dataclass(frozen=True)
class DatabaseConfig:
    host: str
    port: int
    user: str
    password: str
    database: str


@dataclass(frozen=True)
class BuildConfig:
    source: str = "NCI"
    umls_release: str = "unknown"
    min_relation_frequency: int = 50
    semantic_type_fraction: float = 0.01
    random_seed: int = 42
    fetch_size: int = 10_000
    metadata_chunk_size: int = 1_000
    options: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        unknown = set(self.options) - SUPPORTED_OPTIONS
        if unknown:
            raise ValueError(f"Unknown preprocessing options: {sorted(unknown)}")
        if self.min_relation_frequency < 1:
            raise ValueError("min_relation_frequency must be >= 1")
        if not 0.0 <= self.semantic_type_fraction <= 1.0:
            raise ValueError("semantic_type_fraction must be between 0 and 1")
        if self.fetch_size < 1 or self.metadata_chunk_size < 1:
            raise ValueError("fetch sizes must be >= 1")

    @property
    def treatments(self) -> Tuple[str, ...]:
        treatments = list(DEFAULT_TREATMENTS)
        if OPTION_KEEP_RARE in self.options:
            treatments.remove(FILTER_RARE)
        if OPTION_KEEP_INVERSE in self.options:
            treatments.remove(FILTER_INVERSE)
        if OPTION_NO_SMT in self.options:
            treatments.remove(INTEGRATE_SMT)
        return tuple(treatments)


@dataclass(frozen=True, order=True)
class SemanticType:
    tui: str
    name: str

    def to_json(self) -> dict:
        return {"tui": self.tui, "name": self.name}


@dataclass(frozen=True)
class ConceptMetadata:
    cui: str
    label: str
    semantic_types: Tuple[SemanticType, ...]

    @property
    def primary_type(self) -> Optional[str]:
        return self.semantic_types[0].name if self.semantic_types else None


@dataclass(frozen=True)
class EdgeRecord:
    subject_cui: str
    predicate: str
    general_relation: str
    object_cui: str


def inverse_relation_labels(
    rela_inverse: Mapping[str, str],
) -> frozenset[str]:
    """Choose one reproducible inverse label from every MRDOC inverse pair.

    MRDOC documents both directions but does not mark either member as the
    preferred direction. The notebook resolver consistently classifies the
    lexicographically smaller RELA as inverse, so the same deterministic rule
    is applied directly to the MRDOC pairs.
    """

    return frozenset(
        min(relation, inverse)
        for relation, inverse in rela_inverse.items()
        if relation and inverse
    )


def resolve_treatments(options: Sequence[str] = ()) -> Tuple[str, ...]:
    """Resolve user-facing skip options to the applied treatment list."""

    return BuildConfig(options=tuple(options)).treatments


def _chunks(values: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


class SqliteEdgeStore:
    """Disk-backed relation store used to deduplicate large MRREL extracts."""

    def __init__(self, path: Path):
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.executescript(
            """
            CREATE TABLE edges (
                subject_cui TEXT NOT NULL,
                predicate TEXT NOT NULL,
                general_relation TEXT NOT NULL,
                object_cui TEXT NOT NULL,
                PRIMARY KEY (
                    subject_cui, predicate, general_relation, object_cui
                )
            ) WITHOUT ROWID;

            CREATE TABLE nodes (
                cui TEXT PRIMARY KEY
            ) WITHOUT ROWID;

            CREATE TABLE valid_nodes (
                cui TEXT PRIMARY KEY
            ) WITHOUT ROWID;
            """
        )
        self.input_rows = 0
        self.inserted_rows = 0

    def add_many(self, edges: Iterable[EdgeRecord]) -> None:
        edge_rows = []
        node_rows = []
        for edge in edges:
            self.input_rows += 1
            edge_rows.append(
                (
                    edge.subject_cui,
                    edge.predicate,
                    edge.general_relation or "",
                    edge.object_cui,
                )
            )
            node_rows.extend(((edge.subject_cui,), (edge.object_cui,)))

        before = self.connection.total_changes
        self.connection.executemany(
            "INSERT OR IGNORE INTO edges VALUES (?, ?, ?, ?)", edge_rows
        )
        self.inserted_rows += self.connection.total_changes - before
        self.connection.executemany("INSERT OR IGNORE INTO nodes VALUES (?)", node_rows)
        self.connection.commit()

    def finalize(self) -> None:
        self.connection.executescript(
            """
            CREATE INDEX IF NOT EXISTS edges_predicate_idx
                ON edges(predicate);
            CREATE INDEX IF NOT EXISTS edges_object_idx
                ON edges(object_cui);
            """
        )

    def node_ids(self) -> list[str]:
        return [
            row[0]
            for row in self.connection.execute("SELECT cui FROM nodes ORDER BY cui")
        ]

    def set_valid_nodes(self, cuis: Iterable[str]) -> None:
        self.connection.executemany(
            "INSERT OR IGNORE INTO valid_nodes VALUES (?)",
            ((cui,) for cui in cuis),
        )
        self.connection.commit()

    def relation_counts(self) -> Counter:
        rows = self.connection.execute(
            """
            SELECT e.predicate, COUNT(*)
            FROM edges e
            JOIN valid_nodes s ON s.cui = e.subject_cui
            JOIN valid_nodes o ON o.cui = e.object_cui
            GROUP BY e.predicate
            """
        )
        return Counter({predicate: count for predicate, count in rows})

    def count_valid_edges(self) -> int:
        return self.connection.execute(
            """
            SELECT COUNT(*)
            FROM edges e
            JOIN valid_nodes s ON s.cui = e.subject_cui
            JOIN valid_nodes o ON o.cui = e.object_cui
            """
        ).fetchone()[0]

    def iter_edges(
        self, allowed_relations: Optional[set[str]] = None
    ) -> Iterator[EdgeRecord]:
        relation_join = ""
        if allowed_relations is not None:
            self.connection.execute("DROP TABLE IF EXISTS allowed_relations")
            self.connection.execute(
                "CREATE TEMP TABLE allowed_relations "
                "(predicate TEXT PRIMARY KEY) WITHOUT ROWID"
            )
            self.connection.executemany(
                "INSERT INTO allowed_relations VALUES (?)",
                ((predicate,) for predicate in sorted(allowed_relations)),
            )
            relation_join = "JOIN allowed_relations a ON a.predicate = e.predicate"

        query = f"""
            SELECT e.subject_cui, e.predicate,
                   e.general_relation, e.object_cui
            FROM edges e
            JOIN valid_nodes s ON s.cui = e.subject_cui
            JOIN valid_nodes o ON o.cui = e.object_cui
            {relation_join}
            ORDER BY e.subject_cui, e.predicate,
                     e.object_cui, e.general_relation
        """
        for row in self.connection.execute(query):
            yield EdgeRecord(*row)

    def close(self) -> None:
        self.connection.close()


class UMLSRepository:
    """Read-only access to a MySQL installation of the UMLS RRF tables."""

    def __init__(self, connection, source: str, fetch_size: int = 10_000):
        self.connection = connection
        self.source = source
        self.fetch_size = fetch_size

    def inverse_maps(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        cursor = self.connection.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT DOCKEY AS dockey, VALUE AS value, EXPL AS inverse_value
            FROM mrdoc
            WHERE (DOCKEY = 'RELA' AND TYPE = 'rela_inverse')
               OR (DOCKEY = 'REL' AND TYPE = 'rel_inverse')
            ORDER BY DOCKEY, VALUE, EXPL
            """
        )
        rela_inverse: Dict[str, str] = {}
        rel_inverse: Dict[str, str] = {}
        for row in cursor:
            target = rela_inverse if row["dockey"] == "RELA" else rel_inverse
            value = row["value"]
            inverse = row["inverse_value"]
            if value and inverse:
                target[value] = inverse
                target.setdefault(inverse, value)
        cursor.close()
        return rela_inverse, rel_inverse

    def extract_edges(
        self,
        store: SqliteEdgeStore,
        excluded_relations: AbstractSet[str] = frozenset(),
    ) -> dict:
        cursor = self.connection.cursor(dictionary=True, buffered=False)
        cursor.execute(
            """
            SELECT CUI2 AS subject_cui,
                   RELA AS predicate,
                   COALESCE(REL, '') AS general_relation,
                   CUI1 AS object_cui
            FROM mrrel
            WHERE SAB = %s
              AND RELA IS NOT NULL
            """,
            (self.source,),
        )

        rows_read = 0
        rows_excluded = 0
        while True:
            rows = cursor.fetchmany(self.fetch_size)
            if not rows:
                break
            batch = []
            for row in rows:
                rows_read += 1
                if row["predicate"] in excluded_relations:
                    rows_excluded += 1
                    continue
                edge = EdgeRecord(
                    subject_cui=row["subject_cui"],
                    predicate=row["predicate"],
                    general_relation=row["general_relation"] or "",
                    object_cui=row["object_cui"],
                )
                batch.append(edge)
            store.add_many(batch)
        cursor.close()
        store.finalize()
        return {
            "rows_read": rows_read,
            "rows_excluded": rows_excluded,
        }

    def concept_metadata(
        self, cuis: Sequence[str], chunk_size: int = 1_000
    ) -> Dict[str, ConceptMetadata]:
        labels: Dict[str, str] = {}
        types: Dict[str, set[SemanticType]] = defaultdict(set)

        for chunk in _chunks(cuis, chunk_size):
            placeholders = ",".join(["%s"] * len(chunk))
            label_cursor = self.connection.cursor(dictionary=True)
            label_cursor.execute(
                f"""
                SELECT CUI AS cui, STR AS term
                FROM mrconso
                WHERE CUI IN ({placeholders})
                  AND LAT = 'ENG'
                  AND TTY = 'PT'
                """,
                tuple(chunk),
            )
            candidates: Dict[str, list[Mapping[str, object]]] = defaultdict(list)
            for row in label_cursor:
                candidates[row["cui"]].append(row)
            label_cursor.close()
            for cui, rows in candidates.items():
                # Match the notebook's PT/ENG subquery with LIMIT 1: retain
                # the first row returned for each CUI, without source ranking.
                labels[cui] = str(rows[0]["term"])

            type_cursor = self.connection.cursor(dictionary=True)
            type_cursor.execute(
                f"""
                SELECT DISTINCT CUI AS cui, TUI AS tui, STY AS type_name
                FROM mrsty
                WHERE CUI IN ({placeholders})
                ORDER BY CUI, TUI, STY
                """,
                tuple(chunk),
            )
            for row in type_cursor:
                types[row["cui"]].add(
                    SemanticType(tui=row["tui"], name=row["type_name"])
                )
            type_cursor.close()

        return {
            cui: ConceptMetadata(
                cui=cui,
                label=label,
                semantic_types=tuple(sorted(types.get(cui, set()))),
            )
            for cui, label in labels.items()
        }


class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}
        self.size: Dict[str, int] = {}

    def add(self, item: str) -> None:
        if item not in self.parent:
            self.parent[item] = item
            self.size[item] = 1

    def find(self, item: str) -> str:
        parent = self.parent[item]
        while parent != self.parent[parent]:
            parent = self.parent[parent]
        while item != parent:
            next_item = self.parent[item]
            self.parent[item] = parent
            item = next_item
        return parent

    def union(self, left: str, right: str) -> None:
        self.add(left)
        self.add(right)
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.size[root_left] < self.size[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        self.size[root_left] += self.size[root_right]

    def component_sizes(self) -> list[int]:
        counts = Counter(self.find(item) for item in self.parent)
        return sorted(counts.values(), reverse=True)


class GraphStatistics:
    def __init__(self) -> None:
        self.edges = 0
        self.self_loops = 0
        self.synthetic_isa_edges = 0
        self.relations: Counter = Counter()
        self.in_degree: Counter = Counter()
        self.out_degree: Counter = Counter()
        self.nodes: set[str] = set()
        self.concept_cuis: set[str] = set()
        self.semantic_type_nodes: set[str] = set()
        self.union_find = UnionFind()

    def add_edge(
        self,
        subject_id: str,
        predicate: str,
        object_id: str,
        subject_cui: Optional[str],
        object_cui: Optional[str],
        synthetic: bool,
    ) -> None:
        self.edges += 1
        self.relations[predicate] += 1
        self.nodes.update((subject_id, object_id))
        self.out_degree[subject_id] += 1
        self.in_degree[object_id] += 1
        self.union_find.union(subject_id, object_id)
        if subject_id == object_id:
            self.self_loops += 1
        if synthetic and predicate == "isa":
            self.synthetic_isa_edges += 1
        if subject_cui:
            self.concept_cuis.add(subject_cui)
        else:
            self.semantic_type_nodes.add(subject_id)
        if object_cui:
            self.concept_cuis.add(object_cui)
        else:
            self.semantic_type_nodes.add(object_id)

    @staticmethod
    def _degree_summary(nodes: Iterable[str], degrees: Mapping[str, int]) -> dict:
        values = [degrees.get(node, 0) for node in nodes]
        if not values:
            return {"min": 0, "max": 0, "mean": 0.0}
        return {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }

    def as_dict(self, metadata: Mapping[str, ConceptMetadata]) -> dict:
        components = self.union_find.component_sizes()
        node_count = len(self.nodes)
        type_frequency: Counter = Counter()
        for cui in self.concept_cuis:
            concept = metadata.get(cui)
            if concept:
                for semantic_type in concept.semantic_types:
                    type_frequency[semantic_type.name] += 1

        density_denominator = node_count * (node_count - 1)
        return {
            "nodes": node_count,
            "concept_nodes": len(self.concept_cuis),
            "semantic_type_nodes": len(self.semantic_type_nodes),
            "edges": self.edges,
            "relation_types": len(self.relations),
            "self_loops": self.self_loops,
            "synthetic_isa_edges": self.synthetic_isa_edges,
            "weakly_connected_components": len(components),
            "largest_component_nodes": components[0] if components else 0,
            "largest_component_ratio": (
                components[0] / node_count if components and node_count else 0.0
            ),
            "directed_density": (
                self.edges / density_denominator if density_denominator else 0.0
            ),
            "in_degree": self._degree_summary(self.nodes, self.in_degree),
            "out_degree": self._degree_summary(self.nodes, self.out_degree),
            "relation_frequencies": dict(
                sorted(self.relations.items(), key=lambda item: (-item[1], item[0]))
            ),
            "semantic_type_frequencies": dict(
                sorted(type_frequency.items(), key=lambda item: (-item[1], item[0]))
            ),
        }


class JsonArrayWriter:
    """Stream a valid JSON array without retaining the graph in memory."""

    def __init__(self, path: Path):
        self.path = path
        self.handle = None
        self.first = True

    def __enter__(self) -> "JsonArrayWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8", newline="\n")
        self.handle.write("[\n")
        return self

    def write(self, record: Mapping[str, object]) -> None:
        if not self.first:
            self.handle.write(",\n")
        json.dump(record, self.handle, ensure_ascii=False, separators=(",", ":"))
        self.first = False

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.handle is not None:
            if exc_type is None:
                self.handle.write("\n]\n")
            self.handle.close()


def _concept_edge_to_json(
    edge: EdgeRecord,
    metadata: Mapping[str, ConceptMetadata],
    source: str,
) -> dict:
    subject = metadata[edge.subject_cui]
    obj = metadata[edge.object_cui]
    return {
        "subject": subject.label,
        "predicate": edge.predicate,
        "object": obj.label,
        "subject_cui": subject.cui,
        "object_cui": obj.cui,
        "subject_type": subject.primary_type,
        "object_type": obj.primary_type,
        "subject_types": [item.to_json() for item in subject.semantic_types],
        "object_types": [item.to_json() for item in obj.semantic_types],
        "general_relation": edge.general_relation or None,
        "source": source,
        "synthetic": False,
    }


def _semantic_type_edge_to_json(
    concept: ConceptMetadata,
    semantic_type: SemanticType,
) -> dict:
    return {
        "subject": concept.label,
        "predicate": "isa",
        "object": semantic_type.name,
        "subject_cui": concept.cui,
        "object_cui": None,
        "object_tui": semantic_type.tui,
        "subject_type": concept.primary_type,
        "object_type": "Semantic Type",
        "subject_types": [item.to_json() for item in concept.semantic_types],
        "object_types": [],
        "general_relation": "isa",
        "source": "UMLS_SEMANTIC_NETWORK",
        "synthetic": True,
    }


def sample_semantic_type_pairs(
    ordered_cuis: Iterable[str],
    metadata: Mapping[str, ConceptMetadata],
    fraction: float,
    seed: int,
) -> list[Tuple[str, SemanticType]]:
    """Reproduce the notebook's term/type sampling for synthetic isa edges.

    The notebook keeps one semantic type per term, removes duplicate
    ``(term, type)`` pairs, then calls ``sample(random_state=seed)`` separately
    for every type. A representative CUI is retained here so the enriched JSON
    does not lose concept identity.
    """

    if fraction <= 0:
        return []

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "numpy is required to reproduce the notebook's semantic-type " "sampling"
        ) from exc

    by_type: Dict[SemanticType, list[str]] = defaultdict(list)
    seen_term_types: set[Tuple[str, SemanticType]] = set()
    for cui in ordered_cuis:
        concept = metadata[cui]
        if not concept.semantic_types:
            continue
        semantic_type = concept.semantic_types[0]
        term_type = (concept.label, semantic_type)
        if term_type in seen_term_types:
            continue
        seen_term_types.add(term_type)
        by_type[semantic_type].append(cui)

    sampled: list[Tuple[str, SemanticType]] = []
    for semantic_type in sorted(by_type):
        candidates = by_type[semantic_type]
        sample_size = max(1, math.floor(len(candidates) * fraction))
        sample_size = min(sample_size, len(candidates))
        # pandas.DataFrame.sample(random_state=<int>) creates the same legacy
        # NumPy RandomState afresh for each group in the original notebook.
        positions = np.random.RandomState(seed).choice(
            len(candidates), size=sample_size, replace=False
        )
        for cui in sorted(candidates[int(position)] for position in positions):
            sampled.append((cui, semantic_type))
    return sorted(sampled, key=lambda item: (item[0], item[1]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def connect_mysql(config: DatabaseConfig):
    """Create a MySQL connection lazily so pure transformations need no driver."""

    try:
        import mysql.connector
    except ImportError as exc:
        raise RuntimeError(
            "mysql-connector-python is required for UMLS extraction. "
            "Install it with: pip install mysql-connector-python"
        ) from exc

    return mysql.connector.connect(
        host=config.host,
        port=config.port,
        user=config.user,
        password=config.password,
        database=config.database,
        charset="utf8mb4",
        use_unicode=True,
    )


def build_nci_graph(
    database_config: DatabaseConfig,
    output_path: str | Path,
    build_config: Optional[BuildConfig] = None,
    summary_path: Optional[str | Path] = None,
) -> dict:
    """Build one NCI graph variant and write graph and summary JSON files."""

    config = build_config or BuildConfig()
    output = Path(output_path)
    summary_output = (
        Path(summary_path)
        if summary_path is not None
        else output.with_suffix(".summary.json")
    )
    started_at = datetime.now(timezone.utc)
    connection = connect_mysql(database_config)

    try:
        repository = UMLSRepository(
            connection=connection,
            source=config.source,
            fetch_size=config.fetch_size,
        )
        with tempfile.TemporaryDirectory(prefix="tdg_umls_nci_") as temp_dir:
            store = SqliteEdgeStore(Path(temp_dir) / "relations.sqlite3")
            try:
                if FILTER_INVERSE in config.treatments:
                    rela_inverse, rel_inverse = repository.inverse_maps()
                    excluded_inverse_relations = inverse_relation_labels(rela_inverse)
                else:
                    rela_inverse, rel_inverse = {}, {}
                    excluded_inverse_relations = frozenset()

                extraction = repository.extract_edges(
                    store=store,
                    excluded_relations=excluded_inverse_relations,
                )

                all_cuis = store.node_ids()
                metadata = repository.concept_metadata(
                    all_cuis, chunk_size=config.metadata_chunk_size
                )
                store.set_valid_nodes(metadata)

                relation_counts_before_filter = store.relation_counts()
                edges_with_labels = store.count_valid_edges()
                allowed_relations: Optional[set[str]] = None
                if FILTER_RARE in config.treatments:
                    allowed_relations = {
                        relation
                        for relation, count in relation_counts_before_filter.items()
                        if count >= config.min_relation_frequency
                    }

                statistics = GraphStatistics()
                subject_cuis: Dict[str, None] = {}
                object_cuis: Dict[str, None] = {}
                base_edge_count = 0
                semantic_type_pairs: list[Tuple[str, SemanticType]] = []
                with JsonArrayWriter(output) as writer:
                    for edge in store.iter_edges(allowed_relations):
                        writer.write(
                            _concept_edge_to_json(edge, metadata, config.source)
                        )
                        statistics.add_edge(
                            subject_id=f"CUI:{edge.subject_cui}",
                            predicate=edge.predicate,
                            object_id=f"CUI:{edge.object_cui}",
                            subject_cui=edge.subject_cui,
                            object_cui=edge.object_cui,
                            synthetic=False,
                        )
                        subject_cuis.setdefault(edge.subject_cui, None)
                        object_cuis.setdefault(edge.object_cui, None)
                        base_edge_count += 1

                    if INTEGRATE_SMT in config.treatments:
                        semantic_type_pairs = sample_semantic_type_pairs(
                            ordered_cuis=(
                                *subject_cuis.keys(),
                                *object_cuis.keys(),
                            ),
                            metadata=metadata,
                            fraction=config.semantic_type_fraction,
                            seed=config.random_seed,
                        )
                        for cui, semantic_type in semantic_type_pairs:
                            writer.write(
                                _semantic_type_edge_to_json(
                                    metadata[cui], semantic_type
                                )
                            )
                            statistics.add_edge(
                                subject_id=f"CUI:{cui}",
                                predicate="isa",
                                object_id=f"TUI:{semantic_type.tui}",
                                subject_cui=cui,
                                object_cui=None,
                                synthetic=True,
                            )

                finished_at = datetime.now(timezone.utc)
                graph_statistics = statistics.as_dict(metadata)
                effective_relations = (
                    allowed_relations
                    if allowed_relations is not None
                    else set(relation_counts_before_filter)
                )
                summary = {
                    "schema_version": "1.0",
                    "created_at_utc": finished_at.isoformat(),
                    "duration_seconds": (finished_at - started_at).total_seconds(),
                    "graph": {
                        "path": str(output),
                        "sha256": _sha256(output),
                        "size_bytes": output.stat().st_size,
                    },
                    "source": {
                        "vocabulary": config.source,
                        "umls_release": config.umls_release,
                        "database": database_config.database,
                        "host": database_config.host,
                        "port": database_config.port,
                    },
                    "configuration": {
                        "default_treatments": list(DEFAULT_TREATMENTS),
                        "options": list(config.options),
                        "applied_treatments": list(config.treatments),
                        "base_selection": {
                            "mrrel_source": config.source,
                            "require_rela": True,
                            "label_language": "ENG",
                            "label_term_type": "PT",
                            "label_selection": (
                                "first_returned_label_per_cui_like_notebook_limit_1"
                            ),
                            "primary_semantic_type": "first_tui",
                        },
                        "inverse_filter_strategy": (
                            "keep_lexicographic_max_rela_per_mrdoc_pair"
                            if FILTER_INVERSE in config.treatments
                            else None
                        ),
                        "min_relation_frequency": config.min_relation_frequency,
                        "semantic_type_fraction": config.semantic_type_fraction,
                        "random_seed": config.random_seed,
                        "semantic_type_sampling_strategy": (
                            "notebook_primary_type_per_unique_term_and_type"
                            if INTEGRATE_SMT in config.treatments
                            else None
                        ),
                    },
                    "stages": {
                        "mrrel_rows_read": extraction["rows_read"],
                        "inverse_relation_rows_removed": extraction["rows_excluded"],
                        "rows_after_inverse_filter": store.input_rows,
                        "unique_edges_after_deduplication": store.inserted_rows,
                        "documented_inverse_rela_values": len(rela_inverse),
                        "documented_inverse_rel_values": len(rel_inverse),
                        "inverse_rela_values_filtered": len(excluded_inverse_relations),
                        "duplicate_edges_removed": (
                            store.input_rows - store.inserted_rows
                        ),
                        "concepts_seen": len(all_cuis),
                        "concepts_with_english_label": len(metadata),
                        "concepts_without_english_label": (
                            len(all_cuis) - len(metadata)
                        ),
                        "edges_after_label_validation": edges_with_labels,
                        "relations_before_frequency_filter": len(
                            relation_counts_before_filter
                        ),
                        "relations_removed_by_frequency": sorted(
                            set(relation_counts_before_filter) - effective_relations
                        ),
                        "base_edges_written": base_edge_count,
                        "semantic_type_edges_added": len(semantic_type_pairs),
                    },
                    "statistics": graph_statistics,
                }

                summary_output.parent.mkdir(parents=True, exist_ok=True)
                with summary_output.open("w", encoding="utf-8") as handle:
                    json.dump(summary, handle, ensure_ascii=False, indent=2)
                    handle.write("\n")
                summary["summary_path"] = str(summary_output)
                return summary
            finally:
                store.close()
    finally:
        connection.close()
