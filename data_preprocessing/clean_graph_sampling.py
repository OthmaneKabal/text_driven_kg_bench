"""Anchor-guided Frontier Sampling for scale-matched clean graph references.

The sampler keeps common concepts and semantic-type anchors, grows an induced
subgraph with coordinated random walks, and retains the candidate whose
structure is closest to the complete clean graph.  It does not alter labels or
predicates and never calls an LLM.
"""

from __future__ import annotations

import json
import math
import random
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMMON_NODES = ROOT / "datasets" / "common_nodes.xlsx"
DEFAULT_SEMANTIC_TYPES = (
    ROOT / "datasets"
    / "semantic_type_integrations"
    / "umls_nci_clean_1pct_common_terms.json"
)


def normalize_term(value: object) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def _read_records(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    with input_path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{input_path} must be a JSON list of objects")
    return records


def _write_records(path: str | Path, records: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(output_path)


def load_anchor_terms(
    common_nodes_path: str | Path = DEFAULT_COMMON_NODES,
    semantic_types_path: str | Path = DEFAULT_SEMANTIC_TYPES,
) -> set[str]:
    """Return common concepts, selected concepts, and semantic-type nodes."""
    common_nodes = pd.read_excel(common_nodes_path)
    if "term" not in common_nodes.columns:
        raise ValueError(f"{common_nodes_path} must contain a 'term' column")
    anchors = {
        normalize_term(value)
        for value in common_nodes["term"].dropna()
        if str(value).strip()
    }
    for item in _read_records(semantic_types_path):
        for field in ("normalized_term", "semantic_type"):
            value = item.get(field)
            if isinstance(value, str) and value.strip():
                anchors.add(normalize_term(value))
    return anchors


def _build_adjacency(
    records: Iterable[dict[str, Any]],
) -> tuple[dict[str, tuple[tuple[str, int], ...]], set[str]]:
    adjacency_edges: dict[str, dict[str, int]] = {}
    nodes: set[str] = set()
    for index, record in enumerate(records):
        subject, obj = record.get("subject"), record.get("object")
        if not isinstance(subject, str) or not subject.strip():
            continue
        if not isinstance(obj, str) or not obj.strip():
            continue
        subject_id, object_id = normalize_term(subject), normalize_term(obj)
        nodes.update((subject_id, object_id))
        adjacency_edges.setdefault(subject_id, {}).setdefault(object_id, index)
        adjacency_edges.setdefault(object_id, {}).setdefault(subject_id, index)
    return (
        {node: tuple(neighbors.items()) for node, neighbors in adjacency_edges.items()},
        nodes,
    )


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.size: dict[str, int] = {}

    def add(self, node: str) -> None:
        if node not in self.parent:
            self.parent[node] = node
            self.size[node] = 1

    def find(self, node: str) -> str:
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while node != root:
            parent = self.parent[node]
            self.parent[node] = root
            node = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


def _distribution_distance(left: Counter[str | int], right: Counter[str | int]) -> float:
    """Jensen-Shannon distance between two count distributions."""
    keys = set(left) | set(right)
    left_total, right_total = sum(left.values()), sum(right.values())
    if not keys or not left_total or not right_total:
        return 0.0 if left_total == right_total else 1.0
    divergence = 0.0
    for key in keys:
        p, q = left[key] / left_total, right[key] / right_total
        midpoint = (p + q) / 2
        if p:
            divergence += 0.5 * p * math.log2(p / midpoint)
        if q:
            divergence += 0.5 * q * math.log2(q / midpoint)
    return math.sqrt(divergence)


def _record_types(record: dict[str, Any], endpoint: str) -> list[str]:
    types = record.get(f"{endpoint}_types")
    if isinstance(types, list):
        names = [item.get("name") for item in types if isinstance(item, dict)]
        return [name for name in names if isinstance(name, str) and name]
    value = record.get(f"{endpoint}_type")
    return [value] if isinstance(value, str) and value else []


def graph_profile(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Return normalized structural and schema distributions for comparison."""
    union_find = _UnionFind()
    degree: Counter[str] = Counter()
    predicates: Counter[str] = Counter()
    semantic_types: Counter[str] = Counter()
    edges = 0
    for record in records:
        edges += 1
        predicates[str(record.get("predicate", ""))] += 1
        semantic_types.update(_record_types(record, "subject"))
        semantic_types.update(_record_types(record, "object"))
        subject, obj = record.get("subject"), record.get("object")
        if not isinstance(subject, str) or not subject.strip() or not isinstance(obj, str) or not obj.strip():
            continue
        subject_id, object_id = normalize_term(subject), normalize_term(obj)
        union_find.add(subject_id)
        union_find.add(object_id)
        union_find.union(subject_id, object_id)
        degree[subject_id] += 1
        degree[object_id] += 1

    component_sizes: Counter[str] = Counter(union_find.find(node) for node in union_find.parent)
    degree_bins: Counter[int] = Counter(int(math.log2(value)) for value in degree.values() if value > 0)
    nodes = len(union_find.parent)
    # Component sizes must be compared after scaling: a graph reduced to one
    # fifth of its nodes can still preserve a single giant component.
    component_bins: Counter[int] = Counter(
        int(math.floor(math.log2(value / nodes)))
        for value in component_sizes.values()
        if value > 0 and nodes > 0
    )
    return {
        "nodes": nodes,
        "edges": edges,
        "unique_relations": len(predicates),
        "mean_undirected_degree": (sum(degree.values()) / nodes) if nodes else 0.0,
        "weakly_connected_components": len(component_sizes),
        "largest_component_fraction": (max(component_sizes.values()) / nodes) if nodes else 0.0,
        "predicate_distribution": predicates,
        "semantic_type_distribution": semantic_types,
        "degree_distribution": degree_bins,
        "component_size_distribution": component_bins,
    }


def structural_distance(candidate: dict[str, Any], reference: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """Equal-weight structural distance; lower is more similar."""
    parts = {
        "degree": _distribution_distance(candidate["degree_distribution"], reference["degree_distribution"]),
        "components": _distribution_distance(
            candidate["component_size_distribution"], reference["component_size_distribution"]
        ),
        "relations": _distribution_distance(
            candidate["predicate_distribution"], reference["predicate_distribution"]
        ),
        "semantic_types": _distribution_distance(
            candidate["semantic_type_distribution"], reference["semantic_type_distribution"]
        ),
        "largest_component_fraction": abs(
            candidate["largest_component_fraction"] - reference["largest_component_fraction"]
        ),
    }
    return sum(parts.values()), parts


def _bfs_tree(
    adjacency: dict[str, tuple[tuple[str, int], ...]], root: str
) -> dict[str, tuple[str, int] | None]:
    parents: dict[str, tuple[str, int] | None] = {root: None}
    queue = [root]
    for node in queue:
        for neighbor, edge_index in adjacency.get(node, ()):
            if neighbor not in parents:
                parents[neighbor] = (node, edge_index)
                queue.append(neighbor)
    return parents


def _connected_core(
    required_nodes: set[str], root: str, parents: dict[str, tuple[str, int] | None]
) -> tuple[set[str], set[int]]:
    """Join all mandatory nodes to one root through a fixed BFS tree."""
    missing = required_nodes - set(parents)
    if missing:
        raise ValueError(f"{len(missing)} mandatory nodes are outside the root component")
    nodes = {root}
    tree_edges: set[int] = set()
    for node in required_nodes:
        current = node
        while current != root:
            parent = parents[current]
            assert parent is not None
            previous, edge_index = parent
            nodes.update((current, previous))
            tree_edges.add(edge_index)
            current = previous
    return nodes, tree_edges


def _frontier_nodes(
    adjacency: dict[str, tuple[tuple[str, int], ...]],
    required_nodes: set[str],
    tree_edges: set[int],
    target_nodes: int,
    walkers: int,
    rng: random.Random,
) -> tuple[set[str], set[int]]:
    """Grow one connected component with coordinated frontier walks."""
    selected = set(required_nodes)
    anchor_frontier = [node for node in sorted(selected) if adjacency.get(node)]
    # Every anchor remains selected.  The active frontier is capped so a large
    # annotation set does not turn each Frontier Sampling step into thousands
    # of weighted walker updates.
    frontier = (
        rng.sample(anchor_frontier, walkers)
        if len(anchor_frontier) > walkers
        else anchor_frontier
    )
    if not frontier:
        raise ValueError("Connected mandatory core has no traversable node")
    stalled_steps = 0
    while len(selected) < target_nodes:
        weights = [len(adjacency[node]) for node in frontier]
        walker_index = rng.choices(range(len(frontier)), weights=weights, k=1)[0]
        neighbor, edge_index = rng.choice(adjacency[frontier[walker_index]])
        frontier[walker_index] = neighbor
        before = len(selected)
        selected.add(neighbor)
        if len(selected) > before:
            tree_edges.add(edge_index)
        stalled_steps = stalled_steps + 1 if len(selected) == before else 0
        # Keep connectivity: a stalled walk expands from a boundary edge rather
        # than restarting in another component.
        if stalled_steps >= max(100, len(frontier) * 10):
            boundary = [
                (node, neighbor, index)
                for node in selected
                for neighbor, index in adjacency.get(node, ())
                if neighbor not in selected
            ]
            if not boundary:
                break
            node, neighbor, edge_index = rng.choice(boundary)
            selected.add(neighbor)
            tree_edges.add(edge_index)
            frontier[rng.randrange(len(frontier))] = neighbor
            stalled_steps = 0
    if len(selected) != target_nodes:
        raise ValueError(f"Connected frontier reached {len(selected)} nodes, not target {target_nodes}")
    return selected, tree_edges


def _relation_representatives(records: Iterable[dict[str, Any]], rng: random.Random) -> dict[str, int]:
    """Choose one real triple for every predicate by reservoir sampling."""
    representatives: dict[str, int] = {}
    seen: Counter[str] = Counter()
    for index, record in enumerate(records):
        subject, obj = record.get("subject"), record.get("object")
        if not isinstance(subject, str) or not subject.strip() or not isinstance(obj, str) or not obj.strip():
            continue
        predicate = str(record.get("predicate", ""))
        seen[predicate] += 1
        if rng.randrange(seen[predicate]) == 0:
            representatives[predicate] = index
    return representatives


def _endpoints(record: dict[str, Any]) -> tuple[str, str]:
    return normalize_term(record["subject"]), normalize_term(record["object"])


def _sample_edges_with_budget(
    records: list[dict[str, Any]],
    selected_nodes: set[str],
    required_indices: set[int],
    reference_predicates: Counter[str],
    target_edges: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Select real edges with full predicate coverage and a fixed edge budget."""
    groups: dict[str, list[int]] = {}
    eligible_indices: set[int] = set()
    node_support: dict[str, int] = {}
    for index, record in enumerate(records):
        subject, obj = record.get("subject"), record.get("object")
        if not isinstance(subject, str) or not isinstance(obj, str) or not subject.strip() or not obj.strip():
            continue
        subject_id, object_id = _endpoints(record)
        if subject_id not in selected_nodes or object_id not in selected_nodes:
            continue
        predicate = str(record.get("predicate", ""))
        groups.setdefault(predicate, []).append(index)
        eligible_indices.add(index)
        node_support.setdefault(subject_id, index)
        node_support.setdefault(object_id, index)

    missing_nodes = selected_nodes - set(node_support)
    if missing_nodes:
        raise ValueError(f"{len(missing_nodes)} selected nodes have no edge in the sampled subgraph")
    selected_indices = required_indices & eligible_indices
    selected_indices.update(node_support.values())
    if len(selected_indices) > target_edges:
        raise ValueError(
            f"Mandatory edge set has {len(selected_indices)} edges, above target edge budget {target_edges}"
        )

    total_reference_edges = sum(reference_predicates.values())
    for predicate, indices in groups.items():
        desired = max(1, round(target_edges * reference_predicates[predicate] / total_reference_edges))
        already_selected = sum(index in selected_indices for index in indices)
        need = min(
            max(0, desired - already_selected),
            len(indices) - already_selected,
            target_edges - len(selected_indices),
        )
        if need:
            available = [index for index in indices if index not in selected_indices]
            selected_indices.update(rng.sample(available, need))

    if len(selected_indices) < target_edges:
        available = [
            index
            for indices in groups.values()
            for index in indices
            if index not in selected_indices
        ]
        selected_indices.update(rng.sample(available, min(target_edges - len(selected_indices), len(available))))
    return [records[index] for index in sorted(selected_indices)]


def _induced_subgraph(records: Iterable[dict[str, Any]], selected_nodes: set[str]) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if isinstance(record.get("subject"), str)
        and isinstance(record.get("object"), str)
        and normalize_term(record["subject"]) in selected_nodes
        and normalize_term(record["object"]) in selected_nodes
    ]


def sample_scale_matched_clean_graphs(
    clean_graph_path: str | Path,
    target_graph_path: str | Path,
    *,
    common_nodes_path: str | Path = DEFAULT_COMMON_NODES,
    semantic_types_path: str | Path = DEFAULT_SEMANTIC_TYPES,
    candidates: int = 20,
    walkers: int = 64,
    random_seed: int = 42,
    top_k: int = 1,
    with_stats: bool = False,
) -> list[list[dict[str, Any]]] | list[tuple[list[dict[str, Any]], dict[str, Any]]]:
    """Return the ``top_k`` scale-matched clean candidates, best first.

    All common concepts, selected semantic-type concepts, and semantic-type
    nodes are mandatory anchors.  Candidates are ranked by their structural
    distance to the complete clean graph.  This function does not write files.
    """
    if candidates < 1 or walkers < 1 or top_k < 1:
        raise ValueError("candidates, walkers, and top_k must be >= 1")
    if top_k > candidates:
        raise ValueError("top_k cannot exceed candidates")

    clean = _read_records(clean_graph_path)
    target = _read_records(target_graph_path)
    adjacency, clean_nodes = _build_adjacency(clean)
    _target_adjacency, target_nodes_set = _build_adjacency(target)
    target_size = len(target_nodes_set)
    requested_anchors = load_anchor_terms(common_nodes_path, semantic_types_path)
    anchors = requested_anchors & clean_nodes
    missing_anchors = sorted(requested_anchors - clean_nodes)
    if not anchors:
        raise ValueError("None of the required anchors are present in the clean graph")
    if len(anchors) > target_size:
        raise ValueError(
            f"Target has {target_size} nodes but {len(anchors)} mandatory anchors are required"
        )
    if target_size > len(clean_nodes):
        raise ValueError(f"Target has {target_size} nodes but clean graph has only {len(clean_nodes)} nodes")

    reference = graph_profile(clean)
    target_edges = round(target_size * reference["edges"] / reference["nodes"])
    root = max(anchors, key=lambda node: len(adjacency.get(node, ())))
    bfs_parents = _bfs_tree(adjacency, root)
    candidate_results: list[
        tuple[float, int, list[dict[str, Any]], dict[str, Any], dict[str, float]]
    ] = []
    candidate_scores: list[float] = []
    for candidate_index in range(candidates):
        rng = random.Random(random_seed + candidate_index)
        relation_representatives = _relation_representatives(clean, rng)
        required_indices = set(relation_representatives.values())
        # Keep the selected semantic ``isa`` annotations when both endpoints
        # become part of the sample. They are later included in the edge budget.
        required_indices.update(
            index
            for index, record in enumerate(clean)
            if record.get("synthetic") is True and record.get("predicate") == "isa"
        )
        required_nodes = set(anchors)
        for index in required_indices:
            subject, obj = clean[index].get("subject"), clean[index].get("object")
            if isinstance(subject, str) and subject.strip() and isinstance(obj, str) and obj.strip():
                required_nodes.update(_endpoints(clean[index]))
        if len(required_nodes) > target_size:
            raise ValueError(
                f"Mandatory anchors and relation representatives need {len(required_nodes)} nodes, above target {target_size}"
            )
        connected_nodes, connector_edges = _connected_core(required_nodes, root, bfs_parents)
        if len(connected_nodes) > target_size:
            raise ValueError(
                f"Connecting mandatory nodes needs {len(connected_nodes)} nodes, above target {target_size}"
            )
        selected_nodes, connector_edges = _frontier_nodes(
            adjacency, connected_nodes, connector_edges, target_size, walkers, rng
        )
        required_indices.update(connector_edges)
        candidate_graph = _sample_edges_with_budget(
            clean,
            selected_nodes,
            required_indices,
            reference["predicate_distribution"],
            target_edges,
            rng,
        )
        candidate_profile = graph_profile(candidate_graph)
        score, parts = structural_distance(candidate_profile, reference)
        candidate_scores.append(score)
        candidate_results.append((score, candidate_index, candidate_graph, candidate_profile, parts))

    ranked_results = sorted(candidate_results, key=lambda item: (item[0], item[1]))[:top_k]
    ranked: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
    for rank, (score, candidate_index, graph, profile, parts) in enumerate(ranked_results, start=1):
        report = {
            "method": "anchor_guided_frontier_sampling",
            "clean_graph": str(clean_graph_path),
            "target_graph": str(target_graph_path),
            "target_nodes": target_size,
            "target_edges": target_edges,
            "sample_nodes": profile["nodes"],
            "sample_edges": profile["edges"],
            "mandatory_anchors_present": len(anchors),
            "mandatory_anchors_missing_from_clean": len(missing_anchors),
            "candidates": candidates,
            "walkers": walkers,
            "random_seed": random_seed,
            "selected_candidate_rank": rank,
            "selected_candidate_index": candidate_index,
            "selected_candidate_seed": random_seed + candidate_index,
            "selected_candidate_score": score,
            "selected_candidate_score_parts": parts,
            "all_candidate_scores": candidate_scores,
            "sample_statistics": {
                key: value for key, value in profile.items() if not key.endswith("_distribution")
            },
        }
        ranked.append((graph, report))
    if with_stats:
        return ranked
    return [graph for graph, _report in ranked]


def sample_scale_matched_clean_graph(
    clean_graph_path: str | Path,
    target_graph_path: str | Path,
    *,
    common_nodes_path: str | Path = DEFAULT_COMMON_NODES,
    semantic_types_path: str | Path = DEFAULT_SEMANTIC_TYPES,
    candidates: int = 20,
    walkers: int = 64,
    random_seed: int = 42,
    save: bool = False,
    output_path: str | Path | None = None,
    with_stats: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the single best scale-matched clean graph.

    This backward-compatible convenience wrapper returns the best candidate.
    Use :func:`sample_scale_matched_clean_graphs` when several ranked
    candidates are required.
    """
    if save and output_path is None:
        raise ValueError("output_path is required when save=True")
    ranked = sample_scale_matched_clean_graphs(
        clean_graph_path,
        target_graph_path,
        common_nodes_path=common_nodes_path,
        semantic_types_path=semantic_types_path,
        candidates=candidates,
        walkers=walkers,
        random_seed=random_seed,
        top_k=1,
        with_stats=True,
    )
    best_graph, report = ranked[0]
    if save:
        _write_records(output_path, best_graph)
    if with_stats:
        return best_graph, report
    return best_graph
