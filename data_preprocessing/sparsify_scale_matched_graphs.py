"""Create edge-sparsity-matched variants of scale-matched UMLS--NCI graphs.

The input graph already has the target graph's node count.  This module keeps
those nodes, but removes real UMLS edges through stratified sampling until its
edge count equals that of the target graph.  Connectivity is intentionally not
preserved: fragmentation is an expected consequence of matching sparsity.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
TARGETS = (DATASETS / "GT2KG_kg.json", DATASETS / "KG_GEN_kg.json")
COVERAGE_CANDIDATES = 20


def read_records(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    with input_path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{input_path} must be a JSON list of objects")
    return records


def write_records(path: str | Path, records: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(output_path)


def record_endpoints(record: dict[str, Any]) -> tuple[str, str]:
    subject, obj = record.get("subject"), record.get("object")
    if not isinstance(subject, str) or not subject.strip() or not isinstance(obj, str) or not obj.strip():
        raise ValueError("Every source record must have non-empty string subject and object fields")
    # GraphDataPreparation and the pre-generated gold-standard splits resolve
    # node labels by exact string equality.  Do not collapse case or Unicode
    # variants here: ``oxygen`` and ``Oxygen`` must both remain available when
    # both occur in the source graph.
    return subject, obj


def graph_nodes(records: Iterable[dict[str, Any]]) -> set[str]:
    """Return nodes from usable triples, ignoring malformed target records.

    A few existing GT2KG records have an empty endpoint.  They do count toward
    the requested JSON edge budget, but they cannot define a graph node.  This
    is the same endpoint rule used by the scale-matching sampler.
    """
    nodes: set[str] = set()
    for record in records:
        try:
            nodes.update(record_endpoints(record))
        except ValueError:
            continue
    return nodes


def relation_quotas(groups: dict[str, list[int]], edge_budget: int) -> dict[str, int]:
    """Allocate an exact edge budget proportionally, with one edge per relation."""
    relations = sorted(groups)
    if edge_budget < len(relations):
        raise ValueError(
            f"The edge budget ({edge_budget}) is below the {len(relations)} relations that must be retained"
        )
    total_edges = sum(len(indices) for indices in groups.values())
    quotas = {relation: 1 for relation in relations}
    remaining = edge_budget - len(relations)
    fractional: list[tuple[float, str]] = []
    for relation in relations:
        exact = remaining * len(groups[relation]) / total_edges
        addition = min(len(groups[relation]) - 1, int(exact))
        quotas[relation] += addition
        fractional.append((exact - int(exact), relation))
    unallocated = edge_budget - sum(quotas.values())
    for _fraction, relation in sorted(fractional, key=lambda item: (-item[0], item[1])):
        if unallocated == 0:
            break
        if quotas[relation] < len(groups[relation]):
            quotas[relation] += 1
            unallocated -= 1
    if unallocated:
        raise ValueError("Could not allocate the requested edge budget across source relations")
    return quotas


def node_cover_by_relation(
    groups: dict[str, list[int]], endpoints: list[tuple[str, str]], rng: random.Random
) -> dict[str, set[int]]:
    """Build a connectedness-free edge cover, grouped by its predicates."""
    relation_by_index = {
        index: relation for relation, indices in groups.items() for index in indices
    }
    selected = {rng.choice(indices) for indices in groups.values()}
    covered = {node for index in selected for node in endpoints[index]}
    all_nodes = {node for pair in endpoints for node in pair}
    order = list(range(len(endpoints)))
    rng.shuffle(order)
    for index in order:
        left, right = endpoints[index]
        if left not in covered and right not in covered:
            selected.add(index)
            covered.update((left, right))
    for index in order:
        left, right = endpoints[index]
        if left not in covered or right not in covered:
            selected.add(index)
            covered.update((left, right))
    missing = all_nodes - covered
    if missing:
        raise ValueError(f"Could not retain {len(missing)} source node(s)")
    by_relation = {relation: set() for relation in groups}
    for index in selected:
        by_relation[relation_by_index[index]].add(index)
    return by_relation


def quota_distance(quotas: dict[str, int], groups: dict[str, list[int]]) -> float:
    """Jensen--Shannon distance from the source predicate distribution."""
    source_total = sum(len(indices) for indices in groups.values())
    output_total = sum(quotas.values())
    divergence = 0.0
    for relation, indices in groups.items():
        source_share = len(indices) / source_total
        output_share = quotas[relation] / output_total
        midpoint = (source_share + output_share) / 2
        divergence += 0.5 * source_share * math.log2(source_share / midpoint)
        divergence += 0.5 * output_share * math.log2(output_share / midpoint)
    return math.sqrt(divergence)


def mandatory_edges_by_relation(
    groups: dict[str, list[int]], endpoints: list[tuple[str, str]], rng: random.Random
) -> dict[str, set[int]]:
    """Select edges needed by nodes that have only one available relation."""
    relation_by_index = {
        index: relation for relation, indices in groups.items() for index in indices
    }
    incident: dict[str, list[int]] = defaultdict(list)
    for index, pair in enumerate(endpoints):
        for node in set(pair):
            incident[node].append(index)
    dependent_nodes: dict[str, set[str]] = defaultdict(set)
    for node, indices in incident.items():
        relations = {relation_by_index[index] for index in indices}
        if len(relations) == 1:
            dependent_nodes[next(iter(relations))].add(node)

    mandatory = {relation: set() for relation in groups}
    covered: set[str] = set()
    for relation in sorted(groups):
        for node in sorted(dependent_nodes[relation]):
            if node in covered:
                continue
            candidates = [index for index in incident[node] if relation_by_index[index] == relation]
            # Prefer covering another constrained node with the same edge.
            best_score = max(
                sum(endpoint in dependent_nodes[relation] and endpoint not in covered for endpoint in set(endpoints[index]))
                for index in candidates
            )
            best = [
                index
                for index in candidates
                if sum(endpoint in dependent_nodes[relation] and endpoint not in covered for endpoint in set(endpoints[index]))
                == best_score
            ]
            chosen = rng.choice(best)
            mandatory[relation].add(chosen)
            covered.update(endpoints[chosen])
    return mandatory


def relax_quotas_for_mandatory_nodes(
    quotas: dict[str, int], mandatory: dict[str, set[int]], groups: dict[str, list[int]]
) -> tuple[dict[str, int], dict[str, int]]:
    """Increase only forced quotas and compensate from high-frequency donors."""
    adjusted = dict(quotas)
    minimums = {relation: max(1, len(mandatory[relation])) for relation in groups}
    for relation, minimum in minimums.items():
        adjusted[relation] = max(adjusted[relation], minimum)
    excess = sum(adjusted.values()) - sum(quotas.values())
    adjustments = {relation: adjusted[relation] - quotas[relation] for relation in groups}
    while excess:
        donors = [
            relation
            for relation in groups
            if adjusted[relation] > minimums[relation]
        ]
        if not donors:
            raise ValueError(
                "The target edge budget cannot retain all source nodes, even after relaxing every relation quota"
            )
        # Prefer removing from the largest relation with available slack. This
        # minimizes the relative change to a relation's original frequency.
        donor = max(donors, key=lambda relation: (len(groups[relation]), adjusted[relation] - minimums[relation], relation))
        adjusted[donor] -= 1
        adjustments[donor] -= 1
        excess -= 1
    return adjusted, {relation: delta for relation, delta in adjustments.items() if delta}


def stratified_selection(
    groups: dict[str, list[int]],
    quotas: dict[str, int],
    mandatory: dict[str, set[int]],
    rng: random.Random,
) -> tuple[set[int], dict[str, set[int]]]:
    """Draw the exact adjusted quota for every relation."""
    selected_by_relation = {relation: set(indices) for relation, indices in mandatory.items()}
    for relation, indices in groups.items():
        remaining = quotas[relation] - len(selected_by_relation[relation])
        available = [index for index in indices if index not in selected_by_relation[relation]]
        selected_by_relation[relation].update(rng.sample(available, remaining))
    selected = set().union(*selected_by_relation.values())
    return selected, selected_by_relation


def _removable_edge(
    selected_for_relation: set[int],
    endpoints: list[tuple[str, str]],
    support: Counter[str],
    rng: random.Random,
) -> int | None:
    """Find an edge that can be replaced without making a node disappear."""
    candidates = list(selected_for_relation)
    rng.shuffle(candidates)
    for index in candidates:
        if all(support[node] > 1 for node in set(endpoints[index])):
            return index
    return None


def preserve_nodes_within_quotas(
    selected: set[int],
    selected_by_relation: dict[str, set[int]],
    groups: dict[str, list[int]],
    endpoints: list[tuple[str, str]],
    rng: random.Random,
) -> int:
    """Use within-relation swaps to retain every node without altering quotas."""
    incident: dict[str, list[int]] = defaultdict(list)
    relation_by_index = {
        index: relation for relation, indices in groups.items() for index in indices
    }
    all_nodes: set[str] = set()
    support: Counter[str] = Counter()
    for index, pair in enumerate(endpoints):
        for node in set(pair):
            incident[node].append(index)
            all_nodes.add(node)
    for index in selected:
        support.update(set(endpoints[index]))

    swaps = 0
    while True:
        uncovered = [node for node in all_nodes if support[node] == 0]
        if not uncovered:
            return swaps
        rng.shuffle(uncovered)
        progress = False
        for node in uncovered:
            if support[node] > 0:
                continue
            candidates = [index for index in incident[node] if index not in selected]
            rng.shuffle(candidates)
            for addition in candidates:
                relation = relation_by_index[addition]
                removal = _removable_edge(selected_by_relation[relation], endpoints, support, rng)
                if removal is None:
                    continue
                selected.remove(removal)
                selected.add(addition)
                selected_by_relation[relation].remove(removal)
                selected_by_relation[relation].add(addition)
                support.subtract(set(endpoints[removal]))
                support.update(set(endpoints[addition]))
                swaps += 1
                progress = True
                break
        if not progress:
            raise ValueError(
                "Exact relation quotas cannot retain every source node. "
                "Increase the edge budget or allow node removal."
            )


def sparsify_records(
    records: list[dict[str, Any]], target_edges: int, random_seed: int = 42
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Stratify real edges to ``target_edges`` while retaining nodes and relations.

    The returned graph can have multiple weakly connected components. It has
    exactly ``target_edges`` records; each relation receives its proportional
    integer quota, and every source node remains incident to an output edge.
    """
    if target_edges < 1:
        raise ValueError("target_edges must be >= 1")
    if target_edges > len(records):
        raise ValueError(
            f"Target requests {target_edges} edges but the source contains only {len(records)}"
        )
    endpoints = [record_endpoints(record) for record in records]
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[str(record.get("predicate", ""))].append(index)

    proportional_quotas = relation_quotas(groups, target_edges)
    candidates: list[tuple[float, int, dict[str, set[int]], dict[str, int], dict[str, int]]] = []
    for candidate_index in range(COVERAGE_CANDIDATES):
        candidate_rng = random.Random(random_seed + candidate_index)
        mandatory = node_cover_by_relation(groups, endpoints, candidate_rng)
        quotas, quota_adjustments = relax_quotas_for_mandatory_nodes(
            proportional_quotas, mandatory, groups
        )
        candidates.append(
            (
                quota_distance(quotas, groups),
                candidate_index,
                mandatory,
                quotas,
                quota_adjustments,
            )
        )
    _distance, selected_candidate_index, mandatory, quotas, quota_adjustments = min(
        candidates, key=lambda item: (item[0], sum(abs(value) for value in item[4].values()), item[1])
    )
    rng = random.Random(random_seed + COVERAGE_CANDIDATES + selected_candidate_index)
    selected, selected_by_relation = stratified_selection(groups, quotas, mandatory, rng)

    output = [records[index] for index in sorted(selected)]
    source_nodes, output_nodes = graph_nodes(records), graph_nodes(output)
    if source_nodes != output_nodes:
        raise AssertionError("Node-preservation invariant failed during sparsification")
    output_relations = {str(record.get("predicate", "")) for record in output}
    if output_relations != set(groups):
        raise AssertionError("Relation-coverage invariant failed during sparsification")
    actual_counts = Counter(str(record.get("predicate", "")) for record in output)
    if actual_counts != Counter(quotas):
        raise AssertionError("Exact relation-quota invariant failed during sparsification")
    report = {
        "input_edges": len(records),
        "output_edges": len(output),
        "target_edges": target_edges,
        "nodes_preserved": len(output_nodes),
        "relations_preserved": len(output_relations),
        "random_seed": random_seed,
        "quota_adjustment_edges": sum(abs(value) for value in quota_adjustments.values()) // 2,
        "quota_adjustments": quota_adjustments,
        "coverage_candidates": COVERAGE_CANDIDATES,
        "selected_coverage_candidate": selected_candidate_index,
        "relation_distribution_distance": quota_distance(quotas, groups),
    }
    return output, report


def default_source_path(target: Path, rank: int) -> Path:
    return DATASETS / f"UMLS_nci_kg_scale_matched_{target.stem}_top_{rank}.json"


def default_output_path(target: Path, rank: int) -> Path:
    return DATASETS / f"UMLS_nci_kg_scale_matched_{target.stem}_top_{rank}_sparsity_matched.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create edge-sparsity-matched versions of the ranked scale-matched clean graphs."
    )
    parser.add_argument("--targets", type=Path, nargs="+", default=TARGETS)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    plans = [
        (target, rank, default_source_path(target, rank), default_output_path(target, rank))
        for target in args.targets
        for rank in range(1, args.top_k + 1)
    ]
    missing = [source for _target, _rank, source, _output in plans if not source.exists()]
    if missing:
        raise FileNotFoundError("Missing scale-matched input(s): " + ", ".join(map(str, missing)))
    existing = [output for _target, _rank, _source, output in plans if output.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Output already exists; use --overwrite: " + ", ".join(map(str, existing)))

    for target, rank, source, output in plans:
        target_records = read_records(target)
        source_records = read_records(source)
        target_nodes, source_nodes = graph_nodes(target_records), graph_nodes(source_records)
        # The scale-matching stage uses canonicalised labels to count concepts.
        # Here we deliberately preserve the source's *raw* labels so that the
        # gold-standard split resolver can find them exactly.  Consequently a
        # raw-label count may differ from the target's raw-label count; the
        # relevant invariant for sparsification is preserving ``source_nodes``.
        graph, report = sparsify_records(source_records, len(target_records), args.random_seed + rank - 1)
        write_records(output, graph)
        print(
            f"Saved {output.name}: {report['nodes_preserved']} nodes, "
            f"{report['output_edges']} edges, {report['relations_preserved']} relations "
            f"(seed={report['random_seed']}; source raw nodes={len(source_nodes)}, "
            f"target raw nodes={len(target_nodes)})."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
