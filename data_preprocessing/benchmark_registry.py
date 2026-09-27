"""Resolve the labelled-node and split protocol associated with a graph name.

The no-graph baseline needs the same labelled terms and pre-generated splits
as a graph experiment, but it must not read the graph JSON itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from data_preprocessing.graph_catalog import validate_graph_name


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BenchmarkProtocol:
    """Files defining one node-classification benchmark protocol."""

    name: str
    common_nodes_path: Path
    splits_dir: Path


TDG_PROTOCOL = BenchmarkProtocol(
    name="TDG",
    common_nodes_path=ROOT / "datasets" / "common_nodes.xlsx",
    splits_dir=ROOT / "datasets" / "split",
)


def resolve_benchmark_protocol(kg_name: str) -> BenchmarkProtocol:
    """Return the labelled-node protocol associated with ``kg_name``.

    All current graph families in this repository are TDG graph variants and
    therefore share the same gold-standard terms and split files.  A new
    dataset must be registered here explicitly rather than silently reusing
    TDG labels and splits.
    """
    if not isinstance(kg_name, str) or not kg_name.strip():
        raise ValueError("kg_name must be a non-empty graph name.")
    resolved_name = validate_graph_name(kg_name.strip())
    tdg_prefixes = ("GT2KG_kg", "KG_GEN_kg", "UMLS_nci_kg")
    if resolved_name.startswith(tdg_prefixes):
        return TDG_PROTOCOL
    raise ValueError(
        f"No benchmark protocol is registered for kg_name={resolved_name!r}. "
        "Register its common-nodes file and splits directory in "
        "data_preprocessing/benchmark_registry.py."
    )
