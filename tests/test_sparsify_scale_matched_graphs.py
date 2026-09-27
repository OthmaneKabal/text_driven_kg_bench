"""Tests for stratified edge sparsification of scale-matched graphs."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data_preprocessing.sparsify_scale_matched_graphs import graph_nodes, sparsify_records


class SparsifyScaleMatchedGraphsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            {"subject": "A", "predicate": "r1", "object": "B"},
            {"subject": "B", "predicate": "r1", "object": "C"},
            {"subject": "C", "predicate": "r2", "object": "D"},
            {"subject": "D", "predicate": "r2", "object": "A"},
            {"subject": "A", "predicate": "r3", "object": "C"},
            {"subject": "B", "predicate": "r3", "object": "D"},
        ]

    def test_edge_budget_nodes_and_relations_are_preserved(self) -> None:
        sparse, report = sparsify_records(self.records, target_edges=4, random_seed=42)
        self.assertEqual(len(sparse), 4)
        self.assertEqual(graph_nodes(sparse), graph_nodes(self.records))
        self.assertEqual({record["predicate"] for record in sparse}, {"r1", "r2", "r3"})
        self.assertEqual(report["output_edges"], 4)

    def test_impossible_budget_raises_instead_of_dropping_relations(self) -> None:
        with self.assertRaisesRegex(ValueError, "below the 3 relations"):
            sparsify_records(self.records, target_edges=2, random_seed=42)

    def test_raw_node_variants_are_preserved_independently(self) -> None:
        records = [
            {"subject": "oxygen", "predicate": "r1", "object": "A"},
            {"subject": "Oxygen", "predicate": "r1", "object": "B"},
            {"subject": "A", "predicate": "r2", "object": "B"},
        ]
        sparse, _report = sparsify_records(records, target_edges=3, random_seed=42)
        self.assertEqual(graph_nodes(sparse), {"oxygen", "Oxygen", "A", "B"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
