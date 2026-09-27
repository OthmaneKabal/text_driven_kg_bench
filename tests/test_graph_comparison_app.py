"""Tests for the local graph-comparison web interface."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
APPLICATION_DIRECTORY = REPOSITORY_ROOT / "utilities"
if str(APPLICATION_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(APPLICATION_DIRECTORY))

import graph_comparison_app as comparison_app


def write_graph(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(records), encoding="utf-8")


class GraphComparisonAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.datasets = Path(self.temporary_directory.name)
        self.original_datasets_directory = comparison_app.DATASETS_DIRECTORY
        comparison_app.DATASETS_DIRECTORY = self.datasets
        comparison_app.PROFILE_CACHE.clear()
        write_graph(
            self.datasets / "UMLS_nci_kg.json",
            [
                {"subject": "A", "predicate": "treats", "object": "B", "subject_type": "Gene", "object_type": "Disease"},
                {"subject": "B", "predicate": "causes", "object": "C", "subject_type": "Disease", "object_type": "Chemical"},
            ],
        )
        write_graph(
            self.datasets / "candidate.json",
            [{"subject": "A", "predicate": "treats", "object": "B", "subject_type": "Gene", "object_type": "Disease"}],
        )

    def tearDown(self) -> None:
        comparison_app.DATASETS_DIRECTORY = self.original_datasets_directory
        comparison_app.PROFILE_CACHE.clear()
        self.temporary_directory.cleanup()

    def test_profile_reports_core_characteristics(self) -> None:
        profile = comparison_app.calculate_profile(self.datasets / "UMLS_nci_kg.json")
        self.assertEqual(profile.nodes, 3)
        self.assertEqual(profile.edges, 2)
        self.assertEqual(profile.unique_relations, 2)
        self.assertEqual(profile.weakly_connected_components, 1)
        self.assertEqual(profile.unique_semantic_types, 3)
        self.assertAlmostEqual(profile.mean_in_degree, 2 / 3)
        self.assertAlmostEqual(profile.mean_out_degree, 2 / 3)

    def test_compare_endpoint_includes_reference_and_selected_graph(self) -> None:
        client = comparison_app.app.test_client()
        response = client.post("/api/compare", json={"graphs": ["candidate"]})
        self.assertEqual(response.status_code, 200)
        graphs = response.get_json()["graphs"]
        self.assertEqual([graph["graph_name"] for graph in graphs], ["UMLS_nci_kg", "candidate"])
        self.assertEqual(graphs[0]["relation_distribution"], {"causes": 0.5, "treats": 0.5})
        self.assertEqual(graphs[1]["relation_distribution"], {"treats": 1.0})


if __name__ == "__main__":
    unittest.main(verbosity=2)
