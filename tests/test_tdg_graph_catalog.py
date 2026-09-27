"""Executable checks for TDGBench graph-name resolution."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data_preprocessing.graph_catalog import available_graph_names, validate_graph_name


class TDGBenchGraphCatalogTest(unittest.TestCase):
    def test_scale_matched_graphs_are_discoverable(self) -> None:
        available = available_graph_names()
        self.assertIn("UMLS_nci_kg_scale_matched_GT2KG_kg", available)
        self.assertIn("UMLS_nci_kg_scale_matched_KG_GEN_kg", available)

    def test_unknown_graph_name_has_a_clear_message_and_suggestion(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Unknown graph name 'GT2KG'.*Did you mean 'GT2KG_kg'",
        ):
            validate_graph_name("GT2KG")


if __name__ == "__main__":
    unittest.main(verbosity=2)
