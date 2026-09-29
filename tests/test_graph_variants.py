"""Executable tests for data_preprocessing.graph_variants.

Run with either ``python tests/test_graph_variants.py`` or
``python -m unittest tests.test_graph_variants``.  Every graph used here is
created in a temporary directory; no graph variant is saved in ``datasets``.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import warnings
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data_preprocessing.graph_variants import get_graph


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


class GraphVariantsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.datasets = Path(self.temporary_directory.name)
        self.semantic_types = self.datasets / "semantic_types.json"
        self.inverse_resolver = self.datasets / "inverse.xlsx"
        write_json(
            self.semantic_types,
            [
                {
                    "normalized_term": "alpha",
                    "semantic_type": "Type A",
                    "semantic_type_tui": "T001",
                    "anchor_id": "test:alpha",
                },
                {
                    "normalized_term": "beta",
                    "semantic_type": "Type B",
                    "semantic_type_tui": "T002",
                    "anchor_id": "test:beta",
                },
            ],
        )
        pd.DataFrame({"inverse": ["inverse_rel"], "direct": ["direct_rel"]}).to_excel(
            self.inverse_resolver, index=False
        )

        # 50 frequent records, one inverse record, and one rare record.
        raw = [
            {"subject": "Alpha", "predicate": "direct_rel", "object": "Beta"}
            for _ in range(50)
        ]
        raw.extend(
            [
                {"subject": "Beta", "predicate": "inverse_rel", "object": "Alpha"},
                {"subject": "Alpha", "predicate": "rare_rel", "object": "Gamma"},
            ]
        )
        write_json(self.datasets / "UMLS_nci_kg_raw.json", raw)
        write_json(self.datasets / "UMLS_nci_kg.json", [{"subject": "default", "predicate": "p", "object": "graph"}])
        write_json(self.datasets / "Example_kg_raw.json", raw)
        write_json(self.datasets / "Example_kg.json", [{"subject": "default", "predicate": "p", "object": "graph"}])

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def graph(self, kg_name: str, options: list[str] | None = None, **kwargs):
        return get_graph(
            kg_name,
            options=options,
            datasets_dir=self.datasets,
            semantic_types_path=self.semantic_types,
            inverse_resolver_path=self.inverse_resolver,
            **kwargs,
        )

    def test_default_loads_processed_graph_and_raw_loads_raw_graph(self) -> None:
        self.assertEqual(self.graph("Example_kg"), [{"subject": "default", "predicate": "p", "object": "graph"}])
        self.assertEqual(len(self.graph("Example_kg", ["raw"])), 52)

    def test_no_semantic_types_and_no_frequency_filter_apply_together_without_saving(self) -> None:
        graph, stats = self.graph(
            "Example_kg", ["no_smnt", "no_freq_filter"], with_stats=True
        )
        self.assertEqual(len(graph), 52)
        self.assertEqual(stats["records"], 52)
        self.assertEqual(stats["unique_relations"], 3)
        self.assertFalse((self.datasets / "Example_kg_no_freq_filter_no_smnt.json").exists())

    def test_default_treatments_filter_then_add_semantic_types(self) -> None:
        graph = self.graph("Example_kg", ["no_freq_filter"])
        predicates = [record["predicate"] for record in graph]
        # no_freq_filter keeps all 52 raw records, then two isa edges are added.
        self.assertEqual(len(graph), 54)
        self.assertEqual(predicates.count("isa"), 2)

    def test_umls_inverse_is_removed_by_default_and_preserved_when_requested(self) -> None:
        filtered = self.graph("UMLS_nci_kg", ["no_smnt", "no_freq_filter"])
        kept = self.graph("UMLS_nci_kg", ["no_smnt", "no_freq_filter", "with_inverse"])
        self.assertNotIn("inverse_rel", [record["predicate"] for record in filtered])
        self.assertIn("inverse_rel", [record["predicate"] for record in kept])

    def test_save_reuses_existing_variant_without_reprocessing(self) -> None:
        expected_path = self.datasets / "Example_kg_no_smnt.json"
        write_json(expected_path, [{"subject": "cached", "predicate": "p", "object": "variant"}])
        output = StringIO()
        with redirect_stdout(output):
            graph = self.graph("Example_kg", ["no_smnt"], save=False)
        self.assertEqual(graph, [{"subject": "cached", "predicate": "p", "object": "variant"}])
        self.assertIn("Variant already exists", output.getvalue())

    def test_missing_raw_warns_and_uses_default_graph(self) -> None:
        (self.datasets / "Example_kg_raw.json").unlink()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            graph = self.graph("Example_kg", ["no_smnt", "no_freq_filter"])
        self.assertEqual(graph, [{"subject": "default", "predicate": "p", "object": "graph"}])
        self.assertEqual(len(caught), 1)
        self.assertIn("Raw graph not found", str(caught[0].message))

    def test_invalid_options_raise(self) -> None:
        with self.assertRaises(ValueError):
            self.graph("Example_kg", ["unknown"])
        with self.assertRaises(ValueError):
            self.graph("Example_kg", ["raw", "no_smnt"])
        with self.assertRaises(ValueError):
            self.graph("Example_kg", ["with_inverse"])

    def test_rdf_export_writes_paired_rdf_xml_file(self) -> None:
        output = self.graph("Example_kg", ["raw"], format="rdf", save=True)
        self.assertEqual(output, self.datasets / "Example_kg_raw.rdf")
        self.assertTrue(output.exists())
        root = ElementTree.parse(output).getroot()
        self.assertEqual(root.tag, "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}RDF")
        text = output.read_text(encoding="utf-8")
        self.assertIn("Alpha", text)
        self.assertIn("direct_rel", text)

    def test_rdf_requires_save_and_json_remains_default(self) -> None:
        with self.assertRaises(ValueError):
            self.graph("Example_kg", ["raw"], format="rdf")
        self.assertEqual(len(self.graph("Example_kg", ["raw"])), 52)

    def test_rdf_export_preserves_empty_string_nodes(self) -> None:
        write_json(
            self.datasets / "Empty_kg_raw.json",
            [{"subject": "", "predicate": "rel", "object": "Object"}],
        )
        output = self.graph("Empty_kg", ["raw"], format="rdf", save=True)
        root = ElementTree.parse(output).getroot()
        self.assertEqual(root.tag, "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}RDF")
        self.assertIn('rdf:about="https://tdg-bench.org/resource/entity/"', output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
