import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data_preprocessing.umls_nci_graph import (
    FILTER_INVERSE,
    FILTER_RARE,
    INTEGRATE_SMT,
    OPTION_KEEP_INVERSE,
    OPTION_KEEP_RARE,
    OPTION_NO_SMT,
    BuildConfig,
    ConceptMetadata,
    DatabaseConfig,
    EdgeRecord,
    SemanticType,
    SqliteEdgeStore,
    build_nci_graph,
    inverse_relation_labels,
    sample_semantic_type_pairs,
)


class TreatmentConfigurationTests(unittest.TestCase):
    def test_all_treatments_are_enabled_by_default(self):
        self.assertEqual(
            BuildConfig().treatments,
            (FILTER_INVERSE, FILTER_RARE, INTEGRATE_SMT),
        )

    def test_skip_options_disable_only_the_requested_treatments(self):
        config = BuildConfig(
            options=(OPTION_KEEP_RARE, OPTION_KEEP_INVERSE, OPTION_NO_SMT)
        )
        self.assertEqual(config.treatments, ())


class InverseFilteringTests(unittest.TestCase):
    def test_selects_one_stable_inverse_label_per_pair(self):
        labels = inverse_relation_labels({"part_of": "has_part", "has_part": "part_of"})
        self.assertEqual(labels, frozenset({"has_part"}))

    def test_matches_notebook_resolver_for_isa(self):
        labels = inverse_relation_labels({"isa": "inverse_isa", "inverse_isa": "isa"})
        self.assertEqual(labels, frozenset({"inverse_isa"}))

    def test_self_inverse_relation_matches_notebook_resolver(self):
        self.assertEqual(
            inverse_relation_labels({"associated_with": "associated_with"}),
            frozenset({"associated_with"}),
        )


class EdgeStoreTests(unittest.TestCase):
    def test_store_deduplicates_and_filters_by_relation(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SqliteEdgeStore(Path(directory) / "edges.sqlite3")
            try:
                edge = EdgeRecord("C1", "common", "RO", "C2")
                store.add_many(
                    [
                        edge,
                        edge,
                        EdgeRecord("C2", "rare", "RO", "C3"),
                    ]
                )
                store.finalize()
                store.set_valid_nodes(["C1", "C2", "C3"])

                self.assertEqual(store.input_rows, 3)
                self.assertEqual(store.inserted_rows, 2)
                self.assertEqual(store.relation_counts()["common"], 1)
                self.assertEqual(list(store.iter_edges({"common"})), [edge])
            finally:
                store.close()


class SemanticTypeSamplingTests(unittest.TestCase):
    def test_sampling_is_reproducible_and_at_least_one_per_type(self):
        disease = SemanticType("T047", "Disease or Syndrome")
        finding = SemanticType("T033", "Finding")
        metadata = {
            "C1": ConceptMetadata("C1", "One", (disease,)),
            "C2": ConceptMetadata("C2", "Two", (disease,)),
            "C3": ConceptMetadata("C3", "Three", (finding,)),
        }
        first = sample_semantic_type_pairs(metadata, metadata, 0.01, 42)
        second = sample_semantic_type_pairs(metadata, metadata, 0.01, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)

    def test_uses_primary_type_and_deduplicates_equal_term_type_pairs(self):
        disease = SemanticType("T047", "Disease or Syndrome")
        finding = SemanticType("T033", "Finding")
        metadata = {
            "C1": ConceptMetadata("C1", "Same label", (disease, finding)),
            "C2": ConceptMetadata("C2", "Same label", (disease,)),
            "C3": ConceptMetadata("C3", "Other label", (finding,)),
        }
        sampled = sample_semantic_type_pairs(["C1", "C2", "C3"], metadata, 0.01, 42)
        self.assertEqual(
            sampled,
            [("C1", disease), ("C3", finding)],
        )


class BuildPipelineTests(unittest.TestCase):
    def test_writes_enriched_graph_and_summary(self):
        disease = SemanticType("T047", "Disease or Syndrome")
        metadata = {
            cui: ConceptMetadata(cui, f"Concept {cui}", (disease,))
            for cui in ("C1", "C2", "C3", "C4", "C5")
        }

        class FakeConnection:
            def close(self):
                pass

        class FakeRepository:
            def __init__(self, connection, source, fetch_size):
                pass

            def inverse_maps(self):
                return (
                    {"has_part": "part_of", "part_of": "has_part"},
                    {"RB": "RN", "RN": "RB"},
                )

            def extract_edges(self, store, excluded_relations=frozenset()):
                edges = [
                    EdgeRecord("C1", "common", "RO", "C2"),
                    EdgeRecord("C2", "common", "RO", "C3"),
                    EdgeRecord("C3", "rare", "RO", "C4"),
                    EdgeRecord("C4", "has_part", "RB", "C5"),
                ]
                kept = [
                    edge for edge in edges if edge.predicate not in excluded_relations
                ]
                store.add_many(kept)
                store.finalize()
                return {
                    "rows_read": len(edges),
                    "rows_excluded": len(edges) - len(kept),
                }

            def concept_metadata(self, cuis, chunk_size):
                return {cui: metadata[cui] for cui in cuis}

        with tempfile.TemporaryDirectory() as directory:
            graph_path = Path(directory) / "graph.json"
            with patch(
                "data_preprocessing.umls_nci_graph.connect_mysql",
                return_value=FakeConnection(),
            ), patch(
                "data_preprocessing.umls_nci_graph.UMLSRepository",
                FakeRepository,
            ):
                summary = build_nci_graph(
                    database_config=DatabaseConfig(
                        host="localhost",
                        port=3306,
                        user="user",
                        password="secret",
                        database="umls",
                    ),
                    output_path=graph_path,
                    build_config=BuildConfig(
                        min_relation_frequency=2,
                        semantic_type_fraction=0.01,
                    ),
                )

            graph = json.loads(graph_path.read_text(encoding="utf-8"))
            self.assertEqual(len(graph), 3)
            self.assertEqual(
                [edge["predicate"] for edge in graph],
                [
                    "common",
                    "common",
                    "isa",
                ],
            )
            self.assertTrue(all("subject_types" in edge for edge in graph))
            self.assertTrue(all("object_types" in edge for edge in graph))
            self.assertEqual(summary["stages"]["mrrel_rows_read"], 4)
            self.assertEqual(summary["stages"]["inverse_relation_rows_removed"], 1)
            self.assertEqual(summary["stages"]["base_edges_written"], 2)
            self.assertEqual(summary["statistics"]["edges"], 3)
            self.assertNotIn("secret", json.dumps(summary))
            self.assertTrue(graph_path.with_suffix(".summary.json").exists())


if __name__ == "__main__":
    unittest.main()
