"""Smoke tests for the graph-free MLP baseline."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_preprocessing.benchmark_registry import resolve_benchmark_protocol
from data_preprocessing.no_graph_data import build_no_graph_features
from tdg_bench import TDGBench


class NoGraphBaselineTests(unittest.TestCase):
    def test_tdg_graph_family_resolves_to_the_shared_protocol(self) -> None:
        protocol = resolve_benchmark_protocol("UMLS_nci_kg_scale_matched_GT2KG_kg_top_1")
        self.assertEqual(protocol.name, "TDG")
        self.assertTrue(protocol.common_nodes_path.exists())

    def test_random_features_are_seed_specific_and_reproducible(self) -> None:
        terms = ["alpha", "beta"]
        self.assertTrue(
            build_no_graph_features(terms, "random_4", random_seed=7).equal(
                build_no_graph_features(terms, "random_4", random_seed=7)
            )
        )
        self.assertFalse(
            build_no_graph_features(terms, "random_4", random_seed=7).equal(
                build_no_graph_features(terms, "random_4", random_seed=8)
            )
        )

    def test_random_baseline_uses_split_and_randomness_cartesian_product(self) -> None:
        result = TDGBench(use_classifier=True).evaluate_no_graph_baseline(
            kg_name="GT2KG_kg",
            init_embds=["random_8"],
            split_seeds=[42, 123],
            random_seeds=[1, 2],
            hidden_channels=[4],
            epochs=1,
            patience=1,
            verbose=False,
            save_results=False,
            save_models=False,
        )
        run = result["models"]["NoGraphMLP_random_8"]["runs"]["hidden_4_out_8"]
        self.assertEqual(len(run["per_run"]), 4)
        self.assertEqual(run["aggregated"]["test_f1"]["n_scores"], 4)
        self.assertEqual(run["per_run"][0]["artifacts"], {})

if __name__ == "__main__":
    unittest.main(verbosity=2)
