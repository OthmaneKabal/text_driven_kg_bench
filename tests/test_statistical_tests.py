"""Tests for file-based paired t-tests and graph-wise Holm comparisons."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate.statistical_tests import (
    compare_models_by_graph,
    holm_adjust,
    paired_student_t_test,
)


class StatisticalTestsTest(unittest.TestCase):
    def test_paired_student_t_uses_the_usual_paired_standard_error(self) -> None:
        result = paired_student_t_test([0.1, 0.2, 0.3])
        self.assertEqual(result["n_splits"], 3)
        self.assertEqual(result["degrees_of_freedom"], 2)
        self.assertAlmostEqual(result["standard_error"], 0.1 / (3 ** 0.5))
        self.assertGreater(result["p_value_raw"], 0)
        self.assertLessEqual(result["p_value_raw"], 1)

    def test_holm_adjustment_is_monotonic_in_rank_order(self) -> None:
        adjusted, ranks = holm_adjust([0.01, 0.03, 0.04])
        self.assertEqual(ranks, [1, 2, 3])
        self.assertEqual(adjusted, [0.03, 0.06, 0.06])

    def test_file_results_are_paired_and_holm_corrected_per_graph(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            def write_result(model_name: str, values: dict[int, list[float]]) -> Path:
                runs = [
                    {
                        "split_seed": split_seed,
                        "random_seed": random_seed,
                        "best_val_f1": score,
                        "final_test": {"f1": score},
                    }
                    for split_seed, scores in values.items()
                    for random_seed, score in enumerate(scores, start=1)
                ]
                result_file = root / f"results_Example_kg__{model_name}__h64__out64.json"
                result_file.write_text(
                    json.dumps({"graph_variant": "Example_kg", "per_run": runs}),
                    encoding="utf-8",
                )
                return result_file

            first = write_result("GCN", {42: [0.4, 0.6], 123: [0.6, 0.8]})
            second = write_result("RGCN", {42: [0.2, 0.4], 123: [0.4, 0.6]})
            table = compare_models_by_graph([first, second])

            self.assertEqual(len(table), 1)
            self.assertEqual(table.loc[0, "graph_name"], "Example_kg")
            self.assertAlmostEqual(table.loc[0, "mean_difference"], 0.2)
            self.assertEqual(table.loc[0, "holm_rank_within_graph"], 1)
            self.assertAlmostEqual(table.loc[0, "p_value_holm"], table.loc[0, "p_value_raw"])

    def test_representative_is_selected_from_validation_not_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            def write_result(
                name: str,
                validation_f1: float,
                test_f1: float,
            ) -> Path:
                result_file = root / f"results_Example_kg__{name}.json"
                result_file.write_text(
                    json.dumps(
                        {
                            "graph_variant": "Example_kg",
                            "per_run": [
                                {
                                    "split_seed": split_seed,
                                    "random_seed": random_seed,
                                    "best_val_f1": validation_f1,
                                    "final_test": {"f1": test_f1},
                                }
                                for split_seed in [42, 123]
                                for random_seed in [1, 2]
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                return result_file

            # RGCN h128 has lower test F1 but higher validation F1: it must win.
            files = [
                write_result("RGCN__h64__out64", validation_f1=0.40, test_f1=0.90),
                write_result("RGCN__h128__out128", validation_f1=0.50, test_f1=0.20),
                write_result("GAT__h64__out64", validation_f1=0.45, test_f1=0.30),
            ]
            table = compare_models_by_graph(files)

            self.assertEqual(len(table), 1)
            self.assertEqual(table.loc[0, "model_a"], "GAT")
            self.assertEqual(table.loc[0, "model_b"], "RGCN")
            self.assertEqual(table.loc[0, "configuration_b"], "RGCN__h128__out128")
            self.assertAlmostEqual(table.loc[0, "mean_macro_f1_b"], 0.20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
