"""Small executable test for anchor-guided Frontier Sampling."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data_preprocessing.clean_graph_sampling import (
    sample_scale_matched_clean_graph,
    sample_scale_matched_clean_graphs,
)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


class CleanGraphSamplingTest(unittest.TestCase):
    def test_anchors_are_kept_and_target_node_count_is_matched_without_saving(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            clean_path = directory / "clean.json"
            target_path = directory / "target.json"
            common_path = directory / "common.xlsx"
            semantic_path = directory / "types.json"
            output_path = directory / "must_not_exist.json"

            clean = [
                {"subject": "Alpha", "predicate": "rel_a", "object": "Beta", "subject_type": "Gene", "object_type": "Disease"},
                {"subject": "Beta", "predicate": "rel_b", "object": "Gamma", "subject_type": "Disease", "object_type": "Disease"},
                {"subject": "Gamma", "predicate": "rel_a", "object": "Delta", "subject_type": "Disease", "object_type": "Chemical"},
                {"subject": "Delta", "predicate": "rel_c", "object": "Epsilon", "subject_type": "Chemical", "object_type": "Gene"},
                {"subject": "Alpha", "predicate": "isa", "object": "Type A", "synthetic": True},
            ]
            target = [
                {"subject": "one", "predicate": "p", "object": "two"},
                {"subject": "two", "predicate": "p", "object": "three"},
                {"subject": "three", "predicate": "p", "object": "four"},
                {"subject": "four", "predicate": "p", "object": "five"},
                {"subject": "five", "predicate": "p", "object": "six"},
            ]
            write_json(clean_path, clean)
            write_json(target_path, target)
            pd.DataFrame({"term": ["Alpha"]}).to_excel(common_path, index=False)
            write_json(
                semantic_path,
                [{"normalized_term": "alpha", "semantic_type": "Type A", "semantic_type_tui": "T001"}],
            )

            graph, report = sample_scale_matched_clean_graph(
                clean_path,
                target_path,
                common_nodes_path=common_path,
                semantic_types_path=semantic_path,
                candidates=3,
                with_stats=True,
            )

            nodes = {record[field].casefold() for record in graph for field in ("subject", "object")}
            self.assertIn("alpha", nodes)
            self.assertIn("type a", nodes)
            self.assertEqual(report["target_nodes"], 6)
            self.assertEqual(report["sample_nodes"], 6)
            self.assertEqual(report["mandatory_anchors_present"], 2)
            self.assertFalse(output_path.exists())

    def test_top_k_candidates_are_ranked_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            clean_path = directory / "clean.json"
            target_path = directory / "target.json"
            common_path = directory / "common.xlsx"
            semantic_path = directory / "types.json"

            clean = [
                {"subject": "Alpha", "predicate": "rel_a", "object": "Beta"},
                {"subject": "Beta", "predicate": "rel_b", "object": "Gamma"},
                {"subject": "Gamma", "predicate": "rel_c", "object": "Delta"},
                {"subject": "Delta", "predicate": "rel_a", "object": "Epsilon"},
                {"subject": "Alpha", "predicate": "isa", "object": "Type A", "synthetic": True},
            ]
            target = [
                {"subject": "one", "predicate": "p", "object": "two"},
                {"subject": "two", "predicate": "p", "object": "three"},
                {"subject": "three", "predicate": "p", "object": "four"},
                {"subject": "four", "predicate": "p", "object": "five"},
                {"subject": "five", "predicate": "p", "object": "six"},
            ]
            write_json(clean_path, clean)
            write_json(target_path, target)
            pd.DataFrame({"term": ["Alpha"]}).to_excel(common_path, index=False)
            write_json(semantic_path, [{"normalized_term": "alpha", "semantic_type": "Type A"}])

            ranked = sample_scale_matched_clean_graphs(
                clean_path,
                target_path,
                common_nodes_path=common_path,
                semantic_types_path=semantic_path,
                candidates=3,
                top_k=2,
                with_stats=True,
            )

            self.assertEqual(len(ranked), 2)
            reports = [report for _graph, report in ranked]
            self.assertEqual([report["selected_candidate_rank"] for report in reports], [1, 2])
            self.assertLessEqual(
                reports[0]["selected_candidate_score"], reports[1]["selected_candidate_score"]
            )
            selected_seeds = {report["selected_candidate_seed"] for report in reports}
            self.assertEqual(len(selected_seeds), 2)
            self.assertTrue(selected_seeds.issubset({42, 43, 44}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
