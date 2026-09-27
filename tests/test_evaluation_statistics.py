"""Executable tests for the two-level benchmark aggregation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate.evaluation_statistics import aggregate_split_and_randomness


class EvaluationStatisticsTest(unittest.TestCase):
    def test_mean_and_two_variance_sources_are_separate(self) -> None:
        stats = aggregate_split_and_randomness({42: [1.0, 3.0], 123: [5.0, 7.0]})
        self.assertEqual(stats["mean"], 4.0)  # Mean of all four scores.
        self.assertEqual(stats["split_std"], 2.0)  # SD of split means: [2, 6].
        self.assertEqual(stats["randomness_std"], 1.0)  # Mean of intra-split SDs.
        self.assertEqual(stats["n_scores"], 4)
        self.assertEqual(stats["n_splits"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
