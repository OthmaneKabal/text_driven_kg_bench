"""Unit test for per-(split, random) persistence and retry behaviour."""
from __future__ import annotations

import tempfile
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tdg_bench import TDGBench


def fake_result() -> dict:
    return {
        "best_val": {"f1": 0.5, "epoch": 1},
        "final_test": {"accuracy": 0.6, "recall": 0.6, "precision": 0.6, "f1": 0.6},
        "history": {
            "train_acc": [0.7], "train_f1": [0.7],
            "val_acc": [0.5], "val_f1": [0.5],
        },
        "artifacts": {},
    }


class EvaluateResumeTests(unittest.TestCase):
    def test_relaunch_skips_completed_pairs_and_retries_only_failures(self) -> None:
        bench = object.__new__(TDGBench)
        attempts: list[str] = []
        fail_once = {"value": True}

        def evaluate(**kwargs):
            split_path = kwargs["split_path"]
            attempts.append(split_path + kwargs["artifact_prefix"])
            if kwargs["artifact_prefix"].endswith("r2") and fail_once["value"]:
                fail_once["value"] = False
                raise RuntimeError("transient failure")
            return fake_result()

        bench.evaluate = evaluate
        # Keep the temporary output inside the repository: the Windows test
        # environment may deny writes through its redirected system TEMP path.
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            common = dict(
                kg_name="Example_kg",
                model_factory=lambda: None,
                init_embd="random_42",
                split_seeds=[42],
                random_seeds=[1, 2],
                save_models=False,
                save_predictions=False,
                verbose=False,
                results_dir=directory,
                run_id="resume_test",
            )
            first = bench.evaluate_all(**common)
            self.assertEqual(len(first["per_run"]), 1)
            self.assertEqual(len(first["failed_runs"]), 1)
            second = bench.evaluate_all(**common)

        self.assertEqual(len(second["per_run"]), 2)
        self.assertEqual(second["failed_runs"], [])
        self.assertEqual(len(attempts), 3)  # r1 once; r2 fails then retries.


if __name__ == "__main__":
    unittest.main(verbosity=2)
