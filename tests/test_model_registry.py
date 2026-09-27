"""Fast checks for the model registry without importing PyTorch model classes."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_models import ModelRegistry, available_model_names


class ModelRegistryTest(unittest.TestCase):
    def test_builtin_names_are_exposed(self) -> None:
        self.assertIn("GCN", available_model_names())
        self.assertIn("RGCN", available_model_names())
        self.assertIn("RotatEGCN_attn", available_model_names())

    def test_custom_builder_receives_common_arguments(self) -> None:
        registry = ModelRegistry()

        def builder(**kwargs):
            return kwargs

        registry.register("Example", builder)
        built = registry.build(
            "Example",
            in_channels=8,
            hidden_channels=16,
            out_channels=16,
            num_relations=3,
            dropout=0.5,
        )
        self.assertEqual(built["in_channels"], 8)
        self.assertEqual(built["num_relations"], 3)
        self.assertEqual(built["dropout"], 0.5)

    def test_duplicate_or_unknown_names_raise_clear_errors(self) -> None:
        registry = ModelRegistry()
        registry.register("Example", lambda **kwargs: kwargs)
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register("Example", lambda **kwargs: kwargs)
        with self.assertRaisesRegex(ValueError, "Unknown model_name"):
            registry.build("Missing", 8, 16, 16, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
