"""Discover and validate graph JSON files exposed by TDGBench."""

from __future__ import annotations

from difflib import get_close_matches
from pathlib import Path


DATASETS_DIR = Path(__file__).resolve().parents[1] / "datasets"


def available_graph_names(datasets_dir: Path = DATASETS_DIR) -> tuple[str, ...]:
    """Return every graph name stored as ``<name>.json`` in ``datasets_dir``."""
    return tuple(sorted(path.stem for path in datasets_dir.glob("*.json") if path.is_file()))


def validate_graph_name(kg_name: str, datasets_dir: Path = DATASETS_DIR) -> str:
    """Return a valid graph name or raise a helpful ``ValueError``."""
    if not isinstance(kg_name, str) or not kg_name.strip():
        raise ValueError("kg_name must be a non-empty graph name.")

    resolved_name = kg_name.strip()
    available = available_graph_names(datasets_dir)
    if resolved_name in available:
        return resolved_name

    suggestion = get_close_matches(resolved_name, available, n=1, cutoff=0.55)
    message = (
        f"Unknown graph name '{resolved_name}'. "
        f"Available graphs: {', '.join(available)}."
    )
    if suggestion:
        message += f" Did you mean '{suggestion[0]}'?"
    raise ValueError(message)
