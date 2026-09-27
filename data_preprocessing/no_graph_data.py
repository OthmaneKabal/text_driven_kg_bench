"""Data and feature preparation for the text-only no-graph baseline."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from data_preprocessing.initial_embeddings.BertEmbedder import BertEmbedder


ROOT = Path(__file__).resolve().parents[1]
FEATURE_CACHE_ROOT = ROOT / "embeddings_init" / "no_graph"
_RANDOM_SPEC = re.compile(r"random_(\d+)$")


@dataclass(frozen=True)
class NoGraphDataset:
    """The labelled terms used by a text-only node-classification baseline."""

    terms: list[str]
    labels: torch.Tensor
    label_encoder: LabelEncoder

    @property
    def num_classes(self) -> int:
        return len(self.label_encoder.classes_)


def load_no_graph_dataset(common_nodes_path: str | Path) -> NoGraphDataset:
    """Load labelled terms in the exact row order used by the split files."""
    path = Path(common_nodes_path)
    table = pd.read_excel(path)
    missing_columns = {"term", "label"} - set(table.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing required column(s): {sorted(missing_columns)}")
    table = table.dropna(subset=["term", "label"])
    terms = table["term"].astype(str).tolist()
    encoder = LabelEncoder()
    labels = torch.tensor(encoder.fit_transform(table["label"]), dtype=torch.long)
    return NoGraphDataset(terms=terms, labels=labels, label_encoder=encoder)


def load_split_indices(split_path: str | Path, num_terms: int) -> dict[str, torch.Tensor]:
    """Load and validate GS-row split indices without resolving graph nodes."""
    path = Path(split_path)
    with path.open(encoding="utf-8") as handle:
        split = json.load(handle)
    required = ("train_idx", "val_idx", "test_idx")
    if not isinstance(split, dict) or any(name not in split for name in required):
        raise ValueError(f"{path} must contain {list(required)}")
    result: dict[str, torch.Tensor] = {}
    used: set[int] = set()
    for name in required:
        values = split[name]
        if not isinstance(values, list) or not all(isinstance(value, int) for value in values):
            raise ValueError(f"{path}: {name} must be a list of integer GS indices")
        if any(value < 0 or value >= num_terms for value in values):
            raise ValueError(f"{path}: {name} contains an index outside [0, {num_terms - 1}]")
        overlap = used.intersection(values)
        if overlap:
            raise ValueError(f"{path}: split indices overlap (for example {min(overlap)})")
        used.update(values)
        result[name.removesuffix("_idx")] = torch.tensor(values, dtype=torch.long)
    return result


def random_embedding_dimension(init_embd: str, default_dimension: int) -> int | None:
    """Return a requested random-feature dimension, or ``None`` for an LM."""
    if init_embd == "random":
        if default_dimension < 1:
            raise ValueError("random_embedding_dim must be >= 1")
        return default_dimension
    match = _RANDOM_SPEC.fullmatch(init_embd)
    if match is None:
        return None
    dimension = int(match.group(1))
    if dimension < 1:
        raise ValueError("The random embedding dimension must be >= 1")
    return dimension


def _cache_path(init_embd: str, terms: list[str]) -> Path:
    term_digest = hashlib.sha256("\x1f".join(terms).encode("utf-8")).hexdigest()[:16]
    model_digest = hashlib.sha256(init_embd.encode("utf-8")).hexdigest()[:10]
    model_short_name = init_embd.rsplit("/", 1)[-1].replace(" ", "_")
    return FEATURE_CACHE_ROOT / f"{model_short_name}_{model_digest}" / f"terms_{term_digest}.pt"


def _language_model_features(terms: list[str], init_embd: str) -> torch.Tensor:
    cache_path = _cache_path(init_embd, terms)
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        features = cached.get("features") if isinstance(cached, dict) else None
        if (
            isinstance(cached, dict)
            and cached.get("model") == init_embd
            and cached.get("terms") == terms
            and isinstance(features, torch.Tensor)
            and features.ndim == 2
            and features.shape[0] == len(terms)
        ):
            print(f"[INFO] Loading cached no-graph features from {cache_path}")
            return features.float().cpu()

    print(f"[INFO] Building no-graph term features with '{init_embd}'")
    try:
        embedder = BertEmbedder(init_embd)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load language model '{init_embd}' for the no-graph baseline. "
            "Make it available in the local Hugging Face cache or restore network access; "
            "alternatively use 'random' or 'random_<dimension>'."
        ) from exc
    # The language model is an immutable feature extractor for this baseline:
    # disable dropout before caching its term representations.
    embedder.model.eval()
    rows = [embedder.embed_entity(term).detach().cpu().reshape(-1) for term in terms]
    features = torch.stack(rows).float()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": init_embd, "terms": terms, "features": features}, cache_path)
    print(f"[INFO] Saved no-graph features to {cache_path}")
    return features


def build_no_graph_features(
    terms: list[str],
    init_embd: str,
    random_seed: int,
    random_embedding_dim: int = 384,
) -> torch.Tensor:
    """Build fixed LM features or seed-specific independent random features."""
    random_dim = random_embedding_dimension(init_embd, random_embedding_dim)
    if random_dim is None:
        return _language_model_features(terms, init_embd)
    generator = torch.Generator(device="cpu").manual_seed(random_seed)
    return torch.randn((len(terms), random_dim), generator=generator, dtype=torch.float32)
