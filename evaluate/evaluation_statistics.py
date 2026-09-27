"""Hierarchical aggregation for repeated runs over fixed data splits."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def aggregate_split_and_randomness(
    scores_by_split: Mapping[int, Sequence[float]],
) -> dict[str, object]:
    """Aggregate scores while keeping split and training randomness separate.

    The final mean uses every score.  The split standard deviation is computed
    over the per-split means.  The randomness standard deviation is the mean
    of the within-split standard deviations.
    """
    non_empty = {
        int(split_seed): [float(score) for score in scores]
        for split_seed, scores in scores_by_split.items()
        if scores
    }
    if not non_empty:
        raise ValueError("At least one score is required for aggregation")

    all_scores = [score for scores in non_empty.values() for score in scores]
    per_split_means = {
        split_seed: float(np.mean(scores)) for split_seed, scores in non_empty.items()
    }
    per_split_randomness_stds = {
        split_seed: float(np.std(scores)) for split_seed, scores in non_empty.items()
    }
    split_std = float(np.std(list(per_split_means.values())))
    randomness_std = float(np.mean(list(per_split_randomness_stds.values())))
    return {
        "mean": float(np.mean(all_scores)),
        # Backward-compatible alias.  It now has the explicit interpretation
        # of between-split variability rather than a mixed run-level SD.
        "std": split_std,
        "split_std": split_std,
        "randomness_std": randomness_std,
        "overall_std": float(np.std(all_scores)),
        "n_scores": len(all_scores),
        "n_splits": len(non_empty),
        "per_split_means": {str(seed): value for seed, value in per_split_means.items()},
        "per_split_randomness_stds": {
            str(seed): value for seed, value in per_split_randomness_stds.items()
        },
        "values": all_scores,
    }
