"""Paired Student t-tests of completed TDG-Bench experiments.

This module is deliberately independent from training code. It reads saved
``results_*.json`` files, averages test Macro-F1 over training random seeds
inside each split, performs two-sided paired Student t-tests, then applies
Holm's correction separately for every graph.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import shapiro, t as student_t


DEFAULT_ALPHA = 0.05


@dataclass(frozen=True)
class ExperimentScores:
    """Test Macro-F1 values, averaged over random seeds within each split."""

    graph_name: str
    model_name: str
    source_file: Path
    scores_by_split: Mapping[int, float]
    random_seeds_by_split: Mapping[int, tuple[int, ...]]
    mean_validation_f1: float


@dataclass(frozen=True)
class ModelConfiguration:
    """Architecture and widths encoded in a TDG-Bench result filename."""

    architecture: str
    hidden_channels: int
    out_channels: int


_CONFIGURATION_PATTERN = re.compile(
    r"^(?P<architecture>.+)__h(?P<hidden_channels>\d+)__out(?P<out_channels>\d+)$"
)


def _model_name_from_file(result_file: Path, graph_name: str) -> str:
    """Recover the model/configuration name from TDG-Bench's result filename."""
    expected_prefix = f"results_{graph_name}__"
    stem = result_file.stem
    if stem.startswith(expected_prefix):
        name = stem[len(expected_prefix):]
        if name:
            return name

    # Benchmark results are normally stored as <graph>/<model>/<width>/file.
    if result_file.parent.parent.name:
        return result_file.parent.parent.name
    raise ValueError(
        f"Cannot infer the model name from {result_file}. Expected a TDG-Bench "
        "filename such as results_<graph>__<model>__h64__out64.json."
    )


def _parse_model_configuration(model_name: str) -> ModelConfiguration:
    """Parse TDG-Bench's ``<architecture>__h<d>__out<d>`` result label."""
    match = _CONFIGURATION_PATTERN.fullmatch(model_name)
    if match is None:
        raise ValueError(
            f"Cannot select an architecture representative from {model_name!r}. "
            "Expected a TDG-Bench configuration name such as RGCN__h64__out64."
        )
    return ModelConfiguration(
        architecture=match.group("architecture"),
        hidden_channels=int(match.group("hidden_channels")),
        out_channels=int(match.group("out_channels")),
    )


def load_experiment_scores(result_file: str | Path) -> ExperimentScores:
    """Read one TDG-Bench result JSON and average Macro-F1 per split.

    The function requires one score for each ``(split_seed, random_seed)``
    pair. Duplicate runs or malformed records are rejected instead of silently
    changing a paired comparison.
    """
    path = Path(result_file)
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)

    graph_name = result.get("graph_variant") or result.get("kg_name")
    if not isinstance(graph_name, str) or not graph_name:
        raise ValueError(f"{path} has no valid 'graph_variant' or 'kg_name'")
    per_run = result.get("per_run")
    if not isinstance(per_run, list) or not per_run:
        raise ValueError(f"{path} has no non-empty 'per_run' result list")

    values_by_split: dict[int, list[float]] = defaultdict(list)
    seeds_by_split: dict[int, list[int]] = defaultdict(list)
    validation_values: list[float] = []
    observed_pairs: set[tuple[int, int]] = set()
    for index, run in enumerate(per_run):
        try:
            split_seed = int(run["split_seed"])
            random_seed = int(run["random_seed"])
            macro_f1 = float(run["final_test"]["f1"])
            validation_f1 = float(run["best_val_f1"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Malformed test Macro-F1 record #{index} in {path}"
            ) from exc
        pair = (split_seed, random_seed)
        if pair in observed_pairs:
            raise ValueError(f"Duplicate (split_seed, random_seed)={pair} in {path}")
        observed_pairs.add(pair)
        values_by_split[split_seed].append(macro_f1)
        seeds_by_split[split_seed].append(random_seed)
        validation_values.append(validation_f1)

    random_seed_sets = {
        split_seed: tuple(sorted(seeds))
        for split_seed, seeds in seeds_by_split.items()
    }
    return ExperimentScores(
        graph_name=graph_name,
        model_name=_model_name_from_file(path, graph_name),
        source_file=path,
        scores_by_split={
            split_seed: float(np.mean(scores))
            for split_seed, scores in values_by_split.items()
        },
        random_seeds_by_split=random_seed_sets,
        mean_validation_f1=float(np.mean(validation_values)),
    )


def paired_student_t_test(differences: Sequence[float]) -> dict[str, float | int]:
    """Two-sided paired Student t-test of split-level score differences.

    Positive differences mean that the first model is better than the second
    on Macro-F1. The test is conditional on the benchmark's pre-specified
    split protocol; it uses the usual paired t-test standard error
    ``s_d / sqrt(J)``.
    """
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("A paired Student t-test requires at least two paired splits")
    if not np.all(np.isfinite(values)):
        raise ValueError("differences must all be finite")

    n_splits = len(values)
    mean_difference = float(np.mean(values))
    sample_variance = float(np.var(values, ddof=1))
    standard_error = float(np.sqrt(sample_variance / n_splits))
    shapiro_result = shapiro(values) if n_splits >= 3 else None

    if standard_error == 0.0:
        statistic = 0.0 if mean_difference == 0.0 else float(np.sign(mean_difference) * np.inf)
        p_value = 1.0 if mean_difference == 0.0 else 0.0
    else:
        statistic = mean_difference / standard_error
        p_value = float(2.0 * student_t.sf(abs(statistic), df=n_splits - 1))

    return {
        "n_splits": n_splits,
        "degrees_of_freedom": n_splits - 1,
        "mean_difference": mean_difference,
        "difference_std": float(np.sqrt(sample_variance)),
        "standard_error": standard_error,
        "t_statistic": float(statistic),
        "p_value_raw": p_value,
        "shapiro_w": float(shapiro_result.statistic) if shapiro_result else float("nan"),
        "shapiro_p_value": float(shapiro_result.pvalue) if shapiro_result else float("nan"),
    }


def holm_adjust(p_values: Sequence[float]) -> tuple[list[float], list[int]]:
    """Return Holm-adjusted p-values and their original-order ranks."""
    raw = np.asarray(p_values, dtype=float)
    if raw.ndim != 1 or not len(raw):
        raise ValueError("p_values must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(raw)) or np.any((raw < 0.0) | (raw > 1.0)):
        raise ValueError("Each p-value must be finite and between 0 and 1")

    order = np.argsort(raw, kind="stable")
    adjusted = np.empty(len(raw), dtype=float)
    ranks = np.empty(len(raw), dtype=int)
    running_max = 0.0
    for rank, index in enumerate(order, start=1):
        corrected = min(1.0, (len(raw) - rank + 1) * raw[index])
        running_max = max(running_max, corrected)
        adjusted[index] = running_max
        ranks[index] = rank
    return adjusted.tolist(), ranks.tolist()


def _validate_matched_design(
    experiments: Sequence[ExperimentScores],
    graph_name: str,
) -> None:
    """Ensure candidates used in one comparison had the same experiment design."""
    reference = experiments[0]
    reference_splits = set(reference.scores_by_split)
    for experiment in experiments[1:]:
        experiment_splits = set(experiment.scores_by_split)
        if experiment_splits != reference_splits:
            raise ValueError(
                f"Configurations {reference.model_name!r} and {experiment.model_name!r} "
                f"for {graph_name!r} do not have the same split seeds."
            )
        for split_seed in reference_splits:
            if (
                experiment.random_seeds_by_split[split_seed]
                != reference.random_seeds_by_split[split_seed]
            ):
                raise ValueError(
                    f"Configurations {reference.model_name!r} and {experiment.model_name!r} "
                    f"for {graph_name!r} do not use the same random seeds in "
                    f"split {split_seed}."
                )


def select_architecture_representatives(
    experiments: Sequence[ExperimentScores],
    graph_name: str,
) -> list[ExperimentScores]:
    """Select one width per architecture by mean best-validation Macro-F1.

    The chosen configuration is the representative of its architecture for the
    test-set comparison. Exact validation-score ties are resolved by the
    smaller hidden width, then smaller output width, to avoid consulting test
    scores.
    """
    by_architecture: dict[str, list[ExperimentScores]] = defaultdict(list)
    configurations: dict[str, ModelConfiguration] = {}
    for experiment in experiments:
        configuration = _parse_model_configuration(experiment.model_name)
        configurations[experiment.model_name] = configuration
        by_architecture[configuration.architecture].append(experiment)

    representatives: list[ExperimentScores] = []
    for architecture, candidates in sorted(by_architecture.items()):
        _validate_matched_design(candidates, graph_name)
        representatives.append(
            min(
                candidates,
                key=lambda candidate: (
                    -candidate.mean_validation_f1,
                    configurations[candidate.model_name].hidden_channels,
                    configurations[candidate.model_name].out_channels,
                    candidate.model_name,
                ),
            )
        )
    return representatives


def compare_models_by_graph(
    result_files: Iterable[str | Path],
    alpha: float = DEFAULT_ALPHA,
    comparison_unit: str = "representative",
) -> pd.DataFrame:
    """Compare model pairs and apply Holm independently per graph.

    With the default ``comparison_unit='representative'``, configurations are
    first grouped by architecture and the configuration with the greatest mean
    validation Macro-F1 is retained. The paired test therefore compares one
    validation-selected representative per architecture. Set
    ``comparison_unit='configuration'`` to compare every hidden/output
    configuration instead.

    Result files from multiple graphs may be supplied together. The returned
    table contains one row per pair, with raw and graph-wise Holm-adjusted
    paired Student t-test p-values. All compared results must use identical
    split seeds and identical random seed sets inside each split.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    if comparison_unit not in {"representative", "configuration"}:
        raise ValueError(
            "comparison_unit must be 'representative' or 'configuration'"
        )
    experiments = [load_experiment_scores(path) for path in result_files]
    if len(experiments) < 2:
        raise ValueError("At least two result files are required for comparison")

    by_graph: dict[str, list[ExperimentScores]] = defaultdict(list)
    for experiment in experiments:
        by_graph[experiment.graph_name].append(experiment)

    rows: list[dict[str, Any]] = []
    for graph_name, graph_experiments in sorted(by_graph.items()):
        seen_models: set[str] = set()
        for experiment in graph_experiments:
            if experiment.model_name in seen_models:
                raise ValueError(
                    f"Duplicate model configuration {experiment.model_name!r} "
                    f"for graph {graph_name!r}. Provide one result JSON per configuration."
                )
            seen_models.add(experiment.model_name)

        if comparison_unit == "representative":
            compared_experiments = select_architecture_representatives(
                graph_experiments,
                graph_name,
            )
        else:
            compared_experiments = graph_experiments

        graph_rows: list[dict[str, Any]] = []
        for first, second in combinations(compared_experiments, 2):
            first_splits = set(first.scores_by_split)
            second_splits = set(second.scores_by_split)
            if first_splits != second_splits:
                raise ValueError(
                    f"Models {first.model_name!r} and {second.model_name!r} for "
                    f"{graph_name!r} do not have the same split seeds: "
                    f"{sorted(first_splits)} vs {sorted(second_splits)}"
                )
            split_seeds = sorted(first_splits)
            for split_seed in split_seeds:
                if (
                    first.random_seeds_by_split[split_seed]
                    != second.random_seeds_by_split[split_seed]
                ):
                    raise ValueError(
                        f"Models {first.model_name!r} and {second.model_name!r} "
                        f"do not use the same random seeds for split {split_seed}."
                    )

            differences = [
                first.scores_by_split[seed] - second.scores_by_split[seed]
                for seed in split_seeds
            ]
            statistics = paired_student_t_test(differences)
            first_configuration = _parse_model_configuration(first.model_name)
            second_configuration = _parse_model_configuration(second.model_name)
            graph_rows.append(
                {
                    "graph_name": graph_name,
                    "comparison_unit": comparison_unit,
                    "model_a": (
                        first_configuration.architecture
                        if comparison_unit == "representative"
                        else first.model_name
                    ),
                    "model_b": (
                        second_configuration.architecture
                        if comparison_unit == "representative"
                        else second.model_name
                    ),
                    "configuration_a": first.model_name,
                    "configuration_b": second.model_name,
                    "hidden_channels_a": first_configuration.hidden_channels,
                    "hidden_channels_b": second_configuration.hidden_channels,
                    "out_channels_a": first_configuration.out_channels,
                    "out_channels_b": second_configuration.out_channels,
                    "mean_validation_f1_a": first.mean_validation_f1,
                    "mean_validation_f1_b": second.mean_validation_f1,
                    "mean_macro_f1_a": float(np.mean(list(first.scores_by_split.values()))),
                    "mean_macro_f1_b": float(np.mean(list(second.scores_by_split.values()))),
                    "split_seeds": ",".join(map(str, split_seeds)),
                    "random_seeds": ",".join(
                        map(str, first.random_seeds_by_split[split_seeds[0]])
                    ),
                    **statistics,
                }
            )

        if graph_rows:
            adjusted, ranks = holm_adjust([row["p_value_raw"] for row in graph_rows])
            for row, adjusted_p, rank in zip(graph_rows, adjusted, ranks):
                row["holm_rank_within_graph"] = rank
                row["p_value_holm"] = adjusted_p
                row["significant_holm"] = bool(adjusted_p <= alpha)
                row["alpha"] = alpha
            rows.extend(graph_rows)

    if not rows:
        raise ValueError("No graph contains at least two model result files")
    return pd.DataFrame(rows).sort_values(
        ["graph_name", "p_value_holm", "model_a", "model_b"],
        kind="stable",
    ).reset_index(drop=True)


def find_result_files(results_dir: str | Path) -> list[Path]:
    """Find TDG-Bench detailed result JSONs below a benchmark results folder."""
    directory = Path(results_dir)
    files = sorted(directory.rglob("results_*.json"))
    if not files:
        raise FileNotFoundError(f"No results_*.json files found below {directory}")
    return files


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-sided paired Student t-tests with Holm correction per graph."
    )
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument(
        "--results-dir",
        help="Benchmark folder to scan recursively for results_*.json files.",
    )
    sources.add_argument(
        "--result-files",
        nargs="+",
        help="Explicit detailed result JSON files to compare.",
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument(
        "--comparison-unit",
        choices=["representative", "configuration"],
        default="representative",
        help=(
            "'representative' selects one hidden/output configuration per "
            "architecture by mean validation F1; 'configuration' compares all."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV output path. Defaults to <results-dir>/paired_t_holm.csv.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    files = (
        find_result_files(args.results_dir)
        if args.results_dir is not None
        else [Path(path) for path in args.result_files]
    )
    table = compare_models_by_graph(files, alpha=args.alpha, comparison_unit=args.comparison_unit)
    output = (
        Path(args.output)
        if args.output is not None
        else Path(args.results_dir) / "paired_t_holm.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(table.to_string(index=False))
    print(f"\nSaved graph-wise Holm-corrected paired t-tests to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
