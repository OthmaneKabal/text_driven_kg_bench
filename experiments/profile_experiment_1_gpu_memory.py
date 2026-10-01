"""Profile peak GPU memory for the heavy Experiment 1 task classes.

Each graph/model profile is executed in a fresh Python process so that CUDA
allocations from a preceding model cannot contaminate its measurement.  The
parent process samples ``nvidia-smi`` while that child is training and writes
one row per ``graph x architecture`` pair.

The default workload is intentionally the memory upper bound used in
Experiment 1: h=512, one split, one random seed and one epoch.  It is a memory
profile, not an evaluation experiment.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GRAPHS = ("GT2KG_kg", "KG_GEN_kg", "UMLS_nci_kg")
MODELS = (
    "GCN", "GAT", "RGCN", "TransEGCN_conv", "RotatEGCN_conv",
    "TransEGCN_attn", "RotatEGCN_attn",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="+", default=GRAPHS)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--embedding", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=0.2)
    parser.add_argument("--budget-fraction", type=float, default=0.70)
    parser.add_argument(
        "--results-dir", default="results/resource_profiles/experiment_1_h512"
    )
    parser.add_argument("--resume", action="store_true")

    # Internal mode: invoked by the parent once per profile.
    parser.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--graph", help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    return parser.parse_args()


def run_single(args: argparse.Namespace) -> int:
    """Run exactly one training task; called only in a clean child process."""
    if not args.graph or not args.model:
        raise ValueError("--single requires --graph and --model")

    from tdg_bench import TDGBench

    TDGBench(use_classifier=True).evaluate_models(
        kg_name=args.graph,
        model_names=[args.model],
        init_embd=args.embedding,
        split_seeds=[args.split_seed],
        random_seeds=[args.random_seed],
        hidden_channels=[args.hidden],
        random_embd_dim=384,
        epochs=args.epochs,
        patience=args.epochs,
        verbose=False,
        save_models=False,
        save_predictions=False,
        resume=False,
        results_dir=str(Path(args.results_dir) / "training_artifacts"),
    )
    return 0


def gpu_memory_by_pid() -> dict[int, int]:
    """Return ``pid -> used MiB`` reported by nvidia-smi across visible GPUs."""
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {completed.stderr.strip()}")

    values: dict[int, int] = {}
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            values[int(parts[0])] = int(parts[1].replace(" MiB", ""))
        except ValueError:
            continue
    return values


def gpu_total_memory_mib(gpu_index: int) -> int:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--id",
            str(gpu_index),
            "--query-gpu=memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {completed.stderr.strip()}")
    return int(completed.stdout.strip().splitlines()[0].replace(" MiB", ""))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "graph", "model", "hidden_channels", "embedding", "epochs",
        "split_seed", "random_seed", "status", "return_code", "peak_vram_mib",
        "peak_vram_gib", "gpu_total_mib", "budget_fraction",
        "memory_only_worker_limit", "started_at_utc", "finished_at_utc", "log_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def child_command(args: argparse.Namespace, graph: str, model: str) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single",
        "--graph", graph,
        "--model", model,
        "--hidden", str(args.hidden),
        "--epochs", str(args.epochs),
        "--embedding", args.embedding,
        "--split-seed", str(args.split_seed),
        "--random-seed", str(args.random_seed),
        "--results-dir", args.results_dir,
    ]


def profile_task(
    args: argparse.Namespace, graph: str, model: str, gpu_total_mib: int, log_dir: Path
) -> dict[str, Any]:
    log_path = log_dir / f"{graph}__{model}__h{args.hidden}.log"
    started_at = datetime.now(timezone.utc).isoformat()
    peak_mib = 0
    with log_path.open("w", encoding="utf-8") as log_handle:
        child = subprocess.Popen(
            child_command(args, graph, model),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        while child.poll() is None:
            peak_mib = max(peak_mib, gpu_memory_by_pid().get(child.pid, 0))
            time.sleep(args.poll_seconds)
        peak_mib = max(peak_mib, gpu_memory_by_pid().get(child.pid, 0))

    return_code = child.returncode
    worker_limit = (
        math.floor((gpu_total_mib * args.budget_fraction) / peak_mib)
        if peak_mib > 0
        else 0
    )
    return {
        "graph": graph,
        "model": model,
        "hidden_channels": args.hidden,
        "embedding": args.embedding,
        "epochs": args.epochs,
        "split_seed": args.split_seed,
        "random_seed": args.random_seed,
        "status": "completed" if return_code == 0 and peak_mib > 0 else "failed_or_unmeasured",
        "return_code": return_code,
        "peak_vram_mib": peak_mib,
        "peak_vram_gib": round(peak_mib / 1024, 3),
        "gpu_total_mib": gpu_total_mib,
        "budget_fraction": args.budget_fraction,
        "memory_only_worker_limit": worker_limit,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "log_path": str(log_path),
    }


def main() -> int:
    args = parse_args()
    if args.single:
        return run_single(args)
    if args.hidden < 1 or args.epochs < 1:
        raise ValueError("hidden and epochs must be >= 1")
    if args.poll_seconds <= 0:
        raise ValueError("poll-seconds must be > 0")
    if not 0 < args.budget_fraction <= 1:
        raise ValueError("budget-fraction must be in (0, 1]")

    result_dir = Path(args.results_dir)
    result_path = result_dir / "gpu_memory_profile.csv"
    log_dir = result_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if args.resume and result_path.exists():
        with result_path.open(encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))
    completed = {
        (row["graph"], row["model"])
        for row in existing
        if row.get("status") == "completed"
    }

    gpu_total_mib = gpu_total_memory_mib(args.gpu_index)
    rows: list[dict[str, Any]] = existing.copy()
    tasks = [(graph, model) for graph in args.graphs for model in args.models]
    print(f"Profiling {len(tasks)} graph/model pairs on GPU {args.gpu_index} ({gpu_total_mib / 1024:.1f} GiB).")
    for number, (graph, model) in enumerate(tasks, 1):
        if (graph, model) in completed:
            print(f"[{number}/{len(tasks)}] already completed: {graph} | {model}")
            continue
        print(f"[{number}/{len(tasks)}] profiling: {graph} | {model} | h={args.hidden}")
        row = profile_task(args, graph, model, gpu_total_mib, log_dir)
        rows = [item for item in rows if (item["graph"], item["model"]) != (graph, model)]
        rows.append(row)
        write_rows(result_path, rows)
        print(
            f"  {row['status']}: peak={row['peak_vram_gib']} GiB, "
            f"memory-only limit={row['memory_only_worker_limit']}"
        )

    print(f"Saved GPU profile to: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
