import json
from collections import defaultdict
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from tdg_bench import TDGBench


# ------------------------------------------------------------------------------
# Baseline encoder — swap this class to test other encoders.
# Currently: pure identity, no learned parameters.
# The only trainable component is the linear classifier in StandardClassifier.
# -------------------------------------------------------------------------------

class IdentityEncoder(nn.Module):
    """Returns embeddings unchanged. Swap to test learned projections."""
    def __init__(self, in_channels):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x, edge_index, edge_weight=None):
        return x


def main():
    # -----------------------------
    # 1) Paramètres du sweep
    # -----------------------------
    seeds = [42, 123, 456, 789, 2024]
    splits_dir = "datasets/split"

    # Any graph works — the baseline ignores graph structure entirely,
    # so results are identical across graphs (same labeled nodes).
    kg_name = "GT2KG_kg"

    init_embds = [
        "sentence-transformers/all-MiniLM-L6-v2",
        # "distilbert/distilbert-base-uncased",
        # "google-bert/bert-base-cased",
        # "allenai/scibert_scivocab_uncased",
        # "pritamdeka/S-BioBert-snli-multinli-stsb",
        # "sentence-transformers/all-MiniLM-L6-v2",
        "random_42",
        "random_123",
        "random_456",
    ]

    epochs = 100
    patience = 100
    lr = 0.01
    weight_decay = 5e-4

    # -----------------------------
    # 2) Setup sorties globales
    # -----------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    global_dir = Path("results") / f"baseline_{timestamp}"
    global_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        "meta": {
            "timestamp": timestamp,
            "seeds": seeds,
            "init_embds": init_embds,
            "splits_dir": splits_dir,
            "epochs": epochs,
            "patience": patience,
            "lr": lr,
            "weight_decay": weight_decay,
            "model_name": "linear_baseline",
            "hidden_channels": None,
            "use_classifier": True,
        },
        "runs": []
    }

    summary_rows = []

    # -----------------------------
    # 3) Boucle sweep
    # -----------------------------
    total_embds = len(init_embds)
    print(f"[BASELINE] {total_embds} embedding(s) = {total_embds} runs total (graph dimension removed)")
    print(f"[BASELINE] Each run evaluated over {len(seeds)} seed(s)\n")

    for embd_idx, init_embd in enumerate(init_embds, 1):
        embd_short = init_embd.split("/")[-1]
        run_id = f"{embd_short}__linear_baseline"

        print(f"\n[{embd_idx}/{total_embds}] Loading | embd: {embd_short}")
        tdg = TDGBench(use_classifier=True)

        # Load once to get in_channels
        annotated_graph, _, _, _, _ = tdg.get_data(
            kg_name=kg_name,
            init_embd=init_embd,
            split_path=f"{splits_dir}/split_{seeds[0]}.json",
        )
        in_channels = annotated_graph.x.shape[1]
        num_classes = tdg.config["num_classes"]

        print("\n" + "#" * 80)
        print(f"# EMBEDDING: {embd_short}")
        print(f"# Nodes: {annotated_graph.num_nodes} | in_channels={in_channels} | num_classes={num_classes}")
        print("#" * 80)

        per_run_dir = global_dir / embd_short
        per_run_dir.mkdir(parents=True, exist_ok=True)

        def model_factory(_in=in_channels):
            return IdentityEncoder(_in)

        print("\n" + "=" * 80)
        print(f"RUN: {run_id}  [embd {embd_idx}/{total_embds}]")
        print(f"     in_channels={in_channels}, num_classes={num_classes}")
        print("=" * 80)

        results = tdg.evaluate_all(
            kg_name=kg_name,
            model_factory=model_factory,
            init_embd=init_embd,
            seeds=seeds,
            splits_dir=splits_dir,
            epochs=epochs,
            patience=patience,
            lr=lr,
            weight_decay=weight_decay,
            verbose=True,
            save_results=True,
            results_dir=str(per_run_dir),
            run_id=run_id
        )

        all_results["runs"].append({
            "run_id": run_id,
            "init_embd": init_embd,
            "model_name": "linear_baseline",
            "hidden_channels": None,
            "in_channels": in_channels,
            "num_classes": num_classes,
            "results": results,
        })

        agg = results.get("aggregated", {})
        summary_rows.append({
            "run_id": run_id,
            "init_embd": init_embd,
            "model_name": "linear_baseline",
            "test_acc_mean": agg.get("test_acc", {}).get("mean", None),
            "test_acc_std": agg.get("test_acc", {}).get("std", None),
            "test_f1_mean": agg.get("test_f1", {}).get("mean", None),
            "test_f1_std": agg.get("test_f1", {}).get("std", None),
            "val_f1_mean": agg.get("val_f1", {}).get("mean", None),
            "val_f1_std": agg.get("val_f1", {}).get("std", None),
            "best_epoch_mean": agg.get("best_epoch", {}).get("mean", None),
            "best_epoch_std": agg.get("best_epoch", {}).get("std", None),
        })

    # -----------------------------
    # 4) Sauvegardes globales
    # -----------------------------

    # Merge random_* rows into a single "random" entry
    non_random_rows = [r for r in summary_rows if not r["init_embd"].startswith("random_")]
    random_rows     = [r for r in summary_rows if r["init_embd"].startswith("random_")]

    if random_rows:
        metrics = ["test_acc", "test_f1", "val_f1", "best_epoch"]
        random_values = {m: [] for m in metrics}
        for run in all_results["runs"]:
            if run["init_embd"].startswith("random_"):
                agg = run["results"].get("aggregated", {})
                for m in metrics:
                    random_values[m].extend(agg.get(m, {}).get("values", []))

        vals = random_values
        merged_random_rows = [{
            "run_id":          "random__linear_baseline",
            "init_embd":       "random",
            "model_name":      "linear_baseline",
            "test_acc_mean":   float(np.mean(vals["test_acc"]))   if vals["test_acc"]   else None,
            "test_acc_std":    float(np.std(vals["test_acc"]))    if vals["test_acc"]   else None,
            "test_f1_mean":    float(np.mean(vals["test_f1"]))    if vals["test_f1"]    else None,
            "test_f1_std":     float(np.std(vals["test_f1"]))     if vals["test_f1"]    else None,
            "val_f1_mean":     float(np.mean(vals["val_f1"]))     if vals["val_f1"]     else None,
            "val_f1_std":      float(np.std(vals["val_f1"]))      if vals["val_f1"]     else None,
            "best_epoch_mean": float(np.mean(vals["best_epoch"])) if vals["best_epoch"] else None,
            "best_epoch_std":  float(np.std(vals["best_epoch"]))  if vals["best_epoch"] else None,
        }]

        summary_rows = non_random_rows + merged_random_rows

    print(f"\n[BASELINE] All {len(all_results['runs'])} runs completed. Saving global results...")
    json_path = global_dir / "all_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    df = pd.DataFrame(summary_rows)
    csv_path = global_dir / "all_results_summary.csv"
    df.to_csv(csv_path, index=False)

    print("\n" + "#" * 80)
    print("BASELINE DONE ✅")
    print(f"Global JSON : {json_path}")
    print(f"Summary CSV : {csv_path}")
    print("#" * 80)


if __name__ == "__main__":
    main()
