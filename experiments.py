import json
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

from build_models import build_encoder, get_default_model_kwargs
from tdg_bench import TDGBench


# ---------------------------------------------------------------------
# A FOURNIR PAR TOI :
# build_encoder(model_name, in_channels, hidden_channels, out_channels, num_relations, **kwargs)
# -> retourne un encoder torch.nn.Module (GCNEncoder / RGCNEncoder / GATEncoder / etc.)
# ---------------------------------------------------------------------
#


def main():
    # -----------------------------
    # 1) Paramètres du sweep
    # -----------------------------
    seeds = [42, 123, 456, 789, 2024]
    splits_dir = "datasets/split"

    # Liste des graphes (noms kg)
    graphs = [
        # "REF_kg",
        "GT2KG_kg",
        "KG_GEN_kg",
    ]

    # Embedding models to sweep — "random" uses random embeddings, any HF model name uses BERT embeddings
    init_embds = [
        "distilbert/distilbert-base-uncased",
        "google-bert/bert-base-cased",   
        "allenai/scibert_scivocab_uncased",
        "pritamdeka/S-BioBert-snli-multinli-stsb",
        "sentence-transformers/all-MiniLM-L6-v2",
        "random_42",
        "random_123",
        "random_456",
    ]

    # Best GNN model per knowledge graph.
    BEST_GNN_PER_GRAPH = {
        # "REF_kg": "RGCN",
        "GT2KG_kg":  "RotatEGCN_attn",
        "KG_GEN_kg": "RotatEGCN_attn",
    }

    hidden_grid = [256, 64, 128]  # sweep hidden size
    out_channels = 256  # souvent = dim embedding final
    epochs = 100
    patience = 100
    lr = 0.01
    weight_decay = 5e-4

    # kwargs optionnels par modèle
    # ex: num_layers, dropout, heads, num_bases, etc.


    # -----------------------------
    # 2) Setup sorties globales
    # -----------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    global_dir = Path("results") / f"sweep_{timestamp}"
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
            "use_classifier": True,
        },
        "runs": []
    }

    summary_rows = []

    # -----------------------------
    # 3) Boucle sweep
    # -----------------------------
    total_graphs = len(graphs)
    total_embds = len(init_embds)
    total_hidden = len(hidden_grid)
    total_runs = total_graphs * total_embds * total_hidden
    print(f"[SWEEP] {total_graphs} graph(s) x {total_embds} embedding(s) x {total_hidden} hidden size(s) = {total_runs} runs total (1 best GNN per graph)")
    print(f"[SWEEP] Each run evaluated over {len(seeds)} seed(s)\n")

    for graph_idx, kg_name in enumerate(graphs, 1):
        for embd_idx, init_embd in enumerate(init_embds, 1):
            embd_short = init_embd.split("/")[-1]

            print(f"\n[GRAPH {graph_idx}/{total_graphs}] [{embd_idx}/{total_embds}] Loading graph: {kg_name} | embd: {embd_short}")
            tdg = TDGBench(use_classifier=True)

            # Charger une fois pour récupérer dimensions
            print(f"[GRAPH {graph_idx}/{total_graphs}] Building graph and loading split {seeds[0]}...")
            annotated_graph, _, _, _, _ = tdg.get_data(
                kg_name=kg_name,
                init_embd=init_embd,
                split_path=f"{splits_dir}/split_{seeds[0]}.json",
            )
            print(f"[GRAPH {graph_idx}/{total_graphs}] Graph ready.")
            in_channels = annotated_graph.x.shape[1]
            num_relations = len(set(annotated_graph.edge_type.tolist()))

            print("\n" + "#" * 80)
            print(f"# GRAPH: {kg_name} | EMBEDDING: {embd_short}")
            print(f"# Nodes: {annotated_graph.num_nodes}")
            print(f"# Edges: {annotated_graph.num_edges}")
            print(f"# in_channels={in_channels} | num_relations={num_relations}")
            print("#" * 80)

            model_name = BEST_GNN_PER_GRAPH[kg_name]
            extra_kwargs = get_default_model_kwargs(model_name)

            for hidden_channels in hidden_grid:
                run_id = f"{kg_name}__{embd_short}__{model_name}__h{hidden_channels}"

                per_run_dir = global_dir / kg_name / embd_short / model_name / f"h{hidden_channels}"
                per_run_dir.mkdir(parents=True, exist_ok=True)

                # factory => nouveau modèle à chaque seed
                def model_factory(
                        _model_name=model_name,
                        _in=in_channels,
                        _h=hidden_channels,
                        _nr=num_relations,
                        _kwargs=extra_kwargs
                ):
                    return build_encoder(
                        model_name=_model_name,
                        in_channels=_in,
                        hidden_channels=_h,
                        out_channels=_h,
                        num_relations=_nr,
                        **_kwargs
                    )

                print("\n" + "=" * 80)
                print(f"RUN: {run_id}  [graph {graph_idx}/{total_graphs}, embd {embd_idx}/{total_embds}, hidden={hidden_channels}]")
                print(f"     in_channels={in_channels}, out_channels={hidden_channels}, num_relations={num_relations}")
                print(f"     extra_kwargs={extra_kwargs}")
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

                # Stockage global (JSON)
                all_results["runs"].append({
                    "run_id": run_id,
                    "kg_name": kg_name,
                    "init_embd": init_embd,
                    "model_name": model_name,
                    "hidden_channels": hidden_channels,
                    "out_channels": out_channels,
                    "in_channels": in_channels,
                    "num_relations": num_relations,
                    "extra_kwargs": extra_kwargs,
                    "results": results,
                })

                # Ligne de summary (CSV)
                agg = results.get("aggregated", {})
                row = {
                    "run_id": run_id,
                    "kg_name": kg_name,
                    "init_embd": init_embd,
                    "model_name": model_name,
                    "hidden_channels": hidden_channels,
                    "test_acc_mean": agg.get("test_acc", {}).get("mean", None),
                    "test_acc_std": agg.get("test_acc", {}).get("std", None),
                    "test_f1_mean": agg.get("test_f1", {}).get("mean", None),
                    "test_f1_std": agg.get("test_f1", {}).get("std", None),
                    "val_f1_mean": agg.get("val_f1", {}).get("mean", None),
                    "val_f1_std": agg.get("val_f1", {}).get("std", None),
                    "best_epoch_mean": agg.get("best_epoch", {}).get("mean", None),
                    "best_epoch_std": agg.get("best_epoch", {}).get("std", None),
                }
                summary_rows.append(row)

    # -----------------------------
    # 4) Sauvegardes globales
    # -----------------------------

    # Merge random_* summary rows into a single "random" entry per (kg_name, model_name, hidden_channels)
    non_random_rows = [r for r in summary_rows if not r["init_embd"].startswith("random_")]
    random_rows     = [r for r in summary_rows if r["init_embd"].startswith("random_")]

    if random_rows:
        # Pool per-seed values from all matching runs in all_results
        metrics = ["test_acc", "test_f1", "val_f1", "best_epoch"]
        random_values_map = defaultdict(lambda: {m: [] for m in metrics})
        for run in all_results["runs"]:
            if run["init_embd"].startswith("random_"):
                key = (run["kg_name"], run["model_name"], run["hidden_channels"])
                agg = run["results"].get("aggregated", {})
                for m in metrics:
                    random_values_map[key][m].extend(agg.get(m, {}).get("values", []))

        seen_keys = set()
        merged_random_rows = []
        for r in random_rows:
            key = (r["kg_name"], r["model_name"], r["hidden_channels"])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            vals = random_values_map[key]
            merged_random_rows.append({
                "run_id":           f"{r['kg_name']}__random__{r['model_name']}__h{r['hidden_channels']}",
                "kg_name":          r["kg_name"],
                "init_embd":        "random",
                "model_name":       r["model_name"],
                "hidden_channels":  r["hidden_channels"],
                "test_acc_mean":    float(np.mean(vals["test_acc"]))    if vals["test_acc"]    else None,
                "test_acc_std":     float(np.std(vals["test_acc"]))     if vals["test_acc"]    else None,
                "test_f1_mean":     float(np.mean(vals["test_f1"]))     if vals["test_f1"]     else None,
                "test_f1_std":      float(np.std(vals["test_f1"]))      if vals["test_f1"]     else None,
                "val_f1_mean":      float(np.mean(vals["val_f1"]))      if vals["val_f1"]      else None,
                "val_f1_std":       float(np.std(vals["val_f1"]))       if vals["val_f1"]      else None,
                "best_epoch_mean":  float(np.mean(vals["best_epoch"]))  if vals["best_epoch"]  else None,
                "best_epoch_std":   float(np.std(vals["best_epoch"]))   if vals["best_epoch"]  else None,
            })

        summary_rows = non_random_rows + merged_random_rows

    print(f"\n[SWEEP] All {len(all_results['runs'])} runs completed. Saving global results...")
    json_path = global_dir / "all_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    df = pd.DataFrame(summary_rows)
    csv_path = global_dir / "all_results_summary.csv"
    df.to_csv(csv_path, index=False)

    # Optionnel: best configs par graphe/modèle
    best_csv = global_dir / "best_by_graph_model.csv"
    if len(df) > 0 and "test_f1_mean" in df.columns:
        best = (
            df.dropna(subset=["test_f1_mean"])
            .sort_values("test_f1_mean", ascending=False)
            .groupby(["kg_name", "init_embd"], as_index=False)
            .head(1)
        )
        best.to_csv(best_csv, index=False)

    print("\n" + "#" * 80)
    print("SWEEP DONE ✅")
    print(f"Global JSON : {json_path}")
    print(f"Summary CSV : {csv_path}")
    if best_csv.exists():
        print(f"Best CSV    : {best_csv}")
    print("#" * 80)


if __name__ == "__main__":
    main()