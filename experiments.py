import json
from pathlib import Path
from datetime import datetime
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
    seeds = [123, 42, 123, 456, 789, 2024]
    init_embd = "sentence-transformers/all-MiniLM-L6-v2"
    splits_dir = "datasets/split"

    # Liste des graphes (noms kg)
    graphs = [
        "reference_kg",
        "GT2KG_kg",
        # "reference_kg",
        # "another_kg",
    ]

    # Option: mapping embeddings par graphe (recommandé pour éviter incohérences)
    # Mets None si tu veux utiliser random embd / ou si get_data gère autrement.
    embd_paths = {
        "GT2KG_kg": {
            "entities": "GT2KG_kg_init_embd/EntitiesEmbedding.pickle",
            "edges": "GT2KG_kg_init_embd/PredicatesEmbedding.pickle",
        },
        # "KG_GEN_kg": {
        #     "entities": "KG_GEN_kg_init_embd/EntitiesEmbedding.pickle",
        #     "edges": "KG_GEN_kg_init_embd/PredicatesEmbedding.pickle",
        # },
        "reference_kg": {
            "entities": "BioMed_noisy_init_embd/EntitiesBertEmbedding_NCI.pickle",
            "edges": "BioMed_noisy_init_embd/PredicatesBertEmbedding_NCI.pickle",
        },
    }

    # Modèles à tester + configs de sweep
    model_names = ["GAT","GCN", "RGCN", "GAT", "TransEGCN_conv","RotatEGCN_conv", "TransEGCN_attn","RotatEGCN_attn"]

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
            "init_embd": init_embd,
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
    for kg_name in graphs:
        # CRÉER UNE NOUVELLE INSTANCE DE TDGBench PAR GRAPHE
        tdg = TDGBench(use_classifier=True)
        ent_path = embd_paths.get(kg_name, {}).get("entities", None)
        edge_path = embd_paths.get(kg_name, {}).get("edges", None)

        # Charger une fois pour récupérer dimensions
        annotated_graph, _, _, _, _ = tdg.get_data(
            kg_name=kg_name,
            init_embd=init_embd,
            split_path=f"{splits_dir}/split_{seeds[0]}.json",  # n'importe quel split existant
            entities_embd_path=ent_path,
            edges_embd_path=edge_path,
        )
        in_channels = annotated_graph.x.shape[1]
        num_relations = len(set(annotated_graph.edge_type.tolist()))

        print("\n" + "#" * 80)
        print(f"# GRAPH: {kg_name}")
        print(f"# Nodes: {annotated_graph.num_nodes}")
        print(f"# Edges: {annotated_graph.num_edges}")
        print(f"# in_channels={in_channels} | num_relations={num_relations}")
        print(f"# Embeddings entities: {ent_path}")
        print(f"# Embeddings edges: {edge_path}")
        print("#" * 80)

        for model_name in model_names:
            for hidden_channels in hidden_grid:
                run_id = f"{kg_name}__{model_name}__h{hidden_channels}"

                # dossier optionnel par run (si tu veux garder les fichiers par combo)
                per_run_dir = global_dir / kg_name / model_name / f"h{hidden_channels}"
                per_run_dir.mkdir(parents=True, exist_ok=True)

                extra_kwargs = get_default_model_kwargs(model_name)

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
                        out_channels=_h,  #
                        num_relations=_nr,
                        **_kwargs
                    )

                print("\n" + "=" * 80)
                print(f"RUN: {run_id}")
                print("=" * 80)

                results = tdg.evaluate_all(
                    kg_name=kg_name,
                    model_factory=model_factory,
                    init_embd=init_embd,
                    seeds=seeds,
                    splits_dir=splits_dir,
                    entities_embd_path=ent_path,
                    edges_embd_path=edge_path,
                    epochs=epochs,
                    patience=patience,
                    lr=lr,
                    weight_decay=weight_decay,
                    verbose=True,
                    save_results=True,  # garde les exports "classiques" de TDGBench
                    results_dir=str(per_run_dir),  # ici => un sous-dossier par combo
                    run_id=run_id  # ✅ AJOUTER LE run_id POUR DES NOMS UNIQUES
                )

                # Stockage global (JSON)
                all_results["runs"].append({
                    "run_id": run_id,
                    "kg_name": kg_name,
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
            .groupby(["kg_name", "model_name"], as_index=False)
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