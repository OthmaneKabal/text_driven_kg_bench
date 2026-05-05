# import json
# from collections import defaultdict
# from pathlib import Path
# from datetime import datetime

# import numpy as np
# import pandas as pd

# from build_models import build_encoder, get_default_model_kwargs
# from tdg_bench import TDGBench


# def main():
#     # -----------------------------
#     # 1) Paramètres du sweep
#     # -----------------------------
#     seeds = [42, 123, 456, 789, 2024]
#     splits_dir = "datasets/split"

#     onto_incorporation = "align"   # None = baseline | "align" = ontology alignment
#     onto_name = "SM_network"

#     lambda_grid = [0.7]
#     # Tu peux ajouter 0.5 ou 1.0 après si les petites valeurs marchent bien.

#     graphs = [
#         "GT2KG_edc_canonicalized_KG",
#     ]

#     init_embds = [
#         "sentence-transformers/all-MiniLM-L6-v2",
#     ]

#     BEST_GNN_PER_GRAPH = {
#         "KG_GEN_kg": "RotatEGCN_attn",
#         "GT2KG_edc_canonicalized_KG_is_a_aug": "RotatEGCN_attn",
#         "GT2KG_edc_final_kg": "RotatEGCN_attn",
#         "GT2KG_edc_canonicalized_KG": "RotatEGCN_attn",
#         "GT2KG_kg": "RotatEGCN_attn",
#         "GT2KG_is_augmented": "RotatEGCN_attn",
#         "GT2KG_mapped_SMN": "RotatEGCN_attn",
#         "GT2KG_mapped_SMN_is_a_augmented": "RotatEGCN_attn",
#     }

#     hidden_grid = [64, 128, 256, 384, 512]

#     epochs = 100
#     patience = 100
#     lr = 0.01
#     weight_decay = 5e-4

#     # -----------------------------
#     # 2) Sorties
#     # -----------------------------
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     mode_name = onto_incorporation if onto_incorporation else "baseline"

#     global_dir = Path("results") / f"sweep_{mode_name}_lambda_{timestamp}"
#     global_dir.mkdir(parents=True, exist_ok=True)

#     all_results = {
#         "meta": {
#             "timestamp": timestamp,
#             "seeds": seeds,
#             "init_embds": init_embds,
#             "splits_dir": splits_dir,
#             "epochs": epochs,
#             "patience": patience,
#             "lr": lr,
#             "weight_decay": weight_decay,
#             "use_classifier": True,
#             "onto_incorporation": onto_incorporation,
#             "onto_name": onto_name if onto_incorporation == "align" else None,
#             "lambda_grid": lambda_grid if onto_incorporation == "align" else None,
#         },
#         "runs": [],
#     }

#     summary_rows = []

#     # -----------------------------
#     # 3) Sweep
#     # -----------------------------
#     total_runs = (
#         len(graphs)
#         * len(init_embds)
#         * len(hidden_grid)
#         * (len(lambda_grid) if onto_incorporation == "align" else 1)
#     )

#     print(
#         f"[SWEEP] {len(graphs)} graph(s) x {len(init_embds)} embedding(s) "
#         f"x {len(hidden_grid)} hidden size(s) "
#         f"x {(len(lambda_grid) if onto_incorporation == 'align' else 1)} lambda(s) "
#         f"= {total_runs} runs"
#     )
#     print(f"[SWEEP] mode={onto_incorporation} | seeds={seeds}\n")

#     for graph_idx, kg_name in enumerate(graphs, 1):
#         for embd_idx, init_embd in enumerate(init_embds, 1):
#             embd_short = init_embd.split("/")[-1]

#             print(
#                 f"\n[GRAPH {graph_idx}/{len(graphs)}] "
#                 f"[EMBD {embd_idx}/{len(init_embds)}] "
#                 f"Loading {kg_name} | embd={embd_short}"
#             )

#             tdg = TDGBench(use_classifier=True)

#             annotated_graph, _, _, _, _ = tdg.get_data(
#                 kg_name=kg_name,
#                 init_embd=init_embd,
#                 split_path=f"{splits_dir}/split_{seeds[0]}.json",
#             )

#             in_channels = annotated_graph.x.shape[1]
#             num_relations = len(set(annotated_graph.edge_type.tolist()))

#             model_name = BEST_GNN_PER_GRAPH[kg_name]
#             extra_kwargs = get_default_model_kwargs(model_name)

#             print("\n" + "#" * 80)
#             print(f"# GRAPH: {kg_name}")
#             print(f"# EMBEDDING: {embd_short}")
#             print(f"# MODEL: {model_name}")
#             print(f"# Nodes: {annotated_graph.num_nodes}")
#             print(f"# Edges: {annotated_graph.num_edges}")
#             print(f"# in_channels={in_channels} | num_relations={num_relations}")
#             print(f"# onto_incorporation={onto_incorporation}")
#             print("#" * 80)

#             for hidden_channels in hidden_grid:
#                 current_lambda_grid = lambda_grid if onto_incorporation == "align" else [None]

#                 for lambda_align in current_lambda_grid:
#                     lambda_tag = f"lambda{lambda_align}" if lambda_align is not None else "nolambda"

#                     run_id = (
#                         f"{kg_name}__{embd_short}__{model_name}"
#                         f"__h{hidden_channels}__onto_{mode_name}"
#                         f"__{lambda_tag}"
#                     )

#                     per_run_dir = (
#                         global_dir
#                         / kg_name
#                         / embd_short
#                         / model_name
#                         / f"h{hidden_channels}"
#                         / mode_name
#                         / lambda_tag
#                     )
#                     per_run_dir.mkdir(parents=True, exist_ok=True)

#                     def model_factory(
#                         _model_name=model_name,
#                         _in=in_channels,
#                         _h=hidden_channels,
#                         _nr=num_relations,
#                         _kwargs=extra_kwargs,
#                     ):
#                         return build_encoder(
#                             model_name=_model_name,
#                             in_channels=_in,
#                             hidden_channels=_h,
#                             out_channels=_h,
#                             num_relations=_nr,
#                             **_kwargs,
#                         )

#                     print("\n" + "=" * 80)
#                     print(f"RUN: {run_id}")
#                     print(f"hidden={hidden_channels}")
#                     print(f"lambda_align={lambda_align}")
#                     print(f"extra_kwargs={extra_kwargs}")
#                     print("=" * 80)

#                     results = tdg.evaluate_all(
#                         kg_name=kg_name,
#                         model_factory=model_factory,
#                         init_embd=init_embd,
#                         seeds=seeds,
#                         splits_dir=splits_dir,
#                         epochs=epochs,
#                         patience=patience,
#                         lr=lr,
#                         weight_decay=weight_decay,
#                         verbose=True,
#                         save_results=True,
#                         results_dir=str(per_run_dir),
#                         run_id=run_id,
#                         onto_incorporation=onto_incorporation,
#                         onto_name=onto_name,
#                         lambda_align=lambda_align if lambda_align is not None else 0.0,
#                         align_batch_size=None,
#                         align_num_neighbors=[200, 200],
#                     )

#                     all_results["runs"].append(
#                         {
#                             "run_id": run_id,
#                             "kg_name": kg_name,
#                             "init_embd": init_embd,
#                             "model_name": model_name,
#                             "hidden_channels": hidden_channels,
#                             "out_channels": hidden_channels,
#                             "in_channels": in_channels,
#                             "num_relations": num_relations,
#                             "extra_kwargs": extra_kwargs,
#                             "onto_incorporation": onto_incorporation,
#                             "onto_name": onto_name if onto_incorporation == "align" else None,
#                             "lambda_align": lambda_align if onto_incorporation == "align" else None,
#                             "results": results,
#                         }
#                     )

#                     agg = results.get("aggregated", {})

#                     summary_rows.append(
#                         {
#                             "run_id": run_id,
#                             "kg_name": kg_name,
#                             "init_embd": init_embd,
#                             "model_name": model_name,
#                             "hidden_channels": hidden_channels,
#                             "onto_incorporation": onto_incorporation or "baseline",
#                             "onto_name": onto_name if onto_incorporation == "align" else None,
#                             "lambda_align": lambda_align if onto_incorporation == "align" else None,
#                             "test_acc_mean": agg.get("test_acc", {}).get("mean", None),
#                             "test_acc_std": agg.get("test_acc", {}).get("std", None),
#                             "test_f1_mean": agg.get("test_f1", {}).get("mean", None),
#                             "test_f1_std": agg.get("test_f1", {}).get("std", None),
#                             "val_f1_mean": agg.get("val_f1", {}).get("mean", None),
#                             "val_f1_std": agg.get("val_f1", {}).get("std", None),
#                             "best_epoch_mean": agg.get("best_epoch", {}).get("mean", None),
#                             "best_epoch_std": agg.get("best_epoch", {}).get("std", None),
#                         }
#                     )

#     # -----------------------------
#     # 4) Merge random_* rows
#     # -----------------------------
#     non_random_rows = [
#         r for r in summary_rows
#         if not str(r["init_embd"]).startswith("random_")
#     ]

#     random_rows = [
#         r for r in summary_rows
#         if str(r["init_embd"]).startswith("random_")
#     ]

#     if random_rows:
#         metrics = ["test_acc", "test_f1", "val_f1", "best_epoch"]
#         random_values_map = defaultdict(lambda: {m: [] for m in metrics})

#         for run in all_results["runs"]:
#             if str(run["init_embd"]).startswith("random_"):
#                 key = (
#                     run["kg_name"],
#                     run["model_name"],
#                     run["hidden_channels"],
#                     run["onto_incorporation"] or "baseline",
#                     run["lambda_align"],
#                 )

#                 agg = run["results"].get("aggregated", {})

#                 for m in metrics:
#                     random_values_map[key][m].extend(
#                         agg.get(m, {}).get("values", [])
#                     )

#         seen_keys = set()
#         merged_random_rows = []

#         for r in random_rows:
#             key = (
#                 r["kg_name"],
#                 r["model_name"],
#                 r["hidden_channels"],
#                 r["onto_incorporation"],
#                 r["lambda_align"],
#             )

#             if key in seen_keys:
#                 continue

#             seen_keys.add(key)
#             vals = random_values_map[key]

#             merged_random_rows.append(
#                 {
#                     "run_id": (
#                         f"{r['kg_name']}__random__{r['model_name']}"
#                         f"__h{r['hidden_channels']}__onto_{r['onto_incorporation']}"
#                         f"__lambda{r['lambda_align']}"
#                     ),
#                     "kg_name": r["kg_name"],
#                     "init_embd": "random",
#                     "model_name": r["model_name"],
#                     "hidden_channels": r["hidden_channels"],
#                     "onto_incorporation": r["onto_incorporation"],
#                     "onto_name": r["onto_name"],
#                     "lambda_align": r["lambda_align"],
#                     "test_acc_mean": float(np.mean(vals["test_acc"])) if vals["test_acc"] else None,
#                     "test_acc_std": float(np.std(vals["test_acc"])) if vals["test_acc"] else None,
#                     "test_f1_mean": float(np.mean(vals["test_f1"])) if vals["test_f1"] else None,
#                     "test_f1_std": float(np.std(vals["test_f1"])) if vals["test_f1"] else None,
#                     "val_f1_mean": float(np.mean(vals["val_f1"])) if vals["val_f1"] else None,
#                     "val_f1_std": float(np.std(vals["val_f1"])) if vals["val_f1"] else None,
#                     "best_epoch_mean": float(np.mean(vals["best_epoch"])) if vals["best_epoch"] else None,
#                     "best_epoch_std": float(np.std(vals["best_epoch"])) if vals["best_epoch"] else None,
#                 }
#             )

#         summary_rows = non_random_rows + merged_random_rows

#     # -----------------------------
#     # 5) Save global results
#     # -----------------------------
#     print(f"\n[SWEEP] Saving global results to {global_dir}")

#     json_path = global_dir / "all_results.json"
#     with open(json_path, "w") as f:
#         json.dump(all_results, f, indent=2)

#     df = pd.DataFrame(summary_rows)

#     csv_path = global_dir / "all_results_summary.csv"
#     df.to_csv(csv_path, index=False)

#     best_csv = global_dir / "best_by_graph_model.csv"

#     if len(df) > 0 and "test_f1_mean" in df.columns:
#         best = (
#             df.dropna(subset=["test_f1_mean"])
#             .sort_values("test_f1_mean", ascending=False)
#             .groupby(
#                 ["kg_name", "init_embd", "onto_incorporation"],
#                 as_index=False,
#             )
#             .head(1)
#         )
#         best.to_csv(best_csv, index=False)

#     best_lambda_csv = global_dir / "best_lambda_by_hidden.csv"

#     if (
#         len(df) > 0
#         and "test_f1_mean" in df.columns
#         and "lambda_align" in df.columns
#     ):
#         best_lambda = (
#             df.dropna(subset=["test_f1_mean"])
#             .sort_values("test_f1_mean", ascending=False)
#             .groupby(
#                 ["kg_name", "init_embd", "model_name", "hidden_channels"],
#                 as_index=False,
#             )
#             .head(1)
#         )
#         best_lambda.to_csv(best_lambda_csv, index=False)

#     print("\n" + "#" * 80)
#     print("SWEEP DONE ✅")
#     print(f"Global JSON       : {json_path}")
#     print(f"Summary CSV       : {csv_path}")

#     if best_csv.exists():
#         print(f"Best CSV          : {best_csv}")

#     if best_lambda_csv.exists():
#         print(f"Best Lambda CSV   : {best_lambda_csv}")

#     print("#" * 80)


# if __name__ == "__main__":
#     main()


import json
from collections import defaultdict
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from build_models import build_encoder, get_default_model_kwargs
from tdg_bench import TDGBench


def main():
    seeds = [42, 123, 456, 789, 2024]
    splits_dir = "datasets/split"

    onto_incorporation = "align"  # None | "align"
    onto_name = "SM_network"

    lambda_grid = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5,0.7]
    alignment_modes = ["contrastive", "cosine"] ## "cosine", 
    temperature_grid = [0.1, 0.2, 0.5, 0.7]

    graphs = [
        "EDC_canonicalized_kg_v2",
        "EDC_canonicalized_kg_v2_augmented",

    ]

    init_embds = [

        "sentence-transformers/all-MiniLM-L6-v2"
        #         "random_42",
        # "random_123",
        # "random_456",
    ]

    BEST_GNN_PER_GRAPH = {
        "KG_GEN_kg": "RotatEGCN_attn",
        "GT2KG_edc_canonicalized_KG_is_a_aug": "RotatEGCN_attn",
        "GT2KG_edc_final_kg": "RotatEGCN_attn",
        "GT2KG_edc_canonicalized_KG": "RotatEGCN_attn",
        "GT2KG_kg": "RotatEGCN_attn",
        "GT2KG_is_augmented": "RotatEGCN_attn",
        "GT2KG_mapped_SMN": "RotatEGCN_attn",
        "GT2KG_mapped_SMN_is_a_augmented": "RotatEGCN_attn",
        "EDC_canonicalized_kg_v2":"RotatEGCN_attn",
        "EDC_canonicalized_kg_v2_augmented": "RotatEGCN_attn",
    }

    hidden_grid = [64, 128, 256, 384, 512]

    epochs = 100
    patience = 100
    lr = 0.01
    weight_decay = 5e-4

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_name = onto_incorporation if onto_incorporation else "baseline"

    global_dir = Path("results") / f"sweep_{mode_name}_fullopt_{timestamp}"
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
            "onto_incorporation": onto_incorporation,
            "onto_name": onto_name if onto_incorporation == "align" else None,
            "lambda_grid": lambda_grid if onto_incorporation == "align" else None,
            "alignment_modes": alignment_modes if onto_incorporation == "align" else None,
            "temperature_grid": temperature_grid if onto_incorporation == "align" else None,
        },
        "runs": [],
    }

    summary_rows = []

    if onto_incorporation == "align":
        total_align_configs = 0
        for mode in alignment_modes:
            if mode == "contrastive":
                total_align_configs += len(lambda_grid) * len(temperature_grid)
            else:
                total_align_configs += len(lambda_grid)
    else:
        total_align_configs = 1

    total_runs = len(graphs) * len(init_embds) * len(hidden_grid) * total_align_configs

    print(
        f"[SWEEP] {len(graphs)} graph(s) x {len(init_embds)} embedding(s) "
        f"x {len(hidden_grid)} hidden size(s) x {total_align_configs} align config(s) "
        f"= {total_runs} runs"
    )
    print(f"[SWEEP] mode={onto_incorporation} | seeds={seeds}\n")

    for graph_idx, kg_name in enumerate(graphs, 1):
        for embd_idx, init_embd in enumerate(init_embds, 1):
            embd_short = init_embd.split("/")[-1]

            print(
                f"\n[GRAPH {graph_idx}/{len(graphs)}] "
                f"[EMBD {embd_idx}/{len(init_embds)}] "
                f"Loading {kg_name} | embd={embd_short}"
            )

            tdg = TDGBench(use_classifier=True)

            annotated_graph, _, _, _, _ = tdg.get_data(
                kg_name=kg_name,
                init_embd=init_embd,
                split_path=f"{splits_dir}/split_{seeds[0]}.json",
            )

            in_channels = annotated_graph.x.shape[1]
            num_relations = len(set(annotated_graph.edge_type.tolist()))

            model_name = BEST_GNN_PER_GRAPH[kg_name]
            extra_kwargs = get_default_model_kwargs(model_name)

            print("\n" + "#" * 80)
            print(f"# GRAPH: {kg_name}")
            print(f"# EMBEDDING: {embd_short}")
            print(f"# MODEL: {model_name}")
            print(f"# Nodes: {annotated_graph.num_nodes}")
            print(f"# Edges: {annotated_graph.num_edges}")
            print(f"# in_channels={in_channels} | num_relations={num_relations}")
            print(f"# onto_incorporation={onto_incorporation}")
            print("#" * 80)

            for hidden_channels in hidden_grid:
                if onto_incorporation == "align":
                    align_configs = []
                    for alignment_mode in alignment_modes:
                        if alignment_mode == "contrastive":
                            for lambda_align in lambda_grid:
                                for temperature in temperature_grid:
                                    align_configs.append(
                                        {
                                            "lambda_align": lambda_align,
                                            "alignment_mode": alignment_mode,
                                            "temperature": temperature,
                                        }
                                    )
                        else:
                            for lambda_align in lambda_grid:
                                align_configs.append(
                                    {
                                        "lambda_align": lambda_align,
                                        "alignment_mode": alignment_mode,
                                        "temperature": None,
                                    }
                                )
                else:
                    align_configs = [
                        {
                            "lambda_align": None,
                            "alignment_mode": None,
                            "temperature": None,
                        }
                    ]

                for cfg in align_configs:
                    lambda_align = cfg["lambda_align"]
                    alignment_mode = cfg["alignment_mode"]
                    temperature = cfg["temperature"]

                    lambda_tag = (
                        f"lambda{lambda_align}" if lambda_align is not None else "nolambda"
                    )
                    mode_tag = alignment_mode if alignment_mode is not None else "baseline"
                    temp_tag = (
                        f"temp{temperature}" if temperature is not None else "notemp"
                    )

                    run_id = (
                        f"{kg_name}__{embd_short}__{model_name}"
                        f"__h{hidden_channels}__onto_{mode_name}"
                        f"__{mode_tag}__{lambda_tag}__{temp_tag}"
                    )

                    per_run_dir = (
                        global_dir
                        / kg_name
                        / embd_short
                        / model_name
                        / f"h{hidden_channels}"
                        / mode_name
                        / mode_tag
                        / lambda_tag
                        / temp_tag
                    )
                    per_run_dir.mkdir(parents=True, exist_ok=True)

                    def model_factory(
                        _model_name=model_name,
                        _in=in_channels,
                        _h=hidden_channels,
                        _nr=num_relations,
                        _kwargs=extra_kwargs,
                    ):
                        return build_encoder(
                            model_name=_model_name,
                            in_channels=_in,
                            hidden_channels=_h,
                            out_channels=_h,
                            num_relations=_nr,
                            **_kwargs,
                        )

                    print("\n" + "=" * 80)
                    print(f"RUN: {run_id}")
                    print(f"hidden={hidden_channels}")
                    print(f"lambda_align={lambda_align}")
                    print(f"alignment_mode={alignment_mode}")
                    print(f"temperature={temperature}")
                    print(f"extra_kwargs={extra_kwargs}")
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
                        run_id=run_id,
                        onto_incorporation=onto_incorporation,
                        onto_name=onto_name,
                        lambda_align=lambda_align if lambda_align is not None else 0.0,
                        alignment_mode=alignment_mode or "cosine",
                        temperature=temperature if temperature is not None else 0.2,
                        align_batch_size=None,
                        align_num_neighbors=[200, 200],
                    )

                    all_results["runs"].append(
                        {
                            "run_id": run_id,
                            "kg_name": kg_name,
                            "init_embd": init_embd,
                            "model_name": model_name,
                            "hidden_channels": hidden_channels,
                            "out_channels": hidden_channels,
                            "in_channels": in_channels,
                            "num_relations": num_relations,
                            "extra_kwargs": extra_kwargs,
                            "onto_incorporation": onto_incorporation,
                            "onto_name": onto_name if onto_incorporation == "align" else None,
                            "lambda_align": lambda_align if onto_incorporation == "align" else None,
                            "alignment_mode": alignment_mode if onto_incorporation == "align" else None,
                            "temperature": temperature if onto_incorporation == "align" else None,
                            "results": results,
                        }
                    )

                    agg = results.get("aggregated", {})

                    summary_rows.append(
                        {
                            "run_id": run_id,
                            "kg_name": kg_name,
                            "init_embd": init_embd,
                            "model_name": model_name,
                            "hidden_channels": hidden_channels,
                            "onto_incorporation": onto_incorporation or "baseline",
                            "onto_name": onto_name if onto_incorporation == "align" else None,
                            "lambda_align": lambda_align if onto_incorporation == "align" else None,
                            "alignment_mode": alignment_mode if onto_incorporation == "align" else None,
                            "temperature": temperature if onto_incorporation == "align" else None,
                            "test_acc_mean": agg.get("test_acc", {}).get("mean", None),
                            "test_acc_std": agg.get("test_acc", {}).get("std", None),
                            "test_f1_mean": agg.get("test_f1", {}).get("mean", None),
                            "test_f1_std": agg.get("test_f1", {}).get("std", None),
                            "val_f1_mean": agg.get("val_f1", {}).get("mean", None),
                            "val_f1_std": agg.get("val_f1", {}).get("std", None),
                            "best_epoch_mean": agg.get("best_epoch", {}).get("mean", None),
                            "best_epoch_std": agg.get("best_epoch", {}).get("std", None),
                        }
                    )

    non_random_rows = [
        r for r in summary_rows if not str(r["init_embd"]).startswith("random_")
    ]
    random_rows = [
        r for r in summary_rows if str(r["init_embd"]).startswith("random_")
    ]

    if random_rows:
        metrics = ["test_acc", "test_f1", "val_f1", "best_epoch"]
        random_values_map = defaultdict(lambda: {m: [] for m in metrics})

        for run in all_results["runs"]:
            if str(run["init_embd"]).startswith("random_"):
                key = (
                    run["kg_name"],
                    run["model_name"],
                    run["hidden_channels"],
                    run["onto_incorporation"] or "baseline",
                    run["lambda_align"],
                    run["alignment_mode"],
                    run["temperature"],
                )

                agg = run["results"].get("aggregated", {})

                for m in metrics:
                    random_values_map[key][m].extend(
                        agg.get(m, {}).get("values", [])
                    )

        seen_keys = set()
        merged_random_rows = []

        for r in random_rows:
            key = (
                r["kg_name"],
                r["model_name"],
                r["hidden_channels"],
                r["onto_incorporation"],
                r["lambda_align"],
                r["alignment_mode"],
                r["temperature"],
            )

            if key in seen_keys:
                continue

            seen_keys.add(key)
            vals = random_values_map[key]

            merged_random_rows.append(
                {
                    "run_id": (
                        f"{r['kg_name']}__random__{r['model_name']}"
                        f"__h{r['hidden_channels']}__onto_{r['onto_incorporation']}"
                        f"__{r['alignment_mode']}__lambda{r['lambda_align']}"
                        f"__temp{r['temperature']}"
                    ),
                    "kg_name": r["kg_name"],
                    "init_embd": "random",
                    "model_name": r["model_name"],
                    "hidden_channels": r["hidden_channels"],
                    "onto_incorporation": r["onto_incorporation"],
                    "onto_name": r["onto_name"],
                    "lambda_align": r["lambda_align"],
                    "alignment_mode": r["alignment_mode"],
                    "temperature": r["temperature"],
                    "test_acc_mean": float(np.mean(vals["test_acc"]))
                    if vals["test_acc"]
                    else None,
                    "test_acc_std": float(np.std(vals["test_acc"]))
                    if vals["test_acc"]
                    else None,
                    "test_f1_mean": float(np.mean(vals["test_f1"]))
                    if vals["test_f1"]
                    else None,
                    "test_f1_std": float(np.std(vals["test_f1"]))
                    if vals["test_f1"]
                    else None,
                    "val_f1_mean": float(np.mean(vals["val_f1"]))
                    if vals["val_f1"]
                    else None,
                    "val_f1_std": float(np.std(vals["val_f1"]))
                    if vals["val_f1"]
                    else None,
                    "best_epoch_mean": float(np.mean(vals["best_epoch"]))
                    if vals["best_epoch"]
                    else None,
                    "best_epoch_std": float(np.std(vals["best_epoch"]))
                    if vals["best_epoch"]
                    else None,
                }
            )

        summary_rows = non_random_rows + merged_random_rows

    print(f"\n[SWEEP] Saving global results to {global_dir}")

    json_path = global_dir / "all_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    df = pd.DataFrame(summary_rows)

    csv_path = global_dir / "all_results_summary.csv"
    df.to_csv(csv_path, index=False)

    best_csv = global_dir / "best_by_graph_model.csv"

    if len(df) > 0 and "test_f1_mean" in df.columns:
        best = (
            df.dropna(subset=["test_f1_mean"])
            .sort_values("test_f1_mean", ascending=False)
            .groupby(
                ["kg_name", "init_embd", "onto_incorporation"],
                as_index=False,
            )
            .head(1)
        )
        best.to_csv(best_csv, index=False)

    best_config_csv = global_dir / "best_config_by_hidden.csv"

    if len(df) > 0 and "test_f1_mean" in df.columns:
        best_config = (
            df.dropna(subset=["test_f1_mean"])
            .sort_values("test_f1_mean", ascending=False)
            .groupby(
                [
                    "kg_name",
                    "init_embd",
                    "model_name",
                    "hidden_channels",
                    "onto_incorporation",
                ],
                as_index=False,
            )
            .head(1)
        )
        best_config.to_csv(best_config_csv, index=False)

    print("\n" + "#" * 80)
    print("SWEEP DONE ✅")
    print(f"Global JSON       : {json_path}")
    print(f"Summary CSV       : {csv_path}")

    if best_csv.exists():
        print(f"Best CSV          : {best_csv}")

    if best_config_csv.exists():
        print(f"Best Config CSV   : {best_config_csv}")

    print("#" * 80)


if __name__ == "__main__":
    main()
