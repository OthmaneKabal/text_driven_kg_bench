import json
from tqdm import tqdm
from tdg_bench import TDGBench
import pandas as pd
import json
from build_models import build_encoder, get_default_model_kwargs
from data_preprocessing.GraphDataPreparation import GraphDataPreparation
import networkx as nx
from collections import defaultdict, deque, Counter
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

def read_json_file(file_path):
    """
    Read and return the content of a JSON file.

    Parameters
    ----------
    file_path : str
        Path to the JSON file.

    Returns
    -------
    dict or list
        Parsed JSON content.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data

def csv_to_json_triplets(csv_path, output_json_path=None):
    """
    Convert a CSV with columns [subject, predicate, object]
    into a JSON list of triplets.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    output_json_path : str, optional
        Path to save the JSON file.

    Returns
    -------
    list
        List of dictionaries with keys:
        subject, predicate, object
    """
    df = pd.read_csv(csv_path)

    required_cols = ["subject", "predicate", "object"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    triplets = df[required_cols].to_dict(orient="records")

    if output_json_path is not None:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(triplets, f, indent=2, ensure_ascii=False)

    return triplets
def replace_predicate_with_canonical(input_path, output_path):
    """
    Read a graph JSON file, replace 'predicate' by 'canonical_predicate',
    and save the updated graph.
    
    Parameters
    ----------
    input_path : str
        Path to the input JSON file.
    output_path : str
        Path to the output JSON file.
    
    Returns
    -------
    list[dict]
        Updated graph.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    updated_graph = []

    for triple in graph:
        triple = triple.copy()

        canonical_pred = triple.get("canonical_predicate")

        if canonical_pred and str(canonical_pred).strip():
            triple["original_predicate"] = triple.get("predicate")
            triple["predicate"] = canonical_pred

        updated_graph.append(triple)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(updated_graph, f, indent=2, ensure_ascii=False)

    return updated_graph

def merge_json_files(file1_path, file2_path, output_path):
    """
    Merge two JSON files and save the result.

    If both files contain:
    - lists: concatenate them
    - dicts: merge keys (file2 overrides duplicate keys from file1)
    """

    with open(file1_path, "r", encoding="utf-8") as f:
        data1 = json.load(f)

    with open(file2_path, "r", encoding="utf-8") as f:
        data2 = json.load(f)

    if isinstance(data1, list) and isinstance(data2, list):
        merged_data = data1 + data2

    elif isinstance(data1, dict) and isinstance(data2, dict):
        merged_data = {**data1, **data2}

    else:
        raise ValueError("Both JSON files must have the same top-level type (both list or both dict).")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)

    return merged_data



def reduce_ontology_granularity(
    input_path,
    output_path,
    target_level=4,
    reduce_isa_too=False,
    root_mode="both"   # "both", "entity", "event"
):
    """
    Réduit la granularité des types dans une ontologie JSON en remontant chaque type
    à son ancêtre situé au niveau `target_level` dans la hiérarchie `isa`.

    Parameters
    ----------
    input_path : str
        Chemin vers le fichier ontologie JSON.
    output_path : str
        Chemin de sauvegarde du JSON réduit.
    target_level : int
        Niveau maximal conservé depuis la racine.
    reduce_isa_too : bool
        Si True, réduit aussi les triplets de la relation 'isa'.
        Si False, laisse la relation 'isa' telle quelle.
    root_mode : str
        "both"   -> garder les branches sous Entity et Event
        "entity" -> garder seulement la branche sous Entity
        "event"  -> garder seulement la branche sous Event

    Returns
    -------
    reduced_ontology : dict
    type_to_ancestor : dict
    depth : dict
    """

    with open(input_path, "r", encoding="utf-8") as f:
        ontology = json.load(f)

    if "isa" not in ontology or "triplets" not in ontology["isa"]:
        raise ValueError("Le fichier doit contenir une relation 'isa' avec une clé 'triplets'.")

    if root_mode not in {"both", "entity", "event"}:
        raise ValueError("root_mode doit être parmi {'both', 'entity', 'event'}.")

    isa_triplets = ontology["isa"]["triplets"]

    parents_of = defaultdict(set)
    children_of = defaultdict(set)
    all_types = set()

    for triple in isa_triplets:
        if len(triple) != 3:
            continue

        child, rel, parent = triple
        if rel != "isa":
            continue

        parents_of[child].add(parent)
        children_of[parent].add(child)

        all_types.add(child)
        all_types.add(parent)

    # Racines "naturelles" = types sans parent
    natural_roots = [t for t in all_types if t not in parents_of or len(parents_of[t]) == 0]

    if not natural_roots:
        raise ValueError("Aucune racine trouvée dans la hiérarchie 'isa'.")

    # Choix des racines à utiliser
    selected_roots = []
    if root_mode == "both":
        for r in ["Entity", "Event"]:
            if r in all_types:
                selected_roots.append(r)
    elif root_mode == "entity":
        if "Entity" not in all_types:
            raise ValueError("Le type racine 'Entity' est introuvable dans l'ontologie.")
        selected_roots = ["Entity"]
    elif root_mode == "event":
        if "Event" not in all_types:
            raise ValueError("Le type racine 'Event' est introuvable dans l'ontologie.")
        selected_roots = ["Event"]

    if not selected_roots:
        raise ValueError(
            f"Aucune racine sélectionnée trouvée pour root_mode='{root_mode}'. "
            f"Racines naturelles détectées: {natural_roots}"
        )

    # Calcul profondeur minimale depuis les racines sélectionnées
    depth = {}
    queue = deque()

    for root in selected_roots:
        depth[root] = 0
        queue.append(root)

    while queue:
        current = queue.popleft()

        for child in children_of.get(current, []):
            new_depth = depth[current] + 1
            if child not in depth or new_depth < depth[child]:
                depth[child] = new_depth
                queue.append(child)

    # Sous-ensemble gardé = types atteignables depuis les racines choisies
    kept_types = set(depth.keys())

    # Parent principal parmi les parents encore dans la branche gardée
    primary_parent = {}
    for child, parent_set in parents_of.items():
        valid_parents = [p for p in parent_set if p in depth]
        if not valid_parents:
            continue
        best_parent = min(valid_parents, key=lambda p: depth[p])
        primary_parent[child] = best_parent

    def ancestor_at_level(type_name, level):
        if type_name not in depth:
            return None  # hors de la/les branche(s) gardée(s)

        if depth[type_name] <= level:
            return type_name

        current = type_name
        visited = set()

        while current in primary_parent and depth.get(current, 0) > level:
            if current in visited:
                break
            visited.add(current)
            current = primary_parent[current]

        return current

    # Mapping seulement pour les types gardés
    type_to_ancestor = {
        t: ancestor_at_level(t, target_level)
        for t in kept_types
    }

    # Réécriture de l'ontologie
    reduced_ontology = {}

    dropped_triplets_count = 0

    for relation, relation_data in ontology.items():
        triplets = relation_data.get("triplets", [])
        rel_def = relation_data.get("def", "")

        new_triplets = []
        seen = set()

        for triple in triplets:
            if len(triple) != 3:
                continue

            subj, pred, obj = triple

            # Si on filtre une branche, on retire les triplets contenant
            # des types hors de la branche choisie
            if subj not in kept_types or obj not in kept_types:
                dropped_triplets_count += 1
                continue

            if relation == "isa" and not reduce_isa_too:
                new_triple = [subj, pred, obj]
            else:
                new_subj = type_to_ancestor.get(subj, subj)
                new_obj = type_to_ancestor.get(obj, obj)
                new_triple = [new_subj, pred, new_obj]

            triple_key = tuple(new_triple)
            if triple_key not in seen:
                seen.add(triple_key)
                new_triplets.append(new_triple)

        reduced_ontology[relation] = {
            "triplets": new_triplets,
            "def": rel_def
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(reduced_ontology, f, indent=2, ensure_ascii=False)

    # =========================
    # Rapport
    # =========================
    original_type_count = len(all_types)
    kept_type_count = len(kept_types)
    reduced_types = set(type_to_ancestor.values())
    reduced_type_count = len(reduced_types)

    original_depth_distribution = Counter()
    for t, lvl in depth.items():
        original_depth_distribution[lvl] += 1

    reduced_depth_distribution = Counter()
    for t in reduced_types:
        reduced_depth_distribution[depth[t]] += 1

    merged_count = sum(1 for t, a in type_to_ancestor.items() if t != a)

    print("\n" + "=" * 80)
    print("ONTOLOGY GRANULARITY REDUCTION REPORT")
    print("=" * 80)
    print(f"Root mode: {root_mode}")
    print(f"Selected roots: {selected_roots}")
    print(f"Target level: {target_level}")
    print(f"Original number of types in ontology: {original_type_count}")
    print(f"Number of kept types in selected branch(es): {kept_type_count}")
    print(f"Reduced number of final types: {reduced_type_count}")
    print(f"Number of merged/simplified kept types: {merged_count}")
    print(f"Dropped triplets outside selected branch(es): {dropped_triplets_count}")

    if kept_type_count > 0:
        print(f"Reduction ratio inside kept branch(es): {100 * (1 - reduced_type_count / kept_type_count):.2f}%")

    print("\n--- Kept type distribution by level ---")
    for lvl in sorted(original_depth_distribution):
        print(f"Level {lvl}: {original_depth_distribution[lvl]} types")

    print("\n--- Final reduced type distribution by level ---")
    for lvl in sorted(reduced_depth_distribution):
        print(f"Level {lvl}: {reduced_depth_distribution[lvl]} types")

    print("\n--- Example mappings (first 30) ---")
    for original_type, reduced_type in sorted(type_to_ancestor.items())[:30]:
        print(
            f"{original_type} (L{depth[original_type]}) "
            f"-> {reduced_type} (L{depth[reduced_type]})"
        )

    print("=" * 80 + "\n")

    return reduced_ontology, type_to_ancestor, depth

from pathlib import Path
import pandas as pd


def best_by_graph_approach_from_partial_results(
    results_dir,
    metric="test_f1_mean",
    output_name="best_by_graph_approach_partial.csv",
):
    results_dir = Path(results_dir)

    csv_files = list(results_dir.rglob("summary_*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"Aucun fichier summary_*.csv trouvé dans {results_dir}")

    dfs = []

    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df["source_file"] = str(f)
            dfs.append(df)
        except Exception as e:
            print(f"[WARNING] Impossible de lire {f}: {e}")

    df = pd.concat(dfs, ignore_index=True)

    if metric not in df.columns:
        raise ValueError(f"Colonne {metric} absente des fichiers CSV.")

    df = df.dropna(subset=[metric]).copy()

    def get_approach(row):
        parts = []

        if "onto_incorporation" in row and pd.notna(row["onto_incorporation"]):
            parts.append(str(row["onto_incorporation"]))

        if "alignment_mode" in row and pd.notna(row["alignment_mode"]):
            parts.append(str(row["alignment_mode"]))

        if "lambda_align" in row and pd.notna(row["lambda_align"]):
            parts.append(f"lambda={row['lambda_align']}")

        if "temperature" in row and pd.notna(row["temperature"]):
            parts.append(f"temp={row['temperature']}")

        return " | ".join(parts)

    df["approach"] = df.apply(get_approach, axis=1)

    best = (
        df.sort_values(metric, ascending=False)
        .groupby(["kg_name", "approach"], as_index=False)
        .head(1)
        .sort_values(["kg_name", metric], ascending=[True, False])
        .reset_index(drop=True)
    )

    output_path = results_dir / output_name
    best.to_csv(output_path, index=False)

    print(f"[OK] Saved: {output_path}")
    print(f"[INFO] Fichiers lus: {len(csv_files)}")

    return best
"""
parse_partial_results.py
------------------------
Parcourt l'arborescence d'un sweep en cours et génère :
  - partial_summary.csv        : toutes les runs terminées
  - partial_best_by_graph.csv  : meilleur test_f1 par (kg_name, init_embd, onto_incorporation)
  - partial_best_config.csv    : meilleur test_f1 par (kg_name, init_embd, model, hidden, onto)

Usage :
    python parse_partial_results.py --sweep_dir results/sweep_align_fullopt_20260429_175848
"""




# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_mean(vals):
    return float(np.mean(vals)) if vals else None

def safe_std(vals):
    return float(np.std(vals)) if vals else None


def parse_results_json(path: Path) -> dict | None:
    """
    Lit un fichier results_*.json et retourne un dict 'aggregated'.
    Deux formats possibles :
      - {"aggregated": {...}, "per_seed": [...]}   (format complet)
      - {"test_acc": [...], "test_f1": [...], ...} (liste de valeurs directement)
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"  [WARN] Impossible de lire {path}: {e}")
        return None

    # Format 1 : clé "aggregated"
    if "aggregated" in data:
        return data["aggregated"]

    # Format 2 : dict de listes de valeurs brutes
    agg = {}
    for metric in ["test_acc", "test_f1", "val_f1", "best_epoch"]:
        if metric in data:
            vals = data[metric]
            if isinstance(vals, list) and vals:
                agg[metric] = {"mean": safe_mean(vals), "std": safe_std(vals), "values": vals}
    return agg if agg else None


def parse_summary_json(path: Path) -> dict | None:
    """
    Lit un fichier summary_*.json (résumé agrégé direct).
    """
    try:
        with open(path) as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"  [WARN] Impossible de lire {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Extraction des métadonnées depuis le chemin
# ---------------------------------------------------------------------------

def extract_meta_from_path(results_file: Path, sweep_root: Path) -> dict:
    """
    Remonte l'arborescence pour extraire :
      kg_name / init_embd / model_name / hidden / onto_incorp / align_mode / lambda / temp
    
    Structure attendue :
      <sweep_root>/<kg_name>/<init_embd>/<model>/<hN>/<onto>/<mode>/<lambdaX>/<tempY>/results_*.json
    """
    parts = results_file.relative_to(sweep_root).parts
    # parts[0] = kg_name
    # parts[1] = init_embd
    # parts[2] = model_name
    # parts[3] = hN
    # parts[4] = onto_incorp  (align | baseline)
    # parts[5] = mode_tag     (contrastive | cosine | baseline)
    # parts[6] = lambda_tag
    # parts[7] = temp_tag
    # parts[8] = filename

    meta = {
        "kg_name": None,
        "init_embd": None,
        "model_name": None,
        "hidden_channels": None,
        "onto_incorporation": None,
        "alignment_mode": None,
        "lambda_align": None,
        "temperature": None,
    }

    if len(parts) >= 1:
        meta["kg_name"] = parts[0]
    if len(parts) >= 2:
        meta["init_embd"] = parts[1]
    if len(parts) >= 3:
        meta["model_name"] = parts[2]
    if len(parts) >= 4:
        m = re.match(r"h(\d+)", parts[3])
        meta["hidden_channels"] = int(m.group(1)) if m else parts[3]
    if len(parts) >= 5:
        meta["onto_incorporation"] = parts[4]
    if len(parts) >= 6:
        meta["alignment_mode"] = parts[5]
    if len(parts) >= 7:
        tag = parts[6]
        m = re.match(r"lambda(.+)", tag)
        if m:
            try:
                meta["lambda_align"] = float(m.group(1))
            except ValueError:
                meta["lambda_align"] = m.group(1)
    if len(parts) >= 8:
        tag = parts[7]
        m = re.match(r"temp(.+)", tag)
        if m:
            try:
                meta["temperature"] = float(m.group(1))
            except ValueError:
                meta["temperature"] = m.group(1)

    return meta


# ---------------------------------------------------------------------------
# Scan principal
# ---------------------------------------------------------------------------

def collect_rows(sweep_root: Path) -> list[dict]:
    rows = []

    # On cherche tous les fichiers results_*.json
    result_files = sorted(sweep_root.rglob("results_*.json"))
    print(f"[INFO] {len(result_files)} fichier(s) results_*.json trouvé(s)")

    for rf in result_files:
        meta = extract_meta_from_path(rf, sweep_root)
        agg = parse_results_json(rf)

        if agg is None:
            print(f"  [SKIP] Pas d'agrégation dans {rf}")
            continue

        row = {
            **meta,
            "results_file": str(rf.relative_to(sweep_root)),
            "test_acc_mean":  agg.get("test_acc",  {}).get("mean"),
            "test_acc_std":   agg.get("test_acc",  {}).get("std"),
            "test_f1_mean":   agg.get("test_f1",   {}).get("mean"),
            "test_f1_std":    agg.get("test_f1",   {}).get("std"),
            "val_f1_mean":    agg.get("val_f1",    {}).get("mean"),
            "val_f1_std":     agg.get("val_f1",    {}).get("std"),
            "best_epoch_mean": agg.get("best_epoch", {}).get("mean"),
            "best_epoch_std":  agg.get("best_epoch", {}).get("std"),
        }
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep_dir",
        default=None,
        help="Chemin vers le dossier du sweep (ex: results/sweep_align_fullopt_20260429_175848)",
    )
    args = parser.parse_args()

    # Auto-détection du sweep le plus récent si non fourni
    if args.sweep_dir is None:
        results_root = Path("results")
        candidates = sorted(results_root.glob("sweep_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print("[ERROR] Aucun dossier sweep trouvé dans results/")
            return
        sweep_root = candidates[0]
        print(f"[INFO] Sweep auto-détecté : {sweep_root}")
    else:
        sweep_root = Path(args.sweep_dir)

    if not sweep_root.exists():
        print(f"[ERROR] Dossier introuvable : {sweep_root}")
        return

    print(f"[INFO] Scan de : {sweep_root}\n")
    rows = collect_rows(sweep_root)

    if not rows:
        print("[WARN] Aucun résultat trouvé.")
        return

    df = pd.DataFrame(rows)
    print(f"\n[INFO] {len(df)} run(s) avec résultats agrégés\n")

    out_dir = sweep_root

    # --- 1) Summary complet ---
    summary_path = out_dir / "partial_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"[SAVE] partial_summary.csv          ({len(df)} lignes)")

    # --- 2) Best par (kg_name, init_embd, onto_incorporation) ---
    if "test_f1_mean" in df.columns:
        best_graph = (
            df.dropna(subset=["test_f1_mean"])
            .sort_values("test_f1_mean", ascending=False)
            .groupby(["kg_name", "init_embd", "onto_incorporation"], as_index=False)
            .head(1)
        )
        best_graph_path = out_dir / "partial_best_by_graph.csv"
        best_graph.to_csv(best_graph_path, index=False)
        print(f"[SAVE] partial_best_by_graph.csv    ({len(best_graph)} lignes)")

        print("\n=== MEILLEURS RÉSULTATS PAR GRAPHE ===")
        cols_show = [
            "kg_name", "hidden_channels", "alignment_mode",
            "lambda_align", "temperature",
            "test_f1_mean", "test_f1_std",
            "test_acc_mean", "val_f1_mean",
        ]
        cols_show = [c for c in cols_show if c in best_graph.columns]
        print(best_graph[cols_show].to_string(index=False))

    # --- 3) Best par (kg_name, init_embd, model, hidden, onto) ---
    if "test_f1_mean" in df.columns:
        best_config = (
            df.dropna(subset=["test_f1_mean"])
            .sort_values("test_f1_mean", ascending=False)
            .groupby(
                ["kg_name", "init_embd", "model_name", "hidden_channels", "onto_incorporation"],
                as_index=False,
            )
            .head(1)
        )
        best_config_path = out_dir / "partial_best_config.csv"
        best_config.to_csv(best_config_path, index=False)
        print(f"\n[SAVE] partial_best_config.csv      ({len(best_config)} lignes)")

        print("\n=== MEILLEURE CONFIG PAR (GRAPHE × HIDDEN) ===")
        print(best_config[cols_show].to_string(index=False))

    print(f"\n[DONE] Fichiers sauvegardés dans : {out_dir}")




def basic_graph_stats(G):
    """
    Statistiques de base pour un graph NetworkX.
    Compatible avec Graph, DiGraph, MultiGraph, MultiDiGraph.
    """

    # Pour les composantes connexes, on ignore la direction si DiGraph
    G_undirected = G.to_undirected()

    components = list(nx.connected_components(G_undirected))

    # Nombre de types de relations
    relation_types = set()

    if G.is_multigraph():
        for _, _, _, data in G.edges(keys=True, data=True):
            rel = data.get("relation") or data.get("predicate") or data.get("edge_type")
            if rel is not None:
                relation_types.add(rel)
    else:
        for _, _, data in G.edges(data=True):
            rel = data.get("relation") or data.get("predicate") or data.get("edge_type")
            if rel is not None:
                relation_types.add(rel)

    stats = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "num_connected_components": len(components),
        "num_relation_types": len(relation_types),
        "average_degree": sum(dict(G.degree()).values()) / G.number_of_nodes()
        if G.number_of_nodes() > 0 else 0,
        "largest_connected_component_size": max(
            (len(c) for c in components),
            default=0
        ),
    }

    return stats
if __name__ == "__main__":
    # merge_json_files("datasets/SM_network.json", "datasets/GT2KG_edc_canonicalized_KG.json", "datasets/GT2KG_edc_canonicalized_onto_KG.json")
    # print("Done")
    terms = set([i.strip().lower() for i in list(pd.read_excel("datasets/common_nodes.xlsx").term)])
    print(len(terms))
    data = read_json_file("datasets/EDC_canonicalized_kg_v2_.json")
    entities = set()
    for element in data:
        entities.add(element["subject"].strip().lower())            # main()
        entities.add(element["object"].strip().lower())
    print(len(entities))
    print(terms - terms.intersection(entities))

                # main()
    # gdp_reverb = GraphDataPreparation(
    #         kg_name="Reverb45k_linked_vf",
    #         model_name_init="sentence-transformers/all-MiniLM-L6-v2",
    #         entities_embd_path=None,
    #         edges_embd_path=None,
    #         is_directed=True,
    #         with_self_loop=False,
    #     )

    # reverb_data = gdp_reverb.prepare_graph_with_type()
    # print(10*"++++")
    # print(len(set(reverb_data.edge_type.tolist())))
    # print(reverb_data)
    # stats = basic_graph_stats(gdp_reverb.nxGraph)
    # print(stats)




    # kg_name = "GT2KG_kg"
    # onto_name = "SM_network"
    # init_embd = "random"#"sentence-transformers/all-MiniLM-L6-v2"
    # split_path = "datasets/split/split_2024.json"
    # tdg = TDGBench(use_classifier=True)

    # # Charger les données une première fois pour récupérer les dimensions utiles
    # annotated_graph, train_loader, val_loader, test_loader, gdp = tdg.get_data(
    #     kg_name=kg_name,
    #     init_embd=init_embd,
    #     split_path=split_path
    # )

    # print(annotated_graph)

    # in_channels = annotated_graph.x.shape[1]
    # num_relations = int(annotated_graph.edge_type.max().item()) + 1

    # model_name = "RotatEGCN_attn"   # à adapter
    # extra_kwargs = get_default_model_kwargs(model_name)

    # def model_factory(
    #     _model_name=model_name,
    #     _in=in_channels,
    #     _h=256,
    #     _nr=num_relations,
    #     _kwargs=extra_kwargs
    # ):
    #     return build_encoder(
    #         model_name=_model_name,
    #         in_channels=_in,
    #         hidden_channels=_h,
    #         out_channels=_h,
    #         num_relations=_nr,
    #         **_kwargs
    #     )

    # results = tdg.evaluate_with_onto(
    #     kg_name=kg_name,
    #     onto_name=onto_name,
    #     model_factory=model_factory,
    #     init_embd=init_embd,
    #     split_path=split_path,
    #     onto_type_names=[
    #                         "Body Part, Organ, or Organ Component",
    #                         "Disease or Syndrome",
    #                         "Finding",
    #                         "Intellectual Product",
    #                         "Laboratory Procedure",
    #                         "Organic Chemical",
    #                         "Pharmacologic Substance",
    #                         "Therapeutic or Preventive Procedure"
    #                     ],
    #     onto_sim_save_path="results/onto_similarity.png"

    # )


    #########################################
    #########################################


    # print(results)


    # reduced_ontology = read_json_file("ontology_level4_entity.json")
    # types = set()
    # for key,triplet in reduced_ontology.items():
    #     for type_ in triplet["triplets"]:
    #         if key != "isa":
    #             types.add(type_[0])
    #             types.add(type_[2])
    
    # print(len(types))
    # print(types)

    # reduced_ontology, mapping, depth = reduce_ontology_granularity(
    # input_path="datasets/ontology/semantic_relations_with_triplets_and_definitions.json",
    # output_path="ontology_level4_entity.json",
    # target_level=2,
    # reduce_isa_too=False,
    # root_mode="entity"
    # )

    # print("Nombre de types dans le mapping :", len(set(mapping.values())))
    # print("Exemples :")
    # for k, v in list(mapping.items())[:20]:
    #     print(f"{k} -> {v}")
    # print(15*"******")
    # print(set(mapping.values()))



    # file1_path = 'datasets/GT2KG_edc_canonicalized_KG.json'
    # file2_path = 'datasets/is_a_augmentation_MM_mapped_nci_All_R_KG.json'
    # output_path = 'datasets/GT2KG_edc_canonicalized_KG_is_a_aug.json'
    # merge_json_files(file1_path, file2_path, output_path)
    # print("Done")
