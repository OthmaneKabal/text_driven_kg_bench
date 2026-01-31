import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def generate_and_save_splits(
    xlsx_path: str,
    save_dir: str,
    n_splits: int = 5,
    seeds=None,
    train_ratio: float = 0.20,
    val_ratio: float = 0.05,
    test_ratio: float = 0.75,
    stratify: bool = True,
    sheet_name=0,
    index_col: str | None = None,
    label_col: str | None = None,
):
    """
    Lit un .xlsx qui contient (au minimum) une colonne d'index (GS) et une colonne de labels,
    puis génère n_splits splits train/val/test et sauvegarde:
      - split_{i}.json
      - train_idx_{i}.npy / val_idx_{i}.npy / test_idx_{i}.npy
      - splits_summary.json

    Paramètres utiles:
      - sheet_name: nom/idx de la feuille Excel
      - index_col / label_col: si tu veux forcer les noms de colonnes
        (sinon autodétection parmi des noms courants)
    """

    # --- Checks ratios ---
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Les ratios doivent sommer à 1.0, reçu: {total}")

    # --- Load xlsx ---
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, engine="openpyxl")

    # --- Autodétection colonnes si non fournies ---
    if index_col is None:
        index_candidates = [
            "gs_idx", "gs_index", "gs", "index", "idx", "node_idx", "node_index", "id", "term_id"
        ]
        for c in index_candidates:
            if c in df.columns:
                index_col = c
                break

    if label_col is None:
        label_candidates = [
            "label", "labels", "y", "class", "target", "category"
        ]
        for c in label_candidates:
            if c in df.columns:
                label_col = c
                break

    if index_col is None or label_col is None:
        raise ValueError(
            "Impossible de détecter automatiquement les colonnes.\n"
            f"Colonnes disponibles: {list(df.columns)}\n"
            "Passe explicitement index_col=... et label_col=... (ex: index_col='gs_idx', label_col='label')."
        )

    sub = df[[index_col, label_col]].copy()
    sub = sub.dropna(subset=[index_col, label_col])

    # indices (GS) et labels alignés
    gs_indices = sub[index_col].astype(int).tolist()
    labels = sub[label_col].tolist()

    if len(gs_indices) == 0:
        raise ValueError("Aucune ligne valide (index+label) trouvée dans le fichier Excel.")

    # --- Prepare output ---
    os.makedirs(save_dir, exist_ok=True)

    if seeds is None:
        seeds = list(range(n_splits))
    else:
        if len(seeds) != n_splits:
            n_splits = len(seeds)

    splits_summary = {}

    for split_id, seed in enumerate(seeds):
        # Stratify only if meaningful
        stratify_labels = labels if stratify and len(set(labels)) > 1 else None

        # TRAIN vs TEMP
        gs_train, gs_temp, y_train, y_temp = train_test_split(
            gs_indices,
            labels,
            train_size=train_ratio,
            random_state=seed,
            stratify=stratify_labels
        )

        # VAL vs TEST within TEMP
        relative_val = val_ratio / (val_ratio + test_ratio)
        stratify_temp = y_temp if stratify and len(set(y_temp)) > 1 else None

        gs_val, gs_test, _, _ = train_test_split(
            gs_temp,
            y_temp,
            train_size=relative_val,
            random_state=seed,
            stratify=stratify_temp
        )

        split_data = {
            "train_idx": gs_train,  # INDEX GS
            "val_idx": gs_val,      # INDEX GS
            "test_idx": gs_test     # INDEX GS
        }

        # --- Save JSON ---
        with open(os.path.join(save_dir, f"split_{seed}.json"), "w", encoding="utf-8") as f:
            json.dump(split_data, f)

        # --- Save NPY ---
        np.save(os.path.join(save_dir, f"train_idx_{seed}.npy"), np.array(gs_train, dtype=int))
        np.save(os.path.join(save_dir, f"val_idx_{seed}.npy"), np.array(gs_val, dtype=int))
        np.save(os.path.join(save_dir, f"test_idx_{seed}.npy"), np.array(gs_test, dtype=int))

        splits_summary[f"split_{seed}"] = {
            "seed": int(seed),
            "train_size": int(len(gs_train)),
            "val_size": int(len(gs_val)),
            "test_size": int(len(gs_test)),
            "index_col": index_col,
            "label_col": label_col,
            "source_xlsx": os.path.basename(xlsx_path),
            "sheet_name": str(sheet_name),
        }

        print(
            f"[INFO] Split {split_id} (seed={seed}) → "
            f"Train: {len(gs_train)}, Val: {len(gs_val)}, Test: {len(gs_test)}"
        )

    with open(os.path.join(save_dir, "splits_summary.json"), "w", encoding="utf-8") as f:
        json.dump(splits_summary, f, indent=2)

    print(f"[INFO] {len(seeds)} splits sauvegardés dans {save_dir}")
    return splits_summary

