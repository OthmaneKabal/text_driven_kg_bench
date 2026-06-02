import os
import json
import argparse
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

    if label_col is None:
        raise ValueError(
            "Impossible de détecter automatiquement les colonnes.\n"
            f"Colonnes disponibles: {list(df.columns)}\n"
            "Passe explicitement label_col=... (ex: label_col='label')."
        )

    if index_col is None:
        print("[INFO] Aucune colonne d'index detectee: utilisation de l'index de ligne du GS.")
        sub = df[[label_col]].copy()
        sub = sub.dropna(subset=[label_col])
        index_col_for_summary = "__row_index__"
    else:
        sub = df[[index_col, label_col]].copy()
        sub = sub.dropna(subset=[index_col, label_col])
        index_col_for_summary = index_col

    # indices (GS) et labels alignés
    if index_col is None:
        gs_indices = sub.index.astype(int).tolist()
    else:
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
            "index_col": index_col_for_summary,
            "label_col": label_col,
            "source_xlsx": os.path.basename(xlsx_path),
            "sheet_name": str(sheet_name),
        }

        print(
            f"[INFO] Split {split_id} (seed={seed}) -> "
            f"Train: {len(gs_train)}, Val: {len(gs_val)}, Test: {len(gs_test)}"
        )

    with open(os.path.join(save_dir, "splits_summary.json"), "w", encoding="utf-8") as f:
        json.dump(splits_summary, f, indent=2)

    print(f"[INFO] {len(seeds)} splits sauvegardés dans {save_dir}")
    return splits_summary


def _parse_seeds(seeds_text: str | None):
    if seeds_text is None or seeds_text.strip() == "":
        return None

    return [int(seed.strip()) for seed in seeds_text.split(",") if seed.strip()]


def _parse_sheet_name(sheet_name):
    if isinstance(sheet_name, int):
        return sheet_name
    return int(sheet_name) if str(sheet_name).isdigit() else sheet_name


def main():
    parser = argparse.ArgumentParser(
        description="Generate reproducible train/val/test splits from a GS Excel file."
    )
    parser.add_argument("--xlsx-path", required=True, help="Path to the GS .xlsx file.")
    parser.add_argument("--save-dir", required=True, help="Directory where splits will be saved.")
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds, for example: 42,123,456,789,2024.",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.20)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.75)
    parser.add_argument("--sheet-name", default=0)
    parser.add_argument("--index-col", default=None)
    parser.add_argument("--label-col", default=None)
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable label stratification.",
    )

    args = parser.parse_args()
    seeds = _parse_seeds(args.seeds)

    generate_and_save_splits(
        xlsx_path=args.xlsx_path,
        save_dir=args.save_dir,
        n_splits=args.n_splits,
        seeds=seeds,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        stratify=not args.no_stratify,
        sheet_name=_parse_sheet_name(args.sheet_name),
        index_col=args.index_col,
        label_col=args.label_col,
    )


if __name__ == "__main__":
    main()

