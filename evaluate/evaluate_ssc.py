import torch
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

@torch.no_grad()
def ssc_evaluate_split(model, data_loader, split: str, config, gdp=None, save_path=None):
    """
    Évalue un modèle sur un split (train, val, test) avec mini-batches.

    Args:
        model: modèle GNN
        data_loader: NeighborLoader associé au split
        split (str): "train", "val" ou "test"
        config (dict): dictionnaire de configuration
        gdp (GraphDataPreparation, optional): pour décoder les labels/termes
        save_path (str, optional): chemin pour sauvegarder les prédictions dans un Excel
    """
    assert split in ["train", "val", "test"], "split doit être 'train', 'val' ou 'test'"

    model.eval()
    y_true, preds, node_ids = [], [], []

    mask_name = f"{split}_mask"

    for batch in data_loader:
        batch = batch.to(config["device"])
        out = model(batch)

        # On garde seulement les nœuds d'entrée annotés du bon split
        mask_input_id = torch.isin(batch.n_id, batch.input_id)
        mask = mask_input_id & getattr(batch, mask_name) & (batch.y >= 0)

        batch_y_true = batch.y[mask].cpu()
        batch_y_pred = out[mask].argmax(dim=1).cpu()
        batch_node_ids = batch.n_id[mask].cpu()

        y_true.append(batch_y_true)
        preds.append(batch_y_pred)
        node_ids.append(batch_node_ids)

    if len(y_true) == 0:
        print(f"[WARN] Aucun nœud annoté trouvé pour le split '{split}'")
        return None

    # Concaténation
    y_true = torch.cat(y_true, dim=0).numpy()
    preds = torch.cat(preds, dim=0).numpy()
    node_ids = torch.cat(node_ids, dim=0).numpy()

    # Décodage si un label_encoder est fourni
    if gdp and gdp.label_encoder:
        decoded_true = gdp.label_encoder.inverse_transform(y_true)
        decoded_pred = gdp.label_encoder.inverse_transform(preds)
    else:
        decoded_true, decoded_pred = y_true, preds

    # Sauvegarde des prédictions
    if save_path:
        terms = [gdp.decode_indexes()[id] for id in node_ids] if gdp else node_ids
        df = pd.DataFrame({
            "term": terms,
            "label": decoded_true,
            "predictions": decoded_pred
        })
        df.to_excel(save_path, index=False)
        print(f"Node-level predictions for '{split}' saved to {save_path}")

    # Calcul des métriques
    return {
        'split': split,
        'accuracy': accuracy_score(y_true, preds),
        'recall': recall_score(y_true, preds, average='macro', zero_division=0),
        'f1': f1_score(y_true, preds, average='macro', zero_division=0),
        'precision': precision_score(y_true, preds, average='macro', zero_division=0)
    }
