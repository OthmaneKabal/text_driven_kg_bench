import torch
import torch.nn.functional as F

def type_consistency_loss(logits, edge_index, edge_type, valid_pairs_per_type, device):
    """
    Pénalise la masse de probabilité sur les paires de types interdites par l'ontologie.

    logits               : (N, num_classes) — sorties brutes du modèle (avant softmax)
    edge_index           : (2, E) — arêtes du batch courant
    edge_type            : (E,)  — type de chaque arête
    valid_pairs_per_type : {edge_type_id: tensor([[ci,cj],...])} — depuis OntologyConstraintBuilder
    device               : device courant

    Retourne un scalaire.
    """
    probs = F.softmax(logits, dim=1)
    total_loss = torch.tensor(0.0, device=device)
    count = 0

    for rel_id, valid_pairs in valid_pairs_per_type.items():
        mask = (edge_type == rel_id)
        if not mask.any():
            continue

        src = edge_index[0][mask]  # indices locaux du batch
        dst = edge_index[1][mask]

        p_src = probs[src]  # (E_r, num_classes)
        p_dst = probs[dst]

        ci = valid_pairs[:, 0].to(device)  # (K,)
        cj = valid_pairs[:, 1].to(device)

        # Masse sur les paires VALIDES : somme sur K paires de p_src[ci] * p_dst[cj]
        # p_src[:, ci] → (E_r, K)
        valid_mass = (p_src[:, ci] * p_dst[:, cj]).sum(dim=1)  # (E_r,)

        # On veut maximiser valid_mass → on minimise (1 - valid_mass)
        total_loss += (1.0 - valid_mass).mean()
        count += 1

    return total_loss / count if count > 0 else torch.tensor(0.0, device=device)