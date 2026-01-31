import torch
from torch_geometric.data import Data

def view_partial_features_masking(data, max_masking_percentage=0.3, random_seed=42):
    """
    Génère une vue augmentée du graphe en appliquant un masquage partiel exact
    sur les caractéristiques des nœuds, respectant le pourcentage de masquage pour chaque nœud.

    Paramètres:
    - data : objet de type Data avec les attributs x, edge_index, edge_attr, et edge_type
    - max_masking_percentage : pourcentage maximal de masquage pour chaque nœud (par défaut 20%)

    Retourne:
    - data_augmented : une copie de l'objet Data avec les caractéristiques des nœuds masquées
    """
    # Vérifier l'appareil et déplacer tous les éléments de data sur le même appareil



    device = data.x.device
    torch.manual_seed(random_seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(random_seed)

    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    if data.edge_attr is not None:
        data.edge_attr = data.edge_attr.to(device)
    if data.edge_type is not None:
        data.edge_type = data.edge_type.to(device)

    # Récupérer le nombre de nœuds et de caractéristiques
    num_nodes, num_features = data.x.shape

    # Générer un taux de masquage aléatoire pour chaque nœud entre 0 et max_masking_percentage
    node_masking_ratios = (torch.rand(num_nodes, device=device) * max_masking_percentage)

    # Créer un masque exact pour chaque nœud
    mask = torch.ones((num_nodes, num_features), device=device)
    for i in range(num_nodes):
        # Calculer le nombre exact de caractéristiques à masquer pour le nœud i
        num_features_to_mask = int(num_features * node_masking_ratios[i].item())

        # Choisir aléatoirement les indices des caractéristiques à masquer
        mask_indices = torch.randperm(num_features, device=device)[:num_features_to_mask]

        # Appliquer le masquage aux indices sélectionnés
        mask[i, mask_indices] = 0

    # Appliquer le masque aux caractéristiques pour obtenir la vue partiellement masquée
    masked_x = data.x * mask

    # Créer une copie de data pour la vue augmentée avec les caractéristiques masquées
    data_augmented = data.clone()
    data_augmented.x = masked_x

    return data_augmented

## without_limite_per_type
def relation_based_edge_dropping_balanced(data, total_drop_rate, max_drop_fraction_per_node=0.3, random_seed=42):

    """
    Suppression équilibrée des relations dans un graphe en limitant les suppressions par nœud
    tout en évitant l'isolation des nœuds.

    :param data: Objet Data contenant le graphe.
    :param total_drop_rate: Fraction globale d'arêtes à supprimer.
    :param max_drop_fraction_per_node: Fraction maximale d'arêtes pouvant être supprimées par nœud.
    :return: Nouvel objet Data avec les arêtes mises à jour, indices supprimés, et types supprimés.
    """
    # Créer des copies pour éviter de modifier l'objet data original
    device = data.x.device
    torch.manual_seed(random_seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(random_seed)

    assert 0 <= total_drop_rate <= 1, "total_drop_rate doit être entre 0 et 1"
    assert 0 <= max_drop_fraction_per_node <= 1, "max_drop_fraction_per_node doit être entre 0 et 1"

    edge_index = data.edge_index.clone()
    edge_type = data.edge_type.clone()

    # Calcul de la fréquence de chaque type de relation
    edge_types, edge_counts = torch.unique(edge_type, return_counts=True)
    total_edges = edge_index.size(1)

    # Calcul du nombre total d'arêtes à supprimer
    num_edges_to_drop = int(total_edges * total_drop_rate)

    # Calcul du nombre d'arêtes à supprimer par type de relation
    edges_to_drop_per_type = {edge_type.item(): int(num_edges_to_drop * (count.item() / total_edges))
                              for edge_type, count in zip(edge_types, edge_counts)}

    # Suivi des arêtes à garder
    keep_edge_indices = list(range(total_edges))
    removed_edge_indices = []  # Indices des arêtes supprimées

    # Pour chaque type de relation, supprimer les arêtes de manière équilibrée
    for edge_type_value, edges_to_drop in edges_to_drop_per_type.items():
        # Trouver les indices des arêtes de ce type
        edge_indices_of_type = torch.where(edge_type == edge_type_value)[0].tolist()

        # Indices à supprimer
        indices_to_remove = []

        while len(indices_to_remove) < edges_to_drop and edge_indices_of_type:
            candidate = edge_indices_of_type.pop(torch.randint(0, len(edge_indices_of_type), (1,)).item())

            # Vérifier la contrainte de suppression par nœud
            src, dst = edge_index[:, candidate]
            remaining_edges_src = len(torch.where(edge_index[0] == src)[0]) + len(
                torch.where(edge_index[1] == src)[0])
            remaining_edges_dst = len(torch.where(edge_index[0] == dst)[0]) + len(
                torch.where(edge_index[1] == dst)[0])

            max_removable_edges_src = int(remaining_edges_src * max_drop_fraction_per_node)
            max_removable_edges_dst = int(remaining_edges_dst * max_drop_fraction_per_node)

            # Vérifier également la condition de non-isolation des nœuds
            if (remaining_edges_src > max_removable_edges_src and
                    remaining_edges_dst > max_removable_edges_dst and
                    remaining_edges_src > 1 and
                    remaining_edges_dst > 1):
                indices_to_remove.append(candidate)

        # Mettre à jour les listes des arêtes gardées et supprimées
        removed_edge_indices.extend(indices_to_remove)
        keep_edge_indices = [idx for idx in keep_edge_indices if idx not in indices_to_remove]

    # Récupérer les types des arêtes supprimées
    removed_edge_types = edge_type[removed_edge_indices]

    # Création d'un nouvel objet Data avec les arêtes mises à jour
    new_data = Data(
        x=data.x,  # Copie des nœuds
        edge_index=edge_index[:, keep_edge_indices],  # Arêtes mises à jour
        edge_type=edge_type[keep_edge_indices],  # Types d'arêtes mis à jour
        edge_attr= data.edge_attr[keep_edge_indices]
    )

    return new_data, torch.tensor(removed_edge_indices), removed_edge_types

### with limit per type
def relation_based_edge_dropping_balanced_type(data, total_drop_rate, max_drop_fraction_per_node=0.3, type_limite=0.3,
                                          random_seed=42):
    """
    Suppression équilibrée des relations dans un graphe en limitant les suppressions par nœud
    et par type tout en évitant l'isolation des nœuds.

    :param data: Objet Data contenant le graphe.
    :param total_drop_rate: Fraction globale d'arêtes à supprimer.
    :param max_drop_fraction_per_node: Fraction maximale d'arêtes pouvant être supprimées par nœud.
    :param type_limite: Fraction maximale des arêtes d'un même type pouvant être supprimées par nœud.
    :param random_seed: Graine pour la reproductibilité.
    :return: Nouvel objet Data avec les arêtes mises à jour, indices supprimés, et types supprimés.
    """
    import torch
    from torch_geometric.data import Data

    # Initialisation et graine pour la reproductibilité
    device = data.x.device
    torch.manual_seed(random_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(random_seed)

    # Vérification des arguments
    assert 0 <= total_drop_rate <= 1, "total_drop_rate doit être entre 0 et 1"
    assert 0 <= max_drop_fraction_per_node <= 1, "max_drop_fraction_per_node doit être entre 0 et 1"
    assert 0 <= type_limite <= 1, "type_limite doit être entre 0 et 1"

    # Récupération des informations du graphe
    edge_index = data.edge_index.clone()
    edge_type = data.edge_type.clone()
    total_edges = edge_index.size(1)

    # Calcul des statistiques globales
    edge_types, edge_counts = torch.unique(edge_type, return_counts=True)
    num_edges_to_drop = int(total_edges * total_drop_rate)
    edges_to_drop_per_type = {et.item(): int(num_edges_to_drop * (count.item() / total_edges))
                              for et, count in zip(edge_types, edge_counts)}

    # Suivi des suppressions par nœud et par type
    node_edge_counts = {node.item(): 0 for node in torch.unique(edge_index)}
    node_edge_type_counts = {node.item(): {et.item(): 0 for et in edge_types} for node in torch.unique(edge_index)}

    # Listes pour les arêtes conservées et supprimées
    keep_edge_indices = list(range(total_edges))
    removed_edge_indices = []

    # Suppression par type de relation
    for et_value, edges_to_drop in edges_to_drop_per_type.items():
        edge_indices_of_type = torch.where(edge_type == et_value)[0].tolist()
        indices_to_remove = []

        while len(indices_to_remove) < edges_to_drop and edge_indices_of_type:
            candidate = edge_indices_of_type.pop(torch.randint(0, len(edge_indices_of_type), (1,)).item())
            src, dst = edge_index[:, candidate]

            # Comptage des arêtes restantes
            remaining_edges_src = len(torch.where((edge_index[0] == src) | (edge_index[1] == src))[0])
            remaining_edges_dst = len(torch.where((edge_index[0] == dst) | (edge_index[1] == dst))[0])

            # Limite globale de suppression
            max_removable_src = int(remaining_edges_src * max_drop_fraction_per_node)
            max_removable_dst = int(remaining_edges_dst * max_drop_fraction_per_node)

            # Limite par type
            max_type_removable_src = int(node_edge_type_counts[src.item()][et_value] * type_limite)
            max_type_removable_dst = int(node_edge_type_counts[dst.item()][et_value] * type_limite)

            # Vérification des contraintes
            if (node_edge_counts[src.item()] < max_removable_src and
                    node_edge_counts[dst.item()] < max_removable_dst and
                    node_edge_type_counts[src.item()][et_value] < max_type_removable_src and
                    node_edge_type_counts[dst.item()][et_value] < max_type_removable_dst and
                    remaining_edges_src > 1 and remaining_edges_dst > 1):
                indices_to_remove.append(candidate)
                node_edge_counts[src.item()] += 1
                node_edge_counts[dst.item()] += 1
                node_edge_type_counts[src.item()][et_value] += 1
                node_edge_type_counts[dst.item()][et_value] += 1

        removed_edge_indices.extend(indices_to_remove)
        keep_edge_indices = [idx for idx in keep_edge_indices if idx not in indices_to_remove]

    # Types des arêtes supprimées
    removed_edge_types = edge_type[removed_edge_indices]

    # Création d'un nouvel objet Data
    new_data = Data(
        x=data.x,  # Copie des nœuds
        edge_index=edge_index[:, keep_edge_indices],  # Arêtes restantes
        edge_type=edge_type[keep_edge_indices],  # Types d'arêtes mises à jour
    )

    return new_data, torch.tensor(removed_edge_indices), removed_edge_types

