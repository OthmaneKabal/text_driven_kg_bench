import copy
from typing import Optional, List
import torch
from torch_geometric.loader import NeighborLoader
from data_preprocessing.GraphDataPreparation import GraphDataPreparation

_cached = {}

def fflush_cached():
    global _cached
    _cached = {}

def get_data_and_loaders(
    kg_name: str,
    model_name_init: str,
    common_nodes_path: str,
    entities_embd_path: Optional[str] = None,
    edges_embd_path: Optional[str] = None,
    split_file: Optional[str] = None,
    train_ratio: float = 0.1,
    val_ratio: float = 0.1,
    test_ratio: float = 0.80,
    seed: int = 444,
    num_neighbors: List[int] = [200, 200],
    train_batch_size: int = 512,
    val_batch_size: int = 512,
    test_batch_size: int = 512,
    shuffle: bool = False,
    is_directed: bool = True,
    use_cache: bool = True,
    random_embd_dim = 256
):
    if use_cache and "annotated_graph" in _cached:
        return (
            copy.deepcopy(_cached["annotated_graph"]),
            _cached["train_loader"],
            _cached["val_loader"],
            _cached["test_loader"],
            _cached["gdp"]
        )

    print("Initializing data and loaders...")

    # -------- Graph preparation --------
    gdp = GraphDataPreparation(
        kg_name=kg_name,
        model_name_init=model_name_init,
        entities_embd_path=entities_embd_path,
        edges_embd_path=edges_embd_path,
        is_directed=is_directed,
        emb_dim = random_embd_dim
    )

    data = gdp.prepare_graph_with_type_label(common_nodes_path)

    # -------- Splits --------
    if split_file is not None:
        data = gdp.load_split(data, split_file)
        print(f"[INFO] Loaded split from {split_file}")
    else:
        data = gdp.split_data_with_labels(
            data,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_state=seed,
        )
        print("[INFO] Generated random split")

    annotated_graph = data

    # -------- Loaders --------
    train_loader = NeighborLoader(
        annotated_graph,
        input_nodes=annotated_graph.train_mask,
        num_neighbors=num_neighbors,
        batch_size=train_batch_size,
        shuffle=shuffle,
    )

    val_loader = NeighborLoader(
        annotated_graph,
        input_nodes=annotated_graph.val_mask,
        num_neighbors=num_neighbors,
        batch_size=val_batch_size,
        shuffle=False,
    )

    test_loader = NeighborLoader(
        annotated_graph,
        input_nodes=annotated_graph.test_mask,
        num_neighbors=num_neighbors,
        batch_size=test_batch_size,
        shuffle=False,
    )

    # -------- Cache --------
    if use_cache:
        _cached["annotated_graph"] = annotated_graph.cpu()
        _cached["train_loader"] = train_loader
        _cached["val_loader"] = val_loader
        _cached["test_loader"] = test_loader
        _cached["gdp"] = gdp

    return annotated_graph, train_loader, val_loader, test_loader, gdp
