def build_encoder(model_name: str,
                  in_channels: int,
                  hidden_channels: int,
                  out_channels: int,
                  num_relations: int,
                  **kwargs):
    """
    Factory function to build different types of graph encoders.

    Supported models:
    - GCN: Graph Convolutional Network
    - RGCN: Relational Graph Convolutional Network
    - GAT: Graph Attention Network
    - TransGCN_conv_TransE: TransGCN with convolution variant and TransE scoring
    - TransGCN_conv_RotatE: TransGCN with convolution variant and RotatE scoring
    - TransGCN_attn_TransE: TransGCN with attention variant and TransE scoring
    - TransGCN_attn_RotatE: TransGCN with attention variant and RotatE scoring

    Parameters:
    - model_name: Name of the model to build
    - in_channels: Input feature dimension
    - hidden_channels: Hidden dimension (int or list)
    - out_channels: Output embedding dimension
    - num_relations: Number of relation types (for RGCN and TransGCN)
    - **kwargs: Additional model-specific parameters (num_layers, dropout, heads, etc.)

    Returns:
    - encoder: torch.nn.Module
    """

    # ==================== GCN ====================
    if model_name == "GCN":
        from models.GCNEncoder import GCNEncoder
        return GCNEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            **kwargs
        )

    # ==================== RGCN ====================
    elif model_name == "RGCN":
        from models.RGCNEncoder import RGCNEncoder
        return RGCNEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_relations=num_relations,
            **kwargs
        )

    # ==================== GAT ====================
    elif model_name == "GAT":
        from models.GATEncoder import GATEncoder
        return GATEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            **kwargs
        )

    # ==================== TransGCN Variants ====================
    # TransGCN with convolution variant and TransE scoring
    elif model_name == "TransEGCN_conv":
        from models.TransGCNEncoder import TransGCNEncoder
        return TransGCNEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            kg_score_fn="TransE",
            variant="conv",
            **kwargs
        )

    # TransGCN with convolution variant and RotatE scoring
    elif model_name == "RotatEGCN_conv":
        from models.TransGCNEncoder import TransGCNEncoder
        return TransGCNEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            kg_score_fn="RotatE",
            variant="conv",
            **kwargs
        )

    # TransGCN with attention variant and TransE scoring
    elif model_name == "TransEGCN_attn":
        from models.TransGCNEncoder import TransGCNEncoder
        return TransGCNEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            kg_score_fn="TransE",
            variant="attn",
            **kwargs
        )

    # TransGCN with attention variant and RotatE scoring
    elif model_name == "RotatEGCN_attn":
        from models.TransGCNEncoder import TransGCNEncoder
        return TransGCNEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            kg_score_fn="RotatE",
            variant="attn",
            **kwargs
        )

    else:
        raise ValueError(
            f"Unknown model_name: {model_name}\n"
            f"Supported models: GCN, RGCN, GAT, "
            f"TransGCN_conv_TransE, TransGCN_conv_RotatE, "
            f"TransGCN_attn_TransE, TransGCN_attn_RotatE"
        )


# ==================== Configuration par défaut pour chaque modèle ====================
def get_default_model_kwargs(model_name: str):
    """
    Get default hyperparameters for each model type.

    Returns a dictionary of default kwargs that can be overridden.
    """
    defaults = {
        "GCN": {
            "num_layers": 2,
            "dropout": 0.5,
            "batch_norm": True
        },
        "RGCN": {
            "num_layers": 2,
            "dropout": 0.5,
            "num_bases": 50,
            "batch_norm": True
        },
        "GAT": {
            "num_layers": 2,
            "dropout": 0.5,
            "heads": 4,
            "batch_norm": True,
            "concat": False
        },
        "TransEGCN_conv": {
            "num_layers": 2,
            "dropout": 0.5,
            "batch_norm": True,
            "use_edges_info": True,
            "activation": "relu",
            "bias": True
        },
        "RotatEGCN_conv": {
            "num_layers": 2,
            "dropout": 0.5,
            "batch_norm": True,
            "use_edges_info": True,
            "activation": "relu",
            "bias": True
        },
        "TransEGCN_attn": {
            "num_layers": 2,
            "dropout": 0.5,
            "batch_norm": True,
            "use_edges_info": True,
            "activation": "relu",
            "bias": True
        },
        "RotatEGCN_attn": {
            "num_layers": 2,
            "dropout": 0.5,
            "batch_norm": True,
            "use_edges_info": True,
            "activation": "relu",
            "bias": True
        }
    }

    return defaults.get(model_name, {})



