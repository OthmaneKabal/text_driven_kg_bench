import torch

from models.GATEncoder import GATEncoder
from models.GCNEncoder import GCNEncoder
from models.RGCNEncoder import RGCNEncoder
from models.TransGCNEncoder import TransGCNEncoder


def instantiate_encoder(config_, data):
    encoder_type = config_["classifier_encoder"]
    out_channels = config_["encoder_out_channels"]
    device = torch.device(config_.get("device"))
    num_layers = config_["num_layers"]
    use_edges_info = config_.get("use_edges_info", False)
    num_bases = config_.get("num_bases", None)
    msg_sens = config_.get("message_sens", "source_to_target")
    dropout = config_.get("dropout", 0.5)

    if encoder_type == "GCN":
        encoder = GCNEncoder(data, out_channels, num_layers, dropout=dropout,
                             message_sens=msg_sens).to(device)

    elif encoder_type == "RGCN":
        encoder = RGCNEncoder(data, out_channels, num_layers, num_bases=num_bases,
                              dropout=dropout, message_sens=msg_sens).to(device)

    elif encoder_type in ["TransGCN_conv", "TransGCN_attn"]:
        variant = "conv" if "conv" in encoder_type else "attn"
        encoder = TransGCNEncoder(
            data, out_channels, num_layers, dropout=dropout,
            kg_score_fn='TransE', variant=variant,
            use_edges_info=use_edges_info, activation='relu',
            bias=False
        ).to(device)

    elif encoder_type in ["RotatEGCN_conv", "RotatEGCN_attn"]:
        variant = "conv" if "conv" in encoder_type else "attn"
        encoder = TransGCNEncoder(
            data, out_channels, num_layers, dropout=dropout,
            kg_score_fn='RotatE', variant=variant,
            use_edges_info=use_edges_info, activation='relu',
            bias=False
        ).to(device)

    elif encoder_type == "GAT":
        gat_heads = config_.get("gat_heads", 4)
        encoder = GATEncoder(data, out_channels, num_layers,
                             heads=gat_heads, dropout=dropout).to(device)

    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    return encoder
