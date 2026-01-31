from torch import nn
from torch_geometric.data import Data
from typing import List
import sys
# sys.path.append('/trans_gcn_layer')
from models.trans_gcn_layer.trans_gnn import TransGNN


# class TransGCNEncoder(nn.Module):
#     def __init__(self, data: Data, out_channels: List[int], num_layers=2,
#                  dropout=0.5, kg_score_fn='TransE', variant='conv' ,use_edges_info=True, activation='relu', bias=False):
#         """
#         Encodeur basé sur TransGCN utilisant plusieurs couches TransGNN avec ReLU, Dropout, etc.
#
#         :param data: objet Data de PyG
#         :param out_channels: liste des dimensions de sortie par couche
#         :param num_layers: nombre de couches TransGNN
#         :param dropout: probabilité de dropout
#         :param kg_score_fn: nom de la fonction de transformation ('TransE' ou autre)
#         :param use_edges_info: si True, utilise les embeddings des relations
#         :param activation: fonction d'activation
#         :param bias: inclut ou non un biais dans les couches
#         """
#         super().__init__()
#         assert len(out_channels) == num_layers, "out_channels doit avoir autant d'éléments que num_layers"
#         self.out_channels = out_channels[-1]
#         in_channels = data.x.shape[1]
#
#         self.layers = nn.ModuleList()
#         self.dropout = nn.Dropout(p=dropout)
#         self.relu = nn.ReLU()
#         print("transgcn0..**********************************************************************************.")
#         for i in range(num_layers):
#             layer_in = in_channels if i == 0 else out_channels[i - 1]
#             layer_out = out_channels[i]
#
#             self.layers.append(TransGNN(
#                 in_channels=layer_in,
#                 out_channels=layer_out,
#                 use_edges_info=use_edges_info,
#                 kg_score_fn=kg_score_fn,
#                 variant=variant,  # comme dans l'article (pas d'attention)
#                 activation=activation,
#                 bias=bias
#             ))
#         self.convs = self.layers
#
#     def reset_parameters(self):
#         for layer in self.layers:
#             if hasattr(layer, 'reset_parameters'):
#                 layer.reset_parameters()
#
#     def forward(self, data: Data):
#         x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
#
#         for layer in self.layers:
#             x, edge_attr = layer(x, edge_index, edge_attr)
#             x = self.dropout(x)
#         return x, edge_attr
#
#     def encode(self, data: Data):
#         x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
#
#         for layer in self.layers:
#             x, edge_attr = layer(x, edge_index, edge_attr)
#             x = self.dropout(x)
#         return x, edge_attr

from typing import List, Union, Optional
import torch
from torch import nn
from torch_geometric.data import Data
from models.trans_gcn_layer.trans_gnn import TransGNN


class TransGCNEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: Union[int, List[int]],
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        batch_norm: bool = True,
        kg_score_fn: str = "TransE", ## or RotatE
        variant: str = "conv", ## or attn
        use_edges_info: bool = True,
        activation: str = "relu",
        bias: bool = False,
    ):
        """
        Implémentation TransGCN "comme GCNEncoder":
        - init(in_channels, hidden_channels, out_channels, num_layers, ...)
        - self.convs: ModuleList de TransGNN
        - (optionnel) BatchNorm + ReLU + Dropout entre couches (sauf dernière)
        - forward(x, edge_index, edge_attr) -> (x, edge_attr)
        """
        super().__init__()

        self.num_layers = num_layers
        self.dropout_prob = dropout
        self.use_batch_norm = batch_norm

        # hidden_channels peut être int ou list
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels] * (num_layers - 1)
        else:
            assert len(hidden_channels) == num_layers - 1, (
                "hidden_channels doit avoir num_layers-1 éléments si c'est une liste"
            )

        # Dimensions de chaque couche
        layer_dims = [in_channels] + list(hidden_channels) + [out_channels]

        # Create layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if batch_norm else None

        for i in range(num_layers):
            self.convs.append(
                TransGNN(
                    in_channels=layer_dims[i],
                    out_channels=layer_dims[i + 1],
                    use_edges_info=use_edges_info,
                    kg_score_fn=kg_score_fn,
                    variant=variant,
                    activation=activation,
                    bias=bias,
                )
            )
            if batch_norm and i < num_layers - 1:  # Pas de BN après la dernière couche
                self.bns.append(nn.BatchNorm1d(layer_dims[i + 1]))

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

        # Store output dimension for classifier
        self.out_channels = out_channels

    def reset_parameters(self):
        """Reset the parameters of the encoder layers."""
        for conv in self.convs:
            if hasattr(conv, "reset_parameters"):
                conv.reset_parameters()
        if self.bns is not None:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass.

        Parameters:
        - x: Node features [num_nodes, in_channels]
        - edge_index: Edge indices [2, num_edges]
        - edge_attr: Edge attributes/relations (selon ton TransGNN)

        Returns:
        - x: Node embeddings [num_nodes, out_channels]
        - edge_attr: Edge attributes mis à jour (si TransGNN les met à jour)
        """
        for i, conv in enumerate(self.convs):
            x, edge_attr = conv(x, edge_index, edge_attr)

            # Appliquer BN + ReLU + Dropout sauf pour la dernière couche
            if i < self.num_layers - 1:
                if self.bns is not None:
                    x = self.bns[i](x)
                x = self.relu(x)
                x = self.dropout(x)

        return x, edge_attr
