from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

# class GATEncoder(nn.Module):
#     def __init__(self, data: Data, out_channels, num_layers=2, heads=4, dropout=0.5):
#         super(GATEncoder, self).__init__()
#         self.out_channels = out_channels[-1]
#         in_channels = data.x.shape[1]
#         self.convs = nn.ModuleList()
#         self.bns = nn.ModuleList()
#         self.heads = heads
#
#         for i in range(num_layers):
#             input_dim = in_channels if i == 0 else out_channels[i - 1]
#             self.convs.append(GATConv(input_dim, out_channels[i], heads=heads, concat=False, dropout=dropout))
#             # Since concat=False, output dimension is out_channels[i], not out_channels[i] * heads
#             self.bns.append(nn.BatchNorm1d(out_channels[i]))
#
#         self.dropout = nn.Dropout(p=dropout)
#
#         # Final layer - input is out_channels[-1] since concat=False
#         self.final_layer = nn.Linear(out_channels[-1], out_channels[-1])
#         self.relu = nn.ReLU()
#
#     def forward(self, data: Data):
#         x, edge_index = data.x, data.edge_index
#         for conv, bn in zip(self.convs, self.bns):
#             x = conv(x, edge_index)
#             x = bn(x)
#             x = self.relu(x)
#             x = self.dropout(x)
#
#         x = self.final_layer(x)
#         return x

import torch.nn as nn
from torch_geometric.nn import GATConv


class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 heads=4, dropout=0.5, batch_norm=True, concat=False):
        """
        Initialize the GAT encoder with ReLU, BatchNorm, and Dropout between each layer.

        Parameters:
        - in_channels: Input feature dimension
        - hidden_channels: Hidden dimension (int or list)
        - out_channels: Output embedding dimension
        - num_layers: Number of stacked GAT layers
        - heads: Number of attention heads
        - dropout: Dropout probability for regularization
        - batch_norm: Whether to use batch normalization
        - concat: Whether to concatenate attention heads (if False, averages them)
        """
        super(GATEncoder, self).__init__()

        self.num_layers = num_layers
        self.dropout_prob = dropout
        self.use_batch_norm = batch_norm
        self.heads = heads
        self.concat = concat

        # Gérer hidden_channels comme int ou list
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels] * (num_layers - 1)

        # Dimensions de chaque couche
        layer_dims = [in_channels] + hidden_channels + [out_channels]

        # Create layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if batch_norm else None

        for i in range(num_layers):
            # Pour les couches intermédiaires, on peut concat les heads
            # Pour la dernière couche, on moyenne (concat=False) pour avoir out_channels
            is_last_layer = (i == num_layers - 1)
            layer_concat = False if is_last_layer else concat
            layer_heads = 1 if is_last_layer else heads

            self.convs.append(
                GATConv(
                    layer_dims[i],
                    layer_dims[i + 1],
                    heads=layer_heads,
                    concat=layer_concat,
                    dropout=dropout
                )
            )

            # Batch norm dimension depends on concat
            if batch_norm and i < num_layers - 1:
                bn_dim = layer_dims[i + 1] * layer_heads if layer_concat else layer_dims[i + 1]
                self.bns.append(nn.BatchNorm1d(bn_dim))

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

        # Store output dimension for classifier
        self.out_channels = out_channels

    def reset_parameters(self):
        """Reset the parameters of the encoder layers."""
        for conv in self.convs:
            conv.reset_parameters()
        if self.bns is not None:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass through the network.

        Parameters:
        - x: Node features [num_nodes, in_channels]
        - edge_index: Edge indices [2, num_edges]
        - edge_weight: Optional edge weights [num_edges]

        Returns:
        - Node embeddings [num_nodes, out_channels]
        """
        for i, conv in enumerate(self.convs):
            # GAT can optionally use edge_attr, but typically doesn't use edge_weight
            x = conv(x, edge_index)

            # Appliquer BN + ReLU + Dropout sauf pour la dernière couche
            if i < self.num_layers - 1:
                if self.bns is not None:
                    x = self.bns[i](x)
                x = self.relu(x)
                x = self.dropout(x)

        return x