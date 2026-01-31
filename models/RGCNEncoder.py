from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv


# class RGCNEncoder(nn.Module):
#     def __init__(self, data: Data, out_channels, num_layers=2, num_bases=30, dropout=0.5, message_sens = "source_to_target"):
#         """
#         Initialize the RGCN encoder with ReLU, BatchNorm, Dropout, and a final linear layer.
#
#         Parameters:
#         - data: Data object to extract input features and relations.
#         - out_channels: List of output feature dimensions for each layer.
#         - num_layers: Number of stacked RGCN layers (must match the length of out_channels).
#         - num_bases: Number of bases to use in each RGCN layer to reduce parameters.
#         - dropout: Dropout probability for regularization.
#         """
#         super(RGCNEncoder, self).__init__()
#
#         # Ensure the number of layers matches the size of out_channels
#         assert len(out_channels) == num_layers, "The length of out_channels must equal num_layers"
#         self.out_channels = out_channels[-1]
#
#         # Extract input dimensions and the number of relations from the data object
#         in_channels = data.x.shape[1]
#         num_relations = data.edge_type.max().item() + 1
#
#         # Create a list of RGCN layers with varying output sizes
#         self.convs = nn.ModuleList()
#         self.bns = nn.ModuleList()  # Batch normalization layers
#         for i in range(num_layers):
#             input_dim = in_channels if i == 0 else out_channels[i - 1]
#             self.convs.append(RGCNConv(input_dim, out_channels[i], num_relations, num_bases=num_bases, flow=message_sens))
#             self.bns.append(nn.BatchNorm1d(out_channels[i]))
#
#         # Dropout layer
#         self.dropout = nn.Dropout(p=dropout)
#
#         # Final linear layer to produce embeddings
#         self.final_layer = nn.Linear(out_channels[-1], self.out_channels)
#
#         # Activation function
#         self.relu = nn.ReLU()
#
#     def reset_parameters(self):
#         """Reset the parameters of the encoder layers."""
#         for conv in self.convs:
#             conv.reset_parameters()
#         for bn in self.bns:
#             bn.reset_parameters()
#         self.final_layer.reset_parameters()
#
#     def forward(self, data: Data):
#         """
#         Forward pass through the network with a Data object as input.
#
#         Parameters:
#         - data: Data object containing x (node features), edge_index (edge indices), and edge_type (edge types).
#
#         Returns:
#         - Node embeddings after passing through the RGCN encoder.
#         """
#         # Extract attributes from the Data object
#         x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
#
#         # Apply each RGCN layer with BatchNorm, ReLU, and Dropout in between
#         for conv, bn in zip(self.convs, self.bns):
#             x = conv(x, edge_index, edge_type)
#             x = bn(x)
#             x = self.relu(x)
#             x = self.dropout(x)
#
#         # Apply the final linear layer to produce embeddings
#         x = self.final_layer(x)
#
#         return x

import torch.nn as nn
from torch_geometric.nn import RGCNConv


class RGCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations,
                 num_layers=2, num_bases=30, dropout=0.5, batch_norm=True,
                 message_sens="source_to_target"):
        """
        Initialize the RGCN encoder with ReLU, BatchNorm, Dropout between each layer.

        Parameters:
        - in_channels: Input feature dimension
        - hidden_channels: Hidden dimension (int or list)
        - out_channels: Output embedding dimension
        - num_relations: Number of relation types in the graph
        - num_layers: Number of stacked RGCN layers
        - num_bases: Number of bases to reduce parameters
        - dropout: Dropout probability for regularization
        - batch_norm: Whether to use batch normalization
        - message_sens: Message passing direction
        """
        super(RGCNEncoder, self).__init__()

        self.num_layers = num_layers
        self.dropout_prob = dropout
        self.use_batch_norm = batch_norm
        self.num_relations = num_relations

        # Gérer hidden_channels comme int ou list
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels] * (num_layers - 1)

        # Dimensions de chaque couche
        layer_dims = [in_channels] + hidden_channels + [out_channels]

        # Create layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if batch_norm else None

        for i in range(num_layers):
            self.convs.append(
                RGCNConv(
                    layer_dims[i],
                    layer_dims[i + 1],
                    num_relations,
                    num_bases=num_bases,
                    flow=message_sens
                )
            )
            if batch_norm and i < num_layers - 1:
                self.bns.append(nn.BatchNorm1d(layer_dims[i + 1]))

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

    def forward(self, x, edge_index, edge_type, edge_weight=None):
        """
        Forward pass through the network.

        Parameters:
        - x: Node features [num_nodes, in_channels]
        - edge_index: Edge indices [2, num_edges]
        - edge_type: Edge types [num_edges]
        - edge_weight: Optional edge weights [num_edges] (not used by RGCNConv)

        Returns:
        - Node embeddings [num_nodes, out_channels]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)

            # Appliquer BN + ReLU + Dropout sauf pour la dernière couche
            if i < self.num_layers - 1:
                if self.bns is not None:
                    x = self.bns[i](x)
                x = self.relu(x)
                x = self.dropout(x)

        return x
