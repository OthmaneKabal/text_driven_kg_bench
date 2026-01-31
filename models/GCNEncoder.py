# from torch import nn
# from torch_geometric.data import Data
# from torch_geometric.nn import GCNConv
#
#
# class GCNEncoder(nn.Module):
#     def __init__(self, data: Data, out_channels, num_layers=2):
#         """
#         Initialise l'encodeur GCN avec ReLU entre chaque couche.
#
#         Paramètres:
#         - data : objet de type Data pour extraire les caractéristiques d'entrée
#         - out_channels : liste contenant le nombre de caractéristiques de sortie pour chaque couche
#         - num_layers : nombre de couches GCN à empiler (doit correspondre à la longueur de out_channels)
#         """
#         super(GCNEncoder, self).__init__()
#
#         # Assurer que le nombre de couches correspond à la taille de out_channels
#         assert len(out_channels) == num_layers, "La longueur de out_channels doit être égale à num_layers"
#         self.out_channels = out_channels[1]
#
#         # Extraire les dimensions d'entrée depuis l'objet data
#         in_channels = data.x.shape[1]
#
#         # Créer une liste de couches GCN avec des tailles de sortie différentes
#         self.convs = nn.ModuleList()
#         for i in range(num_layers):
#             input_dim = in_channels if i == 0 else out_channels[i - 1]
#             # Ajouter une couche GCN
#             self.convs.append(GCNConv(input_dim, out_channels[i]))
#
#         self.relu = nn.ReLU()
#
#     def reset_parameters(self):
#         """Réinitialise les paramètres des couches de l'encodeur."""
#         for conv in self.convs:
#             conv.reset_parameters()
#
#     def forward(self, data: Data):
#         """
#         Passe en avant dans le réseau en prenant un objet Data comme entrée.
#
#         Paramètres:
#         - data : objet de type Data contenant x (caractéristiques des nœuds) et edge_index (indices des arêtes)
#
#         Retourne:
#         - Embeddings des nœuds après passage dans l'encodeur GCN
#         """
#         # Extraire les attributs de l'objet Data
#         x, edge_index = data.x, data.edge_index
#
#         # Appliquer chaque couche GCN avec une activation ReLU entre chaque couche
#         for conv in self.convs:
#             x = conv(x, edge_index)
#             x = self.relu(x)
#
#         return x
#




from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


# class GCNEncoder(nn.Module):
#     def __init__(self, data: Data, out_channels, num_layers=2, dropout=0.5, message_sens="source_to_target"):
#         """
#         Initialize the GCN encoder with ReLU, BatchNorm, and Dropout between each layer.
#
#         Parameters:
#         - data: Data object to extract input features.
#         - out_channels: List of output feature dimensions for each layer.
#         - num_layers: Number of stacked GCN layers (must match the length of out_channels).
#         - dropout: Dropout probability for regularization.
#         """
#
#         super(GCNEncoder, self).__init__()
#
#         # Ensure the number of layers matches the size of out_channels
#         assert len(out_channels) == num_layers, "The length of out_channels must equal num_layers"
#         self.out_channels = out_channels[-1]
#
#         # Extract input dimensions from the data object
#         in_channels = data.x.shape[1]
#
#         # Create a list of GCN layers with varying output sizes
#         self.convs = nn.ModuleList()
#         self.bns = nn.ModuleList()  # Batch normalization layers
#         for i in range(num_layers):
#             input_dim = in_channels if i == 0 else out_channels[i - 1]
#             self.convs.append(GCNConv(input_dim, out_channels[i], flow=message_sens))
#             self.bns.append(nn.BatchNorm1d(out_channels[i]))
#
#         # Dropout layer
#         self.dropout = nn.Dropout(p=dropout)
#
#         # Final linear layer to produce the embeddings
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
#         - data: Data object containing x (node features) and edge_index (edge indices).
#
#         Returns:
#         - Node embeddings after passing through the GCN encoder.
#         """
#         # Extract attributes from the Data object
#         x, edge_index = data.x, data.edge_index
#
#         # Apply each GCN layer with BatchNorm, ReLU, and Dropout in between
#         for conv, bn in zip(self.convs, self.bns):
#             x = conv(x, edge_index)
#             x = bn(x)
#             x = self.relu(x)
#             x = self.dropout(x)
#
#         # Apply the final linear layer to produce embeddings
#         x = self.final_layer(x)
#
#         return x
class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, batch_norm=True, message_sens="source_to_target"):
        """
        Initialize the GCN encoder with ReLU, BatchNorm, and Dropout between each layer.

        Parameters:
        - in_channels: Input feature dimension (nombre de features des nœuds)
        - hidden_channels: Hidden dimension (peut être int ou list)
        - out_channels: Output embedding dimension
        - num_layers: Number of stacked GCN layers
        - dropout: Dropout probability for regularization
        - batch_norm: Whether to use batch normalization
        - message_sens: Message passing direction
        """
        super(GCNEncoder, self).__init__()

        self.num_layers = num_layers
        self.dropout_prob = dropout
        self.use_batch_norm = batch_norm

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
                GCNConv(layer_dims[i], layer_dims[i + 1], flow=message_sens)
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
            x = conv(x, edge_index, edge_weight)

            # Appliquer BN + ReLU + Dropout sauf pour la dernière couche
            if i < self.num_layers - 1:
                if self.bns is not None:
                    x = self.bns[i](x)
                x = self.relu(x)
                x = self.dropout(x)

        return x