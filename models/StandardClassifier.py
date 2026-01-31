import torch
import torch.nn as nn


class StandardClassifier(nn.Module):
    def __init__(self, encoder, num_classes, dropout=0.0):
        """
        Standard classifier wrapper for different types of encoders.

        Parameters:
        - encoder: The encoder module (GCN, RGCN, GAT, TransGCN, etc.)
        - num_classes: Number of output classes
        - dropout: Dropout probability before classification
        """
        super(StandardClassifier, self).__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(encoder.out_channels, num_classes)

        # Detect encoder type to use correct forward signature
        self.encoder_type = self._detect_encoder_type()
        print(f"[StandardClassifier] Detected encoder type: {self.encoder_type}")

    def _detect_encoder_type(self):
        """
        Detect the encoder type based on its class name.

        Returns:
        - "transgcn": TransGCN variants (expects edge_attr, returns (x, edge_attr))
        - "rgcn": RGCN (expects edge_type)
        - "standard": GCN, GAT (expects edge_weight)
        """
        encoder_name = self.encoder.__class__.__name__

        if "TransGCN" in encoder_name:
            return "transgcn"
        elif "RGCN" in encoder_name:
            return "rgcn"
        elif "GAT" in encoder_name or "GCN" in encoder_name:
            return "standard"

        # Default to standard
        print(f"[WARNING] Unknown encoder type: {encoder_name}, defaulting to 'standard'")
        return "standard"

    def reset_parameters(self):
        """Reset parameters of encoder and classifier."""
        if hasattr(self.encoder, "reset_parameters"):
            self.encoder.reset_parameters()
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x, edge_index, edge_type=None, edge_weight=None, edge_attr=None):
        """
        Forward pass through encoder and classifier.

        Parameters:
        - x: Node features [num_nodes, in_channels]
        - edge_index: Edge indices [2, num_edges]
        - edge_type: Edge types/indices (for RGCN)
        - edge_weight: Edge weights (for GCN/GAT)
        - edge_attr: Edge attribute embeddings (for TransGCN)

        Returns:
        - Logits [num_nodes, num_classes]
        """
        h = self.encode(x, edge_index, edge_type, edge_weight, edge_attr)
        h = self.dropout(h)
        logits = self.classifier(h)
        return logits

    def encode(self, x, edge_index, edge_type=None, edge_weight=None, edge_attr=None):
        """
        Get node embeddings, handling different encoder signatures.

        Parameters:
        - x: Node features [num_nodes, in_channels]
        - edge_index: Edge indices [2, num_edges]
        - edge_type: Edge types/indices [num_edges] (for RGCN)
        - edge_weight: Edge weights [num_edges] (for GCN/GAT)
        - edge_attr: Edge attribute embeddings [num_edges, emb_dim] (for TransGCN)

        Returns:
        - Node embeddings [num_nodes, out_channels]
        """
        if self.encoder_type == "transgcn":
            # TransGCN expects (x, edge_index, edge_attr) and returns (x, edge_attr)
            # edge_attr contains relation embeddings [num_edges, emb_dim]
            out = self.encoder(x, edge_index, edge_attr=edge_attr)

        elif self.encoder_type == "rgcn":
            # RGCN expects (x, edge_index, edge_type)
            # edge_type contains integer indices [num_edges]
            out = self.encoder(x, edge_index, edge_type)

        else:  # standard (GCN, GAT)
            # GCN/GAT expects (x, edge_index, edge_weight)
            out = self.encoder(x, edge_index, edge_weight)

        # Handle tuple outputs (e.g., TransGCN returns (x, edge_attr))
        if isinstance(out, tuple):
            return out[0]
        return out

    def get_embeddings(self, x, edge_index, edge_type=None, edge_weight=None, edge_attr=None):
        """
        Get node embeddings (alias for encode).

        Parameters:
        - x: Node features [num_nodes, in_channels]
        - edge_index: Edge indices [2, num_edges]
        - edge_type: Edge types/indices (for RGCN)
        - edge_weight: Edge weights (for GCN/GAT)
        - edge_attr: Edge attribute embeddings (for TransGCN)

        Returns:
        - Node embeddings [num_nodes, out_channels]
        """
        return self.encode(x, edge_index, edge_type, edge_weight, edge_attr)