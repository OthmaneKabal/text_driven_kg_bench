import sys

import torch
import torch_scatter

sys.path.append('..')
from torch import nn, Tensor
from torch_geometric import nn as gnn, utils as gnn_utils
from models.trans_gcn_layer.common import Activation, ActivationType, create_scoring_fn, KGScoringFn
from typing import Literal, Optional
from dataclasses import dataclass


@dataclass
class BaseTransGNNConfig:
    kg_score_fn : KGScoringFn = 'TransE'
    variant : Literal['conv', 'attn'] = 'attn'
    bias : bool = False
    activation : Optional[ActivationType] = 'relu'
    eps : float = 1e-6

@dataclass
class TransGNNConfig(BaseTransGNNConfig):
    in_channels : int = 384
    out_channels : int = 384
    use_edges_info : bool = True

class TransGNN(gnn.MessagePassing):

    def __init__(self, 
        in_channels : int,
        out_channels : int,
        use_edges_info : bool = True,
        kg_score_fn : KGScoringFn = 'TransE',
        variant : Literal['conv', 'attn'] = 'attn',
        bias : bool = False,
        activation : Optional[ActivationType] = 'relu',
        eps : float = 1e-6,
        **kwargs     
    ) -> None:
        super().__init__(**kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_edges_info = use_edges_info
        self.kg_score_fn = kg_score_fn
        self.variant = variant
        self.bias = bias
        self.activation = activation
        self.eps = eps

        self.score_fn = create_scoring_fn(kg_score_fn)
        self.act = Activation(activation) if activation is not None else nn.Identity()

        self.W_node = nn.Linear(2 * in_channels, out_channels, bias = bias) 

        in_dim = 2 * in_channels 
        
        if use_edges_info:
            in_dim += in_channels

        self.W_edge = nn.Linear(in_dim, out_channels, bias = bias) 

        if variant == 'attn':

            m = 3 if use_edges_info else 2

            self.attn = nn.Sequential(
                nn.Linear(m * in_channels, out_channels),
                nn.LeakyReLU(negative_slope = 0.2),
                nn.Linear(out_channels, 1),
            )

    @staticmethod
    def from_config(config : TransGNNConfig | BaseTransGNNConfig, **kwargs) -> 'TransGNN':

        config = {
            **utils.asdict_shallow(config),
            **kwargs
        }

        return TransGNN(**config)
        
    def forward(self, 
        x : Tensor,
        edge_index : Tensor,
        edge_attr : Optional[Tensor] = None,                          
    ) -> tuple[Tensor, Tensor]:
        
        row, col = edge_index
        og_x = x

        ### Get head, tail and relation
        h, t = x[row], x[col]
        r = edge_attr
        attn_weights = None

        ### If variant is attention then calculate attention weights
        if self.variant == 'attn':
            
            if self.use_edges_info:
                r_inv = self.score_fn.inv_rel(r)
                attn_weights1 = self.attn(torch.cat([h, t, r], dim = -1))
                attn_weights2 = self.attn(torch.cat([t, h, r_inv], dim = -1))
            else:
                attn_weights1 = self.attn(torch.cat([h, t], dim = -1))
                attn_weights2 = self.attn(torch.cat([t, h], dim = -1))

            attn_weights = gnn_utils.softmax(
                src = torch.cat([attn_weights1,attn_weights2],dim=0),
                index = torch.cat([row, col]),
                num_nodes = x.size(0)
            )
        
        ### Get the messages
        messages = self.message_and_aggregate(x, edge_index, edge_attr, attn_weights)

        ### Combine messages with nodes represntations
        x = torch.cat([x, messages], dim = -1)

        ### Linear Transformation
        x = self.act(self.W_node(x)) # W c'est appliqué juste sur les messages

        ### Edge Update
        if self.use_edges_info:
            edge_attr = torch.cat([og_x[row], og_x[col], edge_attr], dim = -1)
        else:
            edge_attr = torch.cat([og_x[row], og_x[col]], dim = -1)

        edge_attr = self.act(self.W_edge(edge_attr))

        return x, edge_attr
        
    def message_and_aggregate(self, 
        x : Tensor,
        edge_index : Tensor,
        edge_attr : Optional[Tensor] = None,       
        attn_weights : Optional[Tensor] = None                   
    ) -> Tensor:
        
        row, col = edge_index

        ### Get head tail and relation
        h, t = x[row], x[col]
        r = edge_attr

        out = self.score_fn.translate(h, r) if self.use_edges_info else h
        in_ = self.score_fn.inv_translate(t, r) if self.use_edges_info else t

        ### Messages
        src = torch.cat([in_,out], dim = 0)

        if attn_weights is not None:
            src = src * attn_weights

        messages = torch_scatter.scatter(
            src = src,
            index = torch.cat([row, col], dim=0),
            dim = 0,
            dim_size = x.size(0),
            reduce = ('mean' if attn_weights is None else 'sum')
        )

        return messages
#
# if __name__ == '__main__':
#
#     N, E, X_e, X_d = 10, 20, 8, 8
#
#     x = torch.randn(N, X_d)
#     edge_index = torch.randint(0, N, (2, E))
#     edge_attr = torch.randn(E, X_e)
#
#     model = TransGNN(
#         in_channels = X_d,
#         out_channels = X_d,
#         kg_score_fn = 'TransE',
#         variant = 'conv',
#         bias = True,
#         use_edges_info=True,
#     activation = 'relu'
#     )
#
#     x, edge_attr = model.forward(x, edge_index, edge_attr)
#
#     print(x.size(), edge_attr.size())