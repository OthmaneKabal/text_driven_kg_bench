import torch
import random
import numpy as np
from torch import Tensor,nn
from torch_scatter import scatter
from typing import Dict

def seed_everything(seed : int) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def edge_score(
    x : Tensor | Dict[str, Tensor],
    x_hat : Tensor | Dict[str, Tensor],
    p : Tensor | Dict[str, Tensor],
    eps : float = 0.05
) -> Tensor:
    
    if isinstance(x, Tensor):
    
        return torch.pow(x - x_hat, 2).mean(dim = -1) - eps * p.log2()
    
    else:

        scores = []

        for key, x_i, x_hat_i, p_i in zip(x.keys(), x.values(), x_hat.values(), p.values()):
            scores.append(
                edge_score(x_i, x_hat_i, p_i)
            )

        return torch.cat(scores, dim=0)

def node_score(
    x : Tensor,
    x_hat : Tensor,
    edge_index : Dict[str, Tensor],
    edge_score : Tensor,
    eps : float = 0.05
) -> Tensor:
    
    if isinstance(edge_index, dict):
        edge_index = torch.cat(list(edge_index.values()), dim = -1)
    
    edge_score = scatter(
        src = edge_score,
        index = edge_index[0],
        dim = 0,
        dim_size = x.size(0),
        reduce = "mean"
    ) 

    return torch.pow(x - x_hat, 2).mean(dim = -1) + eps * edge_score

def reset_parameters(model : nn.Module) -> None:
    
    for module in model.modules():

        if module is model:
            continue

        if hasattr(module,'reset_parameters'):
            module.reset_parameters()
        else:
            reset_parameters(module)