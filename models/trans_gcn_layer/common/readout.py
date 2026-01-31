import torch
from torch import Tensor

def mean_readout(x : Tensor, dim : int = 0) -> Tensor:
    return torch.mean(x, dim = dim)

def max_readout(x : Tensor, dim : int = 0) -> Tensor:
    return torch.max(x, dim = dim).values

def mean_max_readout(x : Tensor, dim : int = 0) -> Tensor:
    return torch.cat([mean_readout(x), max_readout(x)], dim = dim)

AGGREGATIONS = {
    'mean' : mean_readout,
    'max' : max_readout,
    'mean_max' : mean_max_readout
}

def readout(x : Tensor, type : str, dim = 0) -> Tensor:
    
    if type not in AGGREGATIONS:
        raise ValueError(f'Invalid aggregation type: {type}')
    
    readout_fn = AGGREGATIONS[type]

    return readout_fn(x, dim)