import json
import torch
import numpy as np
import pandas as pd
from dataclasses import is_dataclass, fields
from typing import TypeVar,List,Tuple,Dict,Any,Optional
from torch import Tensor
from torch_sparse import SparseTensor
from scipy.sparse import csgraph, csr_matrix
from torch import nn

DS = TypeVar('DS')

def load_dataclass_from_dict(cls : type[DS], data : dict) -> DS:

    new_data = dict()

    for field in fields(cls):

        if field.name in data:
            
            if is_dataclass(field.type):
                new_data[field.name] = load_dataclass_from_dict(field.type, data[field.name])
            else:
                new_data[field.name] = data[field.name]

    return cls(**new_data)

def load_json_file(filename : str) -> dict | list[dict]:
    
    with open(filename, 'r') as f:
        data = json.load(f)

    return data

def load_dataclass(cls : type[DS], filename : str) -> DS:
    json_ = load_json_file(filename)
    return load_dataclass_from_dict(cls, json_)

def entropy(probs : Tensor | np.ndarray) -> Tensor:

    is_np = False

    if isinstance(probs, np.ndarray):
        probs = torch.tensor(probs)
        is_np = True

    result = - torch.sum(probs * torch.log(probs), dim = -1)

    if is_np:
        result = result.numpy()

    return result

def load_data(
    filename : str,
    filter_ : bool = False,
) -> list[dict]:

    with open(filename, 'r') as f:
        data = json.load(f)
    
    if filter_:
        data = list(filter(lambda x : x['is_valid'], data))

    return data

def save_json(obj : object, filename : str) -> None:

    with open(filename, "w") as f:
        json.dump(obj, f, indent = 4)

def move_data_to_device(
    data : Tensor | List[Tensor] | Tuple[Tensor] | Dict[Any,Tensor] | SparseTensor,
    device : str | torch.device                  
) -> Tensor | List[Tensor] | Tuple[Tensor] | Dict[Any,Tensor] | SparseTensor:
        
    if isinstance(data, (Tensor,SparseTensor)):
        return data.to(device)
    
    if isinstance(data, list):
        return [move_data_to_device(item, device) for item in data]
    
    if isinstance(data, tuple):
        return tuple([move_data_to_device(item, device) for item in data])
    
    if isinstance(data, dict):
        return {
            key : move_data_to_device(item, device) 
            for key,item in data.items()
        }
    
    raise Exception(f"{type(data)} is not supported.")

def asdict_shallow(obj : DS) -> dict:

    return {
        field.name : getattr(obj, field.name)
        for field in fields(obj)
    }

def get_components(df : pd.DataFrame) -> pd.Series:

    entities = list(set(df['subject'].to_list() + df['object'].to_list()))
    head = df['subject'].apply(entities.index).to_numpy()
    tail = df['object'].apply(entities.index).to_numpy()

    N = len(entities)
    indices = np.stack([head, tail]).transpose()
    data = np.ones(len(df))

    adj = csr_matrix(
        (data, (indices[:, 0], indices[:, 1])),
        shape=(N, N),
        dtype=int
    )

    _, labels = csgraph.connected_components(adj, directed=False)

    labels = pd.Series(
        index=entities,
        data=labels
    )

    return labels

def calculate_model_size(model : nn.Module) -> float:

    param_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2

    return size_all_mb

def format_dict(data : dict, prefix : Optional[str] = None) -> str:

    keys = []

    for key, value in data.items():

        if isinstance(value, dict):
            
            if prefix is None:
                keys.append(format_dict(value, prefix=key))
            else:
                keys.append(format_dict(value, prefix=f'{prefix}.{key}'))

        else:

            if prefix is None:
                keys.append(f'{key} = {value}')
            else:
                keys.append(f'{prefix}.{key} = {value}')

    return ','.join(keys)