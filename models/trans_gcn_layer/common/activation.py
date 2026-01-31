import torch
from torch import nn
from typing import Literal

ActivationType = Literal['relu', 'leaky_relu', 'elu', 'selu', 'gelu', 'pelu']

class Activation(nn.Module):

    def __init__(self, 
        name : ActivationType = 'relu',
        **kwargs
    ) -> None:
        super().__init__()

        self.name = name

        self.activations = nn.ModuleDict({
            'relu' : nn.ReLU(),
            'leaky_relu' : nn.LeakyReLU(**kwargs),
            'elu' : nn.ELU(**kwargs),
            'selu' : nn.SELU(),
            'gelu' : nn.GELU(),
            'pelu' : nn.PReLU(**kwargs),
        })

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        activation_fn = self.activations[self.name]
        return activation_fn(x)