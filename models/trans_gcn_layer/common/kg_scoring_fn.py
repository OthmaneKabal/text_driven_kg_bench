import torch
from torch import Tensor, nn
from typing import Literal

KGScoringFn = Literal['TransE', 'RotatE']

class KGScoring(nn.Module):

    def translate(self,
        x : Tensor,
        rel : Tensor
    ) -> Tensor:
        raise NotImplementedError
    
    def inv_rel(self, rel : Tensor) -> Tensor:
        NotImplementedError
    
    def inv_translate(self,
        x : Tensor,
        rel : Tensor
    ) -> Tensor:
        return self.translate(x, self.inv_rel(rel))
    
    def score(self,
        x : Tensor,
        y : Tensor
    ) -> Tensor:
        raise NotImplementedError
    
    def forward(self,
        x : Tensor,
        y : Tensor,
        rel : Tensor
    ) -> Tensor:
        x = self.translate(x, rel)
        return self.score(x, y)
    

class TransE(KGScoring):

    def __init__(self,
        p_norm : float = 2.0             
    ) -> None:
        super().__init__()
        self.p_norm = p_norm

    def translate(self,
        x : Tensor,
        rel : Tensor
    ) -> Tensor:
        return x + rel

    def inv_rel(self, rel : Tensor) -> Tensor:
        return - rel

    def score(self,
        x : Tensor,
        y : Tensor
    ) -> Tensor:
        return torch.norm(x - y, p = self.p_norm, dim = -1)
    
class RotatE(KGScoring):

    def __init__(self,
        p_norm : float = 2.0
    ) -> None:
        super().__init__()
        self.p_norm = p_norm
        

    def translate(self,
        x : Tensor,
        rel : Tensor
    ) -> Tensor:
        
        x_im, x_re = torch.chunk(x, 2, dim = -1)
        rel_im, rel_re = torch.chunk(rel, 2, dim = -1)
        
        y_re = x_re * rel_re - x_im * rel_im
        y_im = x_re * rel_im + x_im * rel_re

        return torch.cat([y_im, y_re], dim = -1)
    
    def inv_rel(self, rel : Tensor) -> Tensor:
        rel_im, rel_re = torch.chunk(rel, 2, dim = -1)
        return torch.cat([-rel_im, rel_re], dim = -1)
    
    def score(self,
        x : Tensor,
        y : Tensor
    ) -> Tensor:
        return torch.norm(x - y, p = self.p_norm, dim = -1)
    
def create_scoring_fn(
    name : KGScoringFn,
    **kwargs
) -> KGScoring:
    
    if name == 'TransE':
        return TransE(**kwargs)
    elif name == 'RotatE':
        return RotatE(**kwargs)
    else:
        raise ValueError(f"Unknown scoring function {name}")
    
