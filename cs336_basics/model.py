import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, weights = None, device = None, dtype = None):
        super().__init__()
        # self.weight = torch.nn.Parameter(weight_tensor)
        if weights is not None:
            self.weight = nn.Parameter(weights)
        else:
            self.weight = nn.Parameter(
                torch.empty(
                    (out_features, in_features),
                    device=device,
                    dtype=dtype,
                )
            )

            std = 0.1
            nn.init.trunc_normal_(
                self.weight,
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
            )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.transpose(-1,-2)

class Embedding(nn.Module):
    def __init__(
                self, 
                num_embeddings: int, 
                 embedding_dim: int, 
                 vocab_size: int, 
                 weights = None, 
                 device = None, 
                 dtype = None
                 ):
        super().__init__()
        if weights is not None:
            self.weight = nn.Parameter(weights)
        else:
            self.weight = nn.Parameter(
                torch.empty(
                    (num_embeddings, embedding_dim),
                    device=device,
                    dtype=dtype,
                )
            )

            std = 0.1
            nn.init.trunc_normal_(
                self.weight,
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]
