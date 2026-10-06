import torch
from torch import nn
from jaxtyping import Bool, Float, Int

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weights: Float[torch.Tensor, "out_features in_features"] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
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

            std = 2.0 / (in_features + out_features)
            std = std ** 0.5

            nn.init.trunc_normal_(
                self.weight,
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
            )


    def forward(self, x: Float[torch.Tensor, "... in_features"]) -> Float[torch.Tensor, "... out_features"]:
        return x @ self.weight.transpose(-1,-2)

class Embedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        weights: Float[torch.Tensor, "vocab_size d_model"] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if weights is not None:
            if weights.shape != (vocab_size, d_model):
                raise ValueError(
                    f"Expected weights shape {(vocab_size, d_model)}, "
                    f"got {tuple(weights.shape)}"
                )
            else:
                self.weight = weights
        else:
            self.weight = nn.Parameter(
                torch.empty(
                    (vocab_size, d_model),
                    device=device,
                    dtype=dtype,
                )
            )

            std = 1.0
            nn.init.trunc_normal_(
                self.weight,
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
            )

    def forward(self, x: Int[torch.Tensor, "..."]) -> Float[torch.Tensor, "... d_model"]:
        if x.dtype not in (torch.int32, torch.int64):
            raise TypeError("Token IDs must be int32 or int64.")

        if x.numel() > 0:
            if x.min().item() < 0 or x.max().item() >= self.weight.shape[0]:
                raise ValueError("Token IDs must be in [0, vocab_size).")

        return self.weight[x]

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-8,
        weights: Float[torch.Tensor, "d_model"] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        if weights is not None:
            if weights.shape != (d_model,):
                raise ValueError(
                    f"Expected weights shape {(d_model,)}, "
                    f"got {tuple(weights.shape)}"
                )
            else:
                self.scale = nn.Parameter(weights)
        else:
            self.scale = nn.Parameter(torch.ones(d_model))


    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        input_type = x.dtype
        x = x.to(torch.float32)
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.scale * (x / norm).to(input_type)

def silu(x: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        w1_weight: Float[torch.Tensor, "d_ff d_model"] | None = None,
        w2_weight: Float[torch.Tensor, "d_model d_ff"] | None = None,
        w3_weight: Float[torch.Tensor, "d_ff d_model"] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.linear1 = Linear(d_model, d_ff, weights=w1_weight, device=device, dtype=dtype)
        self.linear2 = Linear(d_ff, d_model, weights=w2_weight, device=device, dtype=dtype)
        self.linear3 = Linear(d_model, d_ff, weights=w3_weight, device=device, dtype=dtype)

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        linear1 = silu(self.linear1(x))
        linear3 = self.linear3(x)
        return self.linear2(linear1 * linear3)

def softmax(x: Float[torch.Tensor, "..."], dim: int) -> Float[torch.Tensor, "..."]:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)

def scaled_dot_product_attention(
    Q: Float[torch.Tensor, "... queries d_k"],
    K: Float[torch.Tensor, "... keys d_k"],
    V: Float[torch.Tensor, "... keys d_v"],
    mask: Bool[torch.Tensor, "..."] | None = None,
) -> Float[torch.Tensor, "... queries d_v"]:
    # mask must broadcast to (..., queries, keys); True allows attention.
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2,-1)) / d_k**0.5

    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))

    scores = softmax(scores, dim = -1) @ V
    return scores
