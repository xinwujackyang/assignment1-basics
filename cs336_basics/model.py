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


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        d_k: int,
        max_seq_len: int,
        theta: float = 10000.0,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if d_k <= 0 or d_k % 2:
            raise ValueError("d_k must be a positive even integer.")
        if max_seq_len <= 0 or theta <= 0:
            raise ValueError("max_seq_len and theta must be positive.")
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Each adjacent component pair has its own rotation frequency.
        inv_freq = theta ** (-torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = torch.einsum("i,j->ij", positions, inv_freq)
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_k"],
        token_positions: Int[torch.Tensor, "... seq_len"],
    ) -> Float[torch.Tensor, "... seq_len d_k"]:
        """Rotate adjacent components using explicit token positions.

        Position dimensions must broadcast to x without its final feature axis.
        For x shaped (B, H, T), plus its feature axis, use positions shaped
        (T,), (B, 1, T), or (B, H, T).
        """
        if x.shape[-1] != self.d_k:
            raise ValueError(f"Expected final dimension {self.d_k}, got {x.shape[-1]}.")
        if token_positions.dtype not in (torch.int32, torch.int64):
            raise TypeError("Token positions must be int32 or int64.")
        if token_positions.numel() and (
            token_positions.min().item() < 0 or token_positions.max().item() >= self.max_seq_len
        ):
            raise ValueError("Token positions must be in [0, max_seq_len).")

        cos_pos = self.cos_cached[token_positions]
        sin_pos = self.sin_cached[token_positions]
        # Use at least float32 arithmetic; preserve float64 inputs.
        working = x.to(torch.float32) if x.dtype in (torch.float16, torch.bfloat16) else x
        even, odd = working[..., ::2], working[..., 1::2]
        rotated_even = even * cos_pos - odd * sin_pos
        rotated_odd = even * sin_pos + odd * cos_pos
        # (..., T, d_k/2, 2) -> (..., T, d_k), preserving adjacent pairs.
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2).to(x.dtype)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        q_proj_weight: Float[torch.Tensor, " d_model d_model"],
        k_proj_weight: Float[torch.Tensor, " d_model d_model"],
        v_proj_weight: Float[torch.Tensor, " d_model d_model"],
        o_proj_weight: Float[torch.Tensor, " d_model d_model"],
        max_seq_len: int | None = None,
        theta: float = 10000.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be a positive integer.")


        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.rope = RotaryPositionalEmbedding(self.d_k, max_seq_len, theta, device=device) if (max_seq_len is not None) else None

        self.Q = Linear(d_model, d_model, weights=q_proj_weight, device=device, dtype=dtype)
        self.K = Linear(d_model, d_model, weights=k_proj_weight, device=device, dtype=dtype)
        self.V = Linear(d_model, d_model, weights=v_proj_weight, device=device, dtype=dtype)
        self.O = Linear(d_model, d_model, weights=o_proj_weight, device=device, dtype=dtype)

    def forward(self, x: Float[torch.Tensor, "... seq_len d_model"], token_positions: Int[torch.Tensor, "... seq_len"] = None) -> Float[torch.Tensor, "... seq_len d_model"]:


        q = self.Q(x).reshape(*x.shape[:-1], self.num_heads, self.d_k).transpose(-2, -3)
        k = self.K(x).reshape(*x.shape[:-1], self.num_heads, self.d_k).transpose(-2, -3)
        v = self.V(x).reshape(*x.shape[:-1], self.num_heads, self.d_k).transpose(-2, -3)

        if self.rope is not None:
            seq_len = x.shape[-2]
            token_positions = token_positions if token_positions is not None else torch.arange(seq_len, device=x.device, dtype=torch.int64)

            positions = token_positions.unsqueeze(-2)
            q = self.rope(q, positions)
            k = self.rope(k, positions)

        seq_len = x.shape[-2]
        casual_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool))

        attn_output = scaled_dot_product_attention(q, k, v, mask=casual_mask)
        attn_output = attn_output.transpose(-2, -3).reshape(*x.shape[:-1], -1)

        return self.O(attn_output)
        




