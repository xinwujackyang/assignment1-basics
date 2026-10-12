from collections.abc import Iterable
import math
import os
from typing import BinaryIO
import numpy as np
import numpy.typing as npt
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
            self.weight = nn.Parameter(weights.to(device=device, dtype=dtype))
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
                self.weight = nn.Parameter(weights.to(device=device, dtype=dtype))
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
        eps: float = 1e-5,
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
                self.scale = nn.Parameter(weights.to(device=device, dtype=dtype))
        else:
            self.scale = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))


    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        input_type = x.dtype
        x = x.to(torch.float32)
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (self.scale * (x / norm)).to(input_type)

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

def softmax(x: Float[torch.Tensor, "..."], dim: int) -> Float[torch.Tensor, ""]:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)

def cross_entropy_loss(logits: Float[torch.Tensor, "... vocab_size"], targets: Int[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
    if logits.shape[:-1] != targets.shape:
        raise ValueError(
            f"Logits shape {tuple(logits.shape)} and targets shape {tuple(targets.shape)} "
            "are incompatible."
        )
    if logits.shape[-1] < 1:
        raise ValueError("Logits last dimension must be > 1.")
    if targets.dtype not in (torch.int32, torch.int64):
        raise TypeError("Targets must be int32 or int64.")
    if targets.numel() > 0:
        if targets.min().item() < 0 or targets.max().item() >= logits.shape[-1]:
            raise ValueError("Targets must be in [0, vocab_size).")

    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1).mean()

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
        q_proj_weight: Float[torch.Tensor, " d_model d_model"] | None = None,
        k_proj_weight: Float[torch.Tensor, " d_model d_model"] | None = None,
        v_proj_weight: Float[torch.Tensor, " d_model d_model"] | None = None,
        o_proj_weight: Float[torch.Tensor, " d_model d_model"] | None = None,
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
        




class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        q_proj_weight: Float[torch.Tensor, " d_model d_model"] | None = None,
        k_proj_weight: Float[torch.Tensor, " d_model d_model"] | None = None,
        v_proj_weight: Float[torch.Tensor, " d_model d_model"] | None = None,
        o_proj_weight: Float[torch.Tensor, " d_model d_model"] | None = None,
        w1_weight: Float[torch.Tensor, "d_ff d_model"] | None = None,
        w2_weight: Float[torch.Tensor, "d_model d_ff"] | None = None,
        w3_weight: Float[torch.Tensor, "d_ff d_model"] | None = None,
        rmsnorm1_weight: Float[torch.Tensor, "d_model"] | None = None,
        rmsnorm2_weight: Float[torch.Tensor, "d_model"] | None = None,
        max_seq_len: int | None = None,
        theta: float = 10000.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            o_proj_weight=o_proj_weight,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype
        )
        self.norm1 = RMSNorm(d_model=d_model, weights=rmsnorm1_weight, device=device, dtype=dtype)
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            w1_weight=w1_weight,
            w2_weight=w2_weight,
            w3_weight=w3_weight,
            device=device,
            dtype=dtype
        )
        self.norm2 = RMSNorm(d_model=d_model, weights=rmsnorm2_weight, device=device, dtype=dtype)

    def forward(self, x: Float[torch.Tensor, "... seq_len d_model"], token_positions: Int[torch.Tensor, "... seq_len"] | None = None) -> Float[torch.Tensor, "... seq_len d_model"]:
        x = x + self.attention(self.norm1(x), token_positions=token_positions)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self,
                vocab_size: int,
                context_length: int,
                d_model: int,
                num_layers: int,
                num_heads: int,
                d_ff: int,
                rope_theta: float,
                weights: dict[str, torch.Tensor] | None = None,
                device: torch.device | str | None = None,
                dtype: torch.dtype | None = None
                 ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        # Missing keys in supplied weights must not silently initialize randomly.
        def get_weight(key: str) -> torch.Tensor | None:
            return None if weights is None else weights[key]

        self.embedding = Embedding(vocab_size, d_model, weights=get_weight("token_embeddings.weight"), device=device, dtype=dtype)

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                q_proj_weight=get_weight(f"layers.{i}.attn.q_proj.weight"),
                k_proj_weight=get_weight(f"layers.{i}.attn.k_proj.weight"),
                v_proj_weight=get_weight(f"layers.{i}.attn.v_proj.weight"),
                o_proj_weight=get_weight(f"layers.{i}.attn.output_proj.weight"),
                w1_weight=get_weight(f"layers.{i}.ffn.w1.weight"),
                w2_weight=get_weight(f"layers.{i}.ffn.w2.weight"),
                w3_weight=get_weight(f"layers.{i}.ffn.w3.weight"),
                rmsnorm1_weight=get_weight(f"layers.{i}.ln1.weight"),
                rmsnorm2_weight=get_weight(f"layers.{i}.ln2.weight"),
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype
            ) for i in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model=d_model, weights=get_weight("ln_final.weight"), device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, weights=get_weight("lm_head.weight"), device=device, dtype=dtype)

    def forward(self, x: Int[torch.Tensor, "... seq_len"]) -> Float[torch.Tensor, "... seq_len vocab_size"]:
        if x.shape[-1] > self.context_length:
            raise ValueError(f"Input sequence length {x.shape[-1]} exceeds context length {self.context_length}.")

        x = self.embedding(x)
        token_positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.int64)

        for layer in self.layers:
            x = layer(x, token_positions=token_positions)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 1
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                

                with torch.no_grad():
                    p.add_(p, alpha=-group["lr"] * group["weight_decay"])

                lr_t = group["lr"] * (1 - group["betas"][1] ** state["step"]) ** 0.5 / (1 - group["betas"][0] ** state["step"])

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                state['step'] = state.get("step", 0) + 1
                beta1, beta2 = group["betas"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                with torch.no_grad():
                    p.add_(exp_avg / (exp_avg_sq.sqrt() + group["eps"]), alpha=-lr_t)


        return loss

PI = math.pi
    
def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if max_learning_rate < 0:
        raise ValueError(f"max_learning_rate must be non-negative, got {max_learning_rate}.")
    if min_learning_rate < 0:
        raise ValueError(f"min_learning_rate must be non-negative, got {min_learning_rate}.")
    if warmup_iters < 0:
        raise ValueError(f"warmup_iters must be non-negative, got {warmup_iters}.")
    if cosine_cycle_iters < 0:
        raise ValueError(f"cosine_cycle_iters must be non-negative, got {cosine_cycle_iters}.")
    
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif it < cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + math.cos(progress * PI))
    else:
        return min_learning_rate


def gradient_clip(
    parameters: Iterable[torch.nn.Parameter], 
    max_l2_norm: float,
    eps: float = 1e-6    
) -> None:
    if max_l2_norm < 0:
        raise ValueError(f"max_l2_norm must be non-negative, got {max_l2_norm}.")

    if eps < 0:
        raise ValueError(f"eps must be non-negative, got {eps}.")

    params = []
    for p in parameters:
        if p.grad is not None:
            params.append(p)

    if not params:
        return

    total_l2_norm = torch.sqrt(sum([
    torch.sum(
        (p.grad.float() if p.grad.dtype in (torch.float16, torch.bfloat16)
         else p.grad) ** 2
    )
    for p in params
    ]))

    with torch.no_grad():
        if total_l2_norm > max_l2_norm:
            clip_coef = max_l2_norm / (total_l2_norm + eps)
            for p in params:
                    p.grad.mul_(clip_coef)

def load_data(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[Int[torch.Tensor, "batch_size context_length"], Int[torch.Tensor, "batch_size context_length"]]:
    """Sample windows from a 1D integer token array, including a memory map.

    batch_size is the number of independently sampled windows; context_length
    is the number of tokens per window. Both int64 outputs are moved to device.
    Targets contain the next token for each corresponding input token.
    """
    if dataset.ndim != 1:
        raise ValueError("dataset must be a one-dimensional array.")
    if not np.issubdtype(dataset.dtype, np.integer):
        raise TypeError("dataset must contain integer token IDs.")
    if batch_size <= 0 or context_length <= 0:
        raise ValueError("batch_size and context_length must be positive.")
    if len(dataset) <= context_length:
        raise ValueError("dataset must contain at least context_length + 1 tokens.")

    # The exclusive upper bound reserves one extra token for the final target.
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    # Copy only sampled windows, not the full dataset, to support large memmaps.
    windows = np.stack([
        dataset[start:start + context_length + 1] for start in starts
    ]).astype(np.int64, copy=False)
    inputs = torch.from_numpy(windows[:, :-1].copy()).to(device)
    targets = torch.from_numpy(windows[:, 1:].copy()).to(device)
    return inputs, targets


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO,
) -> None:
    """Save model state, optimizer state, and completed iteration count.

    out is a file path or a writable binary stream. Model architecture is not
    serialized; loading requires an already constructed compatible model.
    """
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Restore existing model and optimizer objects; return completed iterations.

    src is a file path or a readable binary stream positioned at the checkpoint.
    The optimizer must reference the supplied model's parameters. Load tensors
    on CPU first so a checkpoint does not require its original saving device.
    """
    checkpoint = torch.load(src, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
