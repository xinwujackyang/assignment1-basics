"""Command-line setup for language-model training on tokenized .npy arrays."""

import argparse
from pathlib import Path
import random
import logging

import numpy as np
import numpy.typing as npt
import torch

from cs336_basics.model import (
    AdamW,
    TransformerLM,
    cross_entropy_loss,
    get_lr_cosine_schedule,
    gradient_clip,
    load_checkpoint,
    load_data,
    save_checkpoint,
)

def train(
    model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    train_data: npt.NDArray,
    *,
    batch_size: int,
    context_length: int,
    total_steps: int,
    device: str,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
    max_grad_norm: float = 1.0,
    start_step: int = 0,
    val_data: npt.NDArray | None = None,
    eval_interval: int = 100,
    eval_batches: int = 10,
    checkpoint_path: str | None = None,
    checkpoint_interval: int = 1000,
) -> int:
    """Run updates from start_step to total_steps; return completed updates."""
    model.train()
    for step in range(start_step, total_steps):
        lr = get_lr_cosine_schedule(step, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        inputs, targets = load_data(train_data, batch_size, context_length, device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = cross_entropy_loss(logits, targets)
        loss.backward()
        if max_grad_norm > 0:
            gradient_clip(model.parameters(), max_grad_norm)
        optimizer.step()

        if (step + 1) % eval_interval == 0:
            logging.info(f"Step {step + 1}: training loss = {loss.item():.4f}, learning rate = {lr:.6f}")
            if val_data is not None:
                model.eval()
                with torch.no_grad():
                    val_losses = []
                    for _ in range(eval_batches):
                        val_inputs, val_targets = load_data(val_data, batch_size, context_length, device)
                        val_logits = model(val_inputs)
                        val_loss = cross_entropy_loss(val_logits, val_targets)
                        val_losses.append(val_loss.item())
                    avg_val_loss = sum(val_losses) / len(val_losses)
                    logging.info(f"Step {step + 1}: validation loss = {avg_val_loss:.4f}")
                model.train()

        if checkpoint_path is not None and (step + 1) % checkpoint_interval == 0:
            save_path = Path(checkpoint_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=step + 1,
                out=save_path,
            )
            logging.info(f"Checkpoint saved at step {step + 1} to {save_path}")

    if total_steps % checkpoint_interval != 0 and checkpoint_path is not None:
        save_path = Path(checkpoint_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            iteration=total_steps,
            out=save_path,
        )
        logging.info(f"Final checkpoint saved at step {total_steps} to {save_path}")

    return total_steps





def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=352)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--total-steps", type=int, default=1000)
    parser.add_argument("--max-learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--cosine-cycle-iters", type=int)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/latest.pt"))
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cpu", help="For example: cpu, mps, cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.info("Starting training...")
    for name in ("vocab_size", "context_length", "d_model", "num_layers", "num_heads",
                 "d_ff", "batch_size", "total_steps", "eval_interval", "eval_batches",
                 "checkpoint_interval"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.d_model % args.num_heads or (args.d_model // args.num_heads) % 2:
        parser.error("d_model must be divisible by num_heads with an even head dimension")
    cycle_iters = args.cosine_cycle_iters if args.cosine_cycle_iters is not None else args.total_steps
    if not 0 <= args.warmup_iters < cycle_iters:
        parser.error("require 0 <= warmup_iters < cosine_cycle_iters")
    if not 0 <= args.min_learning_rate <= args.max_learning_rate:
        parser.error("require 0 <= min_learning_rate <= max_learning_rate")
    if args.max_grad_norm < 0:
        parser.error("max_grad_norm must be non-negative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    def open_tokens(path: Path) -> npt.NDArray:
        # Read metadata without loading the whole token dataset into RAM.
        data = np.load(path, mmap_mode="r", allow_pickle=False)
        if not isinstance(data, np.ndarray) or data.ndim != 1:
            raise ValueError(f"{path}: expected a one-dimensional .npy array")
        if not np.issubdtype(data.dtype, np.integer):
            raise TypeError(f"{path}: token IDs must have an integer dtype")
        if len(data) <= args.context_length:
            raise ValueError(f"{path}: need at least context_length + 1 tokens")
        return data

    train_data = open_tokens(args.train_data)
    val_data = open_tokens(args.val_data) if args.val_data is not None else None
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
        dtype=torch.float32,
    )
    # Construct the optimizer only after model parameters are on their device.
    optimizer = AdamW(
        model.parameters(), lr=args.max_learning_rate, betas=tuple(args.betas),
        eps=args.eps, weight_decay=args.weight_decay,
    )
    start_step = 0
    if args.resume is not None:
        # Restores optimizer hyperparameters too. Use matching architecture flags.
        start_step = load_checkpoint(args.resume, model, optimizer)
    if not 0 <= start_step <= args.total_steps:
        parser.error("checkpoint iteration must be between 0 and total_steps")
    if start_step == args.total_steps:
        logging.info(f"Already completed {start_step} updates; no training needed.")
        return

    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.train()
    completed = train(
        model, optimizer, train_data,
        batch_size=args.batch_size, context_length=args.context_length,
        total_steps=args.total_steps, device=args.device,
        max_learning_rate=args.max_learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_iters=args.warmup_iters, cosine_cycle_iters=cycle_iters,
        max_grad_norm=args.max_grad_norm, start_step=start_step,
        val_data=val_data, eval_interval=args.eval_interval,
        eval_batches=args.eval_batches, checkpoint_path=str(args.checkpoint_path),
        checkpoint_interval=args.checkpoint_interval,
    )
    print(f"Completed {completed} updates.")


if __name__ == "__main__":
    main()
