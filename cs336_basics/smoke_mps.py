"""Short MPS training check using the handout's low-resource architecture."""

import argparse
import json
import logging
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from cs336_basics.model import AdamW, TransformerLM
from cs336_basics.train import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/mps-smoke"))
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; this smoke test does not fall back to CPU.")
    if args.steps < 3 or args.batch_size <= 0:
        parser.error("Use at least 3 steps and a positive batch size")
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, handlers=[
        logging.StreamHandler(), logging.FileHandler(args.output / "run.log")])
    config = dict(vocab_size=10000, context_length=256, d_model=512,
                  num_layers=4, num_heads=16, d_ff=1344, rope_theta=10000.0)
    (args.output / "config.json").write_text(json.dumps({
        **config, "batch_size": args.batch_size, "steps": args.steps,
        "lr": args.lr, "seed": 42, "device": "mps", "dtype": "float32",
        "warmup_iters": 2, "min_lr": args.lr / 10, "weight_decay": 0.1,
        "betas": [0.9, 0.999], "eps": 1e-8, "max_grad_norm": 1.0,
    }, indent=2))
    torch.manual_seed(42)
    np.random.seed(42)
    model = TransformerLM(**config, device="mps", dtype=torch.float32)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    train_data = np.load("data/train_tokens.npy", mmap_mode="r")
    valid_data = np.load("data/valid_tokens.npy", mmap_mode="r")
    original_step = optimizer.step
    records = []
    torch.mps.synchronize()
    started = previous = time.perf_counter()

    def measured_step(*step_args, **kwargs):
        nonlocal previous
        result = original_step(*step_args, **kwargs)
        torch.mps.synchronize()
        now = time.perf_counter()
        record = dict(step=len(records) + 1, seconds=now - previous,
                      allocated_gb=torch.mps.current_allocated_memory() / 1e9,
                      driver_gb=torch.mps.driver_allocated_memory() / 1e9)
        records.append(record)
        logging.info("Timing/memory: %s", record)
        previous = now
        return result

    optimizer.step = measured_step
    train(model, optimizer, train_data, batch_size=args.batch_size,
          context_length=256, total_steps=args.steps, device="mps",
          max_learning_rate=args.lr, min_learning_rate=args.lr / 10,
          warmup_iters=2, cosine_cycle_iters=args.steps,
          eval_interval=1, val_data=None, checkpoint_path=None)
    training_seconds = time.perf_counter() - started
    model.eval()
    from cs336_basics.model import load_data, cross_entropy_loss, save_checkpoint, load_checkpoint
    with torch.no_grad():
        x, y = load_data(valid_data, args.batch_size, 256, "mps")
        validation_loss = cross_entropy_loss(model(x), y).item()
    if not np.isfinite(validation_loss):
        raise RuntimeError("Validation loss is non-finite")
    checkpoint = args.output / "checkpoint.pt"
    save_checkpoint(model, optimizer, args.steps, checkpoint)
    assert load_checkpoint(checkpoint, model, optimizer) == args.steps
    assert all(torch.isfinite(p).all().item() for p in model.parameters())
    summary = dict(training_seconds=training_seconds,
                   median_step_seconds=statistics.median(r["seconds"] for r in records[2:]),
                   validation_loss=validation_loss, records=records,
                   parameters=sum(p.numel() for p in model.parameters()))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    logging.info("Smoke test passed: training %.2fs, median step %.3fs, validation loss %.4f",
                 training_seconds, summary["median_step_seconds"], validation_loss)


if __name__ == "__main__":
    main()
