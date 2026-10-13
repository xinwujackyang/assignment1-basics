"""Sequential, reproducible MPS learning-rate screening on TinyStories."""

import argparse
import csv
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from cs336_basics.model import (
    AdamW, TransformerLM, cross_entropy_loss, get_lr_cosine_schedule,
    gradient_clip, load_data, save_checkpoint,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3, 3e-3])
    args = parser.parse_args()
    if not (0 <= args.warmup < args.steps) or min(args.eval_interval, args.eval_batches) <= 0:
        parser.error("Invalid step, warmup, or validation settings")
    if any(not np.isfinite(lr) or lr <= 0 for lr in args.learning_rates):
        parser.error("Learning rates must be finite and positive")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no CPU fallback")
    args.output.mkdir(parents=True, exist_ok=False)
    config = dict(vocab_size=10000, context_length=256, d_model=512,
                  num_layers=4, num_heads=16, d_ff=1344, rope_theta=10000.0)
    metadata = {**config, "steps": args.steps, "batch_size": 32, "warmup": args.warmup,
                "learning_rates": args.learning_rates, "min_lr_ratio": 0.1,
                "eval_interval": args.eval_interval, "eval_batches": args.eval_batches,
                "seed": 42, "validation_seed": 123, "device": "mps", "dtype": "float32",
                "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.1,
                "max_grad_norm": 1.0, "torch_version": str(torch.__version__)}
    (args.output / "config.json").write_text(json.dumps(metadata, indent=2))
    train_data = np.load("data/train_tokens.npy", mmap_mode="r")
    valid_data = np.load("data/valid_tokens.npy", mmap_mode="r")
    np.random.seed(123)
    validation = [load_data(valid_data, 32, 256, "cpu") for _ in range(args.eval_batches)]
    results = []

    for index, peak_lr in enumerate(args.learning_rates):
        directory = args.output / f"{index + 1}-lr-{peak_lr:g}"
        directory.mkdir()
        torch.manual_seed(42)
        np.random.seed(42)
        model = TransformerLM(**config, device="mps", dtype=torch.float32)
        optimizer = AdamW(model.parameters(), lr=peak_lr, weight_decay=0.1)
        torch.mps.synchronize()
        started = time.perf_counter()
        status = "complete"
        final_val = None
        completed = 0
        print(f"START lr={peak_lr:g}", flush=True)
        with (directory / "metrics.csv").open("w", newline="") as stream:
            fields = ["step", "wall_seconds", "learning_rate", "train_loss", "grad_norm", "val_loss", "driver_gb"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for step in range(args.steps):
                model.train()
                lr = get_lr_cosine_schedule(step, peak_lr, peak_lr / 10, args.warmup, args.steps)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                x, y = load_data(train_data, 32, 256, "mps")
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = cross_entropy_loss(logits, y)
                loss_value = loss.item()
                if not np.isfinite(loss_value):
                    status = "nonfinite_loss"
                    break
                loss.backward()
                norm = torch.sqrt(sum(p.grad.float().square().sum() for p in model.parameters() if p.grad is not None)).item()
                if not np.isfinite(norm):
                    status = "nonfinite_gradient"
                    break
                gradient_clip(model.parameters(), 1.0)
                optimizer.step()
                completed = step + 1
                del logits, loss, x, y
                val_loss = None
                if completed % args.eval_interval == 0 or completed == args.steps:
                    model.eval()
                    with torch.no_grad():
                        losses = [cross_entropy_loss(model(vx.to("mps")), vy.to("mps")).item()
                                  for vx, vy in validation]
                    val_loss = float(np.mean(losses))
                    final_val = val_loss
                torch.mps.synchronize()
                elapsed = time.perf_counter() - started
                writer.writerow(dict(step=completed, wall_seconds=elapsed, learning_rate=lr,
                                     train_loss=loss_value, grad_norm=norm, val_loss=val_loss,
                                     driver_gb=torch.mps.driver_allocated_memory() / 1e9))
                stream.flush()
                if val_loss is not None or completed == 1:
                    print(f"lr={peak_lr:g} step={completed} train={loss_value:.4f} val={val_loss} elapsed={elapsed:.1f}s", flush=True)
                if val_loss is not None and not np.isfinite(val_loss):
                    status = "nonfinite_validation"
                    break
        if status == "complete":
            save_checkpoint(model, optimizer, completed, directory / "checkpoint.pt")
        result = dict(peak_lr=peak_lr, status=status, completed_steps=completed,
                      final_validation_loss=final_val if final_val is not None and np.isfinite(final_val) else None,
                      wall_seconds=time.perf_counter() - started)
        results.append(result)
        (args.output / "results.json").write_text(json.dumps(results, indent=2))
        print(f"FINISH {result}", flush=True)
        del model, optimizer
        gc.collect()
        torch.mps.empty_cache()
    print("Sweep complete; compare results.json and per-run metrics.csv.", flush=True)


if __name__ == "__main__":
    main()
