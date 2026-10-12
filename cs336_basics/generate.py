"""Generate text using temperature scaling and nucleus sampling.

Tokenizer JSON format (byte values, not decoded Unicode strings):
{"vocab": {"0": [0], "1": [1]}, "merges": [[[97], [98]]],
 "special_tokens": ["<|endoftext|>"]}
Use the complete tokenizer that produced the training data, not this example.
"""

import argparse
import json
import math
from numbers import Real
from pathlib import Path

import torch

from cs336_basics.model import TransformerLM, softmax
from cs336_basics.tokenizer import Tokenizer


def generate(
    model: TransformerLM,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_id: int,
) -> list[int]:
    """Return prompt IDs followed by generated IDs, including EOS if sampled.

    Stop on EOS or max_new_tokens. Infer the device from model parameters.
    For long sequences, feed only the last model.context_length tokens.
    This function sets evaluation mode and disables gradient recording.
    The model remains in evaluation mode after generation.
    """
    if not isinstance(prompt_ids, list):
        raise TypeError("prompt_ids must be a list of integer token IDs.")
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty.")
    if any(type(token_id) is not int for token_id in prompt_ids):
        raise TypeError("Every prompt token ID must be an integer, not a boolean.")
    if any(not 0 <= token_id < model.vocab_size for token_id in prompt_ids):
        raise ValueError("Prompt token IDs must be in [0, model.vocab_size).")
    if type(max_new_tokens) is not int:
        raise TypeError("max_new_tokens must be an integer.")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")
    for name, value in (("temperature", temperature), ("top_p", top_p)):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number.")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1].")
    if type(eos_token_id) is not int:
        raise TypeError("eos_token_id must be an integer.")
    if not 0 <= eos_token_id < model.vocab_size:
        raise ValueError("eos_token_id must be in [0, model.vocab_size).")
    if max_new_tokens == 0:
        return prompt_ids.copy()

    device = next(model.parameters()).device
    input_ids = torch.tensor(
        prompt_ids, dtype=torch.long, device=device
    ).unsqueeze(0)

    generated_ids = []
    model.eval()  # Ensure model is in evaluation mode

    with torch.no_grad():  # Disable gradient computation for inference
        for _ in range(max_new_tokens):

            context_ids = input_ids[:, -model.context_length:]
            next_logits = model(context_ids)[:, -1, :] # (1, vocab_size)
            probs = softmax(next_logits / temperature, dim=-1) # (1, vocab_size)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs >= top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs[0, indices_to_remove] = 0
            probs /= probs.sum(dim=-1, keepdim=True)

            sample_id = torch.multinomial(probs, num_samples=1).item()
            generated_ids.append(sample_id)

            input_ids = torch.cat([input_ids, torch.tensor([[sample_id]], dtype=torch.long, device=device)], dim=1)

            if sample_id == eos_token_id:
                break

    return prompt_ids + generated_ids




    

    


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True, help="Byte-list JSON tokenizer")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=352)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--eos-token", default="<|endoftext|>")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for name in ("context_length", "d_model", "num_layers", "num_heads", "d_ff"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if args.d_model % args.num_heads or (args.d_model // args.num_heads) % 2:
        parser.error("d_model must be divisible by num_heads with an even head dimension")
    if args.max_new_tokens < 0:
        parser.error("max_new_tokens must be non-negative")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("temperature must be finite and positive")
    if not 0 < args.top_p <= 1:
        parser.error("top_p must be in (0, 1]")
    if not math.isfinite(args.rope_theta) or args.rope_theta <= 0:
        parser.error("rope_theta must be finite and positive")

    with args.tokenizer.open(encoding="utf-8") as stream:
        data = json.load(stream)

    def as_bytes(values: list[int]) -> bytes:
        if not isinstance(values, list) or not all(
            type(value) is int and 0 <= value <= 255 for value in values
        ):
            parser.error("Tokenizer bytes must be lists of integers in [0, 255]")
        return bytes(values)

    vocab = {int(key): as_bytes(value) for key, value in data["vocab"].items()}
    merges = [(as_bytes(a), as_bytes(b)) for a, b in data["merges"]]
    special_tokens = data.get("special_tokens", [])
    if not isinstance(special_tokens, list) or not all(isinstance(t, str) and t for t in special_tokens):
        parser.error("special_tokens must be a list of nonempty strings")
    if set(vocab) != set(range(len(vocab))):
        parser.error("Vocabulary IDs must be contiguous and start at zero")
    if any(token.encode("utf-8") not in vocab.values() for token in special_tokens):
        parser.error("Special tokens must already exist in the saved vocabulary")
    if args.eos_token not in special_tokens:
        parser.error("EOS must be registered in special_tokens")
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        parser.error("Prompt must encode to at least one token")
    eos_id = tokenizer.byte_id[args.eos_token.encode("utf-8")]

    torch.manual_seed(args.seed)
    # Load only model state: inference does not need optimizer state restored.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = TransformerLM(
        vocab_size=len(vocab), context_length=args.context_length,
        d_model=args.d_model, num_layers=args.num_layers,
        num_heads=args.num_heads, d_ff=args.d_ff, rope_theta=args.rope_theta,
        dtype=torch.float32,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(args.device)
    model.eval()
    with torch.no_grad():
        output_ids = prompt_ids if args.max_new_tokens == 0 else generate(
            model, prompt_ids, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p, eos_token_id=eos_id,
        )
    # The generate contract includes the prompt; display the entire completion.
    if output_ids and output_ids[-1] == eos_id:
        output_ids = output_ids[:-1]
    print(tokenizer.decode(output_ids))


if __name__ == "__main__":
    main()
