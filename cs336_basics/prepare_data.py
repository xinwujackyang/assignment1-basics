"""Prepare TinyStories using the training split only for BPE learning."""

import argparse
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_bpe import (
    regular_pretokenizer, pretoken_to_byte_tuple, initialize_vocabulary,
    get_pair_hashmap, pair_count_pretokens, get_most_frequent_pair,
    apply_indexed_merge, merge_pair,
)

EOS = "<|endoftext|>"


def chunks(path, size=4 * 1024 * 1024):
    """Yield exact text chunks ending at document boundaries, preserving newlines."""
    with open(path, encoding="utf-8", newline="") as stream:
        pending = ""
        while block := stream.read(size):
            pending += block
            end = pending.rfind(EOS)
            if end >= 0:
                end += len(EOS)
                yield pending[:end]
                pending = pending[end:]
        if pending:
            yield pending


def learn(path, vocab_size):
    counts = Counter()
    started = time.monotonic()
    processed = 0
    for chunk in chunks(path):
        for text in chunk.split(EOS):
            counts.update(regular_pretokenizer(text))
        processed += len(chunk.encode("utf-8"))
        print(f"counted {processed / 1e6:.1f} MB in {time.monotonic()-started:.1f}s", flush=True)
    byte_counts = {pretoken_to_byte_tuple(text): count for text, count in counts.items()}
    del counts
    sequences = list(byte_counts)
    frequencies = list(byte_counts.values())
    pair_sequences = get_pair_hashmap(sequences)
    pair_counts = pair_count_pretokens(byte_counts)
    del byte_counts
    vocab = initialize_vocabulary([EOS])
    merges = []
    while len(vocab) < vocab_size and pair_counts:
        pair = get_most_frequent_pair(pair_counts)
        apply_indexed_merge(sequences, frequencies, pair_counts, pair_sequences, pair)
        merges.append(pair)
        vocab[len(vocab)] = pair[0] + pair[1]
        if len(merges) % 500 == 0:
            print(f"merges {len(merges)}; elapsed {time.monotonic()-started:.1f}s", flush=True)
    return vocab, merges


class CachedEncoder:
    """Apply learned merges by rank, caching repeated pretoken encodings."""

    def __init__(self, vocab, merges):
        self.ids = {value: key for key, value in vocab.items()}
        self.ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.pretoken = lru_cache(maxsize=100_000)(self._pretoken)

    def _pretoken(self, text):
        sequence = pretoken_to_byte_tuple(text)
        while len(sequence) > 1:
            pair = min(zip(sequence, sequence[1:]),
                       key=lambda p: self.ranks.get(p, float("inf")))
            if pair not in self.ranks:
                break
            sequence = merge_pair(sequence, pair)
        return tuple(self.ids[token] for token in sequence)

    def encode(self, text):
        ids = []
        for index, segment in enumerate(text.split(EOS)):
            if index:
                ids.append(self.ids[EOS.encode()])
            for pretoken in regular_pretokenizer(segment):
                ids.extend(self.pretoken(pretoken))
        return ids


def encode_file(path, output, encoder):
    """Spool uint16 batches, then write a genuine .npy file with bounded RAM."""
    started = time.monotonic()
    count = processed = 0
    with tempfile.TemporaryFile(dir=output.parent) as spool:
        for chunk in chunks(path):
            ids = encoder.encode(chunk)
            array = np.asarray(ids, dtype=np.uint16)
            spool.write(array.tobytes())
            count += len(ids)
            processed += len(chunk.encode("utf-8"))
            print(f"{path.name}: {processed / 1e6:.1f} MB, {count} tokens, "
                  f"{time.monotonic()-started:.1f}s", flush=True)
        partial = output.with_suffix(".npy.part")
        result = np.lib.format.open_memmap(partial, mode="w+", dtype=np.uint16, shape=(count,))
        spool.seek(0)
        offset = 0
        while block := spool.read(8 * 1024 * 1024):
            array = np.frombuffer(block, dtype=np.uint16)
            result[offset:offset + len(array)] = array
            offset += len(array)
        result.flush()
        del result
        partial.replace(output)
    return {"tokens": count, "seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    root = args.data_dir
    train = root / "TinyStoriesV2-GPT4-train.txt"
    valid = root / "TinyStoriesV2-GPT4-valid.txt"
    tokenizer_path = root / "tokenizer.json"
    if tokenizer_path.exists():
        data = json.loads(tokenizer_path.read_text())
        vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
        merges = [(bytes(a), bytes(b)) for a, b in data["merges"]]
    else:
        vocab, merges = learn(train, 10_000)
        data = dict(vocab={str(k): list(v) for k, v in vocab.items()},
                    merges=[[list(a), list(b)] for a, b in merges], special_tokens=[EOS])
        partial = tokenizer_path.with_suffix(".json.part")
        partial.write_text(json.dumps(data), encoding="utf-8")
        partial.replace(tokenizer_path)
    encoder = CachedEncoder(vocab, merges)
    reference = Tokenizer(vocab, merges, [EOS])
    samples = ["Hello world!", "é 中文\n\n", f"a{EOS}b", "aaaaaaa"]
    with valid.open(encoding="utf-8", newline="") as stream:
        samples.append(stream.read(2000))
    for text in samples:
        ids = encoder.encode(text)
        assert ids == reference.encode(text)
        assert reference.decode(ids) == text
    print("Fast encoder matches reference; roundtrip checks passed.", flush=True)
    stats = {}
    for name, path in [("valid", valid), ("train", train)]:
        output = root / f"{name}_tokens.npy"
        if output.exists():
            print(f"Skipping existing {output}", flush=True)
            continue
        stats[name] = encode_file(path, output, encoder)
        tokens = np.load(output, mmap_mode="r")
        assert tokens.ndim == 1 and tokens.dtype == np.uint16
        assert int(tokens.max()) < len(vocab)
    (root / "preparation_stats.json").write_text(json.dumps(stats, indent=2))
    print("Preparation complete.", flush=True)


if __name__ == "__main__":
    main()
