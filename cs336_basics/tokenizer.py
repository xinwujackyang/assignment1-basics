import re
from collections.abc import Iterable, Iterator
from secrets import token_bytes
from typing import Iterator
from cs336_basics.train_bpe import regular_pretokenizer, merge_pair, pretoken_to_byte_tuple

def iter_segments(text: str, special_tokens: list[str]) -> Iterator[tuple[str, bool]]:
    if not special_tokens:
        if text:
            yield text, False
        return
    
    special_token_set = set(special_tokens)  # Create a set of special tokens for quick lookup
    special_tokens = sorted(special_tokens, key=len, reverse=True) # Sort special tokens by length in descending order
    escape_special_tokens = [re.escape(token) for token in special_tokens]  # Escape special characters in the tokens

    pattern = "(" + "|".join(escape_special_tokens) + ")"  # Create a regex pattern to match any of the special tokens
    smaller_chunks = re.split(pattern, text) # Split the chunk into smaller chunks based on special tokens
    for segment in smaller_chunks:
        if segment in special_token_set:
            yield segment, True
        elif segment:
            yield segment, False

class Tokenizer:
    def __init__(self,
                vocab: dict[int, bytes],
                merges: list[tuple[bytes, bytes]],
                special_tokens: list[str] | None = None):
        self.vocab = vocab.copy()
        self.merges = merges.copy()
        self.special_tokens = list(special_tokens or [])

        index = max(self.vocab, default=-1) + 1
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.vocab.values():
                self.vocab[index] = token_bytes
                index += 1

        self.byte_id = {v: k for k, v in self.vocab.items()}


    def decode(self, ids: list[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            ids: A list of token IDs to decode.

        Returns:
            The decoded string.
        """
        tokens = [self.vocab[i] for i in ids]
        return b"".join(tokens).decode("utf-8", errors="replace")

    def encode(self, text: str) -> list[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            text: The string to encode.

        Returns:
            A list of token IDs corresponding to the input string.
        """
        ids = []

        for segment, is_special in iter_segments(text, self.special_tokens):
            if is_special:
                token_bytes = segment.encode("utf-8")
                ids.append(self.byte_id[token_bytes])
            else:
                pretoken_iter = regular_pretokenizer(segment)
                for pretoken in pretoken_iter:
                    byte_tuple = pretoken_to_byte_tuple(pretoken)
                    for pair in self.merges:
                        byte_tuple = merge_pair(byte_tuple, pair)
                    for token_bytes in byte_tuple:
                        ids.append(self.byte_id[token_bytes])

        return ids
        
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)



if __name__ == "__main__":
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = b"ab"
    vocab[257] = b"abc"

    tokenizer = Tokenizer(
        vocab,
        merges=[(b"a", b"b"), (b"ab", b"c")],
        special_tokens=["<|endoftext|>"],
    )

    cases = [
        ("", []),
        ("a", [97]),
        ("ab", [256]),
        ("abc", [257]),
        ("abc abc", [257, 32, 257]),
        ("é", [195, 169]),
        ("abc<|endoftext|>ab", [257, 258, 256]),
        ("<|endoftext|><|endoftext|>", [258, 258]),
    ]

    for text, expected in cases:
        actual = tokenizer.encode(text)
        print(f"{text!r} -> {actual}, expected={expected}")
        assert actual == expected
        assert tokenizer.decode(actual) == text

    print("All encode tests passed!")



