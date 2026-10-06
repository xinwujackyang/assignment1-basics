from typing import Iterator
from collections import Counter, defaultdict
from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re

RULE = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

def regular_pretokenizer(chunk: str) -> Iterator[str]:
    # Tokenize the chunk using the defined rule
    matches = re.finditer(RULE, chunk)
    for m in matches:
        yield m.group()
    

def iter_pretokens(chunk: str, special_tokens: list[str]) -> Iterator[str]:
    if not special_tokens:
        yield from regular_pretokenizer(chunk)
        return
    

    special_tokens = sorted(special_tokens, key=len, reverse=True) # Sort special tokens by length in descending order
    for i, special_token in enumerate(special_tokens):
        special_tokens[i] = re.escape(special_token)  # Escape special characters in the token
     
    smaller_chunks = re.split("|".join(special_tokens), chunk) # Split the chunk into smaller chunks based on special tokens
    for smaller_chunk in smaller_chunks:
        text_groups = regular_pretokenizer(smaller_chunk)
        for text in text_groups:
            yield text


def pretoken_to_byte_tuple(pretoken: str) -> tuple[bytes, ...]:
    pretoken = pretoken.encode("utf-8")  # Encode the pretoken to bytes
    return tuple(pretoken[i-1:i] for i in range(1, len(pretoken) + 1))  # Return a tuple of bytes for each character in the pretoken



def count_pretokens(text_chunk: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    pretoken_counts = {}
    for pretoken in iter_pretokens(text_chunk, special_tokens):
        byte_tuple = pretoken_to_byte_tuple(pretoken)
        pretoken_counts[byte_tuple] = pretoken_counts.get(byte_tuple, 0) + 1  # Count the occurrences of each byte tuple
    return pretoken_counts

def pair_count_pretokens(pretoken_counts: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    pair_counts = {}
    for pretoken_tuple, count in pretoken_counts.items():
        if len(pretoken_tuple) < 2:
            continue  # Skip single-byte pretokens
        for i in range(len(pretoken_tuple) - 1):
            pair = (pretoken_tuple[i], pretoken_tuple[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + count  # Count the occurrences of each byte pair
    return pair_counts

def merge_pair(token_sequence: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    merged_sequence = []
    i = 0
    while i < len(token_sequence):
        if i < len(token_sequence) - 1 and (token_sequence[i], token_sequence[i + 1]) == pair:
            merged_sequence.append(token_sequence[i] + token_sequence[i + 1])  # Merge the pair into a single token
            i += 2  # Skip the next token since it has been merged
        else:
            merged_sequence.append(token_sequence[i])  # Keep the current token as is
            i += 1
    return tuple(merged_sequence)  # Return the new sequence with the merged pair

def get_most_frequent_pair(pair_counts: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes] | None:
    if not pair_counts:
        return None  # Return None if there are no pairs to consider
    return max(pair_counts.items(), key=lambda item: (item[1], item[0]))[0]  # Return the pair with the highest count

def initialize_vocabulary(special_tokens: list[str]) -> dict[int, bytes]:
    vocab = {}
    index = 0
    while index < 256:
        vocab[index] = bytes([index])  # Initialize the vocabulary with single-byte tokens
        index += 1

    for special_token in special_tokens:
        byte = special_token.encode("utf-8")  # Encode the special token to bytes
        vocab[index] = byte
        index += 1
    
    return vocab

def get_pair_hashmap(
    sequences: list[tuple[bytes, ...]],
) -> dict[tuple[bytes, bytes], set[int]]:
    """Build an inverted index of adjacent pairs.

    sequences: Current token tuples for each pre-token; list indices are stable sequence IDs.
    Returns pair -> set of sequence IDs containing it, regardless of repeated occurrences.
    Does not modify the input.
    """
    pair_sequences = defaultdict(set)
    for sequence_id, sequence in enumerate(sequences):
        for pair in zip(sequence, sequence[1:]):
            pair_sequences[pair].add(sequence_id)
    return dict(pair_sequences)


def apply_indexed_merge(
    sequences: list[tuple[bytes, ...]],
    frequencies: list[int],
    pair_counts: dict[tuple[bytes, bytes], int],
    pair_sequences: dict[tuple[bytes, bytes], set[int]],
    selected_pair: tuple[bytes, bytes],
) -> None:
    """Apply one merge and incrementally update statistics for affected sequences.

    sequences: Sequence ID -> current token tuple; affected entries are replaced in place.
    frequencies: Corpus frequency for each sequence ID; aligned with sequences and unchanged.
    pair_counts: Pair -> global weighted occurrence count; updated in place.
    pair_sequences: Pair -> set of sequence IDs containing it; updated in place.
    selected_pair: The fixed pair to merge this round; must exist in the current index.
    Returns None; callers use the updated sequences, pair_counts, and pair_sequences directly.

    Sequence IDs remain stable. Local Counters preserve multiplicity, including two aa pairs in aaa.
    """
    # Updating the index below also changes selected_pair's membership set.
    affected_ids = tuple(pair_sequences[selected_pair])
    for sequence_id in affected_ids:
        old_sequence = sequences[sequence_id]
        new_sequence = merge_pair(old_sequence, selected_pair)
        old_pairs = Counter(zip(old_sequence, old_sequence[1:]))
        new_pairs = Counter(zip(new_sequence, new_sequence[1:]))
        frequency = frequencies[sequence_id]

        for pair in old_pairs.keys() | new_pairs.keys():
            delta = (new_pairs[pair] - old_pairs[pair]) * frequency
            if delta:
                updated_count = pair_counts.get(pair, 0) + delta
                if updated_count:
                    pair_counts[pair] = updated_count
                else:
                    del pair_counts[pair]
            if pair not in new_pairs:
                members = pair_sequences[pair]
                members.remove(sequence_id)
                if not members:
                    del pair_sequences[pair]
            elif pair not in old_pairs:
                pair_sequences.setdefault(pair, set()).add(sequence_id)

        sequences[sequence_id] = new_sequence


def apply_merge(pretoken_counts: dict[tuple[bytes, ...], int], pair: tuple[bytes, bytes]) -> dict[tuple[bytes, ...], int]:
    new_pretoken_counts = {}
    for pretoken_tuple, count in pretoken_counts.items():
        if len(pretoken_tuple) < 2:
            new_pretoken_counts[pretoken_tuple] = new_pretoken_counts.get(pretoken_tuple, 0) + count
            continue
        merged_tuple = merge_pair(pretoken_tuple, pair)  # Merge the specified pair in the pretoken tuple
        new_pretoken_counts[merged_tuple] = new_pretoken_counts.get(merged_tuple, 0) + count  # Update the counts with the merged tuple
    return new_pretoken_counts

def train_bpe(file, special_tokens, vocab_size):

    vocab = initialize_vocabulary(special_tokens)  # Initialize the vocabulary with special tokens

    if vocab_size < len(vocab):
        raise ValueError("vocab_size must be greater than or equal to the number of special tokens and single-byte tokens.")

    if vocab_size == len(vocab):
        return vocab, []  # Return the vocabulary and an empty list of merges if the vocab size is already reached
    
    merges = []

    with open(file, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        
        pretoken_counts = {}
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk_counts = count_pretokens(chunk, special_tokens) # the frequency map of big token tuple
            for pretoken_tuple, count in chunk_counts.items():
                pretoken_counts[pretoken_tuple] = pretoken_counts.get(pretoken_tuple, 0) + count  # Merge counts from all chunks

        # Stable sequence IDs let the inverted index survive tuple replacement.
        sequences = list(pretoken_counts)
        frequencies = list(pretoken_counts.values())
        pair_sequences = get_pair_hashmap(sequences)
        pair_counts = pair_count_pretokens(pretoken_counts)
        most_frequent_pair = get_most_frequent_pair(pair_counts)  # Get the most frequent byte pair

        while most_frequent_pair and len(vocab) < vocab_size:
            merges.append(most_frequent_pair)  # Add the most frequent pair to the list of merges
            apply_indexed_merge(
                sequences, frequencies, pair_counts, pair_sequences, most_frequent_pair
            )
            vocab[len(vocab)] = most_frequent_pair[0] + most_frequent_pair[1]  # Add the merged pair to the vocabulary
            most_frequent_pair = get_most_frequent_pair(pair_counts)  # Get the next most frequent byte pair

    return vocab, merges
        

        


if __name__ == "__main__":
    # Example usage of the regular_tokenizer function
    # 1. 词表初始化：预期 257，以及 b"A"、b"<|endoftext|>"
    vocab = initialize_vocabulary(["<|endoftext|>"])
    print("vocab:", len(vocab), vocab[65], vocab[256])

    # 2. 分隔 special token：预期 ["A", "B"]
    print("pretokens:", list(
        iter_pretokens("A<|endoftext|>B", ["<|endoftext|>"])
    ))

    # 3. Unicode：预期 (b"\xc3", b"\xa9")
    print("bytes:", pretoken_to_byte_tuple("é"))

    # 4. 加权 pair 计数：预期 (a, b): 5，(b, a): 3
    counts = {
        (b"a", b"b", b"a"): 3,
        (b"a", b"b"): 2,
    }
    print("pairs:", pair_count_pretokens(counts))

    # 5. 同频选择：预期 (b"a", b"c")
    print("best pair:", get_most_frequent_pair({
        (b"a", b"b"): 5,
        (b"a", b"c"): 5,
        (b"x", b"y"): 4,
    }))

    # 6. 重叠合并：预期 (b"aa", b"a")
    print("merge:", merge_pair(
        (b"a", b"a", b"a"), (b"a", b"a")
    ))

    # 7. 整体更新：预期 {(b"ab", b"a"): 3, (b"ab",): 2}
    print("apply merge:", apply_merge(counts, (b"a", b"b")))

    # 8. 恰好满足容量：应提前返回，不读取文件
    vocab, merges = train_bpe("unused.txt", [], 256)
    print("exact size:", len(vocab), merges)  # 256 []

    # 9. 容量不足：预期捕获 ValueError
    try:
        train_bpe("unused.txt", ["<|endoftext|>"], 256)
    except ValueError as error:
        print("expected error:", error)

    # 10. Inspect the initial state: sequence IDs are list indices, not token IDs.
    pretoken_counts = {
        (b"a", b"b", b"a", b"b"): 3,
        (b"a", b"b", b"c"): 2,
        (b"x",): 4,
    }
    sequences = list(pretoken_counts)
    frequencies = list(pretoken_counts.values())
    pair_sequences = get_pair_hashmap(sequences)
    pair_counts = pair_count_pretokens(pretoken_counts)
    most_frequent_pair = get_most_frequent_pair(pair_counts)

    print("\n--- Indexed BPE state example ---")
    print("pretoken_counts:", pretoken_counts)
    print("sequences:", sequences)
    print("frequencies:", frequencies)
    for sequence_id, sequence in enumerate(sequences):
        print(f"  sequence ID {sequence_id}: {sequence}, frequency={frequencies[sequence_id]}")
    print("pair_sequences:", pair_sequences)
    print("pair_counts:", pair_counts)
    print("most_frequent_pair:", most_frequent_pair)
    # (a, b) occurs twice in sequence 0 and once in sequence 1: 2 * 3 + 1 * 2 = 8.
    # Its index is {0, 1}: membership records each sequence ID only once.
