"""Compare incremental state with the simple full-scan implementation."""

import random
from collections import Counter

from cs336_basics.train_bpe import (
    apply_indexed_merge,
    apply_merge,
    get_most_frequent_pair,
    get_pair_hashmap,
    pair_count_pretokens,
)


def test_incremental_state_matches_full_recount():
    rng = random.Random(336)
    for _ in range(30):
        reference = Counter()
        for _ in range(25):
            sequence = tuple(rng.choice((b"a", b"b", b"c")) for _ in range(rng.randrange(9)))
            reference[sequence] += rng.randrange(1, 8)
        sequences = list(reference)
        frequencies = list(reference.values())
        pair_counts = pair_count_pretokens(reference)
        index = get_pair_hashmap(sequences)

        while pair_counts:
            selected = get_most_frequent_pair(pair_counts)
            assert selected == get_most_frequent_pair(pair_count_pretokens(reference))
            apply_indexed_merge(sequences, frequencies, pair_counts, index, selected)
            reference = apply_merge(reference, selected)

            reconstructed = Counter()
            for sequence, frequency in zip(sequences, frequencies):
                reconstructed[sequence] += frequency
            assert reconstructed == reference
            assert pair_counts == pair_count_pretokens(reference)
            assert index == get_pair_hashmap(sequences)
