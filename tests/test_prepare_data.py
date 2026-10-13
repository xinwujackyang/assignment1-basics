import numpy as np

from cs336_basics.prepare_data import EOS, CachedEncoder, chunks, encode_file, learn
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_bpe import train_bpe


def test_streaming_preparation(tmp_path):
    text = (f"hello hello! héllo 世界\r\n{EOS}\nbanana aaa aaa!{EOS}\n") * 4
    source = tmp_path / "text.txt"
    source.write_bytes(text.encode("utf-8"))
    assert "".join(chunks(source, size=7)) == text
    vocab, merges = learn(source, 280)
    reference_vocab, reference_merges = train_bpe(str(source), [EOS], 280)
    # The reference file reader uses binary decoding, preserving CRLF too.
    assert vocab == reference_vocab
    assert merges == reference_merges
    reference = Tokenizer(vocab, merges, [EOS])
    encoder = CachedEncoder(vocab, merges)
    assert encoder.encode(text) == reference.encode(text)
    split_ids = [token for chunk in chunks(source, size=7) for token in encoder.encode(chunk)]
    assert split_ids == reference.encode(text)
    output = tmp_path / "tokens.npy"
    stats = encode_file(source, output, encoder)
    tokens = np.load(output, mmap_mode="r")
    assert tokens.dtype == np.uint16
    assert tokens.tolist() == reference.encode(text)
    assert stats["tokens"] == len(tokens)
    assert reference.decode(tokens.tolist()) == text
