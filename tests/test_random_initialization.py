import pytest
import torch

from cs336_basics.model import (
    AdamW, Embedding, MultiHeadSelfAttention, SwiGLU,
    TransformerBlock, TransformerLM, cross_entropy_loss,
)


def test_random_lm_training_step():
    torch.manual_seed(42)
    model = TransformerLM(32, 8, 16, 2, 4, 32, 10000.0, device="cpu")
    assert model.layers[0].attention.Q.weight is not model.layers[1].attention.Q.weight
    assert "embedding.weight" in model.state_dict()
    before = model.embedding.weight.detach().clone()
    optimizer = AdamW(model.parameters())
    logits = model(torch.randint(32, (2, 6)))
    assert logits.shape == (2, 6, 32)
    loss = cross_entropy_loss(logits, torch.randint(32, (2, 6)))
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    optimizer.step()
    assert not torch.equal(before, model.embedding.weight)


def test_optional_component_weights_and_dtype():
    x = torch.randn(2, 5, 16, dtype=torch.float64)
    modules = [
        MultiHeadSelfAttention(16, 4, max_seq_len=8, dtype=torch.float64),
        SwiGLU(16, 32, dtype=torch.float64),
        TransformerBlock(16, 4, 32, max_seq_len=8, dtype=torch.float64),
    ]
    for module in modules:
        assert all(p.dtype == torch.float64 for p in module.parameters())
        output = module(x)
        assert output.shape == x.shape
        assert output.dtype == x.dtype


def test_provided_embedding_is_registered():
    weights = torch.randn(32, 16)
    embedding = Embedding(32, 16, weights=weights, dtype=torch.float64)
    assert isinstance(embedding.weight, torch.nn.Parameter)
    assert embedding.weight.dtype == torch.float64
    torch.testing.assert_close(embedding.weight, weights.double())


def test_incomplete_lm_weights_raise():
    with pytest.raises(KeyError, match="token_embeddings.weight"):
        TransformerLM(32, 8, 16, 2, 4, 32, 10000.0, weights={})
