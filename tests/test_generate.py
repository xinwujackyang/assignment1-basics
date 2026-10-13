from unittest.mock import patch

import pytest
import torch

from cs336_basics.generate import generate


class ToyModel(torch.nn.Module):
    vocab_size = 4
    context_length = 2

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.contexts = []

    def forward(self, ids):
        assert not self.training
        assert not torch.is_grad_enabled()
        self.contexts.append(ids.tolist())
        # Equal probabilities make the exact top-p boundary deterministic.
        return torch.zeros(1, ids.shape[1], self.vocab_size)


def call(model, prompt=None, **kwargs):
    options = dict(max_new_tokens=3, temperature=1.0, top_p=1.0, eos_token_id=3)
    options.update(kwargs)
    return generate(model, [0] if prompt is None else prompt, **options)


@pytest.mark.parametrize("kwargs,error", [
    ({"max_new_tokens": -1}, ValueError),
    ({"max_new_tokens": True}, TypeError),
    ({"temperature": 0}, ValueError),
    ({"temperature": float("nan")}, ValueError),
    ({"temperature": float("inf")}, ValueError),
    ({"temperature": "1"}, TypeError),
    ({"top_p": 0}, ValueError),
    ({"top_p": 1.1}, ValueError),
    ({"top_p": float("nan")}, ValueError),
    ({"eos_token_id": 4}, ValueError),
    ({"eos_token_id": False}, TypeError),
])
def test_invalid_options(kwargs, error):
    with pytest.raises(error):
        call(ToyModel(), **kwargs)


@pytest.mark.parametrize("prompt,error", [
    ([], ValueError), ([-1], ValueError), ([4], ValueError),
    ([1.0], TypeError), ([True], TypeError), ((1,), TypeError),
])
def test_invalid_prompt(prompt, error):
    with pytest.raises(error):
        call(ToyModel(), prompt)


def test_zero_tokens_returns_copy():
    model = ToyModel()
    prompt = [0, 1]
    result = call(model, prompt, max_new_tokens=0)
    assert result == prompt and result is not prompt
    assert not model.contexts


def test_context_limit_and_eos():
    model = ToyModel()
    prompt = [0, 1, 2]
    with patch("torch.multinomial", side_effect=[torch.tensor([[1]]), torch.tensor([[3]])]):
        result = call(model, prompt)
    assert result == [0, 1, 2, 1, 3]
    assert prompt == [0, 1, 2]
    assert model.contexts == [[[1, 2]], [[2, 1]]]


@pytest.mark.parametrize("top_p,count", [(0.1, 1), (0.5, 2), (1.0, 4)])
def test_top_p_and_length_limit(top_p, count):
    def sample(probs, num_samples):
        assert torch.count_nonzero(probs).item() == count
        torch.testing.assert_close(probs.sum(), torch.tensor(1.0))
        return torch.tensor([[0]])

    with patch("torch.multinomial", side_effect=sample):
        result = call(ToyModel(), top_p=top_p)
    assert result == [0, 0, 0, 0]
