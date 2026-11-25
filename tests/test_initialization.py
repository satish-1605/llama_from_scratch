
import math

import pytest
import torch
import torch.nn as nn

from src.models.normalization import RMSNorm
from src.models.initialization import initialize_llama_weights


class DummyLlamaModel(nn.Module):
    """Small model for testing initialization behavior."""

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(1000, 128)
        self.norm = RMSNorm(128)

        self.linear = nn.Linear(128, 128)
        self.residual_projection = nn.Linear(128, 128)


def test_embedding_initialization():
    model = DummyLlamaModel()

    initialize_llama_weights(model, n_layers=32)

    weights = model.embedding.weight

    assert abs(weights.mean().item()) < 0.01
    assert abs(weights.std().item() - 0.02) < 0.003


def test_linear_initialization():
    model = DummyLlamaModel()

    initialize_llama_weights(model, n_layers=32)

    weights = model.linear.weight

    assert abs(weights.mean().item()) < 0.01
    assert abs(weights.std().item() - 0.02) < 0.003


def test_residual_projection_initialization():
    model = DummyLlamaModel()

    n_layers = 32
    init_std = 0.02

    initialize_llama_weights(
        model,
        n_layers=n_layers,
        init_std=init_std,
        residual_projection_names=("residual_projection",),
    )

    expected_std = init_std / math.sqrt(2 * n_layers)

    actual_std = model.residual_projection.weight.std().item()

    assert abs(actual_std - expected_std) < expected_std * 0.15


def test_rmsnorm_initialization():
    model = DummyLlamaModel()

    model.norm.weight.data.fill_(5.0)

    initialize_llama_weights(model, n_layers=32)

    assert torch.allclose(
        model.norm.weight,
        torch.ones_like(model.norm.weight),
    )


def test_linear_bias_initialization():
    model = DummyLlamaModel()

    initialize_llama_weights(model, n_layers=32)

    assert torch.allclose(
        model.linear.bias,
        torch.zeros_like(model.linear.bias),
    )


def test_invalid_layer_count():
    model = DummyLlamaModel()

    with pytest.raises(ValueError):
        initialize_llama_weights(model, n_layers=0)


def test_invalid_init_std():
    model = DummyLlamaModel()

    with pytest.raises(ValueError):
        initialize_llama_weights(
            model,
            n_layers=32,
            init_std=0,
        )


def test_unknown_residual_projection():
    model = DummyLlamaModel()

    with pytest.raises(ValueError):
        initialize_llama_weights(
            model,
            n_layers=32,
            residual_projection_names=("unknown_layer",),
        )
