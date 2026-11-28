
"""
Unit tests for AdamW optimizer construction.

Tests cover:
    1. Optimizer initialization.
    2. Parameter grouping.
    3. Frozen parameter exclusion.
    4. Parameter coverage and uniqueness.
    5. Hyperparameter configuration.
    6. Invalid input handling.
    7. Models without trainable parameters.
    8. Actual parameter updates.
    9. Decoupled weight decay behavior.
"""

import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW

from src.training.optimizer import create_adamw_optimizer


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------


@pytest.fixture
def dummy_model():
    """
    Create a small model containing different parameter types.

    Includes:
        - Embedding weights: 2D
        - Linear weights: 2D
        - Linear bias: 1D
        - LayerNorm weights and bias: 1D
        - Frozen linear layer parameters
    """

    model = nn.ModuleDict(
        {
            "embedding": nn.Embedding(20, 8),
            "linear": nn.Linear(8, 4),
            "norm": nn.LayerNorm(4),
            "frozen": nn.Linear(4, 2),
        }
    )

    for param in model["frozen"].parameters():
        param.requires_grad = False

    return model


@pytest.fixture
def optimizer(dummy_model):
    """Create a default AdamW optimizer for testing."""

    return create_adamw_optimizer(
        model=dummy_model,
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )


# ---------------------------------------------------------
# Optimizer Initialization Tests
# ---------------------------------------------------------


def test_optimizer_initialization(optimizer):
    """Verify that the function returns an AdamW instance."""

    assert isinstance(optimizer, AdamW)


def test_optimizer_hyperparameters(optimizer):
    """Verify optimizer hyperparameters are configured correctly."""

    for group in optimizer.param_groups:

        assert group["lr"] == pytest.approx(3e-4)
        assert group["betas"] == (0.9, 0.999)
        assert group["eps"] == pytest.approx(1e-8)


# ---------------------------------------------------------
# Parameter Grouping Tests
# ---------------------------------------------------------


def test_parameter_grouping(dummy_model, optimizer):
    """
    Verify 2D parameters receive weight decay
    and 1D parameters do not.
    """

    decay_params = optimizer.param_groups[0]
    no_decay_params = optimizer.param_groups[1]

    assert decay_params["weight_decay"] == pytest.approx(0.01)
    assert no_decay_params["weight_decay"] == pytest.approx(0.0)

    assert all(
        param.ndim >= 2
        for param in decay_params["params"]
    )

    assert all(
        param.ndim < 2
        for param in no_decay_params["params"]
    )


def test_embedding_and_linear_weights_receive_decay(
    dummy_model,
    optimizer,
):
    """Verify embedding and linear weights belong to decay group."""

    decay_ids = {
        id(param)
        for param in optimizer.param_groups[0]["params"]
    }

    assert id(dummy_model["embedding"].weight) in decay_ids
    assert id(dummy_model["linear"].weight) in decay_ids


def test_bias_and_norm_parameters_do_not_receive_decay(
    dummy_model,
    optimizer,
):
    """Verify biases and LayerNorm parameters avoid weight decay."""

    no_decay_ids = {
        id(param)
        for param in optimizer.param_groups[1]["params"]
    }

    assert id(dummy_model["linear"].bias) in no_decay_ids
    assert id(dummy_model["norm"].weight) in no_decay_ids
    assert id(dummy_model["norm"].bias) in no_decay_ids


# ---------------------------------------------------------
# Frozen Parameter Tests
# ---------------------------------------------------------


def test_frozen_parameters_are_excluded(dummy_model, optimizer):
    """Verify frozen parameters are not included in optimizer."""

    optimizer_ids = {
        id(param)
        for group in optimizer.param_groups
        for param in group["params"]
    }

    for param in dummy_model["frozen"].parameters():
        assert id(param) not in optimizer_ids


# ---------------------------------------------------------
# Parameter Coverage Tests
# ---------------------------------------------------------


def test_all_trainable_parameters_are_included(
    dummy_model,
    optimizer,
):
    """Verify every trainable parameter is included exactly once."""

    expected_ids = {
        id(param)
        for param in dummy_model.parameters()
        if param.requires_grad
    }

    actual_ids = [
        id(param)
        for group in optimizer.param_groups
        for param in group["params"]
    ]

    assert set(actual_ids) == expected_ids
    assert len(actual_ids) == len(set(actual_ids))


def test_parameter_count(dummy_model, optimizer):
    """Verify optimizer parameter count matches model trainable count."""

    expected_count = sum(
        param.numel()
        for param in dummy_model.parameters()
        if param.requires_grad
    )

    actual_count = sum(
        param.numel()
        for group in optimizer.param_groups
        for param in group["params"]
    )

    assert actual_count == expected_count


# ---------------------------------------------------------
# Invalid Input Tests
# ---------------------------------------------------------


def test_invalid_model_type():
    """Verify non-PyTorch models raise TypeError."""

    with pytest.raises(TypeError, match="nn.Module"):
        create_adamw_optimizer(model="invalid")


@pytest.mark.parametrize("lr", [0, -0.1])
def test_invalid_learning_rate(lr, dummy_model):
    """Verify zero and negative learning rates are rejected."""

    with pytest.raises(ValueError, match="lr"):
        create_adamw_optimizer(
            model=dummy_model,
            lr=lr,
        )


def test_invalid_learning_rate_type(dummy_model):
    """Verify invalid learning rate types are rejected."""

    with pytest.raises(TypeError, match="lr"):
        create_adamw_optimizer(
            model=dummy_model,
            lr="0.001",
        )


@pytest.mark.parametrize(
    "betas",
    [
        (1.0, 0.999),
        (0.9, 1.0),
        (-0.1, 0.999),
        (0.9, -0.1),
        (0.9,),
        (0.9, 0.999, 0.99),
    ],
)
def test_invalid_betas(betas, dummy_model):
    """Verify invalid beta configurations are rejected."""

    with pytest.raises(ValueError):
        create_adamw_optimizer(
            model=dummy_model,
            betas=betas,
        )


def test_invalid_epsilon(dummy_model):
    """Verify zero epsilon is rejected."""

    with pytest.raises(ValueError, match="eps"):
        create_adamw_optimizer(
            model=dummy_model,
            eps=0,
        )


def test_invalid_weight_decay(dummy_model):
    """Verify negative weight decay is rejected."""

    with pytest.raises(ValueError, match="weight_decay"):
        create_adamw_optimizer(
            model=dummy_model,
            weight_decay=-0.01,
        )


# ---------------------------------------------------------
# Empty Model Tests
# ---------------------------------------------------------


def test_model_without_trainable_parameters():
    """Verify models without trainable parameters are rejected."""

    model = nn.Linear(4, 2)

    for param in model.parameters():
        param.requires_grad = False

    with pytest.raises(
        ValueError,
        match="no trainable parameters",
    ):
        create_adamw_optimizer(model=model)


# ---------------------------------------------------------
# Optimizer Update Tests
# ---------------------------------------------------------


def test_optimizer_updates_parameters():
    """Verify optimizer performs an actual parameter update."""

    model = nn.Linear(4, 2)

    optimizer = create_adamw_optimizer(
        model=model,
        lr=0.01,
        weight_decay=0.0,
    )

    initial_weight = model.weight.detach().clone()

    inputs = torch.randn(3, 4)
    targets = torch.randn(3, 2)

    optimizer.zero_grad()

    predictions = model(inputs)
    loss = nn.functional.mse_loss(predictions, targets)

    loss.backward()
    optimizer.step()

    assert not torch.equal(initial_weight, model.weight)


def test_weight_decay_applied_only_to_decay_group():
    """
    Verify AdamW decoupled weight decay is applied only
    to parameters belonging to the decay group.
    """

    model = nn.Module()
    model.weight = nn.Parameter(torch.ones(2, 2))
    model.bias = nn.Parameter(torch.ones(2))

    optimizer = create_adamw_optimizer(
        model=model,
        lr=0.1,
        weight_decay=0.1,
    )

    initial_weight = model.weight.detach().clone()
    initial_bias = model.bias.detach().clone()

    # Zero gradients isolate the decoupled weight decay effect.
    model.weight.grad = torch.zeros_like(model.weight)
    model.bias.grad = torch.zeros_like(model.bias)

    optimizer.step()

    expected_weight = initial_weight * (1 - 0.1 * 0.1)

    assert torch.allclose(
        model.weight,
        expected_weight,
    )

    assert torch.allclose(
        model.bias,
        initial_bias,
    )
