
"""
Unit tests for warmup + cosine learning rate scheduler.
"""

import math

import pytest
import torch
import torch.nn as nn

from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from src.training.scheduler import create_warmup_cosine_scheduler


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture
def optimizer():
    """Create a simple SGD optimizer for scheduler testing."""

    parameter = nn.Parameter(torch.tensor(1.0))

    return SGD(
        [parameter],
        lr=1e-3,
    )


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def collect_learning_rates(
    optimizer,
    scheduler,
    total_steps: int,
) -> list:
    """
    Collect learning rates across the complete schedule.

    Parameters:
        optimizer: Optimizer associated with the scheduler.
        scheduler: Learning rate scheduler.
        total_steps: Number of learning rate values to collect.

    Returns:
        List of learning rates across the schedule.
    """

    learning_rates = [
        optimizer.param_groups[0]["lr"]
    ]

    for _ in range(total_steps - 1):

        optimizer.step()
        scheduler.step()

        learning_rates.append(
            optimizer.param_groups[0]["lr"]
        )

    return learning_rates


# ---------------------------------------------------------
# Scheduler initialization tests
# ---------------------------------------------------------

def test_scheduler_instance(optimizer):
    """Verify that the function returns a LambdaLR scheduler."""

    scheduler = create_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_steps=100,
        warmup_steps=10,
    )

    assert isinstance(scheduler, LambdaLR)


def test_scheduler_initial_learning_rate(optimizer):
    """Verify that the initial LR follows the first warmup value."""

    base_lr = optimizer.param_groups[0]["lr"]
    warmup_steps = 4

    scheduler = create_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_steps=20,
        warmup_steps=warmup_steps,
    )

    expected_lr = base_lr / warmup_steps

    assert math.isclose(
        optimizer.param_groups[0]["lr"],
        expected_lr,
        rel_tol=1e-7,
    )


# ---------------------------------------------------------
# Linear warmup tests
# ---------------------------------------------------------

def test_linear_warmup(optimizer):
    """Verify that learning rate increases linearly during warmup."""

    base_lr = optimizer.param_groups[0]["lr"]
    total_steps = 10
    warmup_steps = 4

    scheduler = create_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )

    learning_rates = collect_learning_rates(
        optimizer,
        scheduler,
        total_steps,
    )

    expected_rates = [
        base_lr * (step + 1) / warmup_steps
        for step in range(warmup_steps)
    ]

    for actual, expected in zip(
        learning_rates[:warmup_steps],
        expected_rates,
    ):
        assert math.isclose(
            actual,
            expected,
            rel_tol=1e-7,
        )


def test_warmup_reaches_peak_learning_rate(optimizer):
    """Verify that warmup reaches the initial learning rate."""

    base_lr = optimizer.param_groups[0]["lr"]
    warmup_steps = 5
    total_steps = 20

    scheduler = create_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )

    learning_rates = collect_learning_rates(
        optimizer,
        scheduler,
        total_steps,
    )

    assert math.isclose(
        learning_rates[warmup_steps - 1],
        base_lr,
        rel_tol=1e-7,
    )


# ---------------------------------------------------------
# Cosine decay tests
# ---------------------------------------------------------

def test_cosine_decay(optimizer):
    """Verify that learning rate follows the cosine decay formula."""

    base_lr = optimizer.param_groups[0]["lr"]
    total_steps = 10
    warmup_steps = 2
    min_lr_ratio = 0.1

    scheduler = create_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
    )

    learning_rates = collect_learning_rates(
        optimizer,
        scheduler,
        total_steps,
    )

    decay_steps = total_steps - warmup_steps

    for step in range(warmup_steps, total_steps):

        progress = (
            step - warmup_steps + 1
        ) / decay_steps

        cosine_factor = 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

        expected_ratio = min_lr_ratio + (
            1.0 - min_lr_ratio
        ) * cosine_factor

        expected_lr = base_lr * expected_ratio

        assert math.isclose(
            learning_rates[step],
            expected_lr,
            rel_tol=1e-7,
        )


def test_learning_rate_decreases_during_cosine_decay(optimizer):
    """Verify that LR decreases monotonically during cosine decay."""

    total_steps = 20
    warmup_steps = 5

    scheduler = create_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )

    learning_rates = collect_learning_rates(
        optimizer,
        scheduler,
        total_steps,
    )

    decay_rates = learning_rates[warmup_steps - 1:]

    for previous_lr, current_lr in zip(
        decay_rates,
        decay_rates[1:],
    ):
        assert current_lr <= previous_lr


# ---------------------------------------------------------
# Minimum learning rate tests
# ---------------------------------------------------------

def test_final_learning_rate_reaches_minimum(optimizer):
    """Verify that final LR equals the configured minimum LR."""

    base_lr = optimizer.param_groups[0]["lr"]
    total_steps = 100
    warmup_steps = 10
    min_lr_ratio = 0.1

    scheduler = create_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
    )

    learning_rates = collect_learning_rates(
        optimizer,
        scheduler,
        total_steps,
    )

    expected_min_lr = base_lr * min_lr_ratio

    assert math.isclose(
        learning_rates[-1],
        expected_min_lr,
        rel_tol=1e-7,
    )


def test_zero_minimum_learning_rate_ratio(optimizer):
    """Verify that scheduler can decay LR to zero."""

    total_steps = 10
    warmup_steps = 2

    scheduler = create_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=0.0,
    )

    learning_rates = collect_learning_rates(
        optimizer,
        scheduler,
        total_steps,
    )

    assert learning_rates[-1] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------

def test_scheduler_without_warmup(optimizer):
    """Verify cosine decay works when warmup is disabled."""

    base_lr = optimizer.param_groups[0]["lr"]
    total_steps = 10
    min_lr_ratio = 0.1

    scheduler = create_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=0,
        min_lr_ratio=min_lr_ratio,
    )

    learning_rates = collect_learning_rates(
        optimizer,
        scheduler,
        total_steps,
    )

    assert math.isclose(
        learning_rates[0],
        base_lr,
        rel_tol=1e-7,
    )

    assert math.isclose(
        learning_rates[-1],
        base_lr * min_lr_ratio,
        rel_tol=1e-7,
    )


# ---------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------

def test_invalid_optimizer():
    """Verify that invalid optimizer types raise TypeError."""

    with pytest.raises(TypeError):
        create_warmup_cosine_scheduler(
            optimizer=object(),
            total_steps=100,
            warmup_steps=10,
        )


@pytest.mark.parametrize(
    "total_steps",
    [True, False, 10.5, "100"],
)
def test_invalid_total_steps_type(optimizer, total_steps):
    """Verify that invalid total_steps types raise TypeError."""

    with pytest.raises(TypeError):
        create_warmup_cosine_scheduler(
            optimizer=optimizer,
            total_steps=total_steps,
            warmup_steps=10,
        )


@pytest.mark.parametrize(
    "total_steps",
    [0, 1, -10],
)
def test_invalid_total_steps_value(optimizer, total_steps):
    """Verify that invalid total_steps values raise ValueError."""

    with pytest.raises(ValueError):
        create_warmup_cosine_scheduler(
            optimizer=optimizer,
            total_steps=total_steps,
            warmup_steps=0,
        )


@pytest.mark.parametrize(
    "warmup_steps",
    [True, False, 10.5, "10"],
)
def test_invalid_warmup_steps_type(optimizer, warmup_steps):
    """Verify that invalid warmup_steps types raise TypeError."""

    with pytest.raises(TypeError):
        create_warmup_cosine_scheduler(
            optimizer=optimizer,
            total_steps=100,
            warmup_steps=warmup_steps,
        )


@pytest.mark.parametrize(
    "warmup_steps",
    [-1, 100],
)
def test_invalid_warmup_steps_value(optimizer, warmup_steps):
    """Verify that invalid warmup_steps values raise ValueError."""

    with pytest.raises(ValueError):
        create_warmup_cosine_scheduler(
            optimizer=optimizer,
            total_steps=100,
            warmup_steps=warmup_steps,
        )


@pytest.mark.parametrize(
    "min_lr_ratio",
    [True, False, "0.1", None],
)
def test_invalid_min_lr_ratio_type(optimizer, min_lr_ratio):
    """Verify that invalid min_lr_ratio types raise TypeError."""

    with pytest.raises(TypeError):
        create_warmup_cosine_scheduler(
            optimizer=optimizer,
            total_steps=100,
            warmup_steps=10,
            min_lr_ratio=min_lr_ratio,
        )


@pytest.mark.parametrize(
    "min_lr_ratio",
    [-0.1, 1.1, float("nan"), float("inf"), float("-inf")],
)
def test_invalid_min_lr_ratio_value(optimizer, min_lr_ratio):
    """Verify that invalid min_lr_ratio values raise ValueError."""

    with pytest.raises(ValueError):
        create_warmup_cosine_scheduler(
            optimizer=optimizer,
            total_steps=100,
            warmup_steps=10,
            min_lr_ratio=min_lr_ratio,
        )
