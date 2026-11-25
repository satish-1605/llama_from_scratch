
"""Unit tests for causal language-model loss."""

import pytest
import torch

from src.model.loss import CausalLMLoss


def test_loss_output_is_scalar():
    """Verify that the loss returns a scalar tensor."""
    loss_fn = CausalLMLoss()

    logits = torch.randn(2, 512, 32000)
    labels = torch.randint(0, 32000, (2, 512))

    loss = loss_fn(logits, labels)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_loss_gradient_flow():
    """Verify that gradients flow back through the logits."""
    loss_fn = CausalLMLoss()

    logits = torch.randn(
        2, 8, 100,
        requires_grad=True,
    )
    labels = torch.randint(0, 100, (2, 8))

    loss = loss_fn(logits, labels)
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_loss_ignores_masked_labels():
    """Verify that labels with ignore_index do not contribute to loss."""
    loss_fn = CausalLMLoss(ignore_index=-100)

    logits = torch.randn(2, 4, 20)

    labels = torch.tensor([
        [1, 2, -100, -100],
        [3, 4, -100, -100],
    ])

    loss = loss_fn(logits, labels)

    expected_loss = torch.nn.functional.cross_entropy(
        logits[:, :2, :].reshape(-1, 20),
        labels[:, :2].reshape(-1),
    )

    assert torch.allclose(loss, expected_loss)


def test_invalid_logits_dimensions():
    """Verify that invalid logits dimensions raise an error."""
    loss_fn = CausalLMLoss()

    logits = torch.randn(2, 100)
    labels = torch.randint(0, 100, (2,))

    with pytest.raises(ValueError):
        loss_fn(logits, labels)


def test_invalid_labels_dimensions():
    """Verify that invalid label dimensions raise an error."""
    loss_fn = CausalLMLoss()

    logits = torch.randn(2, 8, 100)
    labels = torch.randint(0, 100, (2, 8, 1))

    with pytest.raises(ValueError):
        loss_fn(logits, labels)


def test_mismatched_sequence_dimensions():
    """Verify that mismatched batch or sequence dimensions raise an error."""
    loss_fn = CausalLMLoss()

    logits = torch.randn(2, 8, 100)
    labels = torch.randint(0, 100, (2, 7))

    with pytest.raises(ValueError):
        loss_fn(logits, labels)


def test_invalid_label_dtype():
    """Verify that non-integer target labels raise an error."""
    loss_fn = CausalLMLoss()

    logits = torch.randn(2, 8, 100)
    labels = torch.randn(2, 8)

    with pytest.raises(ValueError):
        loss_fn(logits, labels)
