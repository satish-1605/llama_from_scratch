
"""Integration test for packed dataset and causal LM loss."""

import torch
from datasets import load_from_disk

from src.model.loss import CausalLMLoss


def test_packed_dataset_to_causal_lm_loss():
    """Verify that saved packed data works with the causal LM loss."""

    # Load the existing processed training dataset.
    dataset = load_from_disk("artifacts/processed/pretraining/train")

    # Select one sample to keep memory usage manageable.
    sample = dataset[0]

    input_ids = torch.tensor(
        [sample["input_ids"]],
        dtype=torch.long,
    )

    labels = torch.tensor(
        [sample["labels"]],
        dtype=torch.long,
    )

    # Model configuration.
    vocab_size = 32000
    context_length = 512

    # Verify packed dataset dimensions.
    assert input_ids.shape == (1, context_length)
    assert labels.shape == (1, context_length)

    # Verify target token IDs are within vocabulary range.
    assert labels.min() >= 0
    assert labels.max() < vocab_size

    # Simulate model output logits.
    # These are dummy logits, not predictions from a trained model.
    logits = torch.randn(
        1,
        context_length,
        vocab_size,
        requires_grad=True,
    )

    # Calculate causal LM loss.
    loss_fn = CausalLMLoss()
    loss = loss_fn(logits, labels)

    # Verify loss.
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    # Verify backward pass.
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0

    print("Causal LM loss integration test passed!")
    print(f"Input shape: {tuple(input_ids.shape)}")
    print(f"Labels shape: {tuple(labels.shape)}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Loss: {loss.item():.4f}")
