
import pytest
import torch

from src.model.normalization import RMSNorm


def test_output_shape():
    norm = RMSNorm(dim=384)
    x = torch.randn(2, 512, 384)

    output = norm(x)

    assert output.shape == x.shape


def test_rmsnorm_numerical_correctness():
    dim = 4
    eps = 1e-6

    norm = RMSNorm(dim=dim, eps=eps)

    x = torch.tensor([[2.0, 4.0, 6.0, 8.0]])

    expected_rms = torch.sqrt(
        x.pow(2).mean(dim=-1, keepdim=True) + eps
    )

    expected = x / expected_rms

    output = norm(x)

    assert torch.allclose(output, expected, atol=1e-6)


def test_learnable_weight():
    norm = RMSNorm(dim=4, eps = 1e-6)

    assert isinstance(norm.weight, torch.nn.Parameter)
    assert norm.weight.requires_grad

    assert torch.allclose(
        norm.weight,
        torch.ones(4)
    )


def test_gradient_flow():
    norm = RMSNorm(dim=4, eps = 1e-6)

    x = torch.randn(2, 4, requires_grad=True)

    output = norm(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert norm.weight.grad is not None


def test_invalid_dimension():
    with pytest.raises(ValueError):
        RMSNorm(dim=0)


def test_mismatched_input_dimension():
    norm = RMSNorm(dim=4, eps = 1e-6)
    x = torch.randn(2, 8)

    with pytest.raises(ValueError):
        norm(x)


def test_zero_input_stability():
    norm = RMSNorm(dim=4, eps = 1e-6)

    x = torch.zeros(2, 4)
    output = norm(x)

    assert torch.isfinite(output).all()
    assert torch.allclose(output, torch.zeros_like(output))


def test_bfloat16_dtype_preservation():
    norm = RMSNorm(dim=4, eps = 1e-6)
    x = torch.randn(2, 4, dtype=torch.bfloat16)

    output = norm(x)

    assert output.dtype == torch.bfloat16
