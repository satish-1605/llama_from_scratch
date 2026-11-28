import pytest
import torch
import torch.nn as nn

from src.models.swiglu import SwiGLU


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture
def swiglu():
    """
    Create a standard SwiGLU module for testing.

    Returns:
        SwiGLU: Module with d_model=8 and hidden_dim=16.
    """
    return SwiGLU(
        d_model=8,
        hidden_dim=16,
    )


@pytest.fixture
def input_tensor():
    """
    Generate a sample input tensor.

    Returns:
        torch.Tensor: Tensor with shape [2, 4, 8].
    """
    torch.manual_seed(42)

    return torch.randn(
        2, 4, 8
    )


# ---------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------

def test_swiglu_initialization(swiglu):
    """Verify SwiGLU initializes with correct dimensions."""

    assert swiglu.d_model == 8
    assert swiglu.hidden_dim == 16

    assert isinstance(swiglu.w1, nn.Linear)
    assert isinstance(swiglu.w3, nn.Linear)
    assert isinstance(swiglu.w2, nn.Linear)

    assert isinstance(swiglu.act, nn.SiLU)

    assert swiglu.w1.in_features == 8
    assert swiglu.w1.out_features == 16

    assert swiglu.w3.in_features == 8
    assert swiglu.w3.out_features == 16

    assert swiglu.w2.in_features == 16
    assert swiglu.w2.out_features == 8


def test_swiglu_linear_layers_have_no_bias(swiglu):
    """Verify all three projections are bias-free."""

    assert swiglu.w1.bias is None
    assert swiglu.w3.bias is None
    assert swiglu.w2.bias is None


# ---------------------------------------------------------
# Forward Shape Tests
# ---------------------------------------------------------

def test_swiglu_output_shape(swiglu, input_tensor):
    """Verify output shape matches input shape."""

    output = swiglu(input_tensor)

    assert output.shape == input_tensor.shape


@pytest.mark.parametrize(
    "batch_size, seq_len, d_model, hidden_dim",
    [
        (1, 1, 8, 16),
        (2, 4, 8, 16),
        (4, 8, 32, 64),
        (2, 16, 64, 128),
        (1, 10, 128, 256),
    ],
)
def test_swiglu_different_dimensions(
    batch_size,
    seq_len,
    d_model,
    hidden_dim,
):
    """Verify SwiGLU works with different configurations."""

    model = SwiGLU(
        d_model=d_model,
        hidden_dim=hidden_dim,
    )

    x = torch.randn(
        batch_size,
        seq_len,
        d_model,
    )

    output = model(x)

    assert output.shape == (
        batch_size,
        seq_len,
        d_model,
    )


# ---------------------------------------------------------
# Mathematical Correctness Tests
# ---------------------------------------------------------

def test_swiglu_mathematical_correctness(
    swiglu,
    input_tensor,
):
    """
    Verify implementation against the mathematical formula:

        output = W2(SiLU(W1(x)) * W3(x))
    """

    with torch.no_grad():

        gate = torch.nn.functional.silu(
            swiglu.w1(input_tensor)
        )

        value = swiglu.w3(input_tensor)

        hidden = gate * value

        expected_output = swiglu.w2(hidden)

        actual_output = swiglu(input_tensor)

    assert torch.allclose(
        actual_output,
        expected_output,
        atol=1e-6,
        rtol=1e-5,
    )


def test_swiglu_gating_uses_elementwise_multiplication(
    swiglu,
    input_tensor,
):
    """Verify gating produces the expected hidden dimensions."""

    gate = swiglu.act(swiglu.w1(input_tensor))
    value = swiglu.w3(input_tensor)

    hidden = gate * value

    assert gate.shape == (2, 4, 16)
    assert value.shape == (2, 4, 16)
    assert hidden.shape == (2, 4, 16)


# ---------------------------------------------------------
# Parameter Count Tests
# ---------------------------------------------------------

def test_swiglu_parameter_count(swiglu):
    """Verify total trainable parameter count."""

    expected_parameters = 3 * 8 * 16

    assert swiglu.count_parameters() == expected_parameters


def test_swiglu_parameter_count_for_different_dimensions():
    """Verify parameter count with another configuration."""

    model = SwiGLU(
        d_model=32,
        hidden_dim=64,
    )

    expected_parameters = 3 * 32 * 64

    assert model.count_parameters() == expected_parameters


# ---------------------------------------------------------
# Gradient and Backpropagation Tests
# ---------------------------------------------------------

def test_swiglu_gradient_flow(swiglu, input_tensor):
    """Verify gradients flow through input and all projections."""

    x = input_tensor.clone().requires_grad_(True)

    output = swiglu(x)

    loss = output.pow(2).mean()

    loss.backward()

    # Input gradient must exist.
    assert x.grad is not None

    # Input gradient must be finite.
    assert torch.isfinite(x.grad).all()

    # All projection weights must receive gradients.
    assert swiglu.w1.weight.grad is not None
    assert swiglu.w3.weight.grad is not None
    assert swiglu.w2.weight.grad is not None

    assert torch.isfinite(swiglu.w1.weight.grad).all()
    assert torch.isfinite(swiglu.w3.weight.grad).all()
    assert torch.isfinite(swiglu.w2.weight.grad).all()


def test_swiglu_gradients_are_nonzero(swiglu, input_tensor):
    """Verify nonzero gradients for all projection weights."""

    output = swiglu(input_tensor)

    loss = output.pow(2).mean()

    loss.backward()

    assert torch.count_nonzero(
        swiglu.w1.weight.grad
    ) > 0

    assert torch.count_nonzero(
        swiglu.w3.weight.grad
    ) > 0

    assert torch.count_nonzero(
        swiglu.w2.weight.grad
    ) > 0


# ---------------------------------------------------------
# Constructor Validation Tests
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "d_model, hidden_dim",
    [
        (0, 16),
        (8, 0),
        (-8, 16),
        (8, -16),
    ],
)
def test_swiglu_invalid_dimensions(d_model, hidden_dim):
    """Verify non-positive dimensions raise ValueError."""

    with pytest.raises(ValueError):
        SwiGLU(
            d_model=d_model,
            hidden_dim=hidden_dim,
        )


@pytest.mark.parametrize(
    "d_model, hidden_dim",
    [
        (8.5, 16),
        (8, 16.5),
        ("8", 16),
        (8, "16"),
        (True, 16),
    ],
)
def test_swiglu_invalid_dimension_types(d_model, hidden_dim):
    """Verify non-integer dimensions raise TypeError."""

    with pytest.raises(TypeError):
        SwiGLU(
            d_model=d_model,
            hidden_dim=hidden_dim,
        )


# ---------------------------------------------------------
# Input Validation Tests
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "input_shape",
    [
        (8,),
        (4, 8),
        (2, 4, 3, 8),
    ],
)
def test_swiglu_invalid_input_rank(swiglu, input_shape):
    """Verify tensors with incorrect rank raise ValueError."""

    x = torch.randn(*input_shape)

    with pytest.raises(ValueError, match="3D input tensor"):
        swiglu(x)


def test_swiglu_invalid_input_feature_dimension(swiglu):
    """Verify incorrect input feature dimension raises ValueError."""

    x = torch.randn(2, 4, 10)

    with pytest.raises(ValueError, match="feature dimension"):
        swiglu(x)


# ---------------------------------------------------------
# Batch and Sequence Variation Tests
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "batch_size, seq_len",
    [
        (1, 1),
        (1, 32),
        (2, 128),
        (4, 256),
    ],
)
def test_swiglu_variable_batch_and_sequence(
    swiglu,
    batch_size,
    seq_len,
):
    """Verify varying batch sizes and sequence lengths."""

    x = torch.randn(
        batch_size,
        seq_len,
        swiglu.d_model,
    )

    output = swiglu(x)

    assert output.shape == (
        batch_size,
        seq_len,
        swiglu.d_model,
    )

    assert torch.isfinite(output).all()
