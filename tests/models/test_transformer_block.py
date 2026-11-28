
import pytest
import torch
import torch.nn as nn

from src.models.transformer_block import TransformerBlock
from src.models.swiglu import SwiGLU


# ---------------------------------------------------------
# Test Modules
# ---------------------------------------------------------

class DummyNorm(nn.Module):
    """Simple normalization substitute for block unit tests."""

    def forward(self, x):
        return x


class DummyAttention(nn.Module):
    """Simple attention substitute preserving [B, T, D]."""

    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        return self.proj(x)


class DummyFeedForward(nn.Module):
    """Simple FFN substitute preserving [B, T, D]."""

    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        return self.proj(x)


class WrongShapeModule(nn.Module):
    """Module that intentionally returns an invalid shape."""

    def forward(self, x):
        return x[..., :-1]


# ---------------------------------------------------------
# Fixtures
# ---------------------------------------------------------

@pytest.fixture
def d_model():
    return 128


@pytest.fixture
def block(d_model):
    return TransformerBlock(
        d_model=d_model,
        attention=DummyAttention(d_model),
        feed_forward=DummyFeedForward(d_model),
        norm1=DummyNorm(),
        norm2=DummyNorm(),
    )


@pytest.fixture
def input_tensor(d_model):
    return torch.randn(2, 16, d_model)


# ---------------------------------------------------------
# 1. Initialization Tests
# ---------------------------------------------------------

def test_transformer_block_initialization(block, d_model):
    assert isinstance(block, TransformerBlock)
    assert block.d_model == d_model


def test_attention_module_is_registered(block):
    assert isinstance(block.attention, nn.Module)


def test_feed_forward_module_is_registered(block):
    assert isinstance(block.feed_forward, nn.Module)


def test_norm_modules_are_registered(block):
    assert isinstance(block.norm1, nn.Module)
    assert isinstance(block.norm2, nn.Module)


def test_block_parameters_are_registered(block):
    parameter_names = dict(block.named_parameters())

    assert any(name.startswith("attention.") for name in parameter_names)
    assert any(name.startswith("feed_forward.") for name in parameter_names)


# ---------------------------------------------------------
# 2. Forward Shape Tests
# ---------------------------------------------------------

def test_forward_output_shape(block, input_tensor):
    output = block(input_tensor)

    assert output.shape == input_tensor.shape


@pytest.mark.parametrize(
    "batch_size, sequence_length",
    [
        (1, 1),
        (1, 8),
        (2, 16),
        (4, 32),
    ],
)
def test_forward_supports_variable_batch_and_sequence(
    block,
    d_model,
    batch_size,
    sequence_length,
):
    x = torch.randn(batch_size, sequence_length, d_model)

    output = block(x)

    assert output.shape == (batch_size, sequence_length, d_model)


# ---------------------------------------------------------
# 3. Mathematical Correctness Tests
# ---------------------------------------------------------

def test_pre_norm_residual_computation(block, input_tensor):
    """
    Verify:
        x1 = x + attention(norm1(x))
        output = x1 + feed_forward(norm2(x1))
    """

    x = input_tensor

    expected_x = x + block.attention(block.norm1(x))
    expected_output = (
        expected_x
        + block.feed_forward(block.norm2(expected_x))
    )

    actual_output = block(x)

    assert torch.allclose(
        actual_output,
        expected_output,
        atol=1e-6,
    )


def test_attention_residual_is_preserved():
    """
    If attention and FFN return zeros, output must equal input.
    """

    class ZeroModule(nn.Module):
        def forward(self, x):
            return torch.zeros_like(x)

    d_model = 32

    block = TransformerBlock(
        d_model=d_model,
        attention=ZeroModule(),
        feed_forward=ZeroModule(),
        norm1=DummyNorm(),
        norm2=DummyNorm(),
    )

    x = torch.randn(2, 5, d_model)

    output = block(x)

    assert torch.allclose(output, x)


# ---------------------------------------------------------
# 4. Gradient Flow Tests
# ---------------------------------------------------------

def test_input_gradient_is_computed(block, input_tensor):
    x = input_tensor.clone().requires_grad_(True)

    output = block(x)
    loss = output.square().mean()

    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad) > 0


def test_attention_parameters_receive_gradients(block, input_tensor):
    output = block(input_tensor)
    loss = output.mean()

    loss.backward()

    for parameter in block.attention.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_feed_forward_parameters_receive_gradients(block, input_tensor):
    output = block(input_tensor)
    loss = output.mean()

    loss.backward()

    for parameter in block.feed_forward.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


# ---------------------------------------------------------
# 5. Invalid Input Tests
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        (16, 128),
        (2, 16, 128, 1),
        (128,),
    ],
)
def test_invalid_input_rank_raises_error(block, shape):
    x = torch.randn(*shape)

    with pytest.raises(ValueError):
        block(x)


def test_invalid_model_dimension_raises_error():
    with pytest.raises(ValueError):
        TransformerBlock(
            d_model=0,
            attention=DummyAttention(32),
            feed_forward=DummyFeedForward(32),
            norm1=DummyNorm(),
            norm2=DummyNorm(),
        )


def test_invalid_d_model_type_raises_error():
    with pytest.raises(TypeError):
        TransformerBlock(
            d_model=32.5,
            attention=DummyAttention(32),
            feed_forward=DummyFeedForward(32),
            norm1=DummyNorm(),
            norm2=DummyNorm(),
        )




# ---------------------------------------------------------
# 6. Output Contract Tests
# ---------------------------------------------------------

def test_invalid_attention_output_shape_raises_error():
    d_model = 32

    block = TransformerBlock(
        d_model=d_model,
        attention=WrongShapeModule(),
        feed_forward=DummyFeedForward(d_model),
        norm1=DummyNorm(),
        norm2=DummyNorm(),
    )

    x = torch.randn(2, 5, d_model)

    with pytest.raises(ValueError, match="Attention output shape"):
        block(x)



# ---------------------------------------------------------
# 7. SwiGLU Integration Test
# ---------------------------------------------------------

def test_transformer_block_with_real_swiglu(input_tensor, d_model):
    """
    Verify that the actual project SwiGLU module can be
    integrated into TransformerBlock.
    """

    block = TransformerBlock(
        d_model=d_model,
        attention=DummyAttention(d_model),
        feed_forward=SwiGLU(
            d_model=d_model,
            hidden_dim=256,
        ),
        norm1=DummyNorm(),
        norm2=DummyNorm(),
    )

    output = block(input_tensor)

    assert output.shape == input_tensor.shape
    assert torch.isfinite(output).all()


def test_real_swiglu_receives_gradients(input_tensor, d_model):
    block = TransformerBlock(
        d_model=d_model,
        attention=DummyAttention(d_model),
        feed_forward=SwiGLU(
            d_model=d_model,
            hidden_dim=256,
        ),
        norm1=DummyNorm(),
        norm2=DummyNorm(),
    )

    output = block(input_tensor)
    output.square().mean().backward()

    for parameter in block.feed_forward.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
