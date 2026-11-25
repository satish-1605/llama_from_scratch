
"""
Unit tests for Rotary Positional Embeddings (RoPE).

Tests cover:
    1. rotate_half()
    2. RotaryEmbedding initialization
    3. Inverse frequency calculation
    4. Cosine and sine precomputation
    5. Position offset handling
    6. Query and Key rotation
    7. Magnitude preservation
    8. GQA compatibility
    9. Input validation
    10. Gradient propagation
"""

import pytest
import torch

from src.models.rope import (
    rotate_half,
    RotaryEmbedding,
    apply_rotary_pos_emb,
)


# ---------------------------------------------------------
# 1. rotate_half() tests
# ---------------------------------------------------------

def test_rotate_half_correctness():
    """Verify split-half rotation produces the expected output."""

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])

    result = rotate_half(x)

    assert torch.equal(result, expected)


def test_rotate_half_preserves_shape():
    """Verify rotate_half preserves the input tensor shape."""

    x = torch.randn(2, 5, 4, 64)

    result = rotate_half(x)

    assert result.shape == x.shape


def test_rotate_half_odd_dimension_raises_error():
    """Verify odd dimensions are rejected."""

    x = torch.randn(2, 5, 3)

    with pytest.raises(ValueError, match="even head dimension"):
        rotate_half(x)


# ---------------------------------------------------------
# 2. RotaryEmbedding initialization tests
# ---------------------------------------------------------

def test_rotary_embedding_initialization():
    """Verify RotaryEmbedding initializes correctly."""

    rope = RotaryEmbedding(head_dim=64)

    assert rope.head_dim == 64
    assert rope.base == 10000.0
    assert rope.inv_freq.shape == (32,)


@pytest.mark.parametrize(
    "head_dim",
    [0, -2, 7],
)
def test_invalid_head_dimension_raises_error(head_dim):
    """Verify invalid head dimensions are rejected."""

    with pytest.raises(ValueError):
        RotaryEmbedding(head_dim=head_dim)


def test_invalid_base_raises_error():
    """Verify non-positive frequency bases are rejected."""

    with pytest.raises(ValueError, match="base must be positive"):
        RotaryEmbedding(head_dim=64, base=0)


# ---------------------------------------------------------
# 3. Inverse frequency tests
# ---------------------------------------------------------

def test_inverse_frequency_calculation():
    """Verify inverse frequencies match the mathematical formula."""

    head_dim = 8
    base = 10000.0

    rope = RotaryEmbedding(
        head_dim=head_dim,
        base=base,
    )

    indices = torch.arange(0, head_dim, 2).float()

    expected = 1.0 / (
        base ** (indices / head_dim)
    )

    assert torch.allclose(
        rope.inv_freq,
        expected,
        atol=1e-7,
    )


def test_inverse_frequency_values():
    """Verify known inverse frequency values."""

    rope = RotaryEmbedding(head_dim=8)

    expected = torch.tensor([
        1.0,
        0.1,
        0.01,
        0.001,
    ])

    assert torch.allclose(
        rope.inv_freq,
        expected,
        atol=1e-7,
    )


# ---------------------------------------------------------
# 4. Cosine and sine precomputation tests
# ---------------------------------------------------------

def test_cos_sin_output_shapes():
    """Verify cosine and sine tensor shapes."""

    rope = RotaryEmbedding(head_dim=8)

    cos, sin = rope(seq_len=10)

    expected_shape = (1, 10, 1, 8)

    assert cos.shape == expected_shape
    assert sin.shape == expected_shape


def test_cos_sin_values():
    """Verify cosine and sine values at position zero."""

    rope = RotaryEmbedding(head_dim=8)

    cos, sin = rope(seq_len=4)

    # At position zero, all rotation angles are zero.
    assert torch.allclose(
        cos[:, 0],
        torch.ones_like(cos[:, 0]),
    )

    assert torch.allclose(
        sin[:, 0],
        torch.zeros_like(sin[:, 0]),
    )


def test_cos_sin_numerical_correctness():
    """Verify precomputed values against direct calculations."""

    rope = RotaryEmbedding(head_dim=8)

    cos, sin = rope(seq_len=4)

    positions = torch.arange(4, dtype=torch.float32)

    expected_freqs = torch.outer(
        positions,
        rope.inv_freq,
    )

    expected_emb = torch.cat(
        (expected_freqs, expected_freqs),
        dim=-1,
    )

    expected_cos = torch.cos(expected_emb)
    expected_sin = torch.sin(expected_emb)

    expected_cos = expected_cos.unsqueeze(0).unsqueeze(2)
    expected_sin = expected_sin.unsqueeze(0).unsqueeze(2)

    assert torch.allclose(cos, expected_cos)
    assert torch.allclose(sin, expected_sin)


# ---------------------------------------------------------
# 5. Position offset tests
# ---------------------------------------------------------

def test_position_offset():
    """Verify offset generates the correct position-dependent values."""

    rope = RotaryEmbedding(head_dim=8)

    cos_full, sin_full = rope(seq_len=8)

    cos_offset, sin_offset = rope(
        seq_len=4,
        position_offset=4,
    )

    assert torch.allclose(
        cos_offset,
        cos_full[:, 4:8],
    )

    assert torch.allclose(
        sin_offset,
        sin_full[:, 4:8],
    )


@pytest.mark.parametrize(
    "seq_len, position_offset",
    [
        (0, 0),
        (-1, 0),
        (4, -1),
    ],
)
def test_invalid_sequence_arguments(
    seq_len,
    position_offset,
):
    """Verify invalid sequence arguments raise errors."""

    rope = RotaryEmbedding(head_dim=8)

    with pytest.raises(ValueError):
        rope(
            seq_len=seq_len,
            position_offset=position_offset,
        )


# ---------------------------------------------------------
# 6. Dtype tests
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ],
)
def test_cos_sin_dtype(dtype):
    """Verify cosine and sine outputs use the requested dtype."""

    rope = RotaryEmbedding(head_dim=8)

    cos, sin = rope(
        seq_len=4,
        dtype=dtype,
    )

    assert cos.dtype == dtype
    assert sin.dtype == dtype


# ---------------------------------------------------------
# 7. Query and Key rotation tests
# ---------------------------------------------------------

def test_apply_rotary_pos_emb_output_shapes():
    """Verify rotated Query and Key preserve their shapes."""

    batch_size = 2
    seq_len = 10
    num_heads = 4
    head_dim = 64

    rope = RotaryEmbedding(head_dim=head_dim)

    q = torch.randn(
        batch_size,
        seq_len,
        num_heads,
        head_dim,
    )

    k = torch.randn_like(q)

    cos, sin = rope(seq_len=seq_len)

    q_rotated, k_rotated = apply_rotary_pos_emb(
        q,
        k,
        cos,
        sin,
    )

    assert q_rotated.shape == q.shape
    assert k_rotated.shape == k.shape


def test_apply_rotary_pos_emb_changes_vectors():
    """Verify RoPE changes vectors at non-zero positions."""

    rope = RotaryEmbedding(head_dim=8)

    q = torch.randn(2, 4, 3, 8)
    k = torch.randn(2, 4, 3, 8)

    cos, sin = rope(seq_len=4)

    q_rotated, k_rotated = apply_rotary_pos_emb(
        q,
        k,
        cos,
        sin,
    )

    assert not torch.allclose(q_rotated[:, 1:], q[:, 1:])
    assert not torch.allclose(k_rotated[:, 1:], k[:, 1:])


# ---------------------------------------------------------
# 8. Rotation magnitude preservation
# ---------------------------------------------------------

def test_rotation_preserves_vector_norm():
    """
    Verify RoPE preserves the L2 norm of each rotated vector.

    Rotation changes direction, not magnitude.
    """

    rope = RotaryEmbedding(head_dim=8)

    q = torch.randn(2, 4, 3, 8)
    k = torch.randn(2, 4, 3, 8)

    cos, sin = rope(seq_len=4)

    q_rotated, k_rotated = apply_rotary_pos_emb(
        q,
        k,
        cos,
        sin,
    )

    q_norm = torch.linalg.vector_norm(q, dim=-1)
    q_rotated_norm = torch.linalg.vector_norm(
        q_rotated,
        dim=-1,
    )

    k_norm = torch.linalg.vector_norm(k, dim=-1)
    k_rotated_norm = torch.linalg.vector_norm(
        k_rotated,
        dim=-1,
    )

    assert torch.allclose(
        q_norm,
        q_rotated_norm,
        atol=1e-5,
    )

    assert torch.allclose(
        k_norm,
        k_rotated_norm,
        atol=1e-5,
    )


# ---------------------------------------------------------
# 9. GQA compatibility
# ---------------------------------------------------------

def test_gqa_compatibility():
    """
    Verify RoPE supports different Query and Key head counts.

    Example:
        Query heads = 6
        KV heads = 2
    """

    batch_size = 2
    seq_len = 16
    head_dim = 64

    rope = RotaryEmbedding(head_dim=head_dim)

    q = torch.randn(
        batch_size,
        seq_len,
        6,
        head_dim,
    )

    k = torch.randn(
        batch_size,
        seq_len,
        2,
        head_dim,
    )

    cos, sin = rope(seq_len=seq_len)

    q_rotated, k_rotated = apply_rotary_pos_emb(
        q,
        k,
        cos,
        sin,
    )

    assert q_rotated.shape == q.shape
    assert k_rotated.shape == k.shape


# ---------------------------------------------------------
# 10. Invalid input tests
# ---------------------------------------------------------

def test_invalid_query_key_dimensions():
    """Verify Query and Key must be four-dimensional."""

    rope = RotaryEmbedding(head_dim=8)

    q = torch.randn(2, 4, 8)
    k = torch.randn(2, 4, 8)

    cos, sin = rope(seq_len=4)

    with pytest.raises(ValueError, match="4 dimensions"):
        apply_rotary_pos_emb(q, k, cos, sin)


def test_query_key_batch_mismatch():
    """Verify Query and Key batch sizes must match."""

    rope = RotaryEmbedding(head_dim=8)

    q = torch.randn(2, 4, 3, 8)
    k = torch.randn(3, 4, 2, 8)

    cos, sin = rope(seq_len=4)

    with pytest.raises(ValueError, match="batch sizes"):
        apply_rotary_pos_emb(q, k, cos, sin)


def test_query_key_sequence_mismatch():
    """Verify Query and Key sequence lengths must match."""

    rope = RotaryEmbedding(head_dim=8)

    q = torch.randn(2, 4, 3, 8)
    k = torch.randn(2, 5, 2, 8)

    cos, sin = rope(seq_len=4)

    with pytest.raises(ValueError, match="sequence lengths"):
        apply_rotary_pos_emb(q, k, cos, sin)


def test_query_key_head_dimension_mismatch():
    """Verify Query and Key head dimensions must match."""

    rope = RotaryEmbedding(head_dim=8)

    q = torch.randn(2, 4, 3, 8)
    k = torch.randn(2, 4, 2, 16)

    cos, sin = rope(seq_len=4)

    with pytest.raises(ValueError, match="head dimensions"):
        apply_rotary_pos_emb(q, k, cos, sin)


def test_invalid_cos_shape():
    """Verify incompatible cosine shapes are rejected."""

    rope = RotaryEmbedding(head_dim=8)

    q = torch.randn(2, 4, 3, 8)
    k = torch.randn(2, 4, 2, 8)

    cos, sin = rope(seq_len=4)

    invalid_cos = cos.squeeze(2)

    with pytest.raises(ValueError, match="Invalid cos shape"):
        apply_rotary_pos_emb(q, k, invalid_cos, sin)


def test_query_key_dtype_mismatch():
    """Verify Query and Key must have the same dtype."""

    rope = RotaryEmbedding(head_dim=8)

    q = torch.randn(2, 4, 3, 8, dtype=torch.float32)
    k = torch.randn(2, 4, 2, 8, dtype=torch.float16)

    cos, sin = rope(seq_len=4)

    with pytest.raises(ValueError, match="same dtype"):
        apply_rotary_pos_emb(q, k, cos, sin)


# ---------------------------------------------------------
# 11. Gradient propagation
# ---------------------------------------------------------

def test_gradient_propagation():
    """Verify gradients propagate through RoPE to Query and Key."""

    rope = RotaryEmbedding(head_dim=8)

    q = torch.randn(
        2, 4, 3, 8,
        requires_grad=True,
    )

    k = torch.randn(
        2, 4, 2, 8,
        requires_grad=True,
    )

    cos, sin = rope(seq_len=4)

    q_rotated, k_rotated = apply_rotary_pos_emb(
        q,
        k,
        cos,
        sin,
    )

    loss = q_rotated.sum() + k_rotated.sum()

    loss.backward()

    assert q.grad is not None
    assert k.grad is not None

    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(k.grad).all()


# ---------------------------------------------------------
# 12. Buffer persistence
# ---------------------------------------------------------

def test_inverse_frequency_is_non_persistent_buffer():
    """Verify inverse frequencies are excluded from state_dict."""

    rope = RotaryEmbedding(head_dim=64)

    assert "inv_freq" not in rope.state_dict()

    assert "inv_freq" in dict(rope.named_buffers())
