import pytest
import torch
import torch.nn as nn

from src.models.attention import (
    MultiHeadAttention,
    ScaledDotProductAttention,
)

from src.models.gqa import GroupedQueryAttention


# =========================================================
# FIXTURES
# =========================================================

@pytest.fixture
def mha():
    """Create a standard Multi-Head Attention module."""
    return MultiHeadAttention(
        hidden_dim=384,
        num_heads=6,
    )


@pytest.fixture
def sdpa():
    """Create Scaled Dot-Product Attention without dropout."""
    return ScaledDotProductAttention(dropout=0.0)


@pytest.fixture
def input_tensor():
    """Create input tensor with shape [B, T, D]."""
    return torch.randn(2, 5, 384)


@pytest.fixture
def sample_gqa_tensors():
    """Create Q, K and V tensors for GQA attention."""

    batch_size = 2
    num_query_heads = 6
    num_kv_heads = 2
    seq_len = 5
    head_dim = 8

    q = torch.randn(
        batch_size, num_query_heads, seq_len, head_dim
    )

    k = torch.randn(
        batch_size, num_kv_heads, seq_len, head_dim
    )

    v = torch.randn(
        batch_size, num_kv_heads, seq_len, head_dim
    )

    return q, k, v


# =========================================================
# PART A: MULTI-HEAD ATTENTION TESTS (TASK 7)
# =========================================================

# ---------------------------------------------------------
# 1. Initialization
# ---------------------------------------------------------

def test_mha_initialization(mha):
    """Verify MHA configuration."""

    assert mha.hidden_dim == 384
    assert mha.num_heads == 6
    assert mha.head_dim == 64


def test_mha_projection_layers_are_bias_free(mha):
    """Verify Q/K/V/O projections have no bias."""

    assert mha.q_proj.bias is None
    assert mha.k_proj.bias is None
    assert mha.v_proj.bias is None
    assert mha.o_proj.bias is None


def test_mha_projection_layers_are_linear(mha):
    """Verify all projection layers use nn.Linear."""

    assert isinstance(mha.q_proj, nn.Linear)
    assert isinstance(mha.k_proj, nn.Linear)
    assert isinstance(mha.v_proj, nn.Linear)
    assert isinstance(mha.o_proj, nn.Linear)


def test_mha_invalid_hidden_dimension():
    """Reject hidden dimensions not divisible by head count."""

    with pytest.raises(ValueError):
        MultiHeadAttention(hidden_dim=385, num_heads=6)


def test_mha_invalid_num_heads():
    """Reject zero or negative number of heads."""

    with pytest.raises(ValueError):
        MultiHeadAttention(hidden_dim=384, num_heads=0)


def test_mha_invalid_hidden_dim():
    """Reject zero or negative hidden dimensions."""

    with pytest.raises(ValueError):
        MultiHeadAttention(hidden_dim=0, num_heads=6)


# ---------------------------------------------------------
# 2. Q/K/V projections
# ---------------------------------------------------------

def test_mha_qkv_projection_shapes(mha, input_tensor):
    """Verify Q/K/V projection output dimensions."""

    q = mha.q_proj(input_tensor)
    k = mha.k_proj(input_tensor)
    v = mha.v_proj(input_tensor)

    expected_shape = (2, 5, 384)

    assert q.shape == expected_shape
    assert k.shape == expected_shape
    assert v.shape == expected_shape


def test_mha_forward_output_shapes(mha, input_tensor):
    """Verify forward returns correctly split Q/K/V."""

    q, k, v = mha(input_tensor)

    expected_shape = (2, 6, 5, 64)

    assert q.shape == expected_shape
    assert k.shape == expected_shape
    assert v.shape == expected_shape


def test_mha_qkv_use_independent_parameters(mha):
    """Verify Q/K/V projections have separate parameters."""

    assert mha.q_proj.weight is not mha.k_proj.weight
    assert mha.k_proj.weight is not mha.v_proj.weight
    assert mha.q_proj.weight is not mha.v_proj.weight


# ---------------------------------------------------------
# 3. Head splitting
# ---------------------------------------------------------

def test_mha_split_heads_shape(mha):
    """Verify [B,T,D] -> [B,H,T,Dh]."""

    x = torch.randn(2, 5, 384)

    result = mha.split_heads(x)

    assert result.shape == (2, 6, 5, 64)


def test_mha_split_heads_preserves_values(mha):
    """Verify splitting does not modify tensor values."""

    x = torch.randn(2, 5, 384)

    result = mha.split_heads(x)

    reconstructed = (
        result.transpose(1, 2).reshape(2, 5, 384)
    )

    assert torch.equal(x, reconstructed)


def test_mha_split_heads_different_batch_size(mha):
    """Verify splitting supports different batch sizes."""

    x = torch.randn(4, 10, 384)

    result = mha.split_heads(x)

    assert result.shape == (4, 6, 10, 64)


# ---------------------------------------------------------
# 4. Head merging
# ---------------------------------------------------------

def test_mha_merge_heads_shape(mha):
    """Verify [B,H,T,Dh] -> [B,T,D]."""

    x = torch.randn(2, 6, 5, 64)

    result = mha.merge_heads(x)

    assert result.shape == (2, 5, 384)


def test_mha_merge_heads_preserves_values(mha):
    """Verify merging preserves tensor values."""

    x = torch.randn(2, 6, 5, 64)

    result = mha.merge_heads(x)

    reconstructed = mha.split_heads(result)

    assert torch.equal(x, reconstructed)


def test_mha_split_merge_round_trip(mha):
    """Verify splitting followed by merging reconstructs input."""

    x = torch.randn(2, 5, 384)

    split = mha.split_heads(x)
    merged = mha.merge_heads(split)

    assert torch.equal(x, merged)


def test_mha_merge_heads_contiguous_output(mha):
    """Verify merged tensor is contiguous."""

    x = torch.randn(2, 6, 5, 64)

    result = mha.merge_heads(x)

    assert result.is_contiguous()


# ---------------------------------------------------------
# 5. Output projection
# ---------------------------------------------------------

def test_mha_output_projection_shape(mha):
    """Verify merging and output projection."""

    x = torch.randn(2, 6, 5, 64)

    result = mha.project_output(x)

    assert result.shape == (2, 5, 384)


def test_mha_output_projection_is_applied(mha):
    """Verify output projection is applied after merging."""

    x = torch.randn(2, 6, 5, 64)

    expected = mha.o_proj(mha.merge_heads(x))
    actual = mha.project_output(x)

    assert torch.equal(actual, expected)


# ---------------------------------------------------------
# 6. Parameter count
# ---------------------------------------------------------

def test_mha_qkv_parameter_count(mha):
    """Verify Q/K/V parameter count."""

    qkv_params = sum(
        layer.weight.numel()
        for layer in [mha.q_proj, mha.k_proj, mha.v_proj]
    )

    assert qkv_params == 442368


def test_mha_total_parameter_count(mha):
    """Verify total Q/K/V/O parameter count."""

    total_params = sum(
        parameter.numel()
        for parameter in mha.parameters()
    )

    assert total_params == 589824


# ---------------------------------------------------------
# 7. Gradient flow
# ---------------------------------------------------------

def test_mha_qkv_gradient_flow(mha, input_tensor):
    """Verify gradients propagate through Q/K/V projections."""

    q, k, v = mha(input_tensor)

    loss = q.mean() + k.mean() + v.mean()
    loss.backward()

    assert mha.q_proj.weight.grad is not None
    assert mha.k_proj.weight.grad is not None
    assert mha.v_proj.weight.grad is not None


def test_mha_output_projection_gradient_flow(mha):
    """Verify output projection receives gradients."""

    x = torch.randn(2, 6, 5, 64)

    output = mha.project_output(x)
    output.mean().backward()

    assert mha.o_proj.weight.grad is not None


def test_mha_input_gradient_flow(mha):
    """Verify gradients propagate back to input."""

    x = torch.randn(2, 5, 384, requires_grad=True)

    q, k, v = mha(x)

    loss = q.mean() + k.mean() + v.mean()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


# ---------------------------------------------------------
# 8. Invalid inputs
# ---------------------------------------------------------

def test_mha_forward_rejects_invalid_dimensions(mha):
    """Forward requires a 3D tensor."""

    with pytest.raises(ValueError):
        mha(torch.randn(2, 384))


def test_mha_forward_rejects_wrong_hidden_dimension(mha):
    """Forward requires the configured hidden dimension."""

    with pytest.raises(ValueError):
        mha(torch.randn(2, 5, 256))


def test_mha_split_heads_rejects_wrong_hidden_dimension(mha):
    """Head splitting requires the configured hidden dimension."""

    with pytest.raises(ValueError):
        mha.split_heads(torch.randn(2, 5, 256))


def test_mha_split_heads_rejects_non_3d_tensor(mha):
    """Head splitting requires a 3D tensor."""

    with pytest.raises(ValueError):
        mha.split_heads(torch.randn(2, 384))


def test_mha_merge_heads_rejects_wrong_head_count(mha):
    """Head merging requires the configured number of heads."""

    with pytest.raises(ValueError):
        mha.merge_heads(torch.randn(2, 4, 5, 64))


def test_mha_merge_heads_rejects_wrong_head_dimension(mha):
    """Head merging requires the configured head dimension."""

    with pytest.raises(ValueError):
        mha.merge_heads(torch.randn(2, 6, 5, 32))


def test_mha_merge_heads_rejects_non_4d_tensor(mha):
    """Head merging requires a 4D tensor."""

    with pytest.raises(ValueError):
        mha.merge_heads(torch.randn(2, 5, 384))


# =========================================================
# PART B: SCALED DOT-PRODUCT ATTENTION (TASK 9)
# =========================================================

# ---------------------------------------------------------
# 9. Initialization and basic computation
# ---------------------------------------------------------

def test_sdpa_initialization(sdpa):
    """Verify attention dropout configuration."""

    assert isinstance(sdpa.dropout, nn.Dropout)
    assert sdpa.dropout.p == 0.0


def test_sdpa_output_shape(sdpa, sample_gqa_tensors):
    """Verify attention output shape for GQA."""

    q, k, v = sample_gqa_tensors

    output = sdpa(q, k, v)

    assert output.shape == (2, 6, 5, 8)


def test_sdpa_output_is_finite(sdpa, sample_gqa_tensors):
    """Verify output contains no NaN or Inf."""

    q, k, v = sample_gqa_tensors

    output = sdpa(q, k, v)

    assert torch.isfinite(output).all()


def test_sdpa_output_dtype(sdpa, sample_gqa_tensors):
    """Verify output dtype matches Value dtype."""

    q, k, v = sample_gqa_tensors

    output = sdpa(q, k, v)

    assert output.dtype == v.dtype


def test_sdpa_matches_manual_attention_calculation():
    """Compare implementation with manual scaled dot-product attention."""

    torch.manual_seed(42)

    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)

    module = ScaledDotProductAttention(dropout=0.0)

    actual = module(q, k, v, is_causal=False)

    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores / (4 ** 0.5)

    weights = torch.softmax(scores, dim=-1)
    expected = torch.matmul(weights, v)

    assert torch.allclose(actual, expected, atol=1e-6)


# ---------------------------------------------------------
# 10. GQA KV-head expansion
# ---------------------------------------------------------

def test_gqa_head_expansion(sdpa):
    """Verify KV heads are repeated according to GQA groups."""

    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)

    expanded_k, expanded_v = sdpa._expand_kv_heads(
        k, v, num_query_heads=6
    )

    assert expanded_k.shape == (1, 6, 3, 4)
    assert expanded_v.shape == (1, 6, 3, 4)

    # First KV head is shared by Query heads 0, 1, 2.
    assert torch.equal(expanded_k[:, 0], k[:, 0])
    assert torch.equal(expanded_k[:, 1], k[:, 0])
    assert torch.equal(expanded_k[:, 2], k[:, 0])

    # Second KV head is shared by Query heads 3, 4, 5.
    assert torch.equal(expanded_k[:, 3], k[:, 1])
    assert torch.equal(expanded_k[:, 4], k[:, 1])
    assert torch.equal(expanded_k[:, 5], k[:, 1])

    assert torch.equal(expanded_v[:, 0], v[:, 0])
    assert torch.equal(expanded_v[:, 3], v[:, 1])


def test_mha_does_not_expand_kv_heads(sdpa):
    """Verify MHA requires no KV-head expansion."""

    k = torch.randn(1, 4, 3, 8)
    v = torch.randn(1, 4, 3, 8)

    expanded_k, expanded_v = sdpa._expand_kv_heads(
        k, v, num_query_heads=4
    )

    assert expanded_k is k
    assert expanded_v is v


def test_gqa_output_matches_manual_expansion():
    """Verify GQA result matches explicit KV expansion and MHA calculation."""

    torch.manual_seed(42)

    q = torch.randn(1, 6, 3, 4)
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)

    module = ScaledDotProductAttention(dropout=0.0)

    actual = module(q, k, v, is_causal=False)

    expanded_k = k.repeat_interleave(3, dim=1)
    expanded_v = v.repeat_interleave(3, dim=1)

    scores = torch.matmul(q, expanded_k.transpose(-2, -1))
    scores = scores / (4 ** 0.5)

    weights = torch.softmax(scores, dim=-1)
    expected = torch.matmul(weights, expanded_v)

    assert torch.allclose(actual, expected, atol=1e-6)


# ---------------------------------------------------------
# 11. Causal masking
# ---------------------------------------------------------

def test_causal_attention_prevents_future_information(sdpa):
    """Changing future Values must not affect earlier outputs."""

    q = torch.zeros(1, 2, 3, 4)
    k = torch.zeros(1, 1, 3, 4)

    v1 = torch.tensor(
        [[[
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 3.0],
        ]]]
    )

    v2 = v1.clone()
    v2[:, :, 2, :] = 1000.0

    output1 = sdpa(q, k, v1, is_causal=True)
    output2 = sdpa(q, k, v2, is_causal=True)

    assert torch.allclose(
        output1[:, :, :2, :],
        output2[:, :, :2, :],
    )

    assert not torch.allclose(
        output1[:, :, 2, :],
        output2[:, :, 2, :],
    )


def test_non_causal_attention_can_access_future_tokens(sdpa):
    """Verify non-causal attention can access future tokens."""

    q = torch.zeros(1, 2, 3, 4)
    k = torch.zeros(1, 1, 3, 4)

    v1 = torch.zeros(1, 1, 3, 4)
    v2 = v1.clone()

    v2[:, :, 2, :] = 100.0

    output1 = sdpa(q, k, v1, is_causal=False)
    output2 = sdpa(q, k, v2, is_causal=False)

    assert not torch.allclose(
        output1[:, :, 0, :],
        output2[:, :, 0, :],
    )


def test_causal_attention_matches_manual_lower_triangular_mask():
    """Verify causal masking matches an explicit lower-triangular mask."""

    torch.manual_seed(42)

    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)

    module = ScaledDotProductAttention(dropout=0.0)

    actual = module(q, k, v, is_causal=True)

    scores = q @ k.transpose(-2, -1)
    scores = scores / (8 ** 0.5)

    mask = torch.triu(
        torch.ones(4, 4, dtype=torch.bool),
        diagonal=1,
    )

    scores = scores.masked_fill(mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    expected = weights @ v

    assert torch.allclose(actual, expected, atol=1e-6)


# ---------------------------------------------------------
# 12. Attention dropout
# ---------------------------------------------------------

def test_dropout_disabled_in_eval_mode():
    """Verify dropout is disabled during evaluation."""

    module = ScaledDotProductAttention(dropout=1.0)
    module.eval()

    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 1, 3, 4)
    v = torch.randn(1, 1, 3, 4)

    output = module(q, k, v)

    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output).item() > 0


def test_dropout_drops_all_attention_weights_in_training():
    """Verify dropout probability 1.0 zeros all attention weights."""

    module = ScaledDotProductAttention(dropout=1.0)
    module.train()

    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 1, 3, 4)
    v = torch.randn(1, 1, 3, 4)

    output = module(q, k, v)

    assert torch.count_nonzero(output).item() == 0


def test_dropout_probability_validation():
    """Reject dropout probabilities outside [0, 1]."""

    with pytest.raises(ValueError):
        ScaledDotProductAttention(dropout=-0.1)

    with pytest.raises(ValueError):
        ScaledDotProductAttention(dropout=1.1)


# ---------------------------------------------------------
# 13. Numerical stability
# ---------------------------------------------------------

def test_large_attention_scores_are_numerically_stable(sdpa):
    """Verify large scores do not produce NaN or Inf."""

    q = torch.full((1, 2, 3, 4), 1000.0)
    k = torch.full((1, 1, 3, 4), 1000.0)
    v = torch.randn(1, 1, 3, 4)

    output = sdpa(q, k, v)

    assert torch.isfinite(output).all()


def test_attention_with_single_token(sdpa):
    """Verify attention works with sequence length one."""

    q = torch.randn(1, 6, 1, 8)
    k = torch.randn(1, 2, 1, 8)
    v = torch.randn(1, 2, 1, 8)

    output = sdpa(q, k, v)

    assert output.shape == (1, 6, 1, 8)
    assert torch.isfinite(output).all()


# ---------------------------------------------------------
# 14. Input validation
# ---------------------------------------------------------

def test_invalid_head_divisibility(sdpa):
    """Reject incompatible Query and KV head counts."""

    q = torch.randn(1, 5, 3, 8)
    k = torch.randn(1, 2, 3, 8)
    v = torch.randn(1, 2, 3, 8)

    with pytest.raises(ValueError):
        sdpa(q, k, v)


def test_invalid_head_dimension(sdpa):
    """Reject mismatched Query and Key head dimensions."""

    q = torch.randn(1, 4, 3, 8)
    k = torch.randn(1, 2, 3, 6)
    v = torch.randn(1, 2, 3, 8)

    with pytest.raises(ValueError):
        sdpa(q, k, v)


def test_invalid_causal_sequence_lengths(sdpa):
    """Reject unequal Q and K sequence lengths for causal attention."""

    q = torch.randn(1, 4, 3, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)

    with pytest.raises(ValueError):
        sdpa(q, k, v, is_causal=True)


def test_invalid_tensor_dimension(sdpa):
    """Reject non-4D Q/K/V tensors."""

    q = torch.randn(1, 3, 8)
    k = torch.randn(1, 2, 3, 8)
    v = torch.randn(1, 2, 3, 8)

    with pytest.raises(ValueError):
        sdpa(q, k, v)


def test_invalid_batch_size(sdpa):
    """Reject different batch sizes across Q, K and V."""

    q = torch.randn(2, 4, 3, 8)
    k = torch.randn(1, 2, 3, 8)
    v = torch.randn(1, 2, 3, 8)

    with pytest.raises(ValueError):
        sdpa(q, k, v)


def test_invalid_key_value_sequence_lengths(sdpa):
    """Reject different sequence lengths between K and V."""

    q = torch.randn(1, 4, 3, 8)
    k = torch.randn(1, 2, 3, 8)
    v = torch.randn(1, 2, 4, 8)

    with pytest.raises(ValueError):
        sdpa(q, k, v)


def test_invalid_device_dtype_validation(sdpa):
    """Reject mismatched Q/K/V dtypes."""

    q = torch.randn(1, 4, 3, 8)
    k = torch.randn(1, 2, 3, 8)
    v = torch.randn(1, 2, 3, 8, dtype=torch.float64)

    with pytest.raises(ValueError):
        sdpa(q, k, v)


# ---------------------------------------------------------
# 15. Attention gradient flow
# ---------------------------------------------------------

def test_attention_gradient_flow(sdpa):
    """Verify gradients propagate through Q, K and V."""

    q = torch.randn(1, 6, 4, 8, requires_grad=True)
    k = torch.randn(1, 2, 4, 8, requires_grad=True)
    v = torch.randn(1, 2, 4, 8, requires_grad=True)

    output = sdpa(q, k, v)

    output.mean().backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None

    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(k.grad).all()
    assert torch.isfinite(v.grad).all()


# =========================================================
# PART C: END-TO-END INTEGRATION
# =========================================================

# ---------------------------------------------------------
# 16. MHA + Scaled Dot-Product Attention
# ---------------------------------------------------------

def test_mha_attention_integration():
    """
    Verify complete MHA pipeline:

        Input
          -> MHA Q/K/V projections
          -> Scaled Dot-Product Attention
          -> Merge heads
          -> Output projection
          -> Final output
    """

    batch_size = 2
    seq_len = 5
    hidden_dim = 384

    mha = MultiHeadAttention(
        hidden_dim=hidden_dim,
        num_heads=6,
    )

    attention = ScaledDotProductAttention(dropout=0.1)

    mha.train()
    attention.train()

    x = torch.randn(
        batch_size,
        seq_len,
        hidden_dim,
        requires_grad=True,
    )

    q, k, v = mha(x)

    assert q.shape == (2, 6, 5, 64)
    assert k.shape == (2, 6, 5, 64)
    assert v.shape == (2, 6, 5, 64)

    context = attention(
        q, k, v,
        is_causal=True,
    )

    assert context.shape == (2, 6, 5, 64)

    output = mha.project_output(context)

    assert output.shape == (2, 5, 384)

    output.mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    assert mha.q_proj.weight.grad is not None
    assert mha.k_proj.weight.grad is not None
    assert mha.v_proj.weight.grad is not None
    assert mha.o_proj.weight.grad is not None


# ---------------------------------------------------------
# 17. GQA + Scaled Dot-Product Attention
# ---------------------------------------------------------

def test_gqa_attention_integration():
    """
    Verify complete GQA pipeline:

        Input
          -> GQA Q/K/V projections
          -> Scaled Dot-Product Attention
          -> Merge Query heads
          -> Output projection
          -> Final output
    """

    batch_size = 2
    seq_len = 5
    hidden_dim = 384

    gqa = GroupedQueryAttention(
        hidden_dim=hidden_dim,
        num_query_heads=6,
        num_kv_heads=2,
    )

    attention = ScaledDotProductAttention(dropout=0.1)

    gqa.train()
    attention.train()

    x = torch.randn(
        batch_size,
        seq_len,
        hidden_dim,
        requires_grad=True,
    )

    q, k, v = gqa(x)

    assert q.shape == (2, 6, 5, 64)
    assert k.shape == (2, 2, 5, 64)
    assert v.shape == (2, 2, 5, 64)

    context = attention(
        q, k, v,
        is_causal=True,
    )

    assert context.shape == (2, 6, 5, 64)

    # Reuse the existing GQA output projection method.
    output = gqa.project_output(context)

    assert output.shape == (2, 5, 384)

    output.mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    assert gqa.q_proj.weight.grad is not None
    assert gqa.k_proj.weight.grad is not None
    assert gqa.v_proj.weight.grad is not None
    assert gqa.o_proj.weight.grad is not None
