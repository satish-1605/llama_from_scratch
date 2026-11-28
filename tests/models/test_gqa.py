
import pytest
import torch
import torch.nn as nn

from src.models.gqa import GroupedQueryAttention


@pytest.fixture
def gqa():
    """Create GQA using the project configuration."""
    return GroupedQueryAttention(
        hidden_dim=384,
        num_query_heads=6,
        num_kv_heads=2,
    )


@pytest.fixture
def input_tensor():
    """Generate sample input [B, T, D]."""
    return torch.randn(2, 5, 384)


# ---------------------------------------------------------
# 1. Initialization tests
# ---------------------------------------------------------

def test_gqa_initialization(gqa):
    assert gqa.hidden_dim == 384
    assert gqa.num_query_heads == 6
    assert gqa.num_kv_heads == 2
    assert gqa.head_dim == 64
    assert gqa.num_query_groups == 3
    assert gqa.kv_dim == 128


def test_projection_layers_are_linear(gqa):
    assert isinstance(gqa.q_proj, nn.Linear)
    assert isinstance(gqa.k_proj, nn.Linear)
    assert isinstance(gqa.v_proj, nn.Linear)
    assert isinstance(gqa.o_proj, nn.Linear)


def test_projection_layers_are_bias_free(gqa):
    assert gqa.q_proj.bias is None
    assert gqa.k_proj.bias is None
    assert gqa.v_proj.bias is None
    assert gqa.o_proj.bias is None


def test_invalid_hidden_dimension():
    with pytest.raises(ValueError):
        GroupedQueryAttention(385, 6, 2)


def test_invalid_query_heads():
    with pytest.raises(ValueError):
        GroupedQueryAttention(384, 0, 2)


def test_invalid_kv_heads():
    with pytest.raises(ValueError):
        GroupedQueryAttention(384, 6, 0)


def test_query_heads_not_divisible_by_kv_heads():
    with pytest.raises(ValueError):
        GroupedQueryAttention(384, 6, 4)


def test_hidden_dimension_not_divisible_by_query_heads():
    with pytest.raises(ValueError):
        GroupedQueryAttention(384, 5, 1)


# ---------------------------------------------------------
# 2. Projection tests
# ---------------------------------------------------------

def test_q_projection_shape(gqa, input_tensor):
    q = gqa.q_proj(input_tensor)

    assert q.shape == (2, 5, 384)


def test_k_projection_shape(gqa, input_tensor):
    k = gqa.k_proj(input_tensor)

    assert k.shape == (2, 5, 128)


def test_v_projection_shape(gqa, input_tensor):
    v = gqa.v_proj(input_tensor)

    assert v.shape == (2, 5, 128)


def test_qkv_weights_are_independent(gqa):
    assert gqa.q_proj.weight is not gqa.k_proj.weight
    assert gqa.k_proj.weight is not gqa.v_proj.weight
    assert gqa.q_proj.weight is not gqa.v_proj.weight


# ---------------------------------------------------------
# 3. Head splitting tests
# ---------------------------------------------------------

def test_split_query_heads_shape(gqa):
    x = torch.randn(2, 5, 384)

    q = gqa.split_query_heads(x)

    assert q.shape == (2, 6, 5, 64)


def test_split_kv_heads_shape(gqa):
    x = torch.randn(2, 5, 128)

    kv = gqa.split_kv_heads(x)

    assert kv.shape == (2, 2, 5, 64)


def test_split_query_heads_preserves_values(gqa):
    x = torch.randn(2, 5, 384)

    q = gqa.split_query_heads(x)

    reconstructed = (
        q.transpose(1, 2).reshape(2, 5, 384)
    )

    assert torch.equal(x, reconstructed)


def test_split_kv_heads_preserves_values(gqa):
    x = torch.randn(2, 5, 128)

    kv = gqa.split_kv_heads(x)

    reconstructed = (
        kv.transpose(1, 2).reshape(2, 5, 128)
    )

    assert torch.equal(x, reconstructed)


# ---------------------------------------------------------
# 4. Query grouping tests
# ---------------------------------------------------------

def test_group_query_heads_shape(gqa):
    q = torch.randn(2, 6, 5, 64)

    grouped_q = gqa.group_query_heads(q)

    assert grouped_q.shape == (2, 2, 3, 5, 64)


def test_group_query_heads_preserves_values(gqa):
    q = torch.randn(2, 6, 5, 64)

    grouped_q = gqa.group_query_heads(q)

    reconstructed = grouped_q.reshape(2, 6, 5, 64)

    assert torch.equal(q, reconstructed)


def test_group_query_heads_mapping(gqa):
    # Give each Query head a unique identifiable value.
    q = torch.zeros(1, 6, 1, 64)

    for head_index in range(6):
        q[:, head_index, :, :] = head_index

    grouped_q = gqa.group_query_heads(q)

    # KV group 0 must contain Q0, Q1, Q2.
    assert grouped_q[0, 0, 0, 0, 0].item() == 0
    assert grouped_q[0, 0, 1, 0, 0].item() == 1
    assert grouped_q[0, 0, 2, 0, 0].item() == 2

    # KV group 1 must contain Q3, Q4, Q5.
    assert grouped_q[0, 1, 0, 0, 0].item() == 3
    assert grouped_q[0, 1, 1, 0, 0].item() == 4
    assert grouped_q[0, 1, 2, 0, 0].item() == 5


def test_group_query_heads_rejects_invalid_shape(gqa):
    q = torch.randn(2, 6, 5)

    with pytest.raises(ValueError):
        gqa.group_query_heads(q)


def test_group_query_heads_rejects_wrong_head_count(gqa):
    q = torch.randn(2, 4, 5, 64)

    with pytest.raises(ValueError):
        gqa.group_query_heads(q)


def test_group_query_heads_rejects_wrong_head_dim(gqa):
    q = torch.randn(2, 6, 5, 32)

    with pytest.raises(ValueError):
        gqa.group_query_heads(q)


# ---------------------------------------------------------
# 5. Head merging tests
# ---------------------------------------------------------

def test_merge_query_heads_shape(gqa):
    q = torch.randn(2, 6, 5, 64)

    merged = gqa.merge_query_heads(q)

    assert merged.shape == (2, 5, 384)


def test_merge_query_heads_preserves_values(gqa):
    q = torch.randn(2, 6, 5, 64)

    merged = gqa.merge_query_heads(q)

    reconstructed = gqa.split_query_heads(merged)

    assert torch.equal(q, reconstructed)


def test_merge_query_heads_contiguous(gqa):
    q = torch.randn(2, 6, 5, 64)

    merged = gqa.merge_query_heads(q)

    assert merged.is_contiguous()


# ---------------------------------------------------------
# 6. Forward tests
# ---------------------------------------------------------

def test_forward_output_shapes(gqa, input_tensor):
    q, k, v = gqa(input_tensor)

    assert q.shape == (2, 6, 5, 64)
    assert k.shape == (2, 2, 5, 64)
    assert v.shape == (2, 2, 5, 64)


def test_forward_returns_three_tensors(gqa, input_tensor):
    outputs = gqa(input_tensor)

    assert len(outputs) == 3

    for output in outputs:
        assert isinstance(output, torch.Tensor)


def test_forward_supports_different_sequence_length(gqa):
    x = torch.randn(4, 10, 384)

    q, k, v = gqa(x)

    assert q.shape == (4, 6, 10, 64)
    assert k.shape == (4, 2, 10, 64)
    assert v.shape == (4, 2, 10, 64)


def test_forward_rejects_non_3d_input(gqa):
    x = torch.randn(2, 384)

    with pytest.raises(ValueError):
        gqa(x)


def test_forward_rejects_wrong_hidden_dimension(gqa):
    x = torch.randn(2, 5, 256)

    with pytest.raises(ValueError):
        gqa(x)


# ---------------------------------------------------------
# 7. Output projection tests
# ---------------------------------------------------------

def test_output_projection_shape(gqa):
    q = torch.randn(2, 6, 5, 64)

    output = gqa.project_output(q)

    assert output.shape == (2, 5, 384)


def test_output_projection_correctness(gqa):
    q = torch.randn(2, 6, 5, 64)

    merged = gqa.merge_query_heads(q)

    expected = gqa.o_proj(merged)

    actual = gqa.project_output(q)

    assert torch.equal(actual, expected)


# ---------------------------------------------------------
# 8. Parameter count tests
# ---------------------------------------------------------

def test_qkv_parameter_count(gqa):
    qkv_params = (
        gqa.q_proj.weight.numel()
        + gqa.k_proj.weight.numel()
        + gqa.v_proj.weight.numel()
    )

    assert qkv_params == 245760


def test_total_gqa_parameter_count(gqa):
    total_params = sum(
        parameter.numel()
        for parameter in gqa.parameters()
    )

    assert total_params == 393216


# ---------------------------------------------------------
# 9. Gradient flow tests
# ---------------------------------------------------------

def test_qkv_gradient_flow(gqa, input_tensor):
    q, k, v = gqa(input_tensor)

    loss = q.mean() + k.mean() + v.mean()

    loss.backward()

    assert gqa.q_proj.weight.grad is not None
    assert gqa.k_proj.weight.grad is not None
    assert gqa.v_proj.weight.grad is not None


def test_output_projection_gradient_flow(gqa):
    q = torch.randn(2, 6, 5, 64)

    output = gqa.project_output(q)

    loss = output.mean()

    loss.backward()

    assert gqa.o_proj.weight.grad is not None


def test_input_gradient_flow(gqa):
    x = torch.randn(
        2, 5, 384,
        requires_grad=True,
    )

    q, k, v = gqa(x)

    loss = q.mean() + k.mean() + v.mean()

    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
