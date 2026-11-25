
"""
Rotary Positional Embeddings (RoPE) implementation.

This module implements LLaMA-style Rotary Positional Embeddings
using the split-half rotation convention.

Expected input tensor layout:
    [batch_size, seq_len, num_heads, head_dim]

Main components:
    1. rotate_half()
    2. RotaryEmbedding
    3. apply_rotary_pos_emb()
"""

import torch
from loguru import logger
import torch.nn as nn

def rotate_half(x:torch.Tensor)-> torch.Tensor:
    """
    Apply split-half rotation to the last dimension of a tensor.

    The input is divided into two equal halves:

        x = [x1, x2]

    The transformation is:

        rotate_half(x) = [-x2, x1]

    This operation is used to construct the rotation
    required by Rotary Positional Embeddings.

    Args:
        x: Input tensor with an even last dimension.

    Returns:
        Tensor with the same shape as the input.

    Raises:
        ValueError: If the last dimension is not even.
    """

    if x.shape[-1] % 2 != 0:
        raise ValueError(f"Rope requires an even head dimension, got {x.shape[-1]}")

    half_dim = x.shape[-1] // 2

    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]

    return torch.cat([-x2, x1], dim=-1)


class RotaryEmbedding(nn.Module):
    """
    Generate rotary positional embedding cosine and sine values.

    Uses the standard LLaMA-style inverse frequency formula:

        inv_freq[i] = 1 / (base ** (2*i / head_dim))

    Position-dependent frequencies are calculated as:

        freqs = positions outer_product inv_freq

    Cosine and sine values are then precomputed and expanded
    for broadcasting with Query and Key tensors.

    Args:
        head_dim: Dimension of each attention head.
        base: RoPE frequency base. Defaults to 10000.0.
    """

    def __init__(self, head_dim: int, base: float = 10000.0,)->None:
        super().__init__()

        if head_dim <= 0:
            raise ValueError(
                f"head_dim must be positive. Received {head_dim}"
            )

        if head_dim % 2 != 0:
            raise ValueError(
                f"head_dim must be even for RoPE. Received {head_dim}"
            )

        if base <= 0:
            raise ValueError(
                f"base must be positive. Received {base}"
            )

        self.head_dim = head_dim
        self.base= base

        # Dimension indices for each 2D rotation pair.
        dim_indices = torch.arange(0, head_dim, 2, dtype=torch.float32)

        # Calculate inverse frequencies.
        inv_freq= 1 / (base**(dim_indices/head_dim))

        # Keep inverse frequencies as a non-persistent buffer.
        # It automatically follows the module's device.
        # It does not need to be stored in model checkpoints.

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        logger.info("Initialised Rotary Embedding : head_dim {} | base {}", head_dim, base)

    def forward(self,
                seq_len : int,
                device : torch.device | None = None,
                dtype : torch.dtype | None = None,
                position_offset : int = 0
                )-> tuple[torch.Tensor, torch.Tensor]:

        """
        Generate cosine and sine embeddings for a sequence.

        Args:
            seq_len: Number of token positions to generate.
            device: Target device. Defaults to inv_freq device.
            dtype: Output dtype. Defaults to float32.
            position_offset: Starting position index.
                Useful for future KV-cache based generation.

        Returns:
            Tuple containing:
                cos: Shape [1, seq_len, 1, head_dim]
                sin: Shape [1, seq_len, 1, head_dim]

        Raises:
            ValueError: If sequence length or position offset is invalid.
        """
        if seq_len <= 0:
            raise ValueError(
                f"seq_len must be positive. Received {seq_len}"
            )

        if position_offset < 0:
            raise ValueError(
                f"position_offset cannot be negative. "
                f"Received {position_offset}"
            )

        if device is None:
            device = self.inv_freq.device

        if dtype is None:
            dtype = torch.float32

        # generate position indices
        pos = torch.arange(
            position_offset,
            position_offset + seq_len,
            device= device,
            dtype= torch.float32)

        # calculate position dependent freq
        # for e.g if T = 4, inve freq shape - 4 = {4, 4}
        # shape (seq_len, head_dim//2)
        freq = torch.outer(
            pos,
            self.inv_freq.to(device=device)
        )

        # split half convention
        # Duplicate frequencies to match the full head dimension.
        # Shape: [seq_len, head_dim]

        emb = torch.cat((freq, freq), dim=-1)

        #calculate cosine and sine in float 32
        cos = torch.cos(emb)
        sin = torch.sin(emb)

        # Add batch and head broadcasting dimensions.
        # Shape: [1, seq_len, 1, head_dim]
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        # convert to the requested dtype
        cos = cos.to(dtype=dtype)
        sin = sin.to(dtype=dtype)

        logger.debug("Genearted ROPE emebdding, seq_len {} | offset {} |  shape {}", seq_len, position_offset, tuple(cos.shape))
        return cos, sin

def apply_rotary_pos_emb(
        q: torch.Tensor,
        k : torch.Tensor,
        cos: torch.Tensor,
        sin : torch.Tensor        
    )->tuple[torch.Tensor, torch.Tensor]:

    """
    Apply rotary positional embeddings to Query and Key tensors.

    Rotation formula:

        x_rotated = (x * cos) + (rotate_half(x) * sin)

    RoPE is applied to Query and Key, not Value.

    Expected tensor layouts:

        q:   [batch_size, seq_len, num_query_heads, head_dim]
        k:   [batch_size, seq_len, num_kv_heads, head_dim]

        cos: [1, seq_len, 1, head_dim]
        sin: [1, seq_len, 1, head_dim]

    Args:
        q: Query tensor.
        k: Key tensor.
        cos: Precomputed cosine embeddings.
        sin: Precomputed sine embeddings.

    Returns:
        Tuple containing rotated Query and Key tensors.

    Raises:
        ValueError: If input shapes or dimensions are incompatible.
    """

    if q.ndim != 4 or k.ndim != 4:
        raise ValueError(
            "Query and Key must have 4 dimensions: "
            "[batch, seq_len, num_heads, head_dim]"
        )

    if q.shape[0] != k.shape[0]:
        raise ValueError("Query and Key batch sizes must match.")

    if q.shape[1] != k.shape[1]:
        raise ValueError("Query and Key sequence lengths must match.")

    if q.shape[-1] != k.shape[-1]:
        raise ValueError("Query and Key head dimensions must match.")

    if q.shape[-1] % 2 != 0:
        raise ValueError("RoPE requires an even head dimension.")

    expected_shape = (1, q.shape[1], 1, q.shape[-1])

    if tuple(cos.shape) != expected_shape:
        raise ValueError(
            f"Invalid cos shape. Expected {expected_shape}, "
            f"received {tuple(cos.shape)}"
        )

    if tuple(sin.shape) != expected_shape:
        raise ValueError(
            f"Invalid sin shape. Expected {expected_shape}, "
            f"received {tuple(sin.shape)}"
        )

    if q.device != k.device or q.device != cos.device or q.device != sin.device:
        raise ValueError(
            "Query, Key, cosine and sine tensors must be on the same device."
        )

    if q.dtype != k.dtype:
        raise ValueError("Query and Key must have the same dtype.")

    # Ensure cosine and sine use the input tensor dtype.
    cos = cos.to(dtype=q.dtype)
    sin = sin.to(dtype=q.dtype)

    q_rotated = (q * cos) + (rotate_half(q) * sin)
    k_rotated = (k * cos) + (rotate_half(k) * sin)

    logger.debug("Applied RoPE, q_shape {} | k_shape {}", tuple(q.shape), tuple(k.shape))

    return q_rotated, k_rotated









