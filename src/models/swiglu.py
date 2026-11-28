
"""
SwiGLU Feed-Forward Network implementation.

This module implements the SwiGLU feed-forward network used in
modern decoder-only Transformer architectures.

SwiGLU combines:
    1. Two parallel linear projections.
    2. SiLU activation on the first projection.
    3. Element-wise multiplication for gating.
    4. A final linear projection back to d_model.

Mathematical formulation:

    gate = SiLU(W1(x))
    value = W3(x)
    hidden = gate * value
    output = W2(hidden)

Input shape:
    [batch_size, sequence_length, d_model]

Output shape:
    [batch_size, sequence_length, d_model]
"""

import logging

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class SwiGLU(nn.Module):
    """
    Implements the SwiGLU Feed-Forward Network.

    The module uses three bias-free linear projections:

        W1: d_model -> hidden_dim
        W3: d_model -> hidden_dim
        W2: hidden_dim -> d_model

    SiLU activation is applied to the W1 projection.
    Its output is multiplied element-wise with the W3 projection.

    Args:
        d_model (int):
            Input and output embedding dimension.

        hidden_dim (int):
            Intermediate feed-forward dimension.

    Input:
        x (torch.Tensor):
            Input tensor of shape [B, T, d_model].

    Returns:
        torch.Tensor:
            Output tensor of shape [B, T, d_model].

    Raises:
        TypeError:
            If d_model or hidden_dim is not an integer.

        ValueError:
            If either dimension is not positive.

        ValueError:
            If input tensor does not have three dimensions
            or its last dimension does not match d_model.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        # Validate constructor arguments.
        if isinstance(d_model, bool) or not isinstance(d_model, int):
            raise TypeError("d_model must be an integer.")

        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int):
            raise TypeError("hidden_dim must be an integer.")

        if d_model <= 0:
            raise ValueError("d_model must be greater than zero.")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be greater than zero.")

        self.d_model = d_model
        self.hidden_dim = hidden_dim

        # First projection: gate activation branch.
        self.w1 = nn.Linear(
            in_features=d_model,
            out_features=hidden_dim,
            bias=False,
        )

        # Second projection: value branch.
        self.w3 = nn.Linear(
            in_features=d_model,
            out_features=hidden_dim,
            bias=False,
        )

        # SiLU activation function.
        self.act = nn.SiLU()

        # Final projection: hidden_dim -> d_model.
        self.w2 = nn.Linear(
            in_features=hidden_dim,
            out_features=d_model,
            bias=False,
        )

        logger.info(
            "Initialized SwiGLU: d_model=%d, hidden_dim=%d, "
            "parameters=%d",
            d_model,
            hidden_dim,
            self.count_parameters(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU feed-forward computation.

        Computation:

            gate = SiLU(W1(x))
            value = W3(x)
            hidden = gate * value
            output = W2(hidden)

        Args:
            x (torch.Tensor):
                Input tensor with shape [B, T, d_model].

        Returns:
            torch.Tensor:
                Output tensor with shape [B, T, d_model].

        Raises:
            ValueError:
                If input tensor has an invalid shape.
        """

        if x.ndim != 3:
            raise ValueError(
                f"Expected a 3D input tensor [B, T, D], "
                f"but received {x.ndim}D tensor."
            )

        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected input feature dimension {self.d_model}, "
                f"but received {x.shape[-1]}."
            )

        # Parallel projections.
        gate = self.act(self.w1(x))
        value = self.w3(x)

        # Element-wise gating.
        hidden = gate * value

        # Project back to original model dimension.
        output = self.w2(hidden)

        return output

    def count_parameters(self) -> int:
        """
        Calculate the total number of trainable parameters.

        Since all three linear layers are bias-free:

            Parameters = 3 * d_model * hidden_dim

        Returns:
            int: Total number of trainable parameters.
        """

        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
