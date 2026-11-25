import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Normalizes input hidden states using their root mean square
    without subtracting the mean.

    Args:
        dim: Dimension of the input hidden states.
        eps: Small constant for numerical stability.

    Input:
        Tensor of shape (..., dim)

    Output:
        Tensor of shape (..., dim)
    """

    def __init__(self, dim:int, eps:float =  1e-6):
        super().__init__()

        if dim <= 0:
            raise ValueError("dim must be a positive integer")

        if eps <= 0:
            raise ValueError("eps must be positive")

        self.dim = dim
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x:torch.Tensor,)->torch.Tensor:
        """
        Apply RMS normalization to the input tensor.

        Args:
            x: Input tensor with last dimension equal to dim.

        Returns:
            RMS-normalized tensor with learnable scaling.
        """
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Expected last dimension {self.dim}, "
                f"but received {x.shape[-1]}"
            )

        input_dtype = x.dtype

        x_float = x.float()

        rms = torch.sqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normalized_x = x/ rms

        output = normalized_x * self.weight.float()
        return output.to(input_dtype)

        
