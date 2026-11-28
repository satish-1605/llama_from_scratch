from loguru import logger
import torch
import torch.nn as nn
import math

class TransformerBlock(nn.Module):
    """
    Pre-Norm Decoder Transformer Block.

    Components:
        1. First normalization layer.
        2. Grouped Query Attention (or compatible attention module).
        3. First residual connection.
        4. Second normalization layer.
        5. Feed Forward Network (e.g. SwiGLU).
        6. Second residual connection.

    Expected input/output:
        [batch_size, sequence_length, d_model]

    Args:
        d_model: Model embedding dimension.
        attention: Existing attention/GQA module.
        feed_forward: Existing feed-forward module, e.g. SwiGLU.
        norm1: First normalization module, e.g. RMSNorm.
        norm2: Second normalization module, e.g. RMSNorm.
    """

    def __init__(self, 
                 d_model: int,
                 attention : nn.Module,
                 feed_forward : nn.Module,
                 norm1: nn.Module,
                 norm2: nn.Module,
                 )-> None:
        super().__init__()
        if not isinstance(d_model, int) or isinstance(d_model, bool):
            raise TypeError("d_model must be an integer.")

        if d_model <= 0:
            raise ValueError("d_model must be positive.")

        modules = {
            "attention":attention,
            "feed_forward":feed_forward,
            "norm1":norm1,
            "norm2":norm2
        }

        for name, module in modules.items():
            if not isinstance(module, nn.Module):
                raise ValueError(f"{name} must be a torch.nn.Module")

        self.d_model = d_model

        self.norm1 = norm1        
        self.attention = attention

        self.norm2 = norm2
        self.feed_forward = feed_forward

        logger.info(
            "Initialized TransformerBlock | d_model={} | "
            "attention={} | feed_forward={}",
            d_model,
            attention.__class__.__name__,
            feed_forward.__class__.__name__,
        )

    def forward(self, x:torch.Tensor)-> torch.Tensor:
        """
        Execute the Pre-Norm Transformer Block.

        Args:
            x: Input tensor of shape [B, T, d_model].

        Returns:
            Tensor of shape [B, T, d_model].
        """
        if not isinstance(x, torch.Tensor):
            raise ValueError("Input x must be torch.Tensor")

        if x.ndim != 3:
            raise ValueError(
                "Expected input shape [B, T, d_model], "
                f"but received {tuple(x.shape)}."
            )

        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected last dimension {self.d_model}, "
                f"but received {x.shape[-1]}."
            )

        # 1. Attention sublayer: Pre-Norm + Residual
        attention_output = self.attention(self.norm1(x))
        if not isinstance(attention_output, torch.Tensor):
            raise TypeError("Attention module must return a tensor.")

        if attention_output.shape != x.shape:
            raise ValueError(
                "Attention output shape must match input shape. "
                f"Expected {tuple(x.shape)}, "
                f"received {tuple(attention_output.shape)}."
            )

        x = x + attention_output

        # 2. Feed Forward sublayer: Pre-Norm + Residual
        ffn_output = self.feed_forward(self.norm2(x))
        if not isinstance(ffn_output, torch.Tensor):
            raise TypeError("Feedforward module must return a tensor.")
        
        if ffn_output.shape != x.shape:
            raise ValueError(
                "Feedforward output shape must match input shape. "
                f"Expected {tuple(x.shape)}, "
                f"received {tuple(ffn_output.shape)}."
            )

        x = x + ffn_output
        return x 







