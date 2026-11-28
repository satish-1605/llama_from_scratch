from loguru import logger
import math

import torch.nn as nn
from src.models.normalization import RMSNorm

def initialize_llama_weights(
        model: nn.Module,
        n_layers : int,
        init_std : float = 0.02,
        residual_projection_names : tuple[str, ...] = (),
    )-> None:

    """
    Initialize weights of a LLaMA-style Transformer model.

    Standard linear and embedding weights use normal initialization.
    RMSNorm weights are initialized to ones.
    Linear biases are initialized to zeros.

    Selected residual output projections use scaled initialization:
        std = init_std / sqrt(2 * n_layers)

    Args:
        model: Model whose parameters need initialization.
        n_layers: Number of Transformer layers.
        init_std: Base standard deviation for normal initialization.
        residual_projection_names: Exact named module paths of linear
            layers that require residual-scaled initialization.
    """
    if not isinstance(n_layers, int) or isinstance(n_layers, bool):
        raise TypeError("n_layers must be an integer.")
    
    if n_layers <= 0:
        raise ValueError("n_layers must be positive")

    if init_std <= 0:
        raise ValueError("init_st must be positive")

    residual_std = init_std / math.sqrt(2 * n_layers)
    residual_names = set(residual_projection_names)

    linear_names = {
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    }

    unknown_names = residual_names - linear_names

    if unknown_names:
        raise ValueError(f"Unknown residual names {sorted(unknown_names)}")

    for name, module in model.named_modules():
        if isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean= 0.0,
                std= init_std
            )

        elif isinstance(module, nn.Linear):
            std = residual_std if name in residual_names else init_std
            nn.init.normal_(
                    module.weight,
                    mean= 0.0,
                    std= std
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    logger.info(
        "Initialized LLaMA weights | layers={} | "
        "init_std={:.5f} | residual_std={:.6f} | "
        "residual_projections={}",
        n_layers,
        init_std,
        residual_std,
        len(residual_names),
    )

        
