from typing import Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW

from loguru import logger

def create_adamw_optimizer(
        model : nn.Module,
        lr : float = 3e-4,
        betas : Tuple[float, float] = (0.9, 0.999),
        eps : float = 1e-8,
        weight_decay : float = 0.01)-> AdamW:
    """
    Create an AdamW optimizer with separate weight decay groups.

    Parameters:
        model: PyTorch model whose parameters will be optimized.
        lr: Initial learning rate.
        betas: Coefficients used for first and second moment estimates.
        eps: Numerical stability constant.
        weight_decay: Weight decay coefficient for eligible parameters.

    Returns:
        Configured PyTorch AdamW optimizer.

    Raises:
        TypeError: If model or hyperparameter types are invalid.
        ValueError: If hyperparameter values are invalid or
                    the model has no trainable parameters.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be an instance of nn.Module.")

    if isinstance(lr, bool) or not isinstance(lr, (int, float)):
        raise TypeError("lr must be a numeric value.")

    if lr <= 0:
        raise ValueError("lr must be greater than zero.")

    if not isinstance(betas, (tuple, list)) or len(betas) != 2:
        raise ValueError("betas must contain exactly two values.")

    for beta in betas:
        if isinstance(beta, bool) or not isinstance(beta, (int, float)):
            raise TypeError("Each beta must be numeric.")

        if not 0.0 <= beta < 1.0:
            raise ValueError("Each beta must be in [0, 1).")

    if isinstance(eps, bool) or not isinstance(eps, (int, float)):
        raise TypeError("eps must be numeric.")

    if eps <= 0:
        raise ValueError("eps must be greater than zero.")

    if (
        isinstance(weight_decay, bool)
        or not isinstance(weight_decay, (int, float))
    ):
        raise TypeError("weight_decay must be numeric.")

    if weight_decay < 0:
        raise ValueError("weight_decay cannot be negative.")

    decay_params = []
    no_decay_params = []

    decay_count = 0
    no_decay_count = 0

    for name , param in model.named_parameters():
        # Ignore frozen parameters.
        if not param.requires_grad:
            continue

        if param.ndim >= 2:
            decay_params.append(param)
            decay_count += param.numel()
        else:
            no_decay_params.append(param)
            no_decay_count += param.numel()

    if not decay_params and not no_decay_params:
        raise ValueError("Model has no trainable parameters.")

    parameter_groups = []

    if decay_params:
        parameter_groups.append(
            {
                "params": decay_params,
                "weight_decay": weight_decay,
            }
        )

    if no_decay_params:
        parameter_groups.append(
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            }
        )

    optimizer = AdamW(
        parameter_groups,
        lr=lr,
        betas= tuple(betas),
        eps=eps
    )

    logger.info(
        "AdamW optimizer initialized | "
        f"LR={lr} | Betas={betas} | "
        f"Epsilon={eps} | Weight Decay={weight_decay}"
    )

    logger.info(
        f"Decay parameters: {decay_count:,} | "
        f"No-decay parameters: {no_decay_count:,} | "
        f"Total trainable parameters: "
        f"{decay_count + no_decay_count:,}"
    )

    return optimizer

