"""
Single-GPU training utilities.

This module provides a Trainer class for training decoder-only
language models using an optimizer, learning rate scheduler,
cross-entropy loss, and gradient clipping.

The trainer supports two batch formats:

    1. A single token tensor of shape [B, T+1].
       Input and target sequences are created internally.

    2. A tuple containing:
       - input_ids: [B, T]
       - targets:   [B, T]
"""

from typing import Iterable, List, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from loguru import logger

BatchType = Union(Tensor, Tuple[Tensor, Tensor])

class trainer:
    pass
