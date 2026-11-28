import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from loguru import logger

def create_warmup_cosine_scheduler(
        optimizer : Optimizer,
        total_steps : int,
        warmup_steps : int,
        min_lr_ratio : float = 0.1)-> LambdaLR:

    """
    Create a linear warmup + cosine decay learning rate scheduler.

    Parameters:
        optimizer: Optimizer whose learning rate will be scheduled.
        total_steps: Total number of optimizer update steps.
        warmup_steps: Number of optimizer steps used for linear warmup.
        min_lr_ratio: Minimum learning rate as a fraction of the
                      initial learning rate. Must be in [0, 1].

    Returns:
        Configured PyTorch LambdaLR scheduler.

    Raises:
        TypeError: If input parameter types are invalid.
        ValueError: If input parameter values are invalid.
    """

    # Input validation
    if not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be an instance of torch.optim.Optimizer.")

    if isinstance(total_steps, bool) or not isinstance(total_steps, int):
        raise TypeError("total_steps must be an integer.")

    if total_steps < 2:
        raise ValueError("total_steps must be at least 2.")

    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int):
        raise TypeError("warmup_steps must be an integer.")

    if not 0 <= warmup_steps < total_steps:
        raise ValueError(
            "warmup_steps must be greater than or equal to zero "
            "and smaller than total_steps."
        )

    if (
        isinstance(min_lr_ratio, bool)
        or not isinstance(min_lr_ratio, (int, float))
    ):
        raise TypeError("min_lr_ratio must be a numeric value.")

    if not math.isfinite(min_lr_ratio):
        raise ValueError("min_lr_ratio must be finite.")

    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in the range [0, 1].")

    # Learning rate multiplier

    def lr_lambda(current_step:int)->float:
        """
        Calculate the learning rate multiplier for the current step.

        The returned multiplier is applied to the optimizer's
        initial learning rate by PyTorch LambdaLR.

        Parameters:
            current_step: Current scheduler step.

        Returns:
            Learning rate multiplier.
        """

        #Linear warmup
        if warmup_steps > 0 and current_step < warmup_steps:
            return (current_step + 1) / warmup_steps

        # Cosine decay.
        decay_steps = total_steps - warmup_steps

        if warmup_steps == 0:
            progress = current_step / (total_steps -1)
        else:
            progress = (current_step - warmup_steps + 1) / decay_steps

        # Keep the progress within [0, 1].
        progress = min(max(progress, 0.0), 1.0)

        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor

    # Scheduler initialization 
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    logger.info(
        "Warmup + Cosine LR Scheduler initialized | "
        f"Total Steps={total_steps:,} | "
        f"Warmup Steps={warmup_steps:,} | "
        f"Minimum LR Ratio={min_lr_ratio}"
    )

    return scheduler      
    