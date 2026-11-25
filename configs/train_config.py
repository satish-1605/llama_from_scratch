from dataclasses import dataclass

@dataclass
class TrainConfig:
    batch_size :int = 16
    gradient_accumulation_step : int = 4

    learning_rate : float = 3e-4
    weight_decay : float = 0.1
    betas : tuple = (0.9, 0.95)
    eps : float = 1e-8

    max_steps : int = 10000
    warmup_steps : int = 500

    max_grad_norm : float = 1.0

    eval_interval : int = 500
    eval_steps : int = 50

    save_interval : int = 1000
    dtype : str = "bfloat16"

    seed : int = 42

    