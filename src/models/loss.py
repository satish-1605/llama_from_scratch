import torch
import torch.nn as nn

class CausalLMLoss(nn.Module):
    """Calculare next token prediction loss using cross entropy"""

    def __init__(self, ignore_index:int = -100):
        """Initialize cross entropy loss with an optional ignored label"""
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self,
                logits: torch.Tensor,
                labels: torch.Tensor
                )-> torch.Tensor:
        """
        Calculathe the cross entropy loss from predicted logits and target labels.

        Args:
            logits : Predicted vocab logits with shape [B, T, V]
            labels : Target labels with shape [B, T]

        Returns:
            Scaler cross entropy loss.

        Raises:
            Value error if input dim or shapes are invalid.
        """

        if logits.ndim != 3:
            raise ValueError(f"Expected logits with shape [B, T, V], got {tuple(logits.shape)} ")

        if labels.ndim != 2:
                    raise ValueError(f"Expected labels with shape [B, T], got {tuple(labels.shape)} ")

        if logits.shape[:2] != labels.shape:
              raise ValueError(f"Logits and labels dimensions do not match, {tuple(logits.shape[:2])} vs {tuple(labels.shape)}")

        if labels.dtype != torch.long:
            raise ValueError(f"Expected labels dtype torch.long, received {labels.dtype}")

        if logits.shape[-1] == 0:
            raise ValueError("Vocabulary dimension cannot be zero.")

        batch_size, seq_len , vocab_size = logits.shape

        # Flatten batch and seq dim for CE loss
        logits = logits.reshape(batch_size * seq_len, vocab_size)
        labels = labels.reshape(batch_size * seq_len)

        return self.loss_fn(logits, labels)

        

        
