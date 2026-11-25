import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenEmbedding(nn.Module):
    """convert token id into learnable dense representation"""

    def __init__(self, vocab_size:int, hidden_size:int):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim= hidden_size
        )

    def forward(self, input_ids: torch.Tensor)-> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"Input ids shape must be [B, T], got {input_ids.shape}")

        return self.embedding(input_ids)

    @property
    def weight(self):
        return self.embedding.weight

class TiedLMHead(nn.Module):
    """Project hidden states to vocabulary logits using embedding weights."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        embedding_weight: torch.Tensor,
    ) -> torch.Tensor:

        if hidden_states.shape[-1] != embedding_weight.shape[-1]:
            raise ValueError("Hidden dimension and embedding dimension mismatch")

        return F.linear(hidden_states, embedding_weight)
    
        