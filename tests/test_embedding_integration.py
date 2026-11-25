"""Integration testing for token embedding and tied lm head"""

import torch
from datasets import load_from_disk

from src.model.embeddings import TokenEmbedding, TiedLMHead

def test_embedding_to_logits_integration():
    """Vefifying complete packed data, embedding and vocab logits flow """

    # Load the existing training dataset
    dataset = load_from_disk("artifacts/processed/pretraining/train")

    # use a small batch from saved dataset
    batch_size = 2
    samples = dataset.select(range(batch_size))

    input_ids = torch.tensor(samples['input_ids'], dtype= torch.long)

    # mode config 
    vocab_size = 32000
    hidden_size = 384
    context_length = 512

    # inti embedding
    token_embedding = TokenEmbedding(
        vocab_size =vocab_size, hidden_size =hidden_size)

    lm_head = TiedLMHead()

    #forward pass
    hidden_states = token_embedding(input_ids)

    # Verify embedding output.
    assert input_ids.shape == (batch_size, context_length)
    assert hidden_states.shape == (batch_size, context_length, hidden_size)

    # use the same embdding projection for vocab projection
    logits = lm_head(hidden_states, token_embedding.weight)

    # Verify logits output.
    assert logits.shape == (batch_size, context_length, vocab_size)

    # Verify that the LM head uses the exact embedding parameter.
    assert token_embedding.weight.data_ptr() == token_embedding.embedding.weight.data_ptr()

    print("Integration test passed!")
    print(f"Input shape: {tuple(input_ids.shape)}")
    print(f"Embedding shape: {tuple(hidden_states.shape)}")
    print(f"Logits shape: {tuple(logits.shape)}")
