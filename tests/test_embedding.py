
import torch

from src.model.embeddings import TokenEmbedding, TiedLMHead


def test_embedding_output_shape():
    embedding = TokenEmbedding(32000, 384)

    input_ids = torch.randint(0, 32000, (2, 512))
    output = embedding(input_ids)

    assert output.shape == (2, 512, 384)


def test_embedding_parameter_count():
    embedding = TokenEmbedding(32000, 384)

    params = sum(p.numel() for p in embedding.parameters())

    assert params == 32000 * 384


def test_tied_lm_head_output_shape():
    embedding = TokenEmbedding(32000, 384)
    lm_head = TiedLMHead()

    hidden_states = torch.randn(2, 512, 384)

    logits = lm_head(hidden_states, embedding.weight)

    assert logits.shape == (2, 512, 32000)


def test_tied_weights_receive_gradients():
    embedding = TokenEmbedding(32000, 384)
    lm_head = TiedLMHead()

    hidden_states = torch.randn(2, 4, 384)

    logits = lm_head(hidden_states, embedding.weight)
    logits.mean().backward()

    assert embedding.weight.grad is not None
    assert embedding.weight.grad.abs().sum() > 0
