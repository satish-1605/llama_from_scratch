import pytest
import torch

from configs.model_config import ModelConfig
from src.models.llama import Llamma

@pytest.fixture
def config():
    return ModelConfig(vocab_size=256, hidden_size=64, num_layers=2,
        num_attn_heads=4,
        num_kv_heads=2,
        intermediate_size=128,
        max_seq_len=64,
        tie_word_embedding=True,)

@pytest.fixture
def model(config):
    return Llamma(config)

def test_output_forward(model):
    input_ids = torch.randint(0, 256, (2, 32))
    logits = model(input_ids)
    assert logits.shape == (2, 32, 256)
    assert torch.isfinite(logits).all()

def test_single_batch_forward(model):
    input_ids = torch.randint(0, 256, (1, 16))
    logits = model(input_ids)
    assert logits.shape == (1, 16, 256)

def test_number_of_decoder_layers(model):

    assert len(model.layers) == 2

def test_tied_embeddings(model):

    assert (
        model.token_embedding.weight.data_ptr()
        == model.lm_head.weight.data_ptr()
    ) if hasattr(model.lm_head, "weight") else True

def test_parameter_count(model):

    stats = model.count_parameters()

    assert stats["total_parameters"] > 0

    assert (
        stats["total_parameters"]
        == stats["trainable_parameters"]
    )

def gradient_flow_test(model):
    input_ids = torch.randint(0, 256, (2, 16))
    logits = model(input_ids)
    loss = logits.mean()
    loss.backward()
    assert model.token_embedding.weight.grad is not None

    assert model.final_norm.weight.grad is not None

def test_invalid_input_dimensions(model):

    input_ids = torch.randint(0, 256, (2, 3, 4))

    with pytest.raises(ValueError):

        model(input_ids)

def test_invalid_input_dtype(model):

    input_ids = torch.randint(
        0, 256, (2, 16)
    ).float()

    with pytest.raises(TypeError):

        model(input_ids)

def test_sequence_length_exceeded(model):

    input_ids = torch.randint(0, 256, (2, 65))

    with pytest.raises(ValueError):

        model(input_ids)

def test_token_id_out_of_range(model):

    input_ids = torch.randint(0, 256, (2, 16))

    input_ids[0, 0] = 256

    with pytest.raises(ValueError):

        model(input_ids)

def test_empty_batch(model):

    input_ids = torch.empty((0, 16), dtype=torch.long)

    with pytest.raises(ValueError):

        model(input_ids)

def test_empty_sequence(model):

    input_ids = torch.empty((2, 0), dtype=torch.long)

    with pytest.raises(ValueError):

        model(input_ids)

def test_untied_embeddings():

    config = ModelConfig(
        vocab_size=256,
        hidden_size=64,
        num_layers=2,
        num_attn_heads=4,
        num_kv_heads=2,
        intermediate_size=128,
        max_seq_len=64,
        tie_word_embedding=False,
    )

    model = Llamma(config)

    assert model.lm_head.weight.data_ptr() != (
        model.token_embedding.weight.data_ptr()
    )

    input_ids = torch.randint(0, 256, (2, 16))

    logits = model(input_ids)

    assert logits.shape == (2, 16, 256)

    