import torch
from loguru import logger

from configs.model_config import ModelConfig
from src.models.llama import Llamma


def main():
    config = ModelConfig()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    logger.info("Using device: {}", device)

    #model init
    model = Llamma(config).to(device)

    model.eval()

    model.count_paramters()
    
    # Dummy Input
    batch_size = 2
    seq_len = 128

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long,
        device=device,
    )

    with torch.inference_mode():
        logits = model(input_ids)

    expected_shape = (batch_size, seq_len, config.vocab_size)

    assert logits.shape == expected_shape
    assert torch.isfinite(logits).all()

    logger.success("Forward pass successful!")
    logger.info("Input shape: {}", tuple(input_ids.shape))
    logger.info("Logits shape: {}", tuple(logits.shape))

    next_token_logits = logits[:, -1, :]

    next_token = torch.argmax(
        next_token_logits,
        dim=-1,
    )

    logger.info(
        "Next token IDs: {}",
        next_token.tolist(),
    )

    logger.success("LLaMA integration smoke test completed.")

if __name__ == "__main__":
    main()