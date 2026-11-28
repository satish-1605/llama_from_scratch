from src.tokenizer.hf_tokenizer import HFTokenizer
from loguru import logger

TOKENIZER_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def main():
    tokenizer = HFTokenizer(
        model_name= TOKENIZER_NAME,
        expected_vocab_size=32000
    )

    text = "l love deep learning and building LLMs."
    logger.info("Original text: {}", text)

    # convert text into subword tokens
    tokens = tokenizer.tokenize(text)
    logger.info("Tokens: {}", tokens)

    # encode text into tokens ids
    token_ids = tokenizer.encode(text)
    logger.info("Token IDs: {}", token_ids)

    token_ids_with_eos = tokenizer.encode_with_eos(text)
    logger.info("Token IDs with EOS: {}", token_ids_with_eos)

    # convert the tokens id back into the tokens
    reconstructed_tokens = tokenizer.convert_ids_to_tokens(token_ids)
    logger.info("Tokens from IDs: {}", reconstructed_tokens)

    # Decode tokens ids back into the text
    decoded_text = tokenizer.decode(token_ids)
    logger.info("Decoded text: {}", decoded_text)

    assert len(tokens) == len(token_ids)
    assert decoded_text == text

    logger.success("tokenizer test passed")

if __name__ == "__main__":
    main()



