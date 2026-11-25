from transformers import AutoTokenizer
from loguru import logger

class HFTokenizer:
    def __init__(self, model_name: str, expected_vocab_size:int =32000):
        logger.info("Loading tokenizer: {}", model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast = True
        )

        self.vocab_size = len(self.tokenizer)

        logger.success("Tokenizer loaded successfully")
        logger.info("Vocab size {}", self.vocab_size)

        if self.vocab_size != expected_vocab_size:
            raise ValueError(
                f"Vocab mismatch !"
                f"Expected {expected_vocab_size}"
                f"got {self.vocab_size}"
                f"Update model confif accordingly"
            )

        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        self.unk_token = self.tokenizer.unk_token

        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.unk_token_id = self.tokenizer.unk_token_id

        if self.eos_token_id is None:
            raise ValueError("Tokenizer EOS token is missing.")

        logger.info("BOS token {} | ID: {}", self.bos_token, self.bos_token_id)
        logger.info("EOS token {} | ID: {}", self.eos_token, self.eos_token_id)
        logger.info("UNK token {} | ID: {}", self.unk_token, self.unk_token_id)

    def encode(self, text:str, add_special_tokens: bool = False):
        return self.tokenizer.encode(
            text,
            add_special_tokens = add_special_tokens
        )

    def decode(self, token_ids, skip_special_tokens: bool = False):
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens= skip_special_tokens
        )

    def encode_with_eos(self, text):
        token_ids = self.encode(text)

        if not token_ids or token_ids[-1] != self.eos_token_id:
            token_ids.append(self.eos_token_id)

        return token_ids

    def tokenize(self, text:str):
        return self.tokenizer.tokenize(text)
    

    def convert_ids_to_tokens(self, token_ids):
        return self.tokenizer.convert_ids_to_tokens(token_ids)

    
