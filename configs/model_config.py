from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size :int =  32000
    hidden_size : int = 384
    num_layers : int = 6

    num_attn_heads : int = 6
    num_kv_heads : int = 2

    intermediate_size : int = 1024
    max_seq_len : int = 512
    norm_eps : float = 1e-5

    rope_theta : float = 10000.0

    tie_word_embedding : bool = True
    attention_bias : bool = False
    mlp_bias : bool = False

    embedding_init_std : float = 0.02

    def __post_init__(self):
        assert self.hidden_size % self.num_attn_heads == 0
        assert self.num_attn_heads % self.num_kv_heads == 0
        assert self.hidden_size % self.num_attn_heads == 0

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attn_heads

    @property
    def num_query_groups(self):
        return self.num_attn_heads // self.num_kv_heads
