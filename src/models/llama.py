from loguru import logger
import torch
import torch.nn as nn

from configs.model_config import ModelConfig
from src.models.embeddings import TokenEmbedding, TiedLMHead
from src.models.initialization import initialize_llama_weights
from src.models.gqa import GroupedQueryAttention
from src.models.swiglu import SwiGLU
from src.models.normalization import RMSNorm
from src.models.transformer_block import TransformerBlock


class Llamma(nn.Module):
    """
    Complete LLaMA-style decoder-only Transformer language model.

    Architecture:
        Token Embedding
        -> N Decoder Transformer Blocks
        -> Final RMSNorm
        -> Language Modeling Head

    Supports tied or untied input/output word embeddings.
    """
    def __init__(self, config:ModelConfig)-> None:
        """
        Initialize the complete LLaMA model.

        Args:
            config: ModelConfig containing architecture parameters.

        Raises:
            TypeError: If config is not a ModelConfig instance.
            ValueError: If configuration dimensions are invalid.
        """
        super().__init__()
        if not isinstance(config, ModelConfig):
            raise ValueError("config must be an instance of ModelConfig")

        if config.vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")

        if config.hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")

        if config.num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        if config.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")

        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.max_seq_len = config.max_seq_len

        # 1. Token Embedding 
        self.token_embedding = TokenEmbedding(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size
        )

        # 2. Decoder Transformer Blocks
        self.layers = nn.ModuleList()

        for layer_id in range(config.num_layers):
            attention = GroupedQueryAttention(
                hidden_dim= config.hidden_size,
                num_query_heads= config.num_attn_heads,
                num_kv_heads= config.num_kv_heads,
                dropout= 0.1,
                rope_base= config.rope_theta
            )

            feed_forward = SwiGLU(
                d_model=config.hidden_size,
                hidden_dim= config.intermediate_size
            )

            norm1 = RMSNorm(dim = config.hidden_size,
                            eps= config.norm_eps)

            norm2 = RMSNorm(dim = config.hidden_size,
                            eps= config.norm_eps)

            block = TransformerBlock(
                d_model= config.hidden_size,
                attention= attention,
                feed_forward= feed_forward,
                norm1= norm1,
                norm2 = norm2
            )

            self.layers.append(block)

        # 3. Final RMSNorm
        self.final_norm = RMSNorm(
            dim = config.hidden_size,
            eps= config.norm_eps
        )

        # 4. Language Modeling Head
        self.tie_word_emebdding = config.tie_word_embedding

        if self.tie_word_emebdding:
            self.lm_head = TiedLMHead()

        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias= False)

        logger.info(
            "Initialized LlamaModel | "
            "vocab_size={} | hidden_size={} | "
            "num_layers={} | num_attention_heads={} | "
            "num_kv_heads={} | tied_embeddings={}",
            config.vocab_size,
            config.hidden_size,
            config.num_layers,
            config.num_attn_heads,
            config.num_kv_heads,
            config.tie_word_embedding,
        )

        residual_projection_names = self._get_residual_projection_names()

        initialize_llama_weights(model= self, 
                                n_layers=config.num_layers,
                                init_std = config.embedding_init_std,
                                residual_projection_names=residual_projection_names
                                )

    def _get_residual_projection_names(self) -> tuple[str, ...]:
        """
        Identify output projections of attention and feed-forward
        residual branches.
        """

        residual_suffixes = {"o_proj", "w2"}

        names = tuple(
            name
            for name, module in self.named_modules()
            if isinstance(module, nn.Linear)
            and name.rsplit(".", 1)[-1] in residual_suffixes
        )

        expected_count = 2 * self.num_layers

        if len(names) != expected_count:
            raise ValueError(
                "Expected exactly two residual projections per decoder "
                f"layer ({expected_count}), but found {len(names)}. "
                f"Detected names: {names}. "
                "Verify GQA and SwiGLU projection names."
            )
        return names


    def forward(self, input_ids:torch.Tensor)->torch.Tensor:
        """
        Execute the complete decoder-only Transformer forward pass.

        Args:
            input_ids:
                Token IDs of shape [B, T].

        Returns:
            Logits of shape [B, T, vocab_size].
        """

        # Input Validation
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("input_ids must be a torch.Tensor.")

        if input_ids.ndim != 2:
            raise ValueError(
                "Expected input_ids shape [B, T], "
                f"received {tuple(input_ids.shape)}."
            )

        if input_ids.dtype != torch.long:
            raise TypeError(
                f"Expected torch.long input IDs, got {input_ids.dtype}."
            )

        batch_size, seq_len = input_ids.shape

        if batch_size == 0 or seq_len == 0:
            raise ValueError("Batch size and sequence length must be positive.")

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"maximum supported length {self.max_seq_len}."
            )

        if torch.any(input_ids < 0) or torch.any(input_ids >= self.vocab_size):
            raise ValueError(
                f"Token IDs must be in range [0, {self.vocab_size - 1}]."
            )

        # 1. Token Embedding
        # [B, T] -> [B, T, hidden_size]
        hidden_states = self.token_embedding(input_ids)

        # 2. Decoder Transformer Blocks
        # [B, T, hidden_size] -> [B, T, hidden_size]
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        # 3. Final RMSNorm
        hidden_states = self.final_norm(hidden_states)

        # 4. Language Modeling Head
        if self.tie_word_emebdding:
            logits = self.lm_head(hidden_states,
                                  self.token_embedding.weight)
            
        else:
            logits = self.lm_head(hidden_states)

        return logits

    def count_parameters(self)->dict:
        """
        Calculate total and trainable model parameters.

        Returns:
            Dictionary containing total and trainable parameter counts.
        """

        total_params = sum(parameter.numel() for parameter in self.parameters())
        trainable_params = sum(parameter.numel() 
                               for parameter in self.parameters()
                               if parameter.requires_grad)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "total_parameters_millions": total_params / 1e6,
            "trainable_parameters_millions": trainable_params / 1e6,
        } 


            


        
