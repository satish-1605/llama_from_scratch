
from configs.model_config import ModelConfig
from configs.train_config import TrainConfig
from loguru import logger


model_config = ModelConfig()
train_config = TrainConfig()


def calculate_parameters(model: ModelConfig):

    hidden = model.hidden_size
    intermediate = model.intermediate_size
    layers = model.num_layers
    vocab = model.vocab_size

    num_heads = model.num_attn_heads
    num_kv_heads = model.num_kv_heads
    head_dim = model.head_dim

    # 1. Token embeddings
    embedding_params = vocab * hidden

    # 2. Attention projections per layer
    q_proj = hidden * (num_heads * head_dim)
    k_proj = hidden * (num_kv_heads * head_dim)
    v_proj = hidden * (num_kv_heads * head_dim)
    o_proj = (num_heads * head_dim) * hidden

    attention_per_layer = q_proj + k_proj + v_proj + o_proj

    # 3. SwiGLU MLP per layer
    gate_proj = hidden * intermediate
    up_proj = hidden * intermediate
    down_proj = intermediate * hidden

    mlp_per_layer = gate_proj + up_proj + down_proj

    # 4. Transformer layers
    attention_total = attention_per_layer * layers
    mlp_total = mlp_per_layer * layers

    # 5. RMSNorm parameters
    # Two RMSNorms per block + one final RMSNorm
    norm_params = (2 * layers + 1) * hidden

    # 6. Total parameters
    total_params = (
        embedding_params
        + attention_total
        + mlp_total
        + norm_params
    )

    return {
        "Token Embeddings": embedding_params,
        "Attention": attention_total,
        "SwiGLU MLP": mlp_total,
        "RMSNorm": norm_params,
        "Total Parameters": total_params
    }


if __name__ == "__main__":

    logger.info("========== MODEL CONFIGURATION ==========")
    logger.info("{}", model_config)

    logger.info("========== TRAINING CONFIGURATION ==========")
    logger.info("{}", train_config)

    logger.info("========== DERIVED VALUES ==========")
    logger.info("Head Dimension: {}", model_config.head_dim)
    logger.info("Query Groups: {}", model_config.num_query_groups)

    logger.info("========== PARAMETER COUNT ==========")

    params = calculate_parameters(model_config)

    for name, count in params.items():
        logger.info("{}: {:,}", name, count)

    logger.info(
        "Model Size: {:.2f} Million Parameters",
        params["Total Parameters"] / 1_000_000
    )

    logger.success("Configuration validation completed!")
