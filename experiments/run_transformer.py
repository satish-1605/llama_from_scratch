#####
import torch
from src.models.transformer_block import TransformerBlock
from src.models.gqa import GroupedQueryAttention
from src.models.swiglu import SwiGLU
from src.models.normalization import RMSNorm

hidden_dim = 384
num_query_heads = 6
num_kv_heads = 6
d_ff = 3 * 384


attention = GroupedQueryAttention(384, 6, 2)
ffn = SwiGLU(d_model=hidden_dim, hidden_dim=d_ff)
norm1 = RMSNorm(hidden_dim)
norm2 = RMSNorm(hidden_dim)

block = TransformerBlock(
    d_model= 384,
    attention=attention,
    feed_forward=ffn,
    norm1= norm1,
    norm2= norm2
)
x = torch.randn(2, 16, 384)
output = block(x)

print(output.shape)