from loguru import logger
import torch
import math
import torch.nn as nn

class GroupedQueryAttention(nn.Module):
    """
    Implements GQA projections and tensor transformations.

    Multiple Query heads share a single Key and Value head.

    Args:
        hidden_dim (int): Input and output hidden dimension.
        num_query_heads (int): Number of Query heads.
        num_kv_heads (int): Number of Key/Value heads.

    Attributes:
        head_dim (int): Dimension of each attention head.
        num_query_groups (int): Query heads per KV head.
        kv_dim (int): Combined dimension of all KV heads.

    Expected input:
        [batch_size, sequence_length, hidden_dim]

    Query output:
        [batch_size, num_query_heads, sequence_length, head_dim]

    Key and Value output:
        [batch_size, num_kv_heads, sequence_length, head_dim]

    Grouped Query output:
        [batch_size, num_kv_heads, num_query_groups,
         sequence_length, head_dim]
    """

    def __init__(self, hidden_dim:int, num_query_heads:int, num_kv_heads:int)->None:
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be positive, got {hidden_dim}"
            )

        if num_query_heads <= 0:
            raise ValueError(
                f"num_query_heads must be positive, "
                f"got {num_query_heads}"
            )

        if num_kv_heads <= 0:
            raise ValueError(
                f"num_kv_heads must be positive, got {num_kv_heads}"
            )

        if hidden_dim % num_query_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible "
                f"by num_query_heads ({num_query_heads})."
            )

        if num_query_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_query_heads ({num_query_heads}) must be "
                f"divisible by num_kv_heads ({num_kv_heads})."
            )

        self.hidden_dim = hidden_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads

        self.head_dim = hidden_dim // num_query_heads
        self.num_query_groups = (num_query_heads // num_kv_heads)
        self.kv_dim = num_kv_heads * self.head_dim

        # query projection D -> Hq * Dh
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # k and v  projection D -> Hkv * Dh
        self.k_proj = nn.Linear(hidden_dim, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, self.kv_dim, bias=False)

        # output proj D -> D
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        logger.info(
            "Initialized GQA: hidden_dim {}, "
            "query_heads={}, kv_heads={}, "
            "head_dim={}, query_groups={}",
            self.hidden_dim,
            self.num_query_heads,
            self.num_kv_heads,
            self.head_dim,
            self.num_query_groups,
        )

    def split_query_heads(self, x:torch.Tensor)->torch.Tensor:
        """
        Split Query projection into multiple attention heads.

        Input:
            [B, T, D]

        Output:
            [B, Hq, T, Dh]
        """

        if x.ndim != 3:
            raise ValueError(f"Expected a 3D tensor [B, T, D], got {x.shape}")

        batch_size, seq_len, hidden_dim = x.shape

        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Expected hidden dimension {self.hidden_dim}, "
                f"got {hidden_dim}"
            )
        # [B, T, D] => [B, T, Hq, Dh]
        x = x.reshape(batch_size, seq_len, self.num_query_heads, self.head_dim)

        # [B, T, Hq, Dh] => [B, Hq, T, Dh]
        x = x.transpose(1, 2)

        return x

    def split_kv_heads(self, x:torch.Tensor)-> torch.Tensor:
        """
        Split Key or Value projection into KV heads.

        Input:
            [B, T, Hkv * Dh]

        Output:
            [B, Hkv, T, Dh]
        """

        if x.ndim != 3:
            raise ValueError(
                f"Expected a 3D tensor [B, T, Dkv], got {x.shape}"
            )

        batch_size, seq_len, kv_dim = x.shape

        if kv_dim != self.kv_dim:
            raise ValueError(
                f"Expected KV dimension {self.kv_dim}, got {kv_dim}"
            )

        x = x.reshape(
            batch_size,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        )

        x = x.transpose(1, 2)

        return x

    def group_query_heads(self, q:torch.Tensor)-> torch.Tensor:
        """
        Organize Query heads according to their shared KV heads.

        Input:
            [B, Hq, T, Dh]

        Output:
            [B, Hkv, G, T, Dh]

        G represents the number of Query heads per KV head.

        This is a logical reshape only. It does not modify values
        or repeat Key/Value tensors.
        """

        if q.ndim != 4:
            raise ValueError(
                f"Expected a 4D tensor [B, Hq, T, Dh], got {q.shape}"
            )

        batch_size, num_heads, seq_len, head_dim = q.shape

        if num_heads != self.num_query_heads:
            raise ValueError(
                f"Expected {self.num_query_heads} Query heads, "
                f"got {num_heads}"
            )

        if head_dim != self.head_dim:
            raise ValueError(
                f"Expected head_dim {self.head_dim}, got {head_dim}"
            )

        q =q.reshape(batch_size, 
                     self.num_kv_heads, 
                     self.num_query_groups, 
                     seq_len, 
                     self.head_dim)

        return q

    def merge_query_heads(self, q:torch.Tensor)->torch.Tensor:
        """
        Merge Query heads back into the hidden dimension.

        Input:
            [B, Hq, T, Dh]

        Output:
            [B, T, D]
        """

        if q.ndim != 4:
            raise ValueError(
                f"Expected a 4D tensor [B, Hq, T, Dh], got {q.shape}"
            )

        batch_size, num_heads, seq_len, head_dim = q.shape

        if num_heads != self.num_query_heads:
            raise ValueError(
                f"Expected {self.num_query_heads} Query heads, "
                f"got {num_heads}"
            )

        if head_dim != self.head_dim:
            raise ValueError(
                f"Expected head_dim {self.head_dim}, got {head_dim}"
            )

        q = q.transpose(1, 2).contiguous()
        q = q.reshape(
            batch_size,
            seq_len,
            self.hidden_dim,
        )
        return q

    def project_output(self, x:torch.Tensor)->torch.Tensor:
        """
        Merge Query heads and apply the output projection.

        This method expects the output of attention computation
        to have shape [B, Hq, T, Dh].
        """

        x = self.merge_query_heads(x)

        x = self.o_proj(x)

        return x

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        """
        Generate Query, Key and Value representations.

        Args:
            x: Input tensor [B, T, D].

        Returns:
            q: [B, Hq, T, Dh]
            k: [B, Hkv, T, Dh]
            v: [B, Hkv, T, Dh]

        Note:
            Actual attention computation is not included.
        """

        if x.ndim != 3:
            raise ValueError(
                f"Expected input [B, T, D], got {x.shape}"
            )

        if x.size(-1) != self.hidden_dim:
            raise ValueError(
                f"Expected hidden dimension {self.hidden_dim}, "
                f"got {x.size(-1)}"
            )

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self.split_query_heads(q)
        k = self.split_kv_heads(k)
        v = self.split_kv_heads(v)

        logger.debug(
            "GQA tensor shapes: Q={}, K={}, V={}",
            tuple(q.shape),
            tuple(k.shape),
            tuple(v.shape),
        )

        return q, k, v










