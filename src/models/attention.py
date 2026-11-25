from loguru import logger
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """
    Implements Q/K/V projections and multi-head tensor transformations.

    Args:
        hidden_dim (int):
            Input and output hidden dimension.

        num_heads (int):
            Number of attention heads.

    Attributes:
        head_dim (int):
            Dimension allocated to each attention head.

        q_proj (nn.Linear):
            Query projection layer.

        k_proj (nn.Linear):
            Key projection layer.

        v_proj (nn.Linear):
            Value projection layer.

        o_proj (nn.Linear):
            Output projection layer.

    Expected input:
        Tensor of shape [batch_size, sequence_length, hidden_dim].

    Projection output:
        Q, K, V of shape [batch_size, sequence_length, hidden_dim].

    Split-head output:
        Q, K, V of shape
        [batch_size, num_heads, sequence_length, head_dim].

    Note:
        This class does not yet calculate attention scores or
        aggregate Value representations.
    """

    def __init__(self, hidden_dim: int, num_heads: int)-> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be positive, got {hidden_dim}"
            )

        if num_heads <= 0:
            raise ValueError(
                f"num_heads must be positive, got {num_heads}"
            )

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible "
                f"by num_heads ({num_heads})."
            )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Independent, bias-free Q/K/V projections.
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias= False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Combines information from all attention heads.
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        logger.info(
            "Initialized Multi head attention:"
            "hidden dim {} | num_heads {} | head_dim {}",
            hidden_dim, num_heads, self.head_dim)

    def split_heads(self, x: torch.Tensor)-> torch.Tensor:
        """
        Split the hidden dimension into multiple attention heads.

        Args:
            x:
                Tensor of shape [B, T, hidden_dim].

        Returns:
            Tensor of shape [B, H, T, head_dim].
        """

        if x.ndim != 3:
            raise ValueError(f"Expected a 3D tensor [B, T, D], recieved {x.shape}")

        batch_size, seq_len, hidden_dim = x.shape

        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Expected hidden dimension {self.hidden_dim}, "
                f"got {hidden_dim}"
            )

        # [B, T, D] -> [B, T, H, Dh]
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        ## [B, T, H, Dh] -> [B, H, T, Dh]
        x = x.transpose(1,2)

        return x

    def merge_heads(self, x:torch.Tensor)->torch.Tensor:
        """
        Combine multiple attention heads into one hidden representation.

        Args:
            x:
                Tensor of shape [B, H, T, head_dim].

        Returns:
            Tensor of shape [B, T, hidden_dim].
        """
        if x.ndim != 4:
            raise ValueError(f"Expected a 4D tensor [B, H, T, Dh], recieved {x.shape}")

        batch_size, num_heads, seq_len, head_dim = x.shape

        if num_heads != self.num_heads:
            raise ValueError(
                f"Expected {self.num_heads} heads, got {num_heads}"
            )

        if head_dim != self.head_dim:
            raise ValueError(
                f"Expected head_dim {self.head_dim}, got {head_dim}"
            )

        # [B, H, T, Dh] -> [B, T, H, Dh]
        x = x.transpose(1, 2)

        # Make memory layout contiguous before reshaping.
        x = x.contiguous()

        # [B, T, H, Dh] -> [B, T, D]
        x = x.reshape(batch_size, seq_len, self.hidden_dim)

        return x

    def project_output(self, x:torch.Tensor)-> torch.Tensor:
        """
        Merge attention heads and apply the output projection.

        Args:
            x:
                Head representations of shape [B, H, T, Dh].

        Returns:
            Tensor of shape [B, T, hidden_dim].
        """

        x = self.merge_heads(x)

        x = self.o_proj(x)

        return x

    def forward(self, x:torch.Tensor)-> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate Query, Key and Value projections and split into heads.

        Args:
            x:
                Input tensor of shape [B, T, hidden_dim].

        Returns:
            Tuple containing:
                q: [B, H, T, head_dim]
                k: [B, H, T, head_dim]
                v: [B, H, T, head_dim]

        Note:
            Attention score calculation and Value aggregation
            are intentionally excluded at this stage.
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape [B, T, D], got {x.shape}"
            )

        if x.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected input hidden dimension {self.hidden_dim}, "
                f"got {x.shape[-1]}"
            )

        # Step 1: Independent linear projections.
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Step 2: Split each projection into attention heads.
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        logger.debug(
            "Q/K/V projection shapes: Q {}, K{}, V{}",
            tuple(q.shape),
            tuple(k.shape),
            tuple(v.shape),
        )

        return q, k, v


class ScaledDotProductAttention(nn.Module):
    """
    Compute scaled dot-product attention with optional causal masking
    and attention dropout.

    Supports:
        - Multi-Head Attention (Hq == Hkv)
        - Grouped Query Attention (Hq > Hkv)
        - Causal self-attention
        - Numerically stable FP32 Softmax

    Args:
        dropout (float): Attention dropout probability, between 0 and 1.
    """

    def __init__(self, dropout : float = 0.0)-> None:
        super().__init__()

        if not 0.0 <= dropout <= 1.0:
            raise ValueError(
                f"dropout must be between 0 and 1, got {dropout}"
            )

        self.dropout = nn.Dropout(p=dropout)
        logger.info("Initialized ScaledDotProductAttention with dropout {}", dropout)

    def _validate_inputs(self, 
                         q: torch.Tensor,
                         k: torch.Tensor,
                         v: torch.Tensor,
                         is_causal: bool)->None:
        """
        Validate Q, K and V tensor shapes and compatibility.

        Args:
            q: Query tensor [B, Hq, Tq, D].
            k: Key tensor [B, Hkv, Tk, D].
            v: Value tensor [B, Hkv, Tk, Dv].
            is_causal: Whether causal masking is enabled.

        Raises:
            ValueError: If tensor shapes are incompatible.
            TypeError: If inputs are not tensors.
        """
        tensors = {"q": q, "k": k, "v": v}

        for name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")

            if tensor.ndim != 4:
                raise ValueError(
                    f"{name} must be a 4D tensor, "
                    f"got shape {tuple(tensor.shape)}"
                )

        batch_q, num_q_heads, seq_q, q_dim = q.shape
        batch_k, num_kv_heads, seq_k, k_dim = k.shape
        batch_v, num_v_heads, seq_v, _ = v.shape

        if batch_q != batch_k or batch_q != batch_v:
            raise ValueError("Q, K and V must have the same batch size")

        if num_kv_heads != num_v_heads:
            raise ValueError("K and V must have the same number of heads")

        if num_q_heads % num_kv_heads != 0:
            raise ValueError(
                "Number of Query heads must be divisible by "
                "number of KV heads"
            )

        if q_dim != k_dim:
            raise ValueError(
                "Query and Key head dimensions must be equal"
            )

        if seq_k != seq_v:
            raise ValueError("K and V must have the same sequence length")

        if is_causal and seq_q != seq_k:
            raise ValueError(
                "Causal attention currently requires equal Query "
                "and Key sequence lengths. Offset-aware causal "
                "masking will be handled with KV cache."
            )

        if q.device != k.device or q.device != v.device:
            raise ValueError("Q, K and V must be on the same device")

        if q.dtype != k.dtype or q.dtype != v.dtype:
            raise ValueError("Q, K and V must have the same dtype")

        if not q.is_floating_point():
            raise TypeError("Q, K and V must have floating-point dtype")

    def _expand_kv_heads(self,
                         k:torch.Tensor,
                         v: torch.Tensor,
                         num_query_heads : int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Expand KV heads to match the number of Query heads.

        For example:
            Query heads = 6
            KV heads = 2
            Groups = 3

        Each KV head is repeated three times.

        Args:
            k: Key tensor [B, Hkv, T, D].
            v: Value tensor [B, Hkv, T, Dv].
            num_query_heads: Number of Query heads.

        Returns:
            Expanded K and V tensors:
                K: [B, Hq, T, D]
                V: [B, Hq, T, Dv]
        """

        num_kv_heads = k.shape[1]
        if num_query_heads == num_kv_heads:
            return k, v

        groups = num_query_heads // num_kv_heads

        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim = 1)

        logger.debug(
            "Expanded KV heads: {} -> {} using {} groups",
            num_kv_heads,
            num_query_heads,
            groups,
        )

        return k, v

    def forward(self, 
                q: torch.Tensor,
                k:torch.Tensor,
                v: torch.Tensor,
                is_causal:bool = False)-> torch.Tensor:
        """
        Compute scaled dot-product attention.

        Steps:
            1. Validate inputs.
            2. Expand K and V heads for GQA.
            3. Compute QK transpose.
            4. Scale attention scores.
            5. Apply causal mask if enabled.
            6. Compute numerically stable Softmax in FP32.
            7. Apply attention dropout.
            8. Multiply attention weights with V.

        Args:
            q: Query tensor [B, Hq, T, D].
            k: Key tensor [B, Hkv, T, D].
            v: Value tensor [B, Hkv, T, Dv].
            is_causal: Whether to prevent attention to future tokens.

        Returns:
            Attention output [B, Hq, T, Dv].
        """

        self._validate_inputs(q, k, v, is_causal)
        batch_size, num_query_heads, seq_len, head_dim = q.shape

        #Step 1: Expand K and V heads for GQA
        k,v = self._expand_kv_heads(k,v,num_query_heads)

        # Step 2: Compute scaled attention scores
        # Use FP32 for score computation and Softmax stability.
        q_fp32 = q.float()
        k_fp32 = k.float()

        scores = torch.matmul(q_fp32, k_fp32.transpose(-2, -1))
        scores = scores /math.sqrt(head_dim)

        # Step 3: Apply causal mask
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(
                    seq_len, seq_len, 
                    device = q.device, 
                    dtype=torch.bool
                ), diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        # Step 4: Numerically stable Softmax
        attn_weights = torch.softmax(scores, dim=-1)

        # Step 5: Apply attention dropout
        attn_weights = self.dropout(attn_weights)

        # Step 6: Weighted sum of Values
        # Match V dtype for the final matrix multiplication.
        attn_weights = attn_weights.to(dtype = v.dtype)

        output = torch.matmul(attn_weights, v)

        logger.debug(
            "Attention completed: Q={}, K={}, V={}, output={}",
            tuple(q.shape),
            tuple(k.shape),
            tuple(v.shape),
            tuple(output.shape),
        )

        return output


        

        
    



        

        


