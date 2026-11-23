"""
Multi-Head Attention Module utilizing PyTorch Scaled Dot-Product Attention (SDPA)
Supports Bidirectional Self-Attention, Causal Self-Attention, Cross-Attention, and KV Caching.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .config import SentriFormerConfig


class MultiHeadAttention(nn.Module):
    """
    Optimized Multi-Head Attention leveraging PyTorch F.scaled_dot_product_attention (SDPA).
    Automatically leverages FlashAttention-2 and cuDNN kernels where available on GPU.
    """
    def __init__(self, config: SentriFormerConfig, is_cross_attention: bool = False):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.attention_heads
        self.head_dim = config.head_dim
        self.is_cross_attention = is_cross_attention
        self.dropout_p = config.attention_dropout if self.training else 0.0
        
        assert self.num_heads * self.head_dim == self.d_model, (
            f"d_model ({self.d_model}) must equal num_heads ({self.num_heads}) * head_dim ({self.head_dim})"
        )
        
        # Query projection
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=config.use_bias)
        # Key & Value projections (project from encoder hidden state if cross-attention)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=config.use_bias)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=config.use_bias)
        # Output projection
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=config.use_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            hidden_states: [batch_size, seq_len_q, d_model]
            key_value_states: [batch_size, seq_len_kv, d_model] (for cross-attention)
            attention_mask: [batch_size, 1, seq_len_q, seq_len_kv] or boolean mask
            is_causal: bool (enforces autoregressive triangle mask in decoder self-attention)
            past_key_value: Optional tuple of (cached_k, cached_v) for fast generation
            use_cache: whether to return present (k, v) cache
        """
        batch_size, seq_len_q, _ = hidden_states.shape
        
        # 1. Project Queries: [batch_size, num_heads, seq_len_q, head_dim]
        q = self.q_proj(hidden_states)
        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 2. Project Keys & Values
        kv_input = key_value_states if (self.is_cross_attention and key_value_states is not None) else hidden_states
        seq_len_kv = kv_input.shape[1]
        
        k = self.k_proj(kv_input).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_input).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 3. Handle KV Cache (for generation)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            if self.is_cross_attention:
                # For cross attention, encoder keys & values are static
                k, v = past_k, past_v
            else:
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            
        present_key_value = (k, v) if use_cache else None
        
        # 4. Scaled Dot-Product Attention via PyTorch Native SDPA
        # PyTorch SDPA strictly disallows combining attn_mask with is_causal=True.
        # If attention_mask is provided (or when using past_key_value cache), is_causal must be False.
        sdpa_is_causal = is_causal if (attention_mask is None and past_key_value is None) else False
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=sdpa_is_causal
        )
            
        # 5. Reshape and project output: [batch_size, seq_len_q, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        output = self.out_proj(attn_output)
        
        return output, present_key_value
