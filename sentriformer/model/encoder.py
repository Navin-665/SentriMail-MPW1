"""
SentriFormer Transformer Encoder Module (Pre-LayerNorm)
"""
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from .config import SentriFormerConfig
from .attention import MultiHeadAttention
from .feed_forward import FeedForward


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings for robust length generalization."""
    def __init__(self, d_model: int, max_seq_length: int = 512):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, max_len, d_model]

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return self.pe[:, :seq_len].to(device)


class SentriEncoderBlock(nn.Module):
    """
    Pre-LN Transformer Encoder Layer:
    x -> LayerNorm -> Self-Attention -> Dropout + Residual
    x -> LayerNorm -> FeedForward -> Dropout + Residual
    """
    def __init__(self, config: SentriFormerConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.self_attn = MultiHeadAttention(config, is_cross_attention=False)
        self.dropout1 = nn.Dropout(config.dropout)
        
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ffn = FeedForward(config)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-LN Self-Attention
        normed = self.norm1(hidden_states)
        attn_out, _ = self.self_attn(
            hidden_states=normed,
            attention_mask=attention_mask,
            is_causal=False
        )
        hidden_states = hidden_states + self.dropout1(attn_out)
        
        # Pre-LN Feed-Forward
        normed = self.norm2(hidden_states)
        ffn_out = self.ffn(normed)
        hidden_states = hidden_states + self.dropout2(ffn_out)
        
        return hidden_states


class SentriFormerEncoder(nn.Module):
    """
    Complete SentriFormer Encoder stack (4 Pre-LN layers).
    """
    def __init__(self, config: SentriFormerConfig, embedding_layer: Optional[nn.Embedding] = None):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        
        # Shared or internal token embeddings
        if embedding_layer is not None:
            self.embed_tokens = embedding_layer
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
            
        self.pos_emb = SinusoidalPositionalEmbedding(config.d_model, config.max_seq_length)
        self.dropout = nn.Dropout(config.dropout)
        
        # Encoder layer stack (4 layers)
        self.layers = nn.ModuleList([
            SentriEncoderBlock(config) for _ in range(config.encoder_layers)
        ])
        
        # Final LayerNorm
        self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] (1 for valid tokens, 0 for pad)
        Returns:
            last_hidden_state: [batch_size, seq_len, d_model]
        """
        seq_len = input_ids.shape[1]
        
        # Token Embeddings + Positional Embeddings
        token_embs = self.embed_tokens(input_ids) * math.sqrt(self.d_model)
        pos_embs = self.pos_emb(seq_len, input_ids.device)
        hidden_states = self.dropout(token_embs + pos_embs)
        
        # Convert 2D pad mask [batch_size, seq_len] to 4D attention mask [batch_size, 1, 1, seq_len] for SDPA
        extended_mask = None
        if attention_mask is not None:
            # Mask format: False where padded, True where real token
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2).bool()  # [batch, 1, 1, seq_len]
            
        # Pass through all 4 Encoder Blocks
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=extended_mask)
            
        last_hidden_state = self.final_norm(hidden_states)
        return last_hidden_state
