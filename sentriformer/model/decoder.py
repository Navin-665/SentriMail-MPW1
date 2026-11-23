"""
SentriFormer Transformer Decoder Module (Pre-LayerNorm with Causal Self-Attention & Cross-Attention)
"""
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from .config import SentriFormerConfig
from .attention import MultiHeadAttention
from .feed_forward import FeedForward
from .encoder import SinusoidalPositionalEmbedding


class SentriDecoderBlock(nn.Module):
    """
    Pre-LN Transformer Decoder Layer:
    1. LayerNorm -> Causal Self-Attention -> Dropout + Residual
    2. LayerNorm -> Cross-Attention (to Encoder outputs) -> Dropout + Residual
    3. LayerNorm -> FeedForward -> Dropout + Residual
    """
    def __init__(self, config: SentriFormerConfig):
        super().__init__()
        # 1. Self-Attention (Causal)
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.self_attn = MultiHeadAttention(config, is_cross_attention=False)
        self.dropout1 = nn.Dropout(config.dropout)
        
        # 2. Cross-Attention
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.cross_attn = MultiHeadAttention(config, is_cross_attention=True)
        self.dropout2 = nn.Dropout(config.dropout)
        
        # 3. Feed-Forward
        self.norm3 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ffn = FeedForward(config)
        self.dropout3 = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]]:
        
        self_past = past_key_value[0] if past_key_value is not None else None
        cross_past = past_key_value[1] if past_key_value is not None else None
        
        # 1. Causal Self-Attention (causal lower-triangular mask already encoded in attention_mask)
        normed = self.norm1(hidden_states)
        self_attn_out, present_self_kv = self.self_attn(
            hidden_states=normed,
            attention_mask=attention_mask,
            is_causal=False,
            past_key_value=self_past,
            use_cache=use_cache
        )
        hidden_states = hidden_states + self.dropout1(self_attn_out)
        
        # 2. Cross-Attention (only if encoder hidden states are present)
        present_cross_kv = None
        if encoder_hidden_states is not None:
            normed = self.norm2(hidden_states)
            cross_attn_out, present_cross_kv = self.cross_attn(
                hidden_states=normed,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                is_causal=False,
                past_key_value=cross_past,
                use_cache=use_cache
            )
            hidden_states = hidden_states + self.dropout2(cross_attn_out)
            
        # 3. Feed-Forward Network
        normed = self.norm3(hidden_states)
        ffn_out = self.ffn(normed)
        hidden_states = hidden_states + self.dropout3(ffn_out)
        
        present_kv = (present_self_kv, present_cross_kv) if use_cache else None
        return hidden_states, present_kv


class SentriFormerDecoder(nn.Module):
    """
    Complete SentriFormer Decoder stack (4 Pre-LN layers).
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
        
        # Decoder layer stack (4 layers)
        self.layers = nn.ModuleList([
            SentriDecoderBlock(config) for _ in range(config.decoder_layers)
        ])
        
        # Final LayerNorm
        self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List]]:
        """
        Args:
            input_ids: [batch_size, seq_len]
            encoder_hidden_states: [batch_size, enc_seq_len, d_model]
            attention_mask: [batch_size, seq_len]
            encoder_attention_mask: [batch_size, enc_seq_len]
        """
        batch_size, seq_len = input_ids.shape
        
        # Offset position embeddings if using past cache (during generation)
        past_length = 0
        if past_key_values is not None and past_key_values[0][0] is not None:
            past_length = past_key_values[0][0][0].shape[2]
            
        token_embs = self.embed_tokens(input_ids) * math.sqrt(self.d_model)
        pos_embs = self.pos_emb(past_length + seq_len, input_ids.device)[:, past_length:]
        hidden_states = self.dropout(token_embs + pos_embs)
        
        # Decoder self-attention mask (strictly causal)
        # If using past cache during autoregressive generation (seq_len == 1 and past_length > 0):
        # Query at current step can attend to all past tokens in the cache + itself.
        # Since past cache only contains past tokens, no future tokens exist and no causal mask is needed.
        if past_length > 0 and seq_len == 1:
            extended_self_mask = None
        else:
            # During training, validation, or prefill without past cache (seq_len >= 1):
            # Enforce causal lower-triangular mask: position i can only attend to positions <= i.
            causal_mask = torch.tril(
                torch.ones((seq_len, seq_len), dtype=torch.bool, device=input_ids.device)
            ).view(1, 1, seq_len, seq_len)
            
            if attention_mask is not None:
                pad_mask = (attention_mask != 0).view(batch_size, 1, 1, seq_len)
                extended_self_mask = causal_mask & pad_mask
            else:
                extended_self_mask = causal_mask
            
        extended_enc_mask = None
        if encoder_attention_mask is not None:
            extended_enc_mask = (encoder_attention_mask != 0).view(batch_size, 1, 1, encoder_attention_mask.shape[-1])
            
        next_cache = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            hidden_states, present_kv = layer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=extended_self_mask,
                encoder_attention_mask=extended_enc_mask,
                past_key_value=layer_past,
                use_cache=use_cache
            )
            if use_cache:
                next_cache.append(present_kv)
                
        last_hidden_state = self.final_norm(hidden_states)
        return last_hidden_state, next_cache
