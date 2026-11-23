"""
SentriFormer Multi-Task Classification, Sentiment, and CSAT Heads
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from .config import SentriFormerConfig
from .sentriformer import SentriFormerModel


def pool_encoder_states(encoder_hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Mean-pools encoder hidden states across non-padded tokens.
    Args:
        encoder_hidden_states: [batch_size, seq_len, d_model]
        attention_mask: [batch_size, seq_len] (1 for valid, 0 for pad)
    Returns:
        pooled: [batch_size, d_model]
    """
    if attention_mask is None:
        return encoder_hidden_states.mean(dim=1)
    
    # Expand mask: [batch_size, seq_len, 1]
    mask_expanded = attention_mask.unsqueeze(-1).float()
    sum_embeddings = torch.sum(encoder_hidden_states * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


class CategoryClassificationHead(nn.Module):
    """Hierarchical / Multi-class Issue Category Classification Head."""
    def __init__(self, config: SentriFormerConfig):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
        self.out_proj = nn.Linear(config.d_model, config.num_categories)

    def forward(self, pooled_features: torch.Tensor) -> torch.Tensor:
        x = self.dropout(pooled_features)
        x = self.act_fn(self.dense(x))
        x = self.dropout(x)
        logits = self.out_proj(x)
        return logits


class SentimentUrgencyHead(nn.Module):
    """Joint Sentiment Class (3-way) & Continuous Urgency Score (0.0 to 1.0)."""
    def __init__(self, config: SentriFormerConfig):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
        
        self.sentiment_proj = nn.Linear(config.d_model, config.num_sentiment_classes)
        self.urgency_proj = nn.Linear(config.d_model, 1)

    def forward(self, pooled_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.dropout(pooled_features)
        x = self.act_fn(self.dense(x))
        x = self.dropout(x)
        
        sentiment_logits = self.sentiment_proj(x)
        urgency_score = torch.sigmoid(self.urgency_proj(x))
        return {
            "sentiment_logits": sentiment_logits,
            "urgency_score": urgency_score.squeeze(-1)
        }


class CSATPredictionHead(nn.Module):
    """Customer Satisfaction (CSAT 1 to 5) Prediction Head."""
    def __init__(self, config: SentriFormerConfig):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
        self.out_proj = nn.Linear(config.d_model, config.num_csat_classes)

    def forward(self, pooled_features: torch.Tensor) -> torch.Tensor:
        x = self.dropout(pooled_features)
        x = self.act_fn(self.dense(x))
        x = self.dropout(x)
        logits = self.out_proj(x)
        return logits


class SentriFormerForMultiTask(nn.Module):
    """
    Unified SentriFormer Architecture with:
    - Backbone Encoder-Decoder
    - Tied LM Head (Response Generation / Denoising)
    - Category Head (Issue Routing)
    - Sentiment & Urgency Head (Prioritization)
    - CSAT Prediction Head (Customer Satisfaction Forecasting)
    """
    def __init__(self, config: SentriFormerConfig):
        super().__init__()
        self.config = config
        self.sentriformer = SentriFormerModel(config)
        
        # 1. Tied LM Generation Head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.sentriformer.shared_embeddings.weight
            
        # 2. Task-specific Supervised Heads
        self.category_head = CategoryClassificationHead(config)
        self.sentiment_head = SentimentUrgencyHead(config)
        self.csat_head = CSATPredictionHead(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        category_labels: Optional[torch.Tensor] = None,
        sentiment_labels: Optional[torch.Tensor] = None,
        csat_labels: Optional[torch.Tensor] = None,
        generation_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        
        # Encode source text
        encoder_outputs = self.sentriformer.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = pool_encoder_states(encoder_outputs, attention_mask=attention_mask)
        
        # Classification & Sentiment & CSAT Predictions
        category_logits = self.category_head(pooled)
        sentiment_out = self.sentiment_head(pooled)
        csat_logits = self.csat_head(pooled)
        
        # Generation (if decoder inputs or generation labels are provided)
        generation_logits = None
        gen_loss = None
        if decoder_input_ids is not None or generation_labels is not None:
            if decoder_input_ids is None and generation_labels is not None:
                # Shift right
                decoder_input_ids = generation_labels.new_zeros(generation_labels.shape)
                decoder_input_ids[:, 1:] = generation_labels[:, :-1].clone()
                decoder_input_ids[:, 0] = self.config.bos_token_id
                decoder_input_ids.masked_fill_(decoder_input_ids == -100, self.config.pad_token_id)
                if decoder_attention_mask is None:
                    decoder_attention_mask = (decoder_input_ids != self.config.pad_token_id).long()
                
            dec_out, _ = self.sentriformer.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs,
                attention_mask=decoder_attention_mask,
                encoder_attention_mask=attention_mask
            )
            generation_logits = self.lm_head(dec_out)
            
            if generation_labels is not None:
                target_labels = generation_labels.clone()
                target_labels[target_labels == self.config.pad_token_id] = -100
                gen_loss = F.cross_entropy(
                    generation_logits.view(-1, self.config.vocab_size),
                    target_labels.view(-1),
                    ignore_index=-100
                )

        # Losses
        cat_loss = None
        if category_labels is not None:
            cat_loss = F.cross_entropy(category_logits, category_labels)
            
        sent_loss = None
        if sentiment_labels is not None:
            sent_loss = F.cross_entropy(sentiment_out["sentiment_logits"], sentiment_labels)
            
        csat_loss = None
        if csat_labels is not None:
            csat_loss = F.cross_entropy(csat_logits, csat_labels)

        return {
            "category_logits": category_logits,
            "sentiment_logits": sentiment_out["sentiment_logits"],
            "urgency_score": sentiment_out["urgency_score"],
            "csat_logits": csat_logits,
            "generation_logits": generation_logits,
            "category_loss": cat_loss,
            "sentiment_loss": sent_loss,
            "csat_loss": csat_loss,
            "generation_loss": gen_loss,
            "encoder_last_hidden_state": encoder_outputs,
            "pooled_features": pooled,
        }
