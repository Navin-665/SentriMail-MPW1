"""
SentriModel: wraps a pretrained T5 backbone with three classification
heads (sentiment, emotion, category) reading from the encoder's pooled
output, plus the backbone's own decoder for response generation.

T5 has no built-in [CLS]-style summary token the way BERT does, so the
three heads can't just read "the first token" - mean_pool_encoder_output
below does the actual pooling, respecting the attention mask so padding
tokens don't get averaged into the summary vector.
"""

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration

import config


def mean_pool_encoder_output(last_hidden_state, attention_mask):
    """Averages each example's real (non-padding) token vectors into one
    fixed-size summary vector per example."""
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed_hidden_states = (last_hidden_state * mask_expanded).sum(dim=1)
    real_token_counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return summed_hidden_states / real_token_counts


class SentriModel(nn.Module):
    def __init__(self, backbone_model_name=config.BACKBONE_MODEL_NAME):
        super().__init__()
        self.backbone = T5ForConditionalGeneration.from_pretrained(backbone_model_name)
        hidden_size = self.backbone.config.d_model

        self.sentiment_head = nn.Linear(hidden_size, len(config.SENTIMENT_LABELS))
        self.emotion_head = nn.Linear(hidden_size, len(config.EMOTION_LABELS))
        self.category_head = nn.Linear(hidden_size, len(config.CATEGORY_LABELS))

    def forward(self, input_ids, attention_mask, response_labels=None):
        # run the encoder once - both the classification heads and the
        # decoder (for generation) reuse this same pass, rather than
        # encoding the input twice
        encoder_outputs = self.backbone.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        pooled_representation = mean_pool_encoder_output(
            encoder_outputs.last_hidden_state, attention_mask
        )

        sentiment_logits = self.sentiment_head(pooled_representation)
        emotion_logits = self.emotion_head(pooled_representation)
        category_logits = self.category_head(pooled_representation)

        generation_loss = None
        generation_logits = None
        if response_labels is not None:
            # passing the already-computed encoder_outputs skips re-running
            # the encoder inside the backbone's own forward - T5 handles the
            # standard shifted-right decoder input and the -100 ignore_index
            # in response_labels automatically when labels is given
            generation_outputs = self.backbone(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                labels=response_labels,
            )
            generation_loss = generation_outputs.loss
            generation_logits = generation_outputs.logits

        return {
            "sentiment_logits": sentiment_logits,
            "emotion_logits": emotion_logits,
            "category_logits": category_logits,
            "generation_loss": generation_loss,
            "generation_logits": generation_logits,
        }

    @torch.no_grad()
    def generate_response(self, input_ids, attention_mask, max_new_tokens=None):
        """Actual autoregressive decoding for inference - separate from
        forward() above, which only does one teacher-forced training step
        and never produces real generated text."""
        encoder_outputs = self.backbone.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        generated_token_ids = self.backbone.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens or config.MAX_RESPONSE_TOKENS,
        )
        return generated_token_ids