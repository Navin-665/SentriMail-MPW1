"""
SentriFormer Core Encoder-Decoder Model & Conditional Generation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
from .config import SentriFormerConfig
from .encoder import SentriFormerEncoder
from .decoder import SentriFormerDecoder


class SentriFormerModel(nn.Module):
    """
    Base SentriFormer Encoder-Decoder model with shared token embeddings.
    """
    def __init__(self, config: SentriFormerConfig):
        super().__init__()
        self.config = config
        
        # Shared token embedding table
        self.shared_embeddings = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
        )
        
        self.encoder = SentriFormerEncoder(config, embedding_layer=self.shared_embeddings)
        self.decoder = SentriFormerDecoder(config, embedding_layer=self.shared_embeddings)
        
        self._init_weights()

    def _init_weights(self):
        """Initializes weights with Gaussian distribution (std=0.02)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List]]:
        
        # 1. Encode source text
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 2. Decode target sequence with cross-attention to encoder
        decoder_outputs, next_cache = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs,
            attention_mask=decoder_attention_mask,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache
        )
        
        return decoder_outputs, encoder_outputs, next_cache


class SentriFormerForConditionalGeneration(nn.Module):
    """
    SentriFormer with tied LM Head for T5-style Text-to-Text generation & pretraining.
    Ties lm_head.weight == shared_embeddings.weight to save ~4.1M parameters.
    """
    def __init__(self, config: SentriFormerConfig):
        super().__init__()
        self.config = config
        self.sentriformer = SentriFormerModel(config)
        
        # Language Modeling projection head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie LM head weight with shared embedding table
        if config.tie_word_embeddings:
            self.lm_head.weight = self.sentriformer.shared_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List] = None,
        use_cache: bool = False,
    ) -> Dict[str, Any]:
        """
        Standard Seq2Seq forward pass with cross-entropy loss computation.
        """
        # If training with labels and decoder_input_ids not provided, shift labels right
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)

        decoder_outputs, encoder_outputs, next_cache = self.sentriformer(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache
        )
        
        # Project hidden states to vocabulary logits: [batch_size, seq_len, vocab_size]
        logits = self.lm_head(decoder_outputs)
        
        loss = None
        if labels is not None:
            # Flatten for cross-entropy loss ignoring pad_token_id (-100 or 0)
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            # Mask pad token id
            target_labels = labels.clone()
            target_labels[target_labels == self.config.pad_token_id] = -100
            loss = loss_fn(logits.view(-1, self.config.vocab_size), target_labels.view(-1))
            
        return {
            "loss": loss,
            "logits": logits,
            "encoder_last_hidden_state": encoder_outputs,
            "past_key_values": next_cache
        }

    def _shift_right(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Shifts decoder inputs right and prefixes with BOS token."""
        shifted = input_ids.new_zeros(input_ids.shape)
        shifted[:, 1:] = input_ids[:, :-1].clone()
        shifted[:, 0] = self.config.bos_token_id
        # Replace ignored index (-100) with pad token
        shifted.masked_fill_(shifted == -100, self.config.pad_token_id)
        return shifted

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Fast autoregressive sequence generation with KV caching.
        """
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # 1. Encode once
        encoder_outputs = self.sentriformer.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # 2. Start token (BOS)
        curr_token = torch.full((batch_size, 1), self.config.bos_token_id, dtype=torch.long, device=device)
        generated = curr_token.clone()
        past_key_values = None
        
        for _ in range(max_new_tokens):
            decoder_out, past_key_values = self.sentriformer.decoder(
                input_ids=curr_token,
                encoder_hidden_states=encoder_outputs,
                encoder_attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            logits = self.lm_head(decoder_out[:, -1, :]) / max(temperature, 1e-5)
            
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)  # Greedy decoding
            
            generated = torch.cat([generated, next_token], dim=1)
            curr_token = next_token
            
            # Check if all batches reached EOS
            if (next_token == self.config.eos_token_id).all():
                break
                
        return generated
