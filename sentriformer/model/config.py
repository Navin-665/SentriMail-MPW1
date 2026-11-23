"""
SentriFormer Configuration Module
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
import json


@dataclass
class SentriFormerConfig:
    """
    Configuration for SentriFormer Encoder-Decoder Architecture.
    Target: ~11M-12M core parameters + task heads (~12M-15M total).
    """
    # Vocabulary & Sequence Dimensions
    vocab_size: int = 16000
    d_model: int = 256
    encoder_layers: int = 4
    decoder_layers: int = 4
    attention_heads: int = 8
    head_dim: int = 32               # d_model = attention_heads * head_dim (8 * 32 = 256)
    d_ff: int = 1024                 # Feed-Forward hidden dimension (4 * d_model)
    max_seq_length: int = 512
    
    # Regularization & Normalization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    
    # Special Token IDs
    pad_token_id: int = 0
    unk_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 3
    sep_token_id: int = 5
    
    # Architectural Toggles
    tie_word_embeddings: bool = True  # Ties decoder lm_head with input embedding matrix
    use_bias: bool = False            # Modern standard: bias-free Linear layers for stability
    activation_function: str = "gelu"
    
    # Multi-Task Head Configurations
    num_categories: int = 21          # CFPB / Customer support issue categories
    num_sentiment_classes: int = 3    # Negative (Frustrated), Neutral, Positive
    num_csat_classes: int = 5         # CSAT score (1 to 5 ordinal classification)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
        
    def save_pretrained(self, save_path: str):
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
            
    @classmethod
    def from_pretrained(cls, json_path: str) -> "SentriFormerConfig":
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
