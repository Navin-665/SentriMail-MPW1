"""
T5-Style Span Corruption Algorithm for SentriFormer Pretraining
Implements 15% noise density with mean span length of 3 tokens (Raffel et al., 2020).
"""
import random
import numpy as np
from typing import List, Tuple, Dict, Any


class SpanCorruptor:
    """
    Applies T5-style span corruption to a token sequence:
    - 15% of tokens are selected for masking (noise_density = 0.15)
    - Mean span length of 3.0 tokens
    - Replaces each span in the input with a unique sentinel token (<extra_id_0>, <extra_id_1>, ...)
    - Targets contain <extra_id_0> + masked_span_0 + <extra_id_1> + masked_span_1 + ... + </s>
    """
    def __init__(
        self,
        noise_density: float = 0.15,
        mean_noise_span_length: float = 3.0,
        extra_ids_start: int = 6,      # Token ID for <extra_id_0> in SentriTokenizer v1
        num_extra_ids: int = 100,      # 100 sentinel tokens (<extra_id_0> .. <extra_id_99>)
        eos_token_id: int = 3,         # </s>
        pad_token_id: int = 0,         # <pad>
    ):
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.extra_ids_start = extra_ids_start
        self.num_extra_ids = num_extra_ids
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def corrupt(self, input_ids: List[int]) -> Tuple[List[int], List[int]]:
        """
        Args:
            input_ids: List of integer token IDs for a single sequence.
        Returns:
            corrupted_input_ids: Encoder inputs with sentinel tokens.
            target_ids: Decoder target sequence.
        """
        seq_len = len(input_ids)
        if seq_len < 4:
            # Sequence too short for span corruption, return identity target
            return input_ids, [self.extra_ids_start] + input_ids + [self.eos_token_id]

        # Calculate number of noise tokens & spans
        num_noise_tokens = max(1, int(round(seq_len * self.noise_density)))
        # Number of spans
        num_spans = max(1, int(round(num_noise_tokens / self.mean_noise_span_length)))
        num_spans = min(num_spans, self.num_extra_ids - 1)

        # Sample span lengths using geometric distribution (mean = 3.0, p = 1/3)
        p = 1.0 / self.mean_noise_span_length
        span_lengths = np.random.geometric(p, size=num_spans)
        
        # Scale lengths so total masked tokens roughly equals num_noise_tokens
        total_len = sum(span_lengths)
        if total_len != num_noise_tokens and total_len > 0:
            scale = num_noise_tokens / total_len
            span_lengths = np.maximum(1, np.round(span_lengths * scale).astype(int))

        # Sample starting positions without overlapping
        # Available budget for non-noise tokens
        total_noise = min(seq_len - num_spans, sum(span_lengths))
        non_noise_total = seq_len - total_noise
        
        # Generate random gap lengths between spans
        gap_cuts = sorted(random.sample(range(1, non_noise_total + num_spans), num_spans))
        gaps = [gap_cuts[0]] + [gap_cuts[i] - gap_cuts[i - 1] for i in range(1, num_spans)]
        
        # Build mask array: True = noise, False = clean
        mask = [False] * seq_len
        curr_idx = 0
        
        for i in range(num_spans):
            curr_idx += (gaps[i] - 1)
            span_len = min(span_lengths[i], seq_len - curr_idx)
            for j in range(curr_idx, min(seq_len, curr_idx + span_len)):
                mask[j] = True
            curr_idx += span_len
            if curr_idx >= seq_len:
                break

        # Assemble corrupted inputs and target labels
        corrupted_input_ids = []
        target_ids = []
        sentinel_idx = 0
        in_noise = False

        for i, token_id in enumerate(input_ids):
            if mask[i]:
                if not in_noise:
                    # Starting a new noise span
                    sentinel_token_id = self.extra_ids_start + sentinel_idx
                    corrupted_input_ids.append(sentinel_token_id)
                    target_ids.append(sentinel_token_id)
                    sentinel_idx = (sentinel_idx + 1) % self.num_extra_ids
                    in_noise = True
                target_ids.append(token_id)
            else:
                in_noise = False
                corrupted_input_ids.append(token_id)

        # End target sequence with EOS token
        target_ids.append(self.eos_token_id)
        
        return corrupted_input_ids, target_ids


def corrupt_tokens_t5(
    input_ids: List[int],
    noise_density: float = 0.15,
    mean_noise_span_length: float = 3.0,
    extra_ids_start: int = 6,
    num_extra_ids: int = 100,
    eos_token_id: int = 3
) -> Tuple[List[int], List[int]]:
    """Functional interface for T5 span corruption."""
    corruptor = SpanCorruptor(
        noise_density=noise_density,
        mean_noise_span_length=mean_noise_span_length,
        extra_ids_start=extra_ids_start,
        num_extra_ids=num_extra_ids,
        eos_token_id=eos_token_id
    )
    return corruptor.corrupt(input_ids)
