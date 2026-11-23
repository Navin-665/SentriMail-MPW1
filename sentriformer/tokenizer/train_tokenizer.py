"""
SentriTokenizer Trainer — Trains Custom Byte-Pair Encoding (BPE) Tokenizers (16k, 24k, 32k)
Uses pure streaming from train.jsonl (both 'text' and 'response' fields).
Registers standard special tokens + 100 T5 sentinel tokens (<extra_id_0> .. <extra_id_99>).
"""
import os
import sys
import json
import time
from typing import Iterator
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers, processors
from transformers import PreTrainedTokenizerFast

# Base paths
ROOT_DIR = r"c:\Users\nan33\OneDrive\Desktop\6th sem projects\SENTRIMAIL-MPW1"
DATA_DIR = os.path.join(ROOT_DIR, "sentriformer", "data")
TRAIN_JSONL = os.path.join(DATA_DIR, "train.jsonl")
TOKENIZER_DIR = os.path.join(ROOT_DIR, "sentriformer", "tokenizer")

# Special Tokens Specification
SPECIAL_TOKENS = [
    "<pad>",    # ID: 0 (Padding)
    "<unk>",    # ID: 1 (Unknown token fallback)
    "<s>",      # ID: 2 (Beginning of sequence / BOS)
    "</s>",     # ID: 3 (End of sequence / EOS)
    "<mask>",   # ID: 4 (Masked LM token)
    "<sep>",    # ID: 5 (Separator token)
]

# 100 T5-style Sentinel tokens for span corruption pretraining
SENTINEL_TOKENS = [f"<extra_id_{i}>" for i in range(100)]
ALL_SPECIAL_TOKENS = SPECIAL_TOKENS + SENTINEL_TOKENS


def text_iterator(jsonl_path: str, max_samples: int = None) -> Iterator[str]:
    """
    Streams texts from the frozen train.jsonl file.
    Yields both 'text' (customer complaints/dialogue) AND 'response' (brand/agent replies)
    so the tokenizer learns both complaint and response domain vocabularies.
    """
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Train dataset not found at: {jsonl_path}")

    count = 0
    with open(jsonl_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 1. Yield primary customer complaint / conversation context
            text = rec.get("text", "")
            if text and len(text.strip()) > 0:
                yield text.strip()
                count += 1
                if max_samples and count >= max_samples:
                    return

            # 2. Yield agent response text (for dialogue and response generation)
            resp = rec.get("response", "")
            if resp and len(resp.strip()) > 0:
                yield resp.strip()
                count += 1
                if max_samples and count >= max_samples:
                    return


def build_base_tokenizer() -> Tokenizer:
    """
    Constructs a Byte-Pair Encoding (BPE) tokenizer pipeline:
    - Normalizer: NFKC unicode normalization
    - Pre-tokenizer: ByteLevel (preserves byte information, clean spaces, handles emojis)
    - Decoder: ByteLevel
    """
    # 1. Initialize BPE model with <unk>
    bpe_model = models.BPE(unk_token="<unk>")
    tokenizer = Tokenizer(bpe_model)

    # 2. Normalization: NFKC decomposes composite unicode cleanly
    tokenizer.normalizer = normalizers.NFKC()

    # 3. Pre-tokenizer: ByteLevel splits on bytes, handles emojis, punctuation, spaces without information loss
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 4. Decoder: ByteLevel restores exact original strings
    tokenizer.decoder = decoders.ByteLevel()

    return tokenizer


def train_bpe_tokenizer(vocab_size: int, output_dir: str, train_jsonl: str = TRAIN_JSONL, max_samples: int = None):
    """
    Trains a BPE tokenizer for a specified target vocabulary size.
    Saves both native Tokenizer JSON and wrapped HuggingFace PreTrainedTokenizerFast.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n=======================================================")
    print(f"  Training SentriTokenizer — Vocab Size: {vocab_size:,}")
    print(f"=======================================================")
    start_time = time.time()

    tokenizer = build_base_tokenizer()

    # Trainer configuration
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=ALL_SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # Train from streaming iterator
    print(f"  Streaming training corpus from: {os.path.basename(train_jsonl)}...")
    tokenizer.train_from_iterator(text_iterator(train_jsonl, max_samples=max_samples), trainer=trainer)

    # Post-processor for BOS / EOS sequence wrapping
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # Save native tokenizers JSON
    native_save_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(native_save_path)
    print(f"  Saved native tokenizer to: {native_save_path}")

    # Wrap into Hugging Face PreTrainedTokenizerFast for full transformers compatibility
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=native_save_path,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        sep_token="<sep>",
        mask_token="<mask>",
        additional_special_tokens=SENTINEL_TOKENS,
    )
    fast_tokenizer.save_pretrained(output_dir)
    print(f"  Saved HuggingFace fast tokenizer to: {output_dir}")

    elapsed = time.time() - start_time
    print(f"  Completed vocab={vocab_size:,} in {elapsed:.1f} seconds.")
    return fast_tokenizer


def main():
    print("=" * 70)
    print("          SENTRITOKENIZER VOCABULARY TRAINING SUITE")
    print("=" * 70)

    vocab_candidates = [16000, 24000, 32000]
    
    for vocab in vocab_candidates:
        save_path = os.path.join(TOKENIZER_DIR, f"vocab_{vocab//1000}k")
        train_bpe_tokenizer(vocab_size=vocab, output_dir=save_path)

    print("\n" + "=" * 70)
    print("ALL 3 TOKENIZER VOCABULARIES (16k, 24k, 32k) SUCCESSFULLY TRAINED")
    print("=" * 70)


if __name__ == "__main__":
    main()
