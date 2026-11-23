"""
Bitext Customer Support LLM Dataset Extractor
"""
import hashlib
from typing import Iterator, Set, Dict, Any
from datasets import load_dataset
from .schema import SentriRecord, CanonicalSource, ExampleType
from .cleaner import clean_text, normalize_for_dedup, is_valid_text
from .splitter import assign_split


def extract_bitext(
    seen_hashes: Set[str],
    max_records: int = None
) -> Iterator[SentriRecord]:
    """
    Extracts instruction-response pairs from Bitext dataset,
    normalizes and deduplicates, and assigns split based on instruction hash.
    """
    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
    
    count = 0
    for idx, row in enumerate(ds):
        raw_text = str(row.get("instruction", ""))
        raw_resp = str(row.get("response", ""))
        
        cleaned_text = clean_text(raw_text)
        cleaned_resp = clean_text(raw_resp)
        
        if not is_valid_text(cleaned_text, min_chars=4, min_words=1):
            continue
            
        norm_hash = hashlib.sha256(normalize_for_dedup(cleaned_text).encode('utf-8')).hexdigest()[:16]
        if norm_hash in seen_hashes:
            continue
        seen_hashes.add(norm_hash)
        
        # Group by instruction semantic hash so paraphrases stay in the same split
        group_id = f"bitext_grp_{norm_hash[:8]}"
        split = assign_split(group_id)
        
        metadata = {
            "category": row.get("category"),
            "intent": row.get("intent"),
            "flags": row.get("flags"),
        }
        
        record = SentriRecord(
            id=f"bitext_{idx}",
            source=CanonicalSource.BITEXT.value,
            example_type=ExampleType.SUPPORT_RESPONSE.value,
            group_id=group_id,
            parent_id=None,
            text=cleaned_text,
            response=cleaned_resp,
            split=split,
            metadata=metadata
        )
        
        yield record
        count += 1
        if max_records and count >= max_records:
            return
