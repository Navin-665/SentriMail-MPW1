"""
Streaming CFPB Complaints Extractor
"""
import os
import hashlib
from typing import Iterator, Set, Dict, Any
import pandas as pd
from .schema import SentriRecord, CanonicalSource, ExampleType
from .cleaner import clean_text, normalize_for_dedup, is_valid_text
from .splitter import assign_split


def extract_cfpb(
    csv_path: str,
    seen_hashes: Set[str],
    chunk_size: int = 50000,
    max_records: int = None
) -> Iterator[SentriRecord]:
    """
    Streams the 9.18GB CFPB CSV, extracts non-empty narratives, cleans them,
    deduplicates globally via normalized hash, and yields Canonical SentriRecords.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CFPB dataset not found at: {csv_path}")

    columns = [
        "Complaint ID",
        "Product",
        "Sub-product",
        "Issue",
        "Sub-issue",
        "Consumer complaint narrative",
        "Company",
        "State",
        "ZIP code",
        "Date received",
        "Company response to consumer",
        "Timely response?"
    ]

    count = 0
    # Stream in chunks
    for chunk in pd.read_csv(csv_path, usecols=columns, chunksize=chunk_size, low_memory=False):
        # Filter for rows with narrative
        narrative_chunk = chunk.dropna(subset=["Consumer complaint narrative"])
        
        for _, row in narrative_chunk.iterrows():
            raw_narrative = str(row["Consumer complaint narrative"])
            cleaned = clean_text(raw_narrative)
            
            if not is_valid_text(cleaned, min_chars=15, min_words=3):
                continue
                
            norm_hash = hashlib.sha256(normalize_for_dedup(cleaned).encode('utf-8')).hexdigest()[:16]
            if norm_hash in seen_hashes:
                continue
            seen_hashes.add(norm_hash)
            
            complaint_id = str(row["Complaint ID"])
            # Group ID based on text cluster / normalized hash ensures near-duplicates stay in same split
            group_id = f"cfpb_grp_{norm_hash[:8]}"
            split = assign_split(group_id)
            
            metadata = {
                "product": str(row["Product"]) if pd.notna(row["Product"]) else None,
                "sub_product": str(row["Sub-product"]) if pd.notna(row["Sub-product"]) else None,
                "issue": str(row["Issue"]) if pd.notna(row["Issue"]) else None,
                "sub_issue": str(row["Sub-issue"]) if pd.notna(row["Sub-issue"]) else None,
                "company": str(row["Company"]) if pd.notna(row["Company"]) else None,
                "state": str(row["State"]) if pd.notna(row["State"]) else None,
                "date_received": str(row["Date received"]) if pd.notna(row["Date received"]) else None,
                "company_response": str(row["Company response to consumer"]) if pd.notna(row["Company response to consumer"]) else None,
                "timely": str(row["Timely response?"]) if pd.notna(row["Timely response?"]) else None,
            }
            
            record = SentriRecord(
                id=f"cfpb_{complaint_id}",
                source=CanonicalSource.CFPB.value,
                example_type=ExampleType.COMPLAINT.value,
                group_id=group_id,
                parent_id=None,
                text=cleaned,
                response=None,
                split=split,
                metadata=metadata
            )
            
            yield record
            count += 1
            if max_records and count >= max_records:
                return
