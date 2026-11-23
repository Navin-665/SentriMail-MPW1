"""
Customer Support (Shopzilla / E-commerce) Extractor
"""
import os
import hashlib
from typing import Iterator, Set, Dict, Any
import pandas as pd
from .schema import SentriRecord, CanonicalSource, ExampleType
from .cleaner import clean_text, normalize_for_dedup, is_valid_text
from .splitter import assign_split


def extract_customer_support(
    csv_path: str,
    seen_hashes: Set[str],
    max_records: int = None
) -> Iterator[SentriRecord]:
    """
    Extracts non-null customer remarks from the customer support dataset,
    keeps complaint-time metadata cleanly isolated from CSAT target,
    deduplicates globally, and yields canonical SentriRecords.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Customer Support dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    
    # Filter for rows with Customer Remarks
    remarks_df = df.dropna(subset=["Customer Remarks"])
    
    count = 0
    for _, row in remarks_df.iterrows():
        raw_remark = str(row["Customer Remarks"])
        cleaned = clean_text(raw_remark)
        
        # Keep short but valid complaints (e.g. "Fraud on card", "Card stolen")
        if not is_valid_text(cleaned, min_chars=4, min_words=1):
            continue
            
        norm_hash = hashlib.sha256(normalize_for_dedup(cleaned).encode('utf-8')).hexdigest()[:16]
        if norm_hash in seen_hashes:
            continue
        seen_hashes.add(norm_hash)
        
        # Group ID: Prefer Order_id if present, else normalized remark hash cluster
        order_id = str(row["Order_id"]) if pd.notna(row["Order_id"]) else None
        if order_id and order_id.strip() != "":
            group_id = f"custsupp_ord_{order_id.strip()}"
        else:
            group_id = f"custsupp_grp_{norm_hash[:8]}"
            
        split = assign_split(group_id)
        
        # Isolate complaint-time features from post-handling metrics
        metadata = {
            # Complaint-time input features:
            "category": str(row["category"]) if pd.notna(row["category"]) else None,
            "sub_category": str(row["Sub-category"]) if pd.notna(row["Sub-category"]) else None,
            "channel": str(row["channel_name"]) if pd.notna(row["channel_name"]) else None,
            "product_category": str(row["Product_category"]) if pd.notna(row["Product_category"]) else None,
            "item_price": float(row["Item_price"]) if pd.notna(row["Item_price"]) else None,
            "customer_city": str(row["Customer_City"]) if pd.notna(row["Customer_City"]) else None,
            # Supervised Target (never used as pretraining input feature):
            "csat_score": int(row["CSAT Score"]) if pd.notna(row["CSAT Score"]) else None,
            # Post-handling audit info:
            "tenure_bucket": str(row["Tenure Bucket"]) if pd.notna(row["Tenure Bucket"]) else None,
            "agent_shift": str(row["Agent Shift"]) if pd.notna(row["Agent Shift"]) else None,
        }
        
        record = SentriRecord(
            id=f"custsupp_{row['Unique id']}",
            source=CanonicalSource.CUSTOMER_SUPPORT.value,
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
