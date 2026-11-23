"""
TWCS Multi-Turn Conversation Extractor — High Performance
"""
import os
import hashlib
from typing import Iterator, Set, Dict, Any, List
import pandas as pd
from .schema import SentriRecord, CanonicalSource, ExampleType
from .cleaner import clean_text, normalize_for_dedup, is_valid_text, is_deflection
from .splitter import assign_split


def extract_twcs(
    csv_path: str,
    seen_hashes: Set[str],
    max_records: int = None
) -> Iterator[SentriRecord]:
    """
    Parses TWCS tweets, reconstructs multi-turn conversation threads,
    preserves parent/child turns and emits conversation turn records and paired responses.
    Optimized for sub-second dictionary construction on 2.8M rows.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"TWCS dataset not found at: {csv_path}")

    # Load TWCS dataframe
    df = pd.read_csv(
        csv_path,
        usecols=["tweet_id", "author_id", "inbound", "created_at", "text", "in_response_to_tweet_id", "response_tweet_id"],
        dtype={"tweet_id": str, "author_id": str, "in_response_to_tweet_id": str, "response_tweet_id": str}
    )
    
    # Fast vectorized conversion to python lists
    tweet_ids = df["tweet_id"].tolist()
    author_ids = df["author_id"].tolist()
    inbounds = df["inbound"].tolist()
    created_ats = df["created_at"].tolist()
    texts = df["text"].tolist()
    in_resp_list = df["in_response_to_tweet_id"].tolist()
    resp_tw_list = df["response_tweet_id"].tolist()
    
    # Sub-second dict mapping
    tweet_map = {}
    initial_tweet_ids = []
    
    for i in range(len(tweet_ids)):
        t_id = tweet_ids[i]
        inbound = inbounds[i]
        in_resp = in_resp_list[i] if pd.notna(in_resp_list[i]) else None
        resp_id_str = resp_tw_list[i] if pd.notna(resp_tw_list[i]) else None
        
        tweet_map[t_id] = {
            "tweet_id": t_id,
            "author_id": author_ids[i],
            "inbound": inbound,
            "created_at": created_ats[i],
            "text": texts[i],
            "in_response_to": in_resp,
            "response_tweet_id": resp_id_str,
        }
        
        if inbound and in_resp is None and resp_id_str is not None:
            initial_tweet_ids.append(t_id)

    count = 0
    for init_id in initial_tweet_ids:
        init_tweet = tweet_map[init_id]
        conv_id = f"twcs_conv_{init_id}"
        split = assign_split(conv_id)  # All turns in this conversation share the exact same split!
        
        # Follow thread
        current_tweet = init_tweet
        turn_idx = 1
        
        # Linear chain extraction
        while current_tweet and current_tweet.get("response_tweet_id"):
            resp_ids = [r.strip() for r in current_tweet["response_tweet_id"].split(",") if r.strip() in tweet_map]
            if not resp_ids:
                break
                
            resp_tweet = tweet_map[resp_ids[0]]
            
            cust_text = clean_text(current_tweet["text"])
            resp_text = clean_text(resp_tweet["text"])
            
            if is_valid_text(cust_text, min_chars=4, min_words=1) and is_valid_text(resp_text, min_chars=4, min_words=1):
                # Check deflection
                has_deflect = is_deflection(resp_text)
                
                # Check deduplication
                pair_norm = normalize_for_dedup(cust_text + " || " + resp_text)
                norm_hash = hashlib.sha256(pair_norm.encode('utf-8')).hexdigest()[:16]
                
                if norm_hash not in seen_hashes:
                    seen_hashes.add(norm_hash)
                    
                    metadata = {
                        "brand": resp_tweet["author_id"],
                        "customer_author_id": current_tweet["author_id"],
                        "turn_index": turn_idx,
                        "has_deflection": has_deflect,
                        "created_at": current_tweet["created_at"],
                    }
                    
                    record = SentriRecord(
                        id=f"twcs_{current_tweet['tweet_id']}_turn_{turn_idx}",
                        source=CanonicalSource.TWCS.value,
                        example_type=ExampleType.CONVERSATION.value,
                        group_id=conv_id,
                        parent_id=f"twcs_{current_tweet.get('in_response_to')}" if current_tweet.get("in_response_to") else None,
                        text=cust_text,
                        response=resp_text if not has_deflect else None,
                        split=split,
                        metadata=metadata
                    )
                    
                    yield record
                    count += 1
                    if max_records and count >= max_records:
                        return
                        
            # Advance to next turn if customer replied back to brand
            # Look up any tweet responding to resp_tweet['tweet_id']
            if resp_tweet.get("response_tweet_id"):
                next_ids = [r.strip() for r in resp_tweet["response_tweet_id"].split(",") if r.strip() in tweet_map]
                if next_ids and tweet_map[next_ids[0]]["inbound"]:
                    current_tweet = tweet_map[next_ids[0]]
                    turn_idx += 1
                    continue
            break
