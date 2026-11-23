"""
Master Pipeline Orchestrator for Building SentriCorpus
"""
import os
import sys
import json
import time
import hashlib
from typing import Set, Dict, Any, List
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sentriformer.corpus.schema import SentriRecord
from sentriformer.corpus.extract_cfpb import extract_cfpb
from sentriformer.corpus.extract_twcs import extract_twcs
from sentriformer.corpus.extract_custsupp import extract_customer_support
from sentriformer.corpus.extract_bitext import extract_bitext

# Paths
ROOT_DIR = r"c:\Users\nan33\OneDrive\Desktop\6th sem projects\SENTRIMAIL-MPW1"
RAWDATA_DIR = os.path.join(ROOT_DIR, "rawdata")
OUTPUT_DIR = os.path.join(ROOT_DIR, "sentriformer", "data")
SUPERVISED_DIR = os.path.join(OUTPUT_DIR, "supervised")

CFPB_PATH = os.path.join(RAWDATA_DIR, "complaints.csv")
TWCS_PATH = os.path.join(RAWDATA_DIR, "twcs.csv")
CUSTSUPP_PATH = os.path.join(RAWDATA_DIR, "archive (1)", "Customer_support_data.csv")


def init_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SUPERVISED_DIR, exist_ok=True)


class CorpusWriter:
    """Streams records into chunked JSONL and converts to Parquet with zero memory bloat."""
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.files = {
            "train": open(os.path.join(out_dir, "train.jsonl"), "w", encoding="utf-8"),
            "val": open(os.path.join(out_dir, "val.jsonl"), "w", encoding="utf-8"),
            "test": open(os.path.join(out_dir, "test.jsonl"), "w", encoding="utf-8"),
        }
        self.counts = {"train": 0, "val": 0, "test": 0}
        self.source_counts = {}
        self.group_sets = {"train": set(), "val": set(), "test": set()}

    def write(self, record: SentriRecord):
        rec_dict = record.to_dict()
        split = record.split
        line = json.dumps(rec_dict, ensure_ascii=False)
        self.files[split].write(line + "\n")
        self.counts[split] += 1
        
        # Track source
        src = record.source
        self.source_counts[src] = self.source_counts.get(src, 0) + 1
        
        # Track group
        self.group_sets[split].add(record.group_id)

    def close(self):
        for f in self.files.values():
            f.close()


def convert_jsonl_to_parquet(jsonl_path: str, parquet_path: str, chunk_size: int = 100000):
    """Converts large JSONL file to Parquet efficiently in streaming chunks."""
    if not os.path.exists(jsonl_path):
        return 0, 0
    print(f"  Converting {os.path.basename(jsonl_path)} -> {os.path.basename(parquet_path)}...")
    
    records = []
    writer = None
    good = 0
    skipped = 0
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
                good += 1
            except json.JSONDecodeError:
                skipped += 1
                continue
            if len(records) >= chunk_size:
                df = pd.DataFrame(records)
                df["metadata"] = df["metadata"].apply(lambda m: json.dumps(m) if isinstance(m, dict) else str(m))
                table = pa.Table.from_pandas(df)
                if writer is None:
                    writer = pq.ParquetWriter(parquet_path, table.schema, compression="snappy")
                writer.write_table(table)
                records = []
                
        if records:
            df = pd.DataFrame(records)
            df["metadata"] = df["metadata"].apply(lambda m: json.dumps(m) if isinstance(m, dict) else str(m))
            table = pa.Table.from_pandas(df)
            if writer is None:
                writer = pq.ParquetWriter(parquet_path, table.schema, compression="snappy")
            writer.write_table(table)
            
    if writer:
        writer.close()
    if skipped:
        print(f"    Skipped {skipped} malformed lines (out of {good + skipped:,})")
    return good, skipped


def export_supervised_subsets(train_jsonl: str, val_jsonl: str, test_jsonl: str):
    """Exports specialized datasets for downstream classification, dialogue, and CSAT tasks."""
    print("  Exporting supervised task-specific subsets...")
    
    for split_name, jpath in [("train", train_jsonl), ("val", val_jsonl), ("test", test_jsonl)]:
        cls_records = []
        dlg_records = []
        csat_records = []
        
        with open(jpath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                meta = rec.get("metadata", {})
                
                # 1. Classification (CFPB products or Customer Support categories)
                if rec["source"] == "cfpb" and meta.get("product"):
                    cls_records.append({
                        "id": rec["id"],
                        "text": rec["text"],
                        "product": meta.get("product"),
                        "issue": meta.get("issue"),
                        "split": rec["split"]
                    })
                elif rec["source"] == "customer_support" and meta.get("category"):
                    cls_records.append({
                        "id": rec["id"],
                        "text": rec["text"],
                        "product": meta.get("category"),
                        "issue": meta.get("sub_category"),
                        "split": rec["split"]
                    })
                    
                # 2. Dialogue Generation (TWCS clean pairs & Bitext instructions)
                if rec.get("response") and len(rec["response"].strip()) > 0:
                    dlg_records.append({
                        "id": rec["id"],
                        "source": rec["source"],
                        "context": rec["text"],
                        "response": rec["response"],
                        "group_id": rec["group_id"],
                        "split": rec["split"]
                    })
                    
                # 3. CSAT Prediction (Customer Support remarks -> CSAT score)
                if rec["source"] == "customer_support" and meta.get("csat_score") is not None:
                    csat_records.append({
                        "id": rec["id"],
                        "text": rec["text"],
                        "category": meta.get("category"),
                        "sub_category": meta.get("sub_category"),
                        "channel": meta.get("channel"),
                        "csat_score": meta.get("csat_score"),
                        "split": rec["split"]
                    })
                    
        # Save per-split supervised parquet
        if cls_records:
            pd.DataFrame(cls_records).to_parquet(
                os.path.join(SUPERVISED_DIR, f"classification_{split_name}.parquet"), compression="snappy"
            )
        if dlg_records:
            pd.DataFrame(dlg_records).to_parquet(
                os.path.join(SUPERVISED_DIR, f"dialogue_{split_name}.parquet"), compression="snappy"
            )
        if csat_records:
            pd.DataFrame(csat_records).to_parquet(
                os.path.join(SUPERVISED_DIR, f"csat_{split_name}.parquet"), compression="snappy"
            )


def audit_leakage(writer: CorpusWriter) -> Dict[str, Any]:
    """Verifies that there is zero overlap between train, val, and test group sets."""
    train_groups = writer.group_sets["train"]
    val_groups = writer.group_sets["val"]
    test_groups = writer.group_sets["test"]
    
    tv_overlap = len(train_groups.intersection(val_groups))
    tt_overlap = len(train_groups.intersection(test_groups))
    vt_overlap = len(val_groups.intersection(test_groups))
    
    leakage_free = (tv_overlap == 0 and tt_overlap == 0 and vt_overlap == 0)
    
    return {
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "test_groups": len(test_groups),
        "train_val_overlap": tv_overlap,
        "train_test_overlap": tt_overlap,
        "val_test_overlap": vt_overlap,
        "is_strictly_leakage_free": leakage_free
    }


def main():
    print("=" * 70)
    print("      SENTRICORPUS CANONICAL PIPELINE BUILDER")
    print("=" * 70)
    start_time = time.time()
    init_dirs()
    
    seen_hashes: Set[str] = set()
    writer = CorpusWriter(OUTPUT_DIR)
    
    # 1. Customer Support Dataset
    print("\n[1/4] Extracting Customer Support Dataset...")
    cs_count = 0
    for record in extract_customer_support(CUSTSUPP_PATH, seen_hashes):
        writer.write(record)
        cs_count += 1
    print(f"      Extracted {cs_count:,} unique clean customer support records.")
    
    # 2. Bitext Dataset
    print("\n[2/4] Extracting Bitext Instruction Dataset...")
    bi_count = 0
    for record in extract_bitext(seen_hashes):
        writer.write(record)
        bi_count += 1
    print(f"      Extracted {bi_count:,} unique clean Bitext records.")

    # 3. TWCS Dataset
    print("\n[3/4] Extracting TWCS Multi-Turn Conversation Dataset...")
    tw_count = 0
    for record in extract_twcs(TWCS_PATH, seen_hashes):
        writer.write(record)
        tw_count += 1
    print(f"      Extracted {tw_count:,} unique conversation turns.")
    
    # 4. CFPB Dataset (Streaming chunk by chunk)
    print("\n[4/4] Streaming and Extracting CFPB Complaints (9.18 GB)...")
    cfpb_count = 0
    for record in extract_cfpb(CFPB_PATH, seen_hashes, chunk_size=50000):
        writer.write(record)
        cfpb_count += 1
        if cfpb_count % 250000 == 0:
            print(f"      Progress: {cfpb_count:,} unique CFPB complaints processed...")
    print(f"      Extracted {cfpb_count:,} unique clean CFPB complaints.")
    
    writer.close()
    
    # Leakage Audit
    print("\n" + "-" * 50)
    print("Verifying Dataset Split Leakage...")
    leakage_stats = audit_leakage(writer)
    print(f"  Leakage Check Result: {'PASSED (Zero Leakage)' if leakage_stats['is_strictly_leakage_free'] else 'FAILED'}")
    print(f"  Train Groups: {leakage_stats['train_groups']:,} | Val: {leakage_stats['val_groups']:,} | Test: {leakage_stats['test_groups']:,}")
    print(f"  Overlaps: Train-Val={leakage_stats['train_val_overlap']}, Train-Test={leakage_stats['train_test_overlap']}, Val-Test={leakage_stats['val_test_overlap']}")
    
    # Convert JSONL to Parquet
    print("\n" + "-" * 50)
    print("Converting JSONL to High-Performance Parquet Format...")
    train_jsonl = os.path.join(OUTPUT_DIR, "train.jsonl")
    val_jsonl = os.path.join(OUTPUT_DIR, "val.jsonl")
    test_jsonl = os.path.join(OUTPUT_DIR, "test.jsonl")
    
    convert_jsonl_to_parquet(train_jsonl, os.path.join(OUTPUT_DIR, "train.parquet"))
    convert_jsonl_to_parquet(val_jsonl, os.path.join(OUTPUT_DIR, "val.parquet"))
    convert_jsonl_to_parquet(test_jsonl, os.path.join(OUTPUT_DIR, "test.parquet"))
    
    # Export Supervised Subsets
    print("\n" + "-" * 50)
    export_supervised_subsets(train_jsonl, val_jsonl, test_jsonl)
    
    # Calculate corpus statistics
    total_records = sum(writer.counts.values())
    elapsed = time.time() - start_time
    
    stats = {
        "build_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 2),
        "total_canonical_records": total_records,
        "splits": {
            "train": {
                "count": writer.counts["train"],
                "percentage": round(writer.counts["train"] / total_records * 100, 2)
            },
            "val": {
                "count": writer.counts["val"],
                "percentage": round(writer.counts["val"] / total_records * 100, 2)
            },
            "test": {
                "count": writer.counts["test"],
                "percentage": round(writer.counts["test"] / total_records * 100, 2)
            }
        },
        "sources": writer.source_counts,
        "leakage_audit": leakage_stats
    }
    
    stats_path = os.path.join(OUTPUT_DIR, "corpus_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        
    print("\n" + "=" * 70)
    print(f"SENTRICORPUS CREATION COMPLETE IN {elapsed:.1f} SECONDS")
    print(f"Total Unique Canonical Records: {total_records:,}")
    print(f"  - Train: {writer.counts['train']:,} ({writer.counts['train']/total_records*100:.2f}%)")
    print(f"  - Val:   {writer.counts['val']:,} ({writer.counts['val']/total_records*100:.2f}%)")
    print(f"  - Test:  {writer.counts['test']:,} ({writer.counts['test']/total_records*100:.2f}%)")
    print(f"Corpus statistics saved to: {stats_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
