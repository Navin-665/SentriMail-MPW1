"""
Processes each of the 5 data sources independently, checkpointing each one
to its own parquet file so it can be inspected before combining. This
replaces the old single monolithic run - that version had no checkpoints,
so there was no way to verify one source before moving to the next, and no
way to recover partial progress if it needed to be stopped.

Run each source separately first:
    python build_dataset.py --stage bitext
    python build_dataset.py --stage cfpb
    python build_dataset.py --stage twitter
    python build_dataset.py --stage goemotions
    python build_dataset.py --stage ecommerce

Inspect each checkpoint (data/processed/_checkpoint_*.parquet for the
three distilled ones, emotion_supplement.parquet and
validation_ecommerce.parquet for the other two) before moving on.

Once all five look right, pool the three training sources into the final
stratified splits:
    python build_dataset.py --stage combine

Or run everything in sequence in one command, same as the old behavior:
    python build_dataset.py --stage all

Dry run any stage on a small slice first with --max-rows:
    python build_dataset.py --stage bitext --max-rows 500

CFPB and Twitter have no cap on the real run (Twitter caps at 150,000 by
design - see loaders.py). CFPB's full narrative-bearing volume is large,
so its stage is the slow one - that's expected, not a hang.
"""

import argparse
import os
import langdetect
import pandas as pd
from sklearn.model_selection import train_test_split

from loaders import (
    load_bitext, load_cfpb, load_twitter,
    load_ecommerce_validation, load_goemotions,
)
from distill import distill_labels

langdetect.DetectorFactory.seed = 0  # deterministic across runs

CFPB_CSV_PATH = "data/cfpb_complaints.csv"
if not os.path.exists(CFPB_CSV_PATH) and os.path.exists("data/complaints.csv"):
    CFPB_CSV_PATH = "data/complaints.csv"

TWITTER_CSV_PATH = "data/twcs.csv"
OUTPUT_DIR = "data/processed"

CHECKPOINTS = {
    "bitext": f"{OUTPUT_DIR}/_checkpoint_bitext.parquet",
    "cfpb": f"{OUTPUT_DIR}/_checkpoint_cfpb.parquet",
    "twitter": f"{OUTPUT_DIR}/_checkpoint_twitter.parquet",
}


def stage_bitext(max_rows, device, batch_size, fp16):
    print("loading bitext...")
    df = load_bitext(max_rows=max_rows)
    print(f"distilling {len(df)} bitext rows...")
    df = distill_labels(df, device=device, batch_size=batch_size, fp16=fp16)
    df.to_parquet(CHECKPOINTS["bitext"], index=False)
    print(f"saved {CHECKPOINTS['bitext']} ({len(df)} rows)")


def stage_cfpb(max_rows, device, batch_size, fp16):
    print("loading cfpb (target: 300,000 unique context-rich complaints)...")
    df = load_cfpb(CFPB_CSV_PATH, max_rows=max_rows or 300_000)
    print(f"distilling {len(df):,} cfpb rows...")
    df = distill_labels(df, device=device, batch_size=batch_size, fp16=fp16)
    df.to_parquet(CHECKPOINTS["cfpb"], index=False)
    print(f"saved {CHECKPOINTS['cfpb']} ({len(df):,} rows)")


def stage_twitter(max_rows, device, batch_size, fp16):
    print("loading twitter...")
    df = load_twitter(TWITTER_CSV_PATH, max_pairs=max_rows or 150_000)
    print(f"distilling {len(df)} twitter rows...")
    df = distill_labels(df, device=device, batch_size=batch_size, fp16=fp16)
    df.to_parquet(CHECKPOINTS["twitter"], index=False)
    print(f"saved {CHECKPOINTS['twitter']} ({len(df)} rows)")


def stage_goemotions(max_rows):
    print("loading goemotions...")
    df = load_goemotions(max_rows=max_rows)
    df.to_parquet(f"{OUTPUT_DIR}/emotion_supplement.parquet", index=False)
    print(f"saved emotion_supplement.parquet ({len(df)} rows)")


def stage_ecommerce(max_rows, device=None, batch_size=128, fp16=False):
    print("loading ecommerce validation set...")
    df = load_ecommerce_validation(max_rows=max_rows)
    print(f"distilling {len(df)} ecommerce rows...")
    df = distill_labels(df, device=device, batch_size=batch_size, fp16=fp16)
    df.to_parquet(f"{OUTPUT_DIR}/validation_ecommerce.parquet", index=False)
    print(f"saved validation_ecommerce.parquet ({len(df)} rows)")


def stage_combine():
    missing = [name for name, path in CHECKPOINTS.items() if not os.path.exists(path)]
    if missing:
        raise SystemExit(
            f"missing checkpoints for: {missing} - run "
            f"'python build_dataset.py --stage {missing[0]}' first"
        )

    bitext = pd.read_parquet(CHECKPOINTS["bitext"])
    cfpb = pd.read_parquet(CHECKPOINTS["cfpb"])
    twitter = pd.read_parquet(CHECKPOINTS["twitter"])

    pool = pd.concat([bitext, cfpb, twitter], ignore_index=True)
    before_dedup = len(pool)
    pool = pool.dropna(subset=["text"]).drop_duplicates(subset=["text"])
    print(f"pool after dedup: {len(pool)} rows (dropped {before_dedup - len(pool)} duplicate texts)")

    # sentiment/emotion models and the eventual generation model are all
    # English-only. langdetect is unreliable on short text (verified:
    # "help me", "app not working", "iOS 11.1" all misdetect as
    # non-English), so only run detection on text long enough to trust.
    MIN_CHARS_FOR_LANGDETECT = 30

    def _is_english(text):
        if len(text) < MIN_CHARS_FOR_LANGDETECT:
            return True
        try:
            return langdetect.detect(text) == "en"
        except langdetect.LangDetectException:
            return False

    is_en = pool["text"].apply(_is_english)
    print(f"dropping {(~is_en).sum()} non-English rows, {is_en.sum()} remain")
    pool = pool[is_en]

    strat_key = pool["category"].fillna("unlabeled")
    try:
        train, temp = train_test_split(pool, test_size=0.2, random_state=42, stratify=strat_key)
        val, test = train_test_split(
            temp, test_size=0.5, random_state=42, stratify=temp["category"].fillna("unlabeled")
        )
    except ValueError as exc:
        print(f"stratified split failed ({exc}) - falling back to a plain "
              f"random split. Expected on small samples, shouldn't happen "
              f"on the full dataset.")
        train, temp = train_test_split(pool, test_size=0.2, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)

    train.to_parquet(f"{OUTPUT_DIR}/train.parquet", index=False)
    val.to_parquet(f"{OUTPUT_DIR}/val.parquet", index=False)
    test.to_parquet(f"{OUTPUT_DIR}/test.parquet", index=False)

    print(f"\ntrain: {len(train)}  val: {len(val)}  test: {len(test)}")
    print(f"written to {OUTPUT_DIR}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", required=True,
        choices=["bitext", "cfpb", "twitter", "goemotions", "ecommerce", "combine", "all"],
        help="which source to process and checkpoint, 'combine' to pool the "
             "three checkpointed training sources into train/val/test, or "
             "'all' to run every stage in sequence",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="distillation batch size - raised from the old default of 32, "
             "which left the GPU mostly idle between small batches; 6GB "
             "VRAM has comfortable room for 128+ with these small models",
    )
    parser.add_argument(
        "--fp16", action="store_true",
        help="run distillation in half precision - try this if utilization "
             "is still low after raising batch size, or push batch size "
             "even higher alongside it",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stages = {
        "bitext": lambda: stage_bitext(args.max_rows, args.device, args.batch_size, args.fp16),
        "cfpb": lambda: stage_cfpb(args.max_rows, args.device, args.batch_size, args.fp16),
        "twitter": lambda: stage_twitter(args.max_rows, args.device, args.batch_size, args.fp16),
        "goemotions": lambda: stage_goemotions(args.max_rows),
        "ecommerce": lambda: stage_ecommerce(args.max_rows, args.device, args.batch_size, args.fp16),
        "combine": stage_combine,
    }

    if args.stage == "all":
        for name in ["bitext", "cfpb", "twitter", "goemotions", "ecommerce", "combine"]:
            stages[name]()
    else:
        stages[args.stage]()


if __name__ == "__main__":
    main()