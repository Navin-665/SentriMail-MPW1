import random
import re

import pandas as pd
from datasets import load_dataset

from category_map import BITEXT_MAP, ECOMMERCE_MAP, CFPB_MAP, map_category
from emotion_map import GOEMOTIONS_LABELS, GOEMOTIONS_TO_SENTRI

COLUMNS = ["text", "category", "sentiment", "emotion", "response", "source"]

# rough profanity/slur filter for the reddit-sourced goemotions text -
# extend this list if you spot more junk getting through
BAD_WORDS = ["fuck", "nigger", "cunt", "faggot", "retard"]


def _sample(ds, max_rows):
    # Bitext/GoEmotions/ecommerce rows are grouped by category in file
    # order - a plain prefix slice (train[:500]) grabs whatever category
    # happens to sit at the top of the file, not a representative sample.
    # shuffle first so a small --max-rows dry run actually sees variety.
    if not max_rows:
        return ds
    return ds.shuffle(seed=42).select(range(min(max_rows, len(ds))))


def load_bitext(max_rows=None):
    ds = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split="train",
    )
    ds = _sample(ds, max_rows)
    df = ds.to_pandas()
    return pd.DataFrame({
        "text": df["instruction"],
        "category": df["category"].apply(lambda c: map_category(c, BITEXT_MAP)),
        "sentiment": None,
        "emotion": None,
        "response": df["response"],
        "source": "bitext",
    })


def load_ecommerce_validation(max_rows=None):
    # held out for sanity checks only - never goes in the training pool
    ds = load_dataset("Ataur77/ecommerce-customer-support", split="train")
    ds = _sample(ds, max_rows)
    df = ds.to_pandas()
    return pd.DataFrame({
        "text": df["Complaint"],
        "category": df["Category"].apply(lambda c: map_category(c, ECOMMERCE_MAP)),
        "sentiment": None,
        "emotion": None,
        "response": None,
        "urgency": df["Urgency"],
        "source": "ecommerce_validation",
    })


def load_goemotions(max_rows=None):
    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split="train")
    ds = _sample(ds, max_rows)
    df = ds.to_pandas()

    df = df[df["labels"].apply(len) == 1].copy()  # keep it simple, drop multi-label rows
    df["label_name"] = df["labels"].apply(lambda ids: GOEMOTIONS_LABELS[ids[0]])
    df["emotion"] = df["label_name"].map(GOEMOTIONS_TO_SENTRI)

    mask = ~df["text"].str.lower().str.contains("|".join(BAD_WORDS), na=False)
    df = df[mask]

    return pd.DataFrame({
        "text": df["text"],
        "category": None,
        "sentiment": None,
        "emotion": df["emotion"],
        "response": None,
        "source": "goemotions",
    })


def load_cfpb(csv_path, max_rows=None):
    """Loads context-rich, unique CFPB complaint narratives up to max_rows.
    When max_rows is None (default), loads all qualifying rows — the cap is
    controlled by build_dataset.py's --max-rows flag, not here.
    Deduplicates and filters for rich text (>= 50 chars) on the fly."""
    usecols = ["Consumer complaint narrative", "Product"]
    skip = random.Random(42).randint(0, 200_000) if max_rows else 0
    seen_texts = set()
    frames = []
    total = 0
    reader = pd.read_csv(
        csv_path, usecols=usecols, chunksize=50_000,
        skiprows=range(1, skip + 1) if skip else None,
        low_memory=False,
    )
    for chunk in reader:
        chunk = chunk.dropna(subset=["Consumer complaint narrative"])
        # Keep complaints with rich context (>= 50 chars) and drop duplicates upfront
        chunk = chunk[chunk["Consumer complaint narrative"].str.len() >= 50]
        chunk = chunk[~chunk["Consumer complaint narrative"].isin(seen_texts)]
        if chunk.empty:
            continue
        seen_texts.update(chunk["Consumer complaint narrative"])
        frames.append(chunk)
        total += len(chunk)
        if max_rows and total >= max_rows:
            break

    if not frames:
        return pd.DataFrame(columns=COLUMNS)

    df = pd.concat(frames, ignore_index=True)
    if max_rows:
        df = df.head(max_rows)
    print(f"load_cfpb: loaded {len(df):,} unique, context-rich complaints")
    return pd.DataFrame({
        "text": df["Consumer complaint narrative"],
        "category": df["Product"].apply(lambda c: map_category(c, CFPB_MAP)),
        "sentiment": None,
        "emotion": None,
        "response": None,
        "source": "cfpb",
    })


def load_twitter(csv_path, max_pairs=150_000):
    MIN_COMPLAINT_WORDS = 4
    MIN_RESPONSE_WORDS = 3
    BOILERPLATE_RE = re.compile(r"\bDM\b|direct message|reach out to us", re.IGNORECASE)

    df = pd.read_csv(csv_path, usecols=[
        "tweet_id", "author_id", "inbound", "text",
        "response_tweet_id", "in_response_to_tweet_id",
    ], low_memory=False)
    df["inbound"] = df["inbound"].astype(str).str.upper() == "TRUE"

    inbound_all = df[df["inbound"]]
    # only keep inbound tweets that start a thread - one that's itself a
    # reply to the company (in_response_to_tweet_id set) is a mid-thread
    # fragment ("iOS 11.0.3" answering "what's your OS version?"), not a
    # standalone complaint, and means nothing without the prior turns
    top_of_thread = inbound_all[inbound_all["in_response_to_tweet_id"].isna()].set_index("tweet_id")
    outbound = df[~df["inbound"]].set_index("tweet_id")
    print(f"load_twitter: {len(inbound_all):,} inbound tweets, "
          f"{len(top_of_thread):,} start a new thread")

    pairs = []
    no_reply = 0
    too_short = 0
    boilerplate_dropped = 0
    for tweet_id, row in top_of_thread.iterrows():
        reply_id = row["response_tweet_id"]
        if pd.isna(reply_id):
            no_reply += 1
            continue
        first_reply_id = str(reply_id).split(",")[0].strip()  # can be a list of ids
        try:
            reply = outbound.loc[int(first_reply_id)]
            if isinstance(reply, pd.DataFrame):
                reply = reply.iloc[0]  # duplicate tweet_id in the outbound rows
        except (KeyError, ValueError):
            no_reply += 1
            continue

        complaint_text = str(row["text"]).strip()
        response_text = str(reply["text"]).strip()

        # drop exchanges too short to teach the decoder anything about
        # real response phrasing - a bare "Thanks!" isn't a training example
        if len(complaint_text.split()) < MIN_COMPLAINT_WORDS or len(response_text.split()) < MIN_RESPONSE_WORDS:
            too_short += 1
            continue

        # checked here, inside the loop, not after collection - so
        # max_pairs counts final usable pairs, not pre-filter candidates
        # that then get undershot once boilerplate gets stripped out
        if BOILERPLATE_RE.search(response_text):
            boilerplate_dropped += 1
            continue

        pairs.append((complaint_text, response_text))
        if max_pairs and len(pairs) >= max_pairs:
            break

    print(f"load_twitter: {no_reply:,} had no matched reply, "
          f"{too_short:,} dropped by word-count, "
          f"{boilerplate_dropped:,} dropped as boilerplate, "
          f"{len(pairs):,} final pairs kept")

    complaints, replies = zip(*pairs) if pairs else ([], [])
    frame = pd.DataFrame({"text": complaints, "response": replies})
    frame["text"] = frame["text"].str.replace(r"@\w+", "", regex=True).str.strip()
    frame["response"] = frame["response"].str.replace(r"@\w+", "", regex=True).str.strip()
    frame["category"] = None
    frame["sentiment"] = None
    frame["emotion"] = None
    frame["source"] = "twitter"

    return frame[COLUMNS]