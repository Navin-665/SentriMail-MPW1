# SentriMail dataset prep

Pools Bitext, CFPB, and Twitter complaint text into one training-ready
set, using the sentiment/emotion models already in `ai_engine.py` to
pseudo-label the rows that don't come with labels of their own.

## Where this folder lives

Place this whole `model_training/` folder next to `sentrimail/` in the
repo, not inside it:

```
SentriMail-MPW1/
├── sentrimail/        <- existing app, untouched
└── model_training/    <- this folder
```

Add a `.gitignore` here containing:

```
data/
!data/.gitkeep
```

so the raw CSVs and processed output never get committed - both would
blow past GitHub's 100MB file limit anyway.

## Setup

```bash
pip install -r requirements.txt
```

Put the two files you downloaded manually here:

```
data/cfpb_complaints.csv   <- CFPB full complaint export, renamed
data/twcs.csv              <- Kaggle Customer Support on Twitter
```

Bitext, `ecommerce-customer-support`, and GoEmotions load straight from
the Hub the first time you run the script (needs internet access, no
manual download).

## Before the full run: dry-run on a slice

```python
from loaders import load_bitext, load_cfpb, load_twitter

df = load_bitext()
print(df["category"].value_counts())

df = load_cfpb("data/cfpb_complaints.csv", max_rows=2000)
print(df["category"].value_counts())

df = load_twitter("data/twcs.csv", max_pairs=2000)
print(df.head())
```

Check the `value_counts()` against `category_map.py` before trusting a
full run - `map_category()` silently sends anything unmapped to
`"other"`, so a missing key won't error, it'll just quietly mislabel
rows.

## Run

```bash
python build_dataset.py
```

CFPB has **no row cap by default**. That was a deliberate call: rather
than capping or rebalancing the CFPB category distribution upfront
(which is dominated by "Credit reporting" complaints), we're keeping
the full, naturally imbalanced data and handling that downstream with a
stratified train/val/test split (already in this script) plus
class-weighted loss and macro-F1 evaluation (in the training script,
not built yet). That means a full run scans the entire CFPB file and
distills sentiment/emotion over every narrative-bearing row - this is
genuinely slow. Test with a small `max_rows`/`max_pairs` override
first, then run the real pass on a GPU (free Colab is enough) rather
than CPU - distillation is inference-only and batches well.

## What comes out, in `data/processed/`

- `train.parquet`, `val.parquet`, `test.parquet` — the pooled
  Bitext + CFPB + Twitter rows (text, category, sentiment, emotion,
  response — some fields null depending on source), stratified 80/10/10
  split by category so every split reflects the real distribution.
- `emotion_supplement.parquet` — GoEmotions rows, emotion-labeled
  only. Use these to train the emotion head, not as complaint text.
- `validation_ecommerce.parquet` — the ecommerce-customer-support set,
  held out entirely. Use this to sanity-check the trained model, never
  fold it into training.

## Known rough edges

- The category mappings in `category_map.py` are hand-eyeballed from
  the unique values each dataset actually has. Print
  `df["category"].value_counts()` after loading each source and adjust
  the dicts if something's landing in the wrong bucket.
- CFPB rows and most Twitter rows have no ground-truth category or
  response — that's expected, they're there to teach the model
  complaint *language*, not to fill every column.
- `load_twitter` caps at 150k pairs by default (`max_pairs` arg) -
  raise it once the pipeline runs cleanly end to end on a smaller slice.

## Not yet built - next phase

- The multi-task model itself: encoder + 3 classification heads
  (sentiment/emotion/category) + decoder for response generation.
- Response generation is meant to be constrained, not freely
  generated - the decoder should draft tone/phrasing, with factual
  content limited to a verified per-category facts table (an expanded
  version of `CATEGORY_ROOT_CAUSES` in `ai_engine.py`), not invented
  by the model. This matters most for billing/refund categories where
  a wrong generated promise is worse than no response.
- Training script needs class-weighted loss and evaluation via
  macro-F1 plus per-class precision/recall on the untouched, naturally
  distributed test split - not accuracy alone, given the CFPB skew.