# SentriFormer + SentriMail

**SentriFormer** is a compact (11.65M-parameter) multi-task encoder–decoder Transformer, trained from scratch on customer-support data. One model does 33-class complaint routing, sentiment analysis and CSAT (satisfaction) prediction, and drafts support replies.
**SentriMail** is a human-in-the-loop complaint-management web app (FastAPI + MongoDB) that uses SentriFormer for triage and admin-reviewed draft replies.

Each project has a research paper (IEEE two-column) in [`research_publications/`](research_publications/):
[SentriFormer paper (PDF)](research_publications/SentriFormer_Paper/paper.pdf) ·
[SentriMail paper (PDF)](research_publications/SentriMail_Paper/paper.pdf) ·
[summary](research_publications/FINAL_RESEARCH_SUMMARY.md)

## Results

| SentriFormer (sealed test set, scored once) | Value |
|---|---|
| Routing accuracy, 33 classes (n = 123,229) | **73.7%** (majority baseline 27.2%), weighted-F1 0.737, ECE 0.055 |
| Sentiment (n = 20,453) | **88.1%** accuracy, macro-F1 0.758 |
| CSAT, 1–5 stars (n = 870) | 72.4% exact, **82.0% within ±1 star** |
| Replies that loop the customer back to a portal | **31.3% → 0.7%** |
| Deployed T5a: broken-link replies / generic phrasing (validation, pre-registered replication) | 5.8% → **0.4%** / 25.8% → **16.6%** |
| Latency, laptop CPU | 30 ms all classification heads, 275 ms including reply |

The full experiment history, including rejected experiments (label-smoothing collapse, MMI decoding, PMI filter, neuron steering, urgency head), is in [`sentriformer/wsp_runs/WSP_FINAL_REPORT.md`](sentriformer/wsp_runs/WSP_FINAL_REPORT.md) and the `*_DECISION.md` files.


**All training and metric figures** (tokenizer, pre-training, multi-task/SFT training curves, confusion matrices, CIs): [`docs/figures/training_and_metrics/`](docs/figures/training_and_metrics/README.md) · raw-data analysis: [`docs/data_analysis/`](docs/data_analysis/README.md)

## Repository layout

```
sentriformer/            model package
  model/                 encoder, decoder, attention, multi-task heads
  tokenizer/             byte-level BPE (16k) + benchmark of 16k/24k/32k vocabularies
  corpus/                SentriCorpus build: extract, clean, dedup, group-aware split
  training/              pre-training (span corruption) and M-series task training
  wsp/                   Phase-T training (train_v2.py), evaluation harness, inference.py
  wsp_runs/              experiment records: criteria, decisions, metrics, training curves
    FINAL_T5a/           deployed checkpoint (sentriformer_t5a.pt, 46.7 MB) + model card
  results/               plots and reports from pre-training and the M-series
  evaluation/            behavioural audit scripts
  scripts/               data inspection and plotting utilities
apps/sentrimail/         complaint-management web app (FastAPI, Jinja2, MongoDB)
legacy/model_training/   earlier T5-small pipeline (code only, superseded)
docs/                    data sources, project reports, architecture diagram
research_publications/   both papers (LaTeX + PDF), evidence, verification reports, benchmark scripts
```

## Quick start

```bash
pip install torch transformers tokenizers pyarrow numpy
# analyse one complaint with the deployed model
python -I sentriformer/wsp/inference.py "My card was charged twice and I still have no refund."
```

SentriMail:

```bash
cd apps/sentrimail
pip install -r requirements.txt torch
cp .env.example .env      # set MONGODB_URI and the optional SMTP settings
python run.py             # http://localhost:8088
```

SentriMail loads SentriFormer from `../../../sentriformer` relative to `app/` (this repo's layout). Set `SENTRIFORMER_DIR` to override it. Without SentriFormer, the app falls back to DistilBERT, rules and templates.

## Data and licence notes

- **No dataset text is included.** The raw corpora are public; see [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) for sources: CFPB Consumer Complaint Database, Twitter Customer Support (Kaggle), Bitext, and a Kaggle e-commerce support set. Result files that embedded model outputs or reference replies were published as numeric-only `*_numeric.json` copies.
- The model weights were trained partly on TWCS, which is licensed **CC BY-NC-SA 4.0**. Treat the checkpoint as **non-commercial / research use only**.
- `apps/sentrimail/app/auth.py` seeds demo accounts with demo passwords. Change them before any real deployment.

## Known limitations (also stated in the papers)

- The deployed T5a checkpoint has been evaluated on validation data only. The sealed-test numbers above belong to its parent, T4d.
- There is no human evaluation of replies and no SentriMail user study. All runs use a single seed.
- The urgency head is not learned (do not use it).
- Some legacy figures and tables in `docs/reports/research_comparative_evaluation.md` and `sentriformer/results/research_plots/` (efficiency comparison fig4, department matrix fig5) use hard-coded values that were **not measured**. They are kept for history only; use the papers for verified numbers.
