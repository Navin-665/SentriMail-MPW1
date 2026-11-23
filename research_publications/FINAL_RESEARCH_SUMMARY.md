# Final research summary

Two independent IEEE two-column conference papers, built from a read-only audit of the SENTRIMAIL-MPW1 project (2026-11-23).

| | SentriFormer | SentriMail |
|---|---|---|
| PDF | `SentriFormer_Paper/paper.pdf` (7 pp.) | `SentriMail_Paper/paper.pdf` (6 pp.) |
| Source | `main.tex`, `tables/*.tex`, `figures/*`, `references.bib` | same |
| Contribution type | model and methodology (empirical) | system architecture and integration |
| Evidence | `research_evidence.md` | `research_evidence.md` |
| Checks | `verification_report.md` | `verification_report.md` |

To rebuild, run `tectonic -X compile main.tex` in either folder. To regenerate the figures and measurements:
- `python -I _scripts/make_figures.py <sentriformer_dir> SentriFormer_Paper/figures`
- `python -I _scripts/make_sentrimail_figures.py _runs SentriMail_Paper/figures`
- `python -I _scripts/bench_latency.py <sentriformer_dir> _runs/latency_benchmark.json`
- `python -I -B _scripts/bench_pipeline.py <sentrimail_app_dir> _runs/pipeline_benchmark.json`

## Headline results (all traceable to files)

**SentriFormer**

| Result | Value | Split |
|---|---|---|
| Parameters, trained from scratch | 11.65M | — |
| Routing accuracy (33 classes) | **73.71%** (majority 27.2%; weighted-F1 0.737; ECE 0.055) | sealed test, n = 123,229 |
| Sentiment | **88.1% / macro-F1 0.758** | sealed test |
| CSAT | 72.4% exact / **82.0% within ±1 star** | sealed test |
| Portal-loop replies | **31.3% → 0.7%** | sealed test |
| Deployed T5a, broken-link replies | **5.8% → 0.4%** | pre-registered replication, validation |
| Deployed T5a, generic phrasing | **25.8% → 16.6%** | pre-registered replication, validation |
| Latency on laptop CPU | 30 ms all heads; 275 ms with reply | measured |
| Negative results reported | T1/T2 label-smoothing collapse, E2, P1/P2 steering, T5c MMI, T6 PMI filter, T7u urgency | — |

**SentriMail**
- Multilingual (11 languages) and voice/audio intake; hybrid priority engine.
- Human-approved drafts.
- Integrated analysis takes 531 ms median on a laptop GPU.
- On the 30-message benchmark, the only auto-replies were the 2 thank-you messages, and 5 of 30 low-confidence routes were flagged for review.

## Missing evidence (stated in the papers)
- The deployed T5a has never been run on the sealed test set (one inference run would fix this).
- No human evaluation of replies.
- No SentriMail user study and no priority-accuracy study.
- Single seed; no TF-IDF or DistilBERT baselines trained on the same splits.
- Urgency is not learned.
- Authors, affiliations and acknowledgment are placeholders.
- No target venue has been checked; neither paper is submitted or accepted.

## Supporting material
- `_lit/literature_review.md`: themed review and verification table for all references.
- `_lit/ieee_format_notes.md`: IEEEtran requirements.
- `_runs/`: measured latency and pipeline benchmark outputs.
- `_scripts/`: figure and benchmark scripts.
