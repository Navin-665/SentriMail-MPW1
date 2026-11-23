# SentriMail paper: verification report

Checked 2026-11-23. Build: `tectonic -X compile main.tex` (IEEEtran conference) → `paper.pdf`, 6 pages, two columns.

## 1. Compilation and layout
| Check | Result |
|---|---|
| Errors / overfull boxes / undefined citations | none |
| Two-column IEEE layout | yes, checked on the rendered pages |
| Figures | Fig. 1 (TikZ architecture): an overlapping label was found and fixed, then re-checked. Fig. 2 (TikZ workflow), Fig. 3 (two UI screenshots from the original repo, with no personal data visible), Fig. 4 (measured latency) |
| Tables | I stack, II model evidence, III latency, IV pipeline decisions. All fit their columns |

## 2. Every claim traced to its source
`APP` = `SENTRIMAIL-MPW1/Sentrimail/sentrimail`.

| Claim | Source |
|---|---|
| FastAPI, 22 endpoints, MongoDB collections and indexes, cookie sessions, roles | `APP/app/main.py`, `auth.py`, `mongodb.py`, `storage.py` (evidence §2) |
| Whisper base, langdetect + translation, 11 UI languages | `APP/app/main.py` L312–338, L727–745; `templates/submit_complaint.html` |
| Priority formula (Eq. 1), thresholds 75/50/25, 18 triggers, 11 safety keywords | `APP/app/ai_engine.py` L285–333, L68–72; `main.py` L345–351 |
| Hourly escalation, 24 h / 48 h | `APP/app/main.py` L141–163 |
| SentriFormer wiring (sentiment → priority; routing, CSAT, draft and flags shown to admins only; never sent; urgency ignored; fallbacks) | `git diff` of `ai_engine.py`, `main.py`, `complaint_detail.html` (evidence §3.2) |
| Table II model metrics | SentriFormer evidence: `eval_full/T4d_test/metrics.json`, `t5/t5a_retest.json` |
| Table III and Fig. 4 latency | `research_publications/_runs/latency_benchmark.json` (script `_scripts/bench_latency.py`) |
| Table IV and §VI-B pipeline behaviour, 531 ms, first call 43.2 s | `research_publications/_runs/pipeline_benchmark.json` (script `_scripts/bench_pipeline.py`) |
| Auto-ack branch checks only LOW; `auto_resolvable` not enforced | `APP/app/main.py` L365–387 vs `ai_engine.py` L347–364 |

## 3. Integrity checks
- **Benchmark scope.** The benchmark messages were written for this evaluation (30 messages, 4–58 words), and the paper says so. They are not presented as customer traffic, and Table IV is explicitly described as behaviour, not accuracy.
- **No unmeasured claims.** No user study, accuracy-on-traffic or time-saving claim is made. These gaps are listed under Limitations.
- **No unimplemented features.** RAG, inbound e-mail and LLM APIs are not claimed; RAG appears only as future work. Routing is described as advisory.
- **Known defects disclosed.** Auto-reply gate, page re-analysis, session and password handling, JSON fallback.
- **No repeated contribution.** The model contribution (training, failure modes, data filtering) belongs to the SentriFormer paper. This paper cites model metrics only as inputs to the system design.

## 4. Unresolved items
1. Author, affiliation and acknowledgment placeholders.
2. No end-user, admin or priority-accuracy evaluation exists. A study is proposed in §VII.
3. The latency figures exclude translation (external service) and SMTP.
4. The SentriFormer integration is uncommitted in the original SentriMail git repo (it is included in the clean GitHub copy).
5. No target conference has been checked.
