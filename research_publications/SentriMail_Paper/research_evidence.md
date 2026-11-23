# SentriMail — Verified Research Evidence Report

Prepared 2026-11-23 for an IEEE conference paper. Read-only audit: nothing in the source trees was modified, committed or executed against a database.

**Conventions.** Paths are relative to `SENTRIMAIL-MPW1/` unless stated. `APP/` = `SENTRIMAIL-MPW1/Sentrimail/sentrimail/` (the git working tree; repo root is `SENTRIMAIL-MPW1/Sentrimail/`). `SF/` = `SENTRIMAIL-MPW1/sentriformer/`. Line numbers refer to the working-tree files on 2026-11-23, which include the uncommitted SentriFormer edits. Anything not confirmed from code or files is marked **UNVERIFIED**. No corpus text, customer text, email addresses, secrets or passwords are reproduced here.

---

## 0. Key findings

| # | Finding | source |
|---|---|---|
| F1 | SentriMail is a **FastAPI + Jinja2** web app. Complaints come in through a **web form** (typed text, browser speech-to-text, or an uploaded audio file transcribed by Whisper). **It does not read email (no IMAP or Gmail API).** Email is outbound only, over SMTP. | `APP/app/main.py` L1-17, L271-309, L312-400, L727-745 |
| F2 | Before SentriFormer, the AI was a **hybrid of pretrained Hugging Face pipelines and rules**: DistilBERT-SST-2 for sentiment, `j-hartmann/emotion-english-distilroberta-base` for emotion, `google/flan-t5-small` for auto-reply text, rule-based priority scoring, keyword escalation, template replies, and a TF-IDF nearest-neighbour reply retriever. | `APP/app/ai_engine.py` L209-240, L285-333, L367-466, L127-176 |
| F3 | The `model_training/` T5-small multi-task model (`SentriModel`) was **never wired into SentriMail**. No SentriMail file imports it; it only *reused* SentriMail's DistilBERT/emotion models to pseudo-label training data. | `model_training/model.py`, `model_training/distill.py` L1-40, `model_training/README.md`; grep of `APP/app/*.py` |
| F4 | The SentriFormer integration is **uncommitted** (`git status`: `M app/ai_engine.py`, `M app/main.py`, `M templates/complaint_detail.html`; 55 insertions, 1 deletion). It loads `SF/wsp/inference.py` → `SentriFormerPipeline` (default checkpoint `wsp_runs/FINAL_T5a/sentriformer_t5a.pt`). | `git diff --stat`; `APP/app/ai_engine.py` L243-256; `SF/wsp/inference.py` L25 |
| F5 | **Verified:** the SentriFormer **sentiment** output replaces DistilBERT and feeds the rule-based priority score. SentriFormer **category, CSAT stars, draft reply and reply quality flags** are stored under `complaint["sentriformer"]` and shown **only on the admin complaint page**. The **draft reply is never auto-sent and is not pre-filled into the admin reply box**. The **urgency head output is ignored**. | `APP/app/ai_engine.py` L477-494, L547-556; `APP/app/main.py` L499; `APP/templates/complaint_detail.html` (SentriFormer block, ~L159-169; reply textarea L174) |
| F6 | SentriFormer "routing" is **display-only**. The stored `category` field is hard-coded to `"other"` at submission, and nothing routes on the SentriFormer category. | `APP/app/main.py` L341, L355 |
| F7 | **There is no system-level evaluation**: no tests, no latency measurements, no accuracy on SentriMail traffic, no user study. The project's own docs say so. The latency table in `research_comparative_evaluation.md` is **hard-coded in a plotting script**, not measured. | `APP/SENTRIMAIL_INTERVIEW_PREP.md` §14 (L929-945); `SF/generate_research_progression_plots.py` L279-284 |
| F8 | **Auto-reply logic mismatch.** The app auto-replies and emails *every* complaint whose final priority is LOW (`main.py` L365). It ignores the stricter `auto_resolvable` check in `ai_engine.py` (L347-364), which also blocks on refund, legal or fraud terms. When `auto_resolvable` is False, the text auto-sent is the *admin* template. | `APP/app/main.py` L365-387; `APP/app/ai_engine.py` L523-543 |
| F9 | The TF-IDF reply retriever is **effectively disabled**: `data/response_model.json` has 5 samples and 1 unique response, below the code's minimum (≥10 samples and >1 unique response). | `APP/app/ai_engine.py` L143-144; `APP/data/response_model.json` (`num_samples`=5, `unique_response_count`=1) |

---

## 1. Problem, user roles and workflow

### 1.1 Problem statement (as described by the project)
Organisations receive large volumes of unstructured customer complaints. SentriMail aims to triage them automatically (sentiment, emotion, priority), auto-answer simple low-priority cases, and give admins a prioritised inbox with AI-drafted replies, multilingual intake and analytics. source: `APP/README.md` L5-20 ("What This Project Does", "Key Capabilities"); top-level `README.md` §1.1-1.2 (L40-50).

### 1.2 Roles
| Role | How it is defined | Capabilities | source |
|---|---|---|---|
| `user` (customer) | `role: "user"`, assigned on self-registration | register, log in, submit a complaint (text, voice or audio upload, language select), view own complaints and details | `APP/app/auth.py` L79-92; `APP/app/main.py` L235-256, L312-400, L602-619 |
| `admin` | `role: "admin"`, seeded at startup | prioritised inbox and analytics, complaint detail with AI analysis, edit status, send final response, list users, CSV export, JSON APIs | `APP/app/auth.py` L24-51; `APP/app/main.py` L405-457, L459-599, L624-711 |
| anonymous | no session | public complaint tracking by complaint code; `POST /api/transcribe` (**no auth check**) | `APP/app/main.py` L258-268, L727-745 |

There is **no separate "agent" role**. Admins act as support agents. source: `APP/app/auth.py` (only the `admin` and `user` roles exist).

### 1.3 End-to-end workflow (as implemented)
1. **Intake (web form only).** `POST /user/submit` takes `title`, `description` and an optional `language`. Voice input uses the browser Web Speech API on the client side, or an uploaded audio file sent to `POST /api/transcribe` (Whisper `base`). source: `APP/app/main.py` L312-319, L727-745; `APP/templates/submit_complaint.html` (script: `SpeechRecognition` ~L80-117, `fetch('/api/transcribe')` ~L132).
2. **Language handling.** If no language is given, `langdetect.detect` is used. Non-English text is translated to English with `deep_translator.GoogleTranslator`. Note: the form's `<select>` defaults to `en` (11 options: en, hi, ta, te, kn, ml, bn, mr, es, fr, ar), so auto-detection only runs if the field is missing. source: `APP/app/main.py` L324-338; `APP/templates/submit_complaint.html` L34-45.
3. **AI analysis.** `analyze_complaint(translated_text, category="other", username=...)`. source: `APP/app/main.py` L341; pipeline in §3.
4. **Keyword escalation.** If the text contains any of 11 keywords (legal, police, harassment, …), priority is forced to CRITICAL and `keyword_escalated` is set to True. `priority_score` is not updated. source: `APP/app/main.py` L345-351.
5. **Branch on final priority.**
   - `LOW` → status `auto_replied`. The reply (`ai_suggested_response`) is translated back to the user's language, stored as `admin_response`, inserted into `replies` with `is_ai_reply=True`, and emailed in a background task. source: `APP/app/main.py` L365-387.
   - otherwise → status `pending_admin`. source: L374-375.
6. **Persistence.** `save_complaint` assigns a UUID `id` and a sequential `complaint_code` (`SENT-<year>-NNNN`). source: `APP/app/storage.py` L29-53.
7. **Time-based escalation.** An APScheduler job runs hourly: `pending_admin` MEDIUM complaints older than 24 h become HIGH, and HIGH older than 48 h become CRITICAL. source: `APP/app/main.py` L141-163.
8. **Admin review.** The inbox is sorted CRITICAL > HIGH > MEDIUM > LOW. The detail page re-runs `analyze_complaint` for every view (backfill), shows sentiment/emotion/priority meters, a root-cause sentence, the suggested reply (pre-filled into the reply box), and the SentriFormer panel when present. source: `APP/app/main.py` L405-509, L59-138; `APP/templates/complaint_detail.html`.
9. **Resolution.** `POST /admin/complaint/{id}/response` translates the admin text to the user's original language, emails it if status is `resolved`, and stores it. source: `APP/app/main.py` L523-548.
10. **Tracking.** A user or anonymous visitor can look up status by `complaint_code`. source: `APP/app/main.py` L258-268; `APP/templates/track.html`.
11. **CSAT.** There is **no customer CSAT survey or feedback capture**. CSAT appears only as a SentriFormer *prediction* (1-5 stars) on the admin page. source: grep of `APP/` (no CSAT field outside the `sentriformer` dict); `APP/app/ai_engine.py` L553.

---

## 2. Architecture

### 2.1 Stack
| Layer | Technology | source |
|---|---|---|
| Backend | FastAPI 0.111.0 on uvicorn 0.29.0 (`app = FastAPI(title="SentriMail", version="1.0.0")`), CORS `allow_origins=["*"]` | `APP/requirements.txt`; `APP/app/main.py` L41-49 |
| Templates | Jinja2 3.1.4; `static/` is mounted but **empty** | `APP/app/main.py` L52-56; `APP/static/` (empty dir) |
| Frontend JS | inline scripts; Chart.js 4.4.1 from the cdnjs CDN (4 charts: category donut, priority bar, daily line, sentiment-by-category stacked bar); Web Speech API; fetch to `/api/transcribe` and `/api/dashboard-stats` | `APP/templates/admin_dashboard.html` L174-250; `APP/templates/submit_complaint.html` |
| DB | MongoDB via pymongo 4.8.0. Falls back to local JSON files when ping fails. | `APP/app/mongodb.py` L26-31, L152-165 |
| Scheduler | APScheduler `BackgroundScheduler`, hourly | `APP/app/main.py` L156-163 |
| Speech | `openai-whisper` (local model, `base`, loaded **on every request**) | `APP/app/main.py` L724-745 |
| Translation | `langdetect` (local), `deep-translator` GoogleTranslator (calls the external Google Translate web service) | `APP/app/main.py` L10-11, L327-338, L283-288, L538-542 |
| Email | `smtplib` + `MIMEText`, outbound only | `APP/app/main.py` L271-309 |
| ML | `transformers` pipelines (HF Hub download on first load); optional SentriFormer (PyTorch) loaded via `importlib` | `APP/app/ai_engine.py` L209-256 |
| Runtime / deploy | `Procfile`: `web: python run.py`; `runtime.txt`: python-3.11.7; `run.py` reads `PORT` (default 8088); commit history mentions Railway | `APP/Procfile`; `APP/runtime.txt`; `APP/run.py` L38-58; commit `d53808d` "fix Railway deployment" |

**Not present (verified by reading every `APP/app/*.py` file and `requirements.txt`):** no OpenAI, Gemini or Groq API calls, no LLM API of any kind, no vector database, no embeddings, no RAG. The only retrieval is the TF-IDF nearest-neighbour lookup over `data/response_model.json` (`ai_engine.py` L127-176), which is disabled in practice (F9). The mermaid diagram labels match this set: FastAPI, Jinja2, MongoDB, DistilBERT, DistilRoBERTa, TF-IDF, SMTP, Translation (`Sentrimail/mermaid-diagram.svg`).

### 2.2 HTTP endpoints
| Method | Path | Auth | Purpose | source (`APP/app/main.py`) |
|---|---|---|---|---|
| GET | `/` | – | redirect by role | L168-180 |
| GET | `/login` | – | login page | L182-187 |
| POST | `/login` | – | authenticate, log to `login_logs`, set cookie | L190-211 |
| GET | `/register` | – | registration page | L212-214 |
| POST | `/register` | – | create a `user` account | L217-222 |
| GET | `/logout` | – | drop session | L225-229 |
| GET | `/user/dashboard` | user | own complaints and counts | L235-248 |
| GET | `/user/submit` | user | submission form | L251-256 |
| POST | `/user/submit` | user | intake → translate → analyse → escalate → save → auto-reply | L312-400 |
| GET | `/user/complaint/{complaint_id}` | user (owner) | complaint detail | L602-619 |
| GET | `/track` | public | tracking form | L258-260 |
| POST | `/track` | public | look up by `complaint_code` | L262-268 |
| GET | `/admin/dashboard` | admin | prioritised inbox and stats | L405-457 |
| GET | `/admin/complaint/{complaint_id}` | admin | detail with AI analysis (re-runs `analyze_complaint`) | L459-509 |
| POST | `/admin/complaint/{complaint_id}/status` | admin | change status | L511-520 |
| POST | `/admin/complaint/{complaint_id}/response` | admin | send final reply (translate and email if resolved) | L523-548 |
| GET | `/admin/users` | admin | user list (password field excluded) | L551-557 |
| GET | `/admin/export` | admin | CSV export (12 columns) | L560-599 |
| GET | `/api/complaints` | admin | all complaints (JSON) | L624-629 |
| GET | `/api/dashboard-stats` | admin | category/priority/daily/sentiment aggregates, average response hours | L632-711 |
| POST | `/api/analyze` | any logged-in user | run `analyze_complaint` on `{text}` | L714-721 |
| POST | `/api/transcribe` | **none** | Whisper transcription of an uploaded audio file | L727-745 |
| – | `/static` | – | static mount (empty) | L55 |

### 2.3 Data model
The database is MongoDB, named by `MONGODB_DB_NAME` (default `sentrimail`). The JSON fallback is `_LocalDB`, which writes `data/users.json`, `data/complaints.json` and `data/login_logs.json`. source: `APP/app/mongodb.py` L26-31, L131-135.

| Collection | Fields (as written by the code) | Indexes | source |
|---|---|---|---|
| `users` | `username`, `password` (unsalted SHA-256 hex; plaintext is migrated on login), `email`, `role` | `username` unique | `APP/app/auth.py` L20-92; `mongodb.py` L161 |
| `complaints` | `id` (uuid4), `complaint_code`, `title`, `category` (always `"other"` on new submissions), `description` (English/translated), `original_text`, `original_language`, `keyword_escalated`, `username`, `email`, `status` (`auto_replied` / `pending_admin` / `pending` / `resolved` / admin-set), `created_at`, `updated_at`, `admin_response`, plus the analysis fields `sentiment_label`, `sentiment_score`, `emotion_label`, `emotion_score`, `priority`, `priority_color`, `priority_score`, `priority_description`, `root_cause_summary`, `auto_resolvable`, `auto_resolution_reason`, `user_auto_response`, `admin_suggested_response`, `ai_suggested_response`, `model_used`, `response_source`, `reference_id`, `sentriformer` {`category`, `category_confidence`, `needs_human_triage`, `csat_stars`, `draft_reply`, `reply_flags`} | `username`, `status`, `created_at` | `APP/app/main.py` L353-375; `storage.py` L29-53; `ai_engine.py` L531-557; `mongodb.py` L162-164 |
| `replies` | `complaint_id`, `reply_text`, `is_ai_reply`, `replied_at`, `replied_by` (only AI auto-replies are inserted) | – | `APP/app/main.py` L380-386; `migrate.py` L54 |
| `login_logs` | `username`, `role`, `login_time` | (`username`, `login_time`) | `APP/app/main.py` L202-206; `mongodb.py` L165 |

`APP/migrate.py` backfills `original_text`, `translated_text`, `original_language`, `keyword_escalated`, `complaint_code`, `priority` and `status` on older documents. source: `APP/migrate.py` L9-55.

**JSON-fallback limitation (from code reading; not run, so UNVERIFIED at runtime).** `_LocalCursor` has no `.limit()`, but `save_complaint` calls `.limit(1)` (`storage.py` L34). `_LocalDB` has no `replies` collection, but `main.py` L380 writes to one. `_matches_filter` only does equality matching, so `$regex` and `$lt` queries do not work (`mongodb.py` L47-53; `main.py` L145-153). Submissions are therefore expected to fail when MongoDB is offline. The project's own notes record the same thing (`APP/SENTRIMAIL_INTERVIEW_PREP.md` ~L1060).

### 2.4 Auth and sessions
- An opaque `secrets.token_urlsafe(32)` token is kept in an **in-process dict `SESSIONS`**. The cookie `sentrimail_session` is httponly, samesite=lax, max-age 86400. Sessions are lost on restart and do not scale across processes. source: `APP/app/auth.py` L16-17, L95-118.
- Passwords are hashed with unsalted SHA-256. source: `auth.py` L20-21.
- Three demo accounts (one admin, two users) are seeded at startup with **hard-coded demo passwords**. These are also printed by `run.py` and shown on the login page screenshot. source: `auth.py` L24-51; `run.py` L48-49.
- There is no CSRF protection and no rate limiting. source: absence in `main.py` (UNVERIFIED that no middleware exists elsewhere; none found).

### 2.5 Environment variables (names only)
| Name | Used in | Purpose |
|---|---|---|
| `MONGODB_URI`, `MONGODB_DB_NAME` | `APP/app/mongodb.py` L26-27 | DB connection (the local `.env` points to a MongoDB Atlas `mongodb+srv` cluster **with embedded credentials**) |
| `MAIL_SERVER`, `MAIL_PORT`, `MAIL_USE_TLS`, `MAIL_USERNAME`, `MAIL_PASSWORD`, `MAIL_DEFAULT_SENDER` | `APP/app/main.py` L292-305; `APP/.env.example` | SMTP |
| `PORT` | `APP/run.py` L39 | listen port (default 8088; `main.py` `__main__` uses 8000) |
| `SENTRIFORMER_DIR` | `APP/app/ai_engine.py` L31 (uncommitted) | location of the SentriFormer repo (default `../../../sentriformer` relative to `app/`) |
| (historical) a Firebase service-account JSON env var | `app/firebase_config.py`, added in commit `f77e79b`, removed in `cefaa60` | Firestore (abandoned). History contains only the env var name, no key material (checked) |

`.env` is read manually by both `run.py` L22-32 and `mongodb.py` L13-24. `.env` is git-ignored and was never committed (`git log --all -- .env` is empty).

---

## 3. AI components exactly as implemented

All functions are in `APP/app/ai_engine.py` unless stated. Entry point: `analyze_complaint(text, category, username, complaint_id)` L469-557.

### 3.1 Component map
| Component | Implementation | Source of the signal | Lines |
|---|---|---|---|
| SentriFormer load | `_get_sentriformer()`: `importlib` loads `SF/wsp/inference.py`, instantiates `SentriFormerPipeline()` once, and returns None on any failure | SentriFormer T5a | L243-256 |
| SentriFormer call | `sf.analyze(text)`; exceptions are logged and the pipeline continues without it | SentriFormer | L477-483 |
| **Sentiment** | 1) SentriFormer `sentiment.label` upper-cased (NEGATIVE/NEUTRAL/POSITIVE), with `score = max(probs)`; 2) else HF `distilbert-base-uncased-finetuned-sst-2-english` (binary POSITIVE/NEGATIVE, first 512 chars); 3) else `_rule_based_sentiment` (keyword counts over 47 negative and 8 positive keywords) | SF → DistilBERT → rules | L485-494, L217-222, L259-269, L34-46 |
| **Emotion** | HF `j-hartmann/emotion-english-distilroberta-base`; else `_rule_based_emotion` (keyword patterns, 7 emotions). **Not from SentriFormer.** | DistilRoBERTa → rules | L496-503, L223-228, L272-282, L48-56 |
| **Priority** | `_compute_priority`: score = NEGATIVE ? int(40·sent_score) : NEUTRAL ? 10 : 0; + int(40·emo_score) for anger/fear/disgust, or int(20·emo_score) for sadness/surprise; +20 if any of 18 `URGENT_TRIGGERS` appears; +5 if > 100 words. Bands: ≥75 CRITICAL, ≥50 HIGH, ≥25 MEDIUM, else LOW | rules over (SF or DistilBERT sentiment) + emotion | L285-333, L68-72 |
| Keyword escalation | 11 keywords force CRITICAL (in `main.py`, after the analysis) | rules | `main.py` L345-351 |
| Time escalation | MEDIUM→HIGH after 24 h, HIGH→CRITICAL after 48 h while `pending_admin` | rules / scheduler | `main.py` L141-157 |
| **Category** | The user gives no category. The app passes and stores `"other"`. `_infer_issue_type` derives one of billing/auth/technical/delivery/support/general from keywords **only to pick a reply template**. The SentriFormer 33-class category is stored for display only. | rules; SF (display only) | `main.py` L341, L355; `ai_engine.py` L193-206 |
| Root cause | `_generate_root_cause`: fixed sentence per category plus an emotion suffix. Because the category is always `other`, this is effectively the "other" sentence. | template | L336-344, L58-66 |
| Auto-resolvable | `_is_auto_resolvable`: LOW, no urgent trigger, none of 8 hard blockers (refund, chargeback, legal, fraud, threat, injury, security breach, data leak), and not (NEGATIVE with score > 0.75) | rules | L347-364 |
| **Auto reply (user)** | if auto-resolvable: dataset retrieval, else `flan-t5-small` text2text with a support prompt (`do_sample=True, top_p=0.95, max_length=150`), else a fixed template | flan-T5 → template | L367-397, L523-529, L229-232 |
| **Admin suggested reply** | dataset retrieval (dropped if generic for HIGH/CRITICAL), else `_generate_admin_suggestion`: hand-written templates keyed by priority × inferred issue type | templates | L400-466, L508-522, L179-190 |
| Dataset retrieval | TF-IDF cosine nearest neighbour over `data/response_model.json` (+0.03 bonus for same category, +0.02 for same priority; rejected below 0.12). Built by `APP/dataset/new.py` (`model_type: tfidf-nearest-neighbor`). **Disabled by the size guard (F9).** | TF-IDF | L127-176; `APP/dataset/new.py` L120-169 |
| **CSAT** | SentriFormer `csat.stars` (argmax of 5 classes), display only | SentriFormer | L553; `SF/wsp/inference.py` L100 |
| **SentriFormer draft reply** | `sf_result["response"]` (greedy decoding, repetition penalty 1.2 for the T5a multi-turn checkpoint), display only, **never sent** | SentriFormer | L554; `SF/wsp/inference.py` L72-92 |
| Reply safety flags | from `SF/wsp/inference.py` L103-106: repetition, empty, template_placeholder, orphaned_channel_phrase, invented_url, invented_contact, unsupported_name (`eos_terminated` is excluded in SentriMail). Shown to the admin as "Flags". | SentriFormer post-check | `ai_engine.py` L555; `complaint_detail.html` SF block |
| Routing confidence gate | `needs_human_triage = top-1 prob < 0.25` → shown as "low confidence, check manually" | SentriFormer | `SF/wsp/inference.py` L28, L99 |
| Urgency | SentriFormer returns `urgency_experimental` (with a warning) | **ignored by SentriMail** | `SF/wsp/inference.py` L61-63; not referenced in `ai_engine.py`; rationale comment L547-548 → `SF/wsp_runs/t7u/T7U_DECISION.md` |
| Speech | Whisper `base` | OpenAI Whisper (local) | `main.py` L727-745 |
| Translation | GoogleTranslator via deep-translator, in both directions | external service | `main.py` L327-338, L368-372, L538-542, L283-288 |

### 3.2 Verification of the uncommitted SentriFormer wiring (`git diff`)
| Reported claim | Verdict | Evidence |
|---|---|---|
| sentiment → priority | **TRUE.** SF sentiment replaces DistilBERT and goes into `_compute_priority`. Emotion is still DistilRoBERTa. | diff hunk `@@ -453,7` → `ai_engine.py` L485-505 |
| routing category, CSAT, draft reply shown admin-only | **TRUE.** Stored in the `sentriformer` sub-dict and passed only to the admin `complaint_detail` template (`main.py` L499). `user_complaint_detail.html` and `track.html` do not reference it. | `ai_engine.py` L549-556; `main.py` L499; template diff |
| never auto-sent | **TRUE.** The auto-reply path uses `ai_suggested_response` (template/flan-T5/dataset), never `sentriformer.draft_reply`. The admin reply box is pre-filled from `admin_suggested_response`, not the SF draft. | `main.py` L367; `complaint_detail.html` L174 |
| urgency unused | **TRUE.** No reference to `urgency` in `ai_engine.py`. A comment explains the choice. | `ai_engine.py` L547-548 |
| Fallback when SF is unavailable | The existing pipeline runs unchanged and `sentriformer` is None. | `ai_engine.py` L246-256, L556 |

Caveats:
- (a) SF sentiment is only present if the checkpoint's auxiliary heads were trained (`aux_trained`, `SF/wsp/inference.py` L43, L57-59). For T5a this holds: the model card reports sentiment F1 0.756 (`SF/wsp_runs/FINAL_T5a/MODEL_CARD.md`).
- (b) `model_used` still reports `"transformer"` even when SF supplied the sentiment (`ai_engine.py` L544), so stored records do not say which model produced the sentiment.
- (c) `_merge_or_backfill_analysis` calls `analyze_complaint` (including SF generation) on **every admin detail view** but does not persist the `sentriformer` field. Complaints created before the integration never show the SF panel, and every page view pays the SF inference cost (`main.py` L59-138). The cost was not measured.
- (d) Evidence of execution: `app/__pycache__/ai_engine.cpython-314.pyc` is dated 2026-11-09 21:41, which suggests the edited module was imported at least once. **An end-to-end run with SF is UNVERIFIED.** No log or screenshot exists.

### 3.3 Model lineage: what SentriMail actually used
| Period | Sentiment | Emotion | Reply | Category | source |
|---|---|---|---|---|---|
| Initial commit `8ba79d2` (2026-02-24) → `cefaa60` (2026-03-09) | HF transformers / rules | HF / rules | templates | – | `git show --stat` (ai_engine.py 298 lines, rewritten +501/−… in `cefaa60`) |
| `cefaa60` | + TF-IDF reply model (`dataset/new.py`, `data/response_model.json`) | | | | commit `cefaa60` stat |
| `b7e8c45` (2026-04-24) "update the accuracy of the model and the ui" | flan-T5 auto-reply, multilingual, voice, track, CSV export (main.py +306) | | | | commit `b7e8c45` stat. The message says "accuracy", but no evaluation file was added (**UNVERIFIED claim**). |
| Last commit `5a8f80b` (2026-08-02) | DistilBERT-SST-2 / rules | DistilRoBERTa / rules | TF-IDF → flan-T5-small → templates | "other" | `git show HEAD:sentrimail/app/ai_engine.py` (= working tree minus the diff) |
| `model_training/` (Aug 29 – Sep 1, 2026, not in git) | T5-small encoder-decoder + 3 heads (sentiment 2-class, emotion 7, category 7) trained on pooled Bitext/CFPB/TWCS with DistilBERT pseudo-labels | | | | `model_training/model.py`, `config.py` (BACKBONE `t5-small`, 7 categories, threshold 0.65), `distill.py`. **Never integrated** (F3). Checkpoint `model_training/checkpoints/best_model.pt` (~726 MB). No saved eval results file; `config.py` comments cite billing recall 0.70 / precision 0.97 (**UNVERIFIED**, no artifact). |
| Uncommitted (Nov 2026) | **SentriFormer T5a** → DistilBERT → rules | DistilRoBERTa / rules | flan-T5/templates (sent); SF draft (admin-only) | SF 33-class (display) | §3.2 |

`model_training/response_safety.py` (URL / dollar-amount / agent-sign-off filter with template fallback) is **not used by SentriMail**. SentriMail's only reply safety checks are the SF quality flags (displayed, not enforced) and `_is_generic_dataset_response` (L179-190).

### 3.4 SentriFormer model facts relevant to the system (from the model's own artifacts, not SentriMail tests)
| Item | Value | source |
|---|---|---|
| Deployed checkpoint | `SF/wsp_runs/FINAL_T5a/sentriformer_t5a.pt` (46.7 MB). Rollback: `FINAL_T4d`. | `SF/wsp/inference.py` L25; `ls -la` |
| Categories | 33 (`num_classes` 33) | `SF/data/supervised/classification_labels.json` |
| Context lengths | routing 128 tokens; generation 128, or the last 256 tokens of the dialogue history for multi-turn checkpoints | `SF/wsp/inference.py` L27, L66-70 |
| T5a validation (model card) | category acc 73.4%, CSAT F1 0.330, sentiment F1 0.756; generic-phrase rate 17.2%/16.6% (vs T4d 26.8%/25.8%) | `SF/wsp_runs/FINAL_T5a/MODEL_CARD.md` |
| Regression tests | all pass: strict load (123 tensors), 20/20 deterministic, robust inputs (781-token, empty, unicode), originals untouched, 1998/2000 reproduce recorded val predictions | `SF/wsp_runs/FINAL_T5a/regression_tests.json` |
| Known limits | replies short and often generic or unhelpful; urgency only partly validated; escalation untrained; human A/B still pending | `SF/wsp_runs/FINAL_T5a/MODEL_CARD.md` "Known limits" |
| Urgency head | rejected (T7u): MAE worse than the constant-mean baseline | `SF/wsp_runs/t7u/T7U_DECISION.md` |
| Earlier milestone numbers (M9.x) | M9.2 33-class test acc 79.15% (123,229 test rows); M9.4 CSAT exact 72.07%, ±1 star 81.49%; M9.5 multitask cat 73.28% | `classification_results_report.md`, `csat_report.md`, `multitask_report.md` (top level). Their figure links point to a local IDE cache (`.gemini/antigravity-ide/brain/...`), so the figures are **not in the repo**. Copies may exist in `SF/results/` (UNVERIFIED for each one). |

These are **model-level offline** metrics on corpus splits. They are not measurements of SentriMail.

---

## 4. Measured evaluation of the system

**There is no system-level evaluation of SentriMail.**

| Evidence type | Status | source |
|---|---|---|
| Unit/integration tests | **None.** No `test*` files in `APP/`. | `find APP -iname '*test*'` (empty) |
| End-to-end latency (submit → analysed → stored) | **Not measured** | no script or log; `SENTRIMAIL_INTERVIEW_PREP.md` §14 lists "Latency benchmark: Not implemented" |
| Model latency in `research_comparative_evaluation.md` §6.1 (SentriFormer 18.2 ms / 2,811 samples/s; T5-Small 42.5 ms; …) | **Not measured. Hard-coded constants** in a plotting script, with no benchmark code found. **Do not cite as measured.** | `SF/generate_research_progression_plots.py` L279-284; grep for a latency benchmark in `SF/*.py` found none |
| Priority / sentiment / emotion accuracy on SentriMail data | **None.** No labelled SentriMail set exists. | `SENTRIMAIL_INTERVIEW_PREP.md` §14, weakness table ~L1060 |
| Auto-reply precision / admin acceptance rate / resolution time | **None.** `/api/dashboard-stats` computes `avg_response_hours` live, but no report was saved. The screenshot shows 0.4 h on 4 demo complaints. | `main.py` L698; `working_images/admin_dashboard or inbox or admin_analytics_dashboard.png` |
| User study / usability | **None** | – |
| SF integration evidence | code only, plus a `.pyc` timestamp. No screenshot of the SF panel. | §3.2(d) |
| Live data in repo | `data/complaints.json`: 5 demo records (all LOW, resolved, `model_used=rule-based`, category 4×other + 1×technical) | `APP/data/complaints.json` (aggregates only) |

**What a paper would need (missing):**
1. A labelled SentriMail-style test set (priority gold labels) with a confusion matrix and critical-class recall, comparing the rule baseline with DistilBERT, with SentriFormer sentiment, and with the full pipeline.
2. Measured per-stage latency: SF load time, SF analyse with and without generation, Whisper, translation, and end-to-end on CPU and GPU.
3. Auto-reply safety audit (how many LOW auto-replies contained blocker terms, given F8).
4. Admin acceptance or edit distance of suggested replies, and an SF-draft vs template human rating.
5. Routing evaluation of the SF category against the store's category taxonomy (currently not connected).
6. Screenshot of the SF admin panel.

---

## 5. Development history (git)

Repo: `SENTRIMAIL-MPW1/Sentrimail/.git`, remote `github.com/Navin-665/SentriMail-MPW1`. 24 commits, 2026-02-24 → 2026-08-02. Authors: Lohith / Lohith-07-coder (20), Navin (4). source: `git log`.

| Date | Commit | Author | Change |
|---|---|---|---|
| 2026-02-24 | `8ba79d2` | Lohith-07-coder | Initial commit: FastAPI app, ai_engine (298 lines), auth, storage, JSON data, 10 templates |
| 2026-02-24 | `d53808d` … `82917ea` (6 commits) | Lohith-07-coder | `.gitignore`, Procfile, Railway deployment fixes, runtime pin, "production ready deployment", storage rewrite |
| 2026-03-04 | `f77e79b`, `fd820bd`, `1f266fd`, `8760ba8` | Lohith-07-coder | Firebase/Firestore storage and login logging (later removed) |
| 2026-03-05 | `0729199`, `5ce7846`, `77fe4ce`, `b2cb211` | Lohith / Lohith-07-coder | dashboard sort/normalisation, UI |
| 2026-03-09 | `cefaa60` | Lohith-07-coder | ai_engine rewrite (+501), Firebase removed, MongoDB module, TF-IDF reply model (`dataset/new.py`, `data/response_model.json`) |
| 2026-03-10 | `26b563a` | Lohith-07-coder | MongoDB connection and local JSON fallback (mongodb.py +135) |
| 2026-03-25 | `575a41f` | Lohith | removed `.zencoder` folder |
| 2026-04-24 | `b7e8c45` | Lohith-07-coder | main.py +306: multilingual, voice/Whisper, track, CSV export, escalation; `migrate.py`; flan-T5 auto-reply |
| 2026-07-19 | `7216554` | Navin | fix `auto_resolvable` NameError, port 8088, `.env` loader |
| 2026-07-19 | `cccf95b` | Navin | fix session leak on logout |
| 2026-08-02 | `e64bd77` | Navin | README with screenshots (`working_images/`) |
| 2026-08-02 | `5a8f80b` | Navin | top-level README |
| (uncommitted, ~2026-11-09) | – | – | SentriFormer wiring (3 files, +55/−1); untracked `SENTRIMAIL_INTERVIEW_PREP.md`, `../mermaid-diagram.svg` |

The `model_training/` and `sentriformer/` directories are **not** in this git repo. source: `git ls-files` lists only `README.md` and `sentrimail/*`.

---

## 6. Screenshots and figures

All screenshots are in `APP/working_images/`, committed in `e64bd77` (2026-08-02). They show the **pre-SentriFormer** app with demo data dated 2026-07-19. None shows the SentriFormer panel.

| File | Size | Shows |
|---|---|---|
| `login page_dashboard.png` | 276 KB | Login page; feature panel ("DistilBERT & RoBERTa", priority scoring, root cause, RBAC); demo-account tiles (**contain demo passwords, crop before publishing**) |
| `user_dashboard.png` | 63 KB | User "My Complaints": totals (4 submitted / 0 pending / 2 resolved), table with Auto Replied / Resolved / Pending Admin statuses |
| `user_complaint_raise.png` | 37 KB | Submit form: title, language select, description, mic and upload buttons |
| `complaint_in_multiple_lanuages.png` | 24 KB | Language dropdown (11 languages) |
| `complaint_through_voice.png` | 25 KB | Submit form with the mic recording active (voice intake) |
| `admin_dashboard or inbox or admin_analytics_dashboard.png` | 64 KB | Admin Inbox KPIs (4 total, 1 pending, 0 resolved today, 0.4 h average response) and 4 Chart.js charts |
| `admin_dashboard or admin_analytics_dashboard_2.png` | 86 KB | Sentiment-by-category chart, priority filters, Export CSV, complaint table (ID, title, lang, priority, status, date) |
| `user_management.png` | 34 KB | Admin user list (3 seeded demo accounts; example/demo addresses only) |

Other figures:
- `Sentrimail/mermaid-diagram.svg` (33 KB, untracked): architecture diagram (User/Admin → FastAPI/Jinja2 → Language/Translation → DistilBERT/DistilRoBERTa → Priority → TF-IDF → MongoDB → SMTP). It shows the pre-SentriFormer design.
- Mermaid blocks: `APP/README.md` L22-50; top-level `README.md` §8 L326-361. The latter draws a **SentriFormer-centric** architecture (urgency, automated draft) that **overstates the implementation**: urgency is unused, the draft is not sent, and the category is not routed.
- `sentriformer_project_workflow.excalidraw` (top level, 20 KB): SentriFormer project workflow (not opened in detail).
- Filenames contain spaces and the word "or". Rename them for the clean copy.

---

## 7. Cleanup notes for a clean public GitHub copy

### 7.1 Junk / do not copy
| Path | Size | Note |
|---|---|---|
| `Sentrimail/venv/` | ~1.4 GB | virtualenv (outside `sentrimail/`, untracked) |
| `APP/app/__pycache__/` | ~186 KB, 11 `.pyc` | **5 `cpython-313.pyc` files are tracked in git** despite `.gitignore`; remove them |
| `APP/package-lock.json` | 95 B | empty npm lock, no JS project |
| `APP/static/` | empty | mounted by the app; keep with a `.gitkeep` or let the app create it (`main.py` L54 does `makedirs`) |
| `model_training/.venv/`, `model_training/__pycache__/` | large, not measured | venv and caches |
| `model_training/checkpoints/*.pt` | ~1.45 GB (2 × ~726 MB) | T5-small checkpoints; never deploy; too large for GitHub |
| `Sentrimail/.git` | ~888 KB on disk (packed history ~143 KB) | start a fresh history, or keep it but scrub (see 7.2) |

### 7.2 Secrets and personal data: exclude or scrub
| Path | Issue |
|---|---|
| `APP/.env` (254 B) | **Real MongoDB Atlas `mongodb+srv` URI with embedded username and password.** Never copy it. Recommend **rotating** that Atlas credential. Untracked and never committed (verified). |
| `APP/.env.example` | safe: all credential values are empty; only the port and TLS flag have values (verified by checking value lengths) |
| `APP/app/auth.py` L24-51, `APP/run.py` L48-49 | hard-coded demo account passwords (demo only). Acceptable for a demo, but document them or move them to env vars. |
| `APP/data/users.json` (887 B, **tracked**) | 5 users with SHA-256 hashes. 2 non-demo accounts use what look like **real personal email addresses** (one at a gmail domain). **Exclude, or replace with demo-only data.** |
| `APP/data/complaints.json` (6.9 KB, **tracked**) | 5 complaint records with usernames and emails; likely test text by the developers. Treat as personal data and replace with synthetic data. |
| `APP/data/response_model.json` (3.3 KB, tracked) | TF-IDF model built from `complaints.json`; embeds the same complaint text. Regenerate from synthetic data or drop it. |
| git history | contains the files above. The Firebase commits contain only an env var name, no key (checked). Commit author emails are in history. |
| `APP/working_images/login page_dashboard.png` | shows demo passwords (low risk, already public in the README) |
| `rawdata/` (`twcs.csv`, `complaints.csv` = CFPB, `archive (1).zip`, …) | ~9.1 GB. **Third-party corpus. Do not publish.** Link to the sources instead (`rawdata/DATA_SOURCES.md`). |
| `model_training/data/` (`complaints.csv` ~9.2 GB CFPB, `twcs.csv` ~517 MB, `processed/*.parquet`) | ~9.4 GB. Corpus and derived data. Do not publish. Already git-ignored by `model_training/.gitignore`. |
| `SF/corpus`, `SF/data`, `SF/wsp_runs/**/LOCAL_ONLY*` | corpus-derived. Agent 1 / SentriFormer scope; do not publish raw rows. |

### 7.3 Suggested clean layout
`app/`, `templates/`, `dataset/new.py`, `run.py`, `migrate.py`, `requirements.txt` (add `torch` if SF is enabled; SF also needs its own repo or a `SENTRIFORMER_DIR` path), `Procfile`, `runtime.txt`, `.env.example`, `.gitignore`, `README.md`, `docs/screenshots/` (renamed PNGs, login image cropped), `docs/architecture.svg`. Leave out `data/*.json`, or ship a synthetic seed. Mark `SENTRIMAIL_INTERVIEW_PREP.md` as optional (it is a 51 KB self-critique that is useful as a source of limitations).

---

## 8. Known defects relevant to claims (from code reading)
| Defect | source |
|---|---|
| Auto-reply sent for all LOW complaints, bypassing `auto_resolvable` blockers (F8) | `main.py` L365-387 vs `ai_engine.py` L347-364 |
| Category always `"other"`, so the category charts and root-cause text are degenerate; the screenshots show only OTHER | `main.py` L341, L355; screenshots |
| Dashboard "pending" counts `status=="pending"`, but new complaints use `pending_admin` | `main.py` L419 vs L375 |
| Keyword escalation changes `priority` but not `priority_score` or `priority_description` | `main.py` L349-351 |
| Whisper model loaded on every request; endpoint unauthenticated | `main.py` L727-745 |
| SMTP login only when `MAIL_USE_TLS == "True"`; failures are only printed | `main.py` L300-309 |
| Admin detail view re-runs the full analysis (including SF generation) on every page load | `main.py` L69-74, L471 |
| JSON fallback cannot save complaints (`limit`, `replies`, `$regex`) | `storage.py` L34; `mongodb.py` L56-135; `main.py` L380 |
| In-memory sessions, unsalted SHA-256, CORS `*`, no CSRF | `auth.py`; `main.py` L43-49 |
| `analyze_complaint` docstring and `model_used` do not reflect SF | `ai_engine.py` L1-9, L544 |
