# Literature review and verified references

Prepared 2026-11-23 for two IEEEtran conference papers:

- **Paper 1, SentriFormer:** an 11.65M-parameter encoder-decoder Transformer trained from scratch on TWCS and CFPB data. It has multi-task heads for 33-class routing, sentiment, CSAT and urgency (the urgency head failed), plus response generation.
- **Paper 2, SentriMail:** a complaint-management web system (Python backend, MongoDB, admin dashboard) that uses SentriFormer for routing, sentiment-to-priority, CSAT prediction and admin-reviewed draft replies. Drafts are never sent automatically.

The bib files are `../SentriFormer_Paper/references.bib` (43 entries) and `../SentriMail_Paper/references.bib` (32 entries). Every entry was checked against a primary or bibliographic source; the verification table is at the end. Where sources disagreed on pages or title, the published record (Crossref, ACL Anthology, publisher) was used.

---

## Paper 1: SentriFormer

### 1. Transformer and encoder-decoder foundations

- **vaswani2017attention**: introduced the Transformer, built on attention alone. SentriFormer's backbone is a scaled-down version of this encoder-decoder.
- **sutskever2014sequence**: LSTM encoder-decoder (seq2seq) for MT. It is the origin of the encoder-decoder framing used for response generation.
- **devlin2019bert**: bidirectional masked-LM pre-training for an encoder with classification heads. It is the standard pre-train-then-fine-tune baseline that we deliberately do *not* use: we train from scratch.
- **raffel2020exploring** (T5): casts every task as text-to-text and studies objectives and scale. It shows the alternative design of putting classification inside the decoder, where we use separate task heads.
- **lewis2020bart**: denoising seq2seq pre-training that does well on generation. It is the closest pre-trained analogue to our encoder-decoder and a natural comparison point for the response quality we report.
- **xiong2020layer**: shows that Pre-LN Transformers have well-behaved gradients at initialization and can train without a long warm-up. This supports our Pre-LN choice for stable from-scratch training.

### 2. Optimisation, regularisation, class imbalance

- **kingma2015adam / loshchilov2019decoupled**: Adam, and AdamW with decoupled weight decay. AdamW is our optimiser.
- **szegedy2016rethinking**: introduced label smoothing as a regulariser.
- **muller2019labelsmoothing**: shows label smoothing improves calibration but erases information in the logits and hurts distillation. Cite it when you discuss our label-smoothing pitfalls, e.g. side effects on calibrated CSAT/sentiment scores or on minority classes.
- **lin2017focal**: focal loss down-weights easy examples.
- **cui2019classbalanced**: re-weights the loss by the "effective number" of samples per class. Together with focal loss, these are the standard fixes for the long-tailed 33-class routing distribution; say which ones we tried or rejected.

### 3. Tokenisation

- **sennrich2016neural**: BPE subword units for open-vocabulary NMT.
- **kudo2018sentencepiece**: a language-independent subword tokenizer trained on raw text (BPE or unigram). It is our tokenizer. Noisy tweet and complaint text (handles, URLs, emoji) is why raw-text tokenization matters here.

### 4. Multi-task learning

- **caruana1997multitask**: the founding MTL paper. Shared representations act as an inductive bias across related tasks.
- **ruder2017overview**: survey of hard and soft parameter sharing in deep MTL. Our design is hard sharing, with one encoder and several heads.
- **kendall2018multitask**: weights task losses by homoscedastic uncertainty. Relevant to how we weight the routing, sentiment, CSAT, urgency and generation losses. It also gives one possible explanation for the failed urgency regression head: regression and classification losses have different scales.
- **zhang2022survey**: TKDE survey of MTL. A recent general reference for negative transfer between tasks.

### 5. Neural dialogue and the generic-response problem

- **vinyals2015neural**: seq2seq conversational model trained on an IT helpdesk corpus and on movie subtitles. It is an early customer-support-like setting and showed the bland, inconsistent replies typical of MLE training.
- **zhang2020dialogpt**: GPT-2 pre-trained on 147M Reddit exchanges for response generation. A large pre-trained contrast to our small from-scratch model; its authors also apply MMI re-ranking.
- **li2016diversity**: diagnoses the "I don't know" generic-response problem and proposes MMI objectives and decoding. This is our MMI decoding reference.
- **holtzman2020curious**: shows that likelihood-maximising decoding degenerates and proposes nucleus (top-p) sampling. It is the decoding baseline.
- **welleck2020neural**: unlikelihood training penalises repetition and frequent tokens. It is a training-time alternative to our data-filtering approach.
- **su2022contrastive**: SimCTG plus contrastive search, a recent (2022) decoding fix for degeneration.
- **csaky2019improving**: removes high-entropy (generic) utterances from training data and gets more diverse responses. This is the closest prior work to our data-filtering experiments; differentiate on domain (customer support, not open-domain) and on our pre-registered acceptance criteria.
- **akama2020filtering**: scores utterance pairs by connectivity and content relatedness and filters noisy dialogue corpora. A second data-filtering reference; contrast it with our PMI-style filter, which was rejected under our frozen criteria.
- **tevet2021evaluating**: shows that n-gram diversity metrics mostly track surface form, not content diversity. Use it to justify caution in interpreting distinct-n gains.

### 6. Small and efficient Transformers

- **sanh2019distilbert, jiao2020tinybert, sun2020mobilebert**: compress BERT by distillation to about 40% to 4x smaller. They show that compact models are viable, but all of them inherit knowledge from a large teacher; SentriFormer has no teacher and no pre-training.
- **geiping2023cramming**: trains a BERT-like model from scratch on one GPU in one day. It is the closest parallel to our compute-bounded, from-scratch setup.
- **eldan2023tinystories**: models under 10M parameters can produce fluent text when the data distribution is narrow. This supports our claim that a ~12M model can be useful in a narrow domain such as customer support.

### 7. Evaluation methodology

- **efron1979bootstrap**: the bootstrap; the basis of our confidence intervals.
- **koehn2004statistical**: paired bootstrap resampling for MT metrics. It is the standard way to put CIs on BLEU-style generation scores.
- **dror2018hitchhiker**: a guide to significance testing in NLP, which found widespread misuse in ACL/TACL papers. It motivates our explicit CIs.
- **vanmiltenburg2021preregistering**: argues for preregistering NLP studies. It directly supports our frozen criteria, test-once protocol and honest reporting of negative results (the failed urgency head, the rejected T6 filter).

### 8. Customer-support data and complaints

- **xu2017new**: IBM LSTM seq2seq chatbot trained on a large corpus of brand-customer Twitter exchanges (check the exact count in the paper before quoting it), judged comparable to human agents on emotional requests. It is the closest prior generation work on Twitter customer service.
- **hardalov2018towards**: compares retrieval, seq2seq and Transformer models on the same Kaggle Twitter Customer Support corpus that we use; seq2seq scored best. A direct baseline reference for TWCS.
- **preotiucpietro2019automatically**: Twitter complaint dataset across nine domains with feature-based and neural classifiers (about 79 F1). It covers complaint *identification*, whereas we do 33-way *routing*.
- **jin2020complaint**: Transformer models for complaint identification. Shows that pre-trained encoders dominate the task, in contrast to our from-scratch model.
- **oyewola2023consumer**: CFPB complaint classification with a residual 1D-CNN, compared against DistilBERT and Bi-LSTM on about 556k complaints. It is a peer-reviewed CFPB baseline for our routing head.
- **twcs2017dataset / cfpbcomplaints**: our two data sources. The CFPB database notes that it is not a statistically representative sample; mention this under limitations.

---

## Paper 2: SentriMail

### 1. Models used by the system

- **vaswani2017attention, devlin2019bert**: background for the Transformer models behind routing and sentiment.
- **lewis2020rag**: retrieval-augmented generation grounds outputs in retrieved documents. It is a future-work direction for grounding draft replies in KB articles or policy text.
- **ji2023survey**: survey of hallucination in NLG, including dialogue. It is the main justification for never auto-sending generated replies.

### 2. Customer-service chatbots and agent assistance

- **xu2017new, hardalov2018towards**: generative customer-service reply models on Twitter. They generate end-to-end replies; SentriMail places generation behind admin review.
- **cui2017superagent**: e-commerce customer-service chatbot that combines FAQ retrieval, product QA and chit-chat. A fully automated contrast to our human-in-the-loop design.
- **bocklisch2017rasa**: open-source NLU and dialogue-management framework. It is the common engineering baseline for building support bots.
- **adam2021ai**: experiment showing that anthropomorphic chatbot cues increase user compliance in customer service. It covers the user side of automated service.
- **kannan2016smart**: Gmail Smart Reply suggests short replies that a human picks from and never sends automatically. It is the best-known precedent for SentriMail's draft-and-approve pattern.
- **howell2023economic**: ACL industry case study of LLM-drafted replies for human agents. It reports large agent time savings and finds that a small distilled model can be competitive. This supports using a compact model over an API LLM.
- **zhao2025agent**: EMNLP 2025 industry paper in which agent feedback on LLM suggestions continuously improves the support system. Related to our admin edits; future work could log them as training signal.
- **brynjolfsson2025generative**: field study of 5,172 support agents showing about 15% productivity gain from an AI assistant, largest for novices. It is economic evidence for assistive, not autonomous, AI in support.

### 3. Ticket classification and routing

- **molino2018cota**: Uber's ticket-type classification and reply-template ranking, which cut resolution time by about 10% in an A/B test. It is the closest industrial analogue to our routing plus draft pipeline.
- **mandal2018cognitive**: IBM helpdesk email ticket assignment with an ensemble and rule engine; low-confidence tickets go to a human queue. It is a precedent for confidence-gated routing.
- **revina2020it**: finds that simple models with linguistic features match more complex ones for IT ticket classification. Useful framing for model-size trade-offs.
- **zangari2023ticket**: review of ticket automation plus hierarchical multi-level classification (ML-BERT). Relevant to a hierarchical version of our 33-category taxonomy.
- **truss2024ai**: AutoML (Vertex AI) ticket classifier on real Jira Service Desk tickets. A recent low-code alternative to a custom model.

### 4. Complaints

- **preotiucpietro2019automatically, jin2021modeling**: complaint identification, and complaint severity on Twitter. Severity relates directly to our sentiment-to-priority mapping.
- **oyewola2023consumer**: CFPB product classification. A domain baseline for our routing categories.

### 5. Sentiment and customer satisfaction

- **zhang2018deep, wankhade2022survey**: surveys of deep-learning and broader sentiment-analysis methods. Background for the sentiment head and the priority mapping.
- **oraby2017how**: fine-grained dialogue acts in Twitter customer-service threads, used to predict satisfaction, frustration and problem resolution. It is the closest prior work on CSAT prediction from support conversations.
- **park2018positivity**: Samsung live-chat study showing that customers who leave ratings are more positive than those who do not, with an RNN used to infer satisfaction. This is a key caveat: CSAT labels are biased, so predicted CSAT is a triage signal, not ground truth.
- **sun2021simulating**: USS dataset of 6,800 dialogues with turn- and dialogue-level satisfaction labels for task-oriented systems. A recent satisfaction-modelling reference.
- **singh2022linguistic**: empathy and other linguistic features of brand replies that predict customer engagement on Twitter. Relevant to how draft replies should be judged by admins.

### 6. Human-in-the-loop and human-AI interaction

- **amershi2019guidelines**: 18 validated guidelines for human-AI interaction, such as making clear what the system can do, supporting efficient correction and dismissal, and showing confidence. Use them to map SentriMail's UI features (confidence display, edit-before-send, override routing).
- **wu2022survey, mosqueirarey2023human**: surveys of human-in-the-loop ML, covering active learning, interactive ML and machine teaching. They frame the admin-review loop academically.

### 7. Datasets

- **twcs2017dataset, cfpbcomplaints**: the same data sources as Paper 1.

---

## How to differentiate our work (talking points)

1. **From-scratch and compact.** Most customer-support NLP fine-tunes large pre-trained models (BERT, DialoGPT, LLMs). SentriFormer is about 11.65M parameters with no teacher and no pre-training. The nearest relatives are Cramming and TinyStories, neither of which is in this domain.
2. **One model, five outputs.** Prior work does routing (COTA, Mandal), complaint detection (Preoţiuc-Pietro, Jin) and CSAT (Oraby, Park) separately. We combine them under hard-parameter-sharing MTL and report the negative result for urgency honestly.
3. **Data filtering for generic responses in customer support.** Csáky and Akama filter open-domain corpora; we test filtering on TWCS under frozen criteria and report rejection (T6).
4. **Pre-registered evaluation with bootstrap CIs**, following van Miltenburg, Dror and Koehn. This is rare in applied customer-support papers.
5. **SentriMail is assistive, not autonomous**, following Smart Reply, Amershi, Brynjolfsson and Howell. It is backed by the hallucination literature (Ji) and by the CSAT-bias evidence (Park).

---

## Verification table

"How verified": **AA** means ACL Anthology record; **CR** means Crossref API lookup of the DOI on 2026-11-23, with title, pages and volume matched; **PUB** means publisher or proceedings page (NeurIPS, PMLR, JMLR, CVF, AAAI); **AX** means arXiv listing; **WS** means web search with cross-checks from 2 or more bibliographic records.

| Key | Title (short) | Venue | Year | DOI / URL | How verified |
|---|---|---|---|---|---|
| vaswani2017attention | Attention Is All You Need | NIPS 30, pp. 5998-6008 | 2017 | proceedings.neurips.cc/.../3f5ee243...-Abstract.html | PUB metadata (page_first/last) |
| sutskever2014sequence | Sequence to Sequence Learning with NNs | NIPS 27, pp. 3104-3112 | 2014 | proceedings.neurips.cc/.../5a18e133...-Abstract.html | PUB metadata |
| devlin2019bert | BERT | NAACL-HLT, pp. 4171-4186 | 2019 | 10.18653/v1/N19-1423 | AA |
| raffel2020exploring | Exploring the Limits of Transfer Learning (T5) | JMLR 21(140):1-67 | 2020 | jmlr.org/papers/v21/20-074.html | PUB |
| lewis2020bart | BART | ACL, pp. 7871-7880 | 2020 | 10.18653/v1/2020.acl-main.703 | AA |
| xiong2020layer | On Layer Normalization in the Transformer Architecture | ICML, PMLR 119:10524-10533 | 2020 | proceedings.mlr.press/v119/xiong20b.html | PUB (author meta tags) |
| kingma2015adam | Adam | ICLR | 2015 | arxiv.org/abs/1412.6980 | AX |
| loshchilov2019decoupled | Decoupled Weight Decay Regularization | ICLR | 2019 | arxiv.org/abs/1711.05101 | AX + ICLR poster page |
| szegedy2016rethinking | Rethinking the Inception Architecture | CVPR, pp. 2818-2826 | 2016 | 10.1109/CVPR.2016.308 | CR |
| muller2019labelsmoothing | When Does Label Smoothing Help? | NeurIPS 32 | 2019 | proceedings.neurips.cc/.../f1748d6b...-Abstract.html | PUB (page numbers not shown, so omitted) |
| lin2017focal | Focal Loss for Dense Object Detection | ICCV, pp. 2999-3007 | 2017 | 10.1109/ICCV.2017.324 | CR (IEEE pages; CVF lists 2980-2988) |
| cui2019classbalanced | Class-Balanced Loss | CVPR, pp. 9260-9269 | 2019 | 10.1109/CVPR.2019.00949 | CR (IEEE pages; CVF lists 9268-9277) |
| sennrich2016neural | NMT of Rare Words with Subword Units | ACL, pp. 1715-1725 | 2016 | 10.18653/v1/P16-1162 | AA |
| kudo2018sentencepiece | SentencePiece | EMNLP Demos, pp. 66-71 | 2018 | 10.18653/v1/D18-2012 | AA |
| caruana1997multitask | Multitask Learning | Machine Learning 28(1):41-75 | 1997 | 10.1023/A:1007379606734 | CR |
| ruder2017overview | Overview of MTL in Deep NNs | arXiv | 2017 | arxiv.org/abs/1706.05098 | AX |
| kendall2018multitask | MTL Using Uncertainty to Weigh Losses | CVPR, pp. 7482-7491 | 2018 | 10.1109/CVPR.2018.00781 | CR |
| zhang2022survey | A Survey on Multi-Task Learning | IEEE TKDE 34(12):5586-5609 | 2022 | 10.1109/TKDE.2021.3070203 | CR |
| vinyals2015neural | A Neural Conversational Model | ICML DL Workshop / arXiv | 2015 | arxiv.org/abs/1506.05869 | AX |
| li2016diversity | Diversity-Promoting Objective (MMI) | NAACL-HLT, pp. 110-119 | 2016 | 10.18653/v1/N16-1014 | AA |
| holtzman2020curious | Curious Case of Neural Text Degeneration | ICLR | 2020 | arxiv.org/abs/1904.09751 | AX + OpenReview |
| welleck2020neural | Unlikelihood Training | ICLR | 2020 | arxiv.org/abs/1908.04319 | AX + ICLR page |
| su2022contrastive | Contrastive Framework for Neural Text Generation | NeurIPS 35, pp. 21548-21561 | 2022 | proceedings.neurips.cc/.../871cae8f...-Abstract.html | PUB meta tags |
| csaky2019improving | Entropy-Based Data Filtering | ACL, pp. 5650-5669 | 2019 | 10.18653/v1/P19-1567 | AA |
| akama2020filtering | Filtering Noisy Dialogue Corpora | EMNLP, pp. 941-958 | 2020 | 10.18653/v1/2020.emnlp-main.68 | AA |
| tevet2021evaluating | Evaluating the Evaluation of Diversity | EACL, pp. 326-346 | 2021 | 10.18653/v1/2021.eacl-main.25 | AA + CR |
| zhang2020dialogpt | DialoGPT | ACL Demos, pp. 270-278 | 2020 | 10.18653/v1/2020.acl-demos.30 | AA + CR |
| sanh2019distilbert | DistilBERT | EMC2 workshop @ NeurIPS / arXiv | 2019 | arxiv.org/abs/1910.01108 | AX |
| jiao2020tinybert | TinyBERT | Findings of EMNLP, pp. 4163-4174 | 2020 | 10.18653/v1/2020.findings-emnlp.372 | AA |
| sun2020mobilebert | MobileBERT | ACL, pp. 2158-2170 | 2020 | 10.18653/v1/2020.acl-main.195 | AA |
| geiping2023cramming | Cramming | ICML, PMLR 202:11117-11143 | 2023 | proceedings.mlr.press/v202/geiping23a.html | PUB |
| eldan2023tinystories | TinyStories | arXiv | 2023 | arxiv.org/abs/2305.07759 | AX |
| efron1979bootstrap | Bootstrap Methods | Ann. Statist. 7(1):1-26 | 1979 | 10.1214/aos/1176344552 | CR + Project Euclid |
| koehn2004statistical | Statistical Significance Tests for MT Eval | EMNLP, pp. 388-395 | 2004 | aclanthology.org/W04-3250 | AA |
| dror2018hitchhiker | Hitchhiker's Guide to Significance Testing | ACL, pp. 1383-1392 | 2018 | 10.18653/v1/P18-1128 | AA + CR |
| vanmiltenburg2021preregistering | Preregistering NLP Research | NAACL-HLT, pp. 613-623 | 2021 | 10.18653/v1/2021.naacl-main.51 | AA + CR |
| xu2017new | A New Chatbot for Customer Service on Social Media | CHI, pp. 3506-3510 | 2017 | 10.1145/3025453.3025496 | CR |
| hardalov2018towards | Towards Automated Customer Support | AIMSA, LNCS 11089, pp. 48-59 | 2018 | 10.1007/978-3-319-99344-7_5 | CR + AX |
| preotiucpietro2019automatically | Automatically Identifying Complaints in Social Media | ACL, pp. 5008-5019 | 2019 | 10.18653/v1/P19-1495 | AA |
| jin2020complaint | Complaint Identification with Transformer Networks | COLING, pp. 1765-1771 | 2020 | 10.18653/v1/2020.coling-main.157 | AA + CR |
| jin2021modeling | Modeling the Severity of Complaints | NAACL-HLT, pp. 2264-2274 | 2021 | 10.18653/v1/2021.naacl-main.180 | CR (authors + pages) |
| oyewola2023consumer | CFPB Complaints via TSR1DCNN | Data & Inf. Manag. 7(4):100046 | 2023 | 10.1016/j.dim.2023.100046 | CR (bibliographic query + DOI) |
| twcs2017dataset | Customer Support on Twitter | Kaggle dataset | 2017* | kaggle.com/datasets/thoughtvector/customer-support-on-twitter | WS (*upload year not confirmed; data is from 2017 and the set was used by Hardalov in Sep 2018) |
| cfpbcomplaints | Consumer Complaint Database | CFPB website | n/a | consumerfinance.gov/data-research/consumer-complaints/ | WS (official page) |
| lewis2020rag | Retrieval-Augmented Generation | NeurIPS 33, pp. 9459-9474 | 2020 | proceedings.neurips.cc/.../6b493230...-Abstract.html | PUB meta tags |
| ji2023survey | Survey of Hallucination in NLG | ACM CSUR 55(12):1-38 | 2023 | 10.1145/3571730 | CR (full author list) |
| cui2017superagent | SuperAgent | ACL Demos, pp. 97-102 | 2017 | 10.18653/v1/P17-4017 | AA + CR |
| bocklisch2017rasa | Rasa | NIPS ConvAI workshop / arXiv | 2017 | arxiv.org/abs/1712.05181 | AX |
| adam2021ai | AI-based Chatbots in Customer Service | Electronic Markets 31(2):427-445 | 2021 | 10.1007/s12525-020-00414-7 | CR (online-first 2020) |
| kannan2016smart | Smart Reply | KDD, pp. 955-964 | 2016 | 10.1145/2939672.2939801 | CR + DBLP |
| howell2023economic | Economic Trade-offs of LLMs: A Case Study | ACL Industry, pp. 248-267 | 2023 | 10.18653/v1/2023.acl-industry.24 | AA (Crossref holds an older title, "Distilled Language Models are economically efficient for the enterprise. ...mostly."; the Anthology title is used) |
| zhao2025agent | Agent-in-the-Loop Data Flywheel | EMNLP Industry, pp. 1919-1930 | 2025 | 10.18653/v1/2025.emnlp-industry.135 | AA + CR |
| brynjolfsson2025generative | Generative AI at Work | QJE 140(2):889-942 | 2025 | 10.1093/qje/qjae044 | CR |
| molino2018cota | COTA | KDD, pp. 586-595 | 2018 | 10.1145/3219819.3219851 | CR |
| mandal2018cognitive | Helpdesk Email Ticket Assignment | ICSOC, LNCS 11236, pp. 332-341 | 2018 | 10.1007/978-3-030-03596-9_23 | CR |
| revina2020it | IT Ticket Classification: The Simpler, the Better | IEEE Access 8:193380-193395 | 2020 | 10.1109/ACCESS.2020.3032840 | CR |
| zangari2023ticket | Ticket Automation (review) | ESWA 225:119984 | 2023 | 10.1016/j.eswa.2023.119984 | CR |
| truss2024ai | Classification of Support Tickets with AutoML | arXiv (IWEMB 2021/22) | 2024 | arxiv.org/abs/2406.01789 | AX |
| zhang2018deep | Deep Learning for Sentiment Analysis: A Survey | WIREs DMKD 8(4):e1253 | 2018 | 10.1002/widm.1253 | CR |
| wankhade2022survey | Survey on Sentiment Analysis Methods | AI Review 55:5731-5780 | 2022 | 10.1007/s10462-022-10144-1 | CR |
| oraby2017how | "How May I Help You?" Twitter dialogue acts | IUI, pp. 343-355 | 2017 | 10.1145/3025171.3025191 | CR + DBLP |
| park2018positivity | Positivity Bias in CSAT Ratings | WWW Companion, pp. 631-638 | 2018 | 10.1145/3184558.3186579 | CR |
| sun2021simulating | Simulating User Satisfaction | SIGIR, pp. 2499-2506 | 2021 | 10.1145/3404835.3463241 | CR |
| singh2022linguistic | Engaging Customer Service Discourse | NLP+CSS workshop, pp. 105-117 | 2022 | 10.18653/v1/2022.nlpcss-1.12 | AA + CR |
| amershi2019guidelines | Guidelines for Human-AI Interaction | CHI, pp. 1-13 | 2019 | 10.1145/3290605.3300233 | CR + MSR page |
| wu2022survey | Survey of Human-in-the-Loop for ML | FGCS 135:364-381 | 2022 | 10.1016/j.future.2022.05.014 | CR |
| mosqueirarey2023human | HITL ML: A State of the Art | AI Review 56(4):3005-3054 | 2023 | 10.1007/s10462-022-10246-w | CR |

### Considered and dropped (could not be verified to a citable standard)

- Several GitHub and Kaggle notebooks on CFPB BERT classification: not peer reviewed.
- "Enhanced Classification of Consumer Complaints with a Modified Transformer Encoder" (Springer chapter 10.1007/978-981-96-8998-9_16): year, authors and volume not confirmed.
- MDPI and Nature 2026 CFPB papers: surfaced in search but not checked further.
- The CSAT preprints arXiv 2411.12539 and 2008.12376: venue not confirmed.
- Serban et al. AAAI 2016 (HRED) was verified (10.1609/aaai.v30i1.9883) but left out to keep the list focused. Add it if you need a hierarchical dialogue baseline.
