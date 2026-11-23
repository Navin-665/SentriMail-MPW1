# T5c (MMI reranking, decode-time only) — criteria FROZEN 2026-11-08 before any T5c run (hash in T5C_FREEZE.txt)

## Method
The model = the deployed default after the T5 decision (T5a if T5_CRITERIA passes, else T4d); weights are never changed.
Per row: 8 candidates = 1 greedy (deployed decoding: greedy, repetition penalty 1.2) + 7 sampled (temperature 0.7, top-k 50, rep 1.2,
seed 1000·row_position + j). Score = mean log p(r | context) − λ · mean log p(r | "Customer:") (token-averaged, EOS included; Li et al. 2016).
Choose the highest-scoring candidate. λ ∈ {0.15, 0.3, 0.5}.
**Control:** the same candidate pool, with one sampled candidate picked uniformly at random (seed 7·row_position). This isolates reranking
from mere sampling diversity.

## Rows
500 fresh `multi_val` rows with history, random_state = 97, excluding every earlier set (gate rs=11, tune rs=29, Phase-1, P1 rs=53,
P2 rs=67, T5 rs=79). A = first 250 (screen and select λ), B = last 250 (one confirmation).

## Stage A: λ is eligible iff, vs greedy on rows A
1. GENERIC-phrase rate lower, with the paired bootstrap 95% CI of (MMI − greedy) entirely below 0;
2. its reduction is larger than the random-pick control's reduction;
3. no regression: action ≥ −2pp; specificity ≥ −2pp; broken_link ≤ +1pp; orphan ≤ +1pp; privacy ≤ +1pp; portal ≤ +1pp; repetition ≤ +1pp;
   EOS ≥ −1pp; distinct ≥ −0.02; ROUGE-L (non-portal references) ≥ −0.005; mean length within ±20%; empty ≤ +1pp; template ≤ +2pp.
Select the smallest eligible λ. None eligible → REJECT T5c, and Stage B does not run.

## Stage B (once)
The selected λ on rows B; rules 1–3 must hold again. PASS → write `decoding.json` {"mmi_lambda": λ, "n_candidates": 8} next to the deployed
checkpoint, so `inference.py` uses MMI by default for that checkpoint (removable = rollback). Plus a LOCAL-ONLY blind A/B sheet (MMI vs greedy).
