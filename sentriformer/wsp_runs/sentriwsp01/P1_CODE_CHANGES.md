# p1_patch_eval.py code changes (implementation only; P1_CRITERIA.md rules unchanged)

## 2026-11-08, after review round 2, BEFORE any Stage A run
1. **Resumable, strict patch saving** (`save_or_validate_patch`): always rebuilds the model as baseline T4d + this patch (never
   inherits a control edit). A new patch is written atomically (`.tmp` then `os.replace`). An existing patch from a killed run is
   reused only if its `wsp_patch` metadata (id, tensor name, columns, α, base SHA-256) matches exactly AND it reloads strictly to the
   same parameters with the embedding tie intact; otherwise the run stops with an error and nothing is overwritten.
2. **A/B key separated:** `p1_results/AB_P1_SHEET.csv` (shareable, blinded) vs `_sealed_keys/AB_P1_KEY.json` (private). Never upload `_sealed_keys/`.
3. **Decision-gate unit tests** in `selftest` (10 cases): good candidate passes; repetition +5pp rejects; a control beating P1
   rejects; a control tying P1 rejects; CI upper bound 0.01 and exactly 0 reject; −1e-9 passes; classification change rejects;
   template swap rejects; action-word loss >2pp rejects.
4. Cleanup: pandas/csv/random imported at the top; the walrus expression and inline import replaced by `rows_all = pd.concat([A, B])`.
5. The self-test also exercises the save/resume path in the session scratch dir: new save → validated reuse while a control edit
   is deliberately left in memory → refusal on tampered metadata; the temp file is deleted.
Earlier (round 1): activation-shape assertion, eval-mode assertion, strict save/reload check, blinded A/B export.

## 2026-11-08, after review round 3, BEFORE any Stage A run
6. `stageA.json`, `stageB.json` and `AB_P1_KEY.json` are written atomically (`dump_atomic`: `.tmp` then `os.replace`), so a killed
   write cannot leave a partial file that blocks the next run.
7. The Stage B blind A/B sheet and key are generated only if Stage B passes (`if ok:`), matching the frozen criteria order.
8. Boundary tests for every remaining regression gate (specificity, orphan, repetition, EOS, portal, privacy, distinct, ROUGE-L,
   dialogue loss, top-reply share, template concentration): exactly at tolerance passes, 1e-6 beyond fails. 32 gate tests in total.
