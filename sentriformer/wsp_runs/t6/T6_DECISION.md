# T6 decision — REJECTED (2026-11-09 13:18), T5a stays default

Frozen criteria: T6_CRITERIA.md. Numbers: t6_compare.json (500 multi_val rows, rs=131; test set not used).

Failed: P1 (primary), G2_action, G2_template.
- P1 judge-PMI T6 - T6_ctrl = +0.003, 95% CI [-0.012, +0.019] -> includes 0; the PMI filter did not make replies measurably more
  conversation-specific than training on the same data unfiltered.
- G2 action words vs T5a: 39.8% -> 35.4% (-4.4pp, limit -2pp). G2 template concentration 0.278 -> 0.302 (+2.4pp, limit +2pp).
- Also: whole-reply generic rose vs ctrl (+3.2pp, CI [+1.0, +5.4]) — the filter made this worse.
Passed: all G1 heads (category/CSAT/sentiment), G3 dial loss, and the other G2 checks.
Note: T6_ctrl (T5a + 10k more steps, same data) is itself not a candidate; continuing training was not pre-registered.
Filters not human-validated. compare_T6 died silently on its first attempt (11:47), rerun 13:13 completed.
