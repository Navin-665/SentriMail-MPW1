# SentriFormer annotation guidelines — Urgency and Escalation (v1, 2026-11-08)

Two **independent** tasks on the same items. Label each without looking at the other label or any model prediction.
Data never leaves this machine. Do not paste items into any online tool.

## Task A — Urgency (ordinal, 0–3): *How soon does this need attention?*
| Level | Name | Definition | Typical signals |
|---|---|---|---|
| 0 | Low | No time pressure; information, feedback, praise, general questions | "just wondering", compliments, product questions |
| 1 | Medium | A real problem, but waiting a few days causes no extra harm | wrong colour item, promo code not applied, app glitch with workaround |
| 2 | High | Ongoing harm or cost that grows with delay; customer blocked from something they need | money taken but service not provided, missing refund, cannot access account, repeated unresolved contact |
| 3 | Immediate | Risk of serious or irreversible harm right now | fraud / unauthorised transactions, account takeover, safety risk, vulnerable customer in crisis, imminent foreclosure/disconnection |

Rules: judge the **situation**, not the tone. Angry wording alone does not raise urgency; a calm message about fraud is still 3.
If two levels fit, choose the **lower** unless a concrete harm signal is present. Repeated contact (3+ times) raises level by at most one.

## Task B — Escalation (binary): *Can a first-line support reply handle this properly, or does a human specialist/authority need to intervene?*
- **escalate = yes** when resolution needs an action or decision a first-line agent cannot take: fraud/security investigation,
  legal or regulatory complaint, financial hardship arrangement, compensation beyond standard policy, complaint about staff,
  repeated failed resolution (the normal path has already failed), or a safety issue.
- **escalate = no** when the next step is information gathering, troubleshooting, status checks, standard returns/refunds/cancellations.

Urgency ≠ escalation: a locked account (urgency 2) is usually escalate = no; a calm request to formally appeal a credit decision
(urgency 1) is escalate = yes. Negative sentiment alone never implies either.

## Borderline cases (decide using the rules, then tick `borderline`)
1. "I'll go to the regulator if this isn't fixed" — escalation yes only if a formal complaint is being made, not for a threat used as emphasis.
2. Missing refund, first contact, 3 weeks late — urgency 2, escalate no. Same, fourth contact — urgency 2, escalate yes (normal path failed).
3. Fraud already resolved by the bank, customer wants confirmation — urgency 1, escalate no.
4. Elderly customer confused but not at risk — urgency by situation; note vulnerability in `notes`.

## PII redaction (before labelling and before any sharing)
Run `sample_for_annotation.py` (redacts automatically). If you still see a name, email, phone, address, card/account/order number,
replace it with a placeholder ({{Name}}, {{Email}}, {{Phone}}, {{Address}}, {{Account Number}}, {{Order Number}}) and tick `pii_fixed`.

## Process
1. **Pilot:** items P001–P200 are labelled by **two** annotators independently.
2. Agreement: escalation → Cohen's κ; urgency → quadratic-weighted κ plus exact and within-1 agreement (`agreement.py`).
   Proceed to the main batch only if κ(escalation) ≥ 0.6 and weighted κ(urgency) ≥ 0.6; otherwise revise these guidelines and re-pilot.
3. **Adjudication:** a third person (or the two annotators together) resolves every disagreement, writes the reason in
   `adjudication_note`, and adds a borderline example here if a rule was unclear.
4. Main batch: single labelling with 10% random double-labelling to monitor drift.
5. Labels are reviewed for consistency (agreement + adjudication log) **before** any training use.
