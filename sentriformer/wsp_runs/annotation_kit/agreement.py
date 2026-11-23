"""
Inter-annotator agreement for the pilot. Usage: python -I agreement.py pilot_200_annotatorA.csv pilot_200_annotatorB.csv
Escalation: Cohen's kappa. Urgency: quadratic-weighted kappa + exact / within-1 agreement. Writes disagreements_for_adjudication.csv.
"""
import csv, sys, os
import numpy as np


def kappa(a, b, k, weights=None):
    O = np.zeros((k, k))
    for x, y in zip(a, b): O[x, y] += 1
    n = O.sum(); E = np.outer(O.sum(1), O.sum(0)) / n
    W = np.ones((k, k)) - np.eye(k) if weights is None else np.array([[((i - j) / (k - 1)) ** 2 for j in range(k)] for i in range(k)])
    return 1 - (W * O).sum() / max((W * E).sum(), 1e-12)


def _selftest():
    assert abs(kappa([0, 1, 0, 1], [0, 1, 0, 1], 2) - 1) < 1e-9
    assert abs(kappa([0, 1, 2, 3], [0, 1, 2, 3], 4, "quadratic") - 1) < 1e-9
    assert kappa([0, 0, 1, 1], [1, 1, 0, 0], 2) < 0
_selftest()


def read(p): return {r["item_id"]: r for r in csv.DictReader(open(p, encoding="utf-8"))}


def main():
    A, B = read(sys.argv[1]), read(sys.argv[2])
    ids = [i for i in A if i in B and A[i]["urgency_0_3"].strip() and B[i]["urgency_0_3"].strip() and A[i]["escalate_yes_no"].strip() and B[i]["escalate_yes_no"].strip()]
    ua = [int(A[i]["urgency_0_3"]) for i in ids]; ub = [int(B[i]["urgency_0_3"]) for i in ids]
    yn = lambda s: 1 if s.strip().lower().startswith("y") else 0
    ea = [yn(A[i]["escalate_yes_no"]) for i in ids]; eb = [yn(B[i]["escalate_yes_no"]) for i in ids]
    res = {"n": len(ids), "escalation_cohen_kappa": kappa(ea, eb, 2), "urgency_weighted_kappa": kappa(ua, ub, 4, "quadratic"),
           "urgency_exact": float(np.mean(np.array(ua) == np.array(ub))), "urgency_within1": float(np.mean(np.abs(np.array(ua) - np.array(ub)) <= 1))}
    res["proceed_to_main_batch"] = res["escalation_cohen_kappa"] >= 0.6 and res["urgency_weighted_kappa"] >= 0.6
    out = os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), "disagreements_for_adjudication.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["item_id", "text", "urgency_A", "urgency_B", "escalate_A", "escalate_B", "final_urgency", "final_escalate", "adjudication_note"])
        for i, x, y, p, q in zip(ids, ua, ub, ea, eb):
            if x != y or p != q: w.writerow([i, A[i]["text"][:500], x, y, p, q, "", "", ""])
    print(res)


if __name__ == "__main__":
    main()
