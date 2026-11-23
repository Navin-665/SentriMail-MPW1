"""Blind human A/B sheet: first 100 of eval_t4's fixed 300 VAL rows; replies of T3 vs T4d in random order per row.
Writes wsp_runs/ab_test/AB_SHEET.csv (for the rater) and AB_KEY.json (do not open until rating is done).
Score afterwards:  python -I wsp/build_ab_sheet.py score"""
import csv, json, os, random, sys
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wsp_runs"); O = os.path.join(R, "ab_test")
if sys.argv[1:] == ["score"]:
    key = json.load(open(os.path.join(O, "AB_KEY.json"))); w = {"T3": 0, "T4d": 0, "tie": 0}
    for row in csv.DictReader(open(os.path.join(O, "AB_SHEET.csv"), encoding="utf-8-sig")):
        c = row["better (A/B/tie)"].strip().upper()
        if c in ("A", "B"): w[key[row["id"]][c]] += 1
        elif c == "TIE": w["tie"] += 1
    n = w["T3"] + w["T4d"]; print(w, f"T4d win rate excl. ties: {w['T4d'] / n:.2f}" if n else ""); sys.exit()
assert not os.path.exists(os.path.join(O, "AB_SHEET.csv")), "refusing to overwrite"
g = {m: json.load(open(os.path.join(R, "t4", "eval", f"{m}_val", "generations.json"), encoding="utf-8"))[:100] for m in ("T4d", "T3")}
assert [x["group_id"] for x in g["T3"]] == [x["group_id"] for x in g["T4d"]]
rng = random.Random(2026); key = {}
with open(os.path.join(O, "AB_SHEET.csv"), "w", newline="", encoding="utf-8-sig") as f:
    wr = csv.writer(f); wr.writerow(["id", "conversation", "reply A", "reply B", "better (A/B/tie)", "note"])
    for i, (x3, xd) in enumerate(zip(g["T3"], g["T4d"])):
        a, b = (("T3", x3), ("T4d", xd)) if rng.random() < .5 else (("T4d", xd), ("T3", x3))
        rid = f"AB{i + 1:03d}"; key[rid] = {"A": a[0], "B": b[0]}
        wr.writerow([rid, xd["input"].replace(" Agent: ", "\nAgent: ").replace(" Customer: ", "\nCustomer: "), a[1]["generated"], b[1]["generated"], "", ""])
json.dump(key, open(os.path.join(O, "AB_KEY.json"), "w"), indent=0)
print("wrote 100 rows;", sum(v["A"] == "T4d" for v in key.values()), "have T4d as A")
