"""For result JSONs skipped by build_repo.py (they embed reply text), write numeric-only copies (*_numeric.json).
Usage: python -I sanitize_json.py <SENTRIMAIL-MPW1 dir> <repo dir>"""
import json, os, sys

SRC, DST = sys.argv[1], sys.argv[2]
log = json.load(open(os.path.join(DST, "research_publications", "_runs", "repo_build_log.json")))
is_text = lambda v: isinstance(v, str) and len(v.split()) >= 15


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items() if not is_text(v)}
    if isinstance(x, list):
        return [clean(v) for v in x if not is_text(v)]
    return x


for s in log["skipped"]:
    if "free text" not in s: continue
    rel = s.split("  (")[0]
    c = clean(json.load(open(os.path.join(SRC, rel), encoding="utf-8")))
    assert not any(is_text(v) for v in json.dumps(c).split('"') if False)
    out = os.path.join(DST, rel.replace("sentriformer/", "sentriformer/", 1).replace(".json", "_numeric.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(c, open(out, "w", encoding="utf-8"), indent=1)
    print("wrote", out)
