"""Build the clean public repo from SENTRIMAIL-MPW1 (read-only on the source; copies only).
Usage: python -I build_repo.py <SENTRIMAIL-MPW1 dir> <repo dir>
Policy: code, configs, metrics and plots only. No corpus/customer text, no secrets, no checkpoints except the deployed
T5a weights. JSON files are copied only if no string value looks like free text (>=15 words).
"""
import fnmatch, json, os, shutil, sys

SRC, DST = sys.argv[1], sys.argv[2]
copied, skipped = [], []


def free_text_json(p):
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception:
        return True
    stack = [d]
    while stack:
        x = stack.pop()
        if isinstance(x, dict): stack += list(x.values()) + list(x.keys())
        elif isinstance(x, list): stack += x
        elif isinstance(x, str) and len(x.split()) >= 15: return True
    return False


def cp(src_rel, dst_rel):
    s, d = os.path.join(SRC, src_rel), os.path.join(DST, dst_rel)
    if s.endswith(".json") and free_text_json(s):
        skipped.append(src_rel + "  (free text in JSON)"); return
    os.makedirs(os.path.dirname(d), exist_ok=True); shutil.copy2(s, d); copied.append(dst_rel)


def tree(src_rel, dst_rel, include, exclude=()):
    base = os.path.join(SRC, src_rel)
    for root, dirs, files in os.walk(base):
        dirs[:] = [x for x in dirs if x not in ("__pycache__", ".venv", "venv", ".git")]
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), base).replace("\\", "/")
            if any(fnmatch.fnmatch(rel, e) or fnmatch.fnmatch(f, e) for e in exclude): skipped.append(f"{src_rel}/{rel}"); continue
            if any(fnmatch.fnmatch(rel, i) or fnmatch.fnmatch(f, i) for i in include):
                cp(f"{src_rel}/{rel}", f"{dst_rel}/{rel}")


SF = "sentriformer"
CODE = ["*.py"]
# --- SentriFormer model package
for sub in ["model", "corpus", "training", "wsp"]:
    tree(f"{SF}/{sub}", f"sentriformer/{sub}", CODE)
tree(f"{SF}/tokenizer", "sentriformer/tokenizer", CODE + ["*.json"])
tree(f"{SF}/evaluation", "sentriformer/evaluation", CODE, exclude=["scratch_*", "dump_*", "extract_16_failures.py", "print_examples.py", "test_fn.py"])
tree(f"{SF}/data", "sentriformer/data", CODE + ["corpus_stats.json", "classification_labels.json"])
cp(f"{SF}/__init__.py", "sentriformer/__init__.py")
cp(f"{SF}/checkpoints/config.json", "sentriformer/checkpoints/config.json")
for f in os.listdir(os.path.join(SRC, SF)):
    if f.endswith(".py") and f != "__init__.py": cp(f"{SF}/{f}", f"sentriformer/scripts/{f}")
# results: all plots + numeric summaries; behavioural audits quote model/corpus text -> excluded
tree(f"{SF}/results", "sentriformer/results", ["*.png", "*.py", "pretraining_report.md", "classification_report.md",
                                                 "m9_1_audit_summary.json", "assembly_benchmark_results.json",
                                                 "m10_3_dataset_report.json", "m10_3_quality_gate_report.json", "M10_4_GATE_REPORT.md"],
     exclude=["*AUDIT*", "*audit*.json", "*.txt", "*MANUAL_REVIEW*", "*attribution*"])
# experiment records (wsp_runs): decisions, criteria, metrics, histories, configs; deployed weights only
WR_INC = ["*_DECISION.md", "*_CRITERIA.md", "*_FREEZE.txt", "MODEL_CARD.md", "WSP_FINAL_REPORT.md", "SELECTION_CRITERIA.md",
          "RUN_PLAN_*.md", "E0_REPORT.md", "P1_CODE_CHANGES.md", "GUIDELINES.md", "metrics.json", "eval_history.json",
          "config.json", "train_log.txt", "*build_report.json", "acceptance.json", "e0_metrics.json", "e1_metrics.json",
          "e2_report.json", "regression_tests.json", "t5_compare.json", "t5a_retest.json", "t6_compare.json",
          "t7u_results.json", "fit_report.json", "pretrain_overlap.json", "integrity_*.json", "aux_data_report.json", "*.py", "*.sh"]
WR_EXC = ["*LOCAL_ONLY*", "T7U_DECISION.md", "PHASE1_REPORT.md", "WSP_P2_DESIGN.md", "T5_DESIGN.md", "_sealed_keys/*",
          "ab_test/*", "behavior_suite/*", "generations.json", "context_correctness.json", "*.parquet", "*.npz", "*.npy",
          "*.csv", "*.pt", "feats/*", "p1_results/*", "p2_results/*", "t5c_results/*", "phase1*_results.json", "decode/*"]
tree(f"{SF}/wsp_runs", "sentriformer/wsp_runs", WR_INC, WR_EXC)
cp(f"{SF}/wsp_runs/FINAL_T5a/sentriformer_t5a.pt", "sentriformer/wsp_runs/FINAL_T5a/sentriformer_t5a.pt")
# --- SentriMail application
APP = "Sentrimail/sentrimail"
tree(f"{APP}/app", "apps/sentrimail/app", CODE)
tree(f"{APP}/templates", "apps/sentrimail/templates", ["*.html"])
tree(f"{APP}/working_images", "apps/sentrimail/working_images", ["*.png"])
for f in ["run.py", "migrate.py", "requirements.txt", "Procfile", "runtime.txt", ".env.example", "README.md", "SENTRIMAIL_INTERVIEW_PREP.md"]:
    cp(f"{APP}/{f}", f"apps/sentrimail/{f}")
cp(f"{APP}/dataset/new.py", "apps/sentrimail/dataset/new.py")
cp("Sentrimail/mermaid-diagram.svg", "docs/figures/sentrimail_architecture_mermaid.svg")
# --- earlier T5-small pipeline (code only)
tree("model_training", "legacy/model_training", CODE + ["README.md", "requirements.txt"])
# --- top-level reports and data documentation
for f in ["README.md", "classification_results_report.md", "csat_report.md", "dialogue_audit_report.md", "multitask_report.md",
          "pretraining_results_report.md", "research_comparative_evaluation.md", "walkthrough.md", "sentriformer_project_workflow.excalidraw"]:
    cp(f, f"docs/reports/{f}" if f != "README.md" else "docs/reports/PROJECT_README_ORIGINAL.md")
cp("rawdata/DATA_SOURCES.md", "docs/DATA_SOURCES.md")
for d in ["apps/sentrimail/data", "apps/sentrimail/static"]:
    os.makedirs(os.path.join(DST, d), exist_ok=True); open(os.path.join(DST, d, ".gitkeep"), "w").close()

json.dump({"copied": len(copied), "skipped": skipped}, open(os.path.join(DST, "research_publications", "_runs", "repo_build_log.json"), "w"), indent=1)
print("copied", len(copied), "skipped", len(skipped))
for s in skipped:
    if "free text" in s: print("  ", s)
