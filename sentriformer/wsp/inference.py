"""
SentriFormer inference pipeline (WSP deliverable). Loads any SentriFormerForMultiTask checkpoint and returns
category routing, CSAT forecast, a generated response, and automatic quality flags.

Reuses the project's own decoding (evaluation/audit_m10_3_threeway.generate_multitask) and quality checks
(safety_check_response, detect_repetition). Fixes one inference-preprocessing bug without touching weights:
  - inputs were never truncated at inference; training used 128 context tokens and the positional table has
    512 rows, so any complaint > 512 tokens crashed and 129-512 token contexts were out-of-distribution.

Usage:
  python -I wsp/inference.py "My card was charged twice for the same order."
  python -I wsp/inference.py --ckpt checkpoints/m10_3_sft/m10_3_sft_best.pt --greedy "..."
"""
import argparse, json, os, re, sys
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(ROOT))
from transformers import AutoTokenizer
from sentriformer.model.config import SentriFormerConfig
from sentriformer.model.multi_task_heads import SentriFormerForMultiTask
from sentriformer.evaluation.audit_m10_3_threeway import generate_multitask, safety_check_response, detect_repetition

DEFAULT_CKPT = os.path.join("wsp_runs", "FINAL_T5a", "sentriformer_t5a.pt")  # T5a (rollback: FINAL_T4d)  # previous: FINAL_T3/sentriformer_t3.pt, FINAL/sentriformer_final.pt
DIALOGUE_RUNS = ("T4a", "T4d", "T4d_ctrl", "T5a", "T5a_ctrl", "T6", "T6_ctrl")  # trained on role-labelled multi-turn input (left-truncated to 256 tokens)
CAT_LEN, GEN_CTX_LEN = 128, 128   # CAT_LEN overwritten from the final model's metadata if present
LOW_CONF = 0.25                   # routing confidence gate proposed in the M10.2 audit
PLACEHOLDER = re.compile(r"\{\{[^}]*\}\}")


class SentriFormerPipeline:
    def __init__(self, ckpt=DEFAULT_CKPT, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "tokenizer", "sentritokenizer_v1"))
        li = json.load(open(os.path.join(ROOT, "data", "supervised", "classification_labels.json"), encoding="utf-8"))
        self.cat_names = [li["id2label"][str(i)] for i in range(li["num_classes"])]
        self.cfg = SentriFormerConfig.from_pretrained(os.path.join(ROOT, "checkpoints", "config.json"))
        self.cfg.num_categories, self.cfg.num_csat_classes = li["num_classes"], 5
        path = ckpt if os.path.isabs(ckpt) else os.path.join(ROOT, ckpt)
        raw = torch.load(path, map_location="cpu", weights_only=False)
        self.meta = raw.get("wsp_edit", {}) if isinstance(raw, dict) else {}
        self.aux_trained = isinstance(raw, dict) and ("train_run" in raw or raw.get("aux_heads_trained", False))  # sentiment/urgency heads trained (Phase T)
        self.cat_len = int(self.meta.get("category_seq_len", CAT_LEN))
        self.multi_turn = isinstance(raw, dict) and raw.get("train_run") in DIALOGUE_RUNS
        dj = os.path.join(os.path.dirname(path), "decoding.json")   # optional, written only by a passed T5c experiment
        self.mmi = json.load(open(dj)) if os.path.exists(dj) else None
        self.model = SentriFormerForMultiTask(self.cfg)
        self.model.load_state_dict(raw["model_state_dict"] if "model_state_dict" in raw else raw, strict=True)
        self.model.to(self.device).eval().requires_grad_(False)

    def _enc(self, text, max_len):
        ids = self.tok.encode(text, truncation=True, max_length=max_len, add_special_tokens=True) or [self.cfg.unk_token_id]
        t = torch.tensor([ids], device=self.device)
        return t, torch.ones_like(t)

    def _sent_urg(self, out):
        if not self.aux_trained:
            return {"sentiment_urgency": "unavailable: sentiment/urgency head is untrained in this checkpoint"}
        sp = F.softmax(out["sentiment_logits"].float(), -1)[0]
        lv = float(out["urgency_score"].float()[0]) * 3
        return {"sentiment": {"label": ["negative", "neutral", "positive"][int(sp.argmax())], "probs": [round(float(x), 4) for x in sp]},
                "urgency_experimental": {"level_0_to_3": round(lv, 2), "warning": "urgency only partly validated (Spearman CI > 0 but MAE does not beat the mean); do not use for decisions"}}

    @torch.inference_mode()
    def _dialogue_ids(self, text, history):
        """T4 input format: 'Customer: ... Agent: ... Customer: <text>', oldest tokens dropped first (as in training/eval)."""
        ctx = " ".join([f"{'Customer' if r == 'customer' else 'Agent'}: {t}" for r, t in (history or [])] + [f"Customer: {text}"])
        ids = self.tok.encode(ctx, add_special_tokens=True)[-256:] or [self.cfg.unk_token_id]
        t = torch.tensor([ids], device=self.device); return t, torch.ones_like(t)

    def analyze(self, text, max_new_tokens=100, greedy=None, seed=0, history=None):
        """history: optional list of (role, text) earlier turns, role in {"customer", "agent"}. greedy=None -> the
        decoding frozen for this checkpoint (greedy for multi-turn T4 models, sampling for older ones)."""
        if greedy is None: greedy = self.multi_turn
        n_tokens = len(self.tok.encode(text, add_special_tokens=True))
        ids, am = self._enc(text, self.cat_len)
        out = self.model(input_ids=ids, attention_mask=am)
        cp = F.softmax(out["category_logits"].float(), -1)[0]; sp = F.softmax(out["csat_logits"].float(), -1)[0]
        top = torch.topk(cp, 3)
        gids, gam = self._dialogue_ids(text, history) if self.multi_turn else self._enc(text, GEN_CTX_LEN)
        torch.manual_seed(seed)
        if self.mmi and greedy:   # T5c-validated MMI reranking (decode-time; same code as the experiment: wsp/mmi.py)
            import mmi as M
            cands = M.candidates(self.model, self.tok, self.cfg, gids, n=self.mmi["n_candidates"], seed_base=1000 * seed, max_new_tokens=max_new_tokens)
            body = cands[M.pick(M.score_all(self.model, self.tok, self.cfg, gids, cands), self.mmi["mmi_lambda"])]
            gen = [self.cfg.bos_token_id] + body + [self.cfg.eos_token_id] if len(body) < max_new_tokens else [self.cfg.bos_token_id] + body
        else:
            gen = generate_multitask(self.model, self.tok, self.cfg, gids, gam, max_new_tokens=max_new_tokens,
                                     temperature=0.0 if greedy else 0.7, repetition_penalty=1.2)[0].tolist()
        eos = self.cfg.eos_token_id in gen
        body = gen[1:gen.index(self.cfg.eos_token_id)] if eos else gen[1:]
        reply = self.tok.decode(body, skip_special_tokens=True).strip()
        safety = safety_check_response(reply, text)
        return {
            "input_tokens": n_tokens, "truncated_for_routing": n_tokens > self.cat_len, "truncated_for_generation": n_tokens > GEN_CTX_LEN,
            "category": {"label": self.cat_names[top.indices[0]], "confidence": round(float(top.values[0]), 4),
                         "top3": [(self.cat_names[i], round(float(v), 4)) for v, i in zip(top.values, top.indices)],
                         "needs_human_triage": float(top.values[0]) < LOW_CONF},
            "csat": {"stars": int(sp.argmax()) + 1, "probs": [round(float(x), 4) for x in sp]},
            **self._sent_urg(out),
            "response": reply,
            "quality_flags": {"eos_terminated": eos, "repetition": detect_repetition(reply), "empty": not reply,
                              "template_placeholder": bool(PLACEHOLDER.search(reply)),
                              "orphaned_channel_phrase": safety["has_orphaned_phrase"], "invented_url": safety["has_invented_url"],
                              "invented_contact": safety["has_invented_contact"], "unsupported_name": safety["has_unsupported_name"]},
        }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("text"); ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--greedy", action="store_true", default=None); ap.add_argument("--device", default=None)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(SentriFormerPipeline(a.ckpt, a.device).analyze(a.text, greedy=a.greedy), indent=2, ensure_ascii=False))
