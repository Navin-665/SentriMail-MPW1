"""MMI-antiLM reranking (Li et al. 2016), decode-time only; shared by the T5c experiment and wsp/inference.py.
score(r) = mean log p(r | context) - lam * mean log p(r | "Customer:")   (token-averaged over reply tokens + EOS)."""
import torch
import torch.nn.functional as F
from sentriformer.evaluation.audit_m10_3_threeway import generate_multitask

EMPTY_CTX = "Customer:"


def _body(g, eos_id):
    return g[1:g.index(eos_id)] if eos_id in g else g[1:]


@torch.no_grad()
def mean_logp(model, cfg, enc_ids, body):
    """Mean log-prob of body + [EOS] given encoder input enc_ids ([1, T] tensor). Decoder input = [BOS] + body (position t predicts body[t])."""
    am = torch.ones_like(enc_ids); enc = model.sentriformer.encoder(input_ids=enc_ids, attention_mask=am)
    tgt = torch.tensor(body + [cfg.eos_token_id], device=enc_ids.device)
    h, _ = model.sentriformer.decoder(input_ids=torch.tensor([[cfg.bos_token_id] + body], device=enc_ids.device),
                                      encoder_hidden_states=enc, encoder_attention_mask=am)
    lp = F.log_softmax(model.lm_head(h)[0].float(), -1)
    return float(lp[torch.arange(len(tgt)), tgt].mean())


@torch.no_grad()
def candidates(model, tok, cfg, enc_ids, n=8, seed_base=0, max_new_tokens=128):
    """1 greedy (deployed decoding) + n-1 samples (T=0.7, top-k 50, rep 1.2, seeds seed_base+j). Returns token bodies."""
    am = torch.ones_like(enc_ids)
    out = [_body(generate_multitask(model, tok, cfg, enc_ids, am, max_new_tokens=max_new_tokens, temperature=0.0, repetition_penalty=1.2)[0].tolist(), cfg.eos_token_id)]
    for j in range(1, n):
        torch.manual_seed(seed_base + j)
        out.append(_body(generate_multitask(model, tok, cfg, enc_ids, am, max_new_tokens=max_new_tokens, temperature=0.7, top_k=50,
                                            repetition_penalty=1.2)[0].tolist(), cfg.eos_token_id))
    return out


@torch.no_grad()
def score_all(model, tok, cfg, enc_ids, cands):
    empty = torch.tensor([tok.encode(EMPTY_CTX, add_special_tokens=True)], device=enc_ids.device)
    return [(mean_logp(model, cfg, enc_ids, b), mean_logp(model, cfg, empty, b)) for b in cands]


def pick(scores, lam):
    return max(range(len(scores)), key=lambda i: scores[i][0] - lam * scores[i][1])
