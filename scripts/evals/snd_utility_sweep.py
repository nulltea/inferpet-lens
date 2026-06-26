#!/usr/bin/env python3
"""SnD utility-recovery sweep — the UTILITY axis of the SnD privacy–utility tradeoff.

Privatize token embeddings with dχ-privacy (DxPrivacy), let the rest of the model produce the
pooled output embedding e_n, denoise it locally with a noise-aware transformer (Denoiser) →
e_d, and measure how much of the clean pooled embedding e_c the denoiser recovers across the
budget η. Also report teacher-forced perplexity/acc degradation under the dχ noise (generation-
utility cost; the denoiser does not touch that surface). Privacy axis stays in dp_leakage_sweep.

η is the dχ budget, NOT the Gaussian ε of dp_leakage_sweep — not interchangeable.

GPU: ONE process at a time; run via scripts/run_in_rocm.sh. Output JSON under refine-logs/snd/.

  scripts/run_in_rocm.sh python3 scripts/evals/snd_utility_sweep.py \
      --etas inf,100,50,10,1 --train-etas 50,10,1 --out refine-logs/snd/snd_utility_sweep.json
"""
from __future__ import annotations

import numpy as np


def recovery_metrics(e_c, e_n, e_d) -> dict:
    """cos/MSE of noised & denoised pooled embeddings vs clean, and the fraction of the gap closed."""
    def _cos(a, b):
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
        return float((a * b).sum(1).mean())

    cos_n, cos_d = _cos(e_n, e_c), _cos(e_d, e_c)
    mse_n = float(((e_n - e_c) ** 2).mean())
    mse_d = float(((e_d - e_c) ** 2).mean())
    return {
        "cos_noised": cos_n, "cos_denoised": cos_d,
        "mse_noised": mse_n, "mse_denoised": mse_d,
        "recovery_cos": (cos_d - cos_n) / (1 - cos_n) if (1 - cos_n) > 1e-9 else 0.0,
        "recovery_mse": 1 - mse_d / mse_n if mse_n > 1e-12 else 0.0,
    }


import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # scripts/ for defenses.*
from defenses.snd import DxPrivacy, Denoiser                          # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def capture_pooled(model, tok, prompts, eta, C, batch_size=32):
    """Mean-pooled last-hidden (e), per-prompt input-embeddings X̃ (list of (n_i,d)), and lengths.

    eta=inf ⇒ clean (no DxPrivacy hook). A second hook grabs the embedding-layer output AFTER
    DxPrivacy has replaced it (hook return propagates to later hooks + downstream), so grab["x"]
    is exactly the X̃ that the rest of the model consumed. Right-padded; pads dropped per real len."""
    embs, Xt = [], []
    hk = None if math.isinf(eta) else model.get_input_embeddings().register_forward_hook(DxPrivacy(C, eta))
    grab = {}
    h2 = model.get_input_embeddings().register_forward_hook(lambda m, i, o: grab.__setitem__("x", o.detach()))
    try:
        for i in range(0, len(prompts), batch_size):
            enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True)
            ids, mask = enc.input_ids.to(DEV), enc.attention_mask.to(DEV)
            out = model(ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
            last = out.hidden_states[-1].float()                       # (B,T,d) — post-hook, propagated
            m = mask.unsqueeze(-1).float()
            pooled = (last * m).sum(1) / m.sum(1).clamp_min(1.0)        # mean over real tokens
            xpad = grab["x"].float()                                   # (B,T,d): X̃ if hook else clean X
            for b in range(ids.shape[0]):
                n = int(mask[b].sum())
                embs.append(pooled[b].cpu().numpy())
                Xt.append(xpad[b, :n].cpu().numpy())
    finally:
        h2.remove()
        if hk is not None:
            hk.remove()
    return np.stack(embs), Xt


def _pad_batch(Xt_list, idx, d):
    """Stack a list of (n_i, d) arrays at positions `idx` into (B, Tmax, d) + bool pad mask (B,Tmax)."""
    sub = [Xt_list[i] for i in idx]
    Tmax = max(x.shape[0] for x in sub)
    B = len(sub)
    X = np.zeros((B, Tmax, d), np.float32)
    pad = np.ones((B, Tmax), bool)
    for b, x in enumerate(sub):
        X[b, :x.shape[0]] = x
        pad[b, :x.shape[0]] = False
    return X, pad


def _denoise(den, e_n, Xt, Xc, idx, d):
    """One denoiser forward over prompt positions `idx`: build padded X̃, Z=X̃−X, run, return (len(idx),d)."""
    Xtl, pad = _pad_batch(Xt, idx, d)
    Xcl, _ = _pad_batch(Xc, idx, d)
    Z = Xtl - Xcl
    return den(torch.from_numpy(e_n[idx]).to(DEV),
               torch.from_numpy(Xtl).to(DEV), torch.from_numpy(Z).to(DEV),
               pad_mask=torch.from_numpy(pad).to(DEV))


def train_denoiser(model, tok, prompts, train_etas, C, d, epochs, batch_size, seed):
    """Train ONE noise-aware denoiser on (e_n, X̃, Z, e_c) over several η. e_c/X captured once (clean)."""
    torch.manual_seed(seed)
    e_c, Xc = capture_pooled(model, tok, prompts, math.inf, C, batch_size)     # clean pooled + clean X
    den = Denoiser(d=d).to(DEV)
    opt = torch.optim.Adam(den.parameters(), lr=1e-3)
    N = len(prompts)
    loss = torch.tensor(float("nan"))
    for ep in range(epochs):
        for eta in train_etas:
            e_n, Xt = capture_pooled(model, tok, prompts, eta, C, batch_size)
            order = np.random.default_rng(seed + ep).permutation(N)
            for s in range(0, N, batch_size):
                idx = order[s:s + batch_size]
                ed = _denoise(den, e_n, Xt, Xc, idx, d)
                loss = ((ed - torch.from_numpy(e_c[idx]).to(DEV)) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        print(f"[snd] denoiser epoch {ep+1}/{epochs} last-loss {loss.item():.4f}", flush=True)
    return den.eval()


@torch.no_grad()
def utility_pass(model, tok, prompts, eta, C, batch_size):
    """Teacher-forced perplexity + next-token top-1 acc under the dχ hook (eta=inf ⇒ clean)."""
    hk = None if math.isinf(eta) else model.get_input_embeddings().register_forward_hook(DxPrivacy(C, eta))
    ce_sum, n_tok, n_corr = 0.0, 0, 0
    try:
        for i in range(0, len(prompts), batch_size):
            enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True)
            ids, mask = enc.input_ids.to(DEV), enc.attention_mask.to(DEV)
            logits = model(ids, attention_mask=mask, use_cache=False).logits.float()
            pred, tgt, m = logits[:, :-1], ids[:, 1:], mask[:, 1:].bool()
            lp = torch.log_softmax(pred, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            ce_sum += float(-(lp[m]).sum()); n_tok += int(m.sum())
            n_corr += int((pred.argmax(-1)[m] == tgt[m]).sum())
    finally:
        if hk is not None:
            hk.remove()
    return math.exp(ce_sum / max(1, n_tok)), n_corr / max(1, n_tok)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--corpus", default="corpora/rep2text-stratified.txt")
    ap.add_argument("--max-prompts", type=int, default=400)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--etas", default="inf,100,50,10,1", help="dχ budget sweep; 'inf'=clean")
    ap.add_argument("--train-etas", default="50,10,1", help="η values the denoiser trains on")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--clip-percentile", type=float, default=99.9)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260626)
    ap.add_argument("--out", default="refine-logs/snd/snd_utility_sweep.json")
    args = ap.parse_args()

    etas = [math.inf if s.strip().lower().startswith("inf") else float(s)
            for s in args.etas.split(",") if s.strip()]
    if not any(math.isinf(e) for e in etas):
        etas = [math.inf] + etas
    train_etas = [float(s) for s in args.train_etas.split(",") if s.strip()]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    prompts = [l.strip() for l in Path(args.corpus).read_text().splitlines() if l.strip()][: args.max_prompts]
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, attn_implementation="eager", device_map=DEV).eval()
    d = model.config.hidden_size

    cal = []
    h = model.get_input_embeddings().register_forward_hook(
        lambda m, i, o: cal.append(o.float().norm(dim=-1).flatten().cpu()))
    try:
        with torch.no_grad():
            for p in prompts[:48]:
                model(tok(p, return_tensors="pt").input_ids.to(DEV), use_cache=False)
    finally:
        h.remove()
    C = float(np.percentile(torch.cat(cal).numpy(), args.clip_percentile))

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(prompts))
    ntr = int(args.train_frac * len(prompts))
    tr_prompts = [prompts[i] for i in perm[:ntr]]
    te_prompts = [prompts[i] for i in perm[ntr:]]
    print(f"[snd] model={args.model} d={d} C={C:.3f} etas={etas} train_etas={train_etas} "
          f"train/test={len(tr_prompts)}/{len(te_prompts)} dev={DEV}", flush=True)

    den = train_denoiser(model, tok, tr_prompts, train_etas, C, d, args.epochs, args.batch_size, args.seed)

    e_c, Xc = capture_pooled(model, tok, te_prompts, math.inf, C, args.batch_size)      # clean test
    base_ppl, base_acc = utility_pass(model, tok, te_prompts, math.inf, C, args.batch_size)
    records = []
    for eta in etas:
        e_n, Xt = capture_pooled(model, tok, te_prompts, eta, C, args.batch_size)
        e_d = np.zeros_like(e_n)
        with torch.no_grad():
            for s in range(0, len(te_prompts), args.batch_size):
                idx = np.arange(s, min(s + args.batch_size, len(te_prompts)))
                e_d[idx] = _denoise(den, e_n, Xt, Xc, idx, d).cpu().numpy()
        rec = {"eta": (None if math.isinf(eta) else eta)}
        rec.update(recovery_metrics(e_c, e_n, e_d))
        ppl, acc = utility_pass(model, tok, te_prompts, eta, C, args.batch_size)
        rec.update({"perplexity": ppl, "acc": acc,
                    "ppl_degradation": ppl / base_ppl - 1, "retention_acc": acc / base_acc if base_acc else None})
        records.append(rec)
        es = "inf" if math.isinf(eta) else f"{eta:g}"
        print(f"[snd] η={es:>5} | cos {rec['cos_noised']:.3f}->{rec['cos_denoised']:.3f} "
              f"rec_cos={rec['recovery_cos']:.3f} rec_mse={rec['recovery_mse']:.3f} "
              f"ppl_deg={rec['ppl_degradation']:+.2%}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "model": args.model, "corpus": args.corpus, "n_test": len(te_prompts), "hidden": d,
        "defense": "snd_dx", "clip_C": C, "clip_percentile": args.clip_percentile,
        "budget_note": "eta is the dχ-privacy budget (larger=weaker privacy); NOT the Gaussian ε "
                       "of dp_leakage_sweep — the two are not interchangeable.",
        "etas": [None if math.isinf(e) else e for e in etas], "train_etas": train_etas,
        "epochs": args.epochs, "seed": args.seed,
        "readout_note": "recovery_cos = fraction of (1-cos) gap closed by the denoiser; recovery_mse "
                        "= fraction of noised MSE removed; ppl/acc degradation = generation-utility "
                        "cost of the dχ noise (denoiser does not touch logits).",
        "records": records,
    }, indent=2))
    print(f"[snd] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
