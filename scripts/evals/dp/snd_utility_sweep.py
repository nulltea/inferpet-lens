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


def make_eta_groups(train_etas, n_groups):
    """Partition train η values into n_groups contiguous noise-strength bins (paper §A.5.3).

    Returns (groups, reps): groups[g] = sorted η list for bin g (ascending η = strong→mild noise);
    reps[g] = geometric-mean η of the bin (its representative, for routing)."""
    import numpy as _np
    etas = sorted(set(float(e) for e in train_etas))
    n_groups = max(1, min(n_groups, len(etas)))
    groups = [list(a) for a in _np.array_split(_np.array(etas), n_groups)]
    reps = [float(_np.exp(_np.mean(_np.log(g)))) for g in groups]
    return groups, reps


def route_eta(eta, reps):
    """Route an eval η to its denoiser group: nearest representative in log-η space.
    η=inf (clean) → mildest-noise group (largest rep)."""
    import math as _m
    if eta is None or _m.isinf(eta):
        return int(max(range(len(reps)), key=lambda i: reps[i]))
    return int(min(range(len(reps)), key=lambda i: abs(_m.log(eta) - _m.log(reps[i]))))


import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))  # talens.*
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))          # scripts/ for defenses.*
from defenses.snd import DxPrivacy, Denoiser                          # noqa: E402
from talens.probes.utility import (  # noqa: E402  (standardized utility probes; comparable across schemes)
    teacher_forced_pass, next_token_accuracy, perplexity, output_agreement,
    embedding_recovery, retention_thresholds)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def capture_pooled(model, tok, prompts, eta, C, batch_size=32, max_tokens=64):
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
            enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_tokens)   # bound denoiser seq (2T+1) on the iGPU
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


def _cos_mse_loss(pred, target, cos_weight):
    """MSE + cos_weight·(1 − cosine): aligns training with the cosine recovery metric."""
    mse = ((pred - target) ** 2).mean()
    cos = 1.0 - torch.nn.functional.cosine_similarity(pred, target, dim=-1).mean()
    return mse + cos_weight * cos


def _lr_at(ep, epochs, base_lr, warmup_frac=0.1):
    """Linear warmup then cosine decay to 0 over `epochs` (per-epoch granularity)."""
    we = max(1, int(warmup_frac * epochs))
    if ep < we:
        return base_lr * (ep + 1) / we
    prog = (ep - we) / max(1, epochs - we)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * prog))


def train_denoiser(model, tok, prompts, train_etas, C, d, epochs, batch_size, seed,
                   standardize=True, max_tokens=64, n_layers=6, d_ff=0, dropout=0.1,
                   residual=True, lr=1e-3, cos_weight=1.0, patience=4, val_frac=0.15, n_groups=3):
    """Train per-η-GROUP noise-aware denoisers (paper §A.5.3) on (e_n, X̃, Z, e_c). Returns
    (dens, μ, σ, reps): one Denoiser per group, the shared standardizer, and group representatives
    for routing (route_eta). e_c/X captured once (clean); each train-η's noised set captured once and
    reused across epochs (shared across groups).

    Pooled embeddings are STANDARDIZED per-dim by clean-TRAIN stats (μ,σ): LLM hidden states are
    anisotropic (a few outlier dims dominate raw L2/cosine), so raw-space recovery is degenerate
    (cos≈1 regardless of corruption). Training: MSE+cosine loss, AdamW, warmup→cosine LR, grad-clip,
    val-split early-stop per group (best-val weights restored). standardize=False ⇒ raw space."""
    torch.manual_seed(seed)
    e_c, Xc = capture_pooled(model, tok, prompts, math.inf, C, batch_size, max_tokens)   # clean pooled + clean X
    N = len(prompts)
    rng = np.random.default_rng(seed)
    shuf = rng.permutation(N)
    nval = min(max(1, int(val_frac * N)), N - 1)   # ≥1 val, always leave ≥1 train (robust for small N)
    val_idx, tr_idx = shuf[:nval], shuf[nval:]
    if standardize:
        mu, sd = e_c[tr_idx].mean(0), e_c[tr_idx].std(0) + 1e-6     # stats from TRAIN split only
    else:
        mu, sd = np.zeros(d, np.float32), np.ones(d, np.float32)
    e_c_s = ((e_c - mu) / sd).astype(np.float32)

    groups, reps = make_eta_groups(train_etas, n_groups)
    unique_etas = sorted({e for g in groups for e in g})
    dens = [Denoiser(d=d, n_layers=n_layers, d_ff=(d_ff or None), dropout=dropout,
                     residual=residual).to(DEV) for _ in groups]
    opts = [torch.optim.AdamW(m.parameters(), lr=lr) for m in dens]
    best, best_state, bad = [math.inf] * len(groups), [None] * len(groups), [0] * len(groups)
    print(f"[snd] groups={[ [int(x) for x in g] for g in groups]} reps={[round(r) for r in reps]} "
          f"layers={n_layers} d_ff={d_ff or d} dropout={dropout} residual={residual} "
          f"loss=MSE+{cos_weight}·(1-cos) train/val={len(tr_idx)}/{len(val_idx)}", flush=True)

    def _epoch_loss(den, etas_caps, idx_set, train, opt=None, lr_now=None):
        """One pass over idx_set across this group's etas; train (with opt) or eval. Returns mean loss."""
        den.train() if train else den.eval()
        if train and lr_now is not None:
            for pg in opt.param_groups:
                pg["lr"] = lr_now
        tot, nb = 0.0, 0
        order = rng.permutation(len(idx_set)) if train else np.arange(len(idx_set))
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for e_n_s, Xt in etas_caps:
                for s in range(0, len(idx_set), batch_size):
                    idx = idx_set[order[s:s + batch_size]]
                    ed = _denoise(den, e_n_s, Xt, Xc, idx, d)
                    loss = _cos_mse_loss(ed, torch.from_numpy(e_c_s[idx]).to(DEV), cos_weight)
                    if train:
                        opt.zero_grad(); loss.backward()
                        torch.nn.utils.clip_grad_norm_(den.parameters(), 1.0); opt.step()
                    tot += loss.item(); nb += 1
        return tot / max(1, nb)

    # capture each train-η's noised set ONCE and reuse across epochs (standard multi-epoch training on
    # a fixed set). Capturing inside the epoch loop re-ran a full model forward per η per epoch — the
    # dominant cost (≈7× slower) for noise augmentation we don't need at this corpus size.
    caps_s = {}
    for eta in unique_etas:
        e_n, Xt = capture_pooled(model, tok, prompts, eta, C, batch_size, max_tokens)
        caps_s[eta] = (((e_n - mu) / sd).astype(np.float32), Xt)

    for ep in range(epochs):
        lr_now = _lr_at(ep, epochs, lr)
        msg = []
        for gi, g in enumerate(groups):
            if bad[gi] > patience:                   # frozen (early-stopped)
                msg.append(f"g{gi}:stopped"); continue
            gcaps = [caps_s[e] for e in g]
            _epoch_loss(dens[gi], gcaps, tr_idx, train=True, opt=opts[gi], lr_now=lr_now)
            vl = _epoch_loss(dens[gi], gcaps, val_idx, train=False)
            if vl < best[gi] - 1e-4:
                best[gi] = vl; bad[gi] = 0
                best_state[gi] = {k: v.detach().clone() for k, v in dens[gi].state_dict().items()}
            else:
                bad[gi] += 1
            msg.append(f"g{gi}:val={vl:.4f}{'*' if bad[gi] == 0 else ''}")
        print(f"[snd] epoch {ep+1}/{epochs} lr={lr_now:.2e} | " + " ".join(msg), flush=True)
        if all(bad[gi] > patience for gi in range(len(groups))):
            print(f"[snd] all groups early-stopped at epoch {ep+1}", flush=True); break

    for gi in range(len(groups)):                    # restore best-val weights
        if best_state[gi] is not None:
            dens[gi].load_state_dict(best_state[gi])
        dens[gi].eval()
    return dens, mu.astype(np.float32), sd.astype(np.float32), reps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--corpus", default="corpora/rep2text-stratified.txt")
    ap.add_argument("--max-prompts", type=int, default=400)
    ap.add_argument("--train-frac", type=float, default=0.5)
    # defaults calibrated for pythia-160m (token-embed C≈0.82, d=768 ⇒ mean noise mag = d/η): the
    # clean→destroyed band is η≈[800,5000]; other models need their own band (mean noise mag = d/η vs C).
    ap.add_argument("--etas", default="inf,5000,3000,2000,1500,1200,1000,800", help="dχ budget sweep; 'inf'=clean")
    # train across the WHOLE eval band incl. the mild-noise end, else the denoiser over-denoises mild
    # inputs and HURTS recovery at high η (it must learn near-identity when noise is small).
    ap.add_argument("--train-etas", default="5000,2000,1200,800", help="η values the denoiser trains on")
    ap.add_argument("--epochs", type=int, default=15, help="max epochs (val early-stop usually ends sooner)")
    # stronger denoiser (paper §A.5.3 + residual readout). See Denoiser docstring.
    ap.add_argument("--n-groups", type=int, default=3, help="per-η-group denoisers (paper §A.5.3); 1 = single noise-aware model")
    ap.add_argument("--denoiser-layers", type=int, default=6, help="transformer layers (paper Table 8: 6)")
    ap.add_argument("--d-ff", type=int, default=0, help="FFN width (0 = d_model, paper-faithful)")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--no-residual", action="store_true", help="raw h_0^L readout (paper) instead of zero-init residual correction")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cos-weight", type=float, default=1.0, help="λ in MSE + λ(1−cos)")
    ap.add_argument("--patience", type=int, default=4, help="val early-stop patience (epochs)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--clip-percentile", type=float, default=99.9)
    ap.add_argument("--no-standardize", action="store_true",
                    help="measure cos/MSE in RAW pooled-embedding space (default standardizes per-dim "
                         "by clean-train stats — LLM hidden states are anisotropic; raw cos is degenerate)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=64, help="truncate prompts to N tokens — bounds the "
                    "denoiser's 2T+1 sequence (attention is O(T²)) so length-stratified corpora don't OOM the iGPU")
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
                model(tok(p, return_tensors="pt", truncation=True, max_length=args.max_tokens
                          ).input_ids.to(DEV), use_cache=False)
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

    dens, mu, sd, reps = train_denoiser(
        model, tok, tr_prompts, train_etas, C, d, args.epochs, args.batch_size, args.seed,
        standardize=not args.no_standardize, max_tokens=args.max_tokens, n_layers=args.denoiser_layers,
        d_ff=args.d_ff, dropout=args.dropout, residual=not args.no_residual, lr=args.lr,
        cos_weight=args.cos_weight, patience=args.patience, val_frac=args.val_frac, n_groups=args.n_groups)

    if DEV == "cuda":
        torch.cuda.empty_cache()
    e_c, Xc = capture_pooled(model, tok, te_prompts, math.inf, C, args.batch_size, args.max_tokens)  # clean test
    e_c_s = ((e_c - mu) / sd).astype(np.float32)
    # generation utility (standardized talens.probes.utility): clean baseline = unhooked pass
    clean_pass = teacher_forced_pass(model, tok, te_prompts, hook=None,
                                     max_tokens=args.max_tokens, batch_size=args.batch_size)
    records = []
    for eta in etas:
        gi = route_eta(None if math.isinf(eta) else eta, reps)     # paper §A.5.3 routing
        den = dens[gi]
        e_n, Xt = capture_pooled(model, tok, te_prompts, eta, C, args.batch_size, args.max_tokens)
        e_n_s = ((e_n - mu) / sd).astype(np.float32)        # standardized space (cos/MSE + denoiser io)
        e_d_s = np.zeros_like(e_n_s)
        with torch.no_grad():
            for s in range(0, len(te_prompts), args.batch_size):
                idx = np.arange(s, min(s + args.batch_size, len(te_prompts)))
                e_d_s[idx] = _denoise(den, e_n_s, Xt, Xc, idx, d).cpu().numpy()
        rec = {"eta": (None if math.isinf(eta) else eta), "group": gi}
        rec.update(embedding_recovery(e_c_s, e_n_s, e_d_s))                 # embedding-utility probe
        dp = clean_pass if math.isinf(eta) else teacher_forced_pass(
            model, tok, te_prompts, hook=DxPrivacy(C, eta), max_tokens=args.max_tokens, batch_size=args.batch_size)
        acc_r, ppl_r, agr_r = (next_token_accuracy(dp, clean_pass),       # generation-utility probes
                               perplexity(dp, clean_pass), output_agreement(dp, clean_pass))
        rec.update({"perplexity": ppl_r.defended, "acc": acc_r.defended,
                    "ppl_degradation": ppl_r.extra["degradation"], "retention_acc": acc_r.retention,
                    "agree_clean": agr_r.defended,
                    "utility": {r.metric: r.as_dict() for r in (acc_r, ppl_r, agr_r)}})
        records.append(rec)
        es = "inf" if math.isinf(eta) else f"{eta:g}"
        print(f"[snd] η={es:>5} g{gi} | cos {rec['cos_noised']:.3f}->{rec['cos_denoised']:.3f} "
              f"rec_cos={rec['recovery_cos']:.3f} rec_mse={rec['recovery_mse']:.3f} "
              f"ret_acc={acc_r.retention:.3f} ppl_deg={rec['ppl_degradation']:+.2%}", flush=True)

    noised = [r for r in records if r["eta"] is not None]
    eta_thresholds = retention_thresholds([r["eta"] for r in noised], [r["retention_acc"] for r in noised])
    print(f"[snd] η for −10/−20/−50% acc-retention: {eta_thresholds}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "model": args.model, "corpus": args.corpus, "n_test": len(te_prompts), "hidden": d,
        "defense": "snd_dx", "clip_C": C, "clip_percentile": args.clip_percentile,
        "eta_thresholds": eta_thresholds,
        "standardized": not args.no_standardize,
        "standardize_note": "cos/MSE measured on pooled embeddings standardized per-dim by clean-TRAIN "
                            "(μ,σ); the denoiser is trained against the standardized target. Corrects "
                            "LLM anisotropy so cos/MSE reflect content, not outlier-dim magnitude.",
        "budget_note": "eta is the dχ-privacy budget (larger=weaker privacy); NOT the Gaussian ε "
                       "of dp_leakage_sweep — the two are not interchangeable.",
        "etas": [None if math.isinf(e) else e for e in etas], "train_etas": train_etas,
        "epochs": args.epochs, "seed": args.seed,
        "denoiser": {"n_groups": args.n_groups, "group_reps": [round(r, 1) for r in reps],
                     "layers": args.denoiser_layers, "d_ff": args.d_ff or d, "dropout": args.dropout,
                     "residual": not args.no_residual, "lr": args.lr, "cos_weight": args.cos_weight,
                     "patience": args.patience, "val_frac": args.val_frac},
        "denoiser_note": "STRONGER denoiser: per-η-group models (paper §A.5.3, route by η→nearest group "
                         "rep in log-space; clean→mildest group) + zero-init RESIDUAL readout "
                         "(e_d=e_n+head, starts as passthrough → recovery≥0, clean ceiling≈1) + "
                         "MSE+λ(1−cos) loss + warmup/cosine LR + grad-clip + val early-stop.",
        "utility_note": "generation utility via standardized talens.probes.utility (retention∈[0,1], "
                        "1=lossless): next_token_accuracy + perplexity + output_agreement vs the clean "
                        "(unhooked) pass — per-record 'utility' block; comparable across schemes via 'retention'.",
        "readout_note": "recovery_cos = fraction of (1-cos) gap closed by the denoiser; recovery_mse "
                        "= fraction of noised MSE removed (embedding-utility probe); ppl/acc degradation "
                        "= generation-utility cost of the dχ noise (denoiser does not touch logits). "
                        "'group' = routed denoiser.",
        "records": records,
    }, indent=2))
    print(f"[snd] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
