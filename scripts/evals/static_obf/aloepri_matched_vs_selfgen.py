#!/usr/bin/env python3
"""Matched vs Self-generated inversion across (surface × config) — proves "the matched-ridge inverter
(FIG·02) is a realistic/keyless attack ONLY on a surface the obfuscation leaves INVARIANT".
Plan: refine-logs/matched-invariance/EXPERIMENT_PLAN.md, Block 1.

Per (surface × config × seed), split by PROMPT — attacker's own prompts = TRAIN, victim prompts = TEST:
  matched  : ridge(X_dep[train] → emb[y]),  decode X_dep[test]   (ceiling / FIG·02; needs deployment-basis labels)
  self_gen : ridge(X_plain[train] → emb[y]), decode X_dep[test]  (REALISTIC: trains on reps the attacker makes
             itself by running the PUBLIC plaintext model on its OWN prompts — no key, no permutation; valid
             against the deployment ONLY if the surface is invariant)
  floor    : shuffled-label ridge.
Same candidate pool for matched & self_gen within a cell, so the matched−self_gen gap is NOT a pool effect.
self_gen_inset = self_gen map decoded on its OWN (plaintext) test reps → isolates "training works" from
"basis transfers" (rules out the undertraining anti-claim).

Claim proven iff: self_gen ≈ matched on invariant cells, self_gen → floor on rotated cells.
Surfaces: residual (ISA-HiddenState), kqv_out (ISA-AttnValue), kq (ISA-AttnScore).
Configs: plaintext (control), keymat, alg1@1.0, alg2@1.0.

keymat/alg1 WIDEN the residual to d+2h (basis change P̂), so a plaintext-trained residual inverter is
dimension-incompatible with the deployment residual — self_gen is reported "incompat" (the starkest
"not invariant"). kqv_out/kq keep dim d (head axis), so self_gen is always dimensionally applicable and
only the basis differs (the clean same-dim discriminator: invariant under keymat/alg1, rotated under alg2).

GPU: ONE process; runs in the host .venv (gfx1151).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from defenses.aloepri import reparam_pythia  # noqa: E402
from evals.static_obf.aloepri_score_surface_sweep import _load, _CAP  # noqa: E402
from talens.attacks import ridge_W, nearest_token  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"

CONFIGS = {  # name -> reparam kwargs (None = plaintext control)
    "plaintext": None,
    "keymat": dict(config="keymat_only"),
    "alg1": dict(config="full_alg1", alpha_e=1.0, alpha_h=0.2),
    "alg2": dict(config="alg2", alpha_e=1.0, alpha_h=0.0),
}
SURFACES = ("residual", "kqv_out", "kq")


@torch.no_grad()
def capture(model, tok, prompts, layers, window):
    """feats[surface][L]=(rows,dim) for residual|kqv_out|kq, + token ids + per-row prompt index."""
    feats = {s: {L: [] for L in layers} for s in SURFACES}
    ids, pidx = [], []
    for pi, p in enumerate(prompts):
        input_ids = tok(p, return_tensors="pt").input_ids.to(DEV)
        _CAP.clear()
        out = model(input_ids, use_cache=False, output_hidden_states=True)
        toks = input_ids[0].cpu().numpy()
        q = toks.shape[0]
        for L in layers:
            feats["residual"][L].append(out.hidden_states[L + 1][0].float().cpu().numpy().astype(np.float32))
            feats["kqv_out"][L].append(_CAP[("kqv_out", L)].astype(np.float32))
            sc = _CAP[("kq", L)]                                   # (H, q, kv) pre-softmax
            H = sc.shape[0]
            w = np.zeros((q, H, window), np.float32)
            for pos in range(q):
                k = min(window, pos + 1)
                w[pos, :, :k] = sc[:, pos, pos + 1 - k:pos + 1]    # most-recent causal keys, right-aligned
            feats["kq"][L].append(w.reshape(q, H * window))
        ids.append(toks)
        pidx.append(np.full(q, pi))
    ids = np.concatenate(ids).astype(np.int64)
    pidx = np.concatenate(pidx)
    return ({s: {L: np.concatenate(feats[s][L]).astype(np.float32) for L in layers} for s in feats},
            ids, pidx)


def _recover(Xtr, ytr, Xte, yte, emb, pool):
    """ridge Xtr→emb[ytr], decode Xte by nearest-token over `pool`, return top-1 recovery vs yte."""
    W = ridge_W(Xtr, emb[ytr])
    return float((nearest_token(Xte @ W, emb[pool], pool) == yte).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--corpus", default="corpora/release-gate-512.txt")
    ap.add_argument("--max-prompts", type=int, default=160)
    ap.add_argument("--layers", default="0")
    ap.add_argument("--configs", default="plaintext,keymat,alg1,alg2")
    ap.add_argument("--surfaces", default="residual,kqv_out,kq")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--pool-size", type=int, default=2048)
    ap.add_argument("--keymat-h", type=int, default=128)
    ap.add_argument("--keymat-lam", type=float, default=0.3)
    ap.add_argument("--keymat-seed", type=int, default=0)
    ap.add_argument("--out", default="refine-logs/matched-invariance/matched_vs_selfgen.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompts = [l.strip() for l in Path(args.corpus).read_text().splitlines() if l.strip()][: args.max_prompts]
    layers = [int(s) for s in args.layers.split(",") if s.strip()]
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    surfaces = [s.strip() for s in args.surfaces.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    print(f"[mvs] model={args.model} layers={layers} configs={configs} surfaces={surfaces} "
          f"seeds={seeds} prompts={len(prompts)} dev={DEV}", flush=True)

    # capture every config (shared prompts → shared token ids). plaintext first → its embedding table.
    caps, ids0, pidx0, emb, vocab = {}, None, None, None, None
    for cname in configs:
        m = _load(args.model)
        if emb is None:
            emb = m.get_input_embeddings().weight.detach().float().cpu().numpy().astype(np.float32)
            vocab = emb.shape[0]
        rkw = CONFIGS[cname]
        if rkw is not None:
            reparam_pythia(m, h=args.keymat_h, lam=args.keymat_lam, seed=args.keymat_seed, **rkw)
        feats, ids, pidx = capture(m, tok, prompts, layers, args.window)
        del m
        if ids0 is None:
            ids0, pidx0 = ids, pidx
        else:
            assert np.array_equal(ids, ids0), f"token ids drifted under {cname}"
        caps[cname] = feats
        dims = {s: feats[s][layers[0]].shape[1] for s in surfaces}
        print(f"[mvs] captured {cname:>10} | dims={dims} rows={ids.shape[0]}", flush=True)

    y = ids0
    n_prompts = int(pidx0.max()) + 1
    records = []
    for cname in configs:
        for L in layers:
            for surf in surfaces:
                Xd = caps[cname][surf][L]
                Xp = caps["plaintext"][surf][L]
                incompat = Xp.shape[1] != Xd.shape[1]
                for seed in seeds:
                    rng = np.random.default_rng(seed)
                    perm = rng.permutation(n_prompts)
                    tr_p = set(perm[: n_prompts // 2].tolist())     # attacker's own prompts
                    tr = np.where(np.isin(pidx0, list(tr_p)))[0]
                    te = np.where(~np.isin(pidx0, list(tr_p)))[0]
                    true_pool = np.unique(y[te])
                    avail = np.setdiff1d(np.arange(vocab, dtype=np.int64), true_pool)
                    fill = rng.choice(avail, size=max(0, args.pool_size - true_pool.size), replace=False)
                    pool = np.concatenate([true_pool, fill.astype(np.int64)])
                    yshuf = y[tr].copy()
                    rng.shuffle(yshuf)

                    matched = _recover(Xd[tr], y[tr], Xd[te], y[te], emb, pool)
                    floor = _recover(Xd[tr], yshuf, Xd[te], y[te], emb, pool)
                    if incompat:
                        self_gen, self_inset = None, None             # plaintext-basis map cannot apply
                    else:
                        W = ridge_W(Xp[tr], emb[y[tr]])               # fit on the attacker's PLAINTEXT reps
                        self_gen = float((nearest_token(Xd[te] @ W, emb[pool], pool) == y[te]).mean())
                        self_inset = float((nearest_token(Xp[te] @ W, emb[pool], pool) == y[te]).mean())
                    records.append({
                        "config": cname, "layer": L, "surface": surf, "seed": seed,
                        "dim_plain": int(Xp.shape[1]), "dim_dep": int(Xd.shape[1]), "incompat": incompat,
                        "matched": matched, "self_gen": self_gen, "self_inset": self_inset, "floor": floor,
                        "gap": (None if self_gen is None else matched - self_gen),
                        "n_test": int(te.size), "pool": int(pool.size),
                    })

    # aggregate over seeds for the printout / claim verdict
    def _agg(cname, surf, key):
        vals = [r[key] for r in records if r["config"] == cname and r["surface"] == surf and r[key] is not None]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)

    print("\n[mvs] surface × config — matched / self_gen / floor (mean over seeds):", flush=True)
    for surf in surfaces:
        for cname in configs:
            m_, _ = _agg(cname, surf, "matched")
            s_, ssd = _agg(cname, surf, "self_gen")
            f_, _ = _agg(cname, surf, "floor")
            sg = "incompat" if s_ is None else f"{s_:.3f}±{ssd:.3f}"
            print(f"[mvs]   {surf:>9} {cname:>10} | matched={m_:.3f} self_gen={sg} floor={f_:.3f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "model": args.model, "corpus": args.corpus, "n_prompts": len(prompts),
        "layers": layers, "configs": configs, "surfaces": surfaces, "seeds": seeds,
        "split": "prompt-rowsplit (attacker prompts=train, victim prompts=test)",
        "keymat": {"h": args.keymat_h, "lam": args.keymat_lam, "seed": args.keymat_seed},
        "note": "matched=train on deployment reps+true labels (ceiling); self_gen=train on PLAINTEXT-model "
                "reps+true labels, decode deployment reps (realistic keyless); floor=shuffled labels. Same "
                "pool for matched/self_gen per cell. self_inset=self_gen map on its own plaintext test reps "
                "(training-validity control). incompat=plaintext-basis map dim-incompatible with deployment "
                "(keymat/alg1 widen residual to d+2h).",
        "records": records,
    }, indent=2))
    print(f"\n[mvs] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
