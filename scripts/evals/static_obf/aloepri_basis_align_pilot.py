#!/usr/bin/env python3
"""L0 pilot for the harvest-aligned basis-recovery attack (claim:aloepri-kqvout-basis-alignment).

Tests the prediction: on kqv_out under Alg2, a keyless self-generated inverter collapses (§05), but
spending a K-type TFMA harvest on estimating Alg2's secret orthogonal rotation R (NOT on the token map)
lifts held-out recovery from the floor toward the invariant-surface ceiling as K grows.

Setup (L0, kqv_out): capture plaintext (Xp) + alg2 (Xd) + keymat (control). Fix a held-out type set
(never harvested); sweep K = # harvested types (from the rest). Per K:
  R̂       = orthogonal Procrustes on the harvested tokens' aligned (Xp, Xd) pairs
  self-gen = ridge fit on the attacker's own plaintext reps (train-prompt rows)
  basis_align recovery = decode (Xd_held · R̂ᵀ) with the self-gen inverter  ← the attack
  naive recovery       = decode  Xd_held        with the self-gen inverter  ← §05 floor (no un-rotation)
Held-out test set is FIXED across K, so recovery-vs-K is clean. Alignment set = all harvested-token rows
(OPTIMISTIC — the faithful attacker aligns only on fully-known-prefix positions; flagged).
keymat control: R ≈ I, so basis_align ≈ naive ≈ invariant level, K-independent.

GPU: ONE process; host .venv.
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
from evals.static_obf.aloepri_matched_vs_selfgen import capture  # noqa: E402
from evals.static_obf.aloepri_score_surface_sweep import _load  # noqa: E402
from talens.attacks import ridge_W, nearest_token, orthogonal_procrustes_R  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# alg2_0 = Alg2 head rotation with NO αₑ noise (isolates the pure orthogonal rotation the attack
# recovers); alg2 = alg2@1.0 (rotation + noise; noise caps recovery independent of R̂).
CFG = {
    "plaintext": None,
    "keymat": dict(config="keymat_only"),
    "alg2_0": dict(config="alg2", alpha_e=0.0, alpha_h=0.0),
    "alg2": dict(config="alg2", alpha_e=1.0, alpha_h=0.0),
}


def _rel_err(P, D, R):
    return float(np.linalg.norm(P @ R - D) / max(np.linalg.norm(D), 1e-9))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--corpus", default="corpora/release-gate-512.txt")
    ap.add_argument("--max-prompts", type=int, default=160)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--ks", default="0,50,100,300,700")
    ap.add_argument("--held-frac", type=float, default=0.3, help="fraction of types fixed as held-out test set")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--pool-size", type=int, default=2048)
    ap.add_argument("--keymat-h", type=int, default=128)
    ap.add_argument("--keymat-lam", type=float, default=0.3)
    ap.add_argument("--keymat-seed", type=int, default=0)
    ap.add_argument("--out", default="refine-logs/matched-invariance/basis_align_pilot.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompts = [l.strip() for l in Path(args.corpus).read_text().splitlines() if l.strip()][: args.max_prompts]
    L, ks, seeds = args.layer, [int(s) for s in args.ks.split(",")], [int(s) for s in args.seeds.split(",")]
    print(f"[basis-align] model={args.model} L{L} ks={ks} held_frac={args.held_frac} seeds={seeds} dev={DEV}", flush=True)

    caps, ids0, pidx0, table, vocab = {}, None, None, None, None
    for cname in ("plaintext", "keymat", "alg2_0", "alg2"):
        m = _load(args.model)
        if table is None:
            table = m.get_input_embeddings().weight.detach().float().cpu().numpy().astype(np.float32)
            vocab = table.shape[0]
        rkw = CFG[cname]
        if rkw is not None:
            reparam_pythia(m, h=args.keymat_h, lam=args.keymat_lam, seed=args.keymat_seed, **rkw)
        feats, ids, pidx = capture(m, tok, prompts, [L], args.window); del m
        if ids0 is None:
            ids0, pidx0 = ids, pidx
        else:
            assert np.array_equal(ids, ids0), f"ids drifted under {cname}"
        caps[cname] = feats["kqv_out"][L]
        print(f"[basis-align] captured {cname:>9} kqv_out dim={caps[cname].shape[1]} rows={ids.shape[0]}", flush=True)

    y = ids0
    Xp = caps["plaintext"]
    n_prompts = int(pidx0.max()) + 1
    types, counts = np.unique(y, return_counts=True)
    freq_order = types[np.argsort(counts)[::-1]]

    records = []
    for cfg in ("alg2_0", "alg2", "keymat"):
        Xd = caps[cfg]
        for seed in seeds:
            rng = np.random.default_rng(seed)
            # fixed held-out test types (never harvested); harvestable = the rest, ranked by frequency
            perm_types = rng.permutation(types)
            n_held = int(args.held_frac * types.size)
            held_types = set(perm_types[:n_held].tolist())
            harvestable = [t for t in freq_order if t not in held_types]        # freq-ranked
            # self-gen inverter: attacker's own plaintext reps on train-prompts (covers all types)
            tr_p = set(rng.permutation(n_prompts)[: n_prompts // 2].tolist())
            tr_rows = np.where(np.isin(pidx0, list(tr_p)))[0]
            W = ridge_W(Xp[tr_rows], table[y[tr_rows]])
            # fixed test rows = victim (test-prompt) rows whose token is a held-out type
            te_rows = np.where(~np.isin(pidx0, list(tr_p)) & np.isin(y, list(held_types)))[0]
            true_pool = np.unique(y[te_rows])
            fill = rng.choice(np.setdiff1d(np.arange(vocab), true_pool),
                              size=max(0, args.pool_size - true_pool.size), replace=False)
            pool = np.concatenate([true_pool, fill.astype(np.int64)])
            naive = float((nearest_token(Xd[te_rows] @ W, table[pool], pool) == y[te_rows]).mean())
            for k in ks:
                harv = set(harvestable[:k])
                align = np.where(np.isin(y, list(harv)))[0] if k > 0 else np.array([], int)
                if align.size >= Xp.shape[1]:                                   # need ≥ d aligned samples for O(d)
                    R = orthogonal_procrustes_R(Xp[align], Xd[align])
                    ba = float((nearest_token((Xd[te_rows] @ R.T) @ W, table[pool], pool) == y[te_rows]).mean())
                    rerr = _rel_err(Xp[te_rows], Xd[te_rows], R)
                else:
                    ba, rerr = naive, None                                      # too few pairs → no alignment
                records.append({"config": cfg, "seed": seed, "k": k, "n_align": int(align.size),
                                "n_test": int(te_rows.size), "basis_align": ba, "naive": naive, "R_rel_err": rerr})

    def agg(cfg, k, key):
        v = [r[key] for r in records if r["config"] == cfg and r["k"] == k and r[key] is not None]
        return (float(np.mean(v)), float(np.std(v))) if v else (None, None)

    print("\n[basis-align] held-out recovery vs K (mean±sd over seeds):", flush=True)
    for cfg in ("alg2_0", "alg2", "keymat"):
        print(f"  --- {cfg} (naive/self-gen floor = {agg(cfg, ks[0], 'naive')[0]:.3f}) ---", flush=True)
        for k in ks:
            ba, sd = agg(cfg, k, "basis_align"); re_, _ = agg(cfg, k, "R_rel_err")
            na_ = agg(cfg, k, "n_align")[0]
            print(f"    K={k:>4} n_align={na_:>5.0f} | basis_align={ba:.3f}±{sd:.3f} "
                  f"R_rel_err={'NA' if re_ is None else round(re_,3)}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "model": args.model, "layer": L, "ks": ks, "held_frac": args.held_frac, "seeds": seeds,
        "note": "L0 pilot, harvest-aligned basis recovery on kqv_out. basis_align = decode (Xd·R̂ᵀ) with a "
                "self-gen ridge; R̂ = orthogonal Procrustes on harvested-token aligned pairs (OPTIMISTIC: all "
                "harvested-token rows, not just fully-known-prefix). naive = decode Xd without un-rotation "
                "(§05 floor). Fixed held-out test types across K. keymat = control (R≈I).",
        "records": records,
    }, indent=2))
    print(f"\n[basis-align] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
