#!/usr/bin/env python3
"""HONEST-alignment redo of the basis-alignment L0 pilot (claim:aloepri-kqvout-basis-alignment).

The pilot aligned R̂ on ALL harvested-token rows (optimistic — assumes the attacker can compute the
plaintext rep at any harvested-token position). The FAITHFUL attacker can only form an aligned pair at a
position whose ENTIRE causal prefix is harvested (else it can't reconstruct the input to run the public
model). This script:
  1. counts honest aligned rows = per victim prompt, the LEADING run of harvested-type tokens, vs K;
  2. redoes alg2@0 (pure rotation) recovery using ONLY those honest pairs, with
     - GLOBAL orthogonal Procrustes over O(768)  (needs n_align ≥ 768), and
     - PER-HEAD block Procrustes: recover the head-permutation (min-residual assignment) + per-head
       O(64) rotation (needs only ~64 honest rows — exploits R = perm · blkdiag(Û_vo)).
Alignment pairs come only from victim (test-prompt) traffic; self-gen inverter trains on the attacker's
own (train-prompt) plaintext reps. keymat = control (R = I). GPU: ONE process; host .venv.
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
from talens.attacks.dp_inversion import ridge_W, nearest_token, orthogonal_procrustes_R  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CFG = {"plaintext": None, "keymat": dict(config="keymat_only"),
       "alg2_0": dict(config="alg2", alpha_e=0.0, alpha_h=0.0)}


def leading_harvested_mask(y, pidx, harv, prompt_ok):
    """rows usable for HONEST alignment: within each victim prompt (prompt_ok[p]), the leading run of
    harvested-type tokens (a position is usable iff every token at positions 0..i in its prompt is harvested)."""
    H = np.isin(y, list(harv))
    mask = np.zeros(len(y), bool)
    for p in np.unique(pidx):
        if not prompt_ok[p]:
            continue
        idx = np.where(pidx == p)[0]                         # position order within the prompt
        lead = np.cumprod(H[idx].astype(np.int64)).astype(bool)
        mask[idx] = lead
    return np.where(mask)[0]


def blockwise_procrustes_R(Xp, Xd, n_heads=12, hd=64):
    """R = per-head O(hd) rotation composed with a head-permutation, estimated block-wise:
    per (plaintext head h, deployment head h') fit O(hd) Procrustes + residual; assign the permutation by
    min total residual (Hungarian if scipy, else greedy); assemble the block-permuted-orthogonal R (P·R≈D).
    Needs only ~hd honest rows (each row supplies all heads' hd coords)."""
    P = np.asarray(Xp, np.float64).reshape(-1, n_heads, hd)
    D = np.asarray(Xd, np.float64).reshape(-1, n_heads, hd)
    res = np.zeros((n_heads, n_heads)); Q = {}
    for h in range(n_heads):
        for hp in range(n_heads):
            U, _, Vt = np.linalg.svd(P[:, h].T @ D[:, hp], full_matrices=False)
            Q[(h, hp)] = U @ Vt
            res[h, hp] = np.linalg.norm(P[:, h] @ Q[(h, hp)] - D[:, hp])
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(res)
    except Exception:                                        # greedy fallback (12 heads)
        rows, cols, used = list(range(n_heads)), [], set()
        for h in range(n_heads):
            order = np.argsort(res[h])
            hp = next(c for c in order if c not in used); used.add(hp); cols.append(hp)
    R = np.zeros((n_heads * hd, n_heads * hd), np.float32)
    for h, hp in zip(rows, cols):
        R[h * hd:(h + 1) * hd, hp * hd:(hp + 1) * hd] = Q[(h, hp)].astype(np.float32)
    return R


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--corpus", default="corpora/release-gate-512.txt")
    ap.add_argument("--max-prompts", type=int, default=160)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--ks", default="0,50,100,300,700")
    ap.add_argument("--held-frac", type=float, default=0.3)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--pool-size", type=int, default=2048)
    ap.add_argument("--keymat-h", type=int, default=128)
    ap.add_argument("--keymat-lam", type=float, default=0.3)
    ap.add_argument("--keymat-seed", type=int, default=0)
    ap.add_argument("--out", default="refine-logs/matched-invariance/basis_align_honest.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompts = [l.strip() for l in Path(args.corpus).read_text().splitlines() if l.strip()][: args.max_prompts]
    L, ks, seeds = args.layer, [int(s) for s in args.ks.split(",")], [int(s) for s in args.seeds.split(",")]
    print(f"[honest] L{L} ks={ks} held_frac={args.held_frac} seeds={seeds} dev={DEV}", flush=True)

    caps, ids0, pidx0, table, vocab = {}, None, None, None, None
    for cname in ("plaintext", "keymat", "alg2_0"):
        m = _load(args.model)
        if table is None:
            table = m.get_input_embeddings().weight.detach().float().cpu().numpy().astype(np.float32)
            vocab = table.shape[0]
        if CFG[cname] is not None:
            reparam_pythia(m, h=args.keymat_h, lam=args.keymat_lam, seed=args.keymat_seed, **CFG[cname])
        feats, ids, pidx = capture(m, tok, prompts, [L], args.window); del m
        if ids0 is None:
            ids0, pidx0 = ids, pidx
        else:
            assert np.array_equal(ids, ids0), f"ids drifted under {cname}"
        caps[cname] = feats["kqv_out"][L]
        print(f"[honest] captured {cname:>9} rows={ids.shape[0]}", flush=True)

    y, Xp = ids0, caps["plaintext"]
    n_prompts = int(pidx0.max()) + 1
    types, counts = np.unique(y, return_counts=True)
    freq_order = types[np.argsort(counts)[::-1]]

    records = []
    for cfg in ("alg2_0", "keymat"):
        Xd = caps[cfg]
        for seed in seeds:
            rng = np.random.default_rng(seed)
            held = set(rng.permutation(types)[: int(args.held_frac * types.size)].tolist())
            harvestable = [t for t in freq_order if t not in held]
            tr_p = set(rng.permutation(n_prompts)[: n_prompts // 2].tolist())     # attacker's own prompts
            is_test_p = np.array([p not in tr_p for p in range(n_prompts)])       # victim prompts
            tr_rows = np.where(np.isin(pidx0, list(tr_p)))[0]
            W = ridge_W(Xp[tr_rows], table[y[tr_rows]])                           # self-gen inverter
            te_rows = np.where(~np.isin(pidx0, list(tr_p)) & np.isin(y, list(held)))[0]
            true_pool = np.unique(y[te_rows])
            fill = rng.choice(np.setdiff1d(np.arange(vocab), true_pool),
                              size=max(0, args.pool_size - true_pool.size), replace=False)
            pool = np.concatenate([true_pool, fill.astype(np.int64)])
            for k in ks:
                harv = set(harvestable[:k])
                # optimistic (pilot): all harvested-token victim rows
                opt = np.where(np.isin(pidx0, list(set(range(n_prompts)) - tr_p)) & np.isin(y, list(harv)))[0]
                # honest: leading fully-harvested-prefix victim rows
                hon = leading_harvested_mask(y, pidx0, harv, is_test_p) if k > 0 else np.array([], int)
                rec = {"config": cfg, "seed": seed, "k": k, "n_align_opt": int(opt.size),
                       "n_align_honest": int(hon.size), "n_test": int(te_rows.size)}
                for tag, al in (("global", hon), ("block", hon)):
                    if al.size < (768 if tag == "global" else 64):
                        rec[f"ba_{tag}"] = None                                   # too few honest pairs
                        continue
                    R = orthogonal_procrustes_R(Xp[al], Xd[al]) if tag == "global" else blockwise_procrustes_R(Xp[al], Xd[al])
                    rec[f"ba_{tag}"] = float((nearest_token((Xd[te_rows] @ R.T) @ W, table[pool], pool) == y[te_rows]).mean())
                records.append(rec)

    def agg(cfg, k, key):
        v = [r[key] for r in records if r["config"] == cfg and r["k"] == k and r.get(key) is not None]
        return float(np.mean(v)) if v else None

    print("\n[honest] alg2_0 — honest aligned rows and recovery vs K (mean over seeds):", flush=True)
    for cfg in ("alg2_0", "keymat"):
        print(f"  --- {cfg} ---", flush=True)
        for k in ks:
            no, nh = agg(cfg, k, "n_align_opt"), agg(cfg, k, "n_align_honest")
            g, b = agg(cfg, k, "ba_global"), agg(cfg, k, "ba_block")
            fmt = lambda x: "NA(<min)" if x is None else f"{x:.3f}"
            print(f"    K={k:>4} | n_align honest={nh:>6.0f} (opt {no:>5.0f}) | "
                  f"global={fmt(g)} block={fmt(b)}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"model": args.model, "layer": L, "ks": ks, "seeds": seeds,
        "note": "HONEST alignment = leading fully-harvested-prefix victim positions only. global = O(768) "
                "Procrustes (needs n_align≥768); block = per-head O(64)+head-perm (needs ~64). opt = pilot's "
                "optimistic all-harvested-token count for reference.", "records": records}, indent=2))
    print(f"\n[honest] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
