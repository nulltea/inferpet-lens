#!/usr/bin/env python3
"""B2+ — firm up the permutation-Π channel (auto-review fix #4).

B2 found CLUB-on-φ tracks VMA τ-recovery at ρ +0.976, but over only 7 α points,
1 seed, 1 model — and B4 exposed high seed variance in the pooled Π ρ. This
firms it up (GPU-free, weight surface):

* **5 seeds** (perm + noise draws) × **12 α_e** densifying the 0.2–0.7 transition.
* **bootstrap 95% CI** for ρ(CLUB-on-φ, τ-recovery), pooled over seeds.
* **match-mode independence**: CLUB-on-φ is computed once (no assignment); it must
  track VMA under BOTH match="hungarian" and match="nn" → it is independent of the
  attack's assignment algorithm, not a reparameterisation of it.
* **second model width**: Qwen3-4B embedding (d=2560) vs gemma-2-2b (d=2304).

Promotes the Π channel from a go/no-go to a channel claim. Run on host .venv (CPU).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from defenses.aloepri import obfuscate_embedding_table  # noqa: E402

from talens.weights import vma  # noqa: E402
from talens.weights.measures import club_mi_weights, v_information_weights  # noqa: E402

MODELS = {
    "gemma-2-2b": "results/capture_cache/embed-b0c6566474cadb27.pt",
    "qwen3-4b": "results/capture_cache/embed-92e31154209b70d8.pt",
}


def _spear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return 0.0 if np.std(a) < 1e-12 or np.std(b) < 1e-12 else float(stats.spearmanr(a, b).statistic)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="gemma-2-2b,qwen3-4b")
    ap.add_argument("--n-tokens", type=int, default=1000)
    ap.add_argument("--alphas", default="0,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.75,1.0")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--bins", type=int, default=64)
    ap.add_argument("--club-steps", type=int, default=200)
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--out", default="results/b2plus_pi_firmup.json")
    args = ap.parse_args()

    alphas = [float(s) for s in args.alphas.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    def club(pair):
        return club_mi_weights(pair, bins=args.bins, steps=args.club_steps, hidden_size=128, seed=0)["club_mi_bits"]

    def retr(pair, N):
        n_tr = max(64, int(0.66 * N)); n_va = int(0.17 * N)
        return v_information_weights(pair, bins=args.bins, n_train=n_tr, n_val=n_va,
                                     n_test=N - n_tr - n_va, candidate_pool_size=N)["v_information_bits"]

    summary = {}
    all_records = []
    for model in models:
        table = torch.load(MODELS[model], map_location="cpu", weights_only=False).float().numpy().astype(np.float32)
        vocab, d = table.shape
        print(f"\n[b2+] {model}: embed {vocab}x{d}", flush=True)
        # per-seed ρ over the α grid; pooled point cloud for bootstrap
        per_seed_club, per_seed_retr, per_seed_nn = [], [], []
        pool_club, pool_vma_h, pool_vma_n, pool_retr = [], [], [], []
        for seed in seeds:
            rng = np.random.default_rng(20260621 + seed)
            W = table[rng.choice(vocab, size=args.n_tokens, replace=False)].copy()
            cl, vh, vn, rt = [], [], [], []
            for a in alphas:
                pair = obfuscate_embedding_table(W, alpha_e=a, keymat=False, seed=20260621 + seed)
                c = club(pair); r = retr(pair, args.n_tokens)
                vh.append(vma.run(pair, bins=args.bins, match="hungarian").ttrsr_top1)
                vn.append(vma.run(pair, bins=args.bins, match="nn").ttrsr_top1)
                cl.append(c); rt.append(r)
                all_records.append({"model": model, "seed": seed, "alpha_e": a,
                                    "club_bits": c, "retr_pvi_bits": r,
                                    "vma_hungarian": vh[-1], "vma_nn": vn[-1]})
            per_seed_club.append(_spear(cl, vh)); per_seed_retr.append(_spear(rt, vh))
            per_seed_nn.append(_spear(cl, vn))
            pool_club += cl; pool_vma_h += vh; pool_vma_n += vn; pool_retr += rt
            print(f"[b2+] {model} s{seed}: ρ(CLUB,VMA-h)={per_seed_club[-1]:+.3f} "
                  f"ρ(CLUB,VMA-nn)={per_seed_nn[-1]:+.3f} ρ(retr,VMA-h)={per_seed_retr[-1]:+.3f}", flush=True)

        # pooled bootstrap CI for ρ(CLUB, VMA-hungarian)
        pc, pv = np.array(pool_club), np.array(pool_vma_h)
        rng = np.random.default_rng(0); n = pc.size
        boots = [_spear(pc[idx], pv[idx]) for idx in (rng.integers(0, n, n) for _ in range(args.boot))]
        summary[model] = {
            "d": int(d),
            "rho_club_vma_hungarian": {"per_seed_mean": float(np.mean(per_seed_club)),
                                       "per_seed_std": float(np.std(per_seed_club)),
                                       "per_seed_min": float(np.min(per_seed_club)),
                                       "boot_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                                       "pooled": _spear(pool_club, pool_vma_h)},
            "rho_club_vma_nn_mean": float(np.mean(per_seed_nn)),
            "rho_retr_vma_mean": float(np.mean(per_seed_retr)),
        }
        s = summary[model]["rho_club_vma_hungarian"]
        print(f"[b2+] {model} SUMMARY: ρ(CLUB,VMA) per-seed {s['per_seed_mean']:+.3f}±{s['per_seed_std']:.3f} "
              f"(min {s['per_seed_min']:+.3f}); pooled {s['pooled']:+.3f} "
              f"boot95 [{s['boot_ci95'][0]:+.3f},{s['boot_ci95'][1]:+.3f}]; "
              f"match-indep ρ(CLUB,VMA-nn)={summary[model]['rho_club_vma_nn_mean']:+.3f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"alphas": alphas, "seeds": seeds,
                              "summary": summary, "records": all_records}, indent=2))
    print(f"\n[b2+] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
