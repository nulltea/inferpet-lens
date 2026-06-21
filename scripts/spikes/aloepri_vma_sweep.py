#!/usr/bin/env python3
"""B2 — permutation-Π channel: AloePri α_e sweep + the Π-probe bake-off.

GPU-FREE (pure weight-surface table algebra). Loads a real embedding table, takes
a subset of N token rows, and walks the AloePri noise knob α_e over the
permutation-core regime (the VMA-vulnerable one). At each α_e it scores:

* **VMA τ-recovery** (the attack, `weights.vma`) — the ground-truth metric.
* **CLUB I(obf-φ ; plain-φ)** (`weights.measures.club_mi_weights`) — the
  *independent* Π-probe candidate (an MI estimator on the paired sorted-quantile
  signatures, NOT the matching attack).
* **retrieval-PVI on φ** (`weights.measures.v_information_weights`) — the
  *dependent* reference (the docstring is explicit it *is* the VMA in bits).

Decides C4 (which Π-probe is independent yet faithful) on the weight surface:
Spearman(measure, τ-recovery) over α_e. It also confirms the early B0 finding —
the dense Algorithm-1 keymat defeats the sorted-quantile VMA (run once at α_e=0).

The capacity-reader Π-probe candidate is degenerate here (1 row = 1 class on the
weight table) — it applies to the *activation* surface (B4), noted, not run here.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # reach scripts/defenses
from defenses.aloepri import obfuscate_embedding_table  # noqa: E402

from talens.weights import vma  # noqa: E402
from talens.weights.measures import club_mi_weights, v_information_weights  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embed", default="results/capture_cache/embed-b0c6566474cadb27.pt",
                    help="cached embedding table (.pt tensor (vocab, d)); gemma-2-2b default")
    ap.add_argument("--n-tokens", type=int, default=1200)
    ap.add_argument("--alphas", default="0,0.1,0.2,0.35,0.5,0.75,1.0,1.5")
    ap.add_argument("--bins", type=int, default=64)
    ap.add_argument("--club-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260620)
    ap.add_argument("--out", default="results/aloepri_vma_sweep.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    table = torch.load(args.embed, map_location="cpu", weights_only=False)
    table = np.asarray(table.float().numpy(), dtype=np.float32)
    vocab, d = table.shape
    sel = rng.choice(vocab, size=args.n_tokens, replace=False)
    W = table[sel].copy()
    N = W.shape[0]
    alphas = [float(s) for s in args.alphas.split(",") if s.strip()]
    print(f"[b2] embed {vocab}x{d} -> subset N={N} d={d}; alphas={alphas}", flush=True)

    def club(pair):
        return club_mi_weights(pair, bins=args.bins, steps=args.club_steps,
                               hidden_size=128, seed=0)["club_mi_bits"]

    def retr(pair):
        n_tr = max(64, int(0.66 * N)); n_va = int(0.17 * N); n_te = N - n_tr - n_va
        return v_information_weights(pair, bins=args.bins, n_train=n_tr, n_val=n_va,
                                     n_test=n_te, candidate_pool_size=N)["v_information_bits"]

    records = []
    for a in alphas:
        t0 = time.time()
        pair = obfuscate_embedding_table(W, alpha_e=a, keymat=False, seed=args.seed + 1)
        rec_vma = vma.run(pair, bins=args.bins, match="hungarian").ttrsr_top1
        rec = {"alpha_e": a, "keymat": False, "vma_recovery": rec_vma,
               "club_bits": club(pair), "retr_pvi_bits": retr(pair),
               "secs": round(time.time() - t0, 1)}
        records.append(rec)
        print(f"[b2] αe={a:<4} vma={rec_vma:.3f} club={rec['club_bits']:.1f}b "
              f"retr-pvi={rec['retr_pvi_bits']:.2f}b ({rec['secs']}s)", flush=True)

    # one keymat point: the dense key matrix should defeat the sorted VMA
    t0 = time.time()
    pk = obfuscate_embedding_table(W, alpha_e=0.0, keymat=True, seed=args.seed + 1)
    km = {"alpha_e": 0.0, "keymat": True, "vma_recovery": vma.run(pk, bins=args.bins).ttrsr_top1,
          "club_bits": club(pk), "retr_pvi_bits": retr(pk), "secs": round(time.time() - t0, 1)}
    records.append(km)
    print(f"[b2] keymat αe=0  vma={km['vma_recovery']:.3f} (chance≈{1/N:.3f}) "
          f"club={km['club_bits']:.1f}b retr-pvi={km['retr_pvi_bits']:.2f}b", flush=True)

    # C4: Spearman(measure, τ-recovery) over the perm-core α_e sweep
    sweep = [r for r in records if not r["keymat"]]
    rec_v = [r["vma_recovery"] for r in sweep]
    rho = {}
    for key in ("club_bits", "retr_pvi_bits"):
        mv = [r[key] for r in sweep]
        rho[key] = float(stats.spearmanr(mv, rec_v).statistic)
    print(f"\n[b2] Spearman(measure, τ-recovery) over α_e: "
          f"CLUB(indep)={rho['club_bits']:+.3f}  retr-PVI(dep)={rho['retr_pvi_bits']:+.3f}")

    out = {"embed": args.embed, "n_tokens": N, "d": d, "alphas": alphas,
           "channel": "permutation_pi", "surface": "embed_weight_table",
           "ground_truth": "vma_tau_recovery", "spearman_vs_recovery": rho,
           "records": records}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[b2] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
