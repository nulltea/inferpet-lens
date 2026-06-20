#!/usr/bin/env python3
"""Round-3: a NON-DP intervention + the L20 divergence mechanism (model-free).

Reviewer R2's top gaps: (1) all faithfulness evidence is input-DP only — add a
defense that degrades the attack by something OTHER than embedding-input noise;
(2) turn the late-layer divergence into a measured result.

Both are done model-free on the cached gemma capture: we perturb the *hidden
state* X directly with two non-DP, non-input interventions and run the attack +
every measure on the perturbed X:
  * pca_ablate : zero out the top-k principal directions of X (structured
    subspace obfuscation — removes the high-energy geometry the embedding
    reconstruction leans on, without touching the embedding input).
  * iso_noise  : additive isotropic hidden-state Gaussian noise.

Per cell we log TTRSR (attack), cap-PVI (pca64,l2=10) + its shuffle floor + the
reader's token-ID top-1 accuracy, retrieval-PVI, CLUB. Then Spearman(cap, TTRSR)
under the non-DP defense, and the L20 mechanism: token-ID accuracy vs TTRSR.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from talens.attacks._inversion import ridge_inversion
from talens.capture.cache import load_capture, load_embed
from talens.measures.club import club_mi_upper_bound
from talens.measures.vinfo import v_information_retrieval
from talens.measures.vinfo_capacity import _pca_basis, v_information_capacity

CAP = dict(family="pca_softmax", dim=64, l2=10.0)  # overridden by --l2


def _pca_ablate(X, k):
    if k <= 0:
        return X
    mean, comp = _pca_basis(X.astype(np.float32), k)  # comp: (d,k) top-k axes
    proj = (X - mean) @ comp                            # (n,k)
    return (X - proj @ comp.T).astype(np.float32)       # remove top-k subspace


def _iso(X, sigma, seed=11):
    if sigma == 0:
        return X
    rms = np.sqrt((X.astype(np.float64) ** 2).mean(axis=1, keepdims=True))
    return (X + sigma * rms * np.random.default_rng(seed).standard_normal(X.shape)).astype(np.float32)


def cell(X, y, emb):
    att = ridge_inversion(X, y, emb, n_train=10**9, split_mode="vocab")
    ttrsr = att["ttrsr_top1"] if att else None
    cap = v_information_capacity(X, y, **CAP)
    cap_sh = v_information_capacity(X, y, **CAP, control="shuffle")
    retr = v_information_retrieval(X, y, emb, split_mode="row")["v_information_bits"]
    club = club_mi_upper_bound(X, emb[torch.from_numpy(y)].numpy(), max_rows=2500)["club_mi_bits"]
    return {"ttrsr": ttrsr, "cap": cap["v_information_bits"], "cap_sh": cap_sh["v_information_bits"],
            "cap_sel": cap["v_information_bits"] - cap_sh["v_information_bits"],
            "cap_acc": cap["reader_top1_acc"], "retr": retr, "club": club}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default="results/capture_cache/capture-4ca8a33e16bfbec9.pt")
    ap.add_argument("--embed", default="results/capture_cache/embed-b0c6566474cadb27.pt")
    ap.add_argument("--layers", default="5,12,20")
    ap.add_argument("--every-n", type=int, default=2)
    ap.add_argument("--ablate-ks", default="0,8,32,128,384")
    ap.add_argument("--iso-sigmas", default="0,0.5,1,2,4")
    ap.add_argument("--l2", type=float, default=10.0, help="cap reader weight decay")
    ap.add_argument("--out", default="results/nondp_intervention.json")
    args = ap.parse_args()
    CAP["l2"] = args.l2

    cap_cache, _ = load_capture(Path(args.capture))
    emb = load_embed(Path(args.embed))
    layers = [int(s) for s in args.layers.split(",")]
    ks = [int(s) for s in args.ablate_ks.split(",")]
    sigmas = [float(s) for s in args.iso_sigmas.split(",")]
    out = {"interventions": {}, "cap": CAP}

    for itv, knob_name, knobs, fn in [
        ("pca_ablate", "k", ks, _pca_ablate), ("iso_noise", "sigma", sigmas, _iso)]:
        print(f"\n===== non-DP intervention: {itv} =====", flush=True)
        out["interventions"][itv] = {"knob": knob_name, "layers": {}}
        for L in layers:
            X0, y, _ = cap_cache.stack("resid_post", L)
            if args.every_n > 1:
                X0, y = X0[:: args.every_n], y[:: args.every_n]
            rows = []
            for kv in knobs:
                Xp = fn(X0, kv)
                c = cell(Xp, y, emb)
                c[knob_name] = kv
                rows.append(c)
                print(f"  L{L:>2} {knob_name}={kv:<5} ttrsr={c['ttrsr']:.3f} "
                      f"cap={c['cap']:.2f}(sh{c['cap_sh']:.2f}) cap_acc={c['cap_acc']:.3f} "
                      f"retr={c['retr']:.2f} club={c['club']:.0f}", flush=True)
            # Spearman of each measure vs TTRSR across the knob (per layer)
            t = np.array([r["ttrsr"] for r in rows], float)
            sp = {m: round(float(stats.spearmanr([r[m] for r in rows], t).statistic), 3)
                  for m in ["cap_sel", "cap_acc", "retr", "club"]}
            print(f"  L{L:>2} Spearman vs TTRSR: {sp}", flush=True)
            out["interventions"][itv]["layers"][str(L)] = {"rows": rows, "spearman_vs_ttrsr": sp}

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
