#!/usr/bin/env python3
"""DIAG: PVI/V-information overfitting (the −50b shuffled-floor anomaly).

Model-free fast loop on a cached gemma capture. Tests:
  H1 split type — v_information ALWAYS row-splits (no split_mode); confirm PVI
     is identical whatever the runner's --split-mode is.
  H2 class-probe overfit — sweep l2; shuffled PVI should rise −50 → ~0.
  H3 retrieval family — bounded ridge→emb v_information_retrieval; shuffled floor
     should be ~0 (sane), the proper fix.

Signal = the SHUFFLE-control PVI (should be ~0 for a healthy estimator).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from talens.capture.cache import load_capture, load_embed
from talens.probes.vinfo import v_information, v_information_retrieval


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default="results/capture_cache/capture-4ca8a33e16bfbec9.pt")
    ap.add_argument("--embed", default="results/capture_cache/embed-b0c6566474cadb27.pt")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--l2s", default="0.1,1,10,100,1000")
    args = ap.parse_args()

    cap, _ = load_capture(Path(args.capture))
    print(f"[diag] model={cap.model_id} kinds={cap.kinds()} "
          f"layers={cap.layers('resid_post')}", flush=True)
    X, y, _ = cap.stack("resid_post", args.layer)
    print(f"[diag] layer {args.layer}: X{X.shape} y{y.shape} "
          f"distinct_tokens={np.unique(y).size}", flush=True)

    print("\n=== H2: class-probe v_information, l2 sweep (real / shuffle / selectivity) ===")
    for l2 in [float(s) for s in args.l2s.split(",")]:
        real = v_information(X, y, l2=l2)["v_information_bits"]
        shuf = v_information(X, y, l2=l2, control="shuffle")["v_information_bits"]
        print(f"  l2={l2:>7g}  real={real:8.3f}  shuffle={shuf:9.3f}  "
              f"sel={real-shuf:8.3f}  (n_tr/n_te per defaults)", flush=True)

    print("\n=== H3: retrieval-family v_information_retrieval (bounded, generalizes) ===")
    emb = load_embed(Path(args.embed))
    real = v_information_retrieval(X, y, emb, split_mode="vocab")["v_information_bits"]
    # manual shuffle floor: permute y to break X↔Y
    rng = np.random.default_rng(20260616)
    yp = y[rng.permutation(y.size)]
    shuf = v_information_retrieval(X, yp, emb, split_mode="vocab")["v_information_bits"]
    print(f"  retrieval(vocab)  real={real:8.3f}  shuffle={shuf:9.3f}  sel={real-shuf:8.3f}",
          flush=True)
    real_r = v_information_retrieval(X, y, emb, split_mode="row")["v_information_bits"]
    print(f"  retrieval(row)    real={real_r:8.3f}", flush=True)

    # VERIFY FIX: post-hoc Gaussian noise sweep — a healthy measure must fall
    # MONOTONICALLY as noise grows. Class-probe (overfit) vs retrieval (fix).
    print("\n=== VERIFY: PVI vs post-hoc noise σ (should DECREASE monotonically) ===")
    print("   σ    class-probe(l2=.1)   retrieval(row)")
    rms = np.sqrt((X.astype(np.float64) ** 2).mean(axis=1, keepdims=True))
    for sigma in [0.0, 0.25, 0.5, 1.0, 2.0]:
        rng2 = np.random.default_rng(7)
        Xn = (X + sigma * rms * rng2.standard_normal(X.shape)).astype(np.float32)
        cp = v_information(Xn, y, l2=0.1)["v_information_bits"]
        rt = v_information_retrieval(Xn, y, emb, split_mode="row")["v_information_bits"]
        print(f"  {sigma:>4}   {cp:18.3f}   {rt:12.3f}", flush=True)

    print("\nHealthy estimator ⇒ shuffle ≈ 0 and PVI ↓ monotonically with noise.")


if __name__ == "__main__":
    main()
