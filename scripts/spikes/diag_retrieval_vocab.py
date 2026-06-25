#!/usr/bin/env python3
"""VALIDATE: v_information_retrieval under VOCAB-DISJOINT split.

Claims under test (vs the class-probe v_information which floors at −48b):
  (1) sane shuffle-control floor ≈ 0 (permuted y),
  (2) monotonic decrease under post-hoc noise (tracks leakage),
  (3) detects leakage on UNSEEN tokens (vocab-disjoint: train/test share no id)
      — exactly where the free class-probe structurally cannot generalize.

Model-free, on the cached gemma capture.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from talens.capture.cache import load_capture, load_embed
from talens.probes.vinfo import v_information, v_information_retrieval


def stack(cap, layer):
    mats = cap.operands[("resid_post", layer)]
    ids = cap.prompt_token_ids
    Xs, ys = [], []
    for m, idl in zip(mats, ids):
        a = m.detach().cpu().numpy().astype(np.float32)
        idl = np.asarray(idl, dtype=np.int64)
        n = min(a.shape[0], idl.shape[0])
        Xs.append(a[:n]); ys.append(idl[:n])
    return np.concatenate(Xs, 0), np.concatenate(ys, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default="results/capture_cache/capture-4ca8a33e16bfbec9.pt")
    ap.add_argument("--embed", default="results/capture_cache/embed-b0c6566474cadb27.pt")
    ap.add_argument("--layers", default="5,12,20")
    args = ap.parse_args()

    cap, _ = load_capture(Path(args.capture))
    emb = load_embed(Path(args.embed))
    layers = [int(s) for s in args.layers.split(",")]
    print(f"[val] model={cap.model_id} vocab={emb.shape[0]} d={emb.shape[1]}", flush=True)

    def vir_vocab(X, y):
        return v_information_retrieval(X, y, emb, split_mode="vocab")["v_information_bits"]

    for L in layers:
        X, y = stack(cap, L)
        rng = np.random.default_rng(0)
        # claim (1) + (3): real (vocab-disjoint leakage on unseen tokens) + shuffle floor
        real = vir_vocab(X, y)
        shuf = vir_vocab(X, y[rng.permutation(y.size)])
        # class-probe floor for contrast (row-split; it can't even do vocab-disjoint)
        cp_floor = v_information(X, y, control="shuffle")["v_information_bits"]
        print(f"\n[val] L{L}: retrieval(VOCAB)  real={real:7.3f}b  shuffle_floor={shuf:7.3f}b  "
              f"sel={real-shuf:7.3f}b  | class-probe shuffle_floor={cp_floor:8.3f}b", flush=True)
        # claim (2): monotonic decrease under post-hoc noise (vocab-disjoint)
        rms = np.sqrt((X.astype(np.float64) ** 2).mean(axis=1, keepdims=True))
        rng2 = np.random.default_rng(7)
        prev, mono = None, True
        row = []
        for sigma in [0.0, 0.25, 0.5, 1.0, 2.0]:
            Xn = (X + sigma * rms * rng2.standard_normal(X.shape)).astype(np.float32)
            v = vir_vocab(Xn, y)
            row.append((sigma, v))
            if prev is not None and v > prev + 0.3:
                mono = False
            prev = v
        print("       noise σ→PVI(vocab):  " +
              "  ".join(f"σ{s}={v:6.3f}" for s, v in row) +
              f"   monotone={mono}", flush=True)


if __name__ == "__main__":
    main()
