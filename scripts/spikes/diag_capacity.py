#!/usr/bin/env python3
"""M1 SCREEN: capacity-matched class-PVI variants vs class-PVI / retrieval-PVI.

Model-free fast loop on the cached gemma-2-2b capture. For each family the
screen reports the three well-posedness + cost criteria from
``refine-logs/EXPERIMENT_PLAN.md`` Block 1:

  * shuffle-control floor  — healthy ⇒ ≈ 0 (class-PVI sits at ≈ −48 b)
  * monotonicity under post-hoc Gaussian noise σ — healthy ⇒ PVI ↓ monotonically
  * wall-clock / call       — HARD constraint: ≤ class-PVI cost

Run via ``scripts/run_in_rocm.sh`` (softmax fits use the GPU). Writes a JSON
summary for the tracker.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from talens.capture.cache import load_capture, load_embed
from talens.measures.vinfo import v_information, v_information_retrieval
from talens.measures.vinfo_capacity import v_information_capacity

CAP_FAMILIES = ["pca_softmax", "randproj_softmax", "gauss", "knn"]


def _timed(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t0


def _noisy(X: np.ndarray, sigma: float, seed: int = 7) -> np.ndarray:
    if sigma == 0.0:
        return X
    rms = np.sqrt((X.astype(np.float64) ** 2).mean(axis=1, keepdims=True))
    rng = np.random.default_rng(seed)
    return (X + sigma * rms * rng.standard_normal(X.shape)).astype(np.float32)


def screen_layer(X, y, emb, *, dims, sigmas, seeds, n_neighbors):
    res: dict = {"n_rows": int(X.shape[0]), "d": int(X.shape[1]),
                 "distinct_tokens": int(np.unique(y).size)}

    # --- baseline class-PVI: floor + cost ---
    cp_real, cp_shuf, cp_t = [], [], []
    for s in seeds:
        (r, dt) = _timed(v_information, X, y, seed=s)
        cp_real.append(r["v_information_bits"]); cp_t.append(dt)
        cp_shuf.append(v_information(X, y, seed=s, control="shuffle")["v_information_bits"])
    res["class_pvi"] = {"real": float(np.mean(cp_real)), "shuffle": float(np.mean(cp_shuf)),
                        "selectivity": float(np.mean(cp_real) - np.mean(cp_shuf)),
                        "time_s": float(np.mean(cp_t))}
    class_pvi_cost = res["class_pvi"]["time_s"]

    # retrieval-PVI (manual shuffle by permuting y)
    rp_real, rp_t = [], []
    for s in seeds:
        (r, dt) = _timed(v_information_retrieval, X, y, emb, seed=s, split_mode="row")
        rp_real.append(r["v_information_bits"]); rp_t.append(dt)
    res["retrieval_pvi"] = {"real": float(np.mean(rp_real)), "time_s": float(np.mean(rp_t))}

    # --- capacity-matched families: floor + cost over dim sweep ---
    res["capacity"] = {}
    for fam in CAP_FAMILIES:
        fam_dims = dims if fam != "gauss" else dims  # all reduce first
        rows = []
        for dim in fam_dims:
            reals, shufs, ts, edims = [], [], [], []
            for s in seeds:
                (r, dt) = _timed(v_information_capacity, X, y, family=fam, dim=dim,
                                 n_neighbors=n_neighbors, seed=s)
                sh = v_information_capacity(X, y, family=fam, dim=dim,
                                            n_neighbors=n_neighbors, seed=s, control="shuffle")
                reals.append(r["v_information_bits"]); shufs.append(sh["v_information_bits"])
                ts.append(dt); edims.append(r["eff_dim"])
            rows.append({"dim": dim, "eff_dim": int(np.mean(edims)),
                         "real": float(np.mean(reals)), "shuffle": float(np.mean(shufs)),
                         "selectivity": float(np.mean(reals) - np.mean(shufs)),
                         "time_s": float(np.mean(ts)),
                         "cost_ratio_vs_classpvi": float(np.mean(ts) / class_pvi_cost)})
        res["capacity"][fam] = rows

    # --- monotonicity under post-hoc noise (one representative dim per family) ---
    rep_dim = dims[len(dims) // 2]
    res["noise_sweep"] = {"sigmas": sigmas, "rep_dim": rep_dim}
    curves: dict = {"class_pvi": [], "retrieval_pvi": []}
    for fam in CAP_FAMILIES:
        curves[fam] = []
    for sig in sigmas:
        Xn = _noisy(X, sig)
        curves["class_pvi"].append(float(v_information(Xn, y, seed=seeds[0])["v_information_bits"]))
        curves["retrieval_pvi"].append(
            float(v_information_retrieval(Xn, y, emb, seed=seeds[0], split_mode="row")["v_information_bits"]))
        for fam in CAP_FAMILIES:
            curves[fam].append(float(v_information_capacity(
                Xn, y, family=fam, dim=rep_dim, n_neighbors=n_neighbors, seed=seeds[0])["v_information_bits"]))
    res["noise_sweep"]["curves"] = curves
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default="results/capture_cache/capture-4ca8a33e16bfbec9.pt")
    ap.add_argument("--embed", default="results/capture_cache/embed-b0c6566474cadb27.pt")
    ap.add_argument("--layers", default="12")
    ap.add_argument("--dims", default="64,128,256")
    ap.add_argument("--sigmas", default="0,0.25,0.5,1,2")
    ap.add_argument("--seeds", default="20260615,20260620,20260625")
    ap.add_argument("--every-n", type=int, default=1,
                    help="row-stride subsample (2 ≈ half rows, faster screen)")
    ap.add_argument("--n-neighbors", type=int, default=25)
    ap.add_argument("--out", default="results/capacity_screen.json")
    args = ap.parse_args()

    cap, _ = load_capture(Path(args.capture))
    emb = load_embed(Path(args.embed))
    dims = [int(x) for x in args.dims.split(",")]
    sigmas = [float(x) for x in args.sigmas.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    layers = [int(x) for x in args.layers.split(",")]
    print(f"[screen] model={cap.model_id} layers={layers} dims={dims} seeds={len(seeds)}", flush=True)

    out: dict = {"model": cap.model_id, "dims": dims, "sigmas": sigmas,
                 "seeds": seeds, "n_neighbors": args.n_neighbors, "layers": {}}
    for L in layers:
        X, y, _ = cap.stack("resid_post", L)
        if args.every_n > 1:
            X, y = X[:: args.every_n], y[:: args.every_n]
        print(f"\n[screen] === layer {L}: X{X.shape} (every_n={args.every_n}) ===", flush=True)
        r = screen_layer(X, y, emb, dims=dims, sigmas=sigmas, seeds=seeds, n_neighbors=args.n_neighbors)
        out["layers"][str(L)] = r
        cp = r["class_pvi"]
        print(f"  class-PVI    real={cp['real']:7.2f} shuf={cp['shuffle']:8.2f} "
              f"sel={cp['selectivity']:7.2f} t={cp['time_s']:.2f}s", flush=True)
        print(f"  retr-PVI     real={r['retrieval_pvi']['real']:7.2f} t={r['retrieval_pvi']['time_s']:.2f}s", flush=True)
        for fam in CAP_FAMILIES:
            best = max(r["capacity"][fam], key=lambda d: d["selectivity"])
            print(f"  {fam:16s} best dim={best['dim']:4d} real={best['real']:7.2f} "
                  f"shuf={best['shuffle']:7.2f} sel={best['selectivity']:7.2f} "
                  f"t={best['time_s']:.2f}s ({best['cost_ratio_vs_classpvi']:.2f}× class-PVI)", flush=True)
        nc = r["noise_sweep"]["curves"]
        print(f"  noise σ={sigmas}", flush=True)
        for name in ["class_pvi", "retrieval_pvi", *CAP_FAMILIES]:
            print(f"    {name:16s} {['%.2f'%v for v in nc[name]]}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[screen] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
