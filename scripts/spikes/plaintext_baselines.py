#!/usr/bin/env python3
"""Clean-model plaintext-reference baselines for two registered probes (Task 7 piggyback).

Tasks 4/6 queued two probe baselines that had no clean-model reading on disk:
  * SDL (surplus description length) — prequential SDL *selectivity* (real − shuffle) per layer
    across depth, token-identity label.  src: src/talens/measures/mdl.py
  * shared spectral capacity — averaged-row-covariance water-filling capacity per layer × KV
    kind on the clean stack.  src: src/talens/measures/bss_separability.py

Both are computable from CLEAN Qwen3-4B captures ALREADY on disk (no GPU, no model load):
  * resid_post @ depth grid     -> results/capture_cache/capture-*.pt  (for SDL)
  * k / v @ {0,12,20}           -> results/capture_cache/capture-*.pt  (for shared-spectral-cap)

CPU-only: pure estimator passes over cached activations. Run in the host .venv.
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from talens.capture.cache import load_capture  # noqa: E402
from talens.probes.mdl import online_code_length  # noqa: E402
from talens.probes.bss_separability import shared_spectral_capacity_bits  # noqa: E402

SDL_LAYERS = [0, 8, 16, 24, 32]   # depth grid matching the depth-inversion page
KV_LAYERS = [0, 12, 20]


def _qwen_caches():
    """Return (resid_cache_path, kv_cache_path): the clean Qwen3-4B captures with the widest
    resid_post depth grid and the k/v stack, respectively."""
    resid_best, resid_score = None, -1
    kv_best = None
    for f in glob.glob(str(REPO / "results/capture_cache/capture-*.pt")):
        try:
            cap, _ = load_capture(Path(f))
        except Exception:  # noqa: BLE001
            continue
        if "Qwen3-4B" not in (cap.model_id or ""):
            continue
        resid_layers = cap.layers("resid_post")
        score = len([L for L in SDL_LAYERS if L in resid_layers])
        if score > resid_score:
            resid_best, resid_score = (f, cap), score
        if "k" in cap.kinds() and "v" in cap.kinds():
            if all(L in cap.layers("k") for L in KV_LAYERS):
                kv_best = (f, cap)
    return resid_best, kv_best


def sdl_baseline(resid):
    rows = []
    if resid is None:
        return rows, "no clean resid capture found"
    path, cap = resid
    have = cap.layers("resid_post")
    for L in SDL_LAYERS:
        if L not in have:
            continue
        X, y, _ = cap.stack("resid_post", L)
        X = np.asarray(X, dtype=np.float64); y = np.asarray(y)
        t0 = time.time()
        real = online_code_length(X, y, control="none")
        shuf = online_code_length(X, y, control="shuffle")
        sdl_real = real.get("surplus_description_length_bits")
        sdl_shuf = shuf.get("surplus_description_length_bits")
        sel = (sdl_real - sdl_shuf) if (sdl_real is not None and sdl_shuf is not None) else None
        rows.append({
            "probe": "SDL", "layer": L, "n_rows": int(X.shape[0]), "d": int(X.shape[1]),
            "sdl_real_bits": sdl_real, "sdl_shuffle_bits": sdl_shuf, "sdl_selectivity_bits": sel,
            "online_code_length_bits": real.get("online_code_length_bits"),
            "sec": round(time.time() - t0, 1),
        })
        print(f"[sdl] L{L:>2} sdl_real={sdl_real} sdl_shuf={sdl_shuf} sel={sel} "
              f"({rows[-1]['sec']}s)", flush=True)
    return rows, f"{Path(path).name} resid_post layers {have}"


def ssc_baseline(kv):
    rows = []
    if kv is None:
        return rows, "no clean k/v capture found"
    path, cap = kv
    for kind in ("k", "v"):
        for L in KV_LAYERS:
            if L not in cap.layers(kind):
                continue
            t0 = time.time()
            res = shared_spectral_capacity_bits(cap, layer=L, kind=kind)
            cap_per_t = res.get("cap_per_t", {})
            # headline = T=1 (single-observation) capacity; report full per-T too
            head = cap_per_t.get(1) or cap_per_t.get("1")
            rows.append({
                "probe": "shared-spectral-capacity", "kind": kind, "layer": L,
                "spectral_cap_bits_t1": head, "cap_per_t": cap_per_t,
                "d_eff_per_t": res.get("d_eff_per_t", {}), "sec": round(time.time() - t0, 1),
            })
            print(f"[ssc] kind={kind} L{L:>2} cap_t1={head} ({rows[-1]['sec']}s)", flush=True)
    return rows, f"{Path(path).name} kinds k/v layers {KV_LAYERS}"


def main() -> None:
    out = REPO / "refine-logs/utility-tradeoff/plaintext_baselines.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    resid, kv = _qwen_caches()
    sdl_rows, sdl_src = sdl_baseline(resid)
    ssc_rows, ssc_src = ssc_baseline(kv)
    out.write_text(json.dumps({
        "model": "Qwen/Qwen3-4B", "clean": True,
        "sdl": {"source": sdl_src, "rows": sdl_rows},
        "shared_spectral_capacity": {"source": ssc_src, "rows": ssc_rows},
    }, indent=2))
    print(f"[baselines] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
