#!/usr/bin/env python3
"""Spike: does a bounded probe decode the secret (token-id) *more easily*
from the sparse gemma-scope code Z = SAE.encode(x) than from the dense
residual stream x?

This is the pure form of measurement question (2): a V-information
comparison, NOT a raw-MI claim. Because Z is a deterministic function of x,
Shannon I(Z; y) <= I(x; y) (data-processing). What can still differ is the
*usable* information under a fixed probe family — i.e. PVI / V-info. If
I_V(Z -> y) > I_V(x -> y), the sparse overcomplete basis is more linearly
decodable; if <, the SAE bottleneck has cost usable token-identity signal.

Standalone: no pipeline integration. Reuses talens.capture +
talens.probes.v_information. GPU + gated HF models required — run via
scripts/run_in_rocm.sh after `huggingface-cli login` (gemma-2-2b and
gemma-scope are gated under Google's Gemma terms).

Example:
    scripts/run_in_rocm.sh python3 scripts/spikes/sae_vinfo_spike.py \
        --corpus corpora/release-gate-512.txt --max-prompts 256 \
        --layers 5,12,20 --out results/sae_vinfo_spike.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from talens.capture.capture import capture_representations, load_model
from talens.probes.vinfo import v_information


def load_sae(release: str, layer: int, width: str, device: str):
    """Load one gemma-scope residual-stream SAE. Tolerates both the newer
    (returns SAE) and older (returns (sae, cfg, sparsity)) sae_lens API."""
    from sae_lens import SAE

    sae_id = f"layer_{layer}/width_{width}/canonical"
    loaded = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    sae = loaded[0] if isinstance(loaded, tuple) else loaded
    return sae.to(device).eval()


@torch.no_grad()
def encode_features(sae, X: np.ndarray, device: str, batch: int = 4096) -> np.ndarray:
    """X (rows, d) f32 -> Z (rows, n_features) f32 via SAE.encode, batched."""
    out = []
    for i in range(0, X.shape[0], batch):
        xb = torch.from_numpy(X[i : i + batch]).to(device=device, dtype=torch.float32)
        z = sae.encode(xb)
        out.append(z.detach().to(torch.float32).cpu().numpy())
    return np.concatenate(out, 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--release", default="gemma-scope-2b-pt-res-canonical")
    ap.add_argument("--width", default="16k")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--layers", default="5,12,20", help="comma-separated layer indices")
    ap.add_argument("--max-prompts", type=int, default=256)
    ap.add_argument("--max-classes", type=int, default=256)
    ap.add_argument("--control", default="shuffle", choices=["none", "shuffle"])
    ap.add_argument("--out", default="results/sae_vinfo_spike.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = [int(s) for s in args.layers.split(",") if s.strip()]
    prompts = [
        ln.strip()
        for ln in Path(args.corpus).read_text().splitlines()
        if ln.strip()
    ][: args.max_prompts]
    print(f"[spike] device={device} model={args.model} layers={layers} "
          f"prompts={len(prompts)}", flush=True)

    model = load_model(args.model)
    cap = capture_representations(model, prompts, layers=layers, kinds=("resid_post",))

    records = []
    for li in layers:
        t0 = time.time()
        X, y, _ = cap.stack("resid_post", li)
        sae = load_sae(args.release, li, args.width, device)
        Z = encode_features(sae, X, device)
        l0 = float((Z > 0).sum(axis=1).mean())  # avg active features / row

        common = dict(max_classes=args.max_classes, control=args.control)
        vx = v_information(X, y, **common)
        vz = v_information(Z, y, **common)

        rec = {
            "layer": li,
            "n_rows": int(X.shape[0]),
            "d_dense": int(X.shape[1]),
            "d_sparse": int(Z.shape[1]),
            "sae_l0": l0,
            "vinfo_dense_bits": vx["v_information_bits"],
            "vinfo_sparse_bits": vz["v_information_bits"],
            "delta_bits": vz["v_information_bits"] - vx["v_information_bits"],
            "num_classes": vx["num_classes"],
            "secs": round(time.time() - t0, 1),
        }
        if args.control == "shuffle":
            rec["vinfo_dense_shuffle"] = vx.get("v_information_bits_shuffle")
            rec["vinfo_sparse_shuffle"] = vz.get("v_information_bits_shuffle")
        records.append(rec)
        print(
            f"[spike] L{li:>2} rows={rec['n_rows']:>5} L0={l0:6.1f} "
            f"dense={rec['vinfo_dense_bits']:.3f}b  sparse={rec['vinfo_sparse_bits']:.3f}b  "
            f"Δ={rec['delta_bits']:+.3f}b  ({rec['secs']}s)",
            flush=True,
        )

    out = {
        "model": args.model,
        "release": args.release,
        "width": args.width,
        "corpus": args.corpus,
        "n_prompts": len(prompts),
        "max_classes": args.max_classes,
        "control": args.control,
        "records": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[spike] wrote {args.out}", flush=True)

    # Headline read for question (2).
    deltas = [r["delta_bits"] for r in records]
    print("\n=== Question (2): is sparse Z more decodable than dense x? ===")
    print(f"mean Δ V-info (sparse − dense) = {np.mean(deltas):+.3f} bits "
          f"over layers {layers}")
    print(">0 ⇒ SAE basis more usable-decodable; <0 ⇒ bottleneck cost signal")


if __name__ == "__main__":
    main()
