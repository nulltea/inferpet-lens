#!/usr/bin/env python3
"""Spike: SAE-denoising token recovery vs a cut-layer-release defense.

Threat model = split-inference / Shredder cut-layer release: the model runs
CLEAN, then the defense perturbs the **single released layer's** activation
before the untrusted party sees it (no intra-model propagation). So the cover
is a post-hoc Tensor transform on the released residual `x_L`.

Per layer we compare token recovery (existing ridge attack, TTRSR) from:
  - clean    x_L
  - obf      x' = cover(x_L)               (released, defended)
  - denoised x̂ = D·encode(x')              (SAE projects x' onto the manifold)

Covers (`--cover`):
  - noise : x + σ·rms·𝒩   — additive activation noise (denoise-friendly).
  - rotate: (x·Q) + σ·rms·𝒩 — fixed orthogonal basis-change ± noise. NOTE a
            fixed linear map is *absorbed by the linear ridge* (TTRSR(obf) ≈
            TTRSR(noise-only)), but it breaks the SAE's manifold assumption —
            tests whether a basis-change KILLS the noise-case denoising gain
            (AloePri-style off-manifold, predicted ~null).
  - permute: signed-channel permutation ± noise (also orthogonal).

gemma-scope SAEs are gemma-2-2b-only, so this is an AloePri-*style* cover on
gemma, not the Qwen3-4B AloePri GGUF. The in-model hook seam (for a faithful
propagating AloePri) is a separate Qwen3 + llama.cpp effort.

Run via scripts/run_in_rocm.sh.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from talens.attacks._inversion import ridge_inversion


def load_sae(release: str, layer: int, width: str, device: str):
    from sae_lens import SAE
    loaded = SAE.from_pretrained(
        release=release, sae_id=f"layer_{layer}/width_{width}/canonical", device=device)
    sae = loaded[0] if isinstance(loaded, tuple) else loaded
    return sae.to(device).eval()


@torch.no_grad()
def capture(model, tok, prompts, layers, device):
    """Clean forward (no hooks): per layer -> list of (seq,d) residual matrices
    + per-prompt token ids. Cut-layer release perturbs these post-hoc."""
    per = {L: [] for L in layers}
    ids_all = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids.to(device)
        hs = model(ids, output_hidden_states=True, use_cache=False).hidden_states
        for L in layers:
            per[L].append(hs[L + 1][0].float().cpu().numpy())
        ids_all.append(ids[0].cpu().numpy())
    return per, ids_all


def stack(mats, ids_list):
    Xs, ys = [], []
    for m, ids in zip(mats, ids_list):
        n = min(m.shape[0], ids.shape[0])
        Xs.append(m[:n]); ys.append(ids[:n])
    return np.concatenate(Xs, 0), np.concatenate(ys, 0).astype(np.int64)


# ---- cut-layer-release covers (post-hoc Tensor transforms) -------------------
def make_orthogonal(d, device, seed, permute=False):
    g = torch.Generator(device=device).manual_seed(seed)
    if permute:
        perm = torch.randperm(d, generator=g, device=device)
        sign = (torch.randint(0, 2, (d,), generator=g, device=device) * 2 - 1).float()
        Q = torch.zeros(d, d, device=device)
        Q[torch.arange(d, device=device), perm] = sign
        return Q
    A = torch.randn(d, d, generator=g, device=device)
    Q, _ = torch.linalg.qr(A)
    return Q


@torch.no_grad()
def apply_cover(X, cover, sigma, Q, device, seed):
    """X (rows,d) np -> covered np. Noise scaled to per-row RMS."""
    x = torch.from_numpy(X).to(device, torch.float32)
    if Q is not None:
        x = x @ Q
    if sigma > 0:
        g = torch.Generator(device=device).manual_seed(seed)
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        x = x + sigma * rms * torch.randn(x.shape, generator=g, device=device)
    return x.cpu().numpy()


@torch.no_grad()
def sae_denoise(sae, X, device, batch=4096):
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(device, torch.float32)
        out.append(sae.decode(sae.encode(xb)).float().cpu().numpy())
    return np.concatenate(out, 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="unsloth/gemma-2-2b")
    ap.add_argument("--release", default="gemma-scope-2b-pt-res-canonical")
    ap.add_argument("--width", default="16k")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--layers", default="5,12,20")
    ap.add_argument("--max-prompts", type=int, default=256)
    ap.add_argument("--cover", default="noise", choices=["noise", "rotate", "permute"])
    ap.add_argument("--sigmas", default="0.25,0.5")
    ap.add_argument("--split-mode", default="vocab", choices=["vocab", "row"])
    ap.add_argument("--seed", type=int, default=20260618)
    ap.add_argument("--out", default="results/sae_denoise_spike.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = [int(s) for s in args.layers.split(",") if s.strip()]
    sigmas = [float(s) for s in args.sigmas.split(",") if s.strip()]
    prompts = [ln.strip() for ln in Path(args.corpus).read_text().splitlines()
               if ln.strip()][: args.max_prompts]
    print(f"[denoise] cover={args.cover} device={device} layers={layers} "
          f"sigmas={sigmas} prompts={len(prompts)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        attn_implementation="eager", device_map=device).eval()
    embed_table = model.get_input_embeddings().weight.detach().float().cpu()
    d = embed_table.shape[1]

    per_clean, ids = capture(model, tok, prompts, layers, device)
    saes = {L: load_sae(args.release, L, args.width, device) for L in layers}
    Q = None
    if args.cover in ("rotate", "permute"):
        Q = make_orthogonal(d, device, args.seed, permute=(args.cover == "permute"))

    def ttrsr(X, y):
        m = ridge_inversion(X, y, embed_table, n_train=10**9, split_mode=args.split_mode)
        return (m["ttrsr_top1"], m["ttrsr_top10"]) if m else (None, None)

    Xy = {L: stack(per_clean[L], ids) for L in layers}
    clean = {L: ttrsr(*Xy[L]) for L in layers}
    for L in layers:
        print(f"[denoise] clean L{L:>2} top1={clean[L][0]:.3f}", flush=True)

    records = []
    for sigma in sigmas:
        for L in layers:
            t0 = time.time()
            Xc, y = Xy[L]
            Xo = apply_cover(Xc, args.cover, sigma, Q, device, args.seed + L)
            o1, o10 = ttrsr(Xo, y)
            Xd = sae_denoise(saes[L], Xo, device)
            d1, d10 = ttrsr(Xd, y)
            c1 = clean[L][0]
            rec = {"cover": args.cover, "sigma": sigma, "layer": L,
                   "clean_top1": c1, "obf_top1": o1, "denoised_top1": d1,
                   "obf_top10": o10, "denoised_top10": d10,
                   "denoise_gain": (d1 - o1) if None not in (d1, o1) else None,
                   "residual_to_clean": (c1 - d1) if None not in (d1, c1) else None,
                   "secs": round(time.time() - t0, 1)}
            records.append(rec)
            print(f"[denoise] {args.cover} σ={sigma} L{L:>2} clean={c1:.3f} "
                  f"obf={o1:.3f} denoised={d1:.3f} gain={rec['denoise_gain']:+.3f} "
                  f"(clean−denoised={rec['residual_to_clean']:+.3f}) ({rec['secs']}s)",
                  flush=True)

    out = {"model": args.model, "release": args.release, "width": args.width,
           "corpus": args.corpus, "n_prompts": len(prompts),
           "split_mode": args.split_mode, "cover": args.cover,
           "threat_model": "cut-layer-release", "records": records}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[denoise] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
