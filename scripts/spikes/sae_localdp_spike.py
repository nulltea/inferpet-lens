#!/usr/bin/env python3
"""Spike: SAE-denoising token recovery vs INPUT local-DP.

Threat model = local-DP on the input representation. A generic forward hook on
the **embedding layer** (attached after load) clips each token embedding to L2
norm `C` and adds Gaussian noise `σ = C·√(2 ln(1.25/δ))/ε` (Gaussian mechanism,
per-token-embedding (ε,δ)-LDP). The **plaintext-weight** model then runs
normally, so the input noise PROPAGATES through clean layers (the hook seam is
necessary here — unlike the post-hoc cut-layer cover).

Per layer + ε we compare token recovery (learned ridge, TTRSR) from:
  - clean    x      (no DP)
  - dp       x' = model(clip+noise(emb))   (released, LDP-defended)
  - denoised x̂ = D·encode(x')               (SAE manifold projection)

Governing fact (DP post-processing immunity): the SAE cannot change ε. It can
only change *empirical* recovery — and that is exactly what we chart vs ε.
Expectation: input noise is mixed by the model's nonlinear dynamics, so by
layer L it may land partly on-manifold (the model re-naturalizes it) — the open
question is whether that leaves the SAE any denoising headroom. Per-token LDP
needs large σ (ε=1 ⇒ σ≈4.85C), so strong defense at a real utility cost.

Run via scripts/run_in_rocm.sh.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from talens.attacks import ridge_inversion


def load_sae(release, layer, width, device):
    from sae_lens import SAE
    loaded = SAE.from_pretrained(
        release=release, sae_id=f"layer_{layer}/width_{width}/canonical", device=device)
    sae = loaded[0] if isinstance(loaded, tuple) else loaded
    return sae.to(device).eval()


class InputDPCover:
    """Per-token embedding clip-to-C + Gaussian(σ) on the embedding output.

    Noise is drawn FRESH per forward call (per token) — the RNG is seeded once
    per ε in main() for reproducibility, NOT reseeded here (reseeding per call
    would make the noise a deterministic position-only offset the ridge trivially
    absorbs, collapsing the ε dependence)."""
    def __init__(self, C, sigma):
        self.C, self.sigma = C, sigma

    def __call__(self, mod, inp, out):
        f = out.float()
        norm = f.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        f = f * (self.C / norm).clamp_max(1.0)                       # clip ||·||≤C
        if self.sigma > 0:
            f = f + self.sigma * torch.randn_like(f)                 # fresh per call
        return f.to(out.dtype)


@torch.no_grad()
def capture(model, tok, prompts, layers, device):
    per = {L: [] for L in layers}; ids_all = []
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
    ap.add_argument("--epsilons", default="1,4,8")
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--clip-percentile", type=float, default=50.0,
                    help="clip norm C = this percentile of clean embedding L2 norms")
    ap.add_argument("--split-mode", default="vocab", choices=["vocab", "row"])
    ap.add_argument("--seed", type=int, default=20260618)
    ap.add_argument("--out", default="results/sae_localdp_spike.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = [int(s) for s in args.layers.split(",") if s.strip()]
    epsilons = [float(s) for s in args.epsilons.split(",") if s.strip()]
    prompts = [ln.strip() for ln in Path(args.corpus).read_text().splitlines()
               if ln.strip()][: args.max_prompts]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        attn_implementation="eager", device_map=device).eval()
    embed_table = model.get_input_embeddings().weight.detach().float().cpu()

    # Clip norm C from the clean per-token embedding L2 norms over the corpus.
    all_ids = np.concatenate([tok(p, return_tensors="np").input_ids[0] for p in prompts])
    emb_norms = embed_table[torch.from_numpy(all_ids)].norm(dim=-1).numpy()
    C = float(np.percentile(emb_norms, args.clip_percentile))
    z = math.sqrt(2 * math.log(1.25 / args.delta))
    print(f"[localdp] device={device} layers={layers} prompts={len(prompts)} "
          f"C(p{args.clip_percentile})={C:.3f} z={z:.3f} eps={epsilons}", flush=True)

    per_clean, ids = capture(model, tok, prompts, layers, device)
    saes = {L: load_sae(args.release, L, args.width, device) for L in layers}

    def ttrsr(X, y):
        m = ridge_inversion(X, y, embed_table, n_train=10**9, split_mode=args.split_mode)
        return (m["ttrsr_top1"], m["ttrsr_top10"]) if m else (None, None)

    clean = {L: ttrsr(*stack(per_clean[L], ids)) for L in layers}
    for L in layers:
        print(f"[localdp] clean L{L:>2} top1={clean[L][0]:.3f}", flush=True)

    records = []
    for eps in epsilons:
        sigma = C * z / eps
        torch.manual_seed(args.seed + int(eps * 1000))   # reproducible, fresh per call
        cover = InputDPCover(C, sigma)
        h = model.model.embed_tokens.register_forward_hook(cover)
        per_obf, _ = capture(model, tok, prompts, layers, device)
        h.remove()
        for L in layers:
            t0 = time.time()
            Xo, y = stack(per_obf[L], ids)
            o1, o10 = ttrsr(Xo, y)
            Xd = sae_denoise(saes[L], Xo, device)
            d1, d10 = ttrsr(Xd, y)
            c1 = clean[L][0]
            rec = {"epsilon": eps, "delta": args.delta, "sigma": sigma, "clip_C": C,
                   "layer": L, "clean_top1": c1, "dp_top1": o1, "denoised_top1": d1,
                   "dp_top10": o10, "denoised_top10": d10,
                   "denoise_gain": (d1 - o1) if None not in (d1, o1) else None,
                   "secs": round(time.time() - t0, 1)}
            records.append(rec)
            print(f"[localdp] ε={eps} L{L:>2} clean={c1:.3f} dp={o1:.3f} "
                  f"denoised={d1:.3f} gain={rec['denoise_gain']:+.3f} ({rec['secs']}s)",
                  flush=True)

    out = {"model": args.model, "release": args.release, "width": args.width,
           "corpus": args.corpus, "n_prompts": len(prompts),
           "split_mode": args.split_mode, "threat_model": "input-local-DP",
           "delta": args.delta, "clip_C": C, "records": records}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[localdp] wrote {args.out}", flush=True)
    print("\n=== SAE-denoising vs input local-DP ===")
    print("DP post-processing immunity: SAE cannot change ε; gain = empirical recovery only.")


if __name__ == "__main__":
    main()
