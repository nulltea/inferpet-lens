#!/usr/bin/env python3
"""Spike #2 (D-A): cheap, known-weight token recovery from the sparse SAE
code, on PLAINTEXT (Identity) — no cover.

Pipeline per layer: z = SAE.encode(x); x_hat = SAE.decode(z) (the decoder is
known under WEIGHTS-PUB). Feed x_hat to the EXISTING hidden_state ridge attack
(TTRSR top-1/10) and compare to the same attack on the raw residual x.

Why this design (see docs/dev/sae-attack.md):
  - Routing z -> x_hat via the known decoder collapses 16384 -> 2304 BEFORE any
    fit, so the n<<d overfit that wrecked spike #1 cannot occur.
  - TTRSR is bounded [0,1] (unlike the V-info bits that blew up in spike #1).
  - The SAE is just a Transform (x -> D.encode(x)), so the existing attack runs
    unchanged. Baseline is Identity.

Reads: (x vs x_hat) = the SAE-reconstruction cost to a plaintext attacker.
Optional --include-zraw also attacks the raw 16k code z (encode-only) to show
the direct high-dim fit (expensive: 16k-dim ridge).

Run via scripts/run_in_rocm.sh (gemma-2-2b cached, gemma-scope SAEs public).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from talens.attacks.hidden_state import run as attack_run
from talens.capture.capture import load_or_capture
from talens.transforms import Identity


class _SAETransform:
    """Wrap a gemma-scope SAE as a Tensor->Tensor Transform on one layer's
    residual operand. mode='decode' -> x_hat = decode(encode(x)) (D-A route);
    mode='encode' -> z (the raw 16k sparse code)."""

    def __init__(self, sae, device: str, mode: str):
        self.sae = sae
        self.device = device
        self.mode = mode
        self.name = "sae_decode_xhat" if mode == "decode" else "sae_encode_z"

    @torch.no_grad()
    def __call__(self, h: torch.Tensor, *, prompt_index: int) -> torch.Tensor:
        x = h.to(device=self.device, dtype=torch.float32)
        z = self.sae.encode(x)
        return self.sae.decode(z) if self.mode == "decode" else z


def load_sae(release: str, layer: int, width: str, device: str):
    from sae_lens import SAE

    sae_id = f"layer_{layer}/width_{width}/canonical"
    loaded = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    sae = loaded[0] if isinstance(loaded, tuple) else loaded
    return sae.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="unsloth/gemma-2-2b")
    ap.add_argument("--release", default="gemma-scope-2b-pt-res-canonical")
    ap.add_argument("--width", default="16k")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--layers", default="5,12,20")
    ap.add_argument("--max-prompts", type=int, default=512)
    ap.add_argument("--split-mode", default="vocab", choices=["vocab", "row"])
    ap.add_argument("--include-zraw", action="store_true",
                    help="also attack the raw 16k code z (expensive 16k ridge)")
    ap.add_argument("--out", default="results/sae_recover_spike.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = [int(s) for s in args.layers.split(",") if s.strip()]
    prompts = [ln.strip() for ln in Path(args.corpus).read_text().splitlines()
               if ln.strip()][: args.max_prompts]
    print(f"[spike2] device={device} model={args.model} layers={layers} "
          f"prompts={len(prompts)} split={args.split_mode}", flush=True)

    cap, embed_table, source = load_or_capture(
        args.model, prompts, capture_layers=layers, kinds=("resid_post",))
    print(f"[spike2] capture source={source} vocab={embed_table.shape[0]} "
          f"d={embed_table.shape[1]}", flush=True)

    def ttrsr(transform):
        r = attack_run(cap, embed_table, layer=li, kind="resid_post",
                       transform=transform, split_mode=args.split_mode)
        return r.ttrsr_top1, r.ttrsr_top10

    records = []
    for li in layers:
        t0 = time.time()
        sae = load_sae(args.release, li, args.width, device)
        x1, x10 = ttrsr(Identity())
        xh1, xh10 = ttrsr(_SAETransform(sae, device, "decode"))
        rec = {
            "layer": li,
            "x_top1": x1, "x_top10": x10,
            "xhat_top1": xh1, "xhat_top10": xh10,
            "delta_top1": (xh1 - x1) if (xh1 is not None and x1 is not None) else None,
            "secs": round(time.time() - t0, 1),
        }
        if args.include_zraw:
            z1, z10 = ttrsr(_SAETransform(sae, device, "encode"))
            rec["zraw_top1"], rec["zraw_top10"] = z1, z10
        records.append(rec)
        msg = (f"[spike2] L{li:>2} x_top1={x1:.3f} xhat_top1={xh1:.3f} "
               f"Δ={rec['delta_top1']:+.3f} | x_top10={x10:.3f} xhat_top10={xh10:.3f} "
               f"({rec['secs']}s)")
        if args.include_zraw:
            msg += f" | zraw_top1={rec['zraw_top1']:.3f}"
        print(msg, flush=True)

    out = {
        "model": args.model, "release": args.release, "width": args.width,
        "corpus": args.corpus, "n_prompts": len(prompts),
        "split_mode": args.split_mode, "records": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[spike2] wrote {args.out}", flush=True)
    d = [r["delta_top1"] for r in records if r["delta_top1"] is not None]
    if d:
        print(f"\n=== D-A: token recovery from x_hat=D.enc(x) vs raw x (plaintext) ===")
        print(f"mean Δ top-1 (x_hat − x) = {sum(d)/len(d):+.3f} over layers {layers}")
        print(">0 ⇒ SAE recon helps; ≈0 ⇒ recovery survives the SAE; <0 ⇒ recon cost")


if __name__ == "__main__":
    main()
