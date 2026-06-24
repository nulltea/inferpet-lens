#!/usr/bin/env python3
"""Downstream-utility (perplexity) side of the leakage--utility tradeoff for the LOSSY
LM-surface defenses (Task 7, utility-tradeoff). GPU — wrap in scripts/run_in_rocm.sh.

Two modes, each one model load:
  --mode input-dp   gemma-2-2b: per-token-embedding (ε,δ)-LDP cover (clip-to-C + Gaussian
                    σ=C·z/ε) on the input embeddings (same mechanism as the resid-dp-attacks
                    leakage sweep) -> language-model perplexity vs ε.
  --mode pripert    Qwen3-4B: PriPert split-inference cover (per-row top-ρ sparsify + energy-
                    matched Gaussian δ, σ=β·meanRMS(plaintext) fixed floor) at a cut layer
                    -> perplexity vs β (ρ fixed), matching the resid-split leakage sweep.

Utility = mean next-token perplexity on the corpus with the cover active in the forward pass.
The cover is applied via a forward hook so the WHOLE downstream stack sees the protected
activation (exactly what the protected model computes). Each sweep row aligns to the recovery
sweep's parameter value. Incremental save after every row.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))  # defenses.*
from defenses.pripert import pripert_apply, sparsify_rows  # noqa: E402


def parse_eps(t):
    return math.inf if t.strip().lower() in ("inf", "infinity") else float(t)


class InputDPHook:
    """clip-to-C + fresh Gaussian(σ) on the embedding output (per-token LDP)."""
    def __init__(self, C, sigma):
        self.C, self.sigma = C, sigma

    def __call__(self, mod, inp, out):
        f = out.float()
        norm = f.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        f = f * (self.C / norm).clamp_max(1.0)
        if self.sigma > 0:
            f = f + self.sigma * torch.randn_like(f)
        return f.to(out.dtype)


class PriPertHook:
    """per-row top-ρ sparsify + energy-matched Gaussian δ on a decoder layer's hidden states."""
    def __init__(self, rho, beta, sigma_ref, seed=0):
        self.rho, self.beta, self.sigma_ref, self.seed, self._n = rho, beta, sigma_ref, seed, 0

    def __call__(self, mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        H = hs.float()[0].cpu().numpy()  # (seq, d), batch=1
        U, _ = pripert_apply(H, rho=self.rho, beta=self.beta, mode="gauss",
                             seed=self.seed + self._n, sigma=self.sigma_ref * self.beta if self.beta > 0 else 0.0)
        self._n += 1
        u = torch.from_numpy(U).to(hs.device, hs.dtype).unsqueeze(0)
        if isinstance(out, tuple):
            return (u,) + tuple(out[1:])
        return u


@torch.no_grad()
def perplexity(model, tok, prompts, device):
    tot_nll, tot_tok = 0.0, 0
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids.to(device)
        if ids.shape[1] < 2:
            continue
        out = model(ids, labels=ids, use_cache=False)
        n = ids.shape[1] - 1  # next-token positions scored by HF shift
        tot_nll += float(out.loss) * n
        tot_tok += n
    return math.exp(tot_nll / max(tot_tok, 1)), tot_tok


@torch.no_grad()
def layer_rms(model, tok, prompts, layer, device):
    """mean per-row RMS of the clean hidden state at decoder `layer` (PriPert σ floor)."""
    acc = []
    h = model.model.layers[layer].register_forward_hook(
        lambda m, i, o: acc.append((o[0] if isinstance(o, tuple) else o).detach().float()[0].pow(2).mean(-1).sqrt().mean().item()))
    with torch.no_grad():
        for p in prompts[: min(48, len(prompts))]:
            model(tok(p, return_tensors="pt").input_ids.to(device), use_cache=False)
    h.remove()
    return float(np.mean(acc)) if acc else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=["input-dp", "pripert"])
    ap.add_argument("--corpus", default="corpora/release-gate-512.txt")
    ap.add_argument("--max-prompts", type=int, default=64)
    ap.add_argument("--epsilons", default="inf,256,128,96,64")  # input-dp
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--clip-percentile", type=float, default=99.9)
    ap.add_argument("--layer", type=int, default=8)            # pripert cut layer
    ap.add_argument("--rho", type=float, default=0.25)         # pripert sparsity
    ap.add_argument("--betas", default="0.0,0.25,0.5,1.0,2.0")  # pripert noise budget
    ap.add_argument("--seed", type=int, default=20260625)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[util] cuda={torch.cuda.is_available()} mode={args.mode}", flush=True)
    prompts = [ln.strip() for ln in Path(args.corpus).read_text().splitlines() if ln.strip()][: args.max_prompts]
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "unsloth/gemma-2-2b" if args.mode == "input-dp" else "Qwen/Qwen3-4B"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map=device).eval()
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    records = []

    def save(meta_extra):
        out.write_text(json.dumps({
            "mode": args.mode, "model": model_id, "corpus": args.corpus,
            "n_prompts": len(prompts), "config": vars(args), **meta_extra,
            "records": records}, indent=2))

    if args.mode == "input-dp":
        # calibrate clip C from runtime embedding-output norms
        cal = []
        ch = model.model.embed_tokens.register_forward_hook(
            lambda m, i, o: cal.append(o.detach().float().norm(dim=-1).flatten().cpu()))
        with torch.no_grad():
            for p in prompts[: min(48, len(prompts))]:
                model(tok(p, return_tensors="pt").input_ids.to(device), use_cache=False)
        ch.remove()
        cal = torch.cat(cal).numpy()
        C = float(np.percentile(cal, args.clip_percentile))
        z = math.sqrt(2 * math.log(1.25 / args.delta))
        print(f"[util] gemma C(p{args.clip_percentile})={C:.3f} z={z:.3f}", flush=True)
        for es in args.epsilons.split(","):
            eps = parse_eps(es)
            sigma = 0.0 if math.isinf(eps) else C * z / eps
            torch.manual_seed(args.seed + (0 if math.isinf(eps) else int(eps)))
            hk = model.model.embed_tokens.register_forward_hook(InputDPHook(C, sigma))
            t0 = time.time()
            ppl, ntok = perplexity(model, tok, prompts, device)
            hk.remove()
            rec = {"param_name": "epsilon", "param_value": (None if math.isinf(eps) else eps),
                   "sigma": sigma, "utility_metric": "perplexity", "utility_value": ppl,
                   "n_tokens": ntok, "sec": round(time.time() - t0, 1)}
            records.append(rec); save({"clip_C": C, "z_dp": z})
            print(f"[util] ε={es:>5} σ={sigma:.4f} ppl={ppl:.3f} ({rec['sec']}s)", flush=True)

    else:  # pripert
        L = args.layer
        sigma_ref = layer_rms(model, tok, prompts, L, device)
        print(f"[util] Qwen3 L{L} plaintext meanRMS={sigma_ref:.4f} (σ floor base)", flush=True)
        # true no-defense plaintext anchor (no hook): the ρ=1,β=0 baseline perplexity
        t0 = time.time()
        ppl0, ntok0 = perplexity(model, tok, prompts, device)
        records.append({"param_name": "beta", "param_value": 0.0, "rho": 1.0, "layer": L,
                        "sigma": 0.0, "utility_metric": "perplexity", "utility_value": ppl0,
                        "n_tokens": ntok0, "plaintext_anchor": True, "sec": round(time.time() - t0, 1)})
        save({"layer": L, "rho": args.rho, "sigma_ref": sigma_ref})
        print(f"[util] PLAINTEXT (ρ=1,β=0) ppl={ppl0:.3f}", flush=True)
        for bs in args.betas.split(","):
            beta = float(bs)
            torch.manual_seed(args.seed + int(beta * 1000))
            hk = model.model.layers[L].register_forward_hook(
                PriPertHook(args.rho, beta, sigma_ref, seed=args.seed))
            t0 = time.time()
            ppl, ntok = perplexity(model, tok, prompts, device)
            hk.remove()
            rec = {"param_name": "beta", "param_value": beta, "rho": args.rho, "layer": L,
                   "sigma": sigma_ref * beta, "utility_metric": "perplexity", "utility_value": ppl,
                   "n_tokens": ntok, "sec": round(time.time() - t0, 1)}
            records.append(rec); save({"layer": L, "rho": args.rho, "sigma_ref": sigma_ref})
            print(f"[util] β={beta:<5} ρ={args.rho} σ={rec['sigma']:.4f} ppl={ppl:.3f} ({rec['sec']}s)", flush=True)

    print(f"[util] wrote {out} ({len(records)} rows)", flush=True)


if __name__ == "__main__":
    main()
