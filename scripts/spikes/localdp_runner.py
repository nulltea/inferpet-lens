#!/usr/bin/env python3
"""Generic input local-DP experiment runner (no SAE).

Charts the privacy–utility curve for input-space local-DP across a leakage
panel: the learned-ridge attack (TTRSR), PVI / V-information, and CLUB (MI
upper bound), all as a function of the DP budget ε.

A forward hook on the embedding layer clips each token embedding to L2 norm C
(high percentile → clip-only ≈ clean, so the curve is *noise*-driven, not
clip-driven) and adds fresh Gaussian noise σ=C·√(2ln(1.25/δ))/ε
(per-token-embedding (ε,δ)-LDP). The plaintext-weight model then runs; we read
recovery/leakage from resid_post at each layer.

Note the √d blowup: total noise norm ≈ σ·√d, so the meaningful regime sits at
large ε. `r` (printed) = ||noise||/||signal|| vs the median embedding norm.
Pass `inf` in --epsilons for the clip-only (σ=0) no-noise anchor.

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

from talens.attacks._inversion import ridge_inversion
from talens.measures.vinfo import v_information
from talens.measures.club import club_mi_upper_bound


class InputDPCover:
    """Per-token embedding clip-to-C + fresh Gaussian(σ). Seeded once per ε in
    main(); noise drawn fresh each forward (never reseed per call)."""
    def __init__(self, C, sigma):
        self.C, self.sigma = C, sigma

    def __call__(self, mod, inp, out):
        f = out.float()
        norm = f.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        f = f * (self.C / norm).clamp_max(1.0)
        if self.sigma > 0:
            f = f + self.sigma * torch.randn_like(f)
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


def parse_eps(t):
    return math.inf if t.strip().lower() in ("inf", "infinity") else float(t)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="unsloth/gemma-2-2b")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--layers", default="5,12,20")
    ap.add_argument("--max-prompts", type=int, default=256)
    ap.add_argument("--epsilons", default="inf,8192,4096,2048,1024,512")
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--clip-percentile", type=float, default=99.9,
                    help="C = this percentile of clean emb norms (high → clip-only≈clean)")
    ap.add_argument("--club-max-rows", type=int, default=2500)
    ap.add_argument("--split-mode", default="vocab", choices=["vocab", "row"])
    ap.add_argument("--seed", type=int, default=20260618)
    ap.add_argument("--out", default="results/localdp_curve.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = [int(s) for s in args.layers.split(",") if s.strip()]
    epsilons = [parse_eps(s) for s in args.epsilons.split(",") if s.strip()]
    prompts = [ln.strip() for ln in Path(args.corpus).read_text().splitlines()
               if ln.strip()][: args.max_prompts]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        attn_implementation="eager", device_map=device).eval()
    embed_table = model.get_input_embeddings().weight.detach().float().cpu()
    d = embed_table.shape[1]

    # Calibrate C from the RUNTIME embed_tokens output norms (NOT the static
    # table — the hook clips the live output, whose scale can differ, e.g. a
    # folded normalizer). A no-op recording hook over a subset of prompts.
    cal = []
    ch = model.model.embed_tokens.register_forward_hook(
        lambda m, i, o: cal.append(o.float().norm(dim=-1).flatten().cpu()))
    with torch.no_grad():
        for p in prompts[: min(64, len(prompts))]:
            model(tok(p, return_tensors="pt").input_ids.to(device), use_cache=False)
    ch.remove()
    cal = torch.cat(cal).numpy()
    C = float(np.percentile(cal, args.clip_percentile))
    med = float(np.median(cal))
    z = math.sqrt(2 * math.log(1.25 / args.delta))
    print(f"[localdp] d={d} layers={layers} prompts={len(prompts)} "
          f"runtime-emb-norm median={med:.3f} C(p{args.clip_percentile})={C:.3f} "
          f"z={z:.3f}", flush=True)

    # leakage panel on (X, token ids / token embeddings). PVI is measured both
    # real and shuffle-control (Hewitt–Liang): selectivity = real − shuffled
    # subtracts the probe's memorisation/finite-sample floor — the diagnostic
    # for whether a real-PVI rise is signal or estimator floor.
    def panel(X, y):
        r_ridge = ridge_inversion(X, y, embed_table, n_train=10**9, split_mode=args.split_mode)
        top1 = r_ridge["ttrsr_top1"] if r_ridge else None
        pvi = v_information(X, y)["v_information_bits"]
        pvi_sh = v_information(X, y, control="shuffle")["v_information_bits"]
        club = club_mi_upper_bound(X, embed_table[torch.from_numpy(y)].numpy(),
                                   max_rows=args.club_max_rows)["club_mi_bits"]
        return top1, pvi, pvi_sh, club

    per_clean, ids = capture(model, tok, prompts, layers, device)
    clean = {}
    for L in layers:
        clean[L] = panel(*stack(per_clean[L], ids))
        print(f"[localdp] clean L{L:>2} top1={clean[L][0]:.3f} "
              f"pvi={clean[L][1]:.3f}b pvi_sh={clean[L][2]:.3f}b "
              f"sel={clean[L][1]-clean[L][2]:.3f}b club={clean[L][3]:.3f}b", flush=True)

    records = []
    for eps in epsilons:
        sigma = 0.0 if math.isinf(eps) else C * z / eps
        r = sigma * math.sqrt(d) / med if med else 0.0
        torch.manual_seed(args.seed + (0 if math.isinf(eps) else int(eps)))
        h = model.model.embed_tokens.register_forward_hook(InputDPCover(C, sigma))
        per_obf, _ = capture(model, tok, prompts, layers, device)
        h.remove()
        for L in layers:
            t0 = time.time()
            X, y = stack(per_obf[L], ids)
            top1, pvi, pvi_sh, club = panel(X, y)
            c = clean[L]
            rec = {"epsilon": (None if math.isinf(eps) else eps), "sigma": sigma,
                   "noise_to_signal": r, "clip_C": C, "median_norm": med,
                   "delta": args.delta, "layer": L,
                   "clean_top1": c[0], "dp_top1": top1,
                   "clean_pvi_bits": c[1], "dp_pvi_bits": pvi,
                   "clean_pvi_shuffle": c[2], "dp_pvi_shuffle": pvi_sh,
                   "clean_pvi_selectivity": c[1] - c[2],
                   "dp_pvi_selectivity": pvi - pvi_sh,
                   "clean_club_bits": c[3], "dp_club_bits": club,
                   "frac_top1": (top1 / c[0]) if (top1 is not None and c[0]) else None,
                   "secs": round(time.time() - t0, 1)}
            records.append(rec)
            es = "inf" if math.isinf(eps) else f"{eps:g}"
            print(f"[localdp] ε={es:>5} r={r:4.2f} L{L:>2} | top1={top1:.3f}({rec['frac_top1']:.2f}) "
                  f"pvi={pvi:.3f} sh={pvi_sh:.3f} sel={pvi-pvi_sh:.3f}b "
                  f"club={club:.0f}b ({rec['secs']}s)", flush=True)

    out = {"model": args.model, "corpus": args.corpus, "n_prompts": len(prompts),
           "split_mode": args.split_mode, "threat_model": "input-local-DP",
           "delta": args.delta, "clip_C": C, "median_norm": med, "d": d,
           "measures": ["ttrsr_top1", "pvi_bits", "club_mi_bits"], "records": records}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[localdp] wrote {args.out}", flush=True)
    print("\nPanel: TTRSR (attack) + PVI (V-info) + CLUB (MI upper bound) vs ε.")
    print("frac_top1 ~0.5 ⇒ ~50% of attack recovery destroyed.")


if __name__ == "__main__":
    main()
