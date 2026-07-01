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

from talens.attacks import ridge_inversion
from talens.probes.vinfo import v_information, v_information_retrieval  # retrieval family:
# the class-probe v_information overfits on high-d operands (shuffle floor ≈ −48b,
# non-monotonic in noise — see docs/dev/sae-attack.md diagnosis); the bounded
# ridge→embedding retrieval family has a sane ≈0 floor and is monotonic.
# v_information_capacity is the capacity-matched fix (independent token-id family,
# well-posed in d>n) — the M2 survivor candidate under test against TTRSR.
from talens.probes.vinfo_capacity import v_information_capacity
from talens.probes.club import club_mi_upper_bound


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
    ap.add_argument("--capacity-family", default="randproj_softmax",
                    choices=["pca_softmax", "randproj_softmax", "gauss", "knn"],
                    help="capacity-matched class-PVI survivor family (M2)")
    ap.add_argument("--capacity-dim", type=int, default=256)
    ap.add_argument("--capacity-l2", type=float, default=0.1,
                    help="weight decay for the capacity-PVI softmax reader (control-anchored reg)")
    ap.add_argument("--every-n", type=int, default=1,
                    help="row-stride subsample of measure inputs (forward pass stays full)")
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
    def panel(X, y) -> dict:
        r_ridge = ridge_inversion(X, y, embed_table, n_train=10**9, split_mode=args.split_mode)
        top1 = r_ridge["ttrsr_top1"] if r_ridge else None
        # retrieval-PVI (bounded, generalizes — = the attack in bits, mechanical).
        pvi = v_information_retrieval(X, y, embed_table, split_mode="row")["v_information_bits"]
        yp = y[np.random.default_rng(20260616).permutation(y.shape[0])]
        pvi_sh = v_information_retrieval(X, yp, embed_table, split_mode="row")["v_information_bits"]
        club = club_mi_upper_bound(X, embed_table[torch.from_numpy(y)].numpy(),
                                   max_rows=args.club_max_rows)["club_mi_bits"]
        # class-PVI (independent but overfits in d>n) — the broken baseline.
        cls = v_information(X, y)["v_information_bits"]
        cls_sh = v_information(X, y, control="shuffle")["v_information_bits"]
        # capacity-matched class-PVI (independent AND well-posed) — the candidate.
        cap = v_information_capacity(X, y, family=args.capacity_family, dim=args.capacity_dim,
                                     l2=args.capacity_l2)
        cap_sh = v_information_capacity(X, y, family=args.capacity_family, dim=args.capacity_dim,
                                        l2=args.capacity_l2, control="shuffle")
        return {"top1": top1, "pvi": pvi, "pvi_sh": pvi_sh, "club": club,
                "cls": cls, "cls_sh": cls_sh,
                "cap": cap["v_information_bits"], "cap_sh": cap_sh["v_information_bits"],
                "cap_acc": cap["reader_top1_acc"]}

    def stack_sub(mats, ids_list):
        Xs, ys = stack(mats, ids_list)
        return (Xs[:: args.every_n], ys[:: args.every_n]) if args.every_n > 1 else (Xs, ys)

    per_clean, ids = capture(model, tok, prompts, layers, device)
    clean = {}
    for L in layers:
        clean[L] = panel(*stack_sub(per_clean[L], ids))
        c = clean[L]
        print(f"[localdp] clean L{L:>2} top1={c['top1']:.3f} "
              f"retr-pvi={c['pvi']:.2f}b cls-pvi={c['cls']:.2f}(sh{c['cls_sh']:.1f}) "
              f"cap-pvi={c['cap']:.2f}(sh{c['cap_sh']:.2f}) club={c['club']:.0f}b", flush=True)

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
            X, y = stack_sub(per_obf[L], ids)
            p = panel(X, y)
            c = clean[L]
            rec = {"epsilon": (None if math.isinf(eps) else eps), "sigma": sigma,
                   "noise_to_signal": r, "clip_C": C, "median_norm": med,
                   "delta": args.delta, "layer": L,
                   "clean_top1": c["top1"], "dp_top1": p["top1"],
                   "clean_retr_pvi_bits": c["pvi"], "dp_retr_pvi_bits": p["pvi"],
                   "clean_retr_pvi_shuffle": c["pvi_sh"], "dp_retr_pvi_shuffle": p["pvi_sh"],
                   "clean_class_pvi_bits": c["cls"], "dp_class_pvi_bits": p["cls"],
                   "clean_class_pvi_shuffle": c["cls_sh"], "dp_class_pvi_shuffle": p["cls_sh"],
                   "capacity_family": args.capacity_family, "capacity_dim": args.capacity_dim,
                   "clean_cap_acc": c["cap_acc"], "dp_cap_acc": p["cap_acc"],
                   "clean_cap_pvi_bits": c["cap"], "dp_cap_pvi_bits": p["cap"],
                   "clean_cap_pvi_shuffle": c["cap_sh"], "dp_cap_pvi_shuffle": p["cap_sh"],
                   "clean_cap_pvi_selectivity": c["cap"] - c["cap_sh"],
                   "dp_cap_pvi_selectivity": p["cap"] - p["cap_sh"],
                   "clean_club_bits": c["club"], "dp_club_bits": p["club"],
                   "frac_top1": (p["top1"] / c["top1"]) if (p["top1"] is not None and c["top1"]) else None,
                   "secs": round(time.time() - t0, 1)}
            records.append(rec)
            es = "inf" if math.isinf(eps) else f"{eps:g}"
            print(f"[localdp] ε={es:>5} r={r:4.2f} L{L:>2} | top1={p['top1']:.3f}({rec['frac_top1']:.2f}) "
                  f"cap={p['cap']:.2f}(sh{p['cap_sh']:.2f}) cls={p['cls']:.2f} "
                  f"retr={p['pvi']:.2f} club={p['club']:.0f}b ({rec['secs']}s)", flush=True)

    out = {"model": args.model, "corpus": args.corpus, "n_prompts": len(prompts),
           "split_mode": args.split_mode, "threat_model": "input-local-DP",
           "delta": args.delta, "clip_C": C, "median_norm": med, "d": d,
           "capacity_family": args.capacity_family, "capacity_dim": args.capacity_dim,
           "measures": ["ttrsr_top1", "retr_pvi_bits", "class_pvi_bits",
                        "cap_pvi_bits", "club_mi_bits"], "records": records}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[localdp] wrote {args.out}", flush=True)
    print("\nPanel: TTRSR (attack) + PVI (V-info) + CLUB (MI upper bound) vs ε.")
    print("frac_top1 ~0.5 ⇒ ~50% of attack recovery destroyed.")


if __name__ == "__main__":
    main()
