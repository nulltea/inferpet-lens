"""PriPert split-inference defense sweep (Task 6, resid-split).

Reuses the Task-4 inversion pipeline (capture once from cache; ridge/nn/mlp2 token-embedding
inverters with a vocab-disjoint split + label-shuffle control + bootstrap CI on selectivity)
and applies the PriPert defense (``scripts/defenses/pripert.py``: per-row top-ρ magnitude
sparsification + additive perturbation δ) to the observed residual ``U`` before the attacks and
the MATCHED probe see it. The matched, attack-INDEPENDENT probe is ``spectral_channel_mi``:
``I_G = ½Σlog2(1+λ_i/σ²)`` bits on cov(Sparsify_ρ(H)) at the perturbation σ, plus the Fano
recovery ceiling — the empirical realization of PriPert's converse (Thm 1).

Noise floor: σ is referenced to the PLAINTEXT row-RMS per layer (``β·meanRMS(H)``), held fixed
across the ρ-axis, so sparsification reduces signal against a fixed floor and the converse is
comparable across cells (otherwise σ co-scales with ρ and the probe cannot see sparsification).

Measurement loop:
  - C1: best-inverter vocab-disjoint selectivity falls as ρ↓ / β↑ (defended corner CI ∋ 0).
  - C2: does I_G (bits) track best recovery across all (layer,ρ,β) cells? (pooled Spearman ≥0.6)
  - C3: per cell, empirical ttrsr_top1 ≤ fano_exact_ceiling (no converse violation).
  - decision: does mlp2 beat ridge where I_G≈0 (probe–attack gap)?

Usage (always GPU-wrapped):
  scripts/run_in_rocm.sh python3 scripts/spikes/pripert_sweep.py \
      --corpus corpora/release-gate-512.txt --layers 8 --rhos 1.0 0.5 0.25 0.1 0.05 \
      --beta-fixed 0.5 --beta-layers 8 --betas 0.0 0.25 0.5 1.0 --rho-fixed 0.25 \
      --out refine-logs/resid-split/runs/pilot/pripert_pilot.json
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for scripts.defenses

from defenses.pripert import perturbation_sigma, pripert_apply, sparsify_rows  # noqa: E402
from talens.attacks import learned_inversion, nn_inversion, ridge_inversion  # noqa: E402
from talens.capture.capture import load_or_capture  # noqa: E402
from talens.probes import club_mi_upper_bound  # noqa: E402
from talens.probes.spectral_channel_mi import spectral_channel_mi  # noqa: E402

INVERTERS = {"ridge": ridge_inversion, "nn": nn_inversion, "mlp2": learned_inversion}


def _spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return None
    ar = np.argsort(np.argsort(a[ok])); br = np.argsort(np.argsort(b[ok]))
    if ar.std() == 0 or br.std() == 0:
        return None
    return float(np.corrcoef(ar, br)[0, 1])


def _boot_diff_ci(real_hits, shuf_hits, n=2000, seed=0):
    if real_hits is None or shuf_hits is None or len(real_hits) == 0 or len(shuf_hits) == 0:
        return None, None
    rng = np.random.default_rng(seed)
    r = np.asarray(real_hits, float); s = np.asarray(shuf_hits, float)
    diffs = np.empty(n)
    for i in range(n):
        diffs[i] = r[rng.integers(0, len(r), len(r))].mean() - s[rng.integers(0, len(s), len(s))].mean()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def run_cell(X, y, embed_table, *, layer, rho, beta, sigma_ref, mode, args):
    """One (layer, ρ, β) cell: apply PriPert, run inverters + matched probe + CLUB."""
    sigma = 0.0 if beta == 0.0 else float(beta * sigma_ref)
    U, sigma_used = pripert_apply(X, rho=rho, beta=beta, mode=mode, seed=args.seed, sigma=sigma)
    U = U.astype(np.float32)
    S = sparsify_rows(X, rho)  # clean signal (no δ) for the matched-probe covariance

    rec = {"layer": int(layer), "rho": float(rho), "beta": float(beta),
           "sigma": float(sigma_used), "n_rows": int(U.shape[0]), "d": int(U.shape[1]),
           "inverters": {}}
    pool_size = None
    for name, fn in INVERTERS.items():
        kw = {"hidden": args.mlp_hidden, "epochs": args.mlp_epochs} if name == "mlp2" else {}
        real = fn(U, y, embed_table, split_mode="vocab", **kw)
        shuf = fn(U, y, embed_table, split_mode="vocab", control="shuffle", **kw)
        if real is None:
            rec["inverters"][name] = {"note": "too few rows"}
            continue
        pool_size = real.get("candidate_pool_size", pool_size)
        sel = real["ttrsr_top1"] - (shuf["ttrsr_top1"] if shuf else 0.0)
        lo, hi = _boot_diff_ci(real.get("top1_hits"), shuf.get("top1_hits") if shuf else None)
        rec["inverters"][name] = {
            "ttrsr_top1": real["ttrsr_top1"], "ttrsr_top10": real["ttrsr_top10"],
            "embedding_cosine": real["embedding_cosine_similarity"],
            "ttrsr_top1_shuffle": shuf["ttrsr_top1"] if shuf else None,
            "selectivity": sel, "selectivity_ci95": [lo, hi],
            "n_train": real["n_train"], "n_test": real["n_test"],
        }
    # MATCHED probe: spectral channel-MI on cov(S) at the perturbation σ.
    M = pool_size or args.pool_size
    HX = math.log2(M)                       # secret entropy: one token over the pool (bits)
    cov = np.cov(S.astype(np.float64), rowvar=False)
    spec = spectral_channel_mi(cov=cov, sigma=float(sigma_used), center=False,
                               H_X=HX, H_e0=HX, n_tokens=1, vocab=int(M))
    Y = embed_table[torch.from_numpy(y)].numpy()
    club = club_mi_upper_bound(U, Y, steps=args.club_steps, max_rows=args.club_max_rows)
    rec["probes"] = {
        "i_g_bits": (None if not math.isfinite(spec["i_g_bits"]) else float(spec["i_g_bits"])),
        "i_g_is_inf": (not math.isfinite(spec["i_g_bits"])),
        "accessible_bit_ceiling": float(spec["accessible_bit_ceiling"]),
        "fano_recovery_ceiling": (None if spec["fano_exact_ceiling"] is None
                                  else float(spec["fano_exact_ceiling"])),
        "d_eff": int(spec["d_eff"]),
        "H_X_bits": float(HX), "pool_size": int(M),
        "club_mi_bits": club.get("club_mi_bits"),
    }
    best = max((rec["inverters"][k].get("ttrsr_top1", 0) or 0) for k in rec["inverters"])
    # C3 converse check: empirical top-1 must not exceed the Fano accuracy ceiling.
    fc = rec["probes"]["fano_recovery_ceiling"]
    rec["converse_ok"] = (fc is None) or (best <= fc + 1e-9)
    print(f"[L{layer} rho={rho:g} beta={beta:g}] best={best:.3f} "
          f"ridge={rec['inverters'].get('ridge',{}).get('ttrsr_top1')} "
          f"mlp2={rec['inverters'].get('mlp2',{}).get('ttrsr_top1')} "
          f"I_G={rec['probes']['i_g_bits']} fano_ceil={fc} "
          f"club={rec['probes']['club_mi_bits']} converse_ok={rec['converse_ok']}", flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[8])
    ap.add_argument("--rhos", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.1, 0.05])
    ap.add_argument("--beta-fixed", type=float, default=0.5, help="β for the ρ-sweep")
    ap.add_argument("--beta-layers", type=int, nargs="*", default=[8], help="layers for β-sweep")
    ap.add_argument("--betas", type=float, nargs="*", default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--rho-fixed", type=float, default=0.25, help="ρ for the β-sweep")
    ap.add_argument("--mode", default="gauss", choices=["gauss", "pca"])
    ap.add_argument("--mlp-epochs", type=int, default=150)
    ap.add_argument("--mlp-hidden", type=int, default=1024)
    ap.add_argument("--club-steps", type=int, default=150)
    ap.add_argument("--club-max-rows", type=int, default=2500)
    ap.add_argument("--pool-size", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=20260624)
    ap.add_argument("--cache-dir", default="results/capture_cache")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[cuda] available={torch.cuda.is_available()}", flush=True)
    prompts = [ln.strip() for ln in Path(args.corpus).read_text().splitlines() if ln.strip()]
    grid = sorted(set(args.layers) | set(args.beta_layers))
    t0 = time.time()
    cap, embed_table, source = load_or_capture(
        args.model, prompts, capture_layers=grid, kinds=("resid_post",), cache_dir=args.cache_dir)
    print(f"[capture] source={source} layers_present={cap.layers('resid_post')} "
          f"({time.time()-t0:.1f}s)", flush=True)

    # Build the cell list: ρ-sweep @ β_fixed over all layers ∪ β-sweep @ ρ_fixed over beta_layers.
    cells = []
    for L in args.layers:
        for r in args.rhos:
            cells.append((L, r, args.beta_fixed))
    for L in args.beta_layers:
        for b in args.betas:
            cells.append((L, args.rho_fixed, b))
    # always include a plaintext anchor per layer
    for L in grid:
        cells.append((L, 1.0, 0.0))
    seen = set(); uniq = []
    for c in cells:
        if c not in seen:
            seen.add(c); uniq.append(c)
    cells = uniq

    # per-layer plaintext RMS reference (the fixed σ floor)
    sigma_ref = {}
    Xs = {}
    for L in grid:
        if ("resid_post", L) not in cap.operands:
            continue
        X, y, _ = cap.stack("resid_post", L)
        Xs[L] = (np.asarray(X, dtype=np.float64), np.asarray(y))
        sigma_ref[L] = perturbation_sigma(Xs[L][0], 1.0)  # meanRMS(plaintext)
    print(f"[sigma_ref] {{ {', '.join(f'L{L}:{sigma_ref[L]:.3g}' for L in sigma_ref)} }}", flush=True)

    records = []
    for (L, r, b) in cells:
        if L not in Xs:
            print(f"[skip] layer {L} not captured", flush=True); continue
        X, y = Xs[L]
        tl = time.time()
        rec = run_cell(X, y, embed_table, layer=L, rho=r, beta=b,
                       sigma_ref=sigma_ref[L], mode=args.mode, args=args)
        rec["sec"] = round(time.time() - tl, 1)
        records.append(rec)

    def best_rec(r):
        return max((r["inverters"].get(k, {}).get("ttrsr_top1") or 0) for k in r["inverters"])

    def best_sel(r):
        return max((r["inverters"].get(k, {}).get("selectivity") or -9) for k in r["inverters"])

    ig = [r["probes"]["i_g_bits"] for r in records]
    club = [r["probes"]["club_mi_bits"] for r in records]
    brec = [best_rec(r) for r in records]
    bsel = [best_sel(r) for r in records]
    ridge_rec = [r["inverters"].get("ridge", {}).get("ttrsr_top1") for r in records]
    mlp2_rec = [r["inverters"].get("mlp2", {}).get("ttrsr_top1") for r in records]
    # decision: mlp2 − ridge selectivity gap where probe says capacity gone (I_G small)
    gap = [{"layer": r["layer"], "rho": r["rho"], "beta": r["beta"], "i_g_bits": r["probes"]["i_g_bits"],
            "mlp2_minus_ridge_sel": (
                (r["inverters"].get("mlp2", {}).get("selectivity") - r["inverters"].get("ridge", {}).get("selectivity"))
                if (r["inverters"].get("mlp2", {}).get("selectivity") is not None
                    and r["inverters"].get("ridge", {}).get("selectivity") is not None) else None)}
           for r in records]
    converse_violations = [
        {"layer": r["layer"], "rho": r["rho"], "beta": r["beta"],
         "best_top1": best_rec(r), "fano_ceiling": r["probes"]["fano_recovery_ceiling"]}
        for r in records if not r["converse_ok"]]

    corr = {
        "spearman_ig_vs_bestrec": _spearman(ig, brec),
        "spearman_ig_vs_bestsel": _spearman(ig, bsel),
        "spearman_club_vs_bestrec": _spearman(club, brec),
        "spearman_ig_vs_ridge": _spearman(ig, ridge_rec),
        "spearman_ig_vs_mlp2": _spearman(ig, mlp2_rec),
        "n_cells": len(records),
        "n_finite_ig": int(sum(1 for v in ig if v is not None)),
    }
    out = {
        "model": args.model, "corpus": args.corpus, "config": vars(args),
        "cells": [list(c) for c in cells], "records": records,
        "cross_sweep_correlation": corr, "learned_vs_ridge_gap": gap,
        "converse_violations": converse_violations,
        "n_converse_violations": len(converse_violations),
    }
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2, default=float))
    print(f"[done] wrote {outp} ({time.time()-t0:.1f}s total)", flush=True)
    print(f"[corr] {json.dumps(corr, indent=2)}", flush=True)
    print(f"[converse] violations={len(converse_violations)}", flush=True)


if __name__ == "__main__":
    main()
