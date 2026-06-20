#!/usr/bin/env python3
"""Faithfulness analysis for a localdp_runner sweep JSON (Round-2 rigor).

Addresses the reviewer's W2/W3: is the capacity-PVI↔TTRSR rank correlation
robust, or an artifact of n=7 / a single collapse point / the shared noise knob?

For each measure vs TTRSR it reports, pooling across whatever (layer, ε) records
are present:
  * pooled Spearman (+ permutation p)
  * per-layer Spearman
  * leave-one-out Spearman range (min..max) — sensitivity to any single point
  * PARTIAL Spearman controlling for the noise ratio r — does the measure track
    TTRSR beyond what the noise knob alone explains?
  * bootstrap 95% CI on the pooled Spearman
"""
from __future__ import annotations

import argparse
import json
from itertools import permutations

import numpy as np
from scipy import stats

MEASURES = {
    "cap_reader_accuracy": "dp_cap_acc",
    "cap_pvi_raw": "dp_cap_pvi_bits",
    "cap_pvi_selectivity": "dp_cap_pvi_selectivity",
    "class_pvi": "dp_class_pvi_bits",
    "retrieval_pvi": "dp_retr_pvi_bits",
    "club": "dp_club_bits",
}


def _sp(a, b):
    return stats.spearmanr(a, b).statistic


def _perm_p(a, b):
    n = len(a)
    if n > 8:
        return stats.spearmanr(a, b).pvalue
    obs = abs(_sp(a, b))
    rb = stats.rankdata(b)
    cnt = tot = 0
    for p in permutations(range(n)):
        tot += 1
        if abs(_sp(stats.rankdata(a), rb[list(p)])) >= obs - 1e-12:
            cnt += 1
    return cnt / tot


def _partial_sp(m, t, r):
    """Partial Spearman of m,t controlling for r (correlation of rank residuals)."""
    rm, rt, rr = stats.rankdata(m), stats.rankdata(t), stats.rankdata(r)
    r_mt = np.corrcoef(rm, rt)[0, 1]
    r_mr = np.corrcoef(rm, rr)[0, 1]
    r_tr = np.corrcoef(rt, rr)[0, 1]
    denom = np.sqrt(max(1e-12, (1 - r_mr ** 2) * (1 - r_tr ** 2)))
    return (r_mt - r_mr * r_tr) / denom


def _bootstrap_ci(m, t, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(m)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.unique(t[idx]).size < 3 or np.unique(m[idx]).size < 3:
            continue
        vals.append(_sp(m[idx], t[idx]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (None, None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    args = ap.parse_args()
    d = json.load(open(args.json))
    recs = d["records"]
    layers = sorted({r["layer"] for r in recs})
    t = np.array([r["dp_top1"] for r in recs], float)
    rr = np.array([r["noise_to_signal"] for r in recs], float)
    print(f"file={args.json}  n_records={len(recs)}  layers={layers}  "
          f"cap_family={d.get('capacity_family')} dim={d.get('capacity_dim')}")
    print(f"{'measure':22s} {'pooledρ':>8s} {'permp':>7s} {'partialρ|r':>11s} "
          f"{'LOO range':>16s} {'boot95CI':>16s}  per-layer ρ")
    for name, key in MEASURES.items():
        m = np.array([r.get(key, np.nan) for r in recs], float)
        if np.isnan(m).any():
            print(f"{name:22s}  (missing)")
            continue
        pooled = _sp(m, t)
        pp = _perm_p(m, t) if len(recs) <= 8 else stats.spearmanr(m, t).pvalue
        partial = _partial_sp(m, t, rr)
        loo = [_sp(np.delete(m, i), np.delete(t, i)) for i in range(len(m))]
        ci = _bootstrap_ci(m, t)
        per_layer = {L: round(_sp(np.array([r.get(key) for r in recs if r["layer"] == L], float),
                                  np.array([r["dp_top1"] for r in recs if r["layer"] == L], float)), 2)
                     for L in layers}
        ci_s = f"[{ci[0]:.2f},{ci[1]:.2f}]" if ci[0] is not None else "n/a"
        print(f"{name:22s} {pooled:8.3f} {pp:7.3f} {partial:11.3f} "
              f"[{min(loo):.2f},{max(loo):.2f}]{'':4s} {ci_s:>16s}  {per_layer}")

    # --- W5 independence / redundancy: is cap-PVI just retrieval-PVI? ---
    cap = np.array([r.get("dp_cap_pvi_selectivity", np.nan) for r in recs], float)
    retr = np.array([r.get("dp_retr_pvi_bits", np.nan) for r in recs], float)
    if not (np.isnan(cap).any() or np.isnan(retr).any()):
        print("\n[W5 independence] cap-PVI(selectivity) vs retrieval-PVI (the attack in bits):")
        print(f"  ρ(cap, retr)                      = {_sp(cap, retr):+.3f}   (want < ~0.9: not collinear)")
        print(f"  partial ρ(cap, TTRSR | retr)      = {_partial_sp(cap, t, retr):+.3f}   "
              f"(>0 ⇒ cap adds prediction beyond the attack measure)")
        print(f"  partial ρ(cap, TTRSR | r)         = {_partial_sp(cap, t, rr):+.3f}   "
              f"(>0 ⇒ tracks leakage beyond the noise knob — needs layer variation)")


if __name__ == "__main__":
    main()
