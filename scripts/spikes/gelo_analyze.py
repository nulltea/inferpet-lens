"""Analyze the GELO sweep (Task B-5): C0 leak readout, C1 recovery vs kappa/shield,
C2 probe-vs-margin correlation (Spearman), ridge-anchor failure. Writes analysis.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

OUT = Path("refine-logs/resid-gelo")


def main() -> None:
    d = json.loads((OUT / "sweep.json").read_text())
    recs = d["records"]
    ridge = d["ridge"]

    # C2: probe (negentropy bits) vs genuine margin, across all cells with both finite.
    pairs = [(r["negentropy_bits"], r["genuine_margin"]) for r in recs
             if r["negentropy_bits"] is not None and r["genuine_margin"] is not None]
    bits = np.array([p[0] for p in pairs]); marg = np.array([p[1] for p in pairs])
    rho_all, p_all = spearmanr(bits, marg)

    # also within shield_frac == 0 (the cleanest kappa axis)
    s0 = [(r["negentropy_bits"], r["genuine_margin"], r["kappa"], r["layer"]) for r in recs
          if r["shield_frac"] == 0.0 and r["negentropy_bits"] is not None
          and r["genuine_margin"] is not None]
    rho_s0, p_s0 = spearmanr([x[0] for x in s0], [x[1] for x in s0]) if len(s0) > 2 else (None, None)

    # margin/bits vs kappa monotonicity at shield 0 (median over layers per kappa)
    kappas = sorted({r["kappa"] for r in recs})
    by_k = {}
    for k in kappas:
        cells = [r for r in recs if r["kappa"] == k and r["shield_frac"] == 0.0]
        by_k[k] = {
            "jade_p95": float(np.median([c["jade_p95"] for c in cells if c["jade_p95"] is not None])),
            "floor_p95": float(np.median([c["rand_demix_floor_p95"] for c in cells if c["rand_demix_floor_p95"] is not None])),
            "genuine_margin": float(np.median([c["genuine_margin"] for c in cells if c["genuine_margin"] is not None])),
            "negentropy_bits": float(np.median([c["negentropy_bits"] for c in cells if c["negentropy_bits"] is not None])),
            "feat_gram_relerr": float(np.median([c["feat_gram_relerr"] for c in cells if c["feat_gram_relerr"] is not None])),
        }

    # C1 anchor: ridge median p95 vs the matched floor (ridge should be <= floor)
    ridge_med = float(np.median([r["ridge_p95"] for r in ridge if r["ridge_p95"] is not None]))
    floor_med = float(np.median([r["rand_demix_floor_p95"] for r in recs
                                 if r["shield_frac"] == 0.0 and r["rand_demix_floor_p95"] is not None]))

    out = {
        "C0_feat_gram_leak_by_kappa": {str(k): by_k[k]["feat_gram_relerr"] for k in kappas},
        "C1_by_kappa_shield0": by_k,
        "C1_ridge_median_p95": ridge_med,
        "C1_random_floor_median_p95": floor_med,
        "C1_ridge_below_floor": bool(ridge_med < floor_med),
        "C2_spearman_bits_vs_margin_all": {"rho": float(rho_all), "p": float(p_all), "n": len(pairs)},
        "C2_spearman_bits_vs_margin_shield0": {
            "rho": (float(rho_s0) if rho_s0 is not None else None),
            "p": (float(p_s0) if p_s0 is not None else None), "n": len(s0)},
    }
    (OUT / "analysis.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
