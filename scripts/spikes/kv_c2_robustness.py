"""C2 robustness add-on for Task B-1 (kv-accumulation).

Addresses the result-to-claim jury's main statistical concern: the n=9
negentropy->genuine-margin Spearman 0.92 is thin and the rank ordering is
dominated by 3 activation-kind families. Recomputes the SAME deterministic
(negentropy, genuine_margin) pairs already used by kv_bss_analysis.py and adds:
  - exact permutation p-value for the n=9 Spearman (all 9! relabelings),
  - leave-one-kind-out Spearman (drop each family -> 6 remaining cells),
  - within-kind-family Spearman (3 cells each),
  - across-family-MEANS Spearman (the coarse 3-point ordering the jury flagged).
No attack is re-run; this is pure numpy on cached artifacts -> CPU is correct.
"""
import json
from itertools import permutations
from pathlib import Path

import numpy as np


def _rankdata(a):
    """Average-rank (ties handled) — matches scipy.stats.rankdata default."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1)
    # average ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


class _Rho:
    def __init__(self, s):
        self.statistic = s


def spearmanr(x, y):
    rx, ry = _rankdata(x), _rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return _Rho(float("nan"))
    return _Rho(float(np.corrcoef(rx, ry)[0, 1]))

ROOT = Path("refine-logs/kv-accumulation")
pilot = json.loads((ROOT / "pilot_dev24.json").read_text())
analysis = json.loads((ROOT / "analysis_b3.json").read_text())

neg_by_cell = {(d["kind"], d["layer"]): d["negentropy_bits"] for d in pilot["negentropy"]}

cells, xs_neg, ys_margin = [], [], []
for r in analysis["jade_proper_floor"]:
    key = (r["kind"], r["layer"])
    if key in neg_by_cell and neg_by_cell[key] is not None:
        cells.append(key)
        xs_neg.append(neg_by_cell[key])
        ys_margin.append(r["genuine_margin"])

x = np.asarray(xs_neg, float)
y = np.asarray(ys_margin, float)
n = len(x)
rho_obs = spearmanr(x, y).statistic

# Exact two-sided permutation p-value: 9! = 362880 relabelings of y.
count_ge = 0
total = 0
for perm in permutations(range(n)):
    rho = spearmanr(x, y[list(perm)]).statistic
    if abs(rho) >= abs(rho_obs) - 1e-12:
        count_ge += 1
    total += 1
p_perm = count_ge / total

# Leave-one-kind-out: drop each family, Spearman on the remaining 6 cells.
kinds = sorted({k for k, _ in cells})
loko = {}
for drop in kinds:
    idx = [i for i, (k, _) in enumerate(cells) if k != drop]
    loko[drop] = {"n": len(idx),
                  "spearman": float(spearmanr(x[idx], y[idx]).statistic)}

# Within each family (3 cells across layers).
within = {}
for fam in kinds:
    idx = [i for i, (k, _) in enumerate(cells) if k == fam]
    within[fam] = {"n": len(idx),
                   "spearman": float(spearmanr(x[idx], y[idx]).statistic)}

# Across-family means (the coarse 3-point ordering the jury called out).
fam_means_x, fam_means_y = [], []
for fam in kinds:
    idx = [i for i, (k, _) in enumerate(cells) if k == fam]
    fam_means_x.append(float(np.mean(x[idx])))
    fam_means_y.append(float(np.mean(y[idx])))
across = {"n": len(kinds),
          "spearman": float(spearmanr(fam_means_x, fam_means_y).statistic),
          "family_order_negentropy": dict(zip(kinds, fam_means_x)),
          "family_order_margin": dict(zip(kinds, fam_means_y))}

out = {
    "n": n,
    "cells": [list(c) for c in cells],
    "spearman_obs": float(rho_obs),
    "permutation_p_two_sided": float(p_perm),
    "permutation_total": total,
    "leave_one_kind_out": loko,
    "within_family": within,
    "across_family_means": across,
}
(ROOT / "c2_robustness.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
