"""The standardized utility-probe result type + the retention-threshold helper.

A *utility probe* measures how much a defense degrades the model's usefulness, referenced to the
no-defense (clean) baseline. The point of this module is COMPARABILITY across schemes: every utility
probe — for any defense (Gaussian local-DP, dχ-privacy SnD, an obfuscation cover, …) — returns the
same :class:`UtilityResult` whose ``retention`` ∈ [0, 1] is the single axis you line schemes up on.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class UtilityResult:
    """One utility measurement, standardized so it compares across defenses.

    ``retention`` is the canonical comparable scalar: 1.0 = no utility loss, 0.0 = utility destroyed.
    It is derived from the raw ``clean``/``defended`` values per metric direction:

      * higher-is-better raw metric (accuracy, agreement, cosine): ``retention = defended / clean``;
      * lower-is-better raw metric (perplexity): ``retention = clean / defended``.

    Report ``retention`` to compare schemes; keep ``clean``/``defended``/``extra`` for the raw story.
    """

    metric: str               # e.g. "next_token_accuracy", "perplexity", "output_agreement", "embedding_cosine"
    clean: float              # raw metric value at the no-defense baseline
    defended: float           # raw metric value under the defense
    retention: float          # canonical comparable scalar ∈ [0, 1]; 1.0 = lossless
    higher_is_better: bool    # direction of the RAW metric (acc ↑, ppl ↓)
    extra: dict = field(default_factory=dict)   # probe-specific extras (degradation, n_tokens, mse, …)

    def as_dict(self) -> dict:
        return {"metric": self.metric, "clean": self.clean, "defended": self.defended,
                "retention": self.retention, "higher_is_better": self.higher_is_better, **self.extra}


def _retention(clean: float, defended: float, higher_is_better: bool) -> float:
    """Canonical retention ∈ [0, 1] from a clean/defended raw pair (clamped)."""
    if higher_is_better:
        r = defended / clean if clean else 0.0
    else:  # lower-is-better (perplexity): clean is the floor, defended ≥ clean
        r = clean / defended if defended else 0.0
    return float(min(max(r, 0.0), 1.0))


def retention_thresholds(xs, retentions, targets=(0.90, 0.80, 0.50)) -> dict:
    """Privacy-budget value at which retention crosses each target (e.g. ε/η for −10/−20/−50% utility).

    ``xs`` and ``retentions`` are paired sweep points (None entries skipped). Assumes the sweep is
    ordered so retention is monotone in x; crossings found by log-linear interpolation in x. Returns
    ``{"retention_90pct": x, "retention_80pct": x, "retention_50pct": x}`` (None where never crossed).
    """
    pairs = [(float(x), float(r)) for x, r in zip(xs, retentions) if x is not None and r is not None]
    out = {}
    for t in targets:
        cross = None
        for (xa, ra), (xb, rb) in zip(pairs, pairs[1:]):
            lo, hi = sorted((ra, rb))
            if lo <= t <= hi and ra != rb:                 # t lies between the two retentions
                frac = (ra - t) / (ra - rb)
                cross = math.exp(math.log(xa) + frac * (math.log(xb) - math.log(xa)))
                break
        out[f"retention_{int(round(t * 100))}pct"] = cross
    return out
