"""Calibration — does an IT measure predict attack recovery?

The central claim under test: a representation's IT-measure value (PVI,
MDL surplus, CLUB bound) is a calibrated predictor of an inversion
attack's recovery rate across layers. This module fits that relationship
and reports the headline numbers (Spearman rank-correlation, Pearson r,
and the R² of a linear fit).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def calibrate(measure: np.ndarray, recovery: np.ndarray) -> dict[str, Any]:
    """Correlate a measure against recovery. Returns rank- and linear-
    correlation plus a least-squares fit. Drops non-finite pairs.
    """
    m = np.asarray(measure, dtype=np.float64)
    r = np.asarray(recovery, dtype=np.float64)
    ok = np.isfinite(m) & np.isfinite(r)
    m, r = m[ok], r[ok]
    n = m.size
    if n < 3:
        return {"n": int(n), "note": "need >=3 finite pairs"}
    spearman = float(stats.spearmanr(m, r).statistic)
    pearson = float(stats.pearsonr(m, r).statistic)
    slope, intercept, rval, _p, _se = stats.linregress(m, r)
    return {
        "n": int(n),
        "spearman": spearman,
        "pearson": pearson,
        "r_squared": float(rval**2),
        "slope": float(slope),
        "intercept": float(intercept),
    }


def calibrate_records(
    records: list[dict[str, Any]],
    *,
    measure_key: str,
    recovery_key: str = "primary_metric_value",
) -> dict[str, Any]:
    """Calibrate one measure column against the recovery column over a
    list of per-(layer, kind, attack) records.
    """
    measure = np.array([rec.get(measure_key, np.nan) for rec in records], dtype=np.float64)
    recovery = np.array(
        [rec.get(recovery_key, np.nan) if rec.get(recovery_key) is not None else np.nan
         for rec in records],
        dtype=np.float64,
    )
    out = calibrate(measure, recovery)
    out["measure_key"] = measure_key
    out["recovery_key"] = recovery_key
    return out
