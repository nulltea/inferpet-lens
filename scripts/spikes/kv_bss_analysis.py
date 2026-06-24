"""B3 analysis for Task B-1: the CORRECT floor + C1 slope + C2 correlation.

The pilot's ``jd_floor`` compares a recovered source against *unrelated* Gaussian
ground truth (p95 ~ 0.155). That is the wrong control under Identity (U == H): any
demixing ``B`` of ``U`` yields rows in the row-span of ``U == H``, so the Hungarian
p95-cosine against H's own rows is high *regardless of whether the joint-diag found
the right rotation*. The apples-to-apples floor is a RANDOM-demixing floor: same data,
same whiten+Hungarian pipeline, but a random orthogonal rotation in place of the
joint-diagonalisation. Genuine separation = margin of the real attack ABOVE this floor.

Outputs refine-logs/kv-accumulation/analysis_b3.json (CPU; reads the cached capture).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from talens.attacks import bss
from talens.capture.cache import load_capture

OUT = Path("refine-logs/kv-accumulation")
DEV24 = Path("results/capture_cache/capture-3e3a86a58abf0727.pt")
LAYERS = [0, 12, 20]
KINDS = ["kq", "kqv_out", "resid_post"]
TVALS = (1, 2, 4, 8)
MAX_DIM, MAX_FEAT = 64, 256


def _rand_orth(m: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.standard_normal((m, m))
    q, _ = np.linalg.qr(a)
    return q


def jade_proper_floor(cap, kind, layer, n_seeds=5, seed=0) -> dict:
    """Random-rotation floor: same whitened data, random orthogonal demixing, vs true h."""
    rng = np.random.default_rng(seed)
    real, floor = [], []
    for h, u in bss._operands(cap, kind, layer, None):
        if u.shape != h.shape or u.size == 0:
            continue
        u, h = bss._subsample(u, h, MAX_DIM, MAX_FEAT, rng)
        s = u.shape[0]
        if s < 4 or u.shape[1] < 2 * s:
            continue
        try:
            y, w = bss._whiten(u, s)
            q = bss._build_cumulants(y)
            rot = bss._joint_diag(q)
        except np.linalg.LinAlgError:
            continue
        real.append(bss._p95_cosine_with_hungarian((rot.T @ w) @ u, h))
        # random-rotation floor (mean over seeds): demixing irrelevant, subspace held fixed
        fl = [bss._p95_cosine_with_hungarian((_rand_orth(s, rng).T @ w) @ u, h) for _ in range(n_seeds)]
        floor.append(float(np.mean(fl)))
    if not real:
        return {"kind": kind, "layer": layer, "n": 0}
    return {
        "kind": kind, "layer": layer, "n": len(real),
        "jade_p95": float(np.median(real)),
        "proper_floor_p95": float(np.median(floor)),
        "genuine_margin": float(np.median(real) - np.median(floor)),
    }


def jd_proper_floor(cap, kind, layer, n_seeds=3, seed=0) -> dict:
    """Per-T random-rotation floor on the SAME real stacks (demixing irrelevant)."""
    rng = np.random.default_rng(seed)
    pairs = []
    for h, u in bss._operands(cap, kind, layer, None):
        if u.shape != h.shape or u.size == 0:
            continue
        u, h = bss._subsample(u, h, MAX_DIM, MAX_FEAT, rng)
        if u.shape[0] < 4:
            continue
        pairs.append((u, h))
    if not pairs:
        return {"kind": kind, "layer": layer, "floor_per_t": {}}
    ref = pairs[0][0].shape
    pairs = [p for p in pairs if p[0].shape == ref]
    s = ref[0]
    floor_per_t: dict[int, list[float]] = {t: [] for t in TVALS}
    for t_target in TVALS:
        for start in range(0, len(pairs) - t_target + 1, t_target):
            sl = pairs[start: start + t_target]
            u_stack = np.stack([p[0] for p in sl], axis=0)
            h_stack = np.stack([p[1] for p in sl], axis=0)
            try:
                _ys, w = bss._whiten_stack(u_stack)
            except np.linalg.LinAlgError:
                continue
            for _ in range(n_seeds):
                b = _rand_orth(s, rng).T @ w
                for t in range(u_stack.shape[0]):
                    p95 = bss._p95_cosine_with_hungarian(b @ u_stack[t], h_stack[t])
                    if p95 == p95:
                        floor_per_t[t_target].append(p95)
    return {
        "kind": kind, "layer": layer,
        "floor_per_t": {int(t): (float(np.median(v)) if v else None) for t, v in floor_per_t.items()},
    }


def slope_log2t(per_t: dict) -> float | None:
    xs, ys = [], []
    for t, v in per_t.items():
        if v is None:
            continue
        xs.append(np.log2(int(t)))
        ys.append(float(v))
    if len(xs) < 2:
        return None
    return float(np.polyfit(xs, ys, 1)[0])


def main() -> None:
    pilot = json.loads((OUT / "pilot_dev24.json").read_text())
    cap, _ = load_capture(DEV24)

    jade_floor = [jade_proper_floor(cap, k, l) for k in KINDS for l in LAYERS]
    jade_floor = [r for r in jade_floor if r.get("n", 0) > 0]
    jd_floor_proper = [jd_proper_floor(cap, k, l) for k in KINDS for l in LAYERS]

    # C1: jd slope vs log2 T per cell (real attack) + does any cell beat its proper floor at any T
    jd_slopes = []
    jd_by_cell = {(d["kind"], d["layer"]): d for d in pilot["jd"]}
    jdf_by_cell = {(d["kind"], d["layer"]): d for d in jd_floor_proper}
    for (kind, layer), d in jd_by_cell.items():
        fl = jdf_by_cell.get((kind, layer), {}).get("floor_per_t", {})
        margins = {}
        for t, v in d["p95_per_t"].items():
            f = fl.get(int(t)) if isinstance(fl, dict) else None
            if v is not None and f is not None:
                margins[int(t)] = float(v) - float(f)
        jd_slopes.append({
            "kind": kind, "layer": layer,
            "p95_slope_log2t": slope_log2t(d["p95_per_t"]),
            "margin_over_proper_floor": margins,
            "max_margin": (max(margins.values()) if margins else None),
        })

    # C2: probe vs recovery across cells.
    # jade channel: negentropy_bits vs jade genuine_margin
    neg_by_cell = {(d["kind"], d["layer"]): d["negentropy_bits"] for d in pilot["negentropy"]}
    xs_neg, ys_margin, ys_raw = [], [], []
    for r in jade_floor:
        key = (r["kind"], r["layer"])
        if key in neg_by_cell and neg_by_cell[key] is not None:
            xs_neg.append(neg_by_cell[key])
            ys_margin.append(r["genuine_margin"])
            ys_raw.append(r["jade_p95"])

    def corr(x, y):
        if len(x) < 3:
            return {"n": len(x), "spearman": None, "pearson": None}
        return {"n": len(x),
                "spearman": float(spearmanr(x, y).statistic),
                "pearson": float(pearsonr(x, y)[0])}

    # jd channel: shared_spectral_capacity(T) vs jd p95(T), pooled across cells×T
    cap_by_cell = {(d["kind"], d["layer"]): d["cap_per_t"] for d in pilot["shared_spectral_capacity"]}
    xs_cap, ys_jd = [], []
    for (kind, layer), d in jd_by_cell.items():
        capd = cap_by_cell.get((kind, layer), {})
        for t, v in d["p95_per_t"].items():
            c = capd.get(t) if isinstance(capd, dict) else None
            if v is not None and c is not None:
                xs_cap.append(float(c))
                ys_jd.append(float(v))

    out = {
        "jade_proper_floor": jade_floor,
        "jd_proper_floor": jd_floor_proper,
        "C1_jd_accumulation": {
            "per_cell": jd_slopes,
            "median_slope_log2t": float(np.median([s["p95_slope_log2t"] for s in jd_slopes
                                                   if s["p95_slope_log2t"] is not None])),
            "max_genuine_margin_any_cell_any_t": float(max(
                (s["max_margin"] for s in jd_slopes if s["max_margin"] is not None), default=float("nan"))),
        },
        "C2_probe_vs_recovery": {
            "jade__negentropy_vs_genuine_margin": corr(xs_neg, ys_margin),
            "jade__negentropy_vs_raw_p95": corr(xs_neg, ys_raw),
            "jd__shared_capacity_vs_p95": corr(xs_cap, ys_jd),
        },
        "summary": {
            "jade_median_raw_p95": float(np.median([r["jade_p95"] for r in jade_floor])),
            "jade_median_proper_floor": float(np.median([r["proper_floor_p95"] for r in jade_floor])),
            "jade_median_genuine_margin": float(np.median([r["genuine_margin"] for r in jade_floor])),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis_b3.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out["summary"], indent=2))
    print("C1 median jd slope vs log2T:", out["C1_jd_accumulation"]["median_slope_log2t"])
    print("C1 max genuine margin (any cell,T):", out["C1_jd_accumulation"]["max_genuine_margin_any_cell_any_t"])
    print("C2:", json.dumps(out["C2_probe_vs_recovery"], indent=2))


if __name__ == "__main__":
    main()
