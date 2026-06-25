"""GELO leakage sweep (Task B-5): kappa(A) x shield-fraction on the QKVO-feeding residual.

GELO exposes U = A H (fresh secret per-prompt n x n row-mixing of the residual rows; see
scripts/defenses/gelo.py). Unlike plaintext (U == H, where BSS is ill-posed, claim
kv-bss-subspace-floor-and-negentropy-probe), GELO introduces a genuine unknown linear
row-mixing -- the canonical ICA/BSS setting. We grade ICA recovery against a *matched
random-orthogonal-demixing floor* (the kv-accumulation correction), sweep the condition
number kappa(A) and the shield fraction, and ask whether the geometry-only negentropy probe
(bits) tracks the genuine recovery margin. We also confirm the orthogonal-A feature-Gram
leak (U^T U = H^T A^T A H = H^T H at kappa=1) and that an amortized ridge inverter fails
under fresh-per-prompt A.

Reuses talens.attacks.bss internals and talens.probes.bss_separability. CPU-only on the
cached resid_post capture. No GPU.

Modes:
  --sanity   B1: feature-Gram invariance + row-Gram conjugation + un-mix identities vs kappa.
  (default)  B2/B3: sweep + matched floor + ridge anchor + probe; writes refine-logs/resid-gelo/sweep.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts/defenses")
from gelo import GELO, make_mixing  # noqa: E402

from talens.attacks.bss import (  # noqa: E402
    _build_cumulants,
    _joint_diag,
    _p95_cosine_with_hungarian,
    _subsample,
    _whiten,
)
from talens.capture.cache import load_capture  # noqa: E402
from talens.probes.bss_separability import _row_negentropy_nats  # noqa: E402

OUT = Path("refine-logs/resid-gelo")
CAP = Path("results/capture_cache/capture-28a0ee6c41330ee3.pt")
_LN2 = float(np.log(2.0))


def _orth(rng: np.random.Generator, n: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return (q * np.sign(np.diag(r)))


# --------------------------------------------------------------------------- B1 sanity
def run_sanity(layer: int = 12, max_features: int = 256) -> dict:
    cap, _ = load_capture(CAP)
    H = cap.operands[("resid_post", layer)][0].numpy().astype(np.float64)  # (seq, d)
    seq, d = H.shape
    out: dict = {"operand_shape": [seq, d], "layer": layer}
    rng = np.random.default_rng(0)
    cols = np.sort(rng.choice(d, size=min(max_features, d), replace=False))
    Hc = H[:, cols]
    gH = Hc.T @ Hc  # feature (column) Gram, (f, f)
    rowH = H @ H.T  # row Gram, (seq, seq)

    def metrics(kappa: float) -> dict:
        A = make_mixing(np.random.default_rng((0, 55, 0, seq)), seq, kappa)
        U = A @ H
        Uc = U[:, cols]
        gU = Uc.T @ Uc
        rowU = U @ U.T
        feat_gram_relerr = float(np.linalg.norm(gU - gH) / (np.linalg.norm(gH) + 1e-12))
        row_gram_relerr = float(np.linalg.norm(rowU - rowH) / (np.linalg.norm(rowH) + 1e-12))
        row_fro_preserved = float(abs(np.linalg.norm(rowU) - np.linalg.norm(rowH))
                                  / (np.linalg.norm(rowH) + 1e-12))
        # defender un-mix
        Hrec = np.linalg.solve(A, U)
        unmix_err = float(np.linalg.norm(Hrec - H) / (np.linalg.norm(H) + 1e-12))
        cond = float(np.linalg.cond(A))
        return {"kappa": kappa, "cond_A": cond, "feat_gram_relerr": feat_gram_relerr,
                "row_gram_relerr": row_gram_relerr, "row_gram_fro_relerr": row_fro_preserved,
                "unmix_relerr": unmix_err}

    rows = [metrics(k) for k in (1.0, 3.0, 10.0, 30.0, 100.0)]
    out["sweep"] = rows
    m1 = rows[0]
    checks = {
        "kappa=1 A orthogonal (cond~1)": m1["cond_A"] < 1.0 + 1e-6,
        "kappa=1 feature-Gram invariant (relerr~0)": m1["feat_gram_relerr"] < 1e-8,
        "kappa=1 row-Gram Frobenius-norm preserved (orthogonal conjugation)":
            m1["row_gram_fro_relerr"] < 1e-8,
        "kappa=1 row-Gram entries DO change (conjugation != invariance)":
            m1["row_gram_relerr"] > 1e-3,
        "feature-Gram leak grows with kappa": rows[-1]["feat_gram_relerr"] > 1e-2
            and rows[-1]["feat_gram_relerr"] > 1e4 * (m1["feat_gram_relerr"] + 1e-15),
        "defender un-mix exact at all kappa": max(r["unmix_relerr"] for r in rows) < 1e-6,
    }
    out["_checks"] = {k: bool(v) for k, v in checks.items()}
    out["_all_pass"] = bool(all(checks.values()))
    return out


# --------------------------------------------------------------------------- recovery primitives
def _prep(H: np.ndarray, max_dim: int, max_features: int, rng: np.random.Generator):
    """Subsample real rows/cols; return (H_sub, col_idx)."""
    H2, _ = _subsample(H.copy(), H.copy(), max_dim, max_features, rng)
    return H2


def _shielded_observation(H_real: np.ndarray, shield_frac: float, kappa: float,
                          prompt_index: int, rng: np.random.Generator):
    """Build U = A_aug @ [H_real; shields], return (U, n_real). Shields are Gaussian rows
    energy-matched to the median real-row L2 norm. shield_frac=0 -> no shields."""
    n_real, d = H_real.shape
    n_shield = int(round(shield_frac * n_real))
    if n_shield > 0:
        scale = float(np.median(np.linalg.norm(H_real, axis=1))) + 1e-12
        sh = rng.standard_normal((n_shield, d))
        sh = sh / (np.linalg.norm(sh, axis=1, keepdims=True) + 1e-12) * scale
        H_aug = np.vstack([H_real, sh])
    else:
        H_aug = H_real
    n = H_aug.shape[0]
    A = make_mixing(np.random.default_rng((0, 55, prompt_index, n)), n, kappa)
    return A @ H_aug, n_real


def _jade_p95(U: np.ndarray, real_h: np.ndarray) -> float:
    s = U.shape[0]
    if s < 4 or U.shape[1] < 2 * s:
        return float("nan")
    try:
        y, w = _whiten(U, s)
        rot = _joint_diag(_build_cumulants(y))
    except np.linalg.LinAlgError:
        return float("nan")
    return _p95_cosine_with_hungarian((rot.T @ w) @ U, real_h)


def _rand_demix_p95(U: np.ndarray, real_h: np.ndarray, rng: np.random.Generator) -> float:
    """Matched floor: same whitening, a RANDOM rotation instead of joint-diag."""
    s = U.shape[0]
    if s < 4 or U.shape[1] < 2 * s:
        return float("nan")
    try:
        y, w = _whiten(U, s)
    except np.linalg.LinAlgError:
        return float("nan")
    rot = _orth(rng, s)
    return _p95_cosine_with_hungarian((rot.T @ w) @ U, real_h)


def _negentropy_bits(U: np.ndarray) -> float:
    s = U.shape[0]
    if s < 4 or U.shape[1] < 2 * s:
        return float("nan")
    try:
        y, _w = _whiten(U, s)
    except np.linalg.LinAlgError:
        return float("nan")
    return _row_negentropy_nats(y) / _LN2


def _feat_gram_relerr(U: np.ndarray, H: np.ndarray) -> float:
    gU, gH = U.T @ U, H.T @ H
    return float(np.linalg.norm(gU - gH) / (np.linalg.norm(gH) + 1e-12))


# --------------------------------------------------------------------------- B2 sweep
def _cell(prompts: list[np.ndarray], kappa: float, shield_frac: float,
          max_dim: int, max_features: int, seed: int) -> dict:
    """One sweep cell: medians over prompts of recovery, matched floor, margin, probe, leak."""
    rng = np.random.default_rng(seed)
    jade, floor, neg, leak = [], [], [], []
    for pi, H in enumerate(prompts):
        Hs = _prep(H, max_dim, max_features, np.random.default_rng((seed, pi)))
        if Hs.shape[0] < 4:
            continue
        U, n_real = _shielded_observation(Hs, shield_frac, kappa, pi, rng)
        real_h = Hs[:n_real]
        jp = _jade_p95(U, real_h)
        fp = _rand_demix_p95(U, real_h, np.random.default_rng((seed, 99, pi)))
        nb = _negentropy_bits(U)
        if jp == jp:
            jade.append(jp)
        if fp == fp:
            floor.append(fp)
        if nb == nb:
            neg.append(nb)
        if shield_frac == 0.0:
            leak.append(_feat_gram_relerr(U, Hs))
    med = lambda x: float(np.median(x)) if x else None  # noqa: E731
    jade_m, floor_m = med(jade), med(floor)
    margin = (jade_m - floor_m) if (jade_m is not None and floor_m is not None) else None
    return {"kappa": kappa, "shield_frac": shield_frac,
            "jade_p95": jade_m, "rand_demix_floor_p95": floor_m, "genuine_margin": margin,
            "negentropy_bits": med(neg), "feat_gram_relerr": med(leak), "n": len(jade)}


def _ridge_anchor(prompts: list[np.ndarray], kappa: float, max_dim: int, max_features: int,
                  seed: int) -> dict:
    """Amortized linear inverter: fit W: U->H on stacked train rows, test on held-out prompts.
    Under fresh-per-prompt A this should fail (recovery ~ floor)."""
    rng = np.random.default_rng(seed)
    obs = []
    for pi, H in enumerate(prompts):
        Hs = _prep(H, max_dim, max_features, np.random.default_rng((seed, pi)))
        if Hs.shape[0] < 4:
            continue
        U, n_real = _shielded_observation(Hs, 0.0, kappa, pi, rng)
        obs.append((U, Hs))
    if len(obs) < 6:
        return {"kappa": kappa, "ridge_p95": None, "n": 0}
    n_tr = max(3, int(0.6 * len(obs)))
    Utr = np.vstack([u for u, _ in obs[:n_tr]])
    Htr = np.vstack([h for _, h in obs[:n_tr]])
    lam = 1e-2 * np.trace(Utr.T @ Utr) / Utr.shape[1]
    W = np.linalg.solve(Utr.T @ Utr + lam * np.eye(Utr.shape[1]), Utr.T @ Htr)  # (d,d)
    p95 = []
    for u, h in obs[n_tr:]:
        p95.append(_p95_cosine_with_hungarian(u @ W, h))
    return {"kappa": kappa, "ridge_p95": float(np.median(p95)) if p95 else None, "n": len(p95)}


def run_sweep(layers, kappas, shield_fracs, n_prompts, max_dim, max_features, seed) -> dict:
    cap, _ = load_capture(CAP)
    records, ridge_records = [], []
    for layer in layers:
        if layer not in set(cap.layers("resid_post")):
            continue
        prompts = [np.asarray(m[1], dtype=np.float64)
                   for m in cap.per_prompt_matrices("resid_post", layer)][:n_prompts]
        # identity baseline (kappa=1 with no mixing == plaintext): record jade vs floor at U=H
        for kappa in kappas:
            for sf in shield_fracs:
                rec = _cell(prompts, kappa, sf, max_dim, max_features, seed)
                rec["layer"] = layer
                records.append(rec)
            ridge_records.append({**_ridge_anchor(prompts, kappa, max_dim, max_features, seed),
                                  "layer": layer})
    return {"capture": CAP.name, "n_prompts_used": n_prompts,
            "config": {"layers": layers, "kappas": kappas, "shield_fracs": shield_fracs,
                       "max_dim": max_dim, "max_features": max_features, "seed": seed},
            "records": records, "ridge": ridge_records}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--layers", default="0,12,20")
    ap.add_argument("--kappas", default="1,3,10,30,100")
    ap.add_argument("--shield-fracs", default="0,0.5,1.0")
    ap.add_argument("--n-prompts", type=int, default=96)
    ap.add_argument("--max-dim", type=int, default=48)
    ap.add_argument("--max-features", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.sanity:
        out = run_sanity()
        (OUT / "sanity.json").write_text(json.dumps(out, indent=2, default=float))
        print(json.dumps(out, indent=2, default=float))
        print("\nALL_PASS:", out["_all_pass"])
        return

    out = run_sweep(
        [int(x) for x in args.layers.split(",") if x],
        [float(x) for x in args.kappas.split(",") if x],
        [float(x) for x in args.shield_fracs.split(",") if x],
        args.n_prompts, args.max_dim, args.max_features, args.seed,
    )
    (OUT / "sweep.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {OUT/'sweep.json'}: {len(out['records'])} records, {len(out['ridge'])} ridge")


if __name__ == "__main__":
    main()
