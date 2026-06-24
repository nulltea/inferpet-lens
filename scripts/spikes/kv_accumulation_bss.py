"""KV/QKV accumulation-BSS runner (Task B-1).

Two modes:
  --sanity   Block 0: synthetic unit-sanity of the three ports + the probe.
  (default)  Block 1: dev-24 CPU pilot — gram_error / jade / jd(T) / matched probe
             + Hungarian floor control over a layer×kind grid, from the cached capture.

CPU-only by design (operands are cached; attacks/probe are numpy/BLAS). Writes JSON +
a markdown summary under refine-logs/kv-accumulation/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from talens.attacks import bss
from talens.capture.cache import load_capture
from talens.capture.types import CaptureSet
from talens.measures import bss_separability as sep

OUT = Path("refine-logs/kv-accumulation")
DEV24_CAPTURE = Path("results/capture_cache/capture-3e3a86a58abf0727.pt")


# --------------------------------------------------------------------------- sanity
def _synthetic_capture(seed: int = 0) -> tuple[CaptureSet, np.ndarray]:
    """Build a synthetic CaptureSet exercising each attack's expected behaviour.

    kind 'mix'    : U = A·S, A orthogonal, S strongly non-Gaussian rows ⇒ JADE should
                    recover S up to perm/sign (high cosine). (stored as operand = U)
    kind 'gauss'  : i.i.d. Gaussian rows ⇒ JADE/JD near the chance floor.
    All under Identity (H == U) so gram_error reads ~0 (identical Gram).
    """
    rng = np.random.default_rng(seed)
    s, d = 12, 256
    operands: dict[tuple[str, int], list] = {}
    ids: list[list[int]] = []
    import torch

    for p in range(24):
        # non-Gaussian sources (Laplace), orthogonal mixing
        S = rng.laplace(size=(s, d)).astype(np.float32)
        A, _ = np.linalg.qr(rng.standard_normal((s, s)))
        U = (A @ S).astype(np.float32)
        operands.setdefault(("mix", 0), []).append(torch.from_numpy(U))
        G = rng.standard_normal((s, d)).astype(np.float32)
        operands.setdefault(("gauss", 0), []).append(torch.from_numpy(G))
        ids.append(list(range(s)))
    cap = CaptureSet(model_id="synthetic", prompt_token_ids=ids, operands=operands)
    return cap, np.zeros(0)


def _jade_recover(u: np.ndarray, m: int) -> np.ndarray:
    """Run the JADE primitive pipeline on observation ``u`` (s×d), return s_hat (m×d)."""
    y, w = bss._whiten(u, m)
    q = bss._build_cumulants(y)
    rot = bss._joint_diag(q)
    return (rot.T @ w) @ u


def run_sanity() -> dict:
    """Validate the PORTS against known answers.

    Faithfulness is checked at the primitive level on a TRUE mixing problem
    (U = A·S, compare recovered Ŝ to the true sources S) — distinct from the
    repo's plaintext setting (h==u) where Ŝ shares H's row-subspace and the
    Hungarian p95 cosine is intrinsically high (the floor). Both facts are
    asserted so the pilot's floor control is interpreted correctly.
    """
    rng = np.random.default_rng(0)
    s, d = 12, 256
    out: dict = {}

    # (1) JADE recovers true non-Gaussian sources from an orthogonal mixture.
    rec_cos, mix_vs_floor = [], []
    for _ in range(24):
        S = rng.laplace(size=(s, d))
        A, _ = np.linalg.qr(rng.standard_normal((s, s)))
        U = A @ S
        s_hat = _jade_recover(U, s)
        rec_cos.append(bss._p95_cosine_with_hungarian(s_hat, S))   # vs TRUE sources
        # floor: recover from Gaussian, compare to its own (shared-subspace) rows
        G = rng.standard_normal((s, d))
        mix_vs_floor.append(bss._p95_cosine_with_hungarian(_jade_recover(G, s), G))
    out["jade_recover_true_sources_p95"] = float(np.median(rec_cos))
    out["jade_gaussian_selfsubspace_p95"] = float(np.median(mix_vs_floor))

    # whitening correctness: cov(Y) ≈ I (independent of any reference impl)
    Stest = rng.laplace(size=(s, 2000))
    Ytest, _wt = bss._whiten(Stest, s)
    cov_y = (Ytest - Ytest.mean(axis=1, keepdims=True)) @ (Ytest - Ytest.mean(axis=1, keepdims=True)).T / 2000
    out["whiten_cov_offdiag_max"] = float(np.max(np.abs(cov_y - np.diag(np.diag(cov_y)))))
    out["whiten_cov_diag_err"] = float(np.max(np.abs(np.diag(cov_y) - 1.0)))

    # (2) gram_error ~ 0 under Identity (U == H), and JD flat on independent stacks.
    cap, _ = _synthetic_capture()
    out["gram_mix"] = bss.gram_error(cap, layer=0, kind="mix", max_features=256)
    out["jd_gauss"] = bss.jd(cap, layer=0, kind="gauss", t_values=(1, 2, 4, 8), max_dim=12)
    out["negentropy_mix"] = sep.negentropy_bits(cap, layer=0, kind="mix", max_dim=12)
    out["negentropy_gauss"] = sep.negentropy_bits(cap, layer=0, kind="gauss", max_dim=12)

    jd_vals = [v for v in out["jd_gauss"]["p95_per_t"].values() if v is not None]
    jd_flat = (max(jd_vals) - min(jd_vals)) < 0.15 if len(jd_vals) >= 2 else True

    checks = {
        "whitening produces identity covariance": out["whiten_cov_offdiag_max"] < 1e-6
        and out["whiten_cov_diag_err"] < 1e-6,
        "JADE recovery of true sources beats Gaussian self-subspace floor": out[
            "jade_recover_true_sources_p95"
        ]
        > out["jade_gaussian_selfsubspace_p95"],
        "gram_error~0 under Identity (U==H)": out["gram_mix"]["cos_norm_distance"] < 1e-4,
        "JD flat across T on independent stacks": jd_flat,
        "negentropy(non-Gaussian) > negentropy(Gaussian)": out["negentropy_mix"]["negentropy_bits"]
        > out["negentropy_gauss"]["negentropy_bits"],
    }
    out["_checks"] = {k: bool(v) for k, v in checks.items()}
    out["_all_pass"] = all(checks.values())
    return out


# --------------------------------------------------------------------------- pilot
def run_pilot(
    layers_profile: list[int],
    layers_tsweep: list[int],
    kinds: list[str],
    t_values: tuple[int, ...],
    max_dim: int,
    max_features: int,
) -> dict:
    cap, _spec = load_capture(DEV24_CAPTURE)
    res: dict = {
        "model_id": cap.model_id,
        "n_prompts": cap.n_prompts(),
        "config": {
            "layers_profile": layers_profile,
            "layers_tsweep": layers_tsweep,
            "kinds": kinds,
            "t_values": list(t_values),
            "max_dim": max_dim,
            "max_features": max_features,
        },
        "gram_error": [],
        "jade": [],
        "negentropy": [],
        "jd": [],
        "jd_floor": [],
        "shared_spectral_capacity": [],
    }
    for kind in kinds:
        avail = set(cap.layers(kind))
        for layer in layers_profile:
            if layer not in avail:
                continue
            res["gram_error"].append(bss.gram_error(cap, layer=layer, kind=kind, max_features=max_features))
            res["jade"].append(bss.jade(cap, layer=layer, kind=kind, max_dim=max_dim, max_features=max_features))
            res["negentropy"].append(
                sep.negentropy_bits(cap, layer=layer, kind=kind, max_dim=max_dim, max_features=max_features)
            )
        for layer in layers_tsweep:
            if layer not in avail:
                continue
            res["jd"].append(
                bss.jd(cap, layer=layer, kind=kind, t_values=t_values, max_dim=max_dim, max_features=max_features)
            )
            res["jd_floor"].append(
                bss.jd_floor(cap, layer=layer, kind=kind, t_values=t_values, max_dim=max_dim, max_features=max_features)
            )
            res["shared_spectral_capacity"].append(
                sep.shared_spectral_capacity_bits(
                    cap, layer=layer, kind=kind, t_values=t_values, max_dim=max_dim, max_features=max_features
                )
            )
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--layers-profile", default="0,12,20")
    ap.add_argument("--layers-tsweep", default="0,12,20")
    ap.add_argument("--kinds", default="kq,kqv_out,resid_post")
    ap.add_argument("--t-values", default="1,2,4,8,16")
    ap.add_argument("--max-dim", type=int, default=64)
    ap.add_argument("--max-features", type=int, default=256)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.sanity:
        out = run_sanity()
        (OUT / "sanity_bss.json").write_text(json.dumps(out, indent=2, default=float))
        print(json.dumps(out, indent=2, default=float))
        print("\nALL_PASS:", out["_all_pass"])
        return

    out = run_pilot(
        [int(x) for x in args.layers_profile.split(",") if x],
        [int(x) for x in args.layers_tsweep.split(",") if x],
        [k for k in args.kinds.split(",") if k],
        tuple(int(x) for x in args.t_values.split(",") if x),
        args.max_dim,
        args.max_features,
    )
    (OUT / "pilot_dev24.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
