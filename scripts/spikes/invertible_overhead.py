#!/usr/bin/env python3
"""Utility side for the INVERTIBLE-in-TEE defenses (Task 7, utility-tradeoff).

KV-Cloak, orthogonal GELO and AloePri keymat are lossless by construction: the trusted
enclave holds the secret key material and un-mixes exactly before the protected computation
proceeds, so the downstream task metric is the plaintext metric (utility loss = 0). For these
the leakage--utility axis is reported NOT as a degraded task metric but as

  * recon_error   — relative reconstruction error ||inv(apply(H)) - H|| / ||H|| of the
                    TEE-side un-mix (must be ~machine-eps to certify losslessness), and
  * overhead_ms   — wall-time of applying the cover (the compute cost the defense adds),
                    median over repeats, on the real operand shapes.

CPU-only and model-free (pure tensor algebra on the secret key material) — runs in the host
.venv, no GPU. KV-Cloak uses the real captured KV stack (refine-logs/kv-cloak/subcapture_L32.pt)
where present; GELO/AloePri use synthetic operands sized to their real surfaces (Qwen3-4B
resid d=2560; gemma-2-2b embedding d=2304).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))  # for `defenses.*`

from defenses.kv_cloak import KVCloak  # noqa: E402
from defenses.gelo import GELO  # noqa: E402
from defenses import aloepri  # noqa: E402


def _relerr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def _time_ms(fn, repeats: int = 5) -> float:
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


# --------------------------------------------------------------------------- KV-Cloak
def kvcloak_invert(cloak: KVCloak, U: np.ndarray, H: np.ndarray, prompt_index: int) -> np.ndarray:
    """Faithful TEE-side un-mix: undo (per-head M) then (S) then (perm) then (beacon mask),
    in reverse order of KVCloak.__call__. Uses the same key material the cloak holds."""
    seq, feat = H.shape
    h = cloak.head_dim
    n_heads = feat // h
    b = cloak.block_size
    ch = cloak.channel
    scale = float(np.median(np.linalg.norm(H, axis=1))) + 1e-8
    A_unit = cloak._mask(feat) if (ch in ("full", "a") and cloak.mask_energy > 0) else None
    R = U.copy().astype(np.float64)
    for start in range(0, seq, b):
        stop = min(start + b, seq)
        n = stop - start
        blk = R[start:stop].copy()
        block_idx = start // b
        # undo right feature-mix M (orthogonal -> transpose)
        if ch in ("full", "naive", "m"):
            for hd in range(n_heads):
                sl = slice(hd * h, (hd + 1) * h)
                blk[:, sl] = blk[:, sl] @ cloak._M_for(hd).T
        # undo left orth S
        if ch in ("full", "naive", "sp"):
            S = cloak._S_for(n)
            blk = S.T @ blk
        # undo token permutation
        if ch in ("full", "scx", "sp"):
            perm = cloak._Pi(prompt_index, block_idx, n)
            inv = np.argsort(perm)
            blk = blk[inv]
        # undo additive beacon mask
        if A_unit is not None:
            blk = blk - cloak.mask_energy * scale * A_unit[:n]
        R[start:stop] = blk
    return R


def measure_kvcloak() -> list[dict]:
    rows = []
    sub = REPO / "refine-logs/kv-cloak/subcapture_L32.pt"
    if sub.exists():
        obj = torch.load(sub, map_location="cpu", weights_only=False)
        ops = obj.get("operands", {}) if isinstance(obj, dict) else {}
        # real per-prompt KV operand (seq, n_heads*head_dim); prefer kqv_out @ L0
        arr = None
        for key in (("kqv_out", 0), ("kqv_out", 12), ("kqv_out", 20)):
            v = ops.get(key)
            if isinstance(v, (list, tuple)) and v and torch.is_tensor(v[0]) and v[0].dim() == 2:
                arr = v[0]
                break
        if arr is None:
            H = np.random.default_rng(0).standard_normal((128, 32 * 128)).astype(np.float64)
            src = "synthetic(L32 shape fallback)"
        else:
            H = arr.detach().cpu().float().numpy().astype(np.float64)
            src = "refine-logs/kv-cloak/subcapture_L32.pt operands[('kqv_out',0)][0]"
    else:
        H = np.random.default_rng(0).standard_normal((128, 32 * 128)).astype(np.float64)
        src = "synthetic(128x4096)"
    feat = H.shape[1]
    head_dim = 128 if feat % 128 == 0 else feat  # head_dim must divide feat
    if feat % head_dim != 0:
        head_dim = feat
    for channel in ("m", "full"):  # m = sole load-bearing channel; full = eq.9
        cloak = KVCloak(head_dim=head_dim, block_size=32, channel=channel, mask_energy=1.0, seed=0)
        Ht = torch.from_numpy(H.astype(np.float32))
        U = cloak(Ht, prompt_index=0).numpy().astype(np.float64)
        Hrec = kvcloak_invert(cloak, U, H, 0)
        rows.append({
            "surface": "kv", "defense": "kv-cloak", "param_name": "channel",
            "param_value": channel, "utility_metric": "recon_error",
            "utility_value": _relerr(Hrec, H),
            "overhead_ms": _time_ms(lambda: cloak(Ht, prompt_index=0)),
            "operand_shape": list(H.shape), "provenance": src,
        })
    return rows


# --------------------------------------------------------------------------- GELO
def measure_gelo() -> list[dict]:
    rows = []
    d = 2560  # Qwen3-4B resid_post width
    n = 96    # token rows per prompt (representative)
    H = np.random.default_rng(1).standard_normal((n, d)).astype(np.float64)
    Ht = torch.from_numpy(H.astype(np.float32))
    for kappa in (1.0, 10.0, 100.0):
        gelo = GELO(kappa=kappa, seed=0)
        U = gelo(Ht, prompt_index=0).numpy().astype(np.float64)
        A = gelo.mixing_for(0, n)
        # TEE un-mix: orthogonal (kappa=1) -> A^T; else exact solve
        Hrec = (A.T @ U) if kappa == 1.0 else np.linalg.solve(A, U)
        rows.append({
            "surface": "residual", "defense": "gelo", "param_name": "kappa",
            "param_value": kappa, "utility_metric": "recon_error",
            "utility_value": _relerr(Hrec, H),
            "overhead_ms": _time_ms(lambda: gelo(Ht, prompt_index=0)),
            "operand_shape": [n, d], "provenance": "synthetic(Qwen3-4B resid d=2560)",
        })
    return rows


# --------------------------------------------------------------------------- AloePri keymat
def measure_aloepri() -> list[dict]:
    rows = []
    d = 2304  # gemma-2-2b embedding width
    n = 512   # token rows
    rng = np.random.default_rng(2)
    W = (rng.standard_normal((n, d)) / np.sqrt(d)).astype(np.float32)
    h_eff = max(2, (d // 2) - (d // 2) % 2)
    seed = 1
    # Replicate obfuscate_embedding_table(alpha_e=0, keymat=True) round-trip with Q_hat.
    P_hat, Q_hat = aloepri.keymat_gen(d, h_eff, lam=0.1, seed=seed + 7)
    # P̂ Q̂ = I_d identity certificate
    ident_err = _relerr(P_hat @ Q_hat, np.eye(d))
    tau = np.random.default_rng(seed).permutation(n)
    transformed = W @ P_hat            # alpha_e=0 -> noisy == W
    obf = np.empty_like(transformed)
    obf[tau] = transformed             # obf[tau[i]] holds row i  =>  transformed = obf[tau]
    # TEE un-mix: gather rows back via Π then undo keymat via Q̂
    Wrec = obf[tau] @ Q_hat
    rows.append({
        "surface": "embedding-table", "defense": "aloepri-keymat", "param_name": "alpha_e",
        "param_value": 0.0, "utility_metric": "recon_error",
        "utility_value": _relerr(Wrec, W),
        "keymat_identity_relerr": ident_err,
        "overhead_ms": _time_ms(lambda: (W @ P_hat), repeats=5),
        "operand_shape": [n, d], "provenance": "synthetic(gemma-2-2b embed d=2304, alpha_e=0 lossless keymat)",
    })
    return rows


def main() -> None:
    out = REPO / "refine-logs/utility-tradeoff/invertible_utility.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, fn in (("kv-cloak", measure_kvcloak), ("gelo", measure_gelo), ("aloepri", measure_aloepri)):
        try:
            r = fn()
            rows.extend(r)
            for x in r:
                print(f"[inv] {x['defense']:14s} {x['param_name']}={x['param_value']:<8} "
                      f"recon_relerr={x['utility_value']:.2e} overhead={x['overhead_ms']:.3f}ms", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[inv] {name} FAILED: {e}", flush=True)
            rows.append({"defense": name, "error": repr(e)})
    out.write_text(json.dumps({"rows": rows, "note": "invertible-in-TEE defenses: utility=recon_error(~0)+overhead_ms"}, indent=2))
    print(f"[inv] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
