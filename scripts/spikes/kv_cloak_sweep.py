"""KV-CLOAK leakage sweep (Task B-2): channel x block-size x mask-energy.

Reuses the Task-1 BSS attacks (talens.attacks.bss) and matched geometry-only probes
(talens.probes.bss_separability), all of which accept transform=, with the KV-CLOAK
Transform (scripts/defenses/kv_cloak.py) over the raw KV-cache surface (kind 'k').

Modes:
  --sanity   B1: verify the channel-decoupling identities on a sample operand
             (M-only Gram-invariant; S P̂ spectrum-preserving; A spectrum-changing).
  (default)  B2+B3: sweep + matched probe; writes refine-logs/kv-cloak/sweep.json.

CPU-only on the cached K/V capture. No GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts/defenses")
from kv_cloak import KVCloak  # noqa: E402

from talens.attacks import bss  # noqa: E402
from talens.capture.cache import load_capture  # noqa: E402
from talens.probes import bss_separability as sep  # noqa: E402

OUT = Path("refine-logs/kv-cloak")
CAP = Path("results/capture_cache/capture-7de5ef8d6e14afe9.pt")
HEAD_DIM = 128


def _gram_spectrum(x: np.ndarray) -> np.ndarray:
    g = x @ x.T
    return np.sort(np.linalg.eigvalsh(0.5 * (g + g.T)))[::-1]


def run_sanity() -> dict:
    cap, _ = load_capture(CAP)
    H = cap.operands[("k", 0)][0].numpy().astype(np.float64)  # (seq, 1024)
    seq, feat = H.shape
    out: dict = {"operand_shape": [seq, feat], "head_dim": HEAD_DIM, "n_heads": feat // HEAD_DIM}

    def apply(channel, b=32, alpha=1.0):
        t = KVCloak(head_dim=HEAD_DIM, block_size=b, channel=channel, mask_energy=alpha, seed=0)
        import torch
        return t(torch.from_numpy(H.astype(np.float32)), prompt_index=0).numpy().astype(np.float64)

    eig_H = _gram_spectrum(H)
    def spec_rel_err(U):
        e = _gram_spectrum(U)
        return float(np.linalg.norm(e - eig_H) / (np.linalg.norm(eig_H) + 1e-12))
    def gram_cos(U):  # full normalized-Gram Frobenius distance
        gu, gh = U @ U.T, H @ H.T
        return float(np.linalg.norm(gu / np.linalg.norm(gu) - gh / np.linalg.norm(gh)))

    U_m = apply("m"); U_sp = apply("sp"); U_a = apply("a", alpha=1.0); U_scx = apply("scx")
    out["m_only_gram_cos"] = gram_cos(U_m)             # expect ~0 (M Gram-invariant)
    out["m_only_spec_rel_err"] = spec_rel_err(U_m)     # expect ~0
    out["sp_only_spec_rel_err"] = spec_rel_err(U_sp)   # expect ~0 (similarity preserves spectrum)
    out["sp_only_gram_cos"] = gram_cos(U_sp)           # expect > 0 (full Gram rotated)
    out["scx_only_spec_rel_err"] = spec_rel_err(U_scx) # expect ~0 (permutation = similarity)
    out["a_only_spec_rel_err"] = spec_rel_err(U_a)     # expect > 0 (A changes spectrum)
    out["a_only_gram_cos"] = gram_cos(U_a)             # expect > 0

    # orthogonality of the key material
    t = KVCloak(head_dim=HEAD_DIM, block_size=32, channel="full", seed=0)
    S = t._S_for(32); M = t._M_for(0)
    out["S_orth_err"] = float(np.max(np.abs(S @ S.T - np.eye(32))))
    out["M_orth_err"] = float(np.max(np.abs(M @ M.T - np.eye(HEAD_DIM))))

    checks = {
        "M-only leaves row-Gram invariant (gram_cos~0)": out["m_only_gram_cos"] < 1e-4,
        "M-only spectrum invariant": out["m_only_spec_rel_err"] < 1e-4,
        "S.Phat preserve Gram spectrum": out["sp_only_spec_rel_err"] < 1e-4,
        "S.Phat rotate full Gram (gram_cos>0)": out["sp_only_gram_cos"] > 1e-3,
        "SCX (perm) preserves spectrum": out["scx_only_spec_rel_err"] < 1e-4,
        "A changes the Gram spectrum (>> orthogonal channels)": (
            out["a_only_spec_rel_err"] > 1e-2
            and out["a_only_spec_rel_err"] > 1e3 * max(out["m_only_spec_rel_err"],
                                                       out["sp_only_spec_rel_err"],
                                                       out["scx_only_spec_rel_err"])
        ),
        "S orthogonal": out["S_orth_err"] < 1e-8,
        "M orthogonal": out["M_orth_err"] < 1e-8,
    }
    out["_checks"] = {k: bool(v) for k, v in checks.items()}
    out["_all_pass"] = bool(all(checks.values()))
    return out


def run_sweep(
    kinds: list[str], layers: list[int], channels: list[str],
    b_values: list[int], mask_energies: list[float], seeds: list[int],
    max_dim: int, max_features: int,
) -> dict:
    cap, _ = load_capture(CAP)
    records: list[dict] = []

    def measure(kind, layer, transform, label):
        ge = bss.gram_error(cap, layer=layer, kind=kind, transform=transform, max_features=max_features)
        jd_ = bss.jade(cap, layer=layer, kind=kind, transform=transform, max_dim=max_dim, max_features=max_features)
        jdd = bss.jd(cap, layer=layer, kind=kind, transform=transform, t_values=(1, 2, 4),
                     max_dim=max_dim, max_features=max_features)
        neg = sep.negentropy_bits(cap, layer=layer, kind=kind, transform=transform,
                                  max_dim=max_dim, max_features=max_features)
        ssc = sep.shared_spectral_capacity_bits(cap, layer=layer, kind=kind, transform=transform,
                                                t_values=(1, 2, 4), max_dim=max_dim, max_features=max_features)
        return {
            "label": label, "kind": kind, "layer": layer,
            "gram_cos": ge.get("cos_norm_distance"), "gram_spec_err": ge.get("row_gram_spectrum_error"),
            "jade_p95": jd_.get("jade_p95_cosine"),
            "jd_p95_t1": jdd["p95_per_t"].get(1), "jd_p95_t4": jdd["p95_per_t"].get(4),
            "negentropy_bits": neg.get("negentropy_bits"),
            "spectral_cap_t1": ssc["cap_per_t"].get(1),
        }

    for kind in kinds:
        avail = set(cap.layers(kind))
        for layer in layers:
            if layer not in avail:
                continue
            # plaintext baseline + floor
            rec = measure(kind, layer, None, "identity")
            rec.update({"channel": "identity", "b": 0, "mask_energy": 0.0, "seed": -1})
            floor = bss.jd_floor(cap, layer=layer, kind=kind, t_values=(1, 2, 4),
                                 max_dim=max_dim, max_features=max_features)
            rec["jd_floor_t1"] = floor["p95_per_t"].get(1)
            records.append(rec)

            for channel in channels:
                mes = mask_energies if channel in ("a", "full") else [0.0]
                for b in b_values:
                    for alpha in mes:
                        for seed in seeds:
                            t = KVCloak(head_dim=HEAD_DIM, block_size=b, channel=channel,
                                        mask_energy=alpha, seed=seed)
                            r = measure(kind, layer, t,
                                        f"{channel}|b{b}|a{alpha}|s{seed}")
                            r.update({"channel": channel, "b": b, "mask_energy": alpha, "seed": seed})
                            records.append(r)
    return {"capture": CAP.name, "n_prompts": cap.n_prompts(),
            "config": {"kinds": kinds, "layers": layers, "channels": channels,
                       "b_values": b_values, "mask_energies": mask_energies, "seeds": seeds,
                       "max_dim": max_dim, "max_features": max_features},
            "records": records}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--kinds", default="k")
    ap.add_argument("--layers", default="0,12,20")
    ap.add_argument("--channels", default="m,sp,a,scx,naive,full")
    ap.add_argument("--b-values", default="16,32,64")
    ap.add_argument("--mask-energies", default="0,0.5,1,2,4")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--max-dim", type=int, default=64)
    ap.add_argument("--max-features", type=int, default=256)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.sanity:
        out = run_sanity()
        (OUT / "sanity.json").write_text(json.dumps(out, indent=2, default=float))
        print(json.dumps(out, indent=2, default=float))
        print("\nALL_PASS:", out["_all_pass"])
        return

    out = run_sweep(
        [k for k in args.kinds.split(",") if k],
        [int(x) for x in args.layers.split(",") if x],
        [c for c in args.channels.split(",") if c],
        [int(x) for x in args.b_values.split(",") if x],
        [float(x) for x in args.mask_energies.split(",") if x],
        [int(x) for x in args.seeds.split(",") if x],
        args.max_dim, args.max_features,
    )
    (OUT / "sweep.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {OUT/'sweep.json'}: {len(out['records'])} records")


if __name__ == "__main__":
    main()
