"""Analyze the KV-CLOAK sweep (Task B-2): channel decoupling (C1) + matched-probe
correlation / b-flatness / A-tracking (C2). Standardizes to bits + per-secret readout.

Reads refine-logs/kv-cloak/sweep.json; writes analysis.json + RESULTS.md. CPU, instant.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path("refine-logs/kv-cloak")


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return None
    from scipy.stats import spearmanr
    r, p = spearmanr(x[m], y[m])
    return {"rho": float(r), "p": float(p), "n": int(m.sum())}


def _pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return None
    from scipy.stats import pearsonr
    r, p = pearsonr(x[m], y[m])
    return {"r": float(r), "p": float(p), "n": int(m.sum())}


def _agg(recs, key):
    vals = [r[key] for r in recs if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def main() -> None:
    data = json.loads((OUT / "sweep.json").read_text())
    recs = data["records"]
    ident = [r for r in recs if r["channel"] == "identity"]

    # ---- per-layer identity baselines + floor ----
    base = {}
    for r in ident:
        base[r["layer"]] = {
            "jade_p95": r.get("jade_p95"), "jd_p95_t1": r.get("jd_p95_t1"),
            "jd_floor_t1": r.get("jd_floor_t1"), "gram_cos": r.get("gram_cos"),
            "negentropy_bits": r.get("negentropy_bits"), "spectral_cap_t1": r.get("spectral_cap_t1"),
        }

    # ---- C1: channel decoupling (averaged over b, mask, seed; per layer) ----
    c1 = {}
    for layer in sorted(base):
        c1[layer] = {}
        for ch in ["m", "sp", "scx", "naive", "a", "full"]:
            sub = [r for r in recs if r["channel"] == ch and r["layer"] == layer]
            c1[layer][ch] = {
                "gram_cos": _agg(sub, "gram_cos"),
                "gram_spec_err": _agg(sub, "gram_spec_err"),
                "jade_p95": _agg(sub, "jade_p95"),
                "negentropy_bits": _agg(sub, "negentropy_bits"),
                "spectral_cap_t1": _agg(sub, "spectral_cap_t1"),
                "n_cells": len(sub),
            }
        c1[layer]["identity"] = base[layer]

    # ---- C2a: probe vs attack correlations across ALL non-identity cells ----
    nz = [r for r in recs if r["channel"] != "identity"]
    corr = {
        "negentropy_vs_jade": {
            "spearman": _spearman([r["negentropy_bits"] for r in nz], [r["jade_p95"] for r in nz]),
            "pearson": _pearson([r["negentropy_bits"] for r in nz], [r["jade_p95"] for r in nz]),
        },
        "spectral_cap_vs_jade": {
            "spearman": _spearman([r["spectral_cap_t1"] for r in nz], [r["jade_p95"] for r in nz]),
        },
        "negentropy_vs_gram_cos": {
            "spearman": _spearman([r["negentropy_bits"] for r in nz], [r["gram_cos"] for r in nz]),
        },
        "spectral_cap_vs_gram_cos": {
            "spearman": _spearman([r["spectral_cap_t1"] for r in nz], [r["gram_cos"] for r in nz]),
        },
    }

    # ---- C2b: b-flatness — does the spectral probe vary with b under ORTHOGONAL channels? ----
    # Orthogonal channels {m, sp, scx, naive} should leave the spectral probe flat in b.
    bflat = {}
    for ch in ["m", "sp", "scx", "naive", "full"]:
        byb = defaultdict(list)
        for r in [x for x in recs if x["channel"] == ch]:
            if r.get("spectral_cap_t1") is not None:
                byb[r["b"]].append(r["spectral_cap_t1"])
        means = {int(b): float(np.mean(v)) for b, v in sorted(byb.items())}
        # also jade recovery vs b
        byb_j = defaultdict(list)
        for r in [x for x in recs if x["channel"] == ch]:
            if r.get("jade_p95") is not None:
                byb_j[r["b"]].append(r["jade_p95"])
        jmeans = {int(b): float(np.mean(v)) for b, v in sorted(byb_j.items())}
        spread = (max(means.values()) - min(means.values())) if len(means) >= 2 else 0.0
        bflat[ch] = {"spectral_cap_by_b": means, "spectral_cap_spread": spread,
                     "jade_p95_by_b": jmeans}

    # ---- C2c: A-tracking — probe & recovery vs mask energy (channel full, pooled b/seed/layer) ----
    atrack = {}
    for ch in ["a", "full"]:
        bya = defaultdict(lambda: defaultdict(list))
        for r in [x for x in recs if x["channel"] == ch]:
            for k in ("spectral_cap_t1", "negentropy_bits", "jade_p95", "gram_cos", "gram_spec_err"):
                if r.get(k) is not None:
                    bya[r["mask_energy"]][k].append(r[k])
        atrack[ch] = {
            str(a): {k: float(np.mean(v)) for k, v in d.items()}
            for a, d in sorted(bya.items())
        }

    # ---- C2d: is negentropy<->jade just channel separation? within-channel + clustered tests ----
    within = {}
    for ch in ["m", "sp", "scx", "naive", "a", "full"]:
        sub = [r for r in recs if r["channel"] == ch]
        within[ch] = {
            "spearman": _spearman([r["negentropy_bits"] for r in sub], [r["jade_p95"] for r in sub]),
            "jade_var": float(np.nanvar([r["jade_p95"] for r in sub if r.get("jade_p95") is not None])),
            "neg_var": float(np.nanvar([r["negentropy_bits"] for r in sub if r.get("negentropy_bits") is not None])),
        }
    # channel-mean (across-channel) correlation: n=6 pairs
    ch_means = []
    for ch in ["m", "sp", "scx", "naive", "a", "full"]:
        sub = [r for r in recs if r["channel"] == ch]
        ch_means.append((_agg(sub, "negentropy_bits"), _agg(sub, "jade_p95")))
    across = _spearman([x[0] for x in ch_means], [x[1] for x in ch_means])

    # ---- C2e: JD accumulation recovery per channel (T=1 vs T=4), layer 0 + the chance floor ----
    jd_table = {}
    for ch in ["identity", "m", "sp", "scx", "naive", "a", "full"]:
        sub = [r for r in recs if r["channel"] == ch and r["layer"] == 0]
        jd_table[ch] = {"jd_p95_t1": _agg(sub, "jd_p95_t1"), "jd_p95_t4": _agg(sub, "jd_p95_t4")}
    # random-orthogonal-demixing chance floor at T=1 and T=4 (channel-independent), so "at the floor
    # across T" is grounded against the floor at BOTH ends of the accumulation axis.
    import sys as _sys
    _sys.path.insert(0, "src")
    from talens.attacks import bss as _bss
    from talens.capture.cache import load_capture as _lc
    _cap, _ = _lc(Path("results/capture_cache/capture-7de5ef8d6e14afe9.pt"))
    _fl = _bss.jd_floor(_cap, layer=0, kind="k", t_values=(1, 4), max_dim=16, max_features=256)
    jd_table["_floor"] = {"jd_floor_t1": _fl["p95_per_t"].get(1), "jd_floor_t4": _fl["p95_per_t"].get(4)}

    out = {"capture": data["capture"], "n_prompts": data["n_prompts"], "config": data["config"],
           "C1_channel_decoupling": c1, "C2a_correlations": corr,
           "C2b_b_flatness": bflat, "C2c_A_tracking": atrack,
           "C2d_within_channel": within, "C2d_across_channel_mean": across,
           "C2e_jd_recovery_L0": jd_table}
    (OUT / "analysis.json").write_text(json.dumps(out, indent=2, default=float))

    # ---- compact stdout digest ----
    print("== C1 channel decoupling (layer 0, mean over b/mask/seed) ==")
    L0 = c1[0]
    print(f"  identity: gram_cos={L0['identity']['gram_cos']:.4f} jade_p95={L0['identity']['jade_p95']:.3f} "
          f"floor={L0['identity']['jd_floor_t1']} neg={L0['identity']['negentropy_bits']:.2f}b "
          f"spec={L0['identity']['spectral_cap_t1']:.2f}b")
    for ch in ["m", "sp", "scx", "naive", "a", "full"]:
        d = L0[ch]
        print(f"  {ch:6s}: gram_cos={d['gram_cos']:.4f} jade_p95={d['jade_p95']:.3f} "
              f"neg={d['negentropy_bits']:.2f}b spec={d['spectral_cap_t1']:.2f}b")
    print("\n== C2a correlations (non-identity cells) ==")
    for k, v in corr.items():
        sp = v.get("spearman")
        print(f"  {k}: spearman={sp}")
    print("\n== C2b b-flatness (spectral_cap spread across b) ==")
    for ch, d in bflat.items():
        print(f"  {ch:6s}: spec_by_b={ {k:round(x,3) for k,x in d['spectral_cap_by_b'].items()} } "
              f"spread={d['spectral_cap_spread']:.4f}  jade_by_b={ {k:round(x,3) for k,x in d['jade_p95_by_b'].items()} }")
    print("\n== C2c A-tracking (channel full, vs mask energy) ==")
    for a, d in atrack["full"].items():
        print(f"  alpha={a}: spec={d.get('spectral_cap_t1',float('nan')):.2f}b jade_p95={d.get('jade_p95',float('nan')):.3f} "
              f"gram_cos={d.get('gram_cos',float('nan')):.3f} gram_spec_err={d.get('gram_spec_err',float('nan')):.3f}")
    print("\n== C2d within-channel negentropy<->jade (is it channel separation?) ==")
    for ch, d in within.items():
        print(f"  {ch:6s}: within_spearman={d['spearman']} jade_var={d['jade_var']:.4f} neg_var={d['neg_var']:.1f}")
    print(f"  across-channel-mean spearman (n=6): {across}")
    print("\n== C2e JD accumulation recovery (layer 0) ==")
    for ch, d in jd_table.items():
        if ch == "_floor":
            print(f"  {'floor':6s}: jd_t1={d['jd_floor_t1']} jd_t4={d['jd_floor_t4']}")
        else:
            print(f"  {ch:6s}: jd_t1={d['jd_p95_t1']} jd_t4={d['jd_p95_t4']}")
    print("\nwrote", OUT / "analysis.json")


if __name__ == "__main__":
    main()
