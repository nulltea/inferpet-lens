#!/usr/bin/env python3
"""Close the 'MDL/SDL untested' gap: does MDL/SDL join CLUB & capacity-PVI as a
faithful MI-probe that tracks attack recovery under noise? (model-free, CPU)

Reuses cached resid (L12, gemma-2-2b) + the recovery numbers already in
results/b2_lpos_decoder.json, recomputes the noised X at each level, and adds the
MDL surplus-description-length probe (with shuffle-control selectivity) alongside
the CLUB / capacity-PVI numbers. Reports Spearman(MDL-selectivity, ridge recovery)
to see whether MDL tracks like the other probes (which were ρ=1.0 at-layer-noise).
Capped (rows, classes) for CPU speed — MDL is ~6-7x class-PVI cost.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from talens.measures.mdl import online_code_length  # noqa: E402

CACHE = "results/capture_cache/capture-4ca8a33e16bfbec9.pt"
LAYER = 12
LEVELS = [0.0, 0.75, 1.5, 3.0]
MAX_ROWS, MAX_CLASSES = 1500, 128


def main():
    cap = torch.load(CACHE, map_location="cpu", weights_only=False)
    ids_per = [np.asarray(t, dtype=np.int64) for t in cap["prompt_token_ids"]]
    mats = cap["operands"][("resid_post", LAYER)]
    Xs, ys = [], []
    for m, tid in zip(mats, ids_per):
        m = np.asarray(m, np.float32); n = min(m.shape[0], tid.shape[0]); Xs.append(m[:n]); ys.append(tid[:n])
    X0 = np.concatenate(Xs, 0); y = np.concatenate(ys, 0); rms = float(np.sqrt((X0**2).mean()))
    rng = np.random.default_rng(20260621)
    if X0.shape[0] > MAX_ROWS:
        sel = rng.choice(X0.shape[0], MAX_ROWS, replace=False); X0, y = X0[sel], y[sel]
    rec = json.load(open("results/b2_lpos_decoder.json"))["records"]
    rec = {r["level"]: r for r in rec if r["layer"] == LAYER}

    rows = []
    for c in LEVELS:
        sigma = c * rms
        X = X0 + (sigma * np.random.default_rng(20260621 + int(1000*c) + LAYER).standard_normal(X0.shape)).astype(np.float32) if sigma > 0 else X0
        real = online_code_length(X, y, max_classes=MAX_CLASSES)["surplus_description_length_bits"]
        sh = online_code_length(X, y, max_classes=MAX_CLASSES, control="shuffle")["surplus_description_length_bits"]
        mdl_sel = real - sh
        r = rec.get(c, {})
        rows.append({"level": c, "mdl_sdl_bits": real, "mdl_shuffle": sh, "mdl_selectivity": mdl_sel,
                     "ridge_sel": r.get("ridge_selectivity"), "ca_sel": r.get("ca_selectivity"),
                     "capPVI": r.get("cap_pvi_acc"), "club": r.get("club_bits")})
        print(f"[mdl] L{LAYER} c={c:<4} MDL-SDL={real:7.1f}b sel={mdl_sel:+7.1f} | "
              f"ridge_sel={r.get('ridge_selectivity')} capPVI={r.get('cap_pvi_acc')} club={r.get('club_bits')}", flush=True)

    def sp(a, b):
        a = np.asarray([x for x in a if x is not None], float); b = np.asarray([x for x in b if x is not None], float)
        return 0.0 if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9 else float(stats.spearmanr(a, b).statistic)
    mdl = [r["mdl_selectivity"] for r in rows]
    out = {"layer": LAYER, "levels": LEVELS,
           "spearman_mdlSel_vs_ridgeSel": sp(mdl, [r["ridge_sel"] for r in rows]),
           "spearman_mdlSel_vs_capPVI": sp(mdl, [r["capPVI"] for r in rows]),
           "spearman_mdlSel_vs_club": sp(mdl, [r["club"] for r in rows]),
           "spearman_mdlSel_vs_caSel": sp(mdl, [r["ca_sel"] for r in rows]), "records": rows}
    print(f"\n[mdl] Spearman(MDL-SDL selectivity, ·): ridge_sel={out['spearman_mdlSel_vs_ridgeSel']:+.2f} "
          f"capPVI={out['spearman_mdlSel_vs_capPVI']:+.2f} CLUB={out['spearman_mdlSel_vs_club']:+.2f} "
          f"ca_sel={out['spearman_mdlSel_vs_caSel']:+.2f}")
    Path("results/mdl_probe_check.json").write_text(json.dumps(out, indent=2))
    print("[mdl] wrote results/mdl_probe_check.json")


if __name__ == "__main__":
    main()
