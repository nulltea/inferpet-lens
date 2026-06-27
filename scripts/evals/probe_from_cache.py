#!/usr/bin/env python3
"""Re-run probes on a dp_leakage_sweep --cache-dir WITHOUT re-capturing (CPU, model-free).

The cache holds the released victim draw (Xte), the clean rep (X0), and shared meta (y, split,
pool, embedding table). Any array-interface probe re-runs offline here — e.g. swapping the MDL
family or the V_cap reduction — so probe-family experiments cost seconds, not a GPU capture.

Default probe = `mdl` (Voita & Titov faithful class-MDL). Pass --recovery-json to join the live
run's ridge selectivity and print the (bits ↔ recovery) Spearman per layer.

  python3 scripts/evals/probe_from_cache.py --cache-dir refine-logs/pythia-depth/cache \
      --probes mdl --recovery-json refine-logs/pythia-depth/dp_leakage_sweep.json \
      --out refine-logs/pythia-depth/mdl_class_from_cache.json
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dp_leakage_sweep import PROBES, _spearman  # reuse the registry + ranking

_FN = re.compile(r"Xte_eps(?P<eps>inf|[\d.]+)_seed(?P<seed>\d+)_L(?P<L>\d+)\.npy$")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--probes", default="mdl", help=f"subset of {sorted(PROBES)}")
    ap.add_argument("--recovery-json", default="", help="live sweep json to join ridge_sel for correlation")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cdir = Path(args.cache_dir)
    meta = np.load(cdir / "meta.npz", allow_pickle=True)
    y, table = meta["y"], meta["table"]
    emb_y, K = table[y], int(np.unique(y).size)
    probes = [p.strip() for p in args.probes.split(",") if p.strip()]

    rec_lookup = {}
    if args.recovery_json:
        for r in json.load(open(args.recovery_json))["records"]:
            rec_lookup[(r.get("epsilon"), r["layer"], r.get("seed"))] = r.get("ridge_sel")

    cells = []
    for f in sorted(cdir.glob("Xte_*.npy")):
        m = _FN.search(f.name)
        es, seed, L = m["eps"], int(m["seed"]), int(m["L"])
        eps = None if es == "inf" else float(es)
        Xte = np.load(f)
        rec = {"epsilon": eps, "eps_str": es, "layer": L, "seed": seed}
        for p in probes:
            X_clean = np.load(cdir / f"clean_L{L}.npy") if p in ("ig", "ig_unit") else None
            out = PROBES[p](Xte, emb_y, y, K, X_clean=X_clean, full_emb=table)
            for k, v in out.items():
                rec[f"{p}_{k}"] = v
        rec["ridge_sel"] = rec_lookup.get((eps, L, seed))
        cells.append(rec)
        print(f"[cache-probe] ε={es:>4} L{L:>2} " +
              " ".join(f"{p}_bits={rec.get(p + '_bits')}" for p in probes), flush=True)

    # depth × ε table for each probe's headline bits
    eps_order = sorted({c["eps_str"] for c in cells}, key=lambda s: (s != "inf", float(s) if s != "inf" else 0), reverse=True)
    layers = sorted({c["layer"] for c in cells})
    for p in probes:
        print(f"\n=== {p}_bits  (rows=layer, cols=ε) ===")
        print("  L | " + " ".join(f"{e:>10}" for e in eps_order))
        for L in layers:
            vals = []
            for e in eps_order:
                v = next((c[f"{p}_bits"] for c in cells if c["layer"] == L and c["eps_str"] == e), None)
                vals.append(f"{v:>10.1f}" if isinstance(v, (int, float)) else f"{'—':>10}")
            print(f"{L:>3} | " + " ".join(vals))
        if args.recovery_json:
            print(f"--- Spearman({p}_bits, ridge_sel) across ε, per layer ---")
            for L in layers:
                pairs = [(c[f"{p}_bits"], c["ridge_sel"]) for c in cells
                         if c["layer"] == L and c.get(f"{p}_bits") is not None
                         and c.get("ridge_sel") is not None and np.isfinite(c[f"{p}_bits"])]
                rho = _spearman(*zip(*pairs)) if len(pairs) >= 2 else None
                print(f"  L{L:>2}: ρ={rho}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"cache_dir": str(cdir), "probes": probes,
                                              "recovery_json": args.recovery_json, "records": cells}, indent=2))
        print(f"\n[cache-probe] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
