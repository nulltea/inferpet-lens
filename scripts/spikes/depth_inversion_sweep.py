"""Depth hidden-state inversion sweep (Task 4, resid-depth-inversion).

Tests arXiv 2507.16372 "Depth Gives a False Sense of Privacy" on Qwen3-4B
``resid_post``: capture once across a depth grid, then per layer run three
inverters (ridge / nn / mlp2) with a label-shuffle control + bootstrap CI on
selectivity, alongside two attack-INDEPENDENT probes (capacity-matched token-id
V-information reader accuracy + CLUB MI bits). Writes one JSON.

Measurement loop:
  - C1 (depth ≠ privacy): best-inverter vocab-disjoint selectivity > 0 (CI
    excludes 0) at every depth; curve flat / non-monotone, not → 0.
  - decision: does mlp2 (learned) beat ridge at deep layers (probe–attack gap)?
  - C2: does cap-reader-acc / CLUB track best recovery across depth?

Usage (always GPU-wrapped):
  scripts/run_in_rocm.sh python3 scripts/spikes/depth_inversion_sweep.py \
      --corpus corpora/release-gate-512.txt --every-n 4 \
      --out refine-logs/resid-depth-inversion/runs/full/depth_sweep.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from talens.attacks._inversion import learned_inversion, nn_inversion, ridge_inversion
from talens.capture.capture import load_or_capture
from talens.measures import club_mi_upper_bound, v_information_capacity

INVERTERS = {"ridge": ridge_inversion, "nn": nn_inversion, "mlp2": learned_inversion}


def _spearman(a: list[float], b: list[float]) -> float | None:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return None
    ar = np.argsort(np.argsort(a[ok]))
    br = np.argsort(np.argsort(b[ok]))
    if ar.std() == 0 or br.std() == 0:
        return None
    return float(np.corrcoef(ar, br)[0, 1])


def _boot_diff_ci(real_hits, shuf_hits, n=2000, seed=0):
    """95% percentile CI on (mean(real) - mean(shuffle)) over independent
    bootstrap resamples of the two test-row hit vectors."""
    if real_hits is None or shuf_hits is None or len(real_hits) == 0 or len(shuf_hits) == 0:
        return None, None
    rng = np.random.default_rng(seed)
    r = np.asarray(real_hits, float)
    s = np.asarray(shuf_hits, float)
    diffs = np.empty(n)
    for i in range(n):
        diffs[i] = r[rng.integers(0, len(r), len(r))].mean() - s[rng.integers(0, len(s), len(s))].mean()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--every-n", type=int, default=4)
    ap.add_argument("--max-layer", type=int, default=None, help="cap depth (default: all)")
    ap.add_argument("--mlp-epochs", type=int, default=150)
    ap.add_argument("--mlp-hidden", type=int, default=1024)
    ap.add_argument("--club-steps", type=int, default=150)
    ap.add_argument("--club-max-rows", type=int, default=2500)
    ap.add_argument("--pvi-dim", type=int, default=64)
    ap.add_argument("--pvi-max-rows", type=int, default=2500)
    ap.add_argument("--cache-dir", default="results/capture_cache")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[cuda] available={torch.cuda.is_available()}", flush=True)
    prompts = [ln.strip() for ln in Path(args.corpus).read_text().splitlines() if ln.strip()]

    # Resolve the depth grid up front so capture pulls only the needed layers.
    # Qwen3-4B has 36 blocks (0..35); probe with a cheap one-prompt trace? No —
    # just request a generous grid and let capture clamp to existing layers.
    n_blocks = 36
    if args.max_layer is not None:
        n_blocks = min(n_blocks, args.max_layer + 1)
    grid = list(range(0, n_blocks, args.every_n))
    print(f"[grid] layers={grid}", flush=True)

    t0 = time.time()
    cap, embed_table, source = load_or_capture(
        args.model, prompts, capture_layers=grid, kinds=("resid_post",),
        cache_dir=args.cache_dir,
    )
    print(f"[capture] source={source} layers_present={cap.layers('resid_post')} "
          f"({time.time()-t0:.1f}s)", flush=True)

    records = []
    for layer in grid:
        if ("resid_post", layer) not in cap.operands:
            print(f"[skip] layer {layer} not captured", flush=True)
            continue
        X, y, _ = cap.stack("resid_post", layer)
        rec = {"layer": layer, "n_rows": int(X.shape[0]), "d": int(X.shape[1]),
               "inverters": {}}
        tl = time.time()
        for name, fn in INVERTERS.items():
            kw = {}
            if name == "mlp2":
                kw = {"hidden": args.mlp_hidden, "epochs": args.mlp_epochs}
            real = fn(X, y, embed_table, split_mode="vocab", **kw)
            shuf = fn(X, y, embed_table, split_mode="vocab", control="shuffle", **kw)
            if real is None:
                rec["inverters"][name] = {"note": "too few rows"}
                continue
            sel = real["ttrsr_top1"] - (shuf["ttrsr_top1"] if shuf else 0.0)
            lo, hi = _boot_diff_ci(
                real.get("top1_hits"), shuf.get("top1_hits") if shuf else None
            )
            rec["inverters"][name] = {
                "ttrsr_top1": real["ttrsr_top1"],
                "ttrsr_top10": real["ttrsr_top10"],
                "embedding_cosine": real["embedding_cosine_similarity"],
                "ttrsr_top1_shuffle": shuf["ttrsr_top1"] if shuf else None,
                "selectivity": sel,
                "selectivity_ci95": [lo, hi],
                "n_train": real["n_train"], "n_test": real["n_test"],
            }
        # Probes (attack-independent) on the same operand.
        Y = embed_table[torch.from_numpy(y)].numpy()
        capv = v_information_capacity(
            X, y, family="pca_softmax", dim=args.pvi_dim, max_rows=args.pvi_max_rows,
        )
        capv_s = v_information_capacity(
            X, y, family="pca_softmax", dim=args.pvi_dim, max_rows=args.pvi_max_rows,
            control="shuffle",
        )
        club = club_mi_upper_bound(X, Y, steps=args.club_steps, max_rows=args.club_max_rows)
        rec["probes"] = {
            "cap_pvi_bits": capv.get("v_information_bits"),
            "cap_reader_acc": capv.get("reader_top1_acc"),
            "cap_reader_acc_shuffle": capv_s.get("reader_top1_acc"),
            "cap_eff_dim": capv.get("eff_dim"),
            "club_mi_bits": club.get("club_mi_bits"),
        }
        # strip non-serialisable hit arrays
        records.append(rec)
        best = max((rec["inverters"][k].get("ttrsr_top1", 0) or 0)
                   for k in rec["inverters"])
        print(f"[L{layer}] best_ttrsr={best:.3f} "
              f"ridge={rec['inverters'].get('ridge',{}).get('ttrsr_top1')} "
              f"nn={rec['inverters'].get('nn',{}).get('ttrsr_top1')} "
              f"mlp2={rec['inverters'].get('mlp2',{}).get('ttrsr_top1')} "
              f"cap_acc={rec['probes']['cap_reader_acc']} "
              f"club={rec['probes']['club_mi_bits']} ({time.time()-tl:.1f}s)", flush=True)

    # Cross-depth correlations: probe vs BEST-inverter recovery (+ per inverter).
    def col(inv, field):
        return [r["inverters"].get(inv, {}).get(field) for r in records]
    best_sel = [max((r["inverters"].get(k, {}).get("selectivity") or -9)
                    for k in r["inverters"]) for r in records]
    best_rec = [max((r["inverters"].get(k, {}).get("ttrsr_top1") or 0)
                    for k in r["inverters"]) for r in records]
    cap_acc = [r["probes"]["cap_reader_acc"] for r in records]
    club_b = [r["probes"]["club_mi_bits"] for r in records]
    corr = {
        "spearman_capacc_vs_bestrec": _spearman(cap_acc, best_rec),
        "spearman_club_vs_bestrec": _spearman(club_b, best_rec),
        "spearman_capacc_vs_bestsel": _spearman(cap_acc, best_sel),
        "spearman_club_vs_bestsel": _spearman(club_b, best_sel),
        "spearman_capacc_vs_ridge": _spearman(cap_acc, col("ridge", "ttrsr_top1")),
        "spearman_capacc_vs_mlp2": _spearman(cap_acc, col("mlp2", "ttrsr_top1")),
        # independence: cap reader vs ridge recovery should be < 0.9 to not be the attack
        "indep_capacc_vs_ridge_sel": _spearman(cap_acc, col("ridge", "selectivity")),
    }
    # per-layer mlp2 - ridge selectivity gap (decision claim)
    gap = [{"layer": r["layer"],
            "mlp2_minus_ridge_sel": (r["inverters"].get("mlp2", {}).get("selectivity"))
            and (r["inverters"]["mlp2"]["selectivity"] - r["inverters"]["ridge"]["selectivity"])
            if r["inverters"].get("mlp2", {}).get("selectivity") is not None
            and r["inverters"].get("ridge", {}).get("selectivity") is not None else None}
           for r in records]

    out = {
        "model": args.model, "corpus": args.corpus, "grid": grid,
        "config": vars(args), "records": records,
        "cross_depth_correlation": corr, "learned_vs_ridge_gap": gap,
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2, default=float))
    print(f"[done] wrote {outp} ({time.time()-t0:.1f}s total)", flush=True)
    print(f"[corr] {json.dumps(corr, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
