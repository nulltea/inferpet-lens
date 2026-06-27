#!/usr/bin/env python3
"""SPIKE: does the independent-draw protocol change ridge recovery vs a single draw?

Two-draw (current): ridge trains on per_tr, scores on an INDEPENDENT per_te.
Single-draw:        ridge trains AND scores on the same draw (per_tr) — train/test rows are still
                    vocab-disjoint, but they now share within-prompt propagated noise.

If recovery_single ≈ recovery_two, the 2× capture buys nothing for ridge (per-row linear map can't
exploit the shared-context noise correlation) → single-draw is sound at the next scale, halving capture.
If recovery_single ≫ recovery_two, the independent draw is doing real work and must be kept.

Runs offline on the 160M cache (no GPU capture). Secondary: vcap PVI on per_tr vs per_te = draw variance.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "scripts/evals")
import dp_leakage_sweep as S  # ATTACKS["ridge"], PROBES["vcap"]

CDIR = Path("refine-logs/pythia-depth/cache")
SEED = 20260621
EPS = [None, 64.0, 32.0, 16.0, 8.0]
LAYERS = list(range(12))


def es(e):
    return "inf" if e is None else f"{e:g}"


def main():
    m = np.load(CDIR / "meta.npz", allow_pickle=True)
    y, tr, te, pool, table = m["y"], m["tr"], m["te"], m["pool"], m["table"]
    emb_y, pe = table[y], table[pool]
    K = int(np.unique(y).size)
    jr = {(x["epsilon"], x["layer"]): x
          for x in json.load(open("refine-logs/pythia-depth/dp_leakage_sweep.json"))["records"]}

    print(f"{'ε':>4} {'L':>2} | {'rec_single':>10} {'rec_two':>8} {'Δ(s−t)':>8} | {'json_two':>8} "
          f"| {'pvi_tr':>7} {'pvi_te':>7}")
    deltas = []
    for e in EPS:
        for L in LAYERS:
            Xtr = np.load(CDIR / f"Xtr_eps{es(e)}_seed{SEED}_L{L}.npy")
            Xte = np.load(CDIR / f"Xte_eps{es(e)}_seed{SEED}_L{L}.npy")
            # recovery: single-draw (train+test on per_tr) vs two-draw (train per_tr, test per_te)
            yhat_s = S.ATTACKS["ridge"](Xtr[tr], emb_y[tr], Xtr[te], pe, pool, ytr=y[tr], full_emb=table)
            yhat_t = S.ATTACKS["ridge"](Xtr[tr], emb_y[tr], Xte[te], pe, pool, ytr=y[tr], full_emb=table)
            rec_s = float((yhat_s == y[te]).mean())
            rec_t = float((yhat_t == y[te]).mean())
            deltas.append(rec_s - rec_t)
            # probe draw-variance: PVI on per_tr vs per_te (each splits internally)
            pvi_tr = S.PROBES["vcap"](Xtr, emb_y, y, K).get("bits")
            pvi_te = S.PROBES["vcap"](Xte, emb_y, y, K).get("bits")
            jt = jr[(e, L)]["ridge"]
            print(f"{es(e):>4} {L:>2} | {rec_s:>10.4f} {rec_t:>8.4f} {rec_s-rec_t:>+8.4f} | {jt:>8.4f} "
                  f"| {pvi_tr:>7.3f} {pvi_te:>7.3f}")
    d = np.array(deltas)
    print(f"\nΔrecovery (single − two): mean {d.mean():+.4f}  |mean| {np.abs(d).mean():.4f}  "
          f"max {d.max():+.4f}  min {d.min():+.4f}")
    print("Interpretation: |Δ|≈0 ⇒ single-draw sound for ridge (halve capture). "
          "Δ≫0 ⇒ independent draw justified.")


if __name__ == "__main__":
    main()
