#!/usr/bin/env python3
"""Information-efficient PERMUTATION recovery: is RowSort-64 a weak attack? (GPU-free)

VMA (the AloePri baseline) matches the column-perm-invariant **sorted-quantile**
signature φ (64 bins) by cosine + Hungarian. φ shows the same decorrelation
signature as ridge: small α_e noise collapses τ-recovery while CLUB-on-φ barely
moves → the info is preserved, RowSort-64 can't use it. The 64-quantile binning is
a LOSSY compression of the sorted row; the sufficient statistic under column
permutation is the FULL sorted row (Dai-Cullina-Kiyavash MAP / wiki claim
perm-llr-threshold). Stronger matchers tested vs the α_e sweep:

  * rowsort64_cos   — VMA baseline: 64-quantile sorted signature, cosine + Hungarian
  * fullsort_cos    — full sorted row (all d values), cosine + Hungarian
  * fullsort_euc    — full sorted row, NEGATIVE Euclidean (the iid-Gaussian MLE
                      metric on the sorted vector) + Hungarian  [noise-aware]

Reports τ-recovery uplift + low-noise re-correlation with CLUB-on-φ. Pure NumPy/SciPy.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from scipy import stats
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from defenses.aloepri import obfuscate_embedding_table  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from talens.weights.measures import club_mi_weights  # noqa: E402

EMBED = "results/capture_cache/embed-b0c6566474cadb27.pt"


def _l2(F):
    return F / np.clip(np.linalg.norm(F, axis=1, keepdims=True), 1e-9, None)


def sig_quantile(W, bins=64):
    qs = np.linspace(0.0, 1.0, bins)
    f = np.quantile(W, qs, axis=1).T.astype(np.float64)
    f -= f.mean(axis=1, keepdims=True)
    return _l2(f).astype(np.float32)


def sig_fullsort(W):
    f = np.sort(W, axis=1).astype(np.float64)          # column-perm invariant, full
    f -= f.mean(axis=1, keepdims=True)
    return _l2(f).astype(np.float32)


def recover(A, B, perm, metric):
    """A=plain sigs (index=plain i), B=obf sigs (index=obf j). perm: obf[perm[i]]
    is partner of plain[i]. Hungarian on similarity; report per-row τ-recovery."""
    if metric == "cos":
        sims = A @ B.T                                  # (Nplain, Nobf)
    else:  # negative euclidean (iid-Gaussian MLE on the sorted vector)
        a2 = (A ** 2).sum(1)[:, None]; b2 = (B ** 2).sum(1)[None, :]
        sims = -(a2 - 2.0 * A @ B.T + b2)
    pi, oi = linear_sum_assignment(-sims)               # max total similarity
    pred_plain = np.empty(B.shape[0], dtype=np.int64); pred_plain[oi] = pi
    true_plain = np.argsort(perm)                        # true plain index of obf row j
    return float((pred_plain == true_plain).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-tokens", type=int, default=1000)
    ap.add_argument("--alphas", default="0,0.1,0.2,0.35,0.5,0.75,1.0")
    ap.add_argument("--bins", type=int, default=64)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="results/vma_stronger.json")
    args = ap.parse_args()
    table = torch.load(EMBED, map_location="cpu", weights_only=False).float().numpy().astype(np.float32)
    vocab, d = table.shape
    alphas = [float(s) for s in args.alphas.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    print(f"[vma+] embed {vocab}x{d} N={args.n_tokens} alphas={alphas} seeds={seeds}", flush=True)

    per_seed = {}
    for sd in seeds:
        rng = np.random.default_rng(20260621 + sd)
        W = table[rng.choice(vocab, size=args.n_tokens, replace=False)].copy()
        rows = []
        for a in alphas:
            pair = obfuscate_embedding_table(W, alpha_e=a, keymat=False, seed=20260621 + sd)
            plain, obf, perm = pair.plain, pair.obf, pair.perm
            # signatures (column-perm invariant)
            Aq, Bq = sig_quantile(plain, args.bins), sig_quantile(obf, args.bins)
            Af, Bf = sig_fullsort(plain), sig_fullsort(obf)
            r_q = recover(Aq, Bq, perm, "cos")
            r_fc = recover(Af, Bf, perm, "cos")
            r_fe = recover(Af, Bf, perm, "euc")
            club = club_mi_weights(pair, bins=args.bins, steps=150, hidden_size=128, seed=0)["club_mi_bits"]
            rows.append({"alpha_e": a, "rowsort64_cos": r_q, "fullsort_cos": r_fc,
                         "fullsort_euc": r_fe, "club_on_phi": club})
        per_seed[sd] = rows
        print(f"[vma+] seed{sd}: " + " ".join(
            f"a{r['alpha_e']}:rs{r['rowsort64_cos']:.2f}/fe{r['fullsort_euc']:.2f}" for r in rows), flush=True)

    # average over seeds
    def avg(k, i): return float(np.mean([per_seed[s][i][k] for s in seeds]))
    agg = []
    for i, a in enumerate(alphas):
        agg.append({"alpha_e": a, "rowsort64_cos": avg("rowsort64_cos", i),
                    "fullsort_cos": avg("fullsort_cos", i), "fullsort_euc": avg("fullsort_euc", i),
                    "club_on_phi": avg("club_on_phi", i),
                    "uplift_fe_vs_rs": avg("fullsort_euc", i) - avg("rowsort64_cos", i)})
    print("\n[vma+] α_e | rowsort64 | fullsort_cos | fullsort_euc | uplift(fe−rs) | CLUB-φ")
    for r in agg:
        print(f"   {r['alpha_e']:<5} {r['rowsort64_cos']:.3f}      {r['fullsort_cos']:.3f}        "
              f"{r['fullsort_euc']:.3f}        {r['uplift_fe_vs_rs']:+.3f}        {r['club_on_phi']:.0f}")

    def sp(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return 0.0 if np.std(x) < 1e-9 or np.std(y) < 1e-9 else float(stats.spearmanr(x, y).statistic)
    club = [r["club_on_phi"] for r in agg]
    # low-noise subset (α_e ≤ 0.5) — where the weak attack decorrelated
    lo = [i for i, r in enumerate(agg) if r["alpha_e"] <= 0.5]
    corr = {
        "rs_vs_club_all": sp([r["rowsort64_cos"] for r in agg], club),
        "fe_vs_club_all": sp([r["fullsort_euc"] for r in agg], club),
        "rs_vs_club_lownoise": sp([agg[i]["rowsort64_cos"] for i in lo], [club[i] for i in lo]),
        "fe_vs_club_lownoise": sp([agg[i]["fullsort_euc"] for i in lo], [club[i] for i in lo]),
    }
    print(f"\n[vma+] Spearman(τ-recovery, CLUB-on-φ): all-sweep rs={corr['rs_vs_club_all']:+.2f} "
          f"fe={corr['fe_vs_club_all']:+.2f} | low-noise(α≤0.5) rs={corr['rs_vs_club_lownoise']:+.2f} "
          f"fe={corr['fe_vs_club_lownoise']:+.2f}")
    Path(args.out).write_text(json.dumps({"alphas": alphas, "seeds": seeds, "aggregate": agg,
                              "recorrelation": corr, "per_seed": per_seed}, indent=2))
    print(f"[vma+] wrote {args.out}")


if __name__ == "__main__":
    main()
