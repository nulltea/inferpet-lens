#!/usr/bin/env python3
"""Vec2Text-vs-DP leakage evaluation on a pooled GTR sentence embedding.

Charts the leakage-vs-privacy-budget curve for the faithful pretrained Vec2Text
attack (``scripts/evals/vec2text_attack.Vec2TextAttack``) and tests whether the
cheap matched information-theoretic probe predicts what the SOTA attack achieves.

For each ε: embed N texts → DP-noise the released embedding (clip C + Gaussian
σ=C·z/ε) → invert with Vec2Text → score recovery (BLEU / token-F1 / exact / cos)
against the source text; compute CLUB ``I(e';e0)`` and capacity-PVI on the noised
embedding. Reports Spearman(recovery, probe) over the ε sweep. ε=∞ + ``--base``
gives the base-model (0-step) baseline (the information-efficiency floor).

Run inside the ROCm container, single GPU process:

    scripts/run_in_rocm.sh python3 scripts/evals/vec2text_dp_eval.py \
        --n 128 --num-steps 20 --beam 1 --epsilons inf,1024,512,256,128 --base

See ``refine-logs/dp-stronger-attacks/vec2text-pooled/EXPERIMENT_PLAN.md`` (C1–C3).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))           # talens measures
sys.path.insert(0, str(REPO / "scripts" / "evals" / "vec2text"))
import numpy as np  # noqa: E402
from vec2text_attack import Vec2TextAttack, dp_noise, gaussian_sigma  # noqa: E402
from talens.probes.club import club_mi_upper_bound  # noqa: E402
from talens.probes.vinfo_capacity import v_information_capacity  # noqa: E402


def kmeans_labels(X, k, iters=25, seed=0):
    rng = np.random.default_rng(seed)
    c = X[rng.choice(len(X), k, replace=False)].copy()
    lab = np.zeros(len(X), dtype=np.int64)
    for _ in range(iters):
        lab = (((X[:, None, :] - c[None]) ** 2).sum(2)).argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                c[j] = X[m].mean(0)
    return lab.astype(np.int64)


def spearman(a, b):
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    return 0.0 if np.std(a) < 1e-9 or np.std(b) < 1e-9 else float(stats.spearmanr(a, b).statistic)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="corpora/release-gate-512.txt")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=32, help="Morris 32-token regime")
    ap.add_argument("--epsilons", default="inf,1024,512,256,128")
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--beam", type=int, default=1, help="sequence_beam_width (>1 is ~4x+ cost)")
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--clip-percentile", type=float, default=99.9)
    ap.add_argument("--kmeans-k", type=int, default=40)
    ap.add_argument("--base", action="store_true", help="also run the 0-step base model at ε=∞")
    ap.add_argument("--seed", type=int, default=20260622)
    ap.add_argument("--out", default="results/vec2text_dp_eval.json")
    args = ap.parse_args()

    eps_list = [math.inf if s.strip().lower().startswith("inf") else float(s)
                for s in args.epsilons.split(",") if s.strip()]
    rng = np.random.default_rng(args.seed)
    print(f"[v2t-eval] loading GTR encoder + gtr-base corrector", flush=True)
    atk = Vec2TextAttack()

    raw = [l.strip() for l in open(REPO / args.corpus) if l.strip()]
    rng.shuffle(raw)
    texts = atk.canonicalize(raw, max_tokens=args.max_tokens)[: args.n]
    n = len(texts)
    e0 = atk.embed(texts)
    C = float(np.percentile(np.linalg.norm(e0, axis=1), args.clip_percentile))
    labels = kmeans_labels(e0, min(args.kmeans_k, n // 3), seed=args.seed)
    print(f"[v2t-eval] N={n} ({args.max_tokens}-tok) ε={eps_list} steps={args.num_steps} beam={args.beam} "
          f"clip C={C:.3f}(p{args.clip_percentile})", flush=True)

    records = []
    if args.base:
        t0 = time.time(); m = atk.score(atk.invert(e0, num_steps=0), texts, clean_emb=e0)
        records.append({"epsilon": None, "attack": "base_0step", **m, "secs": round(time.time() - t0, 1)})
        print(f"[v2t-eval] base[0-step] ε=inf | bleu={m['bleu']:.1f} tF1={m['token_f1']:.3f} "
              f"exact={m['exact']:.3f} cos={m['cos']:.3f}", flush=True)

    for eps in eps_list:
        sigma = gaussian_sigma(C, eps, args.delta)
        e_noisy = dp_noise(e0, clip_C=C, sigma=sigma, rng=rng)
        t0 = time.time(); recon = atk.invert(e_noisy, num_steps=args.num_steps, beam=args.beam)
        m = atk.score(recon, texts, clean_emb=e0); secs = round(time.time() - t0, 1)
        club = club_mi_upper_bound(e_noisy, e0, max_rows=min(600, n), seed=0)["club_mi_bits"]
        try:
            pdim = min(64, max(2, n // 4))
            capv = v_information_capacity(e_noisy, labels, family="pca_softmax", dim=pdim, l2=0.1)["reader_top1_acc"]
        except Exception as e:  # noqa: BLE001
            print(f"[v2t-eval] capPVI skipped: {e}", flush=True); capv = float("nan")
        rec = {"epsilon": (None if math.isinf(eps) else eps), "sigma": sigma,
               "attack": f"vec2text_{args.num_steps}step_beam{args.beam}",
               **m, "club_bits": club, "cap_pvi_acc": capv, "secs": secs}
        records.append(rec)
        es = "inf" if math.isinf(eps) else f"{eps:g}"
        print(f"[v2t-eval] ε={es:>5} σ={sigma:.3f} | bleu={m['bleu']:.1f} tF1={m['token_f1']:.3f} "
              f"exact={m['exact']:.3f} cos={m['cos']:.3f} | CLUB={club:.1f}b capPVI={capv:.3f} [{secs}s]", flush=True)

    sweep = [r for r in records if r["attack"].startswith("vec2text")]
    corr = {met: {"club": spearman([r[met] for r in sweep], [r["club_bits"] for r in sweep]),
                  "capPVI": spearman([r[met] for r in sweep], [r["cap_pvi_acc"] for r in sweep])}
            for met in ("token_f1", "exact", "cos", "bleu")}
    print("\n[v2t-eval] C3 Spearman(recovery, probe) over ε:")
    for met in ("token_f1", "exact", "cos"):
        print(f"   {met:9s}: CLUB={corr[met]['club']:+.2f}  capPVI={corr[met]['capPVI']:+.2f}")
    out = {"n": n, "clip_C": C, "num_steps": args.num_steps, "beam": args.beam,
           "epsilons": [None if math.isinf(e) else e for e in eps_list],
           "recorrelation": corr, "records": records}
    os.makedirs(REPO / "results", exist_ok=True)
    (REPO / args.out).write_text(json.dumps(out, indent=2))
    print(f"[v2t-eval] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
