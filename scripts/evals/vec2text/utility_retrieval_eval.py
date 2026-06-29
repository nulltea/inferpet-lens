#!/usr/bin/env python3
"""Utility side of the privacy–utility tradeoff: retrieval ranking fidelity of a
DP-noised query embedding vs the clean GTR retriever.

The released object is a GTR mean-pooled sentence embedding — a *retrieval* encoder,
so utility = does the DP-noised query still retrieve the right documents. This is the
standard metric for distance-preserving / query-DP RAG schemes (CAPRISE, RemoteRAG):
compare the perturbed-query ranking against the **plaintext (clean) ranking** as ground
truth — no external labels needed.

For each ε on the same DP mechanism as the leakage side (clip C + Gaussian σ=C·z/ε,
applied to the *query* embedding; corpus stays clean), over a pool of N text embeddings
with each text as a leave-one-out query:
  * nDCG@10   (graded relevance = clean cosine)        — headline
  * Recall@{1,5,10}  (noisy top-k ∩ clean top-k)
  * Spearman rank-corr of the full similarity ordering — robust
  * top-k' expansion k'/k  (smallest k' s.t. noisy top-k' ⊇ clean top-k=10) — CAPRISE
Cheap: cosine over N embeddings, no LLM / no training / no inversion. Pairs with the
leakage curve (Vec2Text recovery, I_G) for the full tradeoff.

Run via scripts/run_in_rocm.sh (GTR encode needs the model); see vec2text-rocm recipe.
    scripts/run_in_rocm.sh python3 scripts/evals/utility_retrieval_eval.py --n 256
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "evals" / "vec2text"))
import numpy as np  # noqa: E402
from vec2text_attack import Vec2TextAttack, dp_noise, gaussian_sigma  # noqa: E402


def _normalize(M):
    return M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-9, None)


def _ndcg_at_k(ranked_rel, ideal_rel, k):
    """graded nDCG@k: ranked_rel = relevance of docs in the NOISY ranking order;
    ideal_rel = relevance sorted desc (clean ideal). Relevance = clean cosine (≥0)."""
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((ranked_rel[:k] * disc).sum())
    idcg = float((ideal_rel[:k] * disc).sum())
    return dcg / idcg if idcg > 1e-12 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="corpora/release-gate-512.txt")
    ap.add_argument("--n", type=int, default=256, help="corpus = query pool size")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--epsilons", default="inf,1024,512,256,128,64")
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--clip-percentile", type=float, default=99.9)
    ap.add_argument("--ks", default="1,5,10")
    ap.add_argument("--ndcg-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260622)
    ap.add_argument("--out", default="results/utility_retrieval_eval.json")
    args = ap.parse_args()

    eps_list = [math.inf if s.strip().lower().startswith("inf") else float(s)
                for s in args.epsilons.split(",") if s.strip()]
    ks = [int(x) for x in args.ks.split(",")]
    rng = np.random.default_rng(args.seed)
    print("[util] loading GTR encoder", flush=True)
    atk = Vec2TextAttack()
    raw = [l.strip() for l in open(REPO / args.corpus) if l.strip()]
    rng.shuffle(raw)
    texts = atk.canonicalize(raw, max_tokens=args.max_tokens)[: args.n]
    n = len(texts)
    e0 = atk.embed(texts).astype(np.float32)            # clean corpus (n,768)
    C = float(np.percentile(np.linalg.norm(e0, axis=1), args.clip_percentile))
    e0n = _normalize(e0)
    # clean similarity matrix (cosine), self masked out → leave-one-out ground truth
    S_clean = e0n @ e0n.T
    np.fill_diagonal(S_clean, -np.inf)
    clean_rank = np.argsort(-S_clean, axis=1)           # (n, n-1 valid) desc by clean cos
    print(f"[util] N={n} d={e0.shape[1]} clip C={C:.3f} | ε={eps_list} ks={ks}", flush=True)

    records = []
    for eps in eps_list:
        sigma = gaussian_sigma(C, eps, args.delta)
        # DP on the QUERY embedding (corpus clean), same mechanism as the leakage side
        q = _normalize(dp_noise(e0, clip_C=C, sigma=sigma, rng=rng))
        S_noisy = q @ e0n.T
        np.fill_diagonal(S_noisy, -np.inf)
        noisy_rank = np.argsort(-S_noisy, axis=1)

        recall = {k: [] for k in ks}
        ndcg, spear, kexp = [], [], []
        from scipy import stats
        for i in range(n):
            cr = clean_rank[i]; nr = noisy_rank[i]
            cset = {k: set(cr[:k].tolist()) for k in ks}
            for k in ks:
                recall[k].append(len(cset[k] & set(nr[:k].tolist())) / k)
            # graded nDCG@K: relevance = clean cosine (shifted to ≥0), docs in noisy order
            rel = np.clip(S_clean[i], 0.0, None)
            ndcg.append(_ndcg_at_k(rel[nr], rel[cr], args.ndcg_k))
            # full-ordering rank correlation (valid docs only)
            valid = np.isfinite(S_clean[i])
            spear.append(stats.spearmanr(S_clean[i][valid], S_noisy[i][valid]).statistic)
            # top-k' expansion (CAPRISE): smallest k' s.t. noisy top-k' ⊇ clean top-10
            target = set(cr[:10].tolist()); seen = set(); kp = 0
            for j, doc in enumerate(nr, 1):
                seen.add(int(doc))
                if target <= seen:
                    kp = j; break
            kexp.append((kp / 10.0) if kp else float(n) / 10.0)

        rec = {"epsilon": (None if math.isinf(eps) else eps), "sigma": sigma,
               "ndcg@%d" % args.ndcg_k: float(np.mean(ndcg)),
               "recall": {k: float(np.mean(recall[k])) for k in ks},
               "spearman_rank": float(np.nanmean(spear)),
               "topk_expansion": float(np.mean(kexp))}
        records.append(rec)
        es = "inf" if math.isinf(eps) else f"{eps:g}"
        rc = " ".join(f"R@{k}={rec['recall'][k]:.3f}" for k in ks)
        print(f"[util] ε={es:>5} σ={sigma:.3f} | nDCG@{args.ndcg_k}={rec['ndcg@%d'%args.ndcg_k]:.3f} "
              f"{rc} rankρ={rec['spearman_rank']:.3f} k'/k={rec['topk_expansion']:.2f}", flush=True)

    out = {"n": n, "clip_C": C, "ks": ks, "ndcg_k": args.ndcg_k,
           "epsilons": [None if math.isinf(e) else e for e in eps_list], "records": records}
    os.makedirs(REPO / "results", exist_ok=True)
    (REPO / args.out).write_text(json.dumps(out, indent=2))
    print(f"[util] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
