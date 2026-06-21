#!/usr/bin/env python3
"""B2-L0 — the EXACT Bayes attack vs ridge under input-DP (GPU-free, L0).

At the embedding layer the input-DP cover output is, exactly,
``Y = clip(e_v) + sigma·z``, ``z~N(0,I)``, with the attacker knowing the vocab
embedding table ``{e_v}``, the noise scale ``sigma`` (the DP parameter), and
(optionally) the token prior ``pi_v``. The Bayes-optimal recoverer is then a
closed-form Gaussian-mixture posterior — so Theorem T1 (docs/research/
info-efficient-attack-guarantee.md) binds the *implemented* attack with NO
approximation slack:

    MAP token  = argmax_v [ log pi_v − ||Y − clip(e_v)||^2 / 2 sigma^2 ]
    MMSE embed = sum_v p(v|Y)·clip(e_v)

The ridge baseline is the degraded version: it fits a linear map obs→clean-emb
then cosine-matches — ignoring sigma, using cosine not the noise metric, and
risking noise amplification in the learned map. This script runs both on the
*identical* vocab-disjoint test split + candidate pool (fair, A4), sweeps the DP
budget, and measures (i) the recovery uplift (C1) and (ii) whether recovery
re-correlates with the MI probes CLUB / capacity-PVI (C2).

Pure NumPy on the real embedding table — no model forward needed at L0.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from talens.measures.club import club_mi_upper_bound  # noqa: E402
from talens.measures.vinfo_capacity import v_information_capacity  # noqa: E402

EMBED = {"gemma-2-2b": "results/capture_cache/embed-b0c6566474cadb27.pt",
         "qwen3-4b": "results/capture_cache/embed-92e31154209b70d8.pt"}


def _clip(E: np.ndarray, C: float) -> np.ndarray:
    n = np.linalg.norm(E, axis=1, keepdims=True)
    return E * np.minimum(1.0, C / np.clip(n, 1e-9, None))


def _ridge_attack(Ytr, Etr_clean, Yte, pool_emb, pool_ids, alphas=(1e-2, 1.0, 1e2)):
    """Fit obs->clean-embedding ridge; cosine-match test to the candidate pool.
    Picks alpha by train-set self-recovery (cheap proxy)."""
    d = Ytr.shape[1]
    best, best_pred = None, None
    pe = pool_emb / np.clip(np.linalg.norm(pool_emb, axis=1, keepdims=True), 1e-9, None)
    for a in alphas:
        W = np.linalg.solve(Ytr.T @ Ytr + a * np.eye(d), Ytr.T @ Etr_clean)  # (d,d)
        pred_tr = Ytr @ W
        pred_tr = pred_tr / np.clip(np.linalg.norm(pred_tr, axis=1, keepdims=True), 1e-9, None)
        # cheap val score: cosine self-match on train target
        score = float((pred_tr * (Etr_clean / np.clip(np.linalg.norm(Etr_clean, axis=1, keepdims=True), 1e-9, None))).sum(1).mean())
        if best is None or score > best[0]:
            ete = Yte @ W
            ete = ete / np.clip(np.linalg.norm(ete, axis=1, keepdims=True), 1e-9, None)
            pred = pool_ids[(ete @ pe.T).argmax(1)]
            best, best_pred = (score, a), pred
    return best_pred


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gemma-2-2b", choices=list(EMBED))
    ap.add_argument("--corpus", default="corpora/release-gate-512.txt")
    ap.add_argument("--max-prompts", type=int, default=256)
    ap.add_argument("--tokenizer", default="unsloth/gemma-2-2b")
    ap.add_argument("--epsilons", default="inf,8192,4096,2048,1024,512,256,128")
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--clip-percentile", type=float, default=99.9)
    ap.add_argument("--pool-size", type=int, default=2048)
    ap.add_argument("--club-max-rows", type=int, default=1500, help="CPU CLUB row cap (speed)")
    ap.add_argument("--cap-dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260621)
    ap.add_argument("--out", default="results/b2_l0_bayes.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    table = torch.load(EMBED[args.model], map_location="cpu", weights_only=False).float().numpy().astype(np.float32)
    vocab, d = table.shape

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        prompts = [l.strip() for l in Path(args.corpus).read_text().splitlines() if l.strip()][: args.max_prompts]
        ids = np.concatenate([np.asarray(tok(p).input_ids, dtype=np.int64) for p in prompts])
        ids = ids[(ids >= 0) & (ids < vocab)]
        print("[b2L0] token ids from real corpus tokenization", flush=True)
    except ModuleNotFoundError:
        # GPU/dep-free fallback: Zipfian token-id multiset over the real vocab
        # (realistic repeats → a non-uniform prior; embeddings are the real table).
        n_tok = 7000
        ranks = np.arange(1, vocab + 1)
        p = (1.0 / ranks); p /= p.sum()                       # Zipf(1) over a random vocab permutation
        perm = rng.permutation(vocab)
        ids = perm[rng.choice(vocab, size=n_tok, replace=True, p=p)].astype(np.int64)
        print("[b2L0] token ids from Zipf fallback (no transformers); real embeddings", flush=True)
    # token prior (smoothed corpus frequency)
    counts = np.bincount(ids, minlength=vocab).astype(np.float64)
    E = table[ids]                                   # clean embeddings (N,d)
    C = float(np.percentile(np.linalg.norm(E, axis=1), args.clip_percentile))
    Ec = _clip(E, C)
    z = math.sqrt(2 * math.log(1.25 / args.delta))
    N = ids.shape[0]
    print(f"[b2L0] {args.model} vocab={vocab} d={d} N={N} C={C:.3f} z={z:.3f}", flush=True)

    # vocab-disjoint split
    distinct = rng.permutation(np.unique(ids))
    n_tr = int(0.7 * distinct.size)
    tr_ids, te_ids = set(distinct[:n_tr].tolist()), set(distinct[n_tr:].tolist())
    tr = np.array([i for i, t in enumerate(ids) if t in tr_ids])
    te = np.array([i for i, t in enumerate(ids) if t in te_ids])
    # candidate pool: ALWAYS keep every test true id (else those rows are
    # unrecoverable and cap recovery), then top up with random fillers. Do NOT
    # sort-truncate (that dropped large-valued true ids — clean recovery fell to
    # 0.616 instead of 1.0).
    true_pool = np.unique(ids[te])
    if true_pool.size < args.pool_size:
        avail = np.setdiff1d(np.arange(vocab, dtype=np.int64), true_pool, assume_unique=False)
        fillers = rng.choice(avail, size=args.pool_size - true_pool.size, replace=False)
        pool = np.concatenate([true_pool, fillers.astype(np.int64)])
    else:
        pool = true_pool
    pool_clean = table[pool]
    pool_clip = _clip(pool_clean, C)
    log_prior_pool = np.log((counts[pool] + 1.0) / (counts.sum() + vocab))

    epsilons = [math.inf if s.strip().lower().startswith("inf") else float(s)
                for s in args.epsilons.split(",") if s.strip()]
    records = []
    for eps in epsilons:
        sigma = 0.0 if math.isinf(eps) else C * z / eps
        Y = Ec + (sigma * rng.standard_normal(Ec.shape)).astype(np.float32) if sigma > 0 else Ec.copy()
        true_te = ids[te]

        # --- ridge baseline (noise-naive, learned linear map) ---
        ridge_pred = _ridge_attack(Y[tr], Ec[tr], Y[te], pool_clip, pool)
        ridge_ttrsr = float((ridge_pred == true_te).mean())

        # --- Bayes MAP (exact, channel-aware): over the same pool ---
        # ||Y - clip(E_v)||^2 = ||Y||^2 - 2 Y·cv + ||cv||^2 ; drop ||Y||^2 (const over v)
        cv2 = (pool_clip ** 2).sum(1)                              # (P,)
        cross = Y[te] @ pool_clip.T                                # (n_te,P)
        neg_d2 = 2.0 * cross - cv2[None, :]                        # = -||Y-cv||^2 + ||Y||^2
        inv2s = 1.0 if sigma == 0 else 1.0 / (2.0 * sigma * sigma)
        score_unif = neg_d2 * inv2s                                # uniform prior
        score_freq = score_unif + log_prior_pool[None, :]          # freq prior
        bayes_unif = float((pool[score_unif.argmax(1)] == true_te).mean())
        bayes_freq = float((pool[score_freq.argmax(1)] == true_te).mean())

        # --- MI probes on (Y, token) ---
        cap = v_information_capacity(Y, ids, family="pca_softmax", dim=args.cap_dim, l2=0.1)["reader_top1_acc"]
        club = club_mi_upper_bound(Y, table[ids], max_rows=args.club_max_rows, seed=0)["club_mi_bits"]

        rec = {"epsilon": (None if math.isinf(eps) else eps), "sigma": sigma,
               "noise_to_signal": (sigma * math.sqrt(d) / C) if C else 0.0,
               "ridge_ttrsr": ridge_ttrsr, "bayes_map_unif": bayes_unif,
               "bayes_map_freq": bayes_freq, "uplift_unif": bayes_unif - ridge_ttrsr,
               "cap_pvi_acc": cap, "club_bits": club, "n_test": int(te.size)}
        records.append(rec)
        es = "inf" if math.isinf(eps) else f"{eps:g}"
        print(f"[b2L0] ε={es:>5} r={rec['noise_to_signal']:.2f} | ridge={ridge_ttrsr:.3f} "
              f"bayesMAP={bayes_unif:.3f}(+{rec['uplift_unif']:+.3f}) bayesFreq={bayes_freq:.3f} "
              f"| capPVI={cap:.3f} club={club:.0f}b", flush=True)

    def sp(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return 0.0 if np.std(a) < 1e-9 or np.std(b) < 1e-9 else float(stats.spearmanr(a, b).statistic)
    R = lambda k: [r[k] for r in records]
    corr = {
        "ridge_vs_club": sp(R("ridge_ttrsr"), R("club_bits")),
        "bayes_vs_club": sp(R("bayes_map_unif"), R("club_bits")),
        "ridge_vs_capPVI": sp(R("ridge_ttrsr"), R("cap_pvi_acc")),
        "bayes_vs_capPVI": sp(R("bayes_map_unif"), R("cap_pvi_acc")),
    }
    print(f"\n[b2L0] Spearman(recovery, MI-probe) over ε:")
    print(f"   ridge↔CLUB={corr['ridge_vs_club']:+.3f}  bayes↔CLUB={corr['bayes_vs_club']:+.3f}")
    print(f"   ridge↔capPVI={corr['ridge_vs_capPVI']:+.3f}  bayes↔capPVI={corr['bayes_vs_capPVI']:+.3f}")
    print(f"[b2L0] mean uplift (bayesMAP−ridge) = {np.mean(R('uplift_unif')):+.3f}")

    out = {"model": args.model, "layer": "L0_embedding", "attack_setting": "exact_bayes_vs_ridge",
           "C": C, "d": d, "n_test": int(te.size), "pool_size": int(pool.size),
           "spearman_vs_probe": corr, "records": records}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[b2L0] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
