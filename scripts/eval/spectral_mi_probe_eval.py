#!/usr/bin/env python3
"""Validate the spectral channel-MI probe vs CLUB / capPVI on the GTR DP sweep.

For each ε on the pooled-GTR Vec2Text pipeline (reusing ``vec2text_attack``):
recovery (token-F1 / exact / cos / positional-token-acc) + three probes —
**I_G** (geometry-only `spectral_channel_mi`, on the *clipped* clean embeddings),
CLUB `I(e';e0)`, capPVI (cluster reader) — with per-probe wall-clock cost. Reports
C1 Spearman(probe, recovery) over ε and the C2 Fano/RD ceilings (with explicit,
flagged H_X/H_e0 proxies). Geometry-only: I_G never sees the attack.

Run inside the ROCm container (see vec2text-rocm dependency recipe):
    scripts/run_in_rocm.sh python3 scripts/eval/spectral_mi_probe_eval.py --n 96
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "eval"))
import numpy as np  # noqa: E402
from vec2text_attack import Vec2TextAttack, dp_noise, gaussian_sigma, normalize_text  # noqa: E402
from talens.measures.club import club_mi_upper_bound  # noqa: E402
from talens.measures.vinfo_capacity import v_information_capacity  # noqa: E402
from talens.measures.spectral_channel_mi import spectral_channel_mi  # noqa: E402


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


def positional_token_acc(atk, recon, true):
    """Mean fraction of positions where the GTR-tokenized recon matches the true
    token (over min length) — the per-token accuracy the RD floor (T3b) ceilings."""
    accs = []
    for r, t in zip(recon, true):
        rt = atk.tokenizer(r)["input_ids"]
        tt = atk.tokenizer(t)["input_ids"]
        m = min(len(rt), len(tt))
        accs.append(np.mean([rt[i] == tt[i] for i in range(m)]) if m else 0.0)
    return float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="corpora/release-gate-512.txt")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--epsilons", default="inf,1024,512,256,128")
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--clip-percentile", type=float, default=99.9)
    ap.add_argument("--kmeans-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260622)
    ap.add_argument("--out", default="results/spectral_mi_probe_eval.json")
    args = ap.parse_args()

    eps_list = [math.inf if s.strip().lower().startswith("inf") else float(s)
                for s in args.epsilons.split(",") if s.strip()]
    rng = np.random.default_rng(args.seed)
    print("[smi] loading GTR + gtr-base corrector", flush=True)
    atk = Vec2TextAttack()
    vocab = atk.tokenizer.vocab_size

    raw = [l.strip() for l in open(REPO / args.corpus) if l.strip()]
    rng.shuffle(raw)
    texts = atk.canonicalize(raw, max_tokens=args.max_tokens)[: args.n]
    n = len(texts)
    e0 = atk.embed(texts)                              # clean pooled (n,768)
    C = float(np.percentile(np.linalg.norm(e0, axis=1), args.clip_percentile))
    # clipped clean embeddings = the channel's e0 (matches DP clip-then-noise); Σ from these
    scale = np.minimum(1.0, C / (np.linalg.norm(e0, axis=1, keepdims=True) + 1e-9))
    e0_clip = (e0 * scale).astype(np.float32)
    labels = kmeans_labels(e0, min(args.kmeans_k, n // 3), seed=args.seed)
    # honest H proxies (flagged): uniform message entropy over the true 32-tok space,
    # H_X = n_tokens·log2(vocab); H_e0 ≤ H_X — use H_e0=H_X as an upper proxy.
    H_X = args.max_tokens * math.log2(vocab)
    H_e0 = H_X
    print(f"[smi] N={n} d={e0.shape[1]} clip C={C:.3f} | H_X≈{H_X:.0f}b (={args.max_tokens}·log2 {vocab}); "
          f"H_e0 proxy=H_X (flagged)", flush=True)

    records = []
    for eps in eps_list:
        sigma = gaussian_sigma(C, eps, args.delta)
        e_noisy = dp_noise(e0, clip_C=C, sigma=sigma, rng=rng)
        t0 = time.time(); recon = atk.invert(e_noisy, num_steps=args.num_steps, beam=args.beam)
        rec_secs = time.time() - t0
        m = atk.score(recon, texts, clean_emb=e0)
        pos_acc = positional_token_acc(atk, recon, texts)

        # --- probe: I_G (geometry-only, on clipped clean embeddings; never sees the attack) ---
        ts = time.time()
        ig = spectral_channel_mi(E0=e0_clip, sigma=(sigma if sigma > 0 else 1e-6),
                                 H_X=H_X, H_e0=H_e0, n_tokens=args.max_tokens, vocab=int(vocab))
        ig_secs = time.time() - ts
        # --- probe: CLUB (trains a variational net) ---
        ts = time.time()
        club = club_mi_upper_bound(e_noisy, e0, max_rows=min(600, n), seed=0)["club_mi_bits"]
        club_secs = time.time() - ts
        # --- probe: capPVI (trains a reader) ---
        ts = time.time()
        try:
            pdim = min(64, max(2, n // 4))
            capv = v_information_capacity(e_noisy, labels, family="pca_softmax", dim=pdim, l2=0.1)["reader_top1_acc"]
        except Exception as e:  # noqa: BLE001
            capv = float("nan"); print(f"[smi] capPVI skipped: {e}", flush=True)
        cap_secs = time.time() - ts

        rec = {"epsilon": (None if math.isinf(eps) else eps), "sigma": sigma,
               "token_f1": m["token_f1"], "exact": m["exact"], "cos": m["cos"], "pos_token_acc": pos_acc,
               "i_g_bits": ig["i_g_bits"], "d_eff": ig["d_eff"], "accessible_bits": ig["accessible_bit_ceiling"],
               "fano_exact_ceiling": ig["fano_exact_ceiling"], "rd_pertoken_floor": ig["rd_pertoken_floor"],
               "club_bits": club, "cap_pvi_acc": capv,
               "cost_secs": {"recovery": round(rec_secs, 1), "i_g": round(ig_secs, 4),
                             "club": round(club_secs, 2), "cappvi": round(cap_secs, 2)}}
        records.append(rec)
        es = "inf" if math.isinf(eps) else f"{eps:g}"
        rd = rec["rd_pertoken_floor"]
        print(f"[smi] ε={es:>5} σ={sigma:.3f} | tF1={m['token_f1']:.3f} exact={m['exact']:.3f} "
              f"posAcc={pos_acc:.3f} cos={m['cos']:.3f} | I_G={ig['i_g_bits']:.1f}b d_eff={ig['d_eff']} "
              f"CLUB={club:.1f}b capPVI={capv:.3f} | RDfloor={rd if rd is None else round(rd,3)} "
              f"[I_G {ig_secs*1e3:.1f}ms vs CLUB {club_secs:.1f}s vs capPVI {cap_secs:.1f}s]", flush=True)

    # C1 — Spearman(probe, recovery) over ε for each probe
    R = lambda k: [r[k] for r in records]
    corr = {}
    for met in ("token_f1", "exact", "cos", "pos_token_acc"):
        corr[met] = {"i_g": spearman(R(met), R("i_g_bits")),
                     "club": spearman(R(met), R("club_bits")),
                     "capPVI": spearman(R(met), R("cap_pvi_acc"))}
    # cost (mean)
    cost = {p: float(np.mean([r["cost_secs"][p] for r in records])) for p in ("i_g", "club", "cappvi")}
    # C2 — ceiling violations (exact-match ≤ fano; pos-token-error ≥ rd floor)
    viol = []
    for r in records:
        if r["fano_exact_ceiling"] is not None and r["exact"] > r["fano_exact_ceiling"] + 1e-9:
            viol.append({"eps": r["epsilon"], "kind": "fano_exact", "actual": r["exact"], "ceiling": r["fano_exact_ceiling"]})
        if r["rd_pertoken_floor"] is not None and (1 - r["pos_token_acc"]) < r["rd_pertoken_floor"] - 1e-9:
            viol.append({"eps": r["epsilon"], "kind": "rd_pertoken", "actual_err": 1 - r["pos_token_acc"], "floor": r["rd_pertoken_floor"]})

    print("\n[smi] C1 Spearman(recovery, probe) over ε  [I_G is geometry-only / closed-form]:")
    for met in ("token_f1", "exact", "cos"):
        c = corr[met]
        print(f"   {met:9s}: I_G={c['i_g']:+.2f}  CLUB={c['club']:+.2f}  capPVI={c['capPVI']:+.2f}")
    print(f"[smi] mean cost: I_G={cost['i_g']*1e3:.1f}ms  CLUB={cost['club']:.1f}s  capPVI={cost['cappvi']:.1f}s")
    print(f"[smi] C2 ceiling violations: {len(viol)} (Fano/RD; H proxies flagged)")
    out = {"n": n, "clip_C": C, "H_X_proxy": H_X, "H_e0_proxy": H_e0, "vocab": int(vocab),
           "num_steps": args.num_steps, "beam": args.beam,
           "epsilons": [None if math.isinf(e) else e for e in eps_list],
           "c1_spearman": corr, "cost_secs_mean": cost, "c2_violations": viol, "records": records}
    os.makedirs(REPO / "results", exist_ok=True)
    (REPO / args.out).write_text(json.dumps(out, indent=2))
    print(f"[smi] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
