#!/usr/bin/env python3
"""SGT (Stained Glass Transform) vs Vec2Text on pooled GTR embeddings — the matched
MI-probe-predicts-attack test under a *learned heteroscedastic* defense (Task 7, Block B).

The Block A study verified the spectral channel-MI ``I_G`` as a matched probe for
**isotropic** DP noise. SGT differs from DP only in noise SHAPE. So for a set of target
MI budgets ``B`` (bits) we build THREE Gaussian channels all hitting the same ``I_G=B``:
  iso        — isotropic σ²I (DP baseline),
  sgt_opt    — distortion-optimal heteroscedastic (the SGT MI-budget optimum; head-heavy),
  tail_dump  — isotropic head + huge near-null-mode noise (head SNR even cleaner than iso).
Head SNR at matched B: sgt_opt < iso < tail_dump. Each is inverted with the *pretrained*
gtr-base corrector. Probe = generalized ``spectral_channel_mi_diag`` (geometry-only, never
sees the attack), plus a head-localized ``I_head(k)`` (tail-truncated) candidate refinement.

Reads:
  C1 (within-shape monotone)   : Spearman(B, recovery) per shape.
  C2 (HEADLINE shape-invariance): at matched B, recovery spread across {iso,sgt_opt,tail_dump};
                                  Spearman(recovery, head_SNR) and (recovery, I_head) over all cells.
  C3 (defense utility)         : distortion D_tot + released-vs-clean cosine per shape at matched B.

Covariance ``Σ`` is estimated from a LARGE clean pool (cheap, no attack) so it is full-rank;
Vec2Text recovery is evaluated on a held-out subset. Run in the ROCm container:
    scripts/run_in_rocm.sh python3 scripts/eval/sgt_attack_eval.py --pool-n 1200 --eval-n 96
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "eval"))
sys.path.insert(0, str(REPO / "scripts" / "defenses"))
import numpy as np  # noqa: E402
from vec2text_attack import Vec2TextAttack  # noqa: E402
from sgt import fit_covariance, build_sgt, _ig_bits  # noqa: E402
from talens.measures.spectral_channel_mi import spectral_channel_mi_diag  # noqa: E402


def spearman(a, b):
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    return 0.0 if np.std(a) < 1e-9 or np.std(b) < 1e-9 else float(stats.spearmanr(a, b).statistic)


def positional_token_acc(atk, recon, true):
    accs = []
    for r, t in zip(recon, true):
        rt = atk.tokenizer(r)["input_ids"]; tt = atk.tokenizer(t)["input_ids"]
        m = min(len(rt), len(tt))
        accs.append(np.mean([rt[i] == tt[i] for i in range(m)]) if m else 0.0)
    return float(np.mean(accs))


def head_localized_ig(lam, v, ks):
    """Tail-truncated channel-MI ``I_head(k) = ½ Σ_{i<k} log2(1+λ_i/v_i)`` (bits)."""
    t = 0.5 * np.log1p(lam / v) / math.log(2.0)
    cs = np.cumsum(t)
    return {int(k): float(cs[min(int(k), len(cs)) - 1]) for k in ks}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="corpora/release-gate-512.txt")
    ap.add_argument("--pool-n", type=int, default=1200, help="texts for Σ estimation (embed only)")
    ap.add_argument("--eval-n", type=int, default=96, help="held-out texts for Vec2Text recovery")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--budgets", default="auto",
                    help="target I_G in bits ('auto' → derive from --ref-sigmas on the fitted spectrum)")
    ap.add_argument("--ref-sigmas", default="0.012,0.025,0.05,0.1",
                    help="Block-A-comparable isotropic σ; their iso I_G become the budgets when --budgets auto")
    ap.add_argument("--shapes", default="iso,sgt_opt,tail_dump")
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--clip-percentile", type=float, default=99.9)
    ap.add_argument("--head-ks", default="8,16,32,64")
    ap.add_argument("--seed", type=int, default=20260624)
    ap.add_argument("--out", default="refine-logs/embed-sgt/runs/sweep/sgt_eval.json")
    args = ap.parse_args()

    shapes = [s.strip() for s in args.shapes.split(",") if s.strip()]
    head_ks = [int(s) for s in args.head_ks.split(",") if s.strip()]
    rng = np.random.default_rng(args.seed)
    print("[sgt] loading GTR + gtr-base corrector", flush=True)
    atk = Vec2TextAttack()
    vocab = atk.tokenizer.vocab_size

    raw = [l.strip() for l in open(REPO / args.corpus) if l.strip()]
    rng.shuffle(raw)
    texts_all = atk.canonicalize(raw, max_tokens=args.max_tokens)
    eval_texts = texts_all[: args.eval_n]
    pool_texts = texts_all[args.eval_n: args.eval_n + args.pool_n]
    if len(pool_texts) < 200:
        pool_texts = texts_all[: max(200, args.pool_n)]   # small corpus → reuse (note in output)
    print(f"[sgt] eval_n={len(eval_texts)} pool_n={len(pool_texts)}", flush=True)

    e0_pool = atk.embed(pool_texts)
    C = float(np.percentile(np.linalg.norm(e0_pool, axis=1), args.clip_percentile))
    sc_pool = np.minimum(1.0, C / (np.linalg.norm(e0_pool, axis=1, keepdims=True) + 1e-9))
    lam, V, _ = fit_covariance((e0_pool * sc_pool).astype(np.float64))   # Σ of clipped pool

    e0_eval = atk.embed(eval_texts)
    sc_eval = np.minimum(1.0, C / (np.linalg.norm(e0_eval, axis=1, keepdims=True) + 1e-9))
    e0_eval_clip = (e0_eval * sc_eval).astype(np.float32)

    H_X = args.max_tokens * math.log2(vocab); H_e0 = H_X
    cov_diag = np.diag(lam)
    if args.budgets.strip().lower() == "auto":
        ref = [float(s) for s in args.ref_sigmas.split(",") if s.strip()]
        budgets = [_ig_bits(lam, np.full_like(lam, s * s)) for s in ref]
        print(f"[sgt] auto budgets from ref σ {ref} → I_G {[round(b,1) for b in budgets]} bits", flush=True)
    else:
        budgets = [float(s) for s in args.budgets.split(",") if s.strip()]
    print(f"[sgt] d={lam.size} clip C={C:.3f} λ1={lam[0]:.3f} rank≈{int((lam>1e-9*lam[0]).sum())} "
          f"H_X≈{H_X:.0f}b", flush=True)

    def score_release(e_rel, tag):
        t0 = time.time(); recon = atk.invert(e_rel, num_steps=args.num_steps, beam=args.beam)
        secs = time.time() - t0
        m = atk.score(recon, eval_texts, clean_emb=e0_eval)
        pos = positional_token_acc(atk, recon, eval_texts)
        rel_cos = float(np.mean(np.sum(e_rel * e0_eval_clip, 1) /
                        (np.linalg.norm(e_rel, axis=1) * np.linalg.norm(e0_eval_clip, axis=1) + 1e-9)))
        print(f"[sgt] {tag:28s} tF1={m['token_f1']:.3f} exact={m['exact']:.3f} posAcc={pos:.3f} "
              f"reconCos={m['cos']:.3f} relCos={rel_cos:.3f} [{secs:.0f}s]", flush=True)
        return {"token_f1": m["token_f1"], "exact": m["exact"], "pos_token_acc": pos,
                "recon_cos": m["cos"], "release_cos": rel_cos, "invert_secs": round(secs, 1)}

    records = []
    # plaintext: clip only, no noise
    rec = {"budget_bits": None, "shape": "plaintext", "i_g_bits": None, "d_eff": lam.size,
           "distortion_total": 0.0, "head_snr_mode0": float("inf"), **score_release(e0_eval_clip, "plaintext(clip-only)")}
    records.append(rec)

    for B in budgets:
        for sh in shapes:
            g = build_sgt(lam, V, shape=sh, budget_bits=B, clip_C=C, seed=args.seed)
            e_rel = g.apply(e0_eval, rng=np.random.default_rng(args.seed + int(B)))
            pr = spectral_channel_mi_diag(cov_diag, g.v, H_X=H_X, H_e0=H_e0,
                                          n_tokens=args.max_tokens, vocab=int(vocab))
            hl = head_localized_ig(lam, g.v, head_ks)
            sc = score_release(e_rel, f"B={B:g} {sh}")
            rec = {"budget_bits": B, "shape": sh, "i_g_bits": pr["i_g_bits"], "d_eff": pr["d_eff"],
                   "distortion_total": g.distortion_total, "head_snr_mode0": float(lam[0] / g.v[0]),
                   "accessible_bits": pr["accessible_bit_ceiling"], "fano_exact_ceiling": pr["fano_exact_ceiling"],
                   "rd_pertoken_floor": pr["rd_pertoken_floor"], "head_ig": hl, **sc}
            records.append(rec)

    # --- persist raw records BEFORE the (cheap, fragile) correlation analysis so the
    #     expensive inversions are never lost to an analysis bug ---
    outp = REPO / args.out; outp.parent.mkdir(parents=True, exist_ok=True)
    base_cfg = {"corpus": args.corpus, "pool_n": len(pool_texts), "eval_n": len(eval_texts),
                "max_tokens": args.max_tokens, "budgets": budgets, "shapes": shapes,
                "num_steps": args.num_steps, "beam": args.beam, "clip_C": C, "d": int(lam.size),
                "H_X_proxy": H_X, "vocab": int(vocab), "seed": args.seed, "head_ks": head_ks}
    outp.write_text(json.dumps({"config": base_cfg, "records": records}, indent=2, default=float))

    # --- analysis ---
    finite = [r for r in records if r["budget_bits"] is not None]
    # C1: within-shape Spearman(I_G, recovery)  (I_G ≈ budget; monotone expected)
    c1 = {}
    for sh in shapes:
        rs = [r for r in finite if r["shape"] == sh]
        c1[sh] = {met: spearman([r["i_g_bits"] for r in rs], [r[met] for r in rs])
                  for met in ("token_f1", "exact", "pos_token_acc", "recon_cos")}
    # C2: at matched budget, recovery spread across shapes; and recovery~head_snr / recovery~I_head over all cells
    spread = {}
    for B in budgets:
        cells = {r["shape"]: r["token_f1"] for r in finite if r["budget_bits"] == B}
        vals = list(cells.values())
        spread[str(B)] = {"per_shape_token_f1": cells, "range": (max(vals) - min(vals)) if vals else 0.0}
    c2_shape = {}
    for met in ("token_f1", "exact", "pos_token_acc"):
        rec_v = [r[met] for r in finite]
        c2_shape[met] = {"vs_i_g": spearman(rec_v, [r["i_g_bits"] for r in finite]),
                         "vs_head_snr": spearman(rec_v, [r["head_snr_mode0"] for r in finite]),
                         "vs_release_cos": spearman(rec_v, [r["release_cos"] for r in finite]),
                         "vs_neg_distortion": spearman(rec_v, [-r["distortion_total"] for r in finite]),
                         **{f"vs_i_head_k{k}": spearman(rec_v, [r["head_ig"][k] for r in finite])
                            for k in head_ks}}
    # C3: utility (distortion + release_cos) per shape at each budget
    c3 = {str(B): {r["shape"]: {"distortion_total": r["distortion_total"], "release_cos": r["release_cos"]}
                   for r in finite if r["budget_bits"] == B} for B in budgets}

    out = {"config": {"corpus": args.corpus, "pool_n": len(pool_texts), "eval_n": len(eval_texts),
                      "max_tokens": args.max_tokens, "budgets": budgets, "shapes": shapes,
                      "num_steps": args.num_steps, "beam": args.beam, "clip_C": C, "d": int(lam.size),
                      "H_X_proxy": H_X, "vocab": int(vocab), "seed": args.seed, "head_ks": head_ks},
           "records": records, "c1_within_shape_spearman": c1,
           "c2_matched_budget_spread": spread, "c2_recovery_vs_probe_spearman": c2_shape, "c3_utility": c3}
    outp.write_text(json.dumps(out, indent=2, default=float))

    print("\n[sgt] C1 within-shape Spearman(I_G, token_f1):",
          {sh: round(c1[sh]["token_f1"], 2) for sh in shapes})
    print("[sgt] C2 matched-budget token_f1 range per B:",
          {b: round(spread[b]["range"], 3) for b in spread})
    print("[sgt] C2 recovery~probe Spearman (token_f1):", {k: round(v, 2) for k, v in c2_shape["token_f1"].items()})
    print(f"[sgt] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
