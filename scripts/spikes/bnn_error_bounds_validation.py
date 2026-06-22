"""Validate the geometry-only BNN/MAP error bounds against the measured
BNN@L0 attack on the real gemma-2-2b embedding codebook (B2 / M1).

Tests claim:bnn-error-bounds-bhattacharyya-fano:
  C1  measured uniform-prior BNN error ∈ [P_e^lb, P_e^ub] at every ε
  C2  the union-bound upper bound reproduces the BNN morphological floor
      from the real distance histogram (top confusable pairs)
  C3  the bounds are computed from the codebook + σ only (no observations)
  C4  the bounds track BNN error across ε (Spearman ρ → 1, co-monotone)

Channel: V ~ Unif(pool), Y = clip(e_V, C_raw) + N(0, σ²I_d), σ = C_raw·z_dp/ε.
BNN = nearest-neighbour (uniform-prior MAP) decode over the clipped pool.
Everything is a table lookup + distance math — no forward pass. Run in the
ROCm container for the table load + GPU Gram/MC.

Usage:
  scripts/run_in_rocm.sh python3 -m scripts.spikes.bnn_error_bounds_validation \
      --out results/bnn_error_bounds_validation.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from talens.measures.channel_error_bounds import (
    fano_equivocation,
    union_bhattacharyya,
)

try:
    import torch
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    torch = None
    DEV = "cpu"


def _clip_np(E: np.ndarray, C: float) -> np.ndarray:
    n = np.linalg.norm(E, axis=1, keepdims=True)
    return E * np.minimum(1.0, C / np.clip(n, 1e-9, None))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2:
        return float("nan")
    try:  # tie-correct (bounds clamp to 0/1 → ties matter)
        from scipy.stats import spearmanr

        rho = spearmanr(a, b).correlation
        return float(rho)
    except Exception:
        ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
        ra -= ra.mean(); rb -= rb.mean()
        den = math.sqrt((ra @ ra) * (rb @ rb))
        return float(ra @ rb / den) if den > 0 else float("nan")


def _bnn_uniform_error(pool_clip: np.ndarray, sigma: float, M: int, seed: int,
                       chunk: int = 64) -> tuple[float, float]:
    """Measured uniform-prior MAP (=BNN) error: for each pool codeword draw
    M noisy observations Y=clip(e_v)+N(0,σ²), decode argmin over the pool,
    error = mean(pred != v). Returns (error, hoeffding_halfwidth_95)."""
    K, d = pool_clip.shape
    if sigma <= 0.0:
        # deterministic argmin (diagonal included): picks self unless an exact
        # duplicate codeword with a lower index collides — that is a real error.
        sq = (pool_clip ** 2).sum(1)
        d2 = sq[:, None] - 2.0 * (pool_clip @ pool_clip.T) + sq[None, :]
        pred = d2.argmin(1)
        err = float((pred != np.arange(K)).mean())
        return err, 0.0
    rng = np.random.default_rng(seed)
    E = pool_clip.astype(np.float32)
    sqE = (E * E).sum(1)
    use_gpu = torch is not None and torch.cuda.is_available()
    if use_gpu:
        Et = torch.from_numpy(E).cuda(); sqEt = torch.from_numpy(sqE).cuda()
    n_err = 0
    n_tot = 0
    for s in range(0, K, chunk):
        idx = np.arange(s, min(s + chunk, K)); B = idx.size
        eps = rng.standard_normal((B, M, d)).astype(np.float32) * sigma
        Y = (E[idx][:, None, :] + eps).reshape(B * M, d)
        true = np.repeat(idx, M)
        if use_gpu:
            Yt = torch.from_numpy(Y).cuda()
            d2 = (Yt * Yt).sum(1, keepdim=True) + sqEt[None, :] - 2.0 * (Yt @ Et.T)
            pred = d2.argmin(1).cpu().numpy()
            del Yt, d2
        else:
            d2 = (Y * Y).sum(1)[:, None] + sqE[None, :] - 2.0 * (Y @ E.T)
            pred = d2.argmin(1)
        n_err += int((pred != true).sum()); n_tot += true.size
    err = n_err / max(1, n_tot)
    hw = math.sqrt(math.log(2 / 0.05) / (2 * n_tot))  # Hoeffding 95%
    return err, hw


def _top_confusable_pairs(pool_clip: np.ndarray, pool_ids: np.ndarray, sigma: float,
                          tok, k: int = 30):
    """Top-k codeword pairs by Bhattacharyya weight exp(-‖Δ‖²/8σ²) — the pairs
    that dominate the union bound (expected: morphological/subword relatives)."""
    sq = (pool_clip ** 2).sum(1)
    d2 = sq[:, None] - 2.0 * (pool_clip @ pool_clip.T) + sq[None, :]
    iu = np.triu_indices(pool_clip.shape[0], k=1)
    w = np.exp(-d2[iu] / (8.0 * sigma ** 2)) if sigma > 0 else np.zeros(iu[0].size)
    order = np.argsort(w)[::-1][:k]
    pairs = []
    for o in order:
        i, j = int(iu[0][o]), int(iu[1][o])
        ti, tj = int(pool_ids[i]), int(pool_ids[j])
        try:
            si, sj = tok.decode([ti]), tok.decode([tj])
        except Exception:
            si, sj = str(ti), str(tj)
        pairs.append({"w": float(w[o]), "dist": float(math.sqrt(max(0.0, d2[i, j]))),
                      "tok_i": ti, "tok_j": tj, "str_i": si, "str_j": sj})
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="unsloth/gemma-2-2b")
    ap.add_argument("--corpus", default="corpora/release-gate-512.txt")
    ap.add_argument("--max-prompts", type=int, default=256)
    ap.add_argument("--epsilons", default="inf,1024,512,256,64")
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--clip-percentile", type=float, default=99.9)
    ap.add_argument("--pool-size", type=int, default=2048)
    ap.add_argument("--mc-fano", type=int, default=64, help="MC draws/codeword for Ĥ_M")
    ap.add_argument("--mc-bnn", type=int, default=64, help="MC draws/codeword for BNN error")
    ap.add_argument("--seed", type=int, default=20260622)
    ap.add_argument("--out", default="results/bnn_error_bounds_validation.json")
    args = ap.parse_args()

    eps_list = [math.inf if s.strip().lower().startswith("inf") else float(s)
                for s in args.epsilons.split(",") if s.strip()]
    rng = np.random.default_rng(args.seed)

    print(f"[bounds] loading {args.model} on {DEV}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        attn_implementation="eager", device_map=DEV).eval()
    table = model.get_input_embeddings().weight.detach().float().cpu().numpy().astype(np.float32)
    vocab, d = table.shape

    prompts = [l.strip() for l in Path(args.corpus).read_text().splitlines()
               if l.strip()][:args.max_prompts]
    ids_per = [tok(p, return_tensors="pt").input_ids[0].numpy().astype(np.int64) for p in prompts]
    y_all = np.concatenate(ids_per).astype(np.int64)

    # C_raw — identical calibration to unified_dp_sweep.py (corpus token norms)
    flat = y_all[(y_all >= 0) & (y_all < vocab)]
    C_raw = float(np.percentile(np.linalg.norm(table[flat], axis=1), args.clip_percentile))
    z_dp = math.sqrt(2 * math.log(1.25 / args.delta))

    # vocab-disjoint split + pool — identical construction to unified_dp_sweep.py
    distinct = rng.permutation(np.unique(y_all))
    n_tr = int(0.7 * distinct.size)
    te_ids = set(distinct[n_tr:].tolist())
    true_pool = np.array(sorted(te_ids), dtype=np.int64)
    avail = np.setdiff1d(np.arange(vocab, dtype=np.int64), true_pool)
    fill = rng.choice(avail, size=max(0, args.pool_size - true_pool.size), replace=False)
    pool = np.concatenate([true_pool, fill.astype(np.int64)])
    pool_clip = _clip_np(table[pool], C_raw)
    K = pool.shape[0]
    print(f"[bounds] vocab={vocab} d={d} C_raw={C_raw:.3f} z_dp={z_dp:.3f} "
          f"|pool|={K} (true={true_pool.size})", flush=True)

    # precompute the codeword Gram once (cached across ε)
    sq = (pool_clip ** 2).sum(1)
    sqd = sq[:, None] - 2.0 * (pool_clip @ pool_clip.T) + sq[None, :]
    sqd = np.maximum(sqd, 0.0)

    recs = []
    print(f"\n{'eps':>6} {'r':>5} {'sigma':>7} | {'BNN_err':>8} {'±':>6} | "
          f"{'P_e^lb':>7} {'P_e^ub':>7} {'P_e^ubB':>8} | {'in?':>3} {'H|Y':>6}", flush=True)
    for eps in eps_list:
        sigma = 0.0 if math.isinf(eps) else C_raw * z_dp / eps
        r = sigma * math.sqrt(d) / C_raw if sigma > 0 else 0.0
        ub = union_bhattacharyya(pool_clip, sigma, sq_dists=sqd)
        lb = fano_equivocation(pool_clip, sigma, M=args.mc_fano, seed=args.seed + 1)
        bnn_err, bnn_hw = _bnn_uniform_error(pool_clip, sigma, args.mc_bnn, args.seed + 2)
        # certified bracketing: use the one-sided LCB on H(V|Y) for the lower
        # side (Ĥ_M raw is unbiased but not itself a certified bound), and the
        # Hoeffding half-width on the measured BNN error.
        inside = bool(lb["p_e_lb_lcb"] <= bnn_err + bnn_hw
                      and bnn_err <= ub["p_e_ub"] + bnn_hw)
        es = "inf" if math.isinf(eps) else f"{eps:g}"
        print(f"{es:>6} {r:>5.2f} {sigma:>7.4f} | {bnn_err:>8.4f} {bnn_hw:>6.4f} | "
              f"{lb['p_e_lb']:>7.4f} {ub['p_e_ub']:>7.4f} {ub['p_e_ub_bhat']:>8.4f} | "
              f"{'Y' if inside else 'N':>3} {lb['h_cond_bits']:>6.2f}", flush=True)
        recs.append({
            "epsilon": None if math.isinf(eps) else eps, "r": r, "sigma": sigma,
            "bnn_err": bnn_err, "bnn_hoeffding_hw": bnn_hw,
            "bnn_ttrsr": 1.0 - bnn_err,
            "p_e_lb": lb["p_e_lb"], "p_e_lb_lcb": lb["p_e_lb_lcb"],
            "p_e_ub": ub["p_e_ub"], "p_e_ub_bhat": ub["p_e_ub_bhat"],
            "p_e_ub_raw": ub["p_e_ub_raw"], "h_cond_bits": lb["h_cond_bits"],
            "h_se": lb["se"], "min_dist": ub["min_dist"], "inside": inside,
        })

    # C4: do the bounds track the BNN error across ε?
    sig = np.array([rr["sigma"] for rr in recs])
    bnn = np.array([rr["bnn_err"] for rr in recs])
    rho_ub = _spearman([rr["p_e_ub"] for rr in recs], bnn)
    rho_lb = _spearman([rr["p_e_lb"] for rr in recs], bnn)
    n_inside = sum(rr["inside"] for rr in recs)
    print(f"\n[bounds] bracketing held at {n_inside}/{len(recs)} ε  |  "
          f"ρ(P_e^ub,BNN_err)={rho_ub:+.3f}  ρ(P_e^lb,BNN_err)={rho_lb:+.3f}", flush=True)

    # C2: morphological floor — top confusable pairs at the HIGHEST-noise ε
    # (smallest finite ε ⇒ largest σ ⇒ where confusion actually happens).
    finite_eps = [e for e in eps_list if not math.isinf(e)]
    floor_eps = min(finite_eps) if finite_eps else None
    floor_pairs = []
    if floor_eps is not None:
        floor_sigma = C_raw * z_dp / floor_eps
        floor_pairs = _top_confusable_pairs(pool_clip, pool, floor_sigma, tok, k=30)
        print(f"\n[bounds] top confusable pairs @ε={floor_eps:g} (σ={floor_sigma:.4f}):", flush=True)
        for p in floor_pairs[:12]:
            print(f"    w={p['w']:.3e} d={p['dist']:.3f}  {p['str_i']!r} ~ {p['str_j']!r}", flush=True)

    out = {
        "model": args.model, "vocab": vocab, "d": d, "C_raw": C_raw, "z_dp": z_dp,
        "pool_size": K, "true_pool": int(true_pool.size), "seed": args.seed,
        "mc_fano": args.mc_fano, "mc_bnn": args.mc_bnn,
        "records": recs, "n_inside": n_inside, "n_eps": len(recs),
        "rho_ub_bnn": rho_ub, "rho_lb_bnn": rho_lb,
        "floor_epsilon": floor_eps, "top_confusable_pairs": floor_pairs,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[bounds] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
