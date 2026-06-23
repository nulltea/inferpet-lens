"""B(-1) geometry diagnostic — go/no-go gate for the shaped-noise study.

The shaped-noise headline (anisotropic Σ beats isotropic against BNN@L0) is
conditional on the codebook's *in-span margin geometry being anisotropic*. This
script measures that, geometry-only, BEFORE the proof gate (B0):

  (i)   centered-span rank + nullspace trace fraction (isotropic's wasted ~11%)
  (ii)  eigen-spectrum + effective rank of the error-weighted pairwise scatter
        S = Σ_{v≠u} w_vu Δ_vu Δ_vuᵀ,  w_vu = exp(-‖Δ_vu‖²/(8σ²)),  across a σ grid
  (iii) gradient anisotropy of the error-surrogate at isotropic
        (∇_Σ U_B|_{Σ=cI} = +⅛ c⁻² S  ⟹ anisotropy of ∇ = anisotropy of S on the span)
  (iv)  top-pair mass (does a handful of morphological pairs dominate S?)
  (v)   eigenspace stability of S across the σ grid

Decision gate: S anisotropic & stable in-span  → GO (strong headline available).
               S ≈ isotropic on the span        → PIVOT (nullspace+morpho audit only).

Efficient exact scatter via the graph-Laplacian identity:
  S = Σ_{v,u} w_vu (e_v−e_u)(e_v−e_u)ᵀ = 2·Eᵀ L_W E,  L_W = diag(W·1) − W
(W symmetric, w_vv=0; E = clipped pool embeddings, K×d). No K² outer products.

Run in the ROCm container (table load + GPU eigh):
  scripts/run_in_rocm.sh python3 scripts/spikes/anisotropic_geometry_diagnostic.py \
      --out results/anisotropic_geometry_diagnostic.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    import torch
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    torch = None
    DEV = "cpu"


def _clip_np(E: np.ndarray, C: float) -> np.ndarray:
    n = np.linalg.norm(E, axis=1, keepdims=True)
    return E * np.minimum(1.0, C / np.clip(n, 1e-9, None))


def _eff_rank(evals: np.ndarray) -> float:
    """Participation-ratio effective rank: (Σλ)² / Σλ²  (λ ≥ 0)."""
    p = np.clip(evals, 0.0, None)
    s1 = p.sum(); s2 = (p * p).sum()
    return float(s1 * s1 / s2) if s2 > 0 else 0.0


def _entropy_eff_rank(evals: np.ndarray) -> float:
    """exp(Shannon entropy of normalized spectrum) — second eff-rank notion."""
    p = np.clip(evals, 0.0, None)
    s = p.sum()
    if s <= 0:
        return 0.0
    q = p / s
    q = q[q > 0]
    return float(math.exp(-(q * np.log(q)).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="unsloth/gemma-2-2b")
    ap.add_argument("--corpus", default="corpora/release-gate-512.txt")
    ap.add_argument("--max-prompts", type=int, default=256)
    ap.add_argument("--pool-size", type=int, default=2048)
    ap.add_argument("--clip-percentile", type=float, default=99.9)
    ap.add_argument("--delta", type=float, default=1e-5)
    # σ grid via the same ε convention as the bounds study (σ = C·z_dp/ε)
    ap.add_argument("--epsilons", default="128,64,32,16")
    ap.add_argument("--seed", type=int, default=20260622)
    ap.add_argument("--topk-pairs", type=int, default=40)
    ap.add_argument("--out", default="results/anisotropic_geometry_diagnostic.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    eps_list = [float(s) for s in args.epsilons.split(",") if s.strip()]

    print(f"[geom] loading {args.model} on {DEV}", flush=True)
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

    flat = y_all[(y_all >= 0) & (y_all < vocab)]
    C_raw = float(np.percentile(np.linalg.norm(table[flat], axis=1), args.clip_percentile))
    z_dp = math.sqrt(2 * math.log(1.25 / args.delta))

    # same vocab-disjoint pool construction as bnn_error_bounds_validation.py
    distinct = rng.permutation(np.unique(y_all))
    n_tr = int(0.7 * distinct.size)
    te_ids = set(distinct[n_tr:].tolist())
    true_pool = np.array(sorted(te_ids), dtype=np.int64)
    avail = np.setdiff1d(np.arange(vocab, dtype=np.int64), true_pool)
    fill = rng.choice(avail, size=max(0, args.pool_size - true_pool.size), replace=False)
    pool = np.concatenate([true_pool, fill.astype(np.int64)])
    E = _clip_np(table[pool], C_raw).astype(np.float32)        # (K, d) clipped codebook
    K = E.shape[0]
    print(f"[geom] vocab={vocab} d={d} C_raw={C_raw:.3f} z_dp={z_dp:.3f} |pool|={K}", flush=True)

    use_gpu = torch is not None and torch.cuda.is_available()
    Et = torch.from_numpy(E).to(DEV) if use_gpu else None

    # ── (i) centered-span rank + nullspace trace fraction ───────────────────────
    Ec = E - E.mean(0, keepdims=True)                          # centered (spans the Δ subspace)
    if use_gpu:
        cov_c = (torch.from_numpy(Ec).to(DEV).T @ torch.from_numpy(Ec).to(DEV)).cpu().numpy()
    else:
        cov_c = Ec.T @ Ec
    ev_c = np.linalg.eigvalsh(cov_c.astype(np.float64))[::-1]  # desc
    rank_at = {t: int((ev_c > ev_c[0] * t).sum()) for t in (1e-5, 1e-6, 1e-7)}
    span_rank = rank_at[1e-6]                                  # primary threshold
    nullspace_dims = d - span_rank
    nullspace_trace_frac = nullspace_dims / d                  # isotropic's wasted fraction
    # NOTE: pool-specific (K=2048 < d=2304 forces ≥ d−(K−1) nullspace dims); this is
    # the right "wasted trace" for the POOL-restricted BNN codebook, not full vocab.
    Ect = torch.from_numpy(Ec.astype(np.float32)).to(DEV) if use_gpu else None
    print(f"[geom] (i) span_rank={span_rank}/{d} (rank@1e-5/6/7={rank_at[1e-5]}/{rank_at[1e-6]}/{rank_at[1e-7]})  "
          f"nullspace_dims={nullspace_dims}  nullspace_trace_frac={nullspace_trace_frac:.4f}", flush=True)

    # squared pairwise distances (K,K), reused per σ
    sqE = (E * E).sum(1)
    if use_gpu:
        D2t = (torch.from_numpy(sqE).to(DEV)[:, None] - 2.0 * (Et @ Et.T)
               + torch.from_numpy(sqE).to(DEV)[None, :]).clamp_min(0.0)
    else:
        D2 = np.maximum(sqE[:, None] - 2.0 * (E @ E.T) + sqE[None, :], 0.0)

    per_eps = []
    top_eigvecs_by_eps = []
    for eps in eps_list:
        sigma = C_raw * z_dp / eps
        r = sigma * math.sqrt(d) / C_raw
        inv8s2 = 1.0 / (8.0 * sigma ** 2)
        # ── weights w_vu = exp(-‖Δ‖²/(8σ²)), w_vv = 0 — log-rescaled per σ for
        # numerical safety (all metrics below are scale-invariant in W, so the
        # constant exp(−m) rescale changes nothing but avoids underflow). ────────
        edge_eff_count = float("nan"); tok_cov = float("nan")
        if use_gpu:
            A = -D2t * inv8s2
            A.fill_diagonal_(float("-inf"))
            m_off = float(A.max().item())                      # = −min off-diag D2·inv8s2
            W = torch.exp(A - m_off)                            # max entry 1, diag→0
            deg = W.sum(1)
            # S = 2 Ecᵀ L_W Ec  (centered coords; L_W·1=0 so identical, better conditioned)
            EcD = Ect.T * deg[None, :]                          # (d,K)
            S = 2.0 * (EcD @ Ect - (Ect.T @ (W @ Ect)))         # (d,d)
            S = (0.5 * (S + S.T)).cpu().numpy().astype(np.float64)
            iu0, iu1 = torch.triu_indices(K, K, offset=1, device=W.device)
            wv = W[iu0, iu1]
            wv_sum = float(wv.sum().item())
            edge_eff_count = float((wv_sum ** 2) / max(1e-30, float((wv * wv).sum().item())))
            topk = min(args.topk_pairs, wv.numel())
            tw, ti = torch.topk(wv, topk)
            top_pair_mass = float(tw.sum().item() / max(1e-12, wv_sum))
            cov = set()
            top_pairs = []
            for mrank in range(topk):
                i, j = int(iu0[ti[mrank]]), int(iu1[ti[mrank]])
                cov.add(int(pool[i])); cov.add(int(pool[j]))
                if mrank < 12:
                    d_ij = float(math.sqrt(max(0.0, float(D2t[i, j].item()))))
                    try:
                        si, sj = tok.decode([int(pool[i])]), tok.decode([int(pool[j])])
                    except Exception:
                        si, sj = str(int(pool[i])), str(int(pool[j]))
                    top_pairs.append({"w": float(tw[mrank].item()), "dist": d_ij, "a": si, "b": sj})
            tok_cov = len(cov) / float(K)                       # unique-token coverage of top-K pairs
            del W, deg, EcD, wv, A
        else:
            A = -D2 * inv8s2; np.fill_diagonal(A, -np.inf)
            m_off = float(A.max()); W = np.exp(A - m_off)
            deg = W.sum(1)
            S = 2.0 * ((Ec.T * deg[None, :]) @ Ec - Ec.T @ (W @ Ec))
            S = (0.5 * (S + S.T)).astype(np.float64)
            iu = np.triu_indices(K, 1); wv = W[iu]; wv_sum = float(wv.sum())
            edge_eff_count = float((wv_sum ** 2) / max(1e-30, float((wv * wv).sum())))
            order = np.argsort(wv)[::-1][:args.topk_pairs]
            top_pair_mass = float(wv[order].sum() / max(1e-12, wv_sum))
            cov = set(); top_pairs = []
            for rank_m, mi in enumerate(order):
                i, j = int(iu[0][mi]), int(iu[1][mi]); cov.add(int(pool[i])); cov.add(int(pool[j]))
                if rank_m < 12:
                    try:
                        si, sj = tok.decode([int(pool[i])]), tok.decode([int(pool[j])])
                    except Exception:
                        si, sj = str(int(pool[i])), str(int(pool[j]))
                    top_pairs.append({"w": float(wv[mi]), "dist": float(math.sqrt(D2[i, j])), "a": si, "b": sj})
            tok_cov = len(cov) / float(K)

        # ── eigendecomp of S ────────────────────────────────────────────────────
        evals, evecs = np.linalg.eigh(S)                       # asc
        evals = evals[::-1]; evecs = evecs[:, ::-1]
        evals = np.clip(evals, 0.0, None)
        eff = _eff_rank(evals)
        eff_ent = _entropy_eff_rank(evals)
        # gradient anisotropy at isotropic: S vs its isotropic part ON THE SPAN.
        # restrict to span_rank dims; isotropic-on-span has all eigenvalues equal.
        span_evals = evals[:span_rank]
        mean_sp = span_evals.mean()
        # anisotropy index: relative Frobenius norm of the traceless part on span
        aniso = float(np.linalg.norm(span_evals - mean_sp) / (np.linalg.norm(span_evals) + 1e-12))
        top1_frac = float(evals[0] / (evals.sum() + 1e-12))
        top10_frac = float(evals[:10].sum() / (evals.sum() + 1e-12))
        top_eigvecs_by_eps.append(evecs[:, :10].copy())
        print(f"[geom] ε={eps:>5g} r={r:.2f} σ={sigma:.4f} │ eff_rank(S)={eff:.1f} "
              f"(ent {eff_ent:.1f}) │ top1={top1_frac:.3f} top10={top10_frac:.3f} "
              f"aniso={aniso:.3f} │ top{args.topk_pairs}-mass={top_pair_mass:.3f} "
              f"edge_eff#={edge_eff_count:.1f} tok_cov={tok_cov:.3f}", flush=True)
        per_eps.append({
            "epsilon": eps, "r": r, "sigma": sigma,
            "eff_rank_S": eff, "eff_rank_S_entropy": eff_ent,
            "top1_eval_frac": top1_frac, "top10_eval_frac": top10_frac,
            "span_anisotropy": aniso, "top_pair_mass": top_pair_mass,
            "edge_eff_count": edge_eff_count, "token_coverage_topk": tok_cov,
            "top_pairs": top_pairs,
            "S_eval_top20": evals[:20].tolist(),
        })

    # ── (v) eigenspace stability across σ (principal angles of top-10 subspaces) ──
    stability = []
    for a in range(len(top_eigvecs_by_eps) - 1):
        Ua, Ub = top_eigvecs_by_eps[a], top_eigvecs_by_eps[a + 1]
        # cos of principal angles = singular values of UaᵀUb
        s = np.linalg.svd(Ua.T @ Ub, compute_uv=False)
        stability.append({"eps_pair": [eps_list[a], eps_list[a + 1]],
                          "mean_cos_principal_angle": float(np.clip(s, 0, 1).mean())})
    if stability:
        print(f"[geom] (v) top-10 eigenspace stability across σ (mean cosθ): "
              + ", ".join(f"{st['mean_cos_principal_angle']:.3f}" for st in stability), flush=True)

    # ── GATE decision (3-way: BROAD vs SPIKY vs PIVOT) ───────────────────────────
    # Anisotropy exists if S is low-eff-rank vs the span AND top-10 carries real mass.
    # But distinguish *broad* anisotropy (many directions, robust) from *spiky*
    # (a handful of morphological pairs dominate) — the latter is real but means the
    # shaped-noise win is concentrated, changing the headline interpretation.
    eff_ranks = [p["eff_rank_S"] for p in per_eps]
    top10 = [p["top10_eval_frac"] for p in per_eps]
    edge_effs = [p["edge_eff_count"] for p in per_eps]
    tok_covs = [p["token_coverage_topk"] for p in per_eps]
    min_stab = min((st["mean_cos_principal_angle"] for st in stability), default=1.0)
    anisotropic = (np.median(eff_ranks) < 0.5 * span_rank) and (np.median(top10) > 0.10)
    stable = min_stab > 0.6
    # spiky if the weighted-edge effective count is tiny (few pairs carry the mass)
    spiky = np.median(edge_effs) < 64 or np.median(tok_covs) < 0.05
    if not anisotropic:
        gate = "PIVOT"
    elif not stable:
        gate = "GO-UNSTABLE"
    elif spiky:
        gate = "GO-SPIKY"      # anisotropy concentrated on few morphological pairs
    else:
        gate = "GO-BROAD"      # broad in-span anisotropy — strongest headline
    print(f"\n[geom] GATE = {gate}", flush=True)
    print(f"[geom]   anisotropic={anisotropic} (median eff_rank(S)={np.median(eff_ranks):.1f} vs span_rank={span_rank}; "
          f"median top10-mass={np.median(top10):.3f})", flush=True)
    print(f"[geom]   stable={stable} (min top-10 eigenspace cosθ={min_stab:.3f}); "
          f"spiky={spiky} (median edge_eff#={np.median(edge_effs):.1f}, median tok_cov={np.median(tok_covs):.3f})", flush=True)
    print(f"[geom]   interpretation: GO-BROAD→strong shaped-noise headline; "
          f"GO-SPIKY→shaping helps but via a few morphological directions; PIVOT→shaping can't beat iso-in-span.", flush=True)

    out = {
        "model": args.model, "vocab": vocab, "d": d, "C_raw": C_raw, "z_dp": z_dp,
        "pool_size": K, "seed": args.seed,
        "span_rank": span_rank, "rank_at_threshold": rank_at,
        "nullspace_dims": nullspace_dims, "nullspace_trace_frac": nullspace_trace_frac,
        "per_epsilon": per_eps, "eigenspace_stability": stability,
        "gate": gate, "anisotropic": bool(anisotropic), "stable": bool(stable), "spiky": bool(spiky),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[geom] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
