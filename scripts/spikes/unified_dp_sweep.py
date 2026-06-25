#!/usr/bin/env python3
"""unified_dp_sweep.py — consistent DP ε-sweep: Ridge / Bayes-NN / FMV × CLUB / CapPVI / MDL.

Resolves the B2-L0 ↔ B6c reporting inconsistency: same prompts, same candidate pool,
same ε grid, same shuffle-selectivity controls at both observation levels.

  L0  (embedding):     Ridge + Bayes-NN TTRSR (exact MAP; FMV≡BNN at L0 — not shown separately)
  L20 (propagated DP): Ridge + FMV Vec2Text (forward-model-in-loop)

Three IT probes at each level (CLUB, CapPVI, MDL-SDL) with shuffle-selectivity controls.
Spearman ρ(attack-TTRSR, probe-selectivity) over ε reported at end.

  scripts/run_in_rocm.sh python3 scripts/spikes/unified_dp_sweep.py
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from talens.probes.club import club_mi_upper_bound
from talens.probes.mdl import online_code_length
from talens.probes.vinfo_capacity import v_information_capacity

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ── DP hook ────────────────────────────────────────────────────────────────────

class _DPHook:
    def __init__(self, C: float, sigma: float):
        self.C, self.sigma = C, sigma

    def __call__(self, mod, inp, out):
        f = out.float(); n = f.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        f = f * (self.C / n).clamp_max(1.0)
        if self.sigma > 0:
            f = f + self.sigma * torch.randn_like(f)
        return f.to(out.dtype)


# ── model / capture ────────────────────────────────────────────────────────────

@torch.no_grad()
def _capture(model, tok, prompts, layer: int, device, hook=None):
    """Capture resid_post at `layer`. Returns (list_of_(T,d), list_of_ids)."""
    mats, ids_list = [], []
    h = model.model.embed_tokens.register_forward_hook(hook) if hook is not None else None
    for p in prompts:
        inp = tok(p, return_tensors="pt").input_ids.to(device)
        hs = model(inp, output_hidden_states=True, use_cache=False).hidden_states
        mats.append(hs[layer + 1][0].float().cpu().numpy())
        ids_list.append(inp[0].cpu().numpy().astype(np.int64))
    if h is not None:
        h.remove()
    return mats, ids_list


def _stack(mats, ids_list):
    Xs, ys = [], []
    for m, ids in zip(mats, ids_list):
        n = min(m.shape[0], ids.shape[0])
        Xs.append(m[:n]); ys.append(ids[:n])
    return np.concatenate(Xs, 0), np.concatenate(ys, 0).astype(np.int64)


# ── attacks ────────────────────────────────────────────────────────────────────

def _clip_np(E: np.ndarray, C: float) -> np.ndarray:
    n = np.linalg.norm(E, axis=1, keepdims=True)
    return E * np.minimum(1.0, C / np.clip(n, 1e-9, None))


def _ridge_idx(Y_tr, E_tr, Y_te, pool_emb_norm, alphas=(1e-2, 1.0, 1e2),
               fixed_alpha=None):
    """Ridge obs→clean-emb + cosine match. Returns pool-position predictions.
    fixed_alpha: skip train-score selection (use for depth layers where
    n≈d and train-score selection picks too-small alpha, overfitting)."""
    d = Y_tr.shape[1]
    ec_n = E_tr / np.clip(np.linalg.norm(E_tr, axis=1, keepdims=True), 1e-9, None)
    alphas_to_try = [fixed_alpha] if fixed_alpha is not None else alphas
    best_sc, best_idx = None, None
    for a in alphas_to_try:
        W = np.linalg.solve(Y_tr.T @ Y_tr + a * np.eye(d), Y_tr.T @ E_tr)
        pred_tr = Y_tr @ W
        pred_tr /= np.clip(np.linalg.norm(pred_tr, axis=1, keepdims=True), 1e-9, None)
        sc = float((pred_tr * ec_n).sum(1).mean())
        if best_sc is None or sc > best_sc:
            best_sc = sc
            pred_te = Y_te @ W
            pred_te /= np.clip(np.linalg.norm(pred_te, axis=1, keepdims=True), 1e-9, None)
            best_idx = (pred_te @ pool_emb_norm.T).argmax(1)
    return best_idx


def _bayes_nn_idx(Y_te, pool_clip, sigma, log_prior=None):
    """Exact MAP: argmax_v [ -||Y-cv||²/(2σ²) + log π_v ]."""
    cv2 = (pool_clip ** 2).sum(1)
    cross = Y_te @ pool_clip.T
    score = 2.0 * cross - cv2[None, :]
    if sigma > 0:
        score /= (2.0 * sigma ** 2)
    if log_prior is not None:
        score = score + log_prior[None, :]
    return score.argmax(1)


def _train_mlp(Xtr: np.ndarray, Ytr: np.ndarray, hidden: int = 384,
               epochs: int = 250, seed: int = 0):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(Xtr.shape[1], hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, Ytr.shape[1]),
    ).to(DEV)
    xt = torch.from_numpy(Xtr).to(DEV)
    yt = torch.from_numpy(Ytr).to(DEV)
    yt = yt / yt.norm(dim=1, keepdim=True).clamp_min(1e-9)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    for _ in range(epochs):
        opt.zero_grad()
        p = net(xt); p = p / p.norm(dim=1, keepdim=True).clamp_min(1e-9)
        (1.0 - (p * yt).sum(1)).mean().backward()
        opt.step()
    return net.eval()


@torch.no_grad()
def _decoder_topk(net, X_te: np.ndarray, pool_emb: np.ndarray, k: int):
    p = net(torch.from_numpy(X_te).to(DEV))
    p = p / p.norm(dim=1, keepdim=True).clamp_min(1e-9)
    pe = torch.from_numpy(pool_emb).to(DEV)
    pe = pe / pe.norm(dim=1, keepdim=True).clamp_min(1e-9)
    sims = p @ pe.T
    topk_pos = sims.topk(k, dim=1).indices.cpu().numpy()
    top1_pos = sims.argmax(1).cpu().numpy()
    return topk_pos, top1_pos


@torch.no_grad()
def _fwd_resid_last(model, seqs: np.ndarray, layer: int, device):
    t = torch.from_numpy(seqs).to(device)
    hs = model(t, output_hidden_states=True, use_cache=False).hidden_states
    return hs[layer + 1][:, -1, :].float().cpu().numpy()


def _run_fmv(model, ids_per, per_mats, pool, pool_emb_raw, table,
             tr_ids, te_ids, layer, C, device, topk, max_pos, seed):
    """FMV Vec2Text at `layer`: decoder top-k → forward-model re-rank.
    Returns (fmv_ttrsr, dec_ttrsr)."""
    rng = np.random.default_rng(seed)
    X, y = _stack(per_mats, ids_per)
    tr = np.array([i for i, t in enumerate(y) if t in tr_ids])
    te = np.array([i for i, t in enumerate(y) if t in te_ids])
    if te.size == 0:
        return float("nan"), float("nan")

    dec = _train_mlp(X[tr], table[y[tr]], seed=seed)
    dtopk_pos, dtop1_pos = _decoder_topk(dec, X[te], pool_emb_raw, topk)
    dec_ttrsr = float((pool[dtop1_pos] == y[te]).mean())

    # Build (pi, pos) index for test token positions, same order as _stack
    pos_index = []
    for pi, (m, ids) in enumerate(zip(per_mats, ids_per)):
        n = min(m.shape[0], ids.shape[0])
        for pos in range(n):
            if ids[pos] in te_ids:
                pos_index.append((pi, pos))

    n_score = min(len(pos_index), max_pos)
    sel = (rng.choice(len(pos_index), n_score, replace=False)
           if len(pos_index) > n_score else np.arange(len(pos_index)))

    clip_h = model.model.embed_tokens.register_forward_hook(_DPHook(C, 0.0))
    fmv_ok = 0
    for j in sel:
        pi, pos = pos_index[j]
        cands_pool = dtopk_pos[j]          # (k,) pool positions
        cands_tok = pool[cands_pool]        # (k,) token ids
        prefix = ids_per[pi][:pos]
        if pos > 0:
            seqs = np.concatenate(
                [np.tile(prefix, (len(cands_tok), 1)), cands_tok[:, None]], axis=1
            ).astype(np.int64)
        else:
            seqs = cands_tok[:, None].astype(np.int64)
        resid = _fwd_resid_last(model, seqs, layer, device)   # (k, d)
        yobs = per_mats[pi][pos]                               # (d,)
        best = cands_pool[int(np.argmin(((resid - yobs[None, :]) ** 2).sum(1)))]
        fmv_ok += int(pool[best] == ids_per[pi][pos])
    clip_h.remove()
    return fmv_ok / max(1, n_score), dec_ttrsr


# ── probes ─────────────────────────────────────────────────────────────────────

def _safe(v) -> float:
    if v is None:
        return float("nan")
    f = float(v)
    return float("nan") if math.isnan(f) else f


def _run_probes(X, y, table, seed, max_rows, cap_dim):
    """CLUB + CapPVI + MDL-SDL, real and shuffle. Returns (real, sel) dicts."""
    ck = dict(max_rows=max_rows, seed=seed)
    pk = dict(family="pca_softmax", dim=cap_dim, l2=0.1, max_rows=max_rows, seed=seed)
    mk = dict(max_classes=256, seed=seed)

    rc = _safe(club_mi_upper_bound(X, table[y], **ck).get("club_mi_bits"))
    sc = _safe(club_mi_upper_bound(X, table[y], control="shuffle", **ck).get("club_mi_bits"))

    rp = _safe(v_information_capacity(X, y, **pk).get("reader_top1_acc"))
    sp_ = _safe(v_information_capacity(X, y, control="shuffle", **pk).get("reader_top1_acc"))

    rm = _safe(online_code_length(X, y, **mk).get("surplus_description_length_bits"))
    sm = _safe(online_code_length(X, y, control="shuffle", **mk).get("surplus_description_length_bits"))

    real = {"club": rc, "pvi": rp, "mdl": rm}
    sel  = {"club": rc - sc, "pvi": rp - sp_, "mdl": rm - sm}
    return real, sel


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="unsloth/gemma-2-2b")
    ap.add_argument("--corpus", default="corpora/release-gate-512.txt")
    ap.add_argument("--max-prompts", type=int, default=256)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--epsilons", default="inf,1024,512,256,64")
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--clip-percentile", type=float, default=99.9)
    ap.add_argument("--pool-size", type=int, default=2048)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--max-test-pos", type=int, default=300)
    ap.add_argument("--cap-dim", type=int, default=64)
    ap.add_argument("--club-max-rows", type=int, default=2000)
    ap.add_argument("--probe-rows", type=int, default=3000,
                    help="subsample rows for probes (speed)")
    ap.add_argument("--seed", type=int, default=20260622)
    ap.add_argument("--out", default="results/unified_dp_sweep.json")
    args = ap.parse_args()

    eps_list = [
        math.inf if s.strip().lower().startswith("inf") else float(s)
        for s in args.epsilons.split(",") if s.strip()
    ]
    rng = np.random.default_rng(args.seed)

    # ── load model ──────────────────────────────────────────────────────────────
    print(f"[unified] loading {args.model} on {DEV}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        attn_implementation="eager", device_map=DEV,
    ).eval()
    table = model.get_input_embeddings().weight.detach().float().cpu().numpy().astype(np.float32)
    vocab, d = table.shape
    prompts = [l.strip() for l in Path(args.corpus).read_text().splitlines()
               if l.strip()][:args.max_prompts]
    ids_raw = [tok(p, return_tensors="pt").input_ids[0].numpy().astype(np.int64) for p in prompts]
    print(f"[unified] vocab={vocab} d={d} prompts={len(prompts)}", flush=True)

    # ── calibrate C ─────────────────────────────────────────────────────────────
    # C_raw: from raw embedding table — used for L0 (table-space) noise.
    flat = np.concatenate(ids_raw)
    flat = flat[(flat >= 0) & (flat < vocab)]
    C_raw = float(np.percentile(np.linalg.norm(table[flat], axis=1), args.clip_percentile))

    # C_runtime: from model's actual embedding *output* norms — used for L20 hook.
    # Gemma-2 (and others) scale embeddings by sqrt(hidden_size) before layer 0;
    # the hook runs on the output tensor, so C must match the *runtime* scale.
    # Matching b2_propagated_dp.py calibration exactly.
    cal: list = []
    _ch = model.model.embed_tokens.register_forward_hook(
        lambda m, i, o: cal.append(o.float().norm(dim=-1).flatten().cpu())
    )
    with torch.no_grad():
        for p in prompts[:48]:
            model(tok(p, return_tensors="pt").input_ids.to(DEV), use_cache=False)
    _ch.remove()
    C_runtime = float(np.percentile(torch.cat(cal).numpy(), args.clip_percentile))

    z_dp = math.sqrt(2 * math.log(1.25 / args.delta))
    print(f"[unified] C_raw={C_raw:.3f}  C_runtime={C_runtime:.3f}  z_dp={z_dp:.3f}", flush=True)

    # ── clean L20 capture (for vocab-disjoint split) ─────────────────────────
    print(f"[unified] capturing clean L{args.layer}...", flush=True)
    t0 = time.time()
    mats_clean, ids_per = _capture(model, tok, prompts, args.layer, DEV)
    print(f"[unified] clean capture {time.time()-t0:.1f}s", flush=True)

    X_clean, y_all = _stack(mats_clean, ids_per)
    N = y_all.shape[0]

    # shared vocab-disjoint split + pool
    distinct = rng.permutation(np.unique(y_all))
    n_tr = int(0.7 * distinct.size)
    tr_ids = set(distinct[:n_tr].tolist())
    te_ids = set(distinct[n_tr:].tolist())
    tr_idx = np.array([i for i, t in enumerate(y_all) if t in tr_ids])
    te_idx = np.array([i for i, t in enumerate(y_all) if t in te_ids])

    true_pool = np.array(sorted(te_ids), dtype=np.int64)
    avail = np.setdiff1d(np.arange(vocab, dtype=np.int64), true_pool)
    fill = rng.choice(avail, size=max(0, args.pool_size - true_pool.size), replace=False)
    pool = np.concatenate([true_pool, fill.astype(np.int64)])
    pool_emb_raw  = table[pool]
    pool_clip     = _clip_np(pool_emb_raw, C_raw)
    pool_emb_norm = pool_clip / np.clip(np.linalg.norm(pool_clip, axis=1, keepdims=True), 1e-9, None)

    counts = np.bincount(y_all, minlength=vocab).astype(np.float64)
    log_prior_pool = np.log((counts[pool] + 1.0) / (counts.sum() + vocab))

    E_clip = _clip_np(table[y_all], C_raw)   # clipped clean embeddings for all tokens (L0)

    print(f"[unified] N={N} n_tr={tr_idx.size} n_te={te_idx.size} pool={pool.size}", flush=True)

    l0_recs, l20_recs = [], []

    # ════════════════ Phase A: L0 (embedding level) ═════════════════════════════
    print("\n[unified] ── Phase A: L0 (embedding) ──", flush=True)
    for eps in eps_list:
        sigma = 0.0 if math.isinf(eps) else C_raw * z_dp / eps
        r = sigma * math.sqrt(d) / C_raw if sigma > 0 else 0.0
        es = "inf" if math.isinf(eps) else f"{eps:g}"

        noise = (rng.standard_normal(E_clip.shape).astype(np.float32) * sigma
                 if sigma > 0 else np.zeros_like(E_clip))
        Y_l0 = E_clip + noise

        ridge_pool = _ridge_idx(Y_l0[tr_idx], E_clip[tr_idx], Y_l0[te_idx], pool_emb_norm)
        ridge_ttrsr = float((pool[ridge_pool] == y_all[te_idx]).mean())

        bnn_pool = _bayes_nn_idx(Y_l0[te_idx], pool_clip, sigma, log_prior_pool)
        bnn_ttrsr = float((pool[bnn_pool] == y_all[te_idx]).mean())

        n_pr = min(Y_l0.shape[0], args.probe_rows)
        pr_idx = rng.choice(Y_l0.shape[0], n_pr, replace=False)
        real_p, sel_p = _run_probes(Y_l0[pr_idx], y_all[pr_idx], table,
                                    args.seed, args.club_max_rows, args.cap_dim)

        print(f"[L0]  ε={es:>5} r={r:.2f} │ ridge={ridge_ttrsr:.3f} BNN={bnn_ttrsr:.3f} "
              f"(↑{bnn_ttrsr-ridge_ttrsr:+.3f}) │ "
              f"CLUB-sel={sel_p['club']:.0f}b  PVI-sel={sel_p['pvi']:.3f}  MDL-sel={sel_p['mdl']:.0f}b",
              flush=True)

        l0_recs.append({
            "epsilon": None if math.isinf(eps) else eps,
            "sigma": sigma, "r": r,
            "ridge": ridge_ttrsr, "bnn": bnn_ttrsr, "bnn_uplift": bnn_ttrsr - ridge_ttrsr,
            "club_real": real_p["club"], "club_sel": sel_p["club"],
            "pvi_real":  real_p["pvi"],  "pvi_sel":  sel_p["pvi"],
            "mdl_real":  real_p["mdl"],  "mdl_sel":  sel_p["mdl"],
        })

    # ════════════════ Phase B: L20 (propagated DP) ══════════════════════════════
    print(f"\n[unified] ── Phase B: L{args.layer} (propagated DP) ──", flush=True)
    for eps in eps_list:
        sigma = 0.0 if math.isinf(eps) else C_runtime * z_dp / eps
        r = sigma * math.sqrt(d) / C_runtime if sigma > 0 else 0.0
        es = "inf" if math.isinf(eps) else f"{eps:g}"

        torch.manual_seed(args.seed + (0 if math.isinf(eps) else int(eps)))
        hook = _DPHook(C_runtime, sigma)  # always clip; sigma=0 at ε=∞ = clip-only; must match FMV
        t0 = time.time()
        mats_dp, _ = _capture(model, tok, prompts, args.layer, DEV, hook=hook)
        print(f"[L{args.layer}] ε={es:>5} r={r:.2f} capture={time.time()-t0:.1f}s", end="", flush=True)

        X_dp, y_dp = _stack(mats_dp, ids_per)
        tr_dp = np.array([i for i, t in enumerate(y_dp) if t in tr_ids])
        te_dp = np.array([i for i, t in enumerate(y_dp) if t in te_ids])

        # fixed alpha=1.0 matches b2_propagated_dp; train-score selection picks 1e-2
        # (too little reg when n≈d=2304) and overfits
        pool_norm_raw = pool_emb_raw / np.clip(np.linalg.norm(pool_emb_raw, axis=1, keepdims=True), 1e-9, None)
        ridge_pool = _ridge_idx(X_dp[tr_dp], table[y_dp[tr_dp]], X_dp[te_dp],
                                pool_norm_raw, fixed_alpha=1.0)
        ridge_ttrsr = float((pool[ridge_pool] == y_dp[te_dp]).mean())

        t1 = time.time()
        fmv_ttrsr, dec_ttrsr = _run_fmv(
            model, ids_per, mats_dp, pool, pool_emb_raw, table,
            tr_ids, te_ids, args.layer, C_runtime, DEV,
            args.topk, args.max_test_pos, args.seed,
        )
        print(f" fmv={time.time()-t1:.1f}s", end="", flush=True)

        n_pr = min(X_dp.shape[0], args.probe_rows)
        pr_idx = rng.choice(X_dp.shape[0], n_pr, replace=False)
        real_p, sel_p = _run_probes(X_dp[pr_idx], y_dp[pr_idx], table,
                                    args.seed, args.club_max_rows, args.cap_dim)

        print(f" │ ridge={ridge_ttrsr:.3f} FMV={fmv_ttrsr:.3f} "
              f"(↑{fmv_ttrsr-ridge_ttrsr:+.3f}) │ "
              f"CLUB-sel={sel_p['club']:.0f}b  PVI-sel={sel_p['pvi']:.3f}  MDL-sel={sel_p['mdl']:.0f}b",
              flush=True)

        l20_recs.append({
            "epsilon": None if math.isinf(eps) else eps,
            "sigma": sigma, "r": r,
            "ridge": ridge_ttrsr, "fmv": fmv_ttrsr, "fmv_uplift": fmv_ttrsr - ridge_ttrsr,
            "dec": dec_ttrsr,
            "club_real": real_p["club"], "club_sel": sel_p["club"],
            "pvi_real":  real_p["pvi"],  "pvi_sel":  sel_p["pvi"],
            "mdl_real":  real_p["mdl"],  "mdl_sel":  sel_p["mdl"],
        })

    # ════════════════ Unified table ══════════════════════════════════════════════
    print("\n")
    hdr = (f"{'ε':>6} {'r':>4} │ {'ridge@L0':>8} {'BNN@L0':>8} {'↑BNN':>6} │"
           f" {'ridge@L20':>9} {'FMV@L20':>7} {'↑FMV':>6} │"
           f" {'CLUB@L0':>7} {'PVI@L0':>6} {'MDL@L0':>7} │"
           f" {'CLUB@L20':>8} {'PVI@L20':>7} {'MDL@L20':>8}")
    bar = "─" * len(hdr)

    def fmt(v, spec=".3f"):
        return "  nan " if (v is None or math.isnan(float(v))) else f"{float(v):{spec}}"

    print(hdr); print(bar)
    for r0, r20 in zip(l0_recs, l20_recs):
        eps_s = "∞" if r0["epsilon"] is None else f"{r0['epsilon']:g}"
        print(
            f"{eps_s:>6} {r0['r']:>4.2f} │"
            f" {fmt(r0['ridge']):>8} {fmt(r0['bnn']):>8} {fmt(r0['bnn_uplift'],'+.3f'):>6} │"
            f" {fmt(r20['ridge']):>9} {fmt(r20['fmv']):>7} {fmt(r20['fmv_uplift'],'+.3f'):>6} │"
            f" {fmt(r0['club_sel'],'.0f'):>7} {fmt(r0['pvi_sel']):>6} {fmt(r0['mdl_sel'],'.0f'):>7} │"
            f" {fmt(r20['club_sel'],'.0f'):>8} {fmt(r20['pvi_sel']):>7} {fmt(r20['mdl_sel'],'.0f'):>8}"
        )
    print(bar)
    print("Probes = selectivity (real − shuffle); CLUB/MDL in bits, PVI in acc units.")

    # ════════════════ Spearman ρ(attack, probe) ══════════════════════════════════
    def _col0(k): return [r[k] for r in l0_recs]
    def _col20(k): return [r[k] for r in l20_recs]

    attacks = [
        ("ridge@L0",  _col0("ridge")),
        ("BNN@L0",    _col0("bnn")),
        ("ridge@L20", _col20("ridge")),
        ("FMV@L20",   _col20("fmv")),
    ]
    probes = [
        ("CLUB@L0",  _col0("club_sel")),
        ("PVI@L0",   _col0("pvi_sel")),
        ("MDL@L0",   _col0("mdl_sel")),
        ("CLUB@L20", _col20("club_sel")),
        ("PVI@L20",  _col20("pvi_sel")),
        ("MDL@L20",  _col20("mdl_sel")),
    ]

    def _sp(a, b) -> float:
        a, b = np.asarray(a, float), np.asarray(b, float)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 3 or np.std(a[m]) < 1e-9 or np.std(b[m]) < 1e-9:
            return float("nan")
        return float(stats.spearmanr(a[m], b[m]).statistic)

    print("\nSpearman ρ(attack TTRSR, probe selectivity) over ε:")
    ph = " ".join(f"{pn:>9}" for pn, _ in probes)
    print(f"  {'':13}  {ph}")
    corr_out = {}
    for att_n, att_v in attacks:
        row = "  " + f"{att_n:13} "
        for pr_n, pr_v in probes:
            rho = _sp(att_v, pr_v)
            corr_out[f"{att_n}|{pr_n}"] = rho
            row += f"  {fmt(rho,'+.2f'):>7}"
        print(row)

    # ════════════════ Save ═══════════════════════════════════════════════════════
    out_d = {
        "model": args.model, "layer": args.layer, "d": d,
        "C_raw": C_raw, "C_runtime": C_runtime,
        "epsilons": args.epsilons,
        "n_train": int(tr_idx.size), "n_test": int(te_idx.size),
        "pool_size": int(pool.size),
        "l0_records": l0_recs,
        "l20_records": l20_recs,
        "spearman": corr_out,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out_d, indent=2))
    print(f"\n[unified] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
