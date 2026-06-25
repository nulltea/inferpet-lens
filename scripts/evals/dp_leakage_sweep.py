#!/usr/bin/env python3
"""DP leakage sweep — reusable cross-layer eval.

One configurable sweep over a defended representation:
  capture residual-stream reps under LOCAL differential privacy (Gaussian mechanism on the
  input embedding) at the SELECTED layers, run the SELECTED attacks (linear ridge /
  non-linear decoder) and the SELECTED attack-independent probes (CLUB MI upper bound,
  V_cap capacity-matched reader), and report bits beside a readable per-secret readout
  (token-id top-1 recovery + reader perplexity).

Promoted from the now-retired validated spikes scripts/spikes/{b2_propagated_dp,b2_lpos_decoder}
into a clean, pluggable module: ATTACKS and PROBES are registries; --layers and --epsilons are
CLI lists, so the same sweep serves any (layers × ε × attacks × probes) cut.

Threat model WEIGHTS-PUB: adversary knows weights + embedding table + published (C, σ) and
synthesizes its own (noised-rep, token) training pairs. The probes never see an attack output.

Conventions (kept consistent with the existing project DP sweeps so results splice together):
  * σ = C·z/ε with z = √(2 ln(1.25/δ)) — add/remove-to-zero adjacency, per-row sensitivity C.
  * recovery = top-1 over the candidate POOL (default 2048), not full vocab.
  * *_sel = recovery − a CLEAN-rep label-shuffle floor (memorization baseline, fixed per layer).
  * probes are computed on ALL captured rows (probe_split=all); attacks score the test split.

GPU: ONE process at a time; run via scripts/run_in_rocm.sh. Output JSON under refine-logs/<surface>/.

Example (the residual cross-layer DP table, campaign-D Task 2):
  scripts/run_in_rocm.sh python3 scripts/evals/dp_leakage_sweep.py \
      --layers 0,5,12,20 --epsilons inf,1024,512,256 --attacks ridge,decoder \
      --probes club,vcap --out refine-logs/dp-decoder-grid/dp_leakage_sweep.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from talens.measures.club import club_mi_upper_bound  # noqa: E402
from talens.measures.vinfo_capacity import v_information_capacity  # noqa: E402
from talens.report import perplexity_from_bits  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ───────────────────────── defense: local DP ─────────────────────────
class LocalDP:
    """Local DP Gaussian mechanism on the input embedding: clip rows to L2 ≤ C, add N(0, σ²I).

    σ=0 is clip-only (≈ clean). Registered as a forward hook on the embedding layer, so the
    noise then propagates through the network to whatever depth is captured. Per-row sensitivity
    is C (add/remove-to-zero adjacency); σ is set by the caller as C·z/ε.
    """

    def __init__(self, C: float, sigma: float):
        self.C, self.sigma = C, sigma

    def __call__(self, mod, inp, out):
        f = out.float()
        n = f.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        f = f * (self.C / n).clamp_max(1.0)
        if self.sigma > 0:
            f = f + self.sigma * torch.randn_like(f)
        return f.to(out.dtype)


@torch.no_grad()
def capture(model, tok, prompts, layers):
    """Per-token residual_post at each requested layer + the token ids (one array per prompt)."""
    per = {L: [] for L in layers}
    ids = []
    for p in prompts:
        i = tok(p, return_tensors="pt").input_ids.to(DEV)
        hs = model(i, output_hidden_states=True, use_cache=False).hidden_states
        for L in layers:
            per[L].append(hs[L + 1][0].float().cpu().numpy())
        ids.append(i[0].cpu().numpy())
    return per, ids


def _stack(per_L, ids):
    Xs, ys = [], []
    for m, t in zip(per_L, ids):
        n = min(m.shape[0], t.shape[0])
        Xs.append(m[:n])
        ys.append(t[:n])
    return np.concatenate(Xs, 0).astype(np.float32), np.concatenate(ys, 0).astype(np.int64)


# ───────────────────────── attacks (rep → embedding → nearest token) ─────────────────────────
def _nearest_token(pred_emb, pool_emb, pool_ids):
    p = pred_emb / np.clip(np.linalg.norm(pred_emb, axis=1, keepdims=True), 1e-9, None)
    e = pool_emb / np.clip(np.linalg.norm(pool_emb, axis=1, keepdims=True), 1e-9, None)
    return pool_ids[(p @ e.T).argmax(1)]


def attack_ridge(Xtr, Etr, Xte, pool_emb, pool_ids, *, alpha=1.0, **_):
    """Linear (ridge) obs→embedding map, then nearest token. The information-inefficient baseline.

    Gram/solve in float64 for stability (d≈2304); prediction back to float32.
    """
    d = Xtr.shape[1]
    A = (Xtr.T @ Xtr).astype(np.float64) + alpha * np.eye(d, dtype=np.float64)
    W = np.linalg.solve(A, (Xtr.T @ Etr).astype(np.float64)).astype(np.float32)
    return _nearest_token(Xte @ W, pool_emb, pool_ids)


def attack_decoder(Xtr, Etr, Xte, pool_emb, pool_ids, *, hidden=384, epochs=200, lr=1e-3, seed=0, **_):
    """Non-linear (MLP) channel-aware decoder obs→embedding (cosine loss), then nearest token.

    Trained on the σ-matched noised reps, so it is noise-aware — the stronger admissible attack.
    """
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(Xtr.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, Etr.shape[1])
    ).to(DEV)
    xt = torch.from_numpy(Xtr).to(DEV)
    yt = torch.from_numpy(Etr).to(DEV)
    yt = yt / yt.norm(dim=1, keepdim=True).clamp_min(1e-9)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    for _ in range(epochs):
        opt.zero_grad()
        p = net(xt)
        p = p / p.norm(dim=1, keepdim=True).clamp_min(1e-9)
        (1.0 - (p * yt).sum(1)).mean().backward()
        opt.step()
    with torch.no_grad():
        pred = net(torch.from_numpy(Xte).to(DEV)).cpu().numpy()
    return _nearest_token(pred, pool_emb, pool_ids)


ATTACKS = {"ridge": attack_ridge, "decoder": attack_decoder}


# ───────────────────────── probes (attack-independent; bits + readout) ─────────────────────────
def probe_club(X, E, y, K, *, club_max_rows=600, **_):
    """CLUB variational MI upper bound I(rep;token), in bits. Continuous MI → no token perplexity."""
    bits = club_mi_upper_bound(X, E, max_rows=club_max_rows, seed=0)["club_mi_bits"]
    return {"bits": None if bits is None else float(bits), "bits_kind": "mi_upper_bound"}


def probe_vcap(X, E, y, K, **_):
    """V_cap capacity-matched predictive V-information (bits) + a readable reader readout.

    Readout: reader_top1_acc, and reader perplexity = 2^(H_cond) where H_cond = H_prior − PVI and
    H_prior is the EMPIRICAL token-label entropy (v_information_capacity's PVI is anchored to the
    empirical prior, not log₂K). Clamped to [1, 2^H_prior] = the effective number of token
    candidates the reader is choosing among.
    """
    r = v_information_capacity(X, y, family="pca_softmax", dim=64, l2=0.1)
    pvi = r.get("v_information_bits")
    acc = r.get("reader_top1_acc")
    ppl = None
    if pvi is not None:
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        h_prior = float(-(p * np.log2(p)).sum())
        h_cond = min(max(h_prior - pvi, 0.0), h_prior)
        ppl = perplexity_from_bits(h_cond)
    return {"bits": pvi, "bits_kind": "capacity_v_info", "reader_top1_acc": acc, "perplexity": ppl}


PROBES = {"club": probe_club, "vcap": probe_vcap}


# ───────────────────────── sweep ─────────────────────────
def _spearman(a, b):
    """Spearman ρ via numpy (rank then Pearson). Dependency-free; ties broken by order
    (fine for the distinct ε-sweep values here). Returns None for degenerate input."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="unsloth/gemma-2-2b")
    ap.add_argument("--corpus", default="corpora/release-gate-512.txt")
    ap.add_argument("--max-prompts", type=int, default=160)
    ap.add_argument("--layers", default="0,5,12,20", help="comma list of layer indices")
    ap.add_argument("--epsilons", default="inf,1024,512,256", help="comma list; 'inf' = clip-only")
    ap.add_argument("--attacks", default="ridge,decoder", help=f"subset of {sorted(ATTACKS)}")
    ap.add_argument("--probes", default="club,vcap", help=f"subset of {sorted(PROBES)}")
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--clip-percentile", type=float, default=99.9)
    ap.add_argument("--pool-size", type=int, default=2048)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--club-max-rows", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20260621)
    ap.add_argument("--out", default="refine-logs/dp-decoder-grid/dp_leakage_sweep.json")
    args = ap.parse_args()

    layers = [int(s) for s in args.layers.split(",") if s.strip()]
    eps_list = [math.inf if s.strip().lower().startswith("inf") else float(s)
                for s in args.epsilons.split(",") if s.strip()]
    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    probes = [p.strip() for p in args.probes.split(",") if p.strip()]
    if not attacks or not probes or not layers or not eps_list:
        ap.error("need at least one each of --layers, --epsilons, --attacks, --probes")
    for a in attacks:
        if a not in ATTACKS:
            ap.error(f"unknown attack {a!r}; choose from {sorted(ATTACKS)}")
    for p in probes:
        if p not in PROBES:
            ap.error(f"unknown probe {p!r}; choose from {sorted(PROBES)}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = [l.strip() for l in Path(args.corpus).read_text().splitlines() if l.strip()][: args.max_prompts]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map=DEV
    ).eval()
    table = model.get_input_embeddings().weight.detach().float().cpu().numpy().astype(np.float32)
    vocab = table.shape[0]

    # clip C from runtime embedding norms (so clip-only ≈ clean; the curve is noise-driven)
    cal = []
    h = model.model.embed_tokens.register_forward_hook(
        lambda m, i, o: cal.append(o.float().norm(dim=-1).flatten().cpu())
    )
    try:
        with torch.no_grad():
            for p in prompts[:48]:
                model(tok(p, return_tensors="pt").input_ids.to(DEV), use_cache=False)
    finally:
        h.remove()
    C = float(np.percentile(torch.cat(cal).numpy(), args.clip_percentile))
    z = math.sqrt(2 * math.log(1.25 / args.delta))
    rng = np.random.default_rng(args.seed)
    print(f"[dp-sweep] C={C:.3f} layers={layers} eps={eps_list} attacks={attacks} probes={probes} "
          f"prompts={len(prompts)} dev={DEV}", flush=True)

    # clean capture once: defines the vocab-disjoint split + candidate pool + per-attack shuffle floor
    perc, idc = capture(model, tok, prompts, layers)
    split = {}
    for L in layers:
        X0, y = _stack(perc[L], idc)
        distinct = rng.permutation(np.unique(y))
        ntr = int(0.7 * distinct.size)
        tr_ids, te_ids = set(distinct[:ntr].tolist()), set(distinct[ntr:].tolist())
        tr = np.array([i for i, t in enumerate(y) if t in tr_ids])
        te = np.array([i for i, t in enumerate(y) if t in te_ids])
        true_pool = np.unique(y[te])
        if true_pool.size > args.pool_size:
            print(f"[dp-sweep] WARN L{L}: {true_pool.size} test tokens > pool-size {args.pool_size}; "
                  f"pool = all test tokens", flush=True)
        avail = np.setdiff1d(np.arange(vocab, dtype=np.int64), true_pool)
        fill = rng.choice(avail, size=max(0, args.pool_size - true_pool.size), replace=False)
        pool = np.concatenate([true_pool, fill.astype(np.int64)])  # true_pool ⊆ pool, disjoint fill
        emb_y = table[y]
        permsh = rng.permutation(tr.size)  # label-shuffle control → CLEAN-rep recovery floor per attack
        floor = {}
        for a in attacks:
            yhat = ATTACKS[a](X0[tr], emb_y[tr][permsh], X0[te], table[pool], pool,
                              hidden=args.hidden, epochs=args.epochs, seed=args.seed)
            floor[a] = float((yhat == y[te]).mean())
        split[L] = dict(y=y, tr=tr, te=te, pool=pool, emb_y=emb_y, floor=floor,
                        K=int(np.unique(y).size))

    records = []
    for ei, eps in enumerate(eps_list):
        sigma = 0.0 if math.isinf(eps) else C * z / eps
        torch.manual_seed(args.seed + 1009 * ei)  # distinct per sweep point (robust to inf / decimal ε)
        hk = model.model.embed_tokens.register_forward_hook(LocalDP(C, sigma))
        try:
            per, ids_chk = capture(model, tok, prompts, layers)
        finally:
            hk.remove()
        assert all(np.array_equal(a, b) for a, b in zip(ids_chk, idc)), "token ids drifted under noise"
        for L in layers:
            X, _ = _stack(per[L], idc)
            s = split[L]
            y, tr, te, pool, emb_y, floor, K = s["y"], s["tr"], s["te"], s["pool"], s["emb_y"], s["floor"], s["K"]
            pe = table[pool]
            rec = {"epsilon": (None if math.isinf(eps) else eps), "layer": L, "sigma": sigma}
            for a in attacks:
                yhat = ATTACKS[a](X[tr], emb_y[tr], X[te], pe, pool,
                                  hidden=args.hidden, epochs=args.epochs, seed=args.seed)
                top1 = float((yhat == y[te]).mean())
                rec[a] = top1
                rec[f"{a}_sel"] = top1 - floor[a]
            for p in probes:
                out = PROBES[p](X, emb_y, y, K, club_max_rows=args.club_max_rows)
                for k, v in out.items():
                    rec[f"{p}_{k}"] = v
            records.append(rec)
            es = "inf" if math.isinf(eps) else f"{eps:g}"
            atxt = " ".join(f"{a}={rec[a]:.3f}" for a in attacks)
            ptxt = " ".join(
                f"{p}={rec.get(p + '_bits')}" + (f"(ppl {rec[p + '_perplexity']:.1f})" if rec.get(p + "_perplexity") else "")
                for p in probes
            )
            print(f"[dp-sweep] ε={es:>5} L{L:>2} | {atxt} | {ptxt}", flush=True)

    # per (layer, attack, probe): does the attack's selectivity track the probe bits across ε?
    corr = {}
    for L in layers:
        r = [x for x in records if x["layer"] == L]
        corr[f"L{L}"] = {}
        for a in attacks:
            for p in probes:
                bk = f"{p}_bits"
                pairs = [(x[f"{a}_sel"], x[bk]) for x in r
                         if x.get(bk) is not None and np.isfinite(x[bk])]
                if len(pairs) >= 2:
                    xs, ys_ = zip(*pairs)
                    corr[f"L{L}"][f"{a}_sel_vs_{p}"] = _spearman(xs, ys_)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "model": args.model, "corpus": args.corpus, "n_prompts": len(prompts),
        "defense": "local_dp", "sigma_convention": "sigma = C*z/eps (sensitivity C, add/remove-to-zero adjacency)",
        "clip_C": C, "delta": args.delta, "pool_size": args.pool_size, "seed": args.seed,
        "layers": layers, "epsilons": [None if math.isinf(e) else e for e in eps_list],
        "attacks": attacks, "probes": probes,
        "readout_note": "recovery = token-id top-1 over the candidate pool; *_sel = recovery minus a "
                        "CLEAN-rep label-shuffle floor (per layer); vcap_perplexity = reader effective "
                        "token candidates 2^(H_prior_empirical - PVI), clamped [1, 2^H_prior]; probes "
                        "computed on all captured rows (probe_split=all).",
        "correlation": corr, "records": records,
    }, indent=2))
    print(f"[dp-sweep] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
