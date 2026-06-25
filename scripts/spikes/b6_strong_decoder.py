#!/usr/bin/env python3
"""B6 — stronger decoder attacks vs ridge/MLP under PROPAGATED input-DP @L20.

Compares, on a propagated-input-DP capture (embedding-DP hook → forward → resid L20):
  * ridge            — linear baseline
  * mlp_oneshot      — 1-hidden MLP (R004c config)
  * deep_oneshot     — 2-hidden wide MLP, long-trained (CAPACITY CONTROL)
  * iterative_T{1,2,3} — Vec2Text-style: ê0=mlp(Y); corrector c([Y‖ê_t])→ê_{t+1}, T rounds
All map obs→embedding then cosine-retrieve over the shared vocab-disjoint pool; selectivity
via shuffle control. Novelty isolation: iterative vs deep (capacity), and T>1 vs T=1.
WEIGHTS-PUB (corrector trained on attacker-generated triples). Single GPU process, ≤20min.
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np, torch
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from b2_propagated_dp import InputDPCover, capture  # noqa: E402
from b2_lpos_decoder import train_decoder, decode_match, ridge_match, DEV  # noqa: E402
sys.path.insert(0, str(HERE.parents[1] / "src"))
from talens.probes.club import club_mi_upper_bound  # noqa: E402
from talens.probes.vinfo_capacity import v_information_capacity  # noqa: E402


def train_deep(Xtr, Ytr, *, hidden=1024, epochs=600, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    di, do = Xtr.shape[1], Ytr.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(di, hidden), torch.nn.ReLU(),
                              torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
                              torch.nn.Linear(hidden, do)).to(DEV)
    xt = torch.from_numpy(Xtr).to(DEV); yt = torch.from_numpy(Ytr).to(DEV)
    yt = yt / yt.norm(dim=1, keepdim=True).clamp_min(1e-9)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    for _ in range(epochs):
        opt.zero_grad(); p = net(xt); p = p / p.norm(dim=1, keepdim=True).clamp_min(1e-9)
        (1.0 - (p * yt).sum(1)).mean().backward(); opt.step()
    return net


def train_corrector(Xtr, e0_tr, Ytr_emb, *, hidden=512, epochs=300, lr=1e-3, seed=0):
    """c([X‖ê]) → emb. Trained on ê0 = base decoder output (one round of data)."""
    torch.manual_seed(seed)
    di = Xtr.shape[1] + e0_tr.shape[1]; do = Ytr_emb.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(di, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, do)).to(DEV)
    inp = torch.from_numpy(np.concatenate([Xtr, e0_tr], axis=1)).to(DEV)
    yt = torch.from_numpy(Ytr_emb).to(DEV); yt = yt / yt.norm(dim=1, keepdim=True).clamp_min(1e-9)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    for _ in range(epochs):
        opt.zero_grad(); p = net(inp); p = p / p.norm(dim=1, keepdim=True).clamp_min(1e-9)
        (1.0 - (p * yt).sum(1)).mean().backward(); opt.step()
    return net


@torch.no_grad()
def corrector_iterate(corr, X, e_init, T):
    x = torch.from_numpy(X).to(DEV); e = torch.from_numpy(e_init).to(DEV)
    for _ in range(T):
        e = corr(torch.cat([x, e], dim=1)); e = e / e.norm(dim=1, keepdim=True).clamp_min(1e-9)
    return e.cpu().numpy()


@torch.no_grad()
def mlp_embed(net, X):
    p = net(torch.from_numpy(X).to(DEV)).cpu().numpy()
    return p / np.clip(np.linalg.norm(p, axis=1, keepdims=True), 1e-9, None)


def retrieve(emb_pred, pool_emb, pool_ids):
    pe = pool_emb / np.clip(np.linalg.norm(pool_emb, axis=1, keepdims=True), 1e-9, None)
    return pool_ids[(emb_pred @ pe.T).argmax(1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="unsloth/gemma-2-2b")
    ap.add_argument("--corpus", default="corpora/release-gate-512.txt")
    ap.add_argument("--max-prompts", type=int, default=160)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--epsilons", default="inf,1024,768,512,384,256")
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--clip-percentile", type=float, default=99.9)
    ap.add_argument("--pool-size", type=int, default=2048)
    ap.add_argument("--club-max-rows", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20260621)
    ap.add_argument("--out", default="results/b6_strong_decoder.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = DEV
    L = args.layer
    eps_list = [math.inf if s.strip().lower().startswith("inf") else float(s) for s in args.epsilons.split(",") if s.strip()]
    prompts = [l.strip() for l in Path(args.corpus).read_text().splitlines() if l.strip()][: args.max_prompts]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map=device).eval()
    table = model.get_input_embeddings().weight.detach().float().cpu().numpy().astype(np.float32)
    vocab = table.shape[0]
    cal = []
    h = model.model.embed_tokens.register_forward_hook(lambda m, i, o: cal.append(o.float().norm(dim=-1).flatten().cpu()))
    with torch.no_grad():
        for p in prompts[:48]: model(tok(p, return_tensors="pt").input_ids.to(device), use_cache=False)
    h.remove()
    C = float(np.percentile(torch.cat(cal).numpy(), args.clip_percentile)); z = math.sqrt(2 * math.log(1.25 / args.delta))
    rng = np.random.default_rng(args.seed)
    print(f"[b6] C={C:.3f} L={L} eps={eps_list} prompts={len(prompts)} dev={device}", flush=True)

    def stack(per_L, ids):
        Xs, ys = [], []
        for m, t in zip(per_L, ids):
            n = min(m.shape[0], t.shape[0]); Xs.append(m[:n]); ys.append(t[:n])
        return np.concatenate(Xs, 0), np.concatenate(ys, 0).astype(np.int64)

    perc, idc = capture(model, tok, prompts, [L], device)
    X0, y = stack(perc[L], idc)
    distinct = rng.permutation(np.unique(y)); ntr = int(0.7 * distinct.size)
    tr_ids = set(distinct[:ntr].tolist()); te_ids = set(distinct[ntr:].tolist())
    tr = np.array([i for i, t in enumerate(y) if t in tr_ids]); te = np.array([i for i, t in enumerate(y) if t in te_ids])
    true_pool = np.unique(y[te]); avail = np.setdiff1d(np.arange(vocab, dtype=np.int64), true_pool)
    fill = rng.choice(avail, size=max(0, args.pool_size - true_pool.size), replace=False)
    pool = np.concatenate([true_pool, fill.astype(np.int64)]); pe = table[pool]; emb_y = table[y]
    # shuffle floors (clean, once)
    permsh = rng.permutation(tr.size)
    rsh = float((ridge_match(X0[tr], emb_y[tr][permsh], X0[te], pe, pool) == y[te]).mean())
    msh_net = train_decoder(X0[tr], emb_y[tr][permsh], hidden=384, epochs=250, seed=args.seed)
    msh = float((retrieve(mlp_embed(msh_net, X0[te]), pe, pool) == y[te]).mean())
    floor = {"ridge": rsh, "mlp": msh}
    print(f"[b6] shuffle floors ridge={rsh:.3f} mlp={msh:.3f} chance={1/pool.size:.4f}", flush=True)

    records = []
    for eps in eps_list:
        sigma = 0.0 if math.isinf(eps) else C * z / eps
        torch.manual_seed(args.seed + (0 if math.isinf(eps) else int(eps)))
        hk = model.model.embed_tokens.register_forward_hook(InputDPCover(C, sigma)); per, _ = capture(model, tok, prompts, [L], device); hk.remove()
        X, _ = stack(per[L], idc)
        att = {}
        att["ridge"] = float((ridge_match(X[tr], emb_y[tr], X[te], pe, pool) == y[te]).mean())
        mlp = train_decoder(X[tr], emb_y[tr], hidden=384, epochs=250, seed=args.seed)
        att["mlp_oneshot"] = float((retrieve(mlp_embed(mlp, X[te]), pe, pool) == y[te]).mean())
        deep = train_deep(X[tr], emb_y[tr], hidden=1024, epochs=600, seed=args.seed)
        att["deep_oneshot"] = float((retrieve(mlp_embed(deep, X[te]), pe, pool) == y[te]).mean())
        e0_tr = mlp_embed(mlp, X[tr]); corr = train_corrector(X[tr], e0_tr, emb_y[tr], hidden=512, epochs=300, seed=args.seed)
        e0_te = mlp_embed(mlp, X[te])
        for T in (1, 2, 3):
            ei = corrector_iterate(corr, X[te], e0_te, T)
            att[f"iter_T{T}"] = float((retrieve(ei, pe, pool) == y[te]).mean())
        capv = v_information_capacity(X, y, family="pca_softmax", dim=64, l2=0.1)["reader_top1_acc"]
        club = club_mi_upper_bound(X, emb_y, max_rows=args.club_max_rows, seed=0)["club_mi_bits"]
        rec = {"epsilon": (None if math.isinf(eps) else eps), "sigma": sigma, **att,
               "ridge_sel": att["ridge"] - floor["ridge"],
               "mlp_sel": att["mlp_oneshot"] - floor["mlp"], "deep_sel": att["deep_oneshot"] - floor["mlp"],
               "iterT3_sel": att["iter_T3"] - floor["mlp"], "cap_pvi_acc": capv, "club_bits": club}
        records.append(rec)
        es = "inf" if math.isinf(eps) else f"{eps:g}"
        print(f"[b6] ε={es:>5} | ridge={att['ridge']:.3f} mlp={att['mlp_oneshot']:.3f} deep={att['deep_oneshot']:.3f} "
              f"iterT1={att['iter_T1']:.3f} T2={att['iter_T2']:.3f} T3={att['iter_T3']:.3f} | capPVI={capv:.3f} club={club:.0f}", flush=True)

    def sp(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return 0.0 if np.std(a) < 1e-9 or np.std(b) < 1e-9 else float(stats.spearmanr(a, b).statistic)
    R = lambda k: [r[k] for r in records]
    corr_stats = {"ridgeSel_vs_capPVI": sp(R("ridge_sel"), R("cap_pvi_acc")),
                  "deepSel_vs_capPVI": sp(R("deep_sel"), R("cap_pvi_acc")),
                  "iterT3Sel_vs_capPVI": sp(R("iterT3_sel"), R("cap_pvi_acc")),
                  "ridgeSel_vs_club": sp(R("ridge_sel"), R("club_bits")),
                  "iterT3Sel_vs_club": sp(R("iterT3_sel"), R("club_bits"))}
    print("\n[b6] re-correlation (Spearman selectivity↔MI over ε):")
    for k, v in corr_stats.items(): print(f"   {k} = {v:+.2f}")
    print("[b6] mean uplift-sel vs ridge:  deep="
          f"{np.mean([r['deep_sel']-r['ridge_sel'] for r in records]):+.3f}  "
          f"iterT3={np.mean([r['iterT3_sel']-r['ridge_sel'] for r in records]):+.3f}")
    print("[b6] mean iter_T3 − deep (iteration vs capacity) = "
          f"{np.mean([r['iter_T3']-r['deep_oneshot'] for r in records]):+.3f}; "
          f"T3 − T1 = {np.mean([r['iter_T3']-r['iter_T1'] for r in records]):+.3f}")
    Path(args.out).write_text(json.dumps({"layer": L, "epsilons": [None if math.isinf(e) else e for e in eps_list],
                              "floors": floor, "recorrelation": corr_stats, "records": records}, indent=2))
    print(f"[b6] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
