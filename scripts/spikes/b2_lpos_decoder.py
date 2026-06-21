#!/usr/bin/env python3
"""B2-L>0 — channel-aware nonlinear decoder vs ridge under at-layer noise (GPU-free).

Uses the cached clean resid_post (L5/12/20, gemma-2-2b) and adds at-layer additive
noise in-memory (Gaussian = DP-like, or Laplace = Shredder profile) at varying
scale. Tests whether an information-efficient attack beats ridge at DEPTH (where
the token is entangled and NN-to-table no longer applies) and whether its recovery
re-correlates with the MI probes (capacity-PVI, CLUB).

Attacks (identical vocab-disjoint test split + candidate pool, A4):
  * ridge            — linear obs->embedding map + cosine (noise-naive baseline)
  * decoder_naive    — nonlinear MLP obs->embedding trained on CLEAN acts (B4 ctrl)
  * decoder_chanaware— same MLP trained on NOISED acts at the deployment sigma
Uplift(decoder_chanaware − ridge) = total; (− decoder_naive) isolates channel-awareness;
(decoder_naive − ridge) isolates nonlinearity. Proof-gated by T1 (nonlinear-MMSE > linear
iff E[emb|Y] non-affine — generic at depth; channel-awareness = training on the true channel).

CPU torch. Noise profile via --profile {gauss,laplace}.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from talens.measures.club import club_mi_upper_bound  # noqa: E402
from talens.measures.vinfo_capacity import v_information_capacity  # noqa: E402

CACHE = "results/capture_cache/capture-4ca8a33e16bfbec9.pt"
EMBED = "results/capture_cache/embed-b0c6566474cadb27.pt"


def add_noise(X, sigma, profile, rng):
    if sigma <= 0:
        return X.copy()
    if profile == "gauss":
        return X + sigma * rng.standard_normal(X.shape).astype(np.float32)
    b = sigma / np.sqrt(2.0)  # Laplace scale matched to std=sigma
    return X + rng.laplace(0.0, b, X.shape).astype(np.float32)


DEV = "cuda" if torch.cuda.is_available() else "cpu"


def train_decoder(Xtr, Ytr, *, hidden=512, epochs=120, lr=1e-3, seed=0):
    """MLP X->embedding, cosine loss. GPU when available (ROCm presents as cuda)."""
    torch.manual_seed(seed)
    d_in, d_out = Xtr.shape[1], Ytr.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(d_in, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, d_out)).to(DEV)
    xt = torch.from_numpy(Xtr).to(DEV); yt = torch.from_numpy(Ytr).to(DEV)
    yt = yt / yt.norm(dim=1, keepdim=True).clamp_min(1e-9)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    for _ in range(epochs):
        opt.zero_grad()
        p = net(xt); p = p / p.norm(dim=1, keepdim=True).clamp_min(1e-9)
        loss = (1.0 - (p * yt).sum(1)).mean()
        loss.backward(); opt.step()
    return net


def decode_match(net, Xte, pool_emb, pool_ids):
    with torch.no_grad():
        p = net(torch.from_numpy(Xte).to(DEV)).cpu().numpy()
    p = p / np.clip(np.linalg.norm(p, axis=1, keepdims=True), 1e-9, None)
    pe = pool_emb / np.clip(np.linalg.norm(pool_emb, axis=1, keepdims=True), 1e-9, None)
    return pool_ids[(p @ pe.T).argmax(1)]


def ridge_match(Xtr, Ytr, Xte, pool_emb, pool_ids, alpha=1.0):
    d = Xtr.shape[1]
    W = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(d), Xtr.T @ Ytr)
    p = Xte @ W; p = p / np.clip(np.linalg.norm(p, axis=1, keepdims=True), 1e-9, None)
    pe = pool_emb / np.clip(np.linalg.norm(pool_emb, axis=1, keepdims=True), 1e-9, None)
    return pool_ids[(p @ pe.T).argmax(1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", default="5,12,20")
    ap.add_argument("--levels", default="0,0.25,0.5,1.0,2.0,4.0", help="noise sigma / act-RMS")
    ap.add_argument("--profile", default="gauss", choices=["gauss", "laplace"])
    ap.add_argument("--pool-size", type=int, default=2048)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--club-max-rows", type=int, default=700)
    ap.add_argument("--seed", type=int, default=20260621)
    ap.add_argument("--out", default="results/b2_lpos_decoder.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cap = torch.load(CACHE, map_location="cpu", weights_only=False)
    table = torch.load(EMBED, map_location="cpu", weights_only=False).float().numpy().astype(np.float32)
    vocab = table.shape[0]
    ids_per = [np.asarray(t, dtype=np.int64) for t in cap["prompt_token_ids"]]
    layers = [int(s) for s in args.layers.split(",") if s.strip()]
    levels = [float(s) for s in args.levels.split(",") if s.strip()]

    def stack(L):
        mats = cap["operands"][("resid_post", L)]
        Xs, ys = [], []
        for m, tid in zip(mats, ids_per):
            m = np.asarray(m, dtype=np.float32); n = min(m.shape[0], tid.shape[0])
            Xs.append(m[:n]); ys.append(tid[:n])
        return np.concatenate(Xs, 0), np.concatenate(ys, 0)

    records = []
    for L in layers:
        X0, y = stack(L)
        rms = float(np.sqrt((X0 ** 2).mean()))
        distinct = rng.permutation(np.unique(y)); ntr = int(0.7 * distinct.size)
        tr_ids = set(distinct[:ntr].tolist()); te_ids = set(distinct[ntr:].tolist())
        tr = np.array([i for i, t in enumerate(y) if t in tr_ids])
        te = np.array([i for i, t in enumerate(y) if t in te_ids])
        true_pool = np.unique(y[te])
        avail = np.setdiff1d(np.arange(vocab, dtype=np.int64), true_pool)
        fill = rng.choice(avail, size=max(0, args.pool_size - true_pool.size), replace=False)
        pool = np.concatenate([true_pool, fill.astype(np.int64)])
        pool_emb = table[pool]; emb_y = table[y]
        # noise-naive decoder trained ONCE on clean acts
        dec_clean = train_decoder(X0[tr], emb_y[tr], hidden=args.hidden, epochs=args.epochs, seed=args.seed)
        # SHUFFLE CONTROL once per layer (Hewitt-Liang): train on permuted (X,token)
        # pairs → no generalizable signal → recovery floor. The floor is noise-
        # INDEPENDENT under vocab-disjoint (shuffled labels carry no signal at any σ),
        # so compute it once at clean and reuse. Selectivity = real − floor rules out
        # memorization; only the generalizing margin counts.
        permsh = rng.permutation(tr.size)
        ridge_sh = float((ridge_match(X0[tr], emb_y[tr][permsh], X0[te], pool_emb, pool) == y[te]).mean())
        dec_sh = train_decoder(X0[tr], emb_y[tr][permsh], hidden=args.hidden, epochs=args.epochs, seed=args.seed)
        ca_sh = float((decode_match(dec_sh, X0[te], pool_emb, pool) == y[te]).mean())
        chance = 1.0 / pool.size
        for c in levels:
            sigma = c * rms
            X = add_noise(X0, sigma, args.profile, np.random.default_rng(args.seed + int(1000 * c) + L))
            ridge_pred = ridge_match(X[tr], emb_y[tr], X[te], pool_emb, pool)
            ridge_t = float((ridge_pred == y[te]).mean())
            naive_pred = decode_match(dec_clean, X[te], pool_emb, pool)
            naive_t = float((naive_pred == y[te]).mean())
            dec_ca = train_decoder(X[tr], emb_y[tr], hidden=args.hidden, epochs=args.epochs, seed=args.seed)
            ca_pred = decode_match(dec_ca, X[te], pool_emb, pool)
            ca_t = float((ca_pred == y[te]).mean())
            capv = v_information_capacity(X, y, family="pca_softmax", dim=64, l2=0.1)["reader_top1_acc"]
            club = club_mi_upper_bound(X, emb_y, max_rows=args.club_max_rows, seed=0)["club_mi_bits"]
            rec = {"layer": L, "level": c, "sigma": sigma, "noise_to_signal": c, "chance": chance,
                   "ridge": ridge_t, "decoder_naive": naive_t, "decoder_chanaware": ca_t,
                   "ridge_shuffle": ridge_sh, "ca_shuffle": ca_sh,
                   "ridge_selectivity": ridge_t - ridge_sh, "ca_selectivity": ca_t - ca_sh,
                   "uplift_selectivity": (ca_t - ca_sh) - (ridge_t - ridge_sh),
                   "chanaware_gain": ca_t - naive_t, "cap_pvi_acc": capv, "club_bits": club}
            records.append(rec)
            print(f"[Lpos] L{L:>2} c={c:<4} | ridge={ridge_t:.3f}(sel{ridge_t-ridge_sh:+.3f}) "
                  f"dec_CA={ca_t:.3f}(sel{ca_t-ca_sh:+.3f}) sh[r={ridge_sh:.3f},ca={ca_sh:.3f}] "
                  f"upliftSel{rec['uplift_selectivity']:+.3f} | capPVI={capv:.3f} club={club:.0f}", flush=True)

    def sp(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return 0.0 if np.std(a) < 1e-9 or np.std(b) < 1e-9 else float(stats.spearmanr(a, b).statistic)
    # per-layer re-correlation: recovery vs MI probe over the noise sweep
    # re-correlation uses SELECTIVITY (generalizing margin), not raw recovery
    corr = {}
    for L in layers:
        r = [x for x in records if x["layer"] == L]
        corr[f"L{L}"] = {
            "ridgeSel_vs_capPVI": sp([x["ridge_selectivity"] for x in r], [x["cap_pvi_acc"] for x in r]),
            "decCASel_vs_capPVI": sp([x["ca_selectivity"] for x in r], [x["cap_pvi_acc"] for x in r]),
            "ridgeSel_vs_club": sp([x["ridge_selectivity"] for x in r], [x["club_bits"] for x in r]),
            "decCASel_vs_club": sp([x["ca_selectivity"] for x in r], [x["club_bits"] for x in r]),
        }
    print("\n[Lpos] re-correlation (Spearman SELECTIVITY↔MI-probe over noise sweep, per layer):")
    for L in layers:
        c = corr[f"L{L}"]
        print(f"   L{L}: ridgeSel↔capPVI={c['ridgeSel_vs_capPVI']:+.2f} decCASel↔capPVI={c['decCASel_vs_capPVI']:+.2f} | "
              f"ridgeSel↔CLUB={c['ridgeSel_vs_club']:+.2f} decCASel↔CLUB={c['decCASel_vs_club']:+.2f}")
    out = {"profile": args.profile, "layers": layers, "levels": levels,
           "recorrelation": corr, "records": records}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[Lpos] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
