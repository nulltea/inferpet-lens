#!/usr/bin/env python3
"""B6c — forward-model-in-loop Vec2Text attack under PROPAGATED input-DP @L20.

The faithful iterative attack the embedding-space corrector (B6) could not be: it
RE-EMBEDS each candidate token through the actual model (clip-only reference,
σ=0) and matches the resulting clean resid to the observed noised resid Y_obs.
The candidate whose forward f_L^clip(prefix+cand) is closest to Y_obs[pos] wins —
this uses the model in the loop (WEIGHTS-PUB: attacker has the weights).

To bound cost, candidates per position are SEEDED by the one-shot decoder's top-k
(re-ranked by the forward model). Teacher-forced (true) prefix → an oracle-prefix
per-position attack that isolates the forward model's per-token discriminability
(the deployable autoregressive variant uses the recovered prefix; noted as the
extension). Compares vs ridge and the one-shot decoder; reports uplift + MI
re-correlation. Single GPU process; budget ≤20 min (few ε, capped positions).

Hypothesis: at LOW noise Y_obs ≈ f_L^clip(true) exactly, so FMV recovers near-perfectly
— closing the gap where ridge/decoder capped — and tracks the MI probes.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from b2_propagated_dp import InputDPCover, capture  # noqa: E402
from b2_lpos_decoder import train_decoder, ridge_match, DEV  # noqa: E402
sys.path.insert(0, str(HERE.parents[1] / "src"))
from talens.measures.vinfo_capacity import v_information_capacity  # noqa: E402
from scipy import stats  # noqa: E402


@torch.no_grad()
def mlp_topk(net, X, pool_emb, pool_ids, k):
    p = net(torch.from_numpy(X).to(DEV)); p = p / p.norm(dim=1, keepdim=True).clamp_min(1e-9)
    pe = torch.from_numpy(pool_emb).to(DEV); pe = pe / pe.norm(dim=1, keepdim=True).clamp_min(1e-9)
    sims = p @ pe.T                                          # (n, P)
    topk = sims.topk(k, dim=1).indices.cpu().numpy()         # (n, k) pool positions
    top1 = pool_ids[sims.argmax(1).cpu().numpy()]
    return pool_ids[topk], top1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="unsloth/gemma-2-2b")
    ap.add_argument("--corpus", default="corpora/release-gate-512.txt")
    ap.add_argument("--max-prompts", type=int, default=120)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--epsilons", default="inf,512,256")
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--clip-percentile", type=float, default=99.9)
    ap.add_argument("--pool-size", type=int, default=2048)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--max-test-pos", type=int, default=400, help="cap total scored test positions (budget)")
    ap.add_argument("--seed", type=int, default=20260621)
    ap.add_argument("--out", default="results/b6c_forward_model.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = DEV; L = args.layer
    eps_list = [math.inf if s.strip().lower().startswith("inf") else float(s) for s in args.epsilons.split(",") if s.strip()]
    prompts = [l.strip() for l in Path(args.corpus).read_text().splitlines() if l.strip()][: args.max_prompts]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map=device).eval()
    table = model.get_input_embeddings().weight.detach().float().cpu().numpy().astype(np.float32)
    vocab = table.shape[0]
    # token ids per prompt (cache once)
    ids_per = [tok(p, return_tensors="pt").input_ids[0].numpy().astype(np.int64) for p in prompts]
    cal = []
    h = model.model.embed_tokens.register_forward_hook(lambda m, i, o: cal.append(o.float().norm(dim=-1).flatten().cpu()))
    with torch.no_grad():
        for p in prompts[:48]: model(tok(p, return_tensors="pt").input_ids.to(device), use_cache=False)
    h.remove()
    C = float(np.percentile(torch.cat(cal).numpy(), args.clip_percentile)); z = math.sqrt(2 * math.log(1.25 / args.delta))
    rng = np.random.default_rng(args.seed)
    print(f"[b6c] C={C:.3f} L={L} eps={eps_list} prompts={len(prompts)} k={args.topk}", flush=True)

    # clip-only forward_fn: resid at L, last position, for a batch of token sequences
    clip_hook = None
    @torch.no_grad()
    def fwd_resid_last(seqs):  # seqs: (B, Lseq) np int64 → (B, d) np float32
        t = torch.from_numpy(seqs).to(device)
        hs = model(t, output_hidden_states=True, use_cache=False).hidden_states
        return hs[L + 1][:, -1, :].float().cpu().numpy()

    def stack(per_L):
        Xs, ys = [], []
        for m, t in zip(per_L, ids_per):
            n = min(m.shape[0], t.shape[0]); Xs.append(m[:n]); ys.append(t[:n])
        return np.concatenate(Xs, 0), np.concatenate(ys, 0).astype(np.int64)

    # split by token id (vocab-disjoint), shared across ε
    _, y_all = stack(capture(model, tok, prompts, [L], device)[0][L])
    distinct = rng.permutation(np.unique(y_all)); ntr = int(0.7 * distinct.size)
    tr_ids = set(distinct[:ntr].tolist()); te_ids = set(distinct[ntr:].tolist())
    true_pool = np.array(sorted(te_ids), dtype=np.int64)
    avail = np.setdiff1d(np.arange(vocab, dtype=np.int64), true_pool)
    fill = rng.choice(avail, size=max(0, args.pool_size - true_pool.size), replace=False)
    pool = np.concatenate([true_pool, fill.astype(np.int64)]); pool_emb = table[pool]; emb_table = table

    records = []
    for eps in eps_list:
        sigma = 0.0 if math.isinf(eps) else C * z / eps
        torch.manual_seed(args.seed + (0 if math.isinf(eps) else int(eps)))
        hk = model.model.embed_tokens.register_forward_hook(InputDPCover(C, sigma))
        per, _ = capture(model, tok, prompts, [L], device); hk.remove()
        per_mats = per[L]
        X, y = stack(per_mats)
        tr = np.array([i for i, t in enumerate(y) if t in tr_ids]); te = np.array([i for i, t in enumerate(y) if t in te_ids])
        emb_y = emb_table[y]
        # ridge + one-shot decoder (channel-aware) baselines on test rows
        ridge_t = float((ridge_match(X[tr], emb_y[tr], X[te], pool_emb, pool) == y[te]).mean())
        dec = train_decoder(X[tr], emb_y[tr], hidden=384, epochs=250, seed=args.seed)
        dtopk, dtop1 = mlp_topk(dec, X[te], pool_emb, pool, args.topk)
        dec_t = float((dtop1 == y[te]).mean())

        # FMV: teacher-forced per-position forward-model re-rank of decoder top-k
        # map global test-row index → (prompt, pos)
        pos_index = []  # (prompt, pos, global_te_order)
        gi = 0
        for pi, tids in enumerate(ids_per):
            n = min(per_mats[pi].shape[0], tids.shape[0])
            for pos in range(n):
                if tids[pos] in te_ids:
                    pos_index.append((pi, pos));
        # align: te is in stack() order = same enumeration → te[order] corresponds to pos_index[order]
        n_score = min(len(pos_index), args.max_test_pos)
        sel = rng.choice(len(pos_index), size=n_score, replace=False) if len(pos_index) > n_score else np.arange(len(pos_index))
        t0 = time.time(); fmv_correct = 0; clip_hook = model.model.embed_tokens.register_forward_hook(InputDPCover(C, 0.0))
        for j in sel:
            pi, pos = pos_index[j]
            cands = dtopk[j]                                   # (k,) candidate token ids (decoder top-k)
            prefix = ids_per[pi][:pos]
            seqs = np.tile(prefix, (len(cands), 1)) if pos > 0 else np.empty((len(cands), 0), dtype=np.int64)
            seqs = np.concatenate([seqs, cands[:, None]], axis=1).astype(np.int64)
            resid = fwd_resid_last(seqs)                       # (k, d) clean (clip-only) resid at last pos
            yobs = per_mats[pi][pos]                            # (d,) observed noised resid
            best = cands[int(np.argmin(((resid - yobs[None, :]) ** 2).sum(1)))]
            fmv_correct += int(best == ids_per[pi][pos])
        clip_hook.remove()
        fmv_t = fmv_correct / max(1, n_score)
        capv = v_information_capacity(X, y, family="pca_softmax", dim=64, l2=0.1)["reader_top1_acc"]
        rec = {"epsilon": (None if math.isinf(eps) else eps), "sigma": sigma, "ridge": ridge_t,
               "decoder_top1": dec_t, "fmv": fmv_t, "uplift_fmv_vs_ridge": fmv_t - ridge_t,
               "uplift_fmv_vs_dec": fmv_t - dec_t, "cap_pvi_acc": capv,
               "n_scored": int(n_score), "fmv_secs": round(time.time() - t0, 1)}
        records.append(rec)
        es = "inf" if math.isinf(eps) else f"{eps:g}"
        print(f"[b6c] ε={es:>5} | ridge={ridge_t:.3f} dec={dec_t:.3f} FMV={fmv_t:.3f} "
              f"(vsRidge{rec['uplift_fmv_vs_ridge']:+.3f} vsDec{rec['uplift_fmv_vs_dec']:+.3f}) "
              f"capPVI={capv:.3f} [{n_score} pos, {rec['fmv_secs']}s]", flush=True)

    def sp(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return 0.0 if np.std(a) < 1e-9 or np.std(b) < 1e-9 else float(stats.spearmanr(a, b).statistic)
    R = lambda k: [r[k] for r in records]
    corr = {"fmv_vs_capPVI": sp(R("fmv"), R("cap_pvi_acc")), "ridge_vs_capPVI": sp(R("ridge"), R("cap_pvi_acc"))}
    print(f"\n[b6c] Spearman(recovery, capPVI) over ε: FMV={corr['fmv_vs_capPVI']:+.2f} ridge={corr['ridge_vs_capPVI']:+.2f}")
    print(f"[b6c] mean uplift FMV vs ridge={np.mean(R('uplift_fmv_vs_ridge')):+.3f} vs dec={np.mean(R('uplift_fmv_vs_dec')):+.3f}")
    Path(args.out).write_text(json.dumps({"layer": L, "topk": args.topk, "recorrelation": corr, "records": records}, indent=2))
    print(f"[b6c] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
