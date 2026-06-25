#!/usr/bin/env python3
"""B7 — FAITHFUL Vec2Text (Morris et al. 2023) under PROPAGATED input-DP @L20.

What B6c (FMV) was NOT: B6c re-ranked a FIXED decoder top-k once, with a
TEACHER-FORCED true prefix. The candidates never change; the forward model only
scores. That is one-shot reranking, not Vec2Text.

This is the genuine iterative hypothesis-refinement loop of Morris et al.:

    h^(0)  = full token sequence: known (train-token) positions = true CONTEXT,
             unknown (test-token) positions = base decoder hypothesis
    for t in 1..T:
        ê^(t) = φ(h^(t))                      # RE-EMBED the current hypothesis
        h^(t+1) = corrector(e, ê^(t), e-ê^(t), h^(t))   # NEW tokens at test positions
        accept h^(t+1) iff it is closer to e (cosine)   # paper's seq-level rule

Known positions are teacher-forced as fixed context (strictly LESS teacher forcing
than B6c, which forced the entire prefix); the test positions are recovered jointly
and iteratively. Recovery is over a test-only candidate pool (train tokens excluded
— B6c vocab-disjoint protocol); the corrector is trained symmetrically on train
positions over a train-only pool, so it never sees a test token as a target.

Here the "embedding" is the resid_post at layer L (per position); e = Y_obs is the
DP-noised observation; φ = clip-only forward (WEIGHTS-PUB, attacker has weights).
The corrector is a per-position network conditioned on the four Vec2Text inputs —
the EmbToSeq(e), EmbToSeq(ê), EmbToSeq(e-ê), word-embeddings(x^(t)) concatenation
of the paper, collapsed to an MLP so it trains in minutes. Each iteration emits a
genuinely NEW sequence that is re-embedded — the residual e-ê^(t) is the fresh
information injected each round.

Decisive faithfulness check (Morris Fig 3): the FEEDBACK corrector (conditioned on
ê^(t)) vs the NO-FEEDBACK ablation (ê,e-ê zeroed). Feedback should keep improving
with t; no-feedback should plateau.

Hypothesis vs B6c's regime-dependence: the corrector is trained on NOISED Y
(noise-aware → robust at high σ like the B6 decoder) AND uses the model-in-loop
re-embedding (the low-σ power of FMV) → it should hold at BOTH ends of the noise
range where FMV (low only) and ridge/decoder (high only) each failed.

Single GPU process; run via scripts/run_in_rocm.sh. Budget ≤20 min.
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
from talens.probes.vinfo_capacity import v_information_capacity  # noqa: E402
from scipy import stats  # noqa: E402


@torch.no_grad()
def forward_resid_ids(model, ids_list, L, device, batch=16):
    """Clip-only forward of token-id sequences → list of (n_i, d) resid at layer L.
    A clip-only InputDPCover hook (σ=0) must be registered by the caller so the
    re-embedding ê matches the clip scale of the (clipped+noised) observation."""
    out = []
    for i in range(0, len(ids_list), batch):
        chunk = ids_list[i:i + batch]
        Lmax = max(len(s) for s in chunk)
        t = torch.zeros(len(chunk), Lmax, dtype=torch.long, device=device)
        for j, s in enumerate(chunk):
            t[j, :len(s)] = torch.from_numpy(np.asarray(s, dtype=np.int64))
        hs = model(t, output_hidden_states=True, use_cache=False).hidden_states[L + 1]
        for j, s in enumerate(chunk):
            out.append(hs[j, :len(s), :].float().cpu().numpy())
    return out


def topk_tokens(pred_emb, pool_emb_n, pool_ids, k=1):
    """argmax/top-k over the candidate pool by cosine. pred_emb:(n,d) np."""
    p = pred_emb / np.clip(np.linalg.norm(pred_emb, axis=1, keepdims=True), 1e-9, None)
    sims = p @ pool_emb_n.T
    if k == 1:
        return pool_ids[sims.argmax(1)]
    idx = np.argpartition(-sims, k, axis=1)[:, :k]
    return pool_ids[idx]


def cos_rows(A, B):
    """per-row cosine of two (n,d) arrays."""
    a = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-9, None)
    b = B / np.clip(np.linalg.norm(B, axis=1, keepdims=True), 1e-9, None)
    return (a * b).sum(1)


def make_feats(yobs, ehat, htok, table, feedback):
    """Vec2Text per-position corrector input: [e, ê, e-ê, emb(x^t)] (4d).
    no-feedback ablation zeroes the ê and e-ê blocks (Morris Fig 3)."""
    he = table[htok]
    if feedback:
        return np.concatenate([yobs, ehat, yobs - ehat, he], axis=1).astype(np.float32)
    z = np.zeros_like(yobs)
    return np.concatenate([yobs, z, z, he], axis=1).astype(np.float32)


@torch.no_grad()
def corrector_emb(net, feats):
    return net(torch.from_numpy(feats).to(DEV)).cpu().numpy()


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
    ap.add_argument("--rounds", type=int, default=4, help="Vec2Text correction rounds T")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--fmv-topk", type=int, default=16, help="teacher-forced FMV reference (B6c)")
    ap.add_argument("--fmv-max-pos", type=int, default=400, help="cap scored FMV positions (budget)")
    ap.add_argument("--seed", type=int, default=20260622)
    ap.add_argument("--out", default="results/b7_vec2text.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = DEV; L = args.layer
    eps_list = [math.inf if s.strip().lower().startswith("inf") else float(s) for s in args.epsilons.split(",") if s.strip()]
    prompts = [l.strip() for l in Path(args.corpus).read_text().splitlines() if l.strip()][: args.max_prompts]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map=device).eval()
    table = model.get_input_embeddings().weight.detach().float().cpu().numpy().astype(np.float32)
    vocab = table.shape[0]
    ids_per = [tok(p, return_tensors="pt").input_ids[0].numpy().astype(np.int64) for p in prompts]

    # clip C from runtime embedding norms (the published DP clip)
    cal = []
    h = model.model.embed_tokens.register_forward_hook(lambda m, i, o: cal.append(o.float().norm(dim=-1).flatten().cpu()))
    with torch.no_grad():
        for p in prompts[:48]: model(tok(p, return_tensors="pt").input_ids.to(device), use_cache=False)
    h.remove()
    C = float(np.percentile(torch.cat(cal).numpy(), args.clip_percentile)); z = math.sqrt(2 * math.log(1.25 / args.delta))
    rng = np.random.default_rng(args.seed)
    print(f"[b7] C={C:.3f} L={L} eps={eps_list} prompts={len(prompts)} T={args.rounds}", flush=True)

    def stack(per_L):
        Xs, ys, idx = [], [], []
        for pi, (m, t) in enumerate(zip(per_L, ids_per)):
            n = min(m.shape[0], t.shape[0]); Xs.append(m[:n]); ys.append(t[:n])
            idx += [(pi, pos) for pos in range(n)]
        return np.concatenate(Xs, 0), np.concatenate(ys, 0).astype(np.int64), idx

    # Vocab-disjoint split on token id (shared across ε). TWO pools (B6c protocol —
    # train tokens NEVER in the eval pool, else the decoder's memorized train-token
    # predictions win at test positions and crush recovery to chance):
    #   te_pool = test-true + fillers   → recover/iterate TEST positions (the metric)
    #   tr_pool = train-true + fillers  → generate corrector training hypotheses
    # Known (train-token) positions are teacher-forced as fixed CONTEXT — strictly
    # less teacher forcing than B6c, which forced the entire prefix; here the test
    # positions are recovered JOINTLY and ITERATIVELY (the Vec2Text loop).
    _, y_all, _ = stack(capture(model, tok, prompts, [L], device)[0][L])
    distinct = rng.permutation(np.unique(y_all)); ntr = int(0.7 * distinct.size)
    tr_ids = set(distinct[:ntr].tolist()); te_ids = set(distinct[ntr:].tolist())
    te_true = np.array(sorted(te_ids), dtype=np.int64); tr_true = np.array(sorted(tr_ids), dtype=np.int64)
    avail = np.setdiff1d(np.arange(vocab, dtype=np.int64), np.unique(y_all))
    fperm = rng.permutation(avail.size)
    def mk_pool(true):
        k = max(0, args.pool_size - true.size)
        f = avail[fperm[:k]] if k else np.empty(0, np.int64)
        return np.concatenate([true, f.astype(np.int64)])
    te_pool = mk_pool(te_true); tr_pool = mk_pool(tr_true)
    te_pool_n = table[te_pool] / np.clip(np.linalg.norm(table[te_pool], axis=1, keepdims=True), 1e-9, None)
    tr_pool_n = table[tr_pool] / np.clip(np.linalg.norm(table[tr_pool], axis=1, keepdims=True), 1e-9, None)
    chance = 1.0 / te_pool.size
    # per-prompt masks (token lengths fixed across ε)
    te_mask_per = [np.array([t in te_ids for t in ids_per[pi]]) for pi in range(len(prompts))]
    tr_mask_per = [np.array([t in tr_ids for t in ids_per[pi]]) for pi in range(len(prompts))]
    print(f"[b7] te_pool={te_pool.size} tr_pool={tr_pool.size} chance={chance:.2e} tr_tok={len(tr_ids)} te_tok={len(te_ids)}", flush=True)

    def recov(h_eval_per):
        """top-1 recovery over te_pool at test positions (h built with train ctx fixed)."""
        ok = tot = 0
        for pi, h in enumerate(h_eval_per):
            m = te_mask_per[pi][:len(h)]; tids = ids_per[pi][:len(h)]
            ok += int(((h == tids) & m).sum()); tot += int(m.sum())
        return ok / max(1, tot)

    def fill_positions(dec_or_cor, yobs_per, ehat_per, mask_per, pool_n, pool_ids, feedback=None):
        """build hypothesis: true tokens everywhere, predicted tokens at `mask` positions
        (decoder if feedback is None, else corrector on the 4-block features)."""
        out = []
        for pi in range(len(prompts)):
            h = ids_per[pi][: yobs_per[pi].shape[0]].copy(); m = mask_per[pi][: h.shape[0]]
            if m.any():
                if feedback is None:
                    pred = corrector_emb(dec_or_cor, yobs_per[pi][m])
                else:
                    f = make_feats(yobs_per[pi][m], ehat_per[pi][m], h[m], table, feedback)
                    pred = corrector_emb(dec_or_cor, f)
                h[m] = topk_tokens(pred, pool_n, pool_ids, 1)
            out.append(h)
        return out

    records = []
    for eps in eps_list:
        sigma = 0.0 if math.isinf(eps) else C * z / eps
        torch.manual_seed(args.seed + (0 if math.isinf(eps) else int(eps)))
        hk = model.model.embed_tokens.register_forward_hook(InputDPCover(C, sigma))
        per, _ = capture(model, tok, prompts, [L], device); hk.remove()
        per_mats = per[L]
        yobs_per = [per_mats[pi][:min(per_mats[pi].shape[0], ids_per[pi].shape[0])] for pi in range(len(prompts))]
        X, y, idx = stack(per_mats)
        tr = np.array([i for i, t in enumerate(y) if t in tr_ids]); te = np.array([i for i, t in enumerate(y) if t in te_ids])
        emb_y = table[y]
        t0 = time.time()

        # --- ridge baseline + base decoder (over te_pool, B6c-comparable) ---
        ridge_t = float((ridge_match(X[tr], emb_y[tr], X[te], table[te_pool], te_pool) == y[te]).mean())
        dec = train_decoder(X[tr], emb_y[tr], hidden=args.hidden, epochs=args.epochs, seed=args.seed)
        clip_hk = model.model.embed_tokens.register_forward_hook(InputDPCover(C, 0.0))

        # h^(0) eval hypothesis: train ctx = true, test positions = base decoder over te_pool
        h0_eval = fill_positions(dec, yobs_per, None, te_mask_per, te_pool_n, te_pool, feedback=None)
        base_acc = recov(h0_eval)

        # --- corrector training (feedback + no-feedback ablation) ---
        # symmetric to eval: TRAIN positions = base decoder over tr_pool, test ctx = true,
        # re-embed → ê, supervise corrector to map (e,ê,e-ê,emb(guess)) → true train emb.
        h_train = fill_positions(dec, yobs_per, None, tr_mask_per, tr_pool_n, tr_pool, feedback=None)
        ehat_train = forward_resid_ids(model, h_train, L, device)
        def train_corrector(feedback):
            F, T = [], []
            for pi in range(len(prompts)):
                m = tr_mask_per[pi][: yobs_per[pi].shape[0]]
                if m.any():
                    F.append(make_feats(yobs_per[pi][m], ehat_train[pi][m], h_train[pi][m], table, feedback))
                    T.append(table[ids_per[pi][: m.shape[0]][m]])
            return train_decoder(np.concatenate(F, 0), np.concatenate(T, 0), hidden=args.hidden, epochs=args.epochs, seed=args.seed)
        cor_fb = train_corrector(True); cor_nf = train_corrector(False)

        # --- Vec2Text iterative inference (feedback & no-feedback), seq-level accept ---
        def vec2text(net, feedback):
            h_per = [a.copy() for a in h0_eval]
            eh_per = forward_resid_ids(model, h_per, L, device)
            curve = [recov(h_per)]  # t=0
            for _ in range(args.rounds):
                hnew = fill_positions(net, yobs_per, eh_per, te_mask_per, te_pool_n, te_pool, feedback=feedback)
                ehnew = forward_resid_ids(model, hnew, L, device)
                for pi in range(len(prompts)):
                    # paper's rule: accept iff closer to e. Judge on the TEST positions we
                    # recover (train ctx is fixed=true → it would drown the sequence mean).
                    m = te_mask_per[pi][: eh_per[pi].shape[0]]
                    if not m.any():
                        continue
                    if cos_rows(ehnew[pi][m], yobs_per[pi][m]).mean() > cos_rows(eh_per[pi][m], yobs_per[pi][m]).mean():
                        h_per[pi] = hnew[pi]; eh_per[pi] = ehnew[pi]
                curve.append(recov(h_per))
            return curve
        curve_fb = vec2text(cor_fb, True)
        curve_nf = vec2text(cor_nf, False)

        # --- teacher-forced FMV reference (B6c): re-rank decoder top-k by clean fwd ---
        dtopk = [topk_tokens(corrector_emb(dec, yo), te_pool_n, te_pool, args.fmv_topk) for yo in yobs_per]
        fmv_pos = [(pi, pos) for pi in range(len(prompts))
                   for pos in range(min(len(ids_per[pi]), per_mats[pi].shape[0])) if ids_per[pi][pos] in te_ids]
        if len(fmv_pos) > args.fmv_max_pos:
            fmv_pos = [fmv_pos[i] for i in rng.choice(len(fmv_pos), args.fmv_max_pos, replace=False)]
        fmv_ok = 0
        for pi, pos in fmv_pos:
            tids = ids_per[pi]; cands = dtopk[pi][pos]; prefix = tids[:pos]
            seqs = [np.concatenate([prefix, [c]]).astype(np.int64) for c in cands]
            rl = np.stack([r[-1] for r in forward_resid_ids(model, seqs, L, device)])  # (k,d) clean resid at last pos
            best = cands[int(np.argmin(((rl - per_mats[pi][pos][None, :]) ** 2).sum(1)))]
            fmv_ok += int(best == tids[pos])
        fmv_t = fmv_ok / max(1, len(fmv_pos))
        clip_hk.remove()

        capv = v_information_capacity(X, y, family="pca_softmax", dim=64, l2=0.1)["reader_top1_acc"]
        rec = {"epsilon": (None if math.isinf(eps) else eps), "sigma": sigma,
               "ridge": ridge_t, "base_decoder": base_acc, "fmv_tf": fmv_t,
               "vec2text_feedback": curve_fb, "vec2text_nofeedback": curve_nf,
               "v2t_fb_final": curve_fb[-1], "v2t_nf_final": curve_nf[-1],
               "cap_pvi_acc": capv, "secs": round(time.time() - t0, 1)}
        records.append(rec)
        es = "inf" if math.isinf(eps) else f"{eps:g}"
        print(f"[b7] ε={es:>5} | ridge={ridge_t:.3f} base={base_acc:.3f} FMV-tf={fmv_t:.3f} "
              f"| V2T-fb {['%.3f'%c for c in curve_fb]} | V2T-nf {['%.3f'%c for c in curve_nf]} "
              f"| capPVI={capv:.3f} [{rec['secs']}s]", flush=True)

    def sp(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return 0.0 if np.std(a) < 1e-9 or np.std(b) < 1e-9 else float(stats.spearmanr(a, b).statistic)
    R = lambda k: [r[k] for r in records]
    corr = {"v2t_fb_vs_capPVI": sp(R("v2t_fb_final"), R("cap_pvi_acc")),
            "fmv_vs_capPVI": sp(R("fmv_tf"), R("cap_pvi_acc")),
            "ridge_vs_capPVI": sp(R("ridge"), R("cap_pvi_acc"))}
    print(f"\n[b7] Spearman(recovery, capPVI) over ε: V2T-fb={corr['v2t_fb_vs_capPVI']:+.2f} "
          f"FMV={corr['fmv_vs_capPVI']:+.2f} ridge={corr['ridge_vs_capPVI']:+.2f}")
    Path(args.out).write_text(json.dumps({"layer": L, "rounds": args.rounds, "recorrelation": corr, "records": records}, indent=2))
    print(f"[b7] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
