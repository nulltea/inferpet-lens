#!/usr/bin/env python3
"""Rep2Text on the Qwen3 residual stream (surface: resid-rep2text, Task 3).

Attack (Rep2Text, arXiv 2511.06571): a trained adapter MLP maps a SINGLE last-token
residual @ layer L of the source model (Qwen3-4B) to k soft-prompt embeddings that
drive a FROZEN pretrained decoder (Qwen3-1.7B, shared Qwen3 tokenizer) to regenerate
the input text. Only the adapter is trained (teacher-forced next-token CE through the
frozen decoder); eval is autoregressive greedy generation from the soft prompts.

Measurement loop:
  * C1 — recovery (token-F1 / ROUGE-L) vs sequence length L; real residual vs two
    controls (mean-residual / shuffled-residual) that subtract the decoder + length prior.
  * C2 — matched, geometry-only spectral channel-MI probe I_G of the last-token
    residual ensemble; rate-distortion proxy accessible_bits / H_X(L) vs recovery,
    swept over length buckets x residual-noise sigma. Spearman(probe, recovery).

GPU: wrap in scripts/run_in_rocm.sh. Capture is cached (load_or_capture); the >10-min
cost is adapter training through the frozen 1.7B decoder. Deterministic seeds.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402
from talens.capture.capture import load_or_capture  # noqa: E402
from talens.measures.spectral_channel_mi import spectral_channel_mi  # noqa: E402


# ----------------------------- recovery metrics -----------------------------
def token_f1(pred_ids: list[int], true_ids: list[int]) -> float:
    """Multiset (order-free) token F1 in the shared decoder token space."""
    if not pred_ids or not true_ids:
        return 0.0
    cp, ct = Counter(pred_ids), Counter(true_ids)
    overlap = sum((cp & ct).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(pred_ids), overlap / len(true_ids)
    return 2 * prec * rec / (prec + rec)


def rouge_l(pred_ids: list[int], true_ids: list[int]) -> float:
    """LCS-based ROUGE-L F1 over token-id sequences."""
    n, m = len(pred_ids), len(true_ids)
    if n == 0 or m == 0:
        return 0.0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        pi = pred_ids[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            row[j] = prev[j - 1] + 1 if pi == true_ids[j - 1] else max(prev[j], row[j - 1])
    lcs = dp[n][m]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / n, lcs / m
    return 2 * prec * rec / (prec + rec)


def spearman(a, b) -> float:
    from scipy import stats

    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(stats.spearmanr(a, b).statistic)


# ----------------------------- adapter -----------------------------
class Adapter(torch.nn.Module):
    def __init__(self, d_src: int, d_dec: int, k: int, hidden: int = 2048):
        super().__init__()
        self.k, self.d_dec = k, d_dec
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_src, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, k * d_dec),
        )

    def forward(self, h):  # h: (B, d_src) -> (B, k, d_dec)
        return self.net(h).view(h.shape[0], self.k, self.d_dec)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="corpora/rep2text-stratified.txt")
    ap.add_argument("--source-model", default="Qwen/Qwen3-4B")
    ap.add_argument("--decoder-model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--mode", choices=["smoke", "full"], default="full")
    ap.add_argument("--n-prompts", type=int, default=900)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-tgt-tokens", type=int, default=64)
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--n-test", type=int, default=140)
    ap.add_argument("--shuffle-draws", type=int, default=5)  # K shuffled-null draws for variance
    ap.add_argument("--sigmas", default="0,0.5,1.0")  # residual-noise inject (xstd units)
    ap.add_argument("--seed", type=int, default=20260624)
    ap.add_argument("--out", default="refine-logs/resid-rep2text/runs/rep2text_results.json")
    args = ap.parse_args()

    if args.mode == "smoke":
        args.n_prompts, args.epochs, args.n_test = 80, 2, 24
        args.sigmas = "0"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[r2t] device={dev} mode={args.mode}", flush=True)

    # ---- corpus ----
    raw = [l.strip() for l in open(REPO / args.corpus) if l.strip()]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(raw)
    prompts = raw[: args.n_prompts]
    print(f"[r2t] {len(prompts)} prompts from {args.corpus}", flush=True)

    # ---- capture last-token residual @ layer L (source model; cached) ----
    t0 = time.time()
    cap, _emb, src = load_or_capture(
        args.source_model, prompts, capture_layers=[args.layer], kinds=("resid_post",)
    )
    ops = cap.operands[("resid_post", args.layer)]
    H = np.stack([op[-1].numpy().astype(np.float32) for op in ops])  # (N, d_src) last token
    d_src = H.shape[1]
    print(f"[r2t] capture src={src} H={H.shape} ({time.time()-t0:.1f}s)", flush=True)

    # ---- decoder (frozen) ----
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.decoder_model)
    dec = AutoModelForCausalLM.from_pretrained(
        args.decoder_model, torch_dtype=torch.bfloat16
    ).to(dev)
    dec.eval()
    for p in dec.parameters():
        p.requires_grad_(False)
    d_dec = dec.config.hidden_size
    vocab = dec.config.vocab_size
    emb_layer = dec.get_input_embeddings()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    # target token ids per prompt (decoder tokenizer), capped
    tgt_ids = [tok(p, add_special_tokens=False)["input_ids"][: args.max_tgt_tokens] for p in prompts]
    tok_len = np.array([len(t) for t in tgt_ids])
    # length buckets by token count
    edges = [0, 12, 18, 24, 32, 48, 10**9]
    bucket = np.digitize(tok_len, edges[1:-1])  # 0..len(edges)-2
    bnames = ["<=12", "13-18", "19-24", "25-32", "33-48", "49+"]

    # ---- split: stratified by bucket -> test gets >= a few per bucket ----
    idx = np.arange(len(prompts))
    rng.shuffle(idx)
    test_idx, train_idx = [], []
    per_bucket_test = max(1, args.n_test // len(bnames))
    seen: dict[int, int] = {}
    for i in idx:
        b = int(bucket[i])
        if seen.get(b, 0) < per_bucket_test and len(test_idx) < args.n_test:
            test_idx.append(i)
            seen[b] = seen.get(b, 0) + 1
        else:
            train_idx.append(i)
    test_idx = np.array(test_idx)
    train_idx = np.array(train_idx)
    print(f"[r2t] train={len(train_idx)} test={len(test_idx)} d_src={d_src} d_dec={d_dec}", flush=True)

    Ht = torch.from_numpy(H).to(dev)
    # standardize residual (per-dim) for stable adapter training
    mu, sd = Ht.mean(0, keepdim=True), Ht.std(0, keepdim=True) + 1e-6
    Hn = (Ht - mu) / sd

    adapter = Adapter(d_src, d_dec, args.k).to(dev).to(torch.float32)
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)

    def make_batch(ids_subset):
        L = max(len(tgt_ids[i]) for i in ids_subset)
        B = len(ids_subset)
        ti = torch.full((B, L), pad_id, dtype=torch.long, device=dev)
        tmask = torch.zeros((B, L), dtype=torch.long, device=dev)
        for r, i in enumerate(ids_subset):
            t = tgt_ids[i]
            ti[r, : len(t)] = torch.tensor(t, device=dev)
            tmask[r, : len(t)] = 1
        return ti, tmask

    # ---- train adapter (teacher forcing through frozen decoder) ----
    t0 = time.time()
    for ep in range(args.epochs):
        perm = train_idx[rng.permutation(len(train_idx))]
        ep_loss, nb = 0.0, 0
        for s in range(0, len(perm), args.batch):
            sub = perm[s : s + args.batch].tolist()
            if len(sub) < 2:
                continue
            ti, tmask = make_batch(sub)
            soft = adapter(Hn[sub]).to(torch.bfloat16)  # (B,k,d_dec)
            temb = emb_layer(ti)  # (B,L,d_dec)
            inp = torch.cat([soft, temb], dim=1)
            amask = torch.cat(
                [torch.ones(len(sub), args.k, dtype=torch.long, device=dev), tmask], dim=1
            )
            labels = torch.cat(
                [torch.full((len(sub), args.k), -100, dtype=torch.long, device=dev),
                 ti.masked_fill(tmask == 0, -100)],
                dim=1,
            )
            out = dec(inputs_embeds=inp, attention_mask=amask, labels=labels)
            opt.zero_grad()
            out.loss.backward()
            opt.step()
            ep_loss += float(out.loss.detach())
            nb += 1
        print(f"[r2t] epoch {ep} CE={ep_loss/max(nb,1):.4f} ({time.time()-t0:.0f}s)", flush=True)

    # ---- eval: greedy generation from soft prompts ----
    adapter.eval()
    # sigma levels are FRACTIONS of the raw-residual RMS scale; noise is injected
    # ISOTROPICALLY in raw space so it matches the spectral probe's σ²I channel.
    fracs = [float(x) for x in args.sigmas.split(",") if x.strip()]
    Hc = H - H.mean(0, keepdims=True)
    rms = float(np.sqrt((Hc ** 2).mean()))  # geometry-only RMS scale of residual
    SNR0 = 100.0                            # fixed reference SNR (geometry-only floor)
    sigma_ref = rms / math.sqrt(SNR0)       # finite noise floor -> I_G is finite & binding
    print(f"[r2t] resid rms={rms:.4f} sigma_ref={sigma_ref:.4f} (SNR0={SNR0})", flush=True)

    @torch.no_grad()
    def generate(h_std_batch):
        soft = adapter(h_std_batch).to(torch.bfloat16)
        amask = torch.ones(soft.shape[0], args.k, dtype=torch.long, device=dev)
        gen = dec.generate(
            inputs_embeds=soft, attention_mask=amask,
            max_new_tokens=args.max_tgt_tokens, do_sample=False, num_beams=1,
            pad_token_id=pad_id,
        )
        return [g.tolist() for g in gen]  # new tokens only

    def eval_variant(h_provider, label):
        """h_provider(test_index_array)->(M,d_src_std) tensor; returns per-prompt metrics."""
        recs = []
        for s in range(0, len(test_idx), args.gen_batch):
            bi = test_idx[s : s + args.gen_batch]
            hb = h_provider(bi)
            preds = generate(hb)
            for j, i in enumerate(bi):
                p = [t for t in preds[j] if t != pad_id]
                recs.append((int(bucket[i]), token_f1(p, tgt_ids[i]), rouge_l(p, tgt_ids[i])))
        return recs

    results = {"config": vars(args), "d_src": d_src, "d_dec": int(d_dec), "vocab": int(vocab),
               "resid_rms": rms, "sigma_ref": sigma_ref, "SNR0": SNR0,
               "bucket_names": bnames, "sweeps": []}

    H_mean = Hn.mean(0, keepdim=True)
    Hraw = torch.from_numpy(H).to(dev)
    n_shuf = args.shuffle_draws

    # K independent shuffled-residual draws (the null: a different prompt's residual),
    # all aligned to test_idx order so real and shuffled pair per example.
    shuffle_draws = []  # each: list[(bucket,f1,rougeL)] aligned to test_idx
    for s in range(n_shuf):
        perm = test_idx[rng.permutation(len(test_idx))]
        sp = {int(a): int(b) for a, b in zip(test_idx, perm)}
        shuffle_draws.append(eval_variant(lambda bi, sp=sp: Hn[[sp[int(i)] for i in bi]], f"shuf{s}"))
    mean_recs = eval_variant(lambda bi: H_mean.expand(len(bi), -1), "mean")
    # avg shuffled f1 per example (over draws) -> the paired null
    shuf_f1_avg = np.mean([[r[1] for r in dr] for dr in shuffle_draws], axis=0)  # (n_test,)
    shuf_f1_std = np.std([np.mean([r[1] for r in dr]) for dr in shuffle_draws])  # across-draw null std
    real_f0_per_ex = None  # set at frac==0

    for frac in fracs:
        raw_inject = frac * rms
        # total channel noise seen by the probe = injected + reference floor (matched)
        raw_sigma = math.sqrt(raw_inject ** 2 + sigma_ref ** 2)

        def real_provider(bi, raw_inject=raw_inject):
            h = Hraw[bi]
            if raw_inject > 0:
                h = h + raw_inject * torch.randn_like(h)
            return (h - mu) / sd  # standardize for the adapter
        recs = eval_variant(real_provider, f"real_f{frac}")
        # one shuffled pass WITH the same noise -> leakage gap across sigma
        permn = test_idx[rng.permutation(len(test_idx))]
        spn = {int(a): int(b) for a, b in zip(test_idx, permn)}

        def shuf_provider(bi, raw_inject=raw_inject, spn=spn):
            h = Hraw[[spn[int(i)] for i in bi]]
            if raw_inject > 0:
                h = h + raw_inject * torch.randn_like(h)
            return (h - mu) / sd
        shuf_recs = eval_variant(shuf_provider, f"shuf_f{frac}")
        ctrl = {}
        ctrl_recs: dict[str, list] = {}
        if frac == 0:
            real_f0_per_ex = np.array([r[1] for r in recs])  # for bootstrap pairing
            ctrl_recs["mean"] = mean_recs
            ctrl_recs["shuffled"] = shuffle_draws[0]
            ctrl = {"mean": {"token_f1": float(np.mean([r[1] for r in mean_recs]))},
                    "shuffled": {"token_f1": float(np.mean(shuf_f1_avg)),
                                 "token_f1_draw_std": float(shuf_f1_std), "n_draws": n_shuf}}

        # --- probe: ONE single-vector capacity I_G from the FULL ensemble at total sigma ---
        # (geometry-only; the bottleneck is a fixed d-dim vector, so capacity is one number).
        ig_full = spectral_channel_mi(E0=H, sigma=raw_sigma)
        I_G = float(ig_full["i_g_bits"])  # finite (sigma_ref floor), bits
        # rate-distortion ceiling per length bucket: recoverable_frac(L) = min(I_G, H_X(L))/H_X(L)
        per_bucket = {}
        for b in range(len(bnames)):
            mask = bucket[test_idx] == b
            allmask = bucket == b
            if allmask.sum() < 4 or mask.sum() == 0:
                continue
            Lb = float(tok_len[allmask].mean())
            H_X = Lb * math.log2(vocab)            # upper proxy (flagged): L·log2(vocab)
            accessible = min(I_G, H_X)
            f1s = [r[1] for r in recs if r[0] == b]
            rls = [r[2] for r in recs if r[0] == b]
            entry = {
                "n_test": int(mask.sum()), "mean_tok_len": Lb,
                "token_f1": float(np.mean(f1s)) if f1s else None,
                "rouge_l": float(np.mean(rls)) if rls else None,
                "i_g_full_bits": I_G,
                "accessible_bits": accessible,
                "H_X_bits": H_X,
                "rd_proxy_recoverable_frac": accessible / H_X if H_X > 0 else None,
            }
            if frac == 0:  # per-bucket genuine-leakage gap with bootstrap CI + null variance
                ex_mask = bucket[test_idx] == b           # examples in this bucket (test_idx order)
                r_b = real_f0_per_ex[ex_mask]              # real per-example f1
                s_b = shuf_f1_avg[ex_mask]                 # avg-over-draws shuffled per-example f1
                mean_b = np.array([r[1] for r in mean_recs])[ex_mask]
                gap_b = r_b - s_b
                # paired bootstrap over examples (B resamples)
                B = 5000
                bidx = np.random.default_rng(args.seed + b).integers(0, len(gap_b), size=(B, len(gap_b)))
                boot = gap_b[bidx].mean(axis=1)
                entry["ctrl_mean_f1"] = float(mean_b.mean())
                entry["ctrl_shuffled_f1"] = float(s_b.mean())
                entry["leakage_gap_f1"] = float(gap_b.mean())
                entry["leakage_gap_ci95"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
                entry["leakage_gap_p_le0"] = float((boot <= 0).mean())  # one-sided bootstrap p
            per_bucket[bnames[b]] = entry
        sweep = {"noise_frac": frac, "raw_sigma": raw_sigma, "per_bucket": per_bucket}
        if ctrl:
            sweep["controls"] = ctrl
        sweep["overall_token_f1"] = float(np.mean([r[1] for r in recs]))
        sweep["overall_shuffled_token_f1"] = float(np.mean([r[1] for r in shuf_recs]))
        sweep["leakage_gap_overall"] = sweep["overall_token_f1"] - sweep["overall_shuffled_token_f1"]
        sweep["i_g_full_bits"] = I_G
        results["sweeps"].append(sweep)
        print(f"[r2t] frac={frac} raw_sigma={raw_sigma:.4f} real={sweep['overall_token_f1']:.3f} "
              f"shuf={sweep['overall_shuffled_token_f1']:.3f} gap={sweep['leakage_gap_overall']:+.3f} "
              f"I_G={I_G:.0f}b", flush=True)

    # ---- C2 correlations ----
    xs, ys, gaps = [], [], []
    for sw in results["sweeps"]:
        for bn, d in sw["per_bucket"].items():
            if d["token_f1"] is not None and d["rd_proxy_recoverable_frac"] is not None:
                xs.append(d["rd_proxy_recoverable_frac"])
                ys.append(d["token_f1"])
                if d.get("leakage_gap_f1") is not None:
                    gaps.append((d["rd_proxy_recoverable_frac"], d["leakage_gap_f1"]))
    # across-sigma: does overall recovery (and the genuine leakage gap) track capacity I_G(sigma)?
    sig_ig = [sw["i_g_full_bits"] for sw in results["sweeps"]]
    sig_f1 = [sw["overall_token_f1"] for sw in results["sweeps"]]
    sig_gap = [sw["leakage_gap_overall"] for sw in results["sweeps"]]
    results["correlation"] = {
        "n_points_bucket_x_sigma": len(xs),
        "spearman_rdproxy_vs_f1": spearman(xs, ys),
        "spearman_rdproxy_vs_leakage_gap": spearman([g[0] for g in gaps], [g[1] for g in gaps]) if gaps else None,
        "across_sigma_spearman_IG_vs_overall_f1": spearman(sig_ig, sig_f1),
        "across_sigma_spearman_IG_vs_leakage_gap": spearman(sig_ig, sig_gap),
        "across_sigma_IG_bits": sig_ig,
        "across_sigma_overall_f1": sig_f1,
        "across_sigma_leakage_gap": sig_gap,
    }
    print(f"[r2t] C2 rd_proxy~f1 ρ={results['correlation']['spearman_rdproxy_vs_f1']:.3f} | "
          f"across-σ I_G~f1 ρ={results['correlation']['across_sigma_spearman_IG_vs_overall_f1']:.3f}", flush=True)

    outp = REPO / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2, default=str))
    print(f"[r2t] wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
