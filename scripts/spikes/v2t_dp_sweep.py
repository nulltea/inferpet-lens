#!/usr/bin/env python3
"""M1+M2 — Faithful Vec2Text on pooled GTR embeddings under DP.

The surface where Vec2Text is the CORRECT attack (single pooled bottleneck vector), vs
the per-position resid where B7 disproved it. Uses the REAL vec2text dependency with its
PRETRAINED gtr-base corrector (no training). Plan:
refine-logs/dp-stronger-attacks/vec2text-pooled/EXPERIMENT_PLAN.md.

Pipeline: GTR-base encodes N 32-token texts → e0 (mean-pooled, 768-d). Defense = DP on the
released embedding: clip to norm C (p99.9 of ‖e0‖) then add N(0,σ²), σ = C·z/ε. For each ε,
invert e' with Vec2Text (num_steps + sequence_beam_width) and score recovery (BLEU / token-F1
/ exact / cos to e0). Matched probe = CLUB I(e'; e0) (and capPVI on a cluster label). C2: does
recovery decay monotonically with ε? C3: does the cheap probe track the SOTA attack's recovery
(Spearman over ε)? ε=∞ + num_steps=0 row = base-model baseline (C1/C4).

Run via scripts/run_in_rocm.sh with HOME + TORCH_EXTENSIONS_DIR pointed at /tmp (apex JIT).
See memory vec2text-rocm-dependency-recipe.
"""
import sys, os, argparse, json, math, time, re
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, ".deps"))   # vec2text + transformers-4.44 shadow
sys.path.insert(0, os.path.join(REPO, "src"))      # talens measures
import numpy as np, torch
import vec2text
from transformers import AutoModel, AutoTokenizer
import sacrebleu
from talens.probes.club import club_mi_upper_bound
from talens.probes.vinfo_capacity import v_information_capacity

DEV = "cuda" if torch.cuda.is_available() else "cpu"
_WS = re.compile(r"\s+")


def norm_txt(s):
    return _WS.sub(" ", s).strip().lower()


def token_f1(pred, true):
    p, t = norm_txt(pred).split(), norm_txt(true).split()
    if not p or not t:
        return 0.0
    from collections import Counter
    cp, ct = Counter(p), Counter(t)
    overlap = sum((cp & ct).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(t)
    return 2 * prec * rec / (prec + rec)


@torch.no_grad()
def get_gtr(texts, enc, tokz, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        chunk = texts[i:i + bs]
        inp = tokz(chunk, return_tensors="pt", max_length=128, truncation=True, padding="max_length").to(DEV)
        o = enc(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
        out.append(vec2text.models.model_utils.mean_pool(o.last_hidden_state, inp["attention_mask"]).cpu())
    return torch.cat(out, 0)


def kmeans_labels(X, k, iters=25, seed=0):
    rng = np.random.default_rng(seed)
    c = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        d = ((X[:, None, :] - c[None]) ** 2).sum(2)
        lab = d.argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                c[j] = X[m].mean(0)
    return lab.astype(np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="corpora/release-gate-512.txt")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=32, help="Morris 32-token regime")
    ap.add_argument("--epsilons", default="inf,1024,512,256,128")
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--clip-percentile", type=float, default=99.9)
    ap.add_argument("--kmeans-k", type=int, default=40)
    ap.add_argument("--base-baseline", action="store_true", help="also run num_steps=0 base model at ε=∞")
    ap.add_argument("--quick", action="store_true", help="tiny sanity: n=16, ε=inf,256")
    ap.add_argument("--seed", type=int, default=20260622)
    ap.add_argument("--out", default="results/v2t_dp_sweep.json")
    args = ap.parse_args()
    if args.quick:
        args.n, args.epsilons = 16, "inf,256"

    eps_list = [math.inf if s.strip().lower().startswith("inf") else float(s) for s in args.epsilons.split(",") if s.strip()]
    rng = np.random.default_rng(args.seed)
    print(f"[v2t-dp] loading GTR encoder + gtr-base corrector on {DEV}", flush=True)
    enc = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to(DEV).eval()
    tokz = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
    cor = vec2text.load_pretrained_corrector("gtr-base")

    # corpus → canonical 32-token texts (truncate via GTR tokenizer, decode back)
    raw = [l.strip() for l in open(os.path.join(REPO, args.corpus)) if l.strip()]
    rng.shuffle(raw)
    texts = []
    for t in raw:
        ids = tokz(t, truncation=True, max_length=args.max_tokens)["input_ids"]
        if len(ids) >= 8:  # skip near-empty
            texts.append(tokz.decode(ids, skip_special_tokens=True))
        if len(texts) >= args.n:
            break
    n = len(texts)
    print(f"[v2t-dp] N={n} texts ({args.max_tokens}-tok), ε={eps_list}, steps={args.num_steps} beam={args.beam}", flush=True)

    e0 = get_gtr(texts, enc, tokz)                       # (n,768) clean pooled
    e0n = e0.numpy().astype(np.float32)
    C = float(np.percentile(np.linalg.norm(e0n, axis=1), args.clip_percentile))
    z = math.sqrt(2 * math.log(1.25 / args.delta))
    labels = kmeans_labels(e0n, min(args.kmeans_k, n // 3), seed=args.seed)
    print(f"[v2t-dp] clip C={C:.3f} (p{args.clip_percentile}); kmeans k={min(args.kmeans_k, n//3)}", flush=True)

    def score(recon):
        bleu = float(np.mean([sacrebleu.sentence_bleu(r, [t]).score for r, t in zip(recon, texts)]))
        tf1 = float(np.mean([token_f1(r, t) for r, t in zip(recon, texts)]))
        exact = float(np.mean([norm_txt(r) == norm_txt(t) for r, t in zip(recon, texts)]))
        er = get_gtr(recon, enc, tokz).numpy().astype(np.float32)
        cos = float(np.mean((er * e0n).sum(1) / (np.linalg.norm(er, axis=1) * np.linalg.norm(e0n, axis=1) + 1e-9)))
        return {"bleu": bleu, "token_f1": tf1, "exact": exact, "cos": cos}

    def invert(emb, steps):
        return vec2text.invert_embeddings(embeddings=emb.to(DEV), corrector=cor,
                                          num_steps=(None if steps == 0 else steps),
                                          sequence_beam_width=(0 if steps == 0 else args.beam))

    records = []
    # optional base-model baseline (num_steps=0) on the clean embedding
    if args.base_baseline:
        t0 = time.time(); rec0 = invert(e0, 0); m = score(rec0)
        records.append({"epsilon": None, "attack": "base_0step", **m, "secs": round(time.time() - t0, 1)})
        print(f"[v2t-dp] base[0-step] ε=inf | bleu={m['bleu']:.1f} tF1={m['token_f1']:.3f} exact={m['exact']:.3f} cos={m['cos']:.3f}", flush=True)

    for eps in eps_list:
        sigma = 0.0 if math.isinf(eps) else C * z / eps
        scale = np.minimum(1.0, C / (np.linalg.norm(e0n, axis=1, keepdims=True) + 1e-9))
        e_clip = e0n * scale
        e_noisy = e_clip + (sigma * rng.standard_normal(e_clip.shape).astype(np.float32) if sigma > 0 else 0.0)
        et = torch.from_numpy(e_noisy.astype(np.float32))
        t0 = time.time(); recon = invert(et, args.num_steps); secs = round(time.time() - t0, 1)
        m = score(recon)
        club = club_mi_upper_bound(e_noisy, e0n, max_rows=min(600, n), seed=0)["club_mi_bits"]
        try:
            pdim = min(64, max(2, n // 4))
            capv = v_information_capacity(e_noisy, labels, family="pca_softmax", dim=pdim, l2=0.1)["reader_top1_acc"]
        except Exception as e:
            print(f"[v2t-dp] capPVI skipped: {e}", flush=True); capv = float("nan")
        rec = {"epsilon": (None if math.isinf(eps) else eps), "sigma": sigma, "attack": f"vec2text_{args.num_steps}step_beam{args.beam}",
               **m, "club_bits": club, "cap_pvi_acc": capv, "secs": secs}
        records.append(rec)
        es = "inf" if math.isinf(eps) else f"{eps:g}"
        print(f"[v2t-dp] ε={es:>5} σ={sigma:.3f} | bleu={m['bleu']:.1f} tF1={m['token_f1']:.3f} exact={m['exact']:.3f} "
              f"cos={m['cos']:.3f} | CLUB={club:.1f}b capPVI={capv:.3f} [{secs}s]", flush=True)

    def sp(a, b):
        from scipy import stats
        a, b = np.asarray(a, float), np.asarray(b, float)
        return 0.0 if np.std(a) < 1e-9 or np.std(b) < 1e-9 else float(stats.spearmanr(a, b).statistic)
    sweep = [r for r in records if r["attack"].startswith("vec2text")]
    corr = {m: {"club": sp([r[m] for r in sweep], [r["club_bits"] for r in sweep]),
                "capPVI": sp([r[m] for r in sweep], [r["cap_pvi_acc"] for r in sweep])}
            for m in ("token_f1", "exact", "cos", "bleu")}
    print("\n[v2t-dp] C3 Spearman(recovery, probe) over ε:")
    for m in ("token_f1", "exact", "cos"):
        print(f"   {m:9s}: CLUB={corr[m]['club']:+.2f}  capPVI={corr[m]['capPVI']:+.2f}")
    out = {"n": n, "clip_C": C, "num_steps": args.num_steps, "beam": args.beam,
           "epsilons": [None if math.isinf(e) else e for e in eps_list], "recorrelation": corr, "records": records}
    os.makedirs(os.path.join(REPO, "results"), exist_ok=True)
    open(os.path.join(REPO, args.out), "w").write(json.dumps(out, indent=2))
    print(f"[v2t-dp] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
