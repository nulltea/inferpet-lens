#!/usr/bin/env python3
"""B2-propagated — stronger decoder vs ridge under PROPAGATED input-DP at depth.

The open regime (B3 L20): input-DP noise injected at the EMBEDDING and reshaped
through the transformer makes ridge's linear obs->emb map fail AND decorrelate from
MI (ρ(ridge-recovery, capPVI/CLUB) went negative at L20). Hypothesis: a channel-aware
nonlinear decoder trained on the propagated-noised resid recovers more AND
re-correlates with the MI probes where ridge anti-correlates.

Captures propagated resid (forward with the embedding-DP hook) at L12/L20 across a
few ε, then on each (L,ε): ridge / channel-aware decoder / capacity-PVI reader, with
vocab-disjoint + shuffle-control selectivity, + CLUB/capPVI MI probes. Reports
Spearman(recovery-selectivity, MI-probe) over ε for ridge vs decoder.

Single GPU process; run via scripts/run_in_rocm.sh. Sized small (few ε, 2 layers).
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np, torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b2_lpos_decoder import train_decoder, decode_match, ridge_match  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from talens.probes.club import club_mi_upper_bound  # noqa: E402
from talens.probes.vinfo_capacity import v_information_capacity  # noqa: E402

EMBED = "results/capture_cache/embed-b0c6566474cadb27.pt"


class InputDPCover:
    def __init__(self, C, sigma): self.C, self.sigma = C, sigma
    def __call__(self, mod, inp, out):
        f = out.float(); n = f.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        f = f * (self.C / n).clamp_max(1.0)
        if self.sigma > 0: f = f + self.sigma * torch.randn_like(f)
        return f.to(out.dtype)


@torch.no_grad()
def capture(model, tok, prompts, layers, device):
    per = {L: [] for L in layers}; ids = []
    for p in prompts:
        i = tok(p, return_tensors="pt").input_ids.to(device)
        hs = model(i, output_hidden_states=True, use_cache=False).hidden_states
        for L in layers: per[L].append(hs[L + 1][0].float().cpu().numpy())
        ids.append(i[0].cpu().numpy())
    return per, ids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="unsloth/gemma-2-2b")
    ap.add_argument("--corpus", default="corpora/release-gate-512.txt")
    ap.add_argument("--max-prompts", type=int, default=160)
    ap.add_argument("--layers", default="12,20")
    ap.add_argument("--epsilons", default="inf,1024,512,256")
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--clip-percentile", type=float, default=99.9)
    ap.add_argument("--pool-size", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--club-max-rows", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20260621)
    ap.add_argument("--out", default="results/b2_propagated_dp.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = [int(s) for s in args.layers.split(",") if s.strip()]
    eps_list = [math.inf if s.strip().lower().startswith("inf") else float(s) for s in args.epsilons.split(",") if s.strip()]
    prompts = [l.strip() for l in Path(args.corpus).read_text().splitlines() if l.strip()][: args.max_prompts]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map=device).eval()
    table = model.get_input_embeddings().weight.detach().float().cpu().numpy().astype(np.float32)
    vocab = table.shape[0]
    # clip C from runtime embed norms
    cal = []
    h = model.model.embed_tokens.register_forward_hook(lambda m, i, o: cal.append(o.float().norm(dim=-1).flatten().cpu()))
    with torch.no_grad():
        for p in prompts[:48]: model(tok(p, return_tensors="pt").input_ids.to(device), use_cache=False)
    h.remove()
    C = float(np.percentile(torch.cat(cal).numpy(), args.clip_percentile)); z = math.sqrt(2 * math.log(1.25 / args.delta))
    rng = np.random.default_rng(args.seed)
    print(f"[prop] C={C:.3f} layers={layers} eps={eps_list} prompts={len(prompts)}", flush=True)

    def stack(per_L, ids):
        Xs, ys = [], []
        for m, t in zip(per_L, ids):
            n = min(m.shape[0], t.shape[0]); Xs.append(m[:n]); ys.append(t[:n])
        return np.concatenate(Xs, 0), np.concatenate(ys, 0).astype(np.int64)

    # capture clean once (for shuffle floor + split definition)
    perc, idc = capture(model, tok, prompts, layers, device)
    records = []
    split = {}
    for L in layers:
        X0, y = stack(perc[L], idc)
        distinct = rng.permutation(np.unique(y)); ntr = int(0.7 * distinct.size)
        tr_ids = set(distinct[:ntr].tolist()); te_ids = set(distinct[ntr:].tolist())
        tr = np.array([i for i, t in enumerate(y) if t in tr_ids]); te = np.array([i for i, t in enumerate(y) if t in te_ids])
        true_pool = np.unique(y[te]); avail = np.setdiff1d(np.arange(vocab, dtype=np.int64), true_pool)
        fill = rng.choice(avail, size=max(0, args.pool_size - true_pool.size), replace=False)
        pool = np.concatenate([true_pool, fill.astype(np.int64)])
        emb_y = table[y]; permsh = rng.permutation(tr.size)
        rsh = float((ridge_match(X0[tr], emb_y[tr][permsh], X0[te], table[pool], pool) == y[te]).mean())
        dsh = train_decoder(X0[tr], emb_y[tr][permsh], hidden=args.hidden, epochs=args.epochs, seed=args.seed)
        csh = float((decode_match(dsh, X0[te], table[pool], pool) == y[te]).mean())
        split[L] = (y, tr, te, pool, emb_y, rsh, csh)

    for eps in eps_list:
        sigma = 0.0 if math.isinf(eps) else C * z / eps
        torch.manual_seed(args.seed + (0 if math.isinf(eps) else int(eps)))
        hk = model.model.embed_tokens.register_forward_hook(InputDPCover(C, sigma))
        per, _ = capture(model, tok, prompts, layers, device); hk.remove()
        for L in layers:
            X, _ = stack(per[L], idc); y, tr, te, pool, emb_y, rsh, csh = split[L]
            pe = table[pool]
            ridge_t = float((ridge_match(X[tr], emb_y[tr], X[te], pe, pool) == y[te]).mean())
            dec = train_decoder(X[tr], emb_y[tr], hidden=args.hidden, epochs=args.epochs, seed=args.seed)
            dec_t = float((decode_match(dec, X[te], pe, pool) == y[te]).mean())
            capv = v_information_capacity(X, y, family="pca_softmax", dim=64, l2=0.1)["reader_top1_acc"]
            club = club_mi_upper_bound(X, emb_y, max_rows=args.club_max_rows, seed=0)["club_mi_bits"]
            rec = {"epsilon": (None if math.isinf(eps) else eps), "layer": L, "sigma": sigma,
                   "ridge": ridge_t, "decoder": dec_t, "ridge_sel": ridge_t - rsh, "dec_sel": dec_t - csh,
                   "uplift_sel": (dec_t - csh) - (ridge_t - rsh), "cap_pvi_acc": capv, "club_bits": club}
            records.append(rec)
            es = "inf" if math.isinf(eps) else f"{eps:g}"
            print(f"[prop] ε={es:>5} L{L:>2} | ridge={ridge_t:.3f}(sel{rec['ridge_sel']:+.3f}) "
                  f"dec={dec_t:.3f}(sel{rec['dec_sel']:+.3f}) upliftSel{rec['uplift_sel']:+.3f} "
                  f"| capPVI={capv:.3f} club={club:.0f}", flush=True)

    def sp(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return 0.0 if np.std(a) < 1e-9 or np.std(b) < 1e-9 else float(stats.spearmanr(a, b).statistic)
    corr = {}
    for L in layers:
        r = [x for x in records if x["layer"] == L]
        corr[f"L{L}"] = {"ridgeSel_vs_capPVI": sp([x["ridge_sel"] for x in r], [x["cap_pvi_acc"] for x in r]),
                         "decSel_vs_capPVI": sp([x["dec_sel"] for x in r], [x["cap_pvi_acc"] for x in r]),
                         "ridgeSel_vs_club": sp([x["ridge_sel"] for x in r], [x["club_bits"] for x in r]),
                         "decSel_vs_club": sp([x["dec_sel"] for x in r], [x["club_bits"] for x in r])}
    print("\n[prop] re-correlation over ε (Spearman selectivity↔MI-probe), per layer:")
    for L in layers:
        c = corr[f"L{L}"]
        print(f"   L{L}: ridgeSel↔capPVI={c['ridgeSel_vs_capPVI']:+.2f} decSel↔capPVI={c['decSel_vs_capPVI']:+.2f} | "
              f"ridgeSel↔CLUB={c['ridgeSel_vs_club']:+.2f} decSel↔CLUB={c['decSel_vs_club']:+.2f}")
    Path(args.out).write_text(json.dumps({"layers": layers, "epsilons": [None if math.isinf(e) else e for e in eps_list],
                              "recorrelation": corr, "records": records}, indent=2))
    print(f"[prop] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
