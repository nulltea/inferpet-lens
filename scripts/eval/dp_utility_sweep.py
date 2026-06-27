#!/usr/bin/env python3
"""Utility side of the DP privacy–utility tradeoff: how much does LocalDP (Gaussian noise on the
input embedding, propagated through the model) degrade the model's *output*, across an ε sweep.

Model-agnostic + scale-invariant by construction: every metric is referenced to the model's own
non-private (ε=∞) baseline, so 160M and 1.4B are directly comparable. Headline follows the DP-LLM
literature (Yu et al. 2021 arXiv:2110.06500; Li et al. 2021 arXiv:2110.05679; DP-Forward, Du et al.
2023 arXiv:2309.06746): utility = task performance vs the non-private baseline.

  perplexity            ppl(ε) = exp(mean teacher-forced next-token CE on held-out text)
  acc                   next-token top-1 accuracy vs the GROUND-TRUTH token (a task-agnostic accuracy)
  retention_acc         acc(ε) / acc(∞)            ← bounded [0,1]; −10/−20/−50% = 0.90/0.80/0.50
  ppl_degradation       ppl(ε) / ppl(∞) − 1        ← DP generation-utility convention
  agree_clean           P[argmax_noised = argmax_clean]  (self-consistency; quantization-eval style)

Teacher-forced, one forward per ε (no generation, no draws, no hidden-state storage) → cheap.

  scripts/run_in_rocm.sh python3 scripts/eval/dp_utility_sweep.py \
      --corpus corpora/rep2text-stratified.txt --max-prompts 1800 \
      --epsilons inf,512,256,128,64,32,16,8 --out refine-logs/pythia-depth/dp_utility.json
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "scripts")
from defenses.local_dp import LocalDP  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def utility_pass(model, tok, prompts, sigma, C, batch_size, clean_argmax=None):
    """One teacher-forced forward sweep at a given σ. Returns (mean_ce_nats, acc, n_tokens,
    argmax_flat, agree_clean). LocalDP hook active when σ>0 (σ=0 ⇒ clip-only ≈ clean)."""
    hk = model.get_input_embeddings().register_forward_hook(LocalDP(C, sigma))
    ce_sum, n_tok, n_correct, n_agree = 0.0, 0, 0, 0
    argmax_flat = []
    ci = 0  # cursor into clean_argmax
    try:
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True)
                ids, mask = enc.input_ids.to(DEV), enc.attention_mask.to(DEV)
                logits = model(ids, attention_mask=mask, use_cache=False).logits.float()
                pred = logits[:, :-1, :]                       # predict token t+1 from ≤t
                tgt = ids[:, 1:]
                m = mask[:, 1:].bool()                         # target position is a real token
                lp = torch.log_softmax(pred, dim=-1)
                tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                am = pred.argmax(dim=-1)
                ce_sum += float(-(tok_lp[m]).sum())
                n_correct += int((am[m] == tgt[m]).sum())
                n_tok += int(m.sum())
                af = am[m].cpu().numpy()
                argmax_flat.append(af)
                if clean_argmax is not None:                   # self-consistency vs clean run
                    n_agree += int((af == clean_argmax[ci:ci + af.size]).sum())
                    ci += af.size
    finally:
        hk.remove()
    argmax_flat = np.concatenate(argmax_flat) if argmax_flat else np.array([], dtype=np.int64)
    mean_ce = ce_sum / max(1, n_tok)
    acc = n_correct / max(1, n_tok)
    agree = (n_agree / max(1, n_tok)) if clean_argmax is not None else None
    return mean_ce, acc, n_tok, argmax_flat, agree


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--corpus", default="corpora/rep2text-stratified.txt")
    ap.add_argument("--max-prompts", type=int, default=1800)
    ap.add_argument("--epsilons", default="inf,512,256,128,64,32,16,8")
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--clip-percentile", type=float, default=99.9)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="refine-logs/pythia-depth/dp_utility.json")
    args = ap.parse_args()

    eps_list = [math.inf if s.strip().lower().startswith("inf") else float(s)
                for s in args.epsilons.split(",") if s.strip()]
    if not any(math.isinf(e) for e in eps_list):
        eps_list = [math.inf] + eps_list                       # ∞ baseline is mandatory
    eps_list = sorted(set(eps_list), key=lambda e: (math.inf if math.isinf(e) else e), reverse=True)  # ∞ first

    from transformers import AutoModelForCausalLM, AutoTokenizer
    prompts = [l.strip() for l in Path(args.corpus).read_text().splitlines() if l.strip()][: args.max_prompts]
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, attn_implementation="eager", device_map=DEV).eval()

    # clip C from embedding norms (same convention as dp_leakage_sweep: clip-only ≈ clean)
    cal = []
    h = model.get_input_embeddings().register_forward_hook(
        lambda m, i, o: cal.append(o.float().norm(dim=-1).flatten().cpu()))
    try:
        with torch.no_grad():
            for p in prompts[:48]:
                model(tok(p, return_tensors="pt").input_ids.to(DEV), use_cache=False)
    finally:
        h.remove()
    C = float(np.percentile(torch.cat(cal).numpy(), args.clip_percentile))
    z = math.sqrt(2 * math.log(1.25 / args.delta))
    print(f"[util] model={args.model} C={C:.3f} z={z:.3f} eps={eps_list} prompts={len(prompts)} dev={DEV}", flush=True)

    recs, clean_argmax, base_ppl, base_acc = [], None, None, None
    for eps in eps_list:
        sigma = 0.0 if math.isinf(eps) else C * z / eps
        ce, acc, ntok, am, agree = utility_pass(model, tok, prompts, sigma, C, args.batch_size, clean_argmax)
        ppl = math.exp(ce)
        if math.isinf(eps):
            clean_argmax, base_ppl, base_acc = am, ppl, acc
        rec = {"epsilon": (None if math.isinf(eps) else eps), "sigma": sigma, "n_tokens": ntok,
               "perplexity": ppl, "acc": acc,
               "retention_acc": acc / base_acc if base_acc else None,
               "ppl_degradation": ppl / base_ppl - 1.0 if base_ppl else None,
               "agree_clean": agree}
        recs.append(rec)
        es = "inf" if math.isinf(eps) else f"{eps:g}"
        print(f"[util] ε={es:>5} σ/C={(0 if math.isinf(eps) else z/eps):.3f} | ppl={ppl:8.2f} "
              f"acc={acc:.4f} retention={rec['retention_acc']:.3f} pplΔ={rec['ppl_degradation']:+.2%} "
              f"agree_clean={'—' if agree is None else f'{agree:.4f}'}", flush=True)

    # locate the ε crossings for −10/−20/−50% accuracy retention (linear interp on log ε)
    thr = {}
    noised = [r for r in recs if r["epsilon"] is not None]
    for target in (0.90, 0.80, 0.50):
        cross = None
        for a, b in zip(noised, noised[1:]):  # noised is descending ε (retention falling)
            ra, rb = a["retention_acc"], b["retention_acc"]
            if ra is not None and rb is not None and ra >= target >= rb:
                la, lb = math.log(a["epsilon"]), math.log(b["epsilon"])
                t = (ra - target) / (ra - rb) if ra != rb else 0.0
                cross = math.exp(la + t * (lb - la))
                break
        thr[f"retention_{int(target*100)}pct"] = cross
    print(f"[util] ε for −10/−20/−50% acc-retention: {thr}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "model": args.model, "corpus": args.corpus, "n_prompts": len(prompts), "clip_C": C, "z": z,
        "utility_note": "teacher-forced next-token; perplexity + ground-truth top-1 acc referenced to "
                        "the ε=∞ baseline (DP-LLM convention); agree_clean = self-consistency vs clean.",
        "epsilon_thresholds": thr, "records": recs}, indent=2))
    print(f"[util] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
