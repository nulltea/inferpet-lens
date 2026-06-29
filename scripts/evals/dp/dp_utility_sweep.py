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

  scripts/run_in_rocm.sh python3 scripts/evals/dp_utility_sweep.py \
      --corpus corpora/rep2text-stratified.txt --max-prompts 1800 \
      --epsilons inf,512,256,128,64,32,16,8 --out refine-logs/pythia-depth/dp_utility.json
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))   # talens.*
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))           # scripts/ for defenses.*
from defenses.local_dp import LocalDP  # noqa: E402
from talens.probes.utility import (  # noqa: E402
    teacher_forced_pass, next_token_accuracy, perplexity, output_agreement, retention_thresholds)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


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

    # clean baseline = clip-only (σ=0) pass, captured once; all retention is referenced to it.
    clean = teacher_forced_pass(model, tok, prompts, hook=LocalDP(C, 0.0), batch_size=args.batch_size)
    recs = []
    for eps in eps_list:
        sigma = 0.0 if math.isinf(eps) else C * z / eps
        dp = clean if sigma == 0 else teacher_forced_pass(
            model, tok, prompts, hook=LocalDP(C, sigma), batch_size=args.batch_size)
        acc_r = next_token_accuracy(dp, clean)        # standardized utility probes (talens.probes.utility)
        ppl_r = perplexity(dp, clean)
        agr_r = output_agreement(dp, clean)
        rec = {"epsilon": (None if math.isinf(eps) else eps), "sigma": sigma,
               "n_tokens": acc_r.extra["n_tokens"],
               "perplexity": ppl_r.defended, "acc": acc_r.defended,
               "retention_acc": acc_r.retention, "ppl_degradation": ppl_r.extra["degradation"],
               "agree_clean": agr_r.defended,
               "utility": {r.metric: r.as_dict() for r in (acc_r, ppl_r, agr_r)}}
        recs.append(rec)
        es = "inf" if math.isinf(eps) else f"{eps:g}"
        print(f"[util] ε={es:>5} σ/C={(0 if math.isinf(eps) else z/eps):.3f} | ppl={ppl_r.defended:8.2f} "
              f"acc={acc_r.defended:.4f} retention={acc_r.retention:.3f} pplΔ={ppl_r.extra['degradation']:+.2%} "
              f"agree_clean={agr_r.defended:.4f}", flush=True)

    noised = [r for r in recs if r["epsilon"] is not None]
    thr = retention_thresholds([r["epsilon"] for r in noised], [r["retention_acc"] for r in noised])
    print(f"[util] ε for −10/−20/−50% acc-retention: {thr}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "model": args.model, "corpus": args.corpus, "n_prompts": len(prompts), "clip_C": C, "z": z,
        "defense": "local_dp_gaussian",
        "utility_note": "teacher-forced next-token; standardized talens.probes.utility (retention∈[0,1], "
                        "1=lossless): next_token_accuracy + perplexity + output_agreement, referenced to "
                        "the clip-only (σ=0) baseline. Comparable across schemes via 'retention'.",
        "epsilon_thresholds": thr, "records": recs}, indent=2))
    print(f"[util] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
