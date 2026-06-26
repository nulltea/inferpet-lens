#!/usr/bin/env python3
"""AloePri token-id surface (Surface B): TFMA τ-recovery vs the M1 perturbation budget ε1.

Unlike Surface A (residual activations, where AloePri is access-control), the token-id surface is a
genuine bits↔recovery sweep. The deterministic token substitution Π (secret map τ over the vocab) is
a monoalphabetic substitution cipher: its unigram-frequency fingerprint lets the Token-Frequency-
Matching Attack (TFMA) recover τ — UNLESS the online perturbation M1 (randomized response, the
exponential mechanism with the transposition metric, §6.2) blurs it. We sweep ε1 and report:

  recovery_topk   — TFMA Top-k token-recovery success rate (the attack; matches obf-stream unigram
                    freqs to a public reference table).  [paper Table 5: Top-100 ≤ 20%]
  mi_bits         — I(released ; true) per token in bits, from the GROUND-TRUTH joint under M1 (the
                    attack-INDEPENDENT leakage probe; not TFMA's output). H(true) is the ceiling.
  keep_p0         — 1/(1+(V−1)e^{−ε1}), the M1 keep-probability.

Reference frequencies default to the SAME corpus (distribution-aware = the strongest attacker, the
TFMA upper bound); pass --ref-corpus for the zero-knowledge / domain-aware settings.

GPU-free (tokenize + count); needs only the tokenizer, so run via scripts/run_in_rocm.sh for the deps.

Example:
  scripts/run_in_rocm.sh python3 scripts/evals/aloepri_tokenid_sweep.py \
      --corpus corpora/release-gate-512.txt --eps1 inf,16,14,12,11,10,8 \
      --out refine-logs/aloepri/aloepri_tokenid_sweep.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ for defenses.*

from defenses.aloepri import m1_randomized_response  # noqa: E402
from talens.attacks.token_frequency import sda_recover, tfma_recover  # noqa: E402


def _tokenize(corpus, tok, max_prompts):
    lines = [l.strip() for l in Path(corpus).read_text().splitlines() if l.strip()][:max_prompts]
    ids = []
    for line in lines:
        ids.extend(tok(line, add_special_tokens=False).input_ids)
    return np.asarray(ids, dtype=np.int64)


def _mi_bits(true, V, p0):
    """I(true; released) in bits, ANALYTIC for the known randomized-response channel M1 (NOT the
    plug-in empirical joint — with V≫n the plug-in estimator is hugely upward-biased: every
    substituted pair is unique and reads as deterministic). Depends only on the mechanism (p0) and
    the empirical true marginal → genuinely attack-independent. Returns (I bits, H(true) bits).

      released = true w.p. p0, else UNIFORM over the other V−1 tokens.
      H(R|T) = H_b(p0) + (1−p0)·log2(V−1)            (same for every input token)
      p_R(r) = p0·p_T(r) + (1−p0)/(V−1)·(1−p_T(r))   (released marginal)
      I = H(R) − H(R|T)
    """
    n = true.shape[0]
    p_t = np.bincount(true, minlength=V).astype(float) / n
    h_true = float(-np.sum(p_t[p_t > 0] * np.log2(p_t[p_t > 0])))
    if p0 >= 1.0:
        return h_true, h_true                              # no perturbation: I = H(true)
    hb = 0.0 if p0 in (0.0, 1.0) else -(p0 * math.log2(p0) + (1 - p0) * math.log2(1 - p0))
    h_r_given_t = hb + (1 - p0) * math.log2(V - 1)
    p_r = p0 * p_t + (1 - p0) / (V - 1) * (1 - p_t)
    h_r = float(-np.sum(p_r * np.log2(p_r)))               # over all V tokens (unseen carry tiny mass)
    return max(0.0, h_r - h_r_given_t), h_true


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--corpus", default="corpora/release-gate-512.txt")
    ap.add_argument("--ref-corpus", default="", help="public reference for TFMA freqs; empty = same "
                    "corpus (distribution-aware, the strongest attacker / TFMA upper bound)")
    ap.add_argument("--max-prompts", type=int, default=512)
    ap.add_argument("--eps1", default="inf,16,14,12,11,10,8", help="comma list; 'inf' = no M1 (pure Π)")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--sda-iters", type=int, default=4000, help="SDA bigram hill-climb iterations")
    ap.add_argument("--tau-seed", type=int, default=0, help="secret token-permutation τ seed")
    ap.add_argument("--m1-seed", type=int, default=20260626)
    ap.add_argument("--out", default="refine-logs/aloepri/aloepri_tokenid_sweep.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    V = len(tok)
    true = _tokenize(args.corpus, tok, args.max_prompts)
    ref_stream = true if not args.ref_corpus else _tokenize(args.ref_corpus, tok, args.max_prompts)
    ref_freq = np.bincount(ref_stream, minlength=V).astype(float)
    rng = np.random.default_rng(args.tau_seed)
    tau = rng.permutation(V)                                     # secret token substitution τ
    eps_list = [math.inf if s.strip().lower().startswith("inf") else float(s)
                for s in args.eps1.split(",") if s.strip()]
    print(f"[tokenid] model={args.model} V={V} tokens={true.size} ref={'same' if not args.ref_corpus else args.ref_corpus} "
          f"eps1={eps_list} top_k={args.top_k}", flush=True)

    records = []
    for eps1 in eps_list:
        released = m1_randomized_response(true, vocab=V, eps1=eps1, seed=args.m1_seed)
        obf = tau[released]                                      # released → obfuscated id via Π
        rec = tfma_recover(obf, ref_freq, tau_true=tau, top_k=args.top_k)["recovery_topk"]
        sda = sda_recover(obf, ref_stream, tau_true=tau, top_k=args.top_k,
                          n_iters=args.sda_iters)["recovery_topk"]
        p0 = 1.0 if math.isinf(eps1) else 1.0 / (1.0 + (V - 1) * math.exp(-eps1))
        mi, h_true = _mi_bits(true, V, p0)
        r = {"eps1": (None if math.isinf(eps1) else eps1), "keep_p0": p0,
             "tfma_recovery_topk": rec, "sda_recovery_topk": sda, "mi_bits": mi, "h_true_bits": h_true}
        records.append(r)
        es = "inf" if math.isinf(eps1) else f"{eps1:g}"
        print(f"[tokenid] ε1={es:>4} p0={p0:.3f} | TFMA top{args.top_k}={rec:.3f} SDA={sda:.3f} | "
              f"I(rel;true)={mi:.2f}b (H={h_true:.2f}b)", flush=True)

    # does TFMA recovery track the independent MI probe across the ε1 sweep?
    xs = [r["tfma_recovery_topk"] for r in records]
    ys = [r["mi_bits"] for r in records]
    rho = None
    if len(xs) >= 2 and np.std(xs) > 1e-9 and np.std(ys) > 1e-9:
        ra, rb = np.argsort(np.argsort(xs)), np.argsort(np.argsort(ys))
        rho = float(np.corrcoef(ra, rb)[0, 1])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "model": args.model, "corpus": args.corpus, "ref_corpus": args.ref_corpus or "(same)",
        "vocab": V, "n_tokens": int(true.size), "surface": "token_id_api",
        "defense": "aloepri_pi_plus_m1_randomized_response", "top_k": args.top_k,
        "note": "M1 = exponential mech (transposition metric) → per-token randomized response, keep "
                "p0=1/(1+(V-1)e^-eps1). TFMA matches obf-stream unigram freqs to the reference. mi_bits "
                "= I(released;true) from the ground-truth joint (attack-independent probe). ref=same "
                "corpus is distribution-aware (strongest attacker / TFMA upper bound).",
        "tfma_recovery_vs_mi_spearman": rho, "records": records,
    }, indent=2))
    print(f"[tokenid] recovery↔MI Spearman={rho} ; wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
