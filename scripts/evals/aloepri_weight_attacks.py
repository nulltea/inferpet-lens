#!/usr/bin/env python3
"""AloePri weight-surface attacks (static obfuscated weights): VMA · IA · IMA-EmbedRow-ridge.

The honest-but-curious server holds the obfuscated weights θ̃ but not the key. These attacks try to
recover the secret token permutation τ (VMA/IA) or invert embedding rows to tokens (IMA) from the
(public plaintext, obfuscated) weight pair. We obfuscate the REAL pythia embedding (with Π — the
secret these attacks target) + the gate/query weights covariantly, and sweep αₑ:

  W̃e = Π·(We + αₑ·σₑ·ε)·P̂        (embedding, dim d→d+2h)
  W̃_gate = W_gate · Q̂ᵀ ; W̃_q = W_q · Q̂ᵀ   (read the obfuscated residual; keymat cancels covariantly)

Attacks (TTRSR = token-recovery success ratio; release gate ≤15% at paper hyperparameters):
  VMA  — RowSort + Hungarian on the sorted-quantile row signature → τ. Defeated by P̂'s dense mixing.
  Gate-IA — match the per-token gate-projection row-mean invariant (survives keymat; broken by αₑ noise).
  Attn-IA — match the per-token attn quadratic form eᵢ(W_qᵀW_q)⁻¹eᵢᵀ (broken by Alg-2 head perms; here
            also stressed by the d→d+2h rank-deficiency of the expanded basis).
  IMA-EmbedRow-ridge — PAIRED-DATA attacker: fit ridge (obf row → plaintext embedding) on known
            (obf-row, token) pairs, recover the rest. With no noise ridge learns P̂⁻¹ in closed form
            → keymat alone gives NO protection (the paper's headline gap); αₑ noise is the defense.

GPU-light (weight algebra); needs the model weights → run via scripts/run_in_rocm.sh.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defenses.aloepri import keymat_gen, obf_read_weight  # noqa: E402
from talens.attacks.dp_inversion import nearest_token, ridge_W  # noqa: E402
from talens.weights import vma  # noqa: E402
from talens.weights.invariant_attack import attn_ia_scalar, gate_ia_scalar, recover_by_invariant  # noqa: E402
from talens.weights.types import WeightPair  # noqa: E402

import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--n-tokens", type=int, default=4000, help="subset of vocab rows attacked (speed)")
    ap.add_argument("--alpha-es", default="0,0.5,1.0,2.0")
    ap.add_argument("--keymat-h", type=int, default=128)
    ap.add_argument("--keymat-lam", type=float, default=0.3)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--vma-bins", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260626)
    ap.add_argument("--out", default="refine-logs/aloepri/aloepri_weight_attacks.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).eval()
    net = model.gpt_neox
    rng = np.random.default_rng(args.seed)
    We_full = net.embed_in.weight.detach().float().numpy().astype(np.float64)
    vocab, d = We_full.shape
    sel = rng.choice(vocab, size=min(args.n_tokens, vocab), replace=False)
    We = We_full[sel].copy()
    N = We.shape[0]
    W_gate = net.layers[0].mlp.dense_h_to_4h.weight.detach().float().numpy().astype(np.float64)  # (4d, d)
    qkv = net.layers[0].attention.query_key_value.weight.detach().float().numpy().astype(np.float64)
    W_q = qkv[:d, :]                                            # query slice (d, d)
    alpha_es = [float(s) for s in args.alpha_es.split(",") if s.strip()]
    print(f"[wattack] model={args.model} vocab={vocab} d={d} N={N} h={args.keymat_h} alphas={alpha_es}", flush=True)

    P_np, Q_np = keymat_gen(d, args.keymat_h, lam=args.keymat_lam, seed=args.seed)
    P, Q = P_np.astype(np.float64), Q_np.astype(np.float64)
    Pt, Qt = torch.from_numpy(P), torch.from_numpy(Q)
    Wg_obf = obf_read_weight(torch.from_numpy(W_gate), Qt).numpy()   # W_gate · Q̂ᵀ  (4d, d+2h)
    Wq_obf = obf_read_weight(torch.from_numpy(W_q), Qt).numpy()      # W_q · Q̂ᵀ      (d, d+2h)
    sig_e = float(We.std())

    # plaintext invariants (computed once)
    a_plain = gate_ia_scalar(We, W_gate)
    b_plain = attn_ia_scalar(We, W_q)
    ntr = int(0.5 * N)
    tr, te = np.arange(ntr), np.arange(ntr, N)

    records = []
    for ae in alpha_es:
        noisy = We + (ae * sig_e * rng.standard_normal(We.shape)) if ae > 0 else We
        tau = rng.permutation(N)                               # secret token permutation Π
        Wobf = np.empty((N, d + 2 * args.keymat_h), dtype=np.float64)
        Wobf[tau] = noisy @ P                                  # W̃e[τ[i]] = (We[i]+noise)·P̂

        # VMA (sorted-quantile RowSort + Hungarian)
        pair = WeightPair(plain=We.astype(np.float32), obf=Wobf.astype(np.float32),
                          perm=tau.astype(np.int64), token_ids=np.asarray(sel, np.int64),
                          kind="embed", model_id="aloepri-pythia")
        vma_rec = float(vma.run(pair, bins=args.vma_bins, match="hungarian").ttrsr_top1)

        # IA: obfuscated invariants from the obf weights (keymat cancels covariantly)
        a_obf = gate_ia_scalar(Wobf, Wg_obf)
        b_obf = attn_ia_scalar(Wobf, Wq_obf)
        gate_rec = recover_by_invariant(a_plain, a_obf, tau_true=tau, top_k=args.top_k)
        attn_rec = recover_by_invariant(b_plain, b_obf, tau_true=tau, top_k=args.top_k)

        # IMA-EmbedRow-ridge: paired-data ridge (obf row → plaintext embedding), recover test tokens
        Xtr, Etr = Wobf[tau[tr]], We[tr]                       # known (obf-row, token) pairs
        Wmap = ridge_W(Xtr.astype(np.float32), Etr.astype(np.float32), alpha=1.0)
        pred = Wobf[tau[te]].astype(np.float32) @ Wmap
        yhat = nearest_token(pred, We[te].astype(np.float32), np.arange(te.size))
        ima_rec = float((yhat == np.arange(te.size)).mean())

        rec = {"alpha_e": ae, "vma": vma_rec, "gate_ia": gate_rec, "attn_ia": attn_rec,
               "ima_embedrow_ridge": ima_rec}
        records.append(rec)
        gate = {k: ("PASS" if v <= 0.15 else "FAIL") for k, v in rec.items() if k != "alpha_e"}
        print(f"[wattack] αe={ae:<4} | VMA={vma_rec:.3f} Gate-IA={gate_rec:.3f} Attn-IA={attn_rec:.3f} "
              f"IMA-ridge={ima_rec:.3f} | gate(≤.15): {gate}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "model": args.model, "surface": "static_obfuscated_weights", "vocab": vocab, "d": d,
        "n_tokens": N, "keymat": {"h": args.keymat_h, "lam": args.keymat_lam}, "top_k": args.top_k,
        "release_gate_ttrsr": 0.15,
        "note": "VMA/Gate-IA/Attn-IA recover the secret token permutation τ; IMA-EmbedRow-ridge is a "
                "paired-data inverter (learns P̂⁻¹ in closed form when αₑ=0 → keymat alone no protection). "
                "TTRSR = token-recovery success ratio; ≤0.15 passes the paper §6.3 gate.",
        "records": records,
    }, indent=2))
    print(f"[wattack] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
