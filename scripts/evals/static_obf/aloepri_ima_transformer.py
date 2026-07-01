#!/usr/bin/env python3
"""AloePri IMA-EmbedRow-transformer (paper §F.1) on the static obfuscated embedding table.

The canonical IMA: a 2-layer / 8-head transformer inverter (talens.attacks.dp_inversion.IMAInverter)
trained on (obfuscated-row → plain-embedding) pairs, decode by nearest-token. Two threat models:

  plain-control (matched, single key) — train + test on the SAME deployment key, row-split. A common
    inverse exists, so this MUST recover (validates the inverter; the control private-rag's driver failed).
  tau-invariant (in-model) — the attacker generates K synthetic own-key obfuscations of the public table
    (own pseudo-τ / P̂ / αₑ, Kerckhoffs) and trains the inverter on those, then decodes the DEPLOYMENT
    table (unseen key, held-out tokens). Recovers only if a key-invariant inverse exists; AloePri's dense
    keymat has none, so the paper reports IMA ≈ 0 % — this measures that on the artifact.

Swept over the αₑ defence level (keymat=0, alg1@x). GPU: ONE process; run via scripts/run_in_rocm.sh.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from defenses.aloepri import obfuscate_embedding_table  # noqa: E402
from talens.attacks import ima_transformer_attack  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _obf_rows(W, alpha_e, h, lam, seed):
    """Token-aligned obfuscated embedding rows: out[i] = obfuscated row of plain token i."""
    wp = obfuscate_embedding_table(W, alpha_e=alpha_e, h=h, lam=lam, keymat=True, seed=seed)
    return wp.obf[wp.perm].astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--configs", default="keymat_only,full_alg1@1.0,full_alg1@2.0")
    ap.add_argument("--n-eval", type=int, default=12000, help="token rows used (subsample of vocab); the 768-d "
                    "inverter needs ample rows to pass the plain control — small sets underfit (≈0).")
    ap.add_argument("--n-keys", type=int, default=4, help="K synthetic attacker keymats for τ-invariant training")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--pool-size", type=int, default=2048)
    ap.add_argument("--keymat-h", type=int, default=128)
    ap.add_argument("--keymat-lam", type=float, default=0.3)
    ap.add_argument("--keymat-seed", type=int, default=0, help="deployment key seed")
    ap.add_argument("--attacker-seed", type=int, default=700000)
    ap.add_argument("--seed", type=int, default=20260629)
    ap.add_argument("--out", default="refine-logs/aloepri/aloepri_ima_transformer.json")
    args = ap.parse_args()

    def cfg_alpha(c):
        if c == "keymat_only":
            return 0.0
        if c.startswith("full_alg1"):
            return float(c.split("@", 1)[1]) if "@" in c else 1.0
        raise ValueError(c)

    from transformers import AutoModelForCausalLM
    W = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).get_input_embeddings(
        ).weight.detach().float().cpu().numpy().astype(np.float32)
    vocab, d = W.shape
    rng = np.random.default_rng(args.seed)
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    sel = rng.choice(vocab, min(args.n_eval, vocab), replace=False)
    ntr = int(0.7 * sel.size)
    tr_tok, te_tok = sel[:ntr], sel[ntr:]                    # vocab-disjoint train/test tokens
    pool = np.concatenate([np.unique(te_tok),
                           rng.choice(np.setdiff1d(np.arange(vocab), te_tok),
                                      size=max(0, args.pool_size - te_tok.size), replace=False).astype(np.int64)])
    print(f"[ima-tf] model={args.model} configs={configs} n_eval={sel.size} K={args.n_keys} dev={DEV}", flush=True)

    records = []
    for c in configs:
        ae = cfg_alpha(c)
        dep = _obf_rows(W, ae, args.keymat_h, args.keymat_lam, args.keymat_seed)   # deployment key

        # plain control: matched single key, row-split over train/test tokens
        ctrl = float((ima_transformer_attack(dep[tr_tok], W[tr_tok], dep[te_tok], W[pool], pool,
                                             hidden=args.hidden, epochs=args.epochs, seed=args.seed) == te_tok).mean())

        # τ-invariant in-model: train on K synthetic own-key obfuscations, decode the deployment rows
        Xsyn = np.concatenate([_obf_rows(W, ae, args.keymat_h, args.keymat_lam, args.attacker_seed + k)[tr_tok]
                               for k in range(args.n_keys)], 0)
        Esyn = np.tile(W[tr_tok], (args.n_keys, 1))
        inv = float((ima_transformer_attack(Xsyn, Esyn, dep[te_tok], W[pool], pool,
                                            hidden=args.hidden, epochs=args.epochs, seed=args.seed) == te_tok).mean())
        records.append({"config": c, "alpha_e": ae, "plain_control_matched": ctrl, "tau_invariant_inmodel": inv,
                        "n_test": int(te_tok.size)})
        print(f"[ima-tf] {c:>14} αₑ={ae} | plain-control(matched)={ctrl:.3f}  τ-invariant(in-model)={inv:.3f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "model": args.model, "surface": "static_obfuscated_embeddings", "attack": "ima_embedrow_transformer",
        "vocab": int(vocab), "d": int(d), "n_keys": args.n_keys, "n_eval": int(sel.size),
        "keymat": {"h": args.keymat_h, "lam": args.keymat_lam, "seed": args.keymat_seed},
        "release_gate_ttrsr": 0.15,
        "note": "IMAInverter (2-layer/8-head, paper §F.1). plain_control_matched = single deployment key, "
                "row-split (validates the inverter inverts). tau_invariant_inmodel = trained on K synthetic "
                "own-key obfuscations, decode the deployment rows (the in-model attack; ~0 ⇒ AloePri defends, "
                "paper IMA≈0%). Swept over αₑ (keymat=0, alg1@x).",
        "records": records,
    }, indent=2))
    print(f"[ima-tf] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
