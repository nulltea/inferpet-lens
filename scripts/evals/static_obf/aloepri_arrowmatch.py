#!/usr/bin/env python3
"""ArrowMatch (Wang et al., USENIX Sec '25, "Game of Arrows") on the AloePri pythia embedding table.

Direction-similarity attack: recover the secret token permutation Π by cosine-matching each obfuscated
embedding row (aligned to plain width by truncation) to its nearest plaintext row (paper Eq. 1):
    σ(i) = argmax_j cos(w_obf^i, w_pre^j)
TTRSR = fraction of obf rows whose top-1 (and top-10) plain match equals the true τ⁻¹.

Obs2 (the attack's foundation): per-vector obfuscation (Π + diagonal scaling) preserves vector
DIRECTION → ArrowMatch recovers it. Only MATRIX-MULTIPLICATION obfuscation changes direction —
exactly AloePri's Algorithm-1 keymat (non-diagonal invertible P̂). So AloePri is expected immune:

  plain_control  — Π + αₑ=0, keymat OFF (pure permutation): direction preserved ⇒ recovers ~1.0
  aloepri        — Π + αₑ=1.0 + keymat P̂ (matrix mult, width d→d+2h): direction destroyed ⇒ ~0.0

CPU-only (embedding-table cosine; no forward pass). Mirrors private-rag m2_7/run_arrowmatch.py.
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


def _topk_match(W_obf, W_pre, topk=10, chunk=512):
    """For each row of W_obf (n,d), top-k cosine-nearest rows of W_pre (V,d). Chunked. Returns (n,topk)."""
    Wp = torch.nn.functional.normalize(torch.from_numpy(W_pre).float(), dim=1)
    out = np.empty((W_obf.shape[0], topk), dtype=np.int64)
    for i in range(0, W_obf.shape[0], chunk):
        q = torch.nn.functional.normalize(torch.from_numpy(W_obf[i:i + chunk]).float(), dim=1)
        sim = q @ Wp.T
        out[i:i + chunk] = sim.topk(topk, dim=1).indices.numpy()
    return out


def _run(plain, mode, alpha_e, h, lam, seed, n_eval, topk, rng):
    """Obfuscate `plain` (V,d) per `mode`, run ArrowMatch on n_eval obf rows against the full plain table.
      mode='perm'   — pure row permutation Π only (per-vector op ⇒ DIRECTION PRESERVED): ArrowMatch's
                      target regime, the sanity control (recovers ~1.0).
      mode='keymat' — AloePri Π + αₑ noise + P̂ matrix-mult change of basis (direction destroyed)."""
    if mode == "perm":
        n = plain.shape[0]
        tau = rng.permutation(n)
        obf = np.empty_like(plain)
        obf[tau] = plain                                         # obf[τ[i]] = plain[i] (direction preserved)
    else:
        wp = obfuscate_embedding_table(plain, alpha_e=alpha_e, h=h, lam=lam, keymat=True, seed=seed)
        obf, tau = wp.obf.astype(np.float32), wp.perm           # obf[τ[i]] = (plain[i]+αₑE)·P̂
    d = plain.shape[1]
    obf_aligned = obf[:, :d] if obf.shape[1] >= d else np.pad(obf, ((0, 0), (0, d - obf.shape[1])))
    tau_inv = np.argsort(tau)                                    # tau_inv[k] = plain id behind obf row k
    rows = rng.choice(obf.shape[0], size=min(n_eval, obf.shape[0]), replace=False)
    pred = _topk_match(obf_aligned[rows], plain, topk=topk)
    truth = tau_inv[rows]
    top1 = float((pred[:, 0] == truth).mean())
    topk_r = float((pred == truth[:, None]).any(1).mean())
    return {"obf_dim": int(obf.shape[1]), "n_eval": int(rows.size),
            "ttrsr_top1": top1, "ttrsr_topk": topk_r, "topk": topk}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--n-eval", type=int, default=4000, help="obf rows scored (vs the full plain table)")
    ap.add_argument("--alpha-e", type=float, default=1.0)
    ap.add_argument("--keymat-h", type=int, default=128)
    ap.add_argument("--keymat-lam", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=20260627)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--out", default="refine-logs/aloepri/aloepri_arrowmatch.json")
    args = ap.parse_args()

    # plain embedding table (no GPU needed; load weights on CPU)
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    plain = m.get_input_embeddings().weight.detach().float().cpu().numpy().astype(np.float32)
    del m
    rng = np.random.default_rng(args.seed)
    print(f"[arrowmatch] model={args.model} vocab={plain.shape[0]} d={plain.shape[1]} n_eval={args.n_eval}",
          flush=True)

    cells = {
        "plain_control": dict(mode="perm", alpha_e=0.0),                  # Π only — direction preserved
        "aloepri_keymat": dict(mode="keymat", alpha_e=args.alpha_e),      # Π + αₑ + P̂ matrix mult
    }
    records = []
    for name, kw in cells.items():
        r = _run(plain, h=args.keymat_h, lam=args.keymat_lam, seed=args.seed, n_eval=args.n_eval,
                 topk=args.topk, rng=np.random.default_rng(args.seed), **kw)
        r["cell"] = name
        records.append(r)
        print(f"[arrowmatch] {name:>16} | top1={r['ttrsr_top1']:.4f} top{args.topk}={r['ttrsr_topk']:.4f} "
              f"(obf_dim={r['obf_dim']})", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "model": args.model, "surface": "static_obfuscated_embeddings", "attack": "arrowmatch",
        "vocab": int(plain.shape[0]), "d": int(plain.shape[1]),
        "release_gate_ttrsr": 0.15,
        "note": "ArrowMatch Stage-1 cosine direction match (paper Eq.1). plain_control=Π only (direction "
                "preserved ⇒ recovers); aloepri_keymat=Π+αₑ+P̂ matrix mult (Obs2: matrix mult destroys "
                "direction ⇒ immune). Mirrors private-rag m2_7-arrowmatch (Q3-4B: plain 0.986 → obf 0.000).",
        "records": records,
    }, indent=2))
    print(f"[arrowmatch] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
