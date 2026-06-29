#!/usr/bin/env python3
"""ISA-AttnScore — Internal-State Attack against attention-score observables (AloePri §5.2.3).

The attacker reads per-head attention scores from GPU memory and inverts them to token ids. Algorithm
1 (keymat) preserves Q·Kᵀ BYTE-FOR-BYTE (forward-correctness requires it), so under keymat_only /
full_alg1 the attention scores are IDENTICAL to plaintext — Alg1 gives ZERO protection on this surface.
That gap is exactly what Algorithm 2's intra-head transforms (R̂qk, Ĥqk, Ẑblock, head perms) close;
since Alg2 is the queued config, this eval measures the Alg1-only artifact and confirms the gap.

Feature: per query position, attention over the most recent `--window` causal keys, per head →
fixed dim n_heads·window. Ridge inverter (attn features → token embedding) → nearest token over the
candidate pool. Reports matched TTRSR for plaintext vs keymat_only (expected ≈ equal: Alg1 inert here).

GPU: ONE process; run via scripts/run_in_rocm.sh.
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

from defenses.aloepri import reparam_pythia  # noqa: E402
from talens.attacks.dp_inversion import nearest_token, ridge_W  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def capture_attn(model, tok, prompts, layer, window):
    """Per query position: attention over the last `window` causal keys, per head → (n_heads·window).
    Returns (X feature rows, token ids). Unbatched (small corpus; output_attentions per prompt)."""
    feats, ids = [], []
    for p in prompts:
        enc = tok(p, return_tensors="pt")
        input_ids = enc.input_ids.to(DEV)
        att = model(input_ids, output_attentions=True, use_cache=False).attentions[layer][0]  # (H,q,kv)
        H, q, _ = att.shape
        toks = input_ids[0].cpu().numpy()
        for pos in range(q):
            row = att[:, pos, :pos + 1]                      # (H, pos+1) causal
            w = torch.zeros(H, window, device=att.device)
            k = min(window, pos + 1)
            w[:, :k] = row[:, pos + 1 - k:pos + 1]           # most-recent `window` keys (right-aligned)
            feats.append(w.reshape(-1).float().cpu().numpy())
            ids.append(int(toks[pos]))
    return np.asarray(feats, np.float32), np.asarray(ids, np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--corpus", default="corpora/release-gate-512.txt")
    ap.add_argument("--max-prompts", type=int, default=160)
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--configs", default="plaintext,keymat_only")
    ap.add_argument("--pool-size", type=int, default=2048)
    ap.add_argument("--keymat-h", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260626)
    ap.add_argument("--out", default="refine-logs/aloepri/aloepri_attnscore.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompts = [l.strip() for l in Path(args.corpus).read_text().splitlines() if l.strip()][: args.max_prompts]
    rng = np.random.default_rng(args.seed)
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    print(f"[attnscore] model={args.model} layer={args.layer} window={args.window} configs={configs} "
          f"prompts={len(prompts)} dev={DEV}", flush=True)

    records = []
    table = None
    split = None
    for cname in configs:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32, attn_implementation="eager", device_map=DEV).eval()
        if table is None:
            table = model.get_input_embeddings().weight.detach().float().cpu().numpy().astype(np.float32)
        if cname != "plaintext":
            reparam_pythia(model, config=cname, h=args.keymat_h, seed=args.seed)
        X, y = capture_attn(model, tok, prompts, args.layer, args.window)
        del model
        if split is None:                                    # shared vocab-disjoint split + pool (once)
            distinct = rng.permutation(np.unique(y))
            ntr = int(0.7 * distinct.size)
            tr_ids = set(distinct[:ntr].tolist())
            tr = np.array([i for i, t in enumerate(y) if t in tr_ids])
            te = np.array([i for i, t in enumerate(y) if t not in tr_ids])
            true_pool = np.unique(y[te])
            avail = np.setdiff1d(np.arange(table.shape[0], dtype=np.int64), true_pool)
            fill = rng.choice(avail, size=max(0, args.pool_size - true_pool.size), replace=False)
            pool = np.concatenate([true_pool, fill.astype(np.int64)])
            split = (tr, te, pool)
        tr, te, pool = split
        emb_y = table[y]
        Wmap = ridge_W(X[tr], emb_y[tr], alpha=1.0)
        yhat = nearest_token(X[te] @ Wmap, table[pool], pool)
        ttrsr = float((yhat == y[te]).mean())
        rec = {"config": cname, "layer": args.layer, "n_rows": int(X.shape[0]),
               "feat_dim": int(X.shape[1]), "isa_attnscore_ttrsr": ttrsr,
               "gate_pass": ttrsr <= 0.15}
        records.append(rec)
        print(f"[attnscore] {cname:>12} | ISA-AttnScore TTRSR={ttrsr:.3f} (gate≤.15: "
              f"{'PASS' if ttrsr <= 0.15 else 'FAIL'})", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "model": args.model, "surface": "attention_scores", "layer": args.layer, "window": args.window,
        "release_gate_ttrsr": 0.15,
        "note": "Alg1 keymat preserves Q·Kᵀ byte-for-byte → keymat_only attention scores == plaintext "
                "(Alg1 inert on this surface; Algorithm 2 is what defends it). matched ridge inverter "
                "on per-head attention over the last `window` causal keys.",
        "records": records,
    }, indent=2))
    print(f"[attnscore] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
