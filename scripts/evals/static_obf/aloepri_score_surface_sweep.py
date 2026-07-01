#!/usr/bin/env python3
"""AloePri score-surface ISA sweep — kq (pre-softmax Q·Kᵀ) and kqv_out (attention output, the
per-head Q/K/V VALUE coordinates) across ALL layers. Replaces the post-softmax windowed
ISA-AttnScore (which floored at a mid-layer + disjoint-split vacuous gate).

This is the in-model demo the ledger queued: Algorithm 2 obfuscates the per-head value coords
(Ûvo rotation + inter-head permutation), so kqv_out is the surface Alg2 actually defends; kq
(scores) is preserved up to a fixed head-perm + rotary non-commutation. Both transforms are
FIXED orthogonal maps, so a *matched* (retrained) ridge absorbs them — any recovery drop is the
αₑ embedding noise, not the rotation. Configs let us attribute rotation vs noise:

  plaintext     — clean reference
  keymat_only   — Alg1 keymat (head-axis untouched ⇒ kq/kqv_out == plaintext; confirms Alg1-inert)
  alg2@0        — Alg2 head transforms, NO αₑ noise (isolates the orthogonal value rotation)
  alg2@1.0      — full deployment: Alg2 + αₑ=1.0 embedding noise (the privacy-relevant config)

Per (config × layer × surface) we fit a matched ridge obs→token-embedding → nearest-token over a
candidate pool, under BOTH vocab-disjoint (held-out tokens; memorization-controlled) and row-split
(shared vocab; private-rag's headline) splits. Reports TTRSR + a shuffled-label floor.

Methodology matches private-rag aloepri-attacks.md (kq/kqv_out, matched ridge, both splits) — no
multi-key on this surface (the deployment's own obf tensor is captured directly).

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
from talens.attacks import nearest_token, ridge_W  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ── capture: a custom attention interface stashes pre-softmax kq; a dense pre-hook stashes kqv_out ──
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402
from transformers.models.gpt_neox.modeling_gpt_neox import eager_attention_forward as _EAGER  # noqa: E402

_CAP: dict = {}  # ("kq"|"kqv_out", layer) -> np.ndarray for the in-flight prompt


def _capture_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    li = getattr(module, "_talens_li", None)
    if li is not None:
        s = (torch.matmul(query, key.transpose(2, 3)) * scaling)[0]   # (H, q, kv) pre-softmax, pre-mask
        _CAP[("kq", li)] = s.detach().float().cpu().numpy()
    return _EAGER(module, query, key, value, attention_mask, scaling, dropout=dropout, **kw)


ALL_ATTENTION_FUNCTIONS["talens_capture"] = _capture_eager


def _parse_config(spec: str):
    if spec == "plaintext":
        return spec, None
    if spec == "keymat_only":
        return spec, dict(config="keymat_only")
    if spec.startswith("alg2"):
        alpha = float(spec.split("@", 1)[1]) if "@" in spec else 0.0
        return spec, dict(config="alg2", alpha_e=alpha, alpha_h=0.0)
    if spec.startswith("full_alg1"):
        alpha = float(spec.split("@", 1)[1]) if "@" in spec else 1.0
        return spec, dict(config="full_alg1", alpha_e=alpha, alpha_h=0.0)
    raise ValueError(f"unknown config spec {spec!r}")


def _load(model_id):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, attn_implementation="eager", device_map=DEV).eval()
    m.config._attn_implementation = "talens_capture"
    for li, layer in enumerate(m.gpt_neox.layers):
        layer.attention._talens_li = li

        def _mk(li):
            def hook(mod, inp):
                _CAP[("kqv_out", li)] = inp[0][0].detach().float().cpu().numpy()   # (q, d)
            return hook
        layer.attention.dense.register_forward_pre_hook(_mk(li))
    return m


@torch.no_grad()
def capture(model, tok, prompts, layers, window):
    """Returns feats[surface][L] = (rows, dim) stacked over positions, and ids (token id per row).
    kqv_out feature = per-position d-vec; kq feature = last `window` causal keys per head (H*window)."""
    feats = {s: {L: [] for L in layers} for s in ("kq", "kqv_out")}
    ids = []
    for p in prompts:
        input_ids = tok(p, return_tensors="pt").input_ids.to(DEV)
        _CAP.clear()
        model(input_ids, use_cache=False)
        toks = input_ids[0].cpu().numpy()
        q = toks.shape[0]
        for L in layers:
            feats["kqv_out"][L].append(_CAP[("kqv_out", L)].astype(np.float32))    # (q, d)
            sc = _CAP[("kq", L)]                                                    # (H, q, kv)
            H = sc.shape[0]
            w = np.zeros((q, H, window), np.float32)
            for pos in range(q):
                k = min(window, pos + 1)
                w[pos, :, :k] = sc[:, pos, pos + 1 - k:pos + 1]                     # most-recent keys, right-aligned
            feats["kq"][L].append(w.reshape(q, H * window))
        ids.append(toks)
    ids = np.concatenate(ids).astype(np.int64)
    out = {s: {L: np.concatenate(feats[s][L]).astype(np.float32) for L in layers} for s in feats}
    return out, ids


def _recovery(Xtr, Xte, ytr, yte, table, pool, *, shuffle=None):
    W = ridge_W(Xtr, table[ytr if shuffle is None else shuffle], alpha=1.0)
    yhat = nearest_token(Xte @ W, table[pool], pool)
    return float((yhat == yte).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--corpus", default="corpora/release-gate-512.txt")
    ap.add_argument("--max-prompts", type=int, default=160)
    ap.add_argument("--layers", default="all", help="'all' or comma list")
    ap.add_argument("--configs", default="plaintext,keymat_only,alg2@0,alg2@1.0")
    ap.add_argument("--surfaces", default="kq,kqv_out")
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--pool-size", type=int, default=2048)
    ap.add_argument("--keymat-h", type=int, default=128)
    ap.add_argument("--keymat-lam", type=float, default=0.3)
    ap.add_argument("--keymat-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260627)
    ap.add_argument("--out", default="refine-logs/aloepri/aloepri_score_surface.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompts = [l.strip() for l in Path(args.corpus).read_text().splitlines() if l.strip()][: args.max_prompts]
    configs = [_parse_config(s.strip()) for s in args.configs.split(",") if s.strip()]
    surfaces = [s.strip() for s in args.surfaces.split(",") if s.strip()]
    rng = np.random.default_rng(args.seed)

    # layer count
    from transformers import AutoConfig
    nL = AutoConfig.from_pretrained(args.model).num_hidden_layers
    layers = list(range(nL)) if args.layers == "all" else [int(s) for s in args.layers.split(",")]
    print(f"[score] model={args.model} layers={layers} configs={[c[0] for c in configs]} "
          f"surfaces={surfaces} prompts={len(prompts)} dev={DEV}", flush=True)

    # capture all configs; verify token ids are identical (forward-preserving obfuscation)
    caps, ids0, table, vocab = {}, None, None, None
    for cname, rkw in configs:
        m = _load(args.model)
        if table is None:
            table = m.get_input_embeddings().weight.detach().float().cpu().numpy().astype(np.float32)
            vocab = table.shape[0]
        if rkw is not None:
            reparam_pythia(m, h=args.keymat_h, lam=args.keymat_lam, seed=args.keymat_seed, **rkw)
        feats, ids = capture(m, tok, prompts, layers, args.window)
        del m
        if ids0 is None:
            ids0 = ids
        else:
            assert np.array_equal(ids, ids0), f"token ids drifted under {cname}"
        caps[cname] = feats
        print(f"[score] captured {cname:>12} | kq dim={feats['kq'][layers[0]].shape[1]} "
              f"kqv_out dim={feats['kqv_out'][layers[0]].shape[1]} rows={ids.shape[0]}", flush=True)

    y = ids0
    # two splits, shared across configs/layers/surfaces
    distinct = rng.permutation(np.unique(y))
    nd = int(0.7 * distinct.size)
    tr_ids = set(distinct[:nd].tolist())
    dj_tr = np.array([i for i, t in enumerate(y) if t in tr_ids])
    dj_te = np.array([i for i, t in enumerate(y) if t not in tr_ids])
    rs = rng.permutation(y.size)
    nr = int(0.7 * y.size)
    rs_tr, rs_te = rs[:nr], rs[nr:]
    splits = {"disjoint": (dj_tr, dj_te), "rowsplit": (rs_tr, rs_te)}

    def _pool(te):
        true_pool = np.unique(y[te])
        avail = np.setdiff1d(np.arange(vocab, dtype=np.int64), true_pool)
        fill = rng.choice(avail, size=max(0, args.pool_size - true_pool.size), replace=False)
        return np.concatenate([true_pool, fill.astype(np.int64)])
    pools = {k: _pool(v[1]) for k, v in splits.items()}
    shuf = {k: rng.permutation(splits[k][0]) for k in splits}   # shuffled-label floor (train rows)

    records = []
    for cname, _ in configs:
        for L in layers:
            for surf in surfaces:
                X = caps[cname][surf][L]
                rec = {"config": cname, "layer": L, "surface": surf, "dim": int(X.shape[1])}
                for sp, (tr, te) in splits.items():
                    pool = pools[sp]
                    rec[f"{sp}_ttrsr"] = _recovery(X[tr], X[te], y[tr], y[te], table, pool)
                    rec[f"{sp}_floor"] = _recovery(X[tr], X[te], y[tr], y[te], table, pool, shuffle=shuf[sp])
                    rec[f"{sp}_sel"] = rec[f"{sp}_ttrsr"] - rec[f"{sp}_floor"]
                records.append(rec)
            kqv = next(r for r in records if r["config"] == cname and r["layer"] == L and r["surface"] == surfaces[-1])
            kq = next((r for r in records if r["config"] == cname and r["layer"] == L and r["surface"] == "kq"), None)
            msg = " ".join(f"{r['surface']}: dj={r['disjoint_ttrsr']:.3f} rs={r['rowsplit_ttrsr']:.3f}"
                           for r in records if r["config"] == cname and r["layer"] == L)
            print(f"[score] {cname:>12} L{L:>2} | {msg}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "model": args.model, "corpus": args.corpus, "n_prompts": len(prompts),
        "surface": "attention_score_and_value", "window": args.window,
        "defense": "aloepri_covariant_obfuscation",
        "keymat": {"h": args.keymat_h, "lam": args.keymat_lam, "seed": args.keymat_seed},
        "release_gate_ttrsr": 0.15,
        "note": "kq=pre-softmax Q·Kᵀ (last `window` causal keys/head); kqv_out=attention output "
                "(input to attention.dense, per-head value coords Alg2 rotates). matched ridge "
                "obs→token-embedding→nearest-token. disjoint=vocab-disjoint (memorization-controlled), "
                "rowsplit=shared-vocab (private-rag headline). alg2@0 isolates the orthogonal rotation "
                "(matched-absorbable); alg2@1.0 adds αₑ noise (the real lever). No multi-key on this surface.",
        "layers": layers, "configs": [c[0] for c in configs], "surfaces": surfaces,
        "pool_size": args.pool_size, "records": records,
    }, indent=2))
    print(f"[score] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
