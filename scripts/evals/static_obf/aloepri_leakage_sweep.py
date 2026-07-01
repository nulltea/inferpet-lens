#!/usr/bin/env python3
"""AloePri static-obfuscation leakage sweep (Surface A: residual activations).

The defense-family analog of dp_leakage_sweep, but the sweep axis is a list of AloePri
CONFIGS (not an ε): the model is covariantly RE-PARAMETERIZED (scripts/defenses/aloepri.py
:reparam_pythia) so the released residual is the obfuscated x'=x P̂ (width d→d+2h), the forward
output preserved up to the embedding/head noise. Configs:

  plaintext        — no obfuscation (the clean reference; trains the adversary's inverter)
  keymat_only      — residual key pair P̂/Q̂ alone (lossless, P̂Q̂=I): MI preserved exactly
  full_alg1@αₑ     — + embedding noise (Wₑ+αₑσₑE)P̂ : the only information-theoretic knob

Threat model (NOT the DP WEIGHTS-PUB synthesis model). AloePri is information-preserving and
invertible, so a linear attack RE-TRAINED on obfuscated reps absorbs the basis change for free —
keymat gives no info-theoretic protection. Its protection is key-gated ACCESS CONTROL. We report
three recoveries per (config, layer, attack):

  matched  — attack trained+tested on the obf reps (adversary has paired obf data) → recovers;
             tracks the probe (the MI content that survives the obfuscation).
  oracle   — plaintext-trained attack tested on X_obf·Q̂_true (knows the key) → recovers.
  blind    — plaintext-trained attack tested on X_obf·Q̂_wrong (right scheme, WRONG key) → fails.

oracle−blind = pure access control (no information lost, just locked behind the key). The αₑ-noise
drop in matched/oracle = the only genuine information reduction. Probes (CLUB/V_cap) are computed on
the obf reps and are ~FLAT across the lossless basis change (invertible ⇒ MI preserved) — the finding
is attack-defeat-by-key, not a bits↔recovery correlation. Π is omitted (activation-inert; its
protection lives on the token-id surface — see the surface-B sweep).

GPU: ONE process at a time; run via scripts/run_in_rocm.sh. Output JSON under refine-logs/<surface>/.

Example (smoke):
  scripts/run_in_rocm.sh python3 scripts/evals/aloepri_leakage_sweep.py \
      --layers 6 --configs plaintext,keymat_only,full_alg1@1.0 --attacks ridge \
      --max-prompts 64 --out refine-logs/aloepri/smoke.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # scripts/ for defenses.*, evals.*

from defenses.aloepri import keymat_gen, reparam_pythia  # noqa: E402
from evals.dp.dp_leakage_sweep import (ATTACKS as _ATTACKS, _spearman, _stack, capture,  # noqa: E402
                                    probe_club, probe_mdl, probe_vcap)
from talens.attacks import nn_attack  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ATTACKS = {**_ATTACKS, "nn": nn_attack}            # + NN (training-free cosine-NN, paper §F.1)
PROBES = {"club": probe_club, "vcap": probe_vcap, "mdl": probe_mdl}


def _parse_config(spec: str):
    """'plaintext' | 'keymat_only' | 'full_alg1@<αₑ>'  →  (name, kwargs-for-reparam | None)."""
    if spec == "plaintext":
        return spec, None
    if spec == "keymat_only":
        return spec, dict(config="keymat_only")
    if spec.startswith("full_alg1"):
        alpha = float(spec.split("@", 1)[1]) if "@" in spec else 1.0
        return spec, dict(config="full_alg1", alpha_e=alpha, alpha_h=0.0)
    if spec.startswith("alg2"):
        alpha = float(spec.split("@", 1)[1]) if "@" in spec else 0.0
        return spec, dict(config="alg2", alpha_e=alpha, alpha_h=0.0)
    raise ValueError(f"unknown config spec {spec!r}")


def _load_model(model_id):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, attn_implementation="eager", device_map=DEV).eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--corpus", default="corpora/release-gate-512.txt")
    ap.add_argument("--max-prompts", type=int, default=160)
    ap.add_argument("--layers", default="0,6,11")
    ap.add_argument("--configs", default="plaintext,keymat_only,full_alg1@0.25,full_alg1@0.5,full_alg1@1.0,full_alg1@2.0")
    ap.add_argument("--attacks", default="ridge,decoder", help=f"subset of {sorted(ATTACKS)}")
    ap.add_argument("--probes", default="club,vcap", help=f"subset of {sorted(PROBES)}")
    ap.add_argument("--keymat-h", type=int, default=128)
    ap.add_argument("--keymat-lam", type=float, default=0.3)
    ap.add_argument("--keymat-seed", type=int, default=0)
    ap.add_argument("--pool-size", type=int, default=2048)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--club-max-rows", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20260626)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="refine-logs/aloepri/aloepri_leakage_sweep.json")
    args = ap.parse_args()

    layers = [int(s) for s in args.layers.split(",") if s.strip()]
    configs = [_parse_config(s.strip()) for s in args.configs.split(",") if s.strip()]
    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    probes = [p.strip() for p in args.probes.split(",") if p.strip()]
    for a in attacks:
        if a not in ATTACKS:
            ap.error(f"unknown attack {a!r}; choose from {sorted(ATTACKS)}")
    for p in probes:
        if p not in PROBES:
            ap.error(f"unknown probe {p!r}; choose from {sorted(PROBES)}")

    from transformers import AutoTokenizer
    prompts = [l.strip() for l in Path(args.corpus).read_text().splitlines() if l.strip()][: args.max_prompts]
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    rng = np.random.default_rng(args.seed)

    print(f"[aloepri] model={args.model} layers={layers} configs={[c[0] for c in configs]} "
          f"attacks={attacks} probes={probes} prompts={len(prompts)} dev={DEV}", flush=True)

    # ── plaintext reference: clean reps, the vocab-disjoint split, candidate pool, shuffle floor ──
    model = _load_model(args.model)
    d = model.gpt_neox.embed_in.weight.shape[1]
    table = model.get_input_embeddings().weight.detach().float().cpu().numpy().astype(np.float32)
    vocab = table.shape[0]
    perc, idc = capture(model, tok, prompts, layers, batch_size=args.batch_size)
    _, y0 = _stack(perc[layers[0]], idc)
    distinct = rng.permutation(np.unique(y0))
    ntr = int(0.7 * distinct.size)
    tr_ids, te_ids = set(distinct[:ntr].tolist()), set(distinct[ntr:].tolist())
    tr = np.array([i for i, t in enumerate(y0) if t in tr_ids])
    te = np.array([i for i, t in enumerate(y0) if t in te_ids])
    true_pool = np.unique(y0[te])
    avail = np.setdiff1d(np.arange(vocab, dtype=np.int64), true_pool)
    fill = rng.choice(avail, size=max(0, args.pool_size - true_pool.size), replace=False)
    pool = np.concatenate([true_pool, fill.astype(np.int64)])
    permsh = rng.permutation(tr.size)
    Xclean, floors = {}, {}
    for L in layers:
        X0, y = _stack(perc[L], idc)
        assert np.array_equal(y, y0)
        Xclean[L] = X0
        emb_y = table[y]
        floors[L] = {}
        for a in attacks:
            yhat = ATTACKS[a](X0[tr], emb_y[tr][permsh], X0[te], table[pool], pool,
                              ytr=y[tr][permsh], full_emb=table, hidden=args.hidden,
                              epochs=args.epochs, seed=args.seed)
            floors[L][a] = float((yhat == y[te]).mean())
    y = y0
    emb_y = table[y]
    K = int(np.unique(y).size)
    del model

    # the WRONG key (right scheme, different seed) — same shape (d+2h, d) as Q̂_true
    _, Q_wrong = keymat_gen(d, args.keymat_h, lam=args.keymat_lam, seed=args.keymat_seed + 99991)

    records = []
    for cname, rkw in configs:
        if rkw is None:                                    # plaintext: residual already = clean
            Xobf = {L: Xclean[L] for L in layers}
            Q_true = None
        else:
            m = _load_model(args.model)
            keys = reparam_pythia(m, h=args.keymat_h, lam=args.keymat_lam, seed=args.keymat_seed, **rkw)
            Q_true = keys["Q"].cpu().numpy().astype(np.float32)
            perco, idco = capture(m, tok, prompts, layers, batch_size=args.batch_size)
            assert all(np.array_equal(a, b) for a, b in zip(idco, idc)), "token ids drifted under obfuscation"
            Xobf = {L: _stack(perco[L], idc)[0] for L in layers}
            del m

        for L in layers:
            Xo = Xobf[L]
            rec = {"config": cname, "layer": L, "obf_dim": int(Xo.shape[1])}
            for a in attacks:
                fl = floors[L][a]
                # matched: adversary has paired obf data → fit+test on obf reps (basis-adaptive)
                yh_m = ATTACKS[a](Xo[tr], emb_y[tr], Xo[te], table[pool], pool, ytr=y[tr],
                                  full_emb=table, hidden=args.hidden, epochs=args.epochs, seed=args.seed)
                rec[f"{a}_matched"] = float((yh_m == y[te]).mean())
                rec[f"{a}_matched_sel"] = rec[f"{a}_matched"] - fl
                if Q_true is not None:                     # oracle (key) / blind (wrong key)
                    Xte_oracle = (Xo[te] @ Q_true).astype(np.float32)
                    Xte_blind = (Xo[te] @ Q_wrong).astype(np.float32)
                    yh_o = ATTACKS[a](Xclean[L][tr], emb_y[tr], Xte_oracle, table[pool], pool, ytr=y[tr],
                                      full_emb=table, hidden=args.hidden, epochs=args.epochs, seed=args.seed)
                    yh_b = ATTACKS[a](Xclean[L][tr], emb_y[tr], Xte_blind, table[pool], pool, ytr=y[tr],
                                      full_emb=table, hidden=args.hidden, epochs=args.epochs, seed=args.seed)
                    rec[f"{a}_oracle"] = float((yh_o == y[te]).mean())
                    rec[f"{a}_blind"] = float((yh_b == y[te]).mean())
                    rec[f"{a}_access_control"] = rec[f"{a}_oracle"] - rec[f"{a}_blind"]
            for p in probes:
                out = PROBES[p](Xo, emb_y, y, K, club_max_rows=args.club_max_rows,
                                full_emb=table, pool_size=args.pool_size, X_clean=Xclean[L])
                for k, v in out.items():
                    rec[f"{p}_{k}"] = v
            records.append(rec)
            atxt = " ".join(
                f"{a}: m={rec[f'{a}_matched']:.3f}"
                + (f" o={rec[f'{a}_oracle']:.3f} b={rec[f'{a}_blind']:.3f}" if f"{a}_oracle" in rec else "")
                for a in attacks)
            ptxt = " ".join(f"{p}={rec.get(p + '_bits')}" for p in probes)
            print(f"[aloepri] {cname:>16} L{L:>2} dim={rec['obf_dim']} | {atxt} | {ptxt}", flush=True)

    # does matched recovery track the probe across the αₑ sweep (the MI-content correlation)?
    corr = {}
    full = [r for r in records if r["config"].startswith("full_alg1")]
    for L in layers:
        r = [x for x in full if x["layer"] == L]
        corr[f"L{L}"] = {}
        for a in attacks:
            for p in probes:
                pairs = [(x[f"{a}_matched_sel"], x[f"{p}_bits"]) for x in r
                         if x.get(f"{p}_bits") is not None and np.isfinite(x.get(f"{p}_bits", np.nan))]
                if len(pairs) >= 2:
                    xs, ys_ = zip(*pairs)
                    corr[f"L{L}"][f"{a}_matched_vs_{p}"] = _spearman(xs, ys_)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "model": args.model, "corpus": args.corpus, "n_prompts": len(prompts),
        "defense": "aloepri_covariant_obfuscation", "surface": "residual_activations",
        "keymat": {"h": args.keymat_h, "lam": args.keymat_lam, "seed": args.keymat_seed},
        "threat_note": "matched=adversary trains on obf reps (basis-adaptive); oracle=plaintext-trained "
                       "tested on X_obf·Q̂_true; blind=tested on X_obf·Q̂_wrong (right scheme, wrong key). "
                       "oracle−blind = key-gated access control; αₑ-noise drop = info-theoretic loss. "
                       "Probes on obf reps ~flat across the lossless basis change (invertible⇒MI preserved).",
        "layers": layers, "configs": [c[0] for c in configs], "attacks": attacks, "probes": probes,
        "pool_size": args.pool_size, "floors": {f"L{L}": floors[L] for L in layers},
        "matched_vs_probe_corr": corr, "records": records,
    }, indent=2))
    print(f"[aloepri] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
