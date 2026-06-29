#!/usr/bin/env python3
"""AloePri ISA-HiddenState — in-model BLIND attack with multi-key synthesis (residual surface, all layers).

The realistic AloePri threat model (honest-but-curious server, Kerckhoffs): the attacker has the public
plaintext model and the obfuscation ALGORITHM, but NOT the deployment's secret key (P̂/Q̂ seed) and NO
(obf, plain) paired data from the live basis. So `matched` (paired ridge on deployment reps) and `oracle`
(knows Q̂) are out of model — only `blind` counts.

Two blind variants:
  single-key — attacker fits a plaintext-trained inverter and tests on the deployment's obf reps with ONE
               wrong key (the naive baseline; floors at ~chance by basis mismatch).
  multi-key  — paper §F.1 / private-rag synthesis: attacker draws K synthetic keymats {P_a^k} (own seeds),
               forms synthetic obf train reps X_clean·P_a^k for every k, and fits ONE ridge over the pooled
               set so it must learn a key-INVARIANT inverse, then decodes the real deployment reps. If the
               Alg1 keymat family has a linear K-invariant inverse, this transfers; if each draw is an
               independent dense basis, it cannot — that itself is the finding (key-gated, not info-hiding).

Reports blind TTRSR (single + multi) vs the matched ceiling and the plaintext floor, per layer, under
vocab-disjoint (generalization) + row-split. GPU: ONE process; run via scripts/run_in_rocm.sh.
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

from defenses.aloepri import keymat_gen, reparam_pythia  # noqa: E402
from evals.dp.dp_leakage_sweep import _stack, capture, probe_vcap  # noqa: E402
from talens.attacks.dp_inversion import multikey_ridge_W, nearest_token, ridge_W  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _parse_config(spec: str):
    if spec == "plaintext":
        return spec, None                                    # no-defence reference (direct inversion)
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
    return AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, attn_implementation="eager", device_map=DEV).eval()


def _ttrsr(W, Xte, yte, table, pool):
    return float((nearest_token(Xte @ W, table[pool], pool) == yte).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--corpus", default="corpora/release-gate-512.txt")
    ap.add_argument("--max-prompts", type=int, default=160)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--configs", default="plaintext,keymat_only,full_alg1@1.0")
    ap.add_argument("--n-keys", type=int, default=64, help="K synthetic attacker keymats for multi-key")
    ap.add_argument("--pvi-fit-keys", type=int, default=16,
                    help="how many synthetic keymats the in-model (blind) PVI reader trains on")
    ap.add_argument("--pool-size", type=int, default=2048)
    ap.add_argument("--keymat-h", type=int, default=128)
    ap.add_argument("--keymat-lam", type=float, default=0.3)
    ap.add_argument("--keymat-seed", type=int, default=0, help="deployment (true) seed")
    ap.add_argument("--attacker-seed", type=int, default=700000, help="base seed for synthetic keymats")
    ap.add_argument("--seed", type=int, default=20260627)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="refine-logs/aloepri/aloepri_multikey_blind.json")
    args = ap.parse_args()

    from transformers import AutoConfig, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    prompts = [l.strip() for l in Path(args.corpus).read_text().splitlines() if l.strip()][: args.max_prompts]
    configs = [_parse_config(s.strip()) for s in args.configs.split(",") if s.strip()]
    nL = AutoConfig.from_pretrained(args.model).num_hidden_layers
    layers = list(range(nL)) if args.layers == "all" else [int(s) for s in args.layers.split(",")]
    rng = np.random.default_rng(args.seed)
    print(f"[multikey] model={args.model} layers={layers} configs={[c[0] for c in configs]} "
          f"K={args.n_keys} prompts={len(prompts)} dev={DEV}", flush=True)

    # plaintext capture (attacker's own, Kerckhoffs) + splits + pool
    m = _load(args.model)
    d = m.gpt_neox.embed_in.weight.shape[1]
    table = m.get_input_embeddings().weight.detach().float().cpu().numpy().astype(np.float32)
    vocab = table.shape[0]
    perc, idc = capture(m, tok, prompts, layers, batch_size=args.batch_size)
    del m
    Xclean = {L: _stack(perc[L], idc)[0] for L in layers}
    _, y = _stack(perc[layers[0]], idc)

    distinct = rng.permutation(np.unique(y))
    nd = int(0.7 * distinct.size)
    tr_ids = set(distinct[:nd].tolist())
    dj_tr = np.array([i for i, t in enumerate(y) if t in tr_ids])
    dj_te = np.array([i for i, t in enumerate(y) if t not in tr_ids])
    rs = rng.permutation(y.size)
    nr = int(0.7 * y.size)
    splits = {"disjoint": (dj_tr, dj_te), "rowsplit": (rs[:nr], rs[nr:])}

    def _pool(te):
        true_pool = np.unique(y[te])
        avail = np.setdiff1d(np.arange(vocab, dtype=np.int64), true_pool)
        fill = rng.choice(avail, size=max(0, args.pool_size - true_pool.size), replace=False)
        return np.concatenate([true_pool, fill.astype(np.int64)])
    pools = {k: _pool(v[1]) for k, v in splits.items()}

    # K synthetic attacker keymats (own seeds, NOT the deployment seed)
    Pks = [keymat_gen(d, args.keymat_h, lam=args.keymat_lam, seed=args.attacker_seed + k)[0]
           for k in range(args.n_keys)]

    # the multi-key blind inverter depends ONLY on plaintext reps + attacker keys (NOT the deployment
    # config): precompute once per (layer, split); only the test reps differ per obfuscated config.
    blindW = {}
    for L in layers:
        Xc = Xclean[L]
        for sp, (tr, _te) in splits.items():
            G0 = (Xc[tr].T.astype(np.float64)) @ Xc[tr].astype(np.float64)
            H0 = (Xc[tr].T.astype(np.float64)) @ table[y[tr]].astype(np.float64)
            blindW[(L, sp)] = {"multikey": multikey_ridge_W(G0, H0, Pks)}

    K = int(np.unique(y).size)
    emb_y = table[y]
    records = []
    for cname, rkw in configs:
        if rkw is None:                                      # no-defence: released rep IS the plaintext residual
            Xcfg = Xclean
        else:
            m = _load(args.model)
            reparam_pythia(m, h=args.keymat_h, lam=args.keymat_lam, seed=args.keymat_seed, **rkw)
            perco, idco = capture(m, tok, prompts, layers, batch_size=args.batch_size)
            del m
            assert all(np.array_equal(a, b) for a, b in zip(idco, idc)), "token ids drifted"
            Xcfg = {L: _stack(perco[L], idc)[0] for L in layers}

        for L in layers:
            Xo = Xcfg[L]
            rec = {"config": cname, "layer": L, "obf_dim": int(Xo.shape[1])}
            # PVI probe, IN THREAT MODEL: the reader trains only on attack-accessible fit pairs.
            #   no-defence  → attacker reproduces the plaintext release → default (row-split of Xo).
            #   obfuscated  → attacker has no deployment-basis labels → fit on its OWN synthetic-key
            #                 reps (Xclean·P_a^k, the multi-key set) and score on the deployment reps.
            if rkw is None:
                rec["pvi_bits"] = probe_vcap(Xo, emb_y, y, K)["bits"]
            else:
                rs_tr, rs_te = splits["rowsplit"]
                Pk_pvi = Pks[: args.pvi_fit_keys]
                fit_X = np.concatenate([(Xclean[L][rs_tr] @ Pk).astype(np.float32) for Pk in Pk_pvi], 0)
                fit_y = np.tile(y[rs_tr], len(Pk_pvi))
                rec["pvi_bits"] = probe_vcap(Xo[rs_te], table[y[rs_te]], y[rs_te], K,
                                             fit_X=fit_X, fit_y=fit_y)["bits"]
            for sp, (tr, te) in splits.items():
                pool = pools[sp]
                if rkw is None:                              # no-defence in-model recovery = direct ridge
                    Wr = ridge_W(Xo[tr], table[y[tr]], alpha=1.0)
                    rec[f"{sp}_recovery"] = _ttrsr(Wr, Xo[te], y[te], table, pool)
                else:                                        # obfuscated in-model recovery = multi-key blind
                    rec[f"{sp}_recovery"] = _ttrsr(blindW[(L, sp)]["multikey"], Xo[te], y[te], table, pool)
            records.append(rec)
            print(f"[multikey] {cname:>14} L{L:>2} | pvi={rec['pvi_bits']:.2f} bits | "
                  f"recovery dj={rec['disjoint_recovery']:.3f} rs={rec['rowsplit_recovery']:.3f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "model": args.model, "corpus": args.corpus, "n_prompts": len(prompts),
        "surface": "residual_activations", "defense": "aloepri_covariant_obfuscation",
        "n_keys": args.n_keys, "keymat": {"h": args.keymat_h, "lam": args.keymat_lam, "seed": args.keymat_seed},
        "release_gate_ttrsr": 0.15,
        "note": "recovery = in-model ISA-HiddenState: no-defence (plaintext) inverts the residual directly; "
                "obfuscated configs use the K-synthetic-key multi-key blind inverter (paper §F.1) tested on the "
                "deployment reps. pvi_bits = capacity-matched V-information computed IN THREAT MODEL (the reader "
                "trains only on attack-accessible fit pairs: plaintext reps for no-defence, synthetic own-key reps "
                "for obfuscated; scored on the released reps). Under obfuscation the synthetic-basis reader cannot "
                "transfer to the secret deployment basis, so pvi_bits goes negative (worse than the class prior = "
                "zero accessible info) and tracks the collapsed recovery. disjoint=vocab-disjoint, rowsplit=shared-vocab.",
        "layers": layers, "configs": [c[0] for c in configs], "pool_size": args.pool_size,
        "records": records,
    }, indent=2))
    print(f"[multikey] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
