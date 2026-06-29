#!/usr/bin/env python3
"""AloePri partial-τ bootstrap sweep (Item 3) — does a TFMA-style τ-harvest of k token TYPES let a
SUPERVISED inverter generalize to the unharvested tokens? Generic over (attack × surface) via the one
cascade primitive (talens.attacks.dp_inversion.cascade_attack): harvest top-k types → train the attack
on those (deployment-rep, token) pairs (+ optional blind aug) → score recovery on the HELD-OUT types.

Beneficiaries of a partial-τ leak = supervised inverters that consume (rep, token) pairs:
  --attack ridge | decoder        (the per-position inverter; decoder = gated GELU skip net)
  --surface residual              ISA-HiddenState (resid_post), aug = multi-key synthetic
           | kq | kqv_out         ISA score / value surface (attention internals; pure k-pair bootstrap)
           | embed_table          IMA-EmbedRow-ridge on the static obf table W̃ (no capture, CPU)
(Structural / cipher attacks — VMA, ArrowMatch, Gate/Attn-IA — and the harvest producers TFMA/SDA do
NOT consume pairs, so they are out of this family. LM-prior/BeamClean is a separate sequence harness.)

k=0 = blind baseline; the knee in held-out recovery vs k is the bootstrap threshold. --order freq is the
realistic TFMA priority (harvest the most frequent first); random is the low-diversity control.
--label-noise f mislabels a fraction of harvested types (TFMA is ~52% clean; oracle = 0). --no-aug drops
the blind augmentation. GPU: ONE process; run via scripts/run_in_rocm.sh (embed_table is CPU-only).
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

from defenses.aloepri import keymat_gen, obfuscate_embedding_table, reparam_pythia  # noqa: E402
from evals.static_obf.aloepri_score_surface_sweep import _load as score_load, capture as score_capture  # noqa: E402
from evals.dp.dp_leakage_sweep import _stack, capture as resid_capture  # noqa: E402
from talens.attacks.dp_inversion import cascade_attack, ridge_attack, skip_decoder_attack  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ATTACKS = {"ridge": ridge_attack, "decoder": skip_decoder_attack}


def _reparam_kwargs(spec: str):
    if spec == "plaintext":
        return None
    if spec == "keymat_only":
        return dict(config="keymat_only")
    if spec.startswith("alg2"):
        return dict(config="alg2", alpha_e=float(spec.split("@", 1)[1]) if "@" in spec else 0.0, alpha_h=0.0)
    if spec.startswith("full_alg1"):
        # α_h = paper default 0.2 (noises the OUTPUT HEAD only — irrelevant to the residual / embedding-table
        # / attention-value surfaces these attacks read; set for faithfulness, does not change those results).
        return dict(config="full_alg1", alpha_e=float(spec.split("@", 1)[1]) if "@" in spec else 1.0, alpha_h=0.2)
    raise ValueError(f"unknown config {spec!r}")


def _hf_load(model_id):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, attn_implementation="eager", device_map=DEV).eval()


def _pool(types, vocab, pool_size, rng):
    fill = rng.choice(np.setdiff1d(np.arange(vocab), types), size=max(0, pool_size - types.size), replace=False)
    return np.concatenate([types, fill.astype(np.int64)])


def prepare(args, tok, prompts, rng):
    """Per-surface: returns (cells, y, table, pool, order) where cells = [(label, X_deploy, aug_X, aug_y)].
    order = harvest priority (token ids, most-valuable first)."""
    surf, cfg = args.surface, args.config
    if surf == "embed_table":                                        # static IMA-EmbedRow (no model forward)
        from transformers import AutoModelForCausalLM
        W = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).get_input_embeddings(
            ).weight.detach().float().cpu().numpy().astype(np.float32)
        vocab, d = W.shape
        rkw = _reparam_kwargs(cfg) or dict(config="keymat_only")
        # keymat = αₑ 0 (pure keymat); full_alg1@x = αₑ x. Default 0 so keymat_only is noiseless.
        wp = obfuscate_embedding_table(W, alpha_e=rkw.get("alpha_e", 0.0), h=args.keymat_h,
                                       lam=args.keymat_lam, keymat=True, seed=args.keymat_seed)
        X_all = wp.obf[wp.perm].astype(np.float32)                   # X_all[t] = token t's obf row
        sel = rng.choice(vocab, min(args.n_eval, vocab), replace=False)
        X, y = X_all[sel], sel.astype(np.int64)
        # realistic harvest order = corpus token frequency
        cnt = np.bincount(np.concatenate([tok(p).input_ids for p in prompts]), minlength=vocab)
        order = y[np.argsort(cnt[y])[::-1]] if args.order == "freq" else rng.permutation(y)
        pool = _pool(np.unique(y), vocab, max(args.pool_size, y.size), rng)
        return [("table", X, None, None)], y, W, pool, order

    # activation surfaces — capture deployment reps under `cfg`
    layers = [int(s) for s in args.layers.split(",") if s.strip()]
    if surf == "residual":
        m = _hf_load(args.model)
        table = m.get_input_embeddings().weight.detach().float().cpu().numpy().astype(np.float32)
        d, vocab = table.shape[1], table.shape[0]
        perc, idc = resid_capture(m, tok, prompts, layers, batch_size=args.batch_size)
        del m
        Xclean = {L: _stack(perc[L], idc)[0] for L in layers}
        _, y = _stack(perc[layers[0]], idc)
        rkw = _reparam_kwargs(cfg) or dict(config="keymat_only")
        m = _hf_load(args.model); reparam_pythia(m, h=args.keymat_h, lam=args.keymat_lam, seed=args.keymat_seed, **rkw)
        perco, idco = resid_capture(m, tok, prompts, layers, batch_size=args.batch_size); del m
        assert all(np.array_equal(a, b) for a, b in zip(idco, idc)), "ids drifted"
        Xobf = {L: _stack(perco[L], idc)[0] for L in layers}
        aug = {L: (None, None) for L in layers}
        if not args.no_aug and args.n_keys > 0:                      # blind multi-key synthetic aug
            Pks = [keymat_gen(d, args.keymat_h, lam=args.keymat_lam, seed=args.attacker_seed + k)[0]
                   for k in range(args.n_keys)]
            aug = {L: (np.concatenate([(Xclean[L] @ Pk).astype(np.float32) for Pk in Pks], 0),
                       np.tile(y, len(Pks))) for L in layers}
        cells = [(f"L{L}", Xobf[L], aug[L][0], aug[L][1]) for L in layers]
    else:                                                            # kq | kqv_out (pure k-pair bootstrap)
        m = score_load(args.model)
        table = m.get_input_embeddings().weight.detach().float().cpu().numpy().astype(np.float32)
        vocab = table.shape[0]
        rkw = _reparam_kwargs(cfg)
        if rkw is not None:
            reparam_pythia(m, h=args.keymat_h, lam=args.keymat_lam, seed=args.keymat_seed, **rkw)
        feats, y = score_capture(m, tok, prompts, layers, args.window); del m
        cells = [(f"L{L}", feats[surf][L], None, None) for L in layers]
    types, counts = np.unique(y, return_counts=True)
    order = types[np.argsort(counts)[::-1]] if args.order == "freq" else rng.permutation(types)
    pool = _pool(types, vocab, max(args.pool_size, types.size), rng)
    return cells, y, table, pool, order


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--corpus", default="corpora/release-gate-512.txt")
    ap.add_argument("--max-prompts", type=int, default=160)
    ap.add_argument("--attack", default="ridge", choices=list(ATTACKS))
    ap.add_argument("--surface", default="residual", choices=["residual", "kq", "kqv_out", "embed_table"])
    ap.add_argument("--config", default="keymat_only", help="deployment obfuscation (alg2@1.0 for kq/kqv_out)")
    ap.add_argument("--layers", default="0,6")
    ap.add_argument("--ks", default="0,20,100,293,313,1024")
    ap.add_argument("--order", default="freq", choices=["freq", "random"])
    ap.add_argument("--label-noise", type=float, default=0.0)
    ap.add_argument("--n-keys", type=int, default=16)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--n-eval", type=int, default=4000, help="embed_table: token rows scored")
    ap.add_argument("--window", type=int, default=16, help="kq feature window")
    ap.add_argument("--hidden", type=int, default=256, help="decoder hidden")
    ap.add_argument("--epochs", type=int, default=300, help="decoder epochs")
    ap.add_argument("--pool-size", type=int, default=2048)
    ap.add_argument("--keymat-h", type=int, default=128)
    ap.add_argument("--keymat-lam", type=float, default=0.3)
    ap.add_argument("--keymat-seed", type=int, default=0)
    ap.add_argument("--attacker-seed", type=int, default=700000)
    ap.add_argument("--seed", type=int, default=20260629)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="refine-logs/aloepri/aloepri_partial_tau.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    prompts = [l.strip() for l in Path(args.corpus).read_text().splitlines() if l.strip()][: args.max_prompts]
    ks = [int(s) for s in args.ks.split(",") if s.strip()]
    rng = np.random.default_rng(args.seed)
    attack_kw = dict(hidden=args.hidden, epochs=args.epochs, seed=args.seed) if args.attack == "decoder" else {}
    print(f"[partial-τ] attack={args.attack} surface={args.surface} config={args.config} ks={ks} "
          f"order={args.order} noise={args.label_noise} aug={'no' if args.no_aug else args.n_keys} dev={DEV}", flush=True)

    cells, y, table, pool, order = prepare(args, tok, prompts, rng)

    records = []
    for label, Xd, augX, augY in cells:
        for k in ks:
            harvested = order[:k]
            ytr = y
            if args.label_noise > 0 and k > 0:
                noisy = rng.choice(harvested, size=int(args.label_noise * k), replace=False)
                mask = np.isin(y, noisy)
                ytr = y.copy(); ytr[mask] = rng.choice(harvested, size=int(mask.sum()))
            r = cascade_attack(ATTACKS[args.attack], Xd, ytr, harvested, table, pool,
                               X_aug=augX, y_aug=augY, **attack_kw)
            rec = {"surface": args.surface, "attack": args.attack, "cell": label, "k": int(k),
                   "order": args.order, "label_noise": args.label_noise,
                   "unharvested": r["unharvested"], "harvested": r["harvested"],
                   "n_harv_types": r["n_harv_types"], "n_held": r["n_held"]}
            records.append(rec)
            uh = rec["unharvested"]
            print(f"[partial-τ] {label:>6} k={k:>4} | held-out={uh if uh is None else round(uh,3)} "
                  f"(n_held={rec['n_held']})", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "model": args.model, "corpus": args.corpus, "n_prompts": len(prompts),
        "attack": args.attack, "surface": args.surface, "config": args.config,
        "keymat": {"h": args.keymat_h, "lam": args.keymat_lam, "seed": args.keymat_seed},
        "n_keys_aug": 0 if args.no_aug else args.n_keys, "order": args.order, "label_noise": args.label_noise,
        "release_gate_ttrsr": 0.15,
        "note": "two-stage τ-leak cascade (cascade_attack): harvest top-k frequency token TYPES, train the "
                "attack on those deployment-basis pairs (+ blind aug for residual), score recovery on HELD-OUT "
                "types. unharvested = generalization (bootstrap signal); k=0 = blind baseline. Generic over "
                "(attack × surface). embed_table = IMA-EmbedRow-ridge on the static obf table.",
        "layers": args.layers, "ks": ks, "pool_size": args.pool_size, "records": records,
    }, indent=2))
    print(f"[partial-τ] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
