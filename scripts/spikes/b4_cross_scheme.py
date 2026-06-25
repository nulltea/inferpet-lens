#!/usr/bin/env python3
"""B4 — cross-scheme calibration: does the channel-specific decoupling transfer
across defense families, or is it specific to input-DP's propagation?

B3 found the decoupling on the DEPTH axis under **input-local-DP** (noise injected
at the embedding, propagating L layers → token-id↔attack flips sign at L12). B4
adds a SECOND lossy family — **Shredder static-Laplace injected directly at the
captured layer** — and asks:

  * Does the matched diagonal still dominate under Shredder?
  * **Prediction:** Shredder hits layer L directly (no propagation), so it should
    behave like input-DP@L0 at *every* depth → NO depth sign-flip (token-id stays
    positive at all L). If so, the decoupling is *defense-injection-specific*, not
    a generic property — a sharper claim.
  * **Cross-scheme transfer (C3):** does one calibration curve P_c→A_c fit BOTH
    families? Report per-channel ρ within-DP, within-Shredder, and pooled.

Efficient: Shredder is a post-capture Transform, so we capture CLEAN activations
once per (seed, layer) and sweep the noise in-memory (no GPU re-capture per level).
Loads `results/b3_decoupling_matrix.json` for the input-DP arm. Run via run_in_rocm.sh.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from defenses.aloepri import obfuscate_embedding_table  # noqa: E402
from defenses.shredder import ShredderStaticLaplace  # noqa: E402

from talens.attacks._inversion import ridge_inversion  # noqa: E402
from talens.probes.club import club_mi_upper_bound  # noqa: E402
from talens.probes.vinfo_capacity import v_information_capacity  # noqa: E402
from talens.weights import vma  # noqa: E402
from talens.weights.measures import club_mi_weights  # noqa: E402

CHANNELS = ["token_id", "embedding", "perm_pi"]


@torch.no_grad()
def capture(model, tok, prompts, layers, device):
    per = {L: [] for L in layers}; ids = []
    for p in prompts:
        i = tok(p, return_tensors="pt").input_ids.to(device)
        hs = model(i, output_hidden_states=True, use_cache=False).hidden_states
        for L in layers:
            per[L].append(hs[L + 1][0].float().cpu().numpy())
        ids.append(i[0].cpu().numpy())
    return per, ids


def stack(mats, ids, every=1):
    Xs, ys = [], []
    for m, i in zip(mats, ids):
        n = min(m.shape[0], i.shape[0]); Xs.append(m[:n]); ys.append(i[:n])
    X = np.concatenate(Xs, 0); y = np.concatenate(ys, 0).astype(np.int64)
    return (X[::every], y[::every]) if every > 1 else (X, y)


def _spear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return 0.0 if np.std(a) < 1e-12 or np.std(b) < 1e-12 else float(stats.spearmanr(a, b).statistic)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="unsloth/gemma-2-2b")
    ap.add_argument("--corpus", default="corpora/release-gate-512.txt")
    ap.add_argument("--max-prompts", type=int, default=192)
    ap.add_argument("--layers", default="0,5,12,20")
    ap.add_argument("--levels", default="0,0.1,0.2,0.35,0.5,0.75", help="Shredder b / Π α_e per level (frac of act-RMS)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n-pi", type=int, default=1000)
    ap.add_argument("--cap-dim", type=int, default=64)
    ap.add_argument("--club-max-rows", type=int, default=2500)
    ap.add_argument("--b3", default="results/b3_decoupling_matrix.json")
    ap.add_argument("--out", default="results/b4_cross_scheme.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = [int(s) for s in args.layers.split(",") if s.strip()]
    levels = [float(s) for s in args.levels.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    prompts = [ln.strip() for ln in Path(args.corpus).read_text().splitlines() if ln.strip()][: args.max_prompts]

    print(f"[b4] device={device} layers={layers} levels={levels} seeds={seeds} prompts={len(prompts)}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map=device).eval()
    embed_table = model.get_input_embeddings().weight.detach().float().cpu()
    et = embed_table

    # capture CLEAN activations once per (seed, layer)
    clean = {}
    for seed in seeds:
        torch.manual_seed(20260621 + seed)
        per, ids = capture(model, tok, prompts, layers, device)
        for L in layers:
            clean[(seed, L)] = stack(per[L], ids)
    scale = float(np.median([np.median(np.abs(X)) for (X, _) in clean.values()]))
    print(f"[b4] captured clean; act-|median|={scale:.4f}", flush=True)

    def act_panel(X, y):
        cap = v_information_capacity(X, y, family="pca_softmax", dim=args.cap_dim, l2=0.1)
        rg = ridge_inversion(X, y, et, n_train=10**9, split_mode="vocab") or {}
        return {"token_probe": cap["reader_top1_acc"], "token_attack": rg.get("ttrsr_top1", 0.0),
                "embed_probe": club_mi_upper_bound(X, et[torch.from_numpy(y)].numpy(),
                                                   max_rows=args.club_max_rows)["club_mi_bits"],
                "embed_attack": rg.get("embedding_cosine_similarity", 0.0)}

    pi_cache = {}
    def pi_panel(seed, k):
        if (seed, k) in pi_cache:
            return pi_cache[(seed, k)]
        rng = np.random.default_rng(20260621 + seed)
        W = embed_table[rng.choice(embed_table.shape[0], size=args.n_pi, replace=False)].numpy().astype(np.float32)
        pair = obfuscate_embedding_table(W, alpha_e=levels[k], keymat=False, seed=20260621 + seed)
        out = {"pi_probe": club_mi_weights(pair, bins=64, steps=150, hidden_size=128, seed=0)["club_mi_bits"],
               "pi_attack": vma.run(pair, bins=64, match="hungarian").ttrsr_top1}
        pi_cache[(seed, k)] = out
        return out

    rows = []
    for seed in seeds:
        for k, c in enumerate(levels):
            b = c * scale
            pic = pi_panel(seed, k)
            for L in layers:
                t0 = time.time()
                X0, y = clean[(seed, L)]
                Xn = ShredderStaticLaplace(b=b, seed=seed)(torch.from_numpy(X0), prompt_index=0).numpy() if b > 0 else X0
                ap_ = act_panel(Xn, y)
                rows.append({"seed": seed, "k": k, "b": b, "layer": L, "noise_level": k, **ap_, **pic,
                             "secs": round(time.time() - t0, 1)})
                r = rows[-1]
                print(f"[b4] s{seed} k{k} L{L:>2} | tok P={r['token_probe']:.3f} A={r['token_attack']:.3f} "
                      f"| emb A={r['embed_attack']:.3f} | Π A={r['pi_attack']:.3f} ({r['secs']}s)", flush=True)

    # matrix + per-layer diagonal under Shredder
    P = {"token_id": [r["token_probe"] for r in rows], "embedding": [r["embed_probe"] for r in rows],
         "perm_pi": [r["pi_probe"] for r in rows]}
    A = {"token_id": [r["token_attack"] for r in rows], "embedding": [r["embed_attack"] for r in rows],
         "perm_pi": [r["pi_attack"] for r in rows]}
    M = {pi: {aj: _spear(P[pi], A[aj]) for aj in CHANNELS} for pi in CHANNELS}
    per_layer = {L: {"token_id": _spear([r["token_probe"] for r in rows if r["layer"] == L],
                                        [r["token_attack"] for r in rows if r["layer"] == L]),
                     "embedding": _spear([r["embed_probe"] for r in rows if r["layer"] == L],
                                         [r["embed_attack"] for r in rows if r["layer"] == L])} for L in layers}

    # cross-scheme transfer vs B3 (input-DP)
    transfer = {}
    try:
        b3 = json.load(open(args.b3))["records"]
        for ch, pk, ak in [("token_id", "token_probe", "token_attack"),
                           ("embedding", "embed_probe", "embed_attack"),
                           ("perm_pi", "pi_probe", "pi_attack")]:
            dp_p = [r[pk] for r in b3]; dp_a = [r[ak] for r in b3]
            sh_p = [r[pk] for r in rows]; sh_a = [r[ak] for r in rows]
            transfer[ch] = {"rho_dp": _spear(dp_p, dp_a), "rho_shredder": _spear(sh_p, sh_a),
                            "rho_pooled": _spear(dp_p + sh_p, dp_a + sh_a)}
        b3_per_layer = {int(k): v for k, v in json.load(open(args.b3))["per_layer_diagonal"].items()}
    except Exception as e:
        transfer = {"error": str(e)}; b3_per_layer = {}

    out = {"model": args.model, "defense": "shredder_static_laplace", "levels": levels,
           "layers": layers, "seeds": seeds, "matrix_spearman": M,
           "per_layer_diagonal_shredder": {str(k): v for k, v in per_layer.items()},
           "per_layer_diagonal_inputdp": {str(k): v for k, v in b3_per_layer.items()},
           "cross_scheme_transfer": transfer, "records": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print("\n[b4] === Shredder matrix M[probe][attack] ===")
    print("            " + "".join(f"{a:>12}" for a in CHANNELS))
    for pi in CHANNELS:
        print(f"{pi:>11} " + "".join(f"{M[pi][aj]:>12.3f}" for aj in CHANNELS))
    print("\n[b4] per-layer token-id diagonal (decoupling test):")
    print(f"   Shredder (direct inject): " + " ".join(f"L{L}={per_layer[L]['token_id']:+.2f}" for L in layers))
    if b3_per_layer:
        print(f"   input-DP (propagated)   : " + " ".join(
            f"L{L}={b3_per_layer.get(L, {}).get('token_id', float('nan')):+.2f}" for L in layers))
    print("\n[b4] cross-scheme transfer (per channel: ρ_DP / ρ_Shredder / ρ_pooled):")
    for ch in CHANNELS:
        t = transfer.get(ch, {})
        if "rho_dp" in t:
            print(f"   {ch:>11}: {t['rho_dp']:+.3f} / {t['rho_shredder']:+.3f} / {t['rho_pooled']:+.3f}")
    print(f"[b4] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
