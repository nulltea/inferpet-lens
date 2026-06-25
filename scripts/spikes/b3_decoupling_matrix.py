#!/usr/bin/env python3
"""B3 — the channel-specific decoupling matrix (headline).

Builds the K×K probe×attack Spearman matrix over a SHARED defense-setting grid and
tests **diagonal dominance** (matched probe predicts its own attack better than
others') with bootstrap CIs, per the round-1 reviewer protocol.

Shared grid: input-local-DP budget ε (k levels) × depth L ∈ {0,5,12,20} × seeds.
At each (ε, L, seed) we capture resid_post under DP(ε) and score the two
*activation* channels on the SAME X:
  * token-identity : probe = capacity-PVI reader accuracy ; attack = ridge TTRSR
  * embedding-geom : probe = CLUB I(X; emb[y])           ; attack = ridge cosine
The permutation-Π channel lives on the weight table; its defense knob α_e[k] is
index-aligned to the DP level k (a shared "noise level"), depth-agnostic (tiled
across L):
  * permutation-Π  : probe = CLUB-on-φ ; attack = VMA τ-recovery (AloePri perm-core)

M[i,j] = Spearman over all settings of (probe_i, attack_j). Diagonal-dominance
Δ_i = ρ(i,i) − max_{j≠i} ρ(i,j), bootstrapped over settings. Controls: a random
probe (≈0), retrieval-PVI (dependent ref), a monotone noise-index baseline
(shows a single monotone knob can fake diagonal dominance), and shuffled-pairing.

GPU job — run via scripts/run_in_rocm.sh. Restricted to L0/5/12/20.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/ for defenses
from defenses.aloepri import obfuscate_embedding_table  # noqa: E402

from talens.attacks._inversion import ridge_inversion  # noqa: E402
from talens.probes.club import club_mi_upper_bound  # noqa: E402
from talens.probes.vinfo_capacity import v_information_capacity  # noqa: E402
from talens.probes.vinfo import v_information_retrieval  # noqa: E402
from talens.weights import vma  # noqa: E402
from talens.weights.measures import club_mi_weights  # noqa: E402

CHANNELS = ["token_id", "embedding", "perm_pi"]


class InputDPCover:
    def __init__(self, C, sigma):
        self.C, self.sigma = C, sigma

    def __call__(self, mod, inp, out):
        f = out.float()
        norm = f.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        f = f * (self.C / norm).clamp_max(1.0)
        if self.sigma > 0:
            f = f + self.sigma * torch.randn_like(f)
        return f.to(out.dtype)


@torch.no_grad()
def capture(model, tok, prompts, layers, device):
    per = {L: [] for L in layers}; ids_all = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids.to(device)
        hs = model(ids, output_hidden_states=True, use_cache=False).hidden_states
        for L in layers:
            per[L].append(hs[L + 1][0].float().cpu().numpy())
        ids_all.append(ids[0].cpu().numpy())
    return per, ids_all


def stack(mats, ids_list, every_n=1):
    Xs, ys = [], []
    for m, ids in zip(mats, ids_list):
        n = min(m.shape[0], ids.shape[0]); Xs.append(m[:n]); ys.append(ids[:n])
    X = np.concatenate(Xs, 0); y = np.concatenate(ys, 0).astype(np.int64)
    return (X[::every_n], y[::every_n]) if every_n > 1 else (X, y)


def _spear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(stats.spearmanr(a, b).statistic)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="unsloth/gemma-2-2b")
    ap.add_argument("--corpus", default="corpora/release-gate-512.txt")
    ap.add_argument("--max-prompts", type=int, default=192)
    ap.add_argument("--layers", default="0,5,12,20")
    ap.add_argument("--epsilons", default="inf,4096,2048,1024,512,256")
    ap.add_argument("--alphas", default="0,0.1,0.2,0.35,0.5,0.75", help="Π noise per ε level (index-aligned)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--clip-percentile", type=float, default=99.9)
    ap.add_argument("--n-pi", type=int, default=1000, help="embed rows for the Π weight-surface channel")
    ap.add_argument("--cap-dim", type=int, default=64)
    ap.add_argument("--club-max-rows", type=int, default=2500)
    ap.add_argument("--every-n", type=int, default=1)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true", help="1 ε × 1 layer × 1 seed sanity")
    ap.add_argument("--out", default="results/b3_decoupling_matrix.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = [int(s) for s in args.layers.split(",") if s.strip()]
    eps_list = [math.inf if s.strip().lower().startswith("inf") else float(s)
                for s in args.epsilons.split(",") if s.strip()]
    alphas = [float(s) for s in args.alphas.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.smoke:
        layers, eps_list, alphas, seeds = layers[:1], eps_list[:1], alphas[:1], seeds[:1]
    assert len(alphas) == len(eps_list), "alphas must be index-aligned to epsilons"
    prompts = [ln.strip() for ln in Path(args.corpus).read_text().splitlines() if ln.strip()][: args.max_prompts]

    print(f"[b3] device={device} layers={layers} eps={eps_list} seeds={seeds} "
          f"prompts={len(prompts)} grid={len(eps_list)}x{len(layers)}x{len(seeds)}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map=device).eval()
    embed_table = model.get_input_embeddings().weight.detach().float().cpu()
    d = embed_table.shape[1]

    # clip C from runtime embed-norms
    cal = []
    ch = model.model.embed_tokens.register_forward_hook(
        lambda m, i, o: cal.append(o.float().norm(dim=-1).flatten().cpu()))
    with torch.no_grad():
        for p in prompts[: min(64, len(prompts))]:
            model(tok(p, return_tensors="pt").input_ids.to(device), use_cache=False)
    ch.remove()
    cal = torch.cat(cal).numpy()
    C = float(np.percentile(cal, args.clip_percentile)); med = float(np.median(cal))
    z = math.sqrt(2 * math.log(1.25 / args.delta))
    print(f"[b3] d={d} C={C:.3f} med={med:.3f}", flush=True)

    et = embed_table  # for ridge/CLUB targets

    def activation_panel(X, y):
        cap = v_information_capacity(X, y, family="pca_softmax", dim=args.cap_dim, l2=0.1)
        rg = ridge_inversion(X, y, et, n_train=10**9, split_mode="vocab")
        retr = v_information_retrieval(X, y, et, split_mode="row")["v_information_bits"]
        club = club_mi_upper_bound(X, et[torch.from_numpy(y)].numpy(), max_rows=args.club_max_rows)["club_mi_bits"]
        return {"token_probe": cap["reader_top1_acc"], "token_attack": (rg or {}).get("ttrsr_top1", 0.0),
                "embed_probe": club, "embed_attack": (rg or {}).get("embedding_cosine_similarity", 0.0),
                "retr_pvi": retr}

    # Π channel: depend only on (seed, k). Precompute per (seed,k).
    pi_cache: dict[tuple[int, int], dict] = {}
    def pi_panel(seed, k):
        key = (seed, k)
        if key in pi_cache:
            return pi_cache[key]
        rng = np.random.default_rng(20260620 + seed)
        sel = rng.choice(embed_table.shape[0], size=args.n_pi, replace=False)
        W = embed_table[sel].numpy().astype(np.float32)
        pair = obfuscate_embedding_table(W, alpha_e=alphas[k], keymat=False, seed=20260620 + seed)
        out = {"pi_probe": club_mi_weights(pair, bins=64, steps=150, hidden_size=128, seed=0)["club_mi_bits"],
               "pi_attack": vma.run(pair, bins=64, match="hungarian").ttrsr_top1}
        pi_cache[key] = out
        return out

    rows = []  # each: dict with all 3 probes + 3 attacks + meta + controls
    for seed in seeds:
        for k, eps in enumerate(eps_list):
            sigma = 0.0 if math.isinf(eps) else C * z / eps
            torch.manual_seed(20260620 + seed + (0 if math.isinf(eps) else int(eps)))
            h = model.model.embed_tokens.register_forward_hook(InputDPCover(C, sigma))
            per, ids = capture(model, tok, prompts, layers, device)
            h.remove()
            pic = pi_panel(seed, k)
            for L in layers:
                t0 = time.time()
                X, y = stack(per[L], ids, args.every_n)
                ap_ = activation_panel(X, y)
                row = {"seed": seed, "k": k, "epsilon": (None if math.isinf(eps) else eps),
                       "layer": L, "noise_level": k, **ap_, **pic, "secs": round(time.time() - t0, 1)}
                rows.append(row)
                print(f"[b3] s{seed} k{k} L{L:>2} | tok P={row['token_probe']:.3f} A={row['token_attack']:.3f} "
                      f"| emb P={row['embed_probe']:.0f} A={row['embed_attack']:.3f} "
                      f"| Π P={row['pi_probe']:.0f} A={row['pi_attack']:.3f} ({row['secs']}s)", flush=True)

    # ---- assemble matrix ----
    P = {"token_id": [r["token_probe"] for r in rows],
         "embedding": [r["embed_probe"] for r in rows],
         "perm_pi": [r["pi_probe"] for r in rows]}
    A = {"token_id": [r["token_attack"] for r in rows],
         "embedding": [r["embed_attack"] for r in rows],
         "perm_pi": [r["pi_attack"] for r in rows]}
    M = {pi: {aj: _spear(P[pi], A[aj]) for aj in CHANNELS} for pi in CHANNELS}

    # bootstrap Δ_i over setting rows
    n = len(rows); rng = np.random.default_rng(0)
    deltas = {c: [] for c in CHANNELS}
    Pa = {c: np.array(P[c]) for c in CHANNELS}; Aa = {c: np.array(A[c]) for c in CHANNELS}
    for _ in range(args.boot if not args.smoke else 50):
        idx = rng.integers(0, n, n)
        for i, ci in enumerate(CHANNELS):
            diag = _spear(Pa[ci][idx], Aa[ci][idx])
            off = max(_spear(Pa[ci][idx], Aa[cj][idx]) for cj in CHANNELS if cj != ci)
            deltas[ci].append(diag - off)
    delta_ci = {c: {"mean": float(np.mean(deltas[c])),
                    "lo": float(np.percentile(deltas[c], 2.5)),
                    "hi": float(np.percentile(deltas[c], 97.5))} for c in CHANNELS}

    # per-layer diagonal (expose the depth sign-flip)
    per_layer = {}
    for L in layers:
        lr = [r for r in rows if r["layer"] == L]
        per_layer[L] = {
            "token_id": _spear([r["token_probe"] for r in lr], [r["token_attack"] for r in lr]),
            "embedding": _spear([r["embed_probe"] for r in lr], [r["embed_attack"] for r in lr])}

    # controls
    rand = rng.standard_normal(n)
    monotone = [r["noise_level"] for r in rows]
    controls = {
        "random_probe_vs_attacks": {aj: _spear(rand, A[aj]) for aj in CHANNELS},
        "monotone_index_vs_attacks": {aj: _spear(monotone, A[aj]) for aj in CHANNELS},
        "retr_pvi_vs_token_attack_dependent": _spear([r["retr_pvi"] for r in rows], A["token_id"]),
        "shuffled_pairing_token": _spear(P["token_id"], list(np.array(A["token_id"])[rng.permutation(n)])),
    }

    sign_flips = [{"layer": L, "channel": c, "rho": per_layer[L][c]}
                  for L in layers for c in ("token_id", "embedding")
                  if per_layer[L][c] < 0]

    out = {"model": args.model, "grid": {"epsilons": [None if math.isinf(e) else e for e in eps_list],
           "alphas": alphas, "layers": layers, "seeds": seeds, "n_settings": n},
           "matrix_spearman": M, "diagonal_dominance_delta": delta_ci,
           "per_layer_diagonal": {str(k): v for k, v in per_layer.items()},
           "controls": controls, "sign_flips": sign_flips, "records": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print("\n[b3] === Spearman matrix M[probe][attack] ===")
    print("            " + "".join(f"{a:>12}" for a in CHANNELS))
    for pi in CHANNELS:
        print(f"{pi:>11} " + "".join(f"{M[pi][aj]:>12.3f}" for aj in CHANNELS))
    print("\n[b3] diagonal-dominance Δ_i (95% CI):")
    for c in CHANNELS:
        dd = delta_ci[c]; ok = "✓" if dd["lo"] > 0 else "✗"
        print(f"  {c:>11}: Δ={dd['mean']:+.3f} [{dd['lo']:+.3f},{dd['hi']:+.3f}] {ok}")
    print(f"[b3] per-layer diagonal: {per_layer}")
    print(f"[b3] sign-flips: {sign_flips}")
    print(f"[b3] controls: rand≈{controls['random_probe_vs_attacks']}")
    print(f"[b3] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
