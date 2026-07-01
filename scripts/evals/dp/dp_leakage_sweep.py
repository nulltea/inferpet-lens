#!/usr/bin/env python3
"""DP leakage sweep — reusable cross-layer eval.

One configurable sweep over a defended representation:
  capture residual-stream reps under LOCAL differential privacy (Gaussian mechanism on the
  input embedding) at the SELECTED layers, run the SELECTED attacks (linear ridge /
  non-linear decoder) and the SELECTED attack-independent probes (CLUB MI upper bound,
  V_cap capacity-matched reader), and report bits beside a readable per-secret readout
  (token-id top-1 recovery + reader perplexity).

Promoted from the now-retired validated spikes scripts/spikes/{b2_propagated_dp,b2_lpos_decoder}
into a clean, pluggable module: ATTACKS and PROBES are registries; --layers and --epsilons are
CLI lists, so the same sweep serves any (layers × ε × attacks × probes) cut.

Threat model WEIGHTS-PUB: adversary knows weights + embedding table + published (C, σ) and
synthesizes its own (noised-rep, token) training pairs. The probes never see an attack output.

Conventions (kept consistent with the existing project DP sweeps so results splice together):
  * σ = C·z/ε with z = √(2 ln(1.25/δ)) — add/remove-to-zero adjacency, per-row sensitivity C.
  * recovery = top-1 over the candidate POOL (default 2048), not full vocab.
  * *_sel = recovery − a CLEAN-rep label-shuffle floor (memorization baseline, fixed per layer).
  * probes are computed on ALL captured rows (probe_split=all); attacks score the test split.

GPU: ONE process at a time; run via scripts/run_in_rocm.sh. Output JSON under refine-logs/<surface>/.

Example (the residual cross-layer DP table, campaign-D Task 2):
  scripts/run_in_rocm.sh python3 scripts/evals/dp_leakage_sweep.py \
      --layers 0,5,12,20 --epsilons inf,512,256,128 --attacks ridge,decoder \
      --probes club,vcap --out refine-logs/dp-decoder-grid/dp_leakage_sweep.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # scripts/ for defenses.*
from talens.probes.club import club_mi_upper_bound  # noqa: E402
from talens.probes.vinfo_capacity import v_information_capacity  # noqa: E402
from talens.report import perplexity_from_bits  # noqa: E402
from talens.attacks import ridge_attack, skip_decoder_attack, logit_lens_attack  # noqa: E402
from defenses.local_dp import LocalDP  # noqa: E402
import functools  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def capture(model, tok, prompts, layers, batch_size=32):
    """Per-token residual_post at each requested layer + the token ids (one array per prompt).

    Batched + right-padded to saturate the iGPU (a 160M model is idle at batch=1). Right padding
    keeps real-token rotary positions and causal context identical to the unbatched forward; pads are
    masked (attention_mask) and dropped per row via the real length, so reps match unbatched exactly.
    """
    per = {L: [] for L in layers}
    ids = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True)
        input_ids, mask = enc.input_ids.to(DEV), enc.attention_mask.to(DEV)
        hs = model(input_ids, attention_mask=mask, output_hidden_states=True, use_cache=False).hidden_states
        lens = mask.sum(dim=1).tolist()
        for b, n in enumerate(lens):
            for L in layers:
                per[L].append(hs[L + 1][b, :n].float().cpu().numpy())
            ids.append(input_ids[b, :n].cpu().numpy())
    return per, ids


def _stack(per_L, ids):
    Xs, ys = [], []
    for m, t in zip(per_L, ids):
        n = min(m.shape[0], t.shape[0])
        Xs.append(m[:n])
        ys.append(t[:n])
    return np.concatenate(Xs, 0).astype(np.float32), np.concatenate(ys, 0).astype(np.int64)


# ── attacks: imported from the library (src/talens/attacks/dp_inversion.py) ──
#   lens = CE tuned-lens affine; declens = + gated GELU correction (the fair non-linearity test).
ATTACKS = {
    "ridge": ridge_attack,
    "decoder": skip_decoder_attack,
    "lens": functools.partial(logit_lens_attack, nonlinear=False),
    "declens": functools.partial(logit_lens_attack, nonlinear=True),
}


# ───────────────────────── probes (attack-independent; bits + readout) ─────────────────────────
def probe_club(X, E, y, K, *, club_max_rows=600, **_):
    """CLUB variational MI upper bound I(rep;token), in bits. Continuous MI → no token perplexity."""
    bits = club_mi_upper_bound(X, E, max_rows=club_max_rows, seed=0)["club_mi_bits"]
    return {"bits": None if bits is None else float(bits), "bits_kind": "mi_upper_bound"}


def _vcap_ppl(pvi, y):
    """reader perplexity = 2^(H_cond), H_cond = H_prior − PVI, clamped to [1, 2^H_prior]."""
    if pvi is None:
        return None
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    h_prior = float(-(p * np.log2(p)).sum())
    return perplexity_from_bits(min(max(h_prior - pvi, 0.0), h_prior))


def probe_vcap(X, E, y, K, *, fit_X=None, fit_y=None, **_):
    """V_cap capacity-matched predictive V-information (bits) + a readable reader readout.

    Emits TWO families on the SAME (X, y) to adjudicate whether the k=64 PCA reduction discards
    token signal (PCA keeps top-VARIANCE axes; under DP noise variance ≠ discriminative):
      * `bits`        — pca_softmax, dim=64  : capacity-bounded by truncation (current headline).
      * `bits_gauss`  — gauss, dim=768       : LDA-diagonal in a full-rank PCA rotation (no
        truncation; capacity bounded by the diagonal-covariance structure, not by dropping dims).
    gauss ≥ pca ⇒ PCA-64 was discarding signal (use gauss as the no-truncation reference); they
    agree ⇒ the k=64 capacity bound was the whole story. Readout: reader_top1_acc + perplexity.
    H_prior is the EMPIRICAL token-label entropy (PVI is anchored to the empirical prior, not log₂K).

    `fit_X` / `fit_y` (optional): the attack-accessible (representation, label) pairs the reader is
    allowed to train on. Omit for attacker-reproducible releases (DP); supply synthetic own-key reps
    for secret-key schemes so the probe is scored on the released X but never trains on a
    deployment-basis true-label pair the attack could not obtain. See v_information_capacity."""
    r = v_information_capacity(X, y, family="pca_softmax", dim=64, l2=0.1, fit_X=fit_X, fit_y=fit_y)
    g = v_information_capacity(X, y, family="gauss", dim=768, fit_X=fit_X, fit_y=fit_y)  # full-rank, no trunc
    pvi = r.get("v_information_bits")
    return {"bits": pvi, "bits_kind": "capacity_v_info",
            "reader_top1_acc": r.get("reader_top1_acc"), "perplexity": _vcap_ppl(pvi, y),
            "bits_gauss": g.get("v_information_bits"), "gauss_eff_dim": g.get("eff_dim"),
            "gauss_top1_acc": g.get("reader_top1_acc")}


def probe_mdl(X, E, y, K, *, mdl_classes=256, **_):
    """Voita & Titov (2020) FAITHFUL MDL — prequential online code length of the discrete token-id
    given the rep, under a CLOSED-SET softmax probe (top-`mdl_classes` token-ids). This is the
    canonical class-MDL: train the probe on a growing data prefix, pay the next block's CE; sum =
    online codelength = area under the learning curve. NOT the retrieval variant (ridge X→emb is
    circular with the attack) — that one is dropped.

    Headline `bits` is leakage-POSITIVE: info = uniform_code − online_code (rises with leakage →
    correlates the right way with recovery). compression = uniform/online and the Whitney SDL are
    kept beside it (SDL is non-monotone across ε, so info — not SDL — is the representative signal).
    """
    from talens.probes.mdl import online_code_length
    r = online_code_length(X, y, max_classes=mdl_classes, seed=0)
    code = r.get("online_code_length_bits")
    uni = r.get("uniform_code_length_bits")
    info = None if (code is None or uni is None) else float(uni - code)
    return {"bits": info, "bits_kind": "mdl_info_class",
            "mdl_code_bits": code, "compression": r.get("compression"),
            "sdl_bits": r.get("surplus_description_length_bits"),
            "num_classes": r.get("num_classes"),
            "floor_ce_bits_per_row": r.get("floor_ce_bits_per_row")}


def probe_ig(X, E, y, K, *, X_clean=None, ig_ridge=1e-6, unit=False, **_):
    """Geometry-only Gaussian channel-MI I_G of the DP channel at this layer (attack-INDEPENDENT).

    Whitens the signal covariance (CLEAN rep Σ) by the empirical propagated-NOISE covariance N
    (noised − clean): I_G = ½ Σ log2(1+μ_i), μ_i = eig(N^{-1/2} Σ N^{-1/2}). No fitted predictor →
    independent of every attack/probe-family. Exact token-MI ceiling only at L0 (clean rep ↔ token
    bijection); at depth it is representation-survival MI (a geometry upper bound on token-MI).

    unit=True → DIRECTION-SPACE control: unit-normalise each clean+noised row first, removing residual
    magnitude growth with depth, to test whether a depth bump is angular structure vs. just scale."""
    if X_clean is None:
        return {"bits": None, "bits_kind": "ig_channel"}
    noise = X - X_clean
    if float(noise.std()) < 1e-9:
        return {"bits": float("inf"), "bits_kind": "ig_channel", "note": "sigma~0"}
    if unit:  # direction-space: strip per-row magnitude from clean AND noised, then re-form the noise
        Xu = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
        Xcu = X_clean / np.clip(np.linalg.norm(X_clean, axis=1, keepdims=True), 1e-9, None)
        X_clean, noise = Xcu, (Xu - Xcu)
    Xc = torch.from_numpy(np.ascontiguousarray(X_clean)).to(DEV).double()
    Nn = torch.from_numpy(np.ascontiguousarray(noise)).to(DEV).double()
    Sig, Nz = torch.cov(Xc.T), torch.cov(Nn.T)
    w, Vv = torch.linalg.eigh(0.5 * (Nz + Nz.T))
    w = torch.clamp(w, min=ig_ridge * float(w.max()))      # PD floor (conditioning of N^{-1/2})
    Ninv = (Vv / w.sqrt()) @ Vv.T                            # N^{-1/2}
    Wm = Ninv @ Sig @ Ninv
    mu = torch.clamp(torch.linalg.eigvalsh(0.5 * (Wm + Wm.T)), min=0.0)
    return {"bits": float((0.5 * torch.log2(1.0 + mu)).sum()), "bits_kind": "ig_channel_survival",
            "d_eff": int((mu >= 1.0).sum())}


def probe_sep(X, E, y, K, *, sep_classes=None, **_):
    """Token-class separability (Voita-style) of a fixed closed-class set, attack-INDEPENDENT.

    Converse (geometry-only Bhattacharyya/Fano channel-MI, headline `bits`) + achievable (class-probe
    MDL info) on the rows whose token id ∈ a small class set (e.g. {is,are,was,were}). Direct
    TOKEN-class separability — adjudicates whether the L20 information peak is real token info ridge
    can't read (separability peaks at L20) or a representation-context artifact (separability tracks
    recovery's monotone decline). `sep_classes` maps class label → token ids (resolved in main).
    """
    from talens.probes.class_separability import class_separability
    if not sep_classes:
        return {"bits": None, "bits_kind": "class_separability", "note": "no --sep-words"}
    r = class_separability(X, y, sep_classes, seed=0)
    return {"bits": r.get("bits"), "bits_kind": "class_separability",
            "bhat_dist": r.get("bhat_dist"), "mi_converse_bits": r.get("mi_converse_bits"),
            "mdl_info_bits": r.get("mdl_info_bits"), "compression": r.get("compression"),
            "p_e_ub": r.get("p_e_ub"), "p_e_lb": r.get("p_e_lb"),
            "k_present": r.get("k_present"), "n_rows": r.get("n_rows"),
            "labels": r.get("labels"), "per_class_n": r.get("per_class_n"),
            "coords": r.get("coords")}


PROBES = {"club": probe_club, "vcap": probe_vcap, "mdl": probe_mdl, "ig": probe_ig,
          "ig_unit": functools.partial(probe_ig, unit=True), "sep": probe_sep}


# ───────────────────────── sweep ─────────────────────────
def _spearman(a, b):
    """Spearman ρ via numpy (rank then Pearson). Dependency-free; ties broken by order
    (fine for the distinct ε-sweep values here). Returns None for degenerate input."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--corpus", default="corpora/release-gate-512.txt")
    ap.add_argument("--max-prompts", type=int, default=160)
    ap.add_argument("--layers", default="0,2,4,6,8,10,11", help="comma list of layer indices (pythia-160m has 12 blocks)")
    ap.add_argument("--epsilons", default="inf,512,256,128", help="comma list; 'inf' = clip-only")
    ap.add_argument("--attacks", default="ridge,lens,declens", help=f"subset of {sorted(ATTACKS)}")
    ap.add_argument("--probes", default="club,vcap", help=f"subset of {sorted(PROBES)}")
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--clip-percentile", type=float, default=99.9)
    ap.add_argument("--pool-size", type=int, default=2048)
    ap.add_argument("--hidden", type=int, default=384, help="MLP-branch width of the gated linear-skip decoder (narrow ≤ input)")
    ap.add_argument("--epochs", type=int, default=500, help="decoder max epochs (early-stopped on a disjoint val split)")
    ap.add_argument("--club-max-rows", type=int, default=600)
    ap.add_argument("--sep-words", default="is,are,was,were",
                    help="separability class words for the 'sep' probe (single-token, leading-space matched)")
    ap.add_argument("--seed", type=int, default=20260621, help="base seed (split, floor, probe/attack init — fixed across noise seeds)")
    ap.add_argument("--seeds", default="", help="comma list of DP-noise seeds for multi-seed error bars (varies ONLY the noise draw); empty = single run at --seed")
    ap.add_argument("--out", default="refine-logs/dp-decoder-grid/dp_leakage_sweep.json")
    ap.add_argument("--cache-dir", default="", help="if set, dump victim Xte / clean X0 / shared meta "
                    "as .npy so probe-family experiments re-run offline (CPU) without re-capturing")
    ap.add_argument("--batch-size", type=int, default=32, help="prompts per forward pass (batched + "
                    "right-padded to saturate the GPU; reps match unbatched exactly)")
    args = ap.parse_args()

    layers = [int(s) for s in args.layers.split(",") if s.strip()]
    eps_list = [math.inf if s.strip().lower().startswith("inf") else float(s)
                for s in args.epsilons.split(",") if s.strip()]
    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    probes = [p.strip() for p in args.probes.split(",") if p.strip()]
    if not attacks or not probes or not layers or not eps_list:
        ap.error("need at least one each of --layers, --epsilons, --attacks, --probes")
    for a in attacks:
        if a not in ATTACKS:
            ap.error(f"unknown attack {a!r}; choose from {sorted(ATTACKS)}")
    for p in probes:
        if p not in PROBES:
            ap.error(f"unknown probe {p!r}; choose from {sorted(PROBES)}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = [l.strip() for l in Path(args.corpus).read_text().splitlines() if l.strip()][: args.max_prompts]
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:           # GPT-NeoX tokenizer ships no pad token; reuse EOS for masking
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"          # real-token positions/context unchanged vs unbatched
    # fp32, NOT bf16: batched bf16 GEMM on ROCm is not batch-invariant (verified: reps drift ~18% by
    # mid-depth vs unbatched, fp32 drifts ~1e-5). Batched capture is required to saturate the iGPU, and
    # the I_G probe reads noise = Xte − X0, so a batch-dependent rounding artifact would corrupt the
    # small-ε noise estimate. fp32 is deterministic + batch-invariant and trivially cheap at 160M.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, attn_implementation="eager", device_map=DEV
    ).eval()
    table = model.get_input_embeddings().weight.detach().float().cpu().numpy().astype(np.float32)
    vocab = table.shape[0]

    # separability class set: resolve each word to its single leading-space token id (skip multi-token)
    sep_classes = None
    if "sep" in probes:
        sep_classes = {}
        for w in (x.strip() for x in args.sep_words.split(",") if x.strip()):
            tids = tok(" " + w, add_special_tokens=False).input_ids
            if len(tids) == 1:
                sep_classes[w] = (int(tids[0]),)
            else:
                print(f"[dp-sweep] WARN sep word {w!r} → {len(tids)} tokens {tids}; skipped (need single-token)", flush=True)
        print(f"[dp-sweep] sep classes: {sep_classes}", flush=True)

    # clip C from runtime embedding norms (so clip-only ≈ clean; the curve is noise-driven)
    cal = []
    h = model.get_input_embeddings().register_forward_hook(  # model-agnostic (gemma/pythia/qwen)
        lambda m, i, o: cal.append(o.float().norm(dim=-1).flatten().cpu())
    )
    try:
        with torch.no_grad():
            for p in prompts[:48]:
                model(tok(p, return_tensors="pt").input_ids.to(DEV), use_cache=False)
    finally:
        h.remove()
    C = float(np.percentile(torch.cat(cal).numpy(), args.clip_percentile))
    z = math.sqrt(2 * math.log(1.25 / args.delta))
    rng = np.random.default_rng(args.seed)
    print(f"[dp-sweep] C={C:.3f} layers={layers} eps={eps_list} attacks={attacks} probes={probes} "
          f"prompts={len(prompts)} dev={DEV}", flush=True)

    # clean capture once: defines the vocab-disjoint split + candidate pool + per-attack shuffle floor
    perc, idc = capture(model, tok, prompts, layers, batch_size=args.batch_size)
    # The token-id split, candidate pool and label-shuffle are LAYER-INDEPENDENT — token ids are identical
    # across layers (same prompts/positions), only the reps differ. Compute them ONCE so every depth is
    # compared on the SAME held-out tokens and the SAME pool. (Previously these were drawn inside the layer
    # loop with an advancing RNG, giving each layer a different split/pool — a depth-comparison confound.)
    _, y0 = _stack(perc[layers[0]], idc)
    distinct = rng.permutation(np.unique(y0))
    ntr = int(0.7 * distinct.size)
    tr_ids, te_ids = set(distinct[:ntr].tolist()), set(distinct[ntr:].tolist())
    tr = np.array([i for i, t in enumerate(y0) if t in tr_ids])
    te = np.array([i for i, t in enumerate(y0) if t in te_ids])
    true_pool = np.unique(y0[te])
    if true_pool.size > args.pool_size:
        print(f"[dp-sweep] WARN: {true_pool.size} test tokens > pool-size {args.pool_size}; "
              f"pool = all test tokens", flush=True)
    avail = np.setdiff1d(np.arange(vocab, dtype=np.int64), true_pool)
    fill = rng.choice(avail, size=max(0, args.pool_size - true_pool.size), replace=False)
    pool = np.concatenate([true_pool, fill.astype(np.int64)])  # true_pool ⊆ pool, disjoint fill
    permsh = rng.permutation(tr.size)  # label-shuffle control → CLEAN-rep recovery floor per attack
    split = {}
    for L in layers:
        X0, y = _stack(perc[L], idc)
        assert np.array_equal(y, y0), "token ids must be identical across layers (shared split)"
        emb_y = table[y]
        floor = {}
        for a in attacks:
            yhat = ATTACKS[a](X0[tr], emb_y[tr][permsh], X0[te], table[pool], pool,
                              ytr=y[tr][permsh], full_emb=table,  # CE: same shuffle as Etr for the floor
                              hidden=args.hidden, epochs=args.epochs, seed=args.seed)
            floor[a] = float((yhat == y[te]).mean())
        split[L] = dict(y=y, tr=tr, te=te, pool=pool, emb_y=emb_y, floor=floor,
                        K=int(np.unique(y).size), X0=X0)  # X0 = clean rep (for I_G noise estimate)

    # cache shared (layer/ε-independent) meta + clean reps once, so any probe re-runs offline on
    # the SAME rows the live run measured (probes consume Xte / X0 / y — Xtr is attack-only, skipped).
    cdir = Path(args.cache_dir) if args.cache_dir else None
    if cdir:
        cdir.mkdir(parents=True, exist_ok=True)
        np.savez(cdir / "meta.npz", y=y0, tr=tr, te=te, pool=pool, table=table, layers=np.array(layers),
                 draw_seed_note=np.array("Xtr draw seed = base+1009*eps_idx; Xte draw seed = "
                                         "base+1009*eps_idx+5_000_003 — independent DP draws"))
        for L in layers:
            np.save(cdir / f"clean_L{L}.npy", split[L]["X0"])
        print(f"[dp-sweep] cached meta + clean reps → {cdir}", flush=True)

    # noise seeds: hoist all noise-INDEPENDENT work (model, clean capture, split, floor) above; only
    # the noised capture + probes + attacks repeat per seed. Fixed split + probe init → isolates the
    # DP-noise-draw variance (the multi-seed error bar on the probe/recovery values).
    noise_seeds = [int(s) for s in args.seeds.split(",") if s.strip()] if args.seeds else [args.seed]
    records = []
    for ei, eps in enumerate(eps_list):
        sigma = 0.0 if math.isinf(eps) else C * z / eps
        for si, nseed in enumerate(noise_seeds):
            def _noised_capture(noise_seed):
                torch.manual_seed(noise_seed)
                hk = model.get_input_embeddings().register_forward_hook(LocalDP(C, sigma))  # model-agnostic
                try:
                    per, ids_chk = capture(model, tok, prompts, layers, batch_size=args.batch_size)
                finally:
                    hk.remove()
                assert all(np.array_equal(a, b) for a, b in zip(ids_chk, idc)), "token ids drifted under noise"
                return per
            # WEIGHTS-PUB realism: the adversary trains on its OWN noise draw; the victim's released
            # rep is an INDEPENDENT draw it never sees. Two captures ⇒ train/test DP randomness is
            # structurally disjoint (not just per-position-iid). Train on per_tr, score/probe per_te.
            tr_seed = nseed + 1009 * ei                    # adversary/train DP draw seed
            te_seed = nseed + 1009 * ei + 5_000_003        # victim/released DP draw seed (≠ tr_seed)
            per_tr = _noised_capture(tr_seed)
            per_te = _noised_capture(te_seed)
            for L in layers:
                Xtr, _ = _stack(per_tr[L], idc)   # adversary-train draw
                Xte, _ = _stack(per_te[L], idc)   # victim-test draw (the released surface)
                if cdir:  # cache BOTH independent draws: adversary-train (tr_seed) + victim/released
                    es_c = "inf" if math.isinf(eps) else f"{eps:g}"  # (te_seed). Distinct DP seeds.
                    np.save(cdir / f"Xtr_eps{es_c}_seed{nseed}_L{L}.npy", Xtr)
                    np.save(cdir / f"Xte_eps{es_c}_seed{nseed}_L{L}.npy", Xte)
                s = split[L]
                y, tr, te, pool, emb_y, floor, K = s["y"], s["tr"], s["te"], s["pool"], s["emb_y"], s["floor"], s["K"]
                pe = table[pool]
                rec = {"epsilon": (None if math.isinf(eps) else eps), "layer": L, "sigma": sigma, "seed": nseed}
                for a in attacks:
                    yhat = ATTACKS[a](Xtr[tr], emb_y[tr], Xte[te], pe, pool,
                                      ytr=y[tr], full_emb=table,  # CE attacks use ids + frozen table
                                      hidden=args.hidden, epochs=args.epochs, seed=args.seed)
                    top1 = float((yhat == y[te]).mean())
                    rec[a] = top1
                    rec[f"{a}_sel"] = top1 - floor[a]
                for p in probes:
                    out = PROBES[p](Xte, emb_y, y, K, club_max_rows=args.club_max_rows,
                                    full_emb=table, pool_size=args.pool_size, X_clean=s["X0"],
                                    sep_classes=sep_classes)
                    for k, v in out.items():
                        rec[f"{p}_{k}"] = v
                records.append(rec)
                es = "inf" if math.isinf(eps) else f"{eps:g}"
                atxt = " ".join(f"{a}={rec[a]:.3f}" for a in attacks)
                ptxt = " ".join(
                    f"{p}={rec.get(p + '_bits')}" + (f"(ppl {rec[p + '_perplexity']:.1f})" if rec.get(p + "_perplexity") else "")
                    for p in probes
                )
                print(f"[dp-sweep] ε={es:>5} L{L:>2} seed={nseed} | {atxt} | {ptxt}", flush=True)

    # per (layer, attack, probe): does the attack's selectivity track the probe bits across ε?
    corr = {}
    for L in layers:
        r = [x for x in records if x["layer"] == L]
        corr[f"L{L}"] = {}
        for a in attacks:
            for p in probes:
                bk = f"{p}_bits"
                pairs = [(x[f"{a}_sel"], x[bk]) for x in r
                         if x.get(bk) is not None and np.isfinite(x[bk])]
                if len(pairs) >= 2:
                    xs, ys_ = zip(*pairs)
                    corr[f"L{L}"][f"{a}_sel_vs_{p}"] = _spearman(xs, ys_)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "model": args.model, "corpus": args.corpus, "n_prompts": len(prompts),
        "defense": "local_dp", "sigma_convention": "sigma = C*z/eps (sensitivity C, add/remove-to-zero adjacency)",
        "noise_protocol": "channel-matched + independent DP draws: attack trains on an adversary noise "
                          "draw, scored on an INDEPENDENT victim draw (the released surface); probes "
                          "measure the victim draw. Train/test DP randomness structurally disjoint.",
        "clip_C": C, "delta": args.delta, "pool_size": args.pool_size, "seed": args.seed,
        "layers": layers, "epsilons": [None if math.isinf(e) else e for e in eps_list],
        "attacks": attacks, "probes": probes,
        "readout_note": "recovery = token-id top-1 over the candidate pool; *_sel = recovery minus a "
                        "CLEAN-rep label-shuffle floor (per layer); vcap_perplexity = reader effective "
                        "token candidates 2^(H_prior_empirical - PVI), clamped [1, 2^H_prior]; probes "
                        "computed on all captured rows (probe_split=all).",
        "correlation": corr, "records": records,
    }, indent=2))
    print(f"[dp-sweep] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
