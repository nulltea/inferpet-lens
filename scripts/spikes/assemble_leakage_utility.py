#!/usr/bin/env python3
"""Assemble the canonical leakage--utility dataset (Task 7, utility-tradeoff).

Joins the on-disk (leakage_bits, recovery) sweeps with the utility measured in this phase
into one row per sweep point with the Task-7 schema:
  {surface, defense, param_name, param_value, leakage_bits, bits_kind, recovery,
   recovery_metric, utility_metric, utility_value, provenance}
Invertible-in-TEE rows set utility_metric="recon_error"/"overhead_ms".

Also writes queue_results.json recording the two consumed queue files (plaintext baselines +
BNN H(V|Y) precision). CPU-only, model-free: reads result JSONs already on disk.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
UT = REPO / "refine-logs/utility-tradeoff"


def _load(p):
    return json.loads(Path(p).read_text())


def rows_input_dp():
    """gemma-2-2b L0 input-DP: CLUB bits + Bayes/ridge recovery (RESULTS_STANDARDIZED R1) x perplexity."""
    util = {r["param_value"]: r for r in _load(UT / "dp_perplexity.json")["records"]}
    # R1 standardized table (refine-logs/resid-dp-attacks/RESULTS_STANDARDIZED.md)
    R1 = {  # eps: (club_bits, bayes_ttrsr, ridge_ttrsr)
        None: (3084, 1.000, 1.000), 512.0: (2942, 1.000, 0.993), 256.0: (2624, 1.000, 0.202),
        128.0: (1912, 1.000, 0.020), 96.0: (None, 1.000, 0.008), 64.0: (None, 0.993, 0.002)}
    out = []
    for eps, u in util.items():
        club, bayes, ridge = R1.get(eps, (None, None, None))
        out.append({
            "surface": "residual-input-dp", "defense": "input-dp", "param_name": "epsilon",
            "param_value": eps, "leakage_bits": club, "bits_kind": "club_mi_upper_bound",
            "recovery": bayes, "recovery_metric": "bayes_nn_token_ttrsr",
            "recovery_secondary": ridge, "recovery_secondary_metric": "ridge_token_ttrsr",
            "utility_metric": "perplexity", "utility_value": u["utility_value"],
            "provenance": "recovery+bits: refine-logs/resid-dp-attacks/RESULTS_STANDARDIZED.md R1; "
                          "utility: refine-logs/utility-tradeoff/dp_perplexity.json"})
    return out


def rows_input_dp_capacity_pvi():
    """gemma-2-2b L0 input-DP, capacity-PVI sweep variant (synthesis residual table): V_cap acc + CLUB
    + ridge recovery (resid-capacity-pvi L0) x perplexity. Same DP mechanism as R1, different N/probe run."""
    util = {r["param_value"]: r for r in _load(UT / "dp_perplexity.json")["records"]}
    # synthesis.html residual input-DP @L0 table (source: resid-capacity-pvi / b2 L0 run)
    T = {  # eps: (vcap_acc, club_bits, ridge_ttrsr)
        None: (0.882, 3850, 0.809), 1024.0: (0.878, 3640, 0.661),
        512.0: (0.846, 3340, 0.428), 256.0: (0.684, 2810, 0.140)}
    out = []
    for eps, (vcap, club, ridge) in T.items():
        u = util.get(eps)
        out.append({
            "surface": "residual-input-dp-l0-capacity", "defense": "input-dp", "param_name": "epsilon",
            "param_value": eps, "leakage_bits": club, "bits_kind": "club_mi_upper_bound",
            "vcap_reader_acc": vcap, "recovery": ridge, "recovery_metric": "ridge_token_ttrsr",
            "utility_metric": "perplexity", "utility_value": (u["utility_value"] if u else None),
            "provenance": "recovery+bits: refine-logs/resid-capacity-pvi/RESULTS_STANDARDIZED.md (L0 input-DP "
                          "V_cap/CLUB/ridge table, gemma-2-2b); utility: refine-logs/utility-tradeoff/dp_perplexity.json "
                          "(same per-token-embedding DP mechanism as the R1 rows, so the perplexity curve applies to both)"})
    return out


def rows_pripert():
    """Qwen3-4B resid L8/rho=0.25 PriPert: I_G bits + best-inverter selectivity x perplexity."""
    sweep = _load(REPO / "refine-logs/resid-split/runs/sweep/pripert_sweep.json")["records"]
    cells = {r["beta"]: r for r in sweep if r.get("layer") == 8 and abs(r.get("rho", 9) - 0.25) < 1e-6}
    udoc = _load(UT / "pripert_perplexity.json")
    util = {r["param_value"]: r for r in udoc["records"]}
    sigma_ref = udoc.get("sigma_ref")  # σ floor the utility run used (aligned to the sweep)
    out = []
    for beta, c in sorted(cells.items()):
        inv = c["inverters"]
        best = max((inv[k]["selectivity"] for k in inv if inv[k].get("selectivity") is not None), default=None)
        pr = c["probes"]
        ig = pr.get("i_g_bits")
        u = util.get(beta)
        out.append({
            "surface": "residual-split", "defense": "pripert", "param_name": "beta",
            "param_value": beta, "rho": 0.25, "layer": 8,
            "leakage_bits": (None if (ig is not None and math.isinf(ig)) else ig),
            "bits_kind": "spectral_channel_mi_i_g", "i_g_is_inf": pr.get("i_g_is_inf"),
            "fano_recovery_ceiling": pr.get("fano_recovery_ceiling"),
            "recovery": best, "recovery_metric": "best_inverter_token_ttrsr_selectivity",
            "sigma_sweep": c.get("sigma"), "sigma_utility": (u.get("sigma") if u else None),
            "utility_metric": "perplexity", "utility_value": (u["utility_value"] if u else None),
            "provenance": "recovery+bits: refine-logs/resid-split/runs/sweep/pripert_sweep.json (L8,ρ=0.25); "
                          "utility: refine-logs/utility-tradeoff/pripert_perplexity.json. Aligned by ABSOLUTE σ: "
                          f"utility run used σ_ref={sigma_ref} (the sweep's plaintext meanRMS), so sigma_sweep==sigma_utility at every β."})
    # plaintext anchor (ρ=1,β=0) perplexity
    anchor = next((r for r in _load(UT / "pripert_perplexity.json")["records"] if r.get("plaintext_anchor")), None)
    if anchor:
        out.insert(0, {
            "surface": "residual-split", "defense": "pripert", "param_name": "beta",
            "param_value": 0.0, "rho": 1.0, "layer": 8, "plaintext_anchor": True,
            "leakage_bits": None, "bits_kind": "spectral_channel_mi_i_g",
            "recovery": None, "recovery_metric": "best_inverter_token_ttrsr_selectivity",
            "utility_metric": "perplexity", "utility_value": anchor["utility_value"],
            "provenance": "plaintext (no defense) anchor: refine-logs/utility-tradeoff/pripert_perplexity.json"})
    return out


def rows_vec2text_dp():
    """GTR pooled embedding Vec2Text-under-DP: I_G bits + token-F1 x retrieval nDCG@10/Recall."""
    probe = _load(REPO / "results/spectral_mi_probe_eval.json")
    precs = probe.get("records", probe.get("rows", []))
    pby = {(r.get("epsilon")): r for r in precs}
    ret = _load(UT / "gtr_retrieval.json")
    rrecs = ret.get("records", ret.get("rows", ret.get("sweep", [])))
    rby = {}
    for r in (rrecs if isinstance(rrecs, list) else []):
        eps = r.get("epsilon", r.get("eps"))
        rby[eps] = r
    out = []
    for eps, u in rby.items():
        p = pby.get(eps)
        ndcg = u.get("ndcg@10") or next((v for k, v in u.items() if k.startswith("ndcg")), None)
        out.append({
            "surface": "embedding-vec2text", "defense": "input-dp-embedding", "param_name": "epsilon",
            "param_value": eps, "leakage_bits": (p.get("i_g_bits") if p else None),
            "bits_kind": "spectral_channel_mi_i_g",
            "recovery": (p.get("token_f1") if p else None), "recovery_metric": "vec2text_token_f1",
            "recovery_note": (None if p else "utility measured at this ε; no recovery sweep point (TODO)"),
            "utility_metric": "retrieval_ndcg@10", "utility_value": ndcg,
            "utility_recall": u.get("recall"), "utility_rank_spearman": u.get("spearman") or u.get("rank_spearman"),
            "provenance": "recovery+bits: results/spectral_mi_probe_eval.json; "
                          "utility: refine-logs/utility-tradeoff/gtr_retrieval.json"})
    return out


def rows_sgt():
    """GTR SGT shaped-noise: I_G bits + token-F1 x release-cosine fidelity (utility on disk)."""
    sgt = _load(REPO / "refine-logs/embed-sgt/runs/sweep/sgt_eval.json")
    recs = sgt.get("records", sgt if isinstance(sgt, list) else sgt.get("rows", []))
    out = []
    for r in recs:
        out.append({
            "surface": "embedding-sgt", "defense": "sgt", "param_name": "budget_bits",
            "param_value": r.get("budget_bits"), "shape": r.get("shape"),
            "leakage_bits": r.get("i_g_bits"), "bits_kind": "spectral_channel_mi_i_g",
            "recovery": r.get("token_f1"), "recovery_metric": "vec2text_token_f1",
            "utility_metric": "release_cosine", "utility_value": r.get("release_cos"),
            "provenance": "refine-logs/embed-sgt/runs/sweep/sgt_eval.json (release_cos = released-vs-clean "
                          "embedding cosine fidelity; retrieval nDCG measured separately for the DP-mechanism embed surface)"})
    return out


def rows_invertible():
    """KV-Cloak / GELO / AloePri: recon_error (~0) + overhead_ms; recovery from each defense sweep."""
    inv = _load(UT / "invertible_utility.json")["rows"]
    # headline recovery for context (from each defense's leakage sweep)
    rec_ctx = {
        ("kv-cloak", "m"): ("jade_p95=0.126 (BSS), neg=1.47b", "refine-logs/kv-cloak/analysis.json"),
        ("kv-cloak", "full"): ("jade_p95=0.126 (BSS), neg=0.72b", "refine-logs/kv-cloak/analysis.json"),
        ("gelo", 1.0): ("jade_p95=0.721, genuine_margin=0.047", "refine-logs/resid-gelo/analysis.json"),
        ("gelo", 10.0): ("jade_p95=0.705, genuine_margin=0.022", "refine-logs/resid-gelo/analysis.json"),
        ("gelo", 100.0): ("jade_p95=0.692, feat_gram_relerr=1780", "refine-logs/resid-gelo/analysis.json"),
        ("aloepri-keymat", 0.0): ("VMA τ-recovery=0.000 (chance), CLUB-on-φ=−2.4b", "refine-logs/dp-stronger-attacks/EXPERIMENT_RESULTS.md"),
    }
    out = []
    for r in inv:
        if "error" in r:
            out.append(r); continue
        ctx, src = rec_ctx.get((r["defense"], r["param_value"]), (None, None))
        out.append({
            "surface": r["surface"], "defense": r["defense"], "param_name": r["param_name"],
            "param_value": r["param_value"], "leakage_bits": None,
            "bits_kind": "n/a (invertible-in-TEE: lossless un-mix; leakage probe = negentropy/CLUB on the released surface)",
            "recovery": None, "recovery_metric": "see recovery_context",
            "recovery_context": ctx, "recovery_source": src,
            "utility_metric": r["utility_metric"], "utility_value": r["utility_value"],
            "overhead_ms": r.get("overhead_ms"), "keymat_identity_relerr": r.get("keymat_identity_relerr"),
            "provenance": f"recon+overhead: refine-logs/utility-tradeoff/invertible_utility.json ({r.get('provenance')})"})
    return out


def rows_shredder():
    """gemma-2-2b Shredder static-Laplace: permutation+token-id recovery (defenses-existing R1) x perplexity."""
    util = {r["param_value"]: r for r in _load(UT / "shredder_perplexity.json")["records"]}
    # recovery from refine-logs/defenses-existing/RESULTS_STANDARDIZED.md R1 (b: perm_rec, tokid_rec_L0)
    R = {0.0: (1.000, 0.747), 0.109: (0.977, None), 0.218: (0.565, None), 0.381: (0.204, None),
         0.545: (0.099, None), 0.817: (0.037, 0.670)}
    out = []
    for b, u in sorted(util.items()):
        perm, tokid = R.get(b, (None, None))
        out.append({
            "surface": "embedding-shredder", "defense": "shredder", "param_name": "shredder_b",
            "param_value": b, "leakage_bits": None,
            "bits_kind": "n/a (source embedding probe NaN across seeds; recovery-only per defenses-existing)",
            "recovery": perm, "recovery_metric": "permutation_vma_recovery",
            "recovery_secondary": tokid, "recovery_secondary_metric": "token_id_recovery_L0",
            "utility_metric": "perplexity", "utility_value": u["utility_value"],
            "provenance": "recovery: refine-logs/defenses-existing/RESULTS_STANDARDIZED.md R1; "
                          "utility: refine-logs/utility-tradeoff/shredder_perplexity.json"})
    return out


def main():
    UT.mkdir(parents=True, exist_ok=True)
    rows = []
    rows += rows_input_dp()
    rows += rows_input_dp_capacity_pvi()
    rows += rows_pripert()
    rows += rows_vec2text_dp()
    rows += rows_sgt()
    rows += rows_invertible()
    rows += rows_shredder()
    # row_status: machine-readable coverage tag (so consumers need not parse prose/provenance)
    for r in rows:
        if r.get("utility_value") is None and r.get("recovery") is None and r.get("leakage_bits") is None:
            r["row_status"] = "todo"
        elif r.get("plaintext_anchor"):
            r["row_status"] = "anchor"
        elif r.get("utility_metric") in ("recon_error", "overhead_ms"):
            r["row_status"] = "invertible_context"
        elif r.get("recovery") is None and r.get("leakage_bits") is None and r.get("utility_value") is not None:
            r["row_status"] = "utility_only"
        elif r.get("utility_value") is not None and (r.get("recovery") is not None or r.get("leakage_bits") is not None):
            r["row_status"] = "aligned"
        else:
            r["row_status"] = "partial"
    (UT / "leakage_utility.json").write_text(json.dumps({
        "schema": ["surface", "defense", "param_name", "param_value", "leakage_bits", "bits_kind",
                   "recovery", "recovery_metric", "utility_metric", "utility_value", "provenance", "row_status"],
        "row_status_values": {"aligned": "leakage+recovery+utility at the same sweep point",
                              "anchor": "plaintext/no-defense baseline", "utility_only": "utility measured; no recovery/bits sweep point",
                              "invertible_context": "invertible-in-TEE: utility=recon_error+overhead, recovery is context",
                              "partial": "some axes present", "todo": "explicit TODO + reason in provenance"},
        "note": "Task-7 leakage-utility dataset. Lossy defenses: utility=task metric (perplexity / "
                "retrieval nDCG@10 / release-cosine). Invertible-in-TEE defenses: utility=recon_error(~0)"
                "+overhead_ms (lossless by construction). One row per sweep point.",
        "n_rows": len(rows), "rows": rows}, indent=2))
    print(f"[assemble] wrote leakage_utility.json ({len(rows)} rows)")

    # queue_results.json — record both consumed queue files
    base = _load(UT / "plaintext_baselines.json")
    bnn = _load(REPO / "results/bnn_error_bounds_validation_dense.json")
    hvy = {r["epsilon"]: r["h_cond_bits"] for r in bnn["records"] if r.get("epsilon") in (128.0, 96.0, 80.0, 64.0, 56.0)}
    qr = {
        "consumed": ["refine-logs/probe-pages/queued-for-utility.md", "refine-logs/readout-metrics/queued-for-utility.md"],
        "emitted": {
            "SDL_plaintext_per_layer": {
                "status": "EMITTED (CPU from clean Qwen3 resid capture; no GPU needed)",
                "source": "refine-logs/utility-tradeoff/plaintext_baselines.json -> sdl",
                "backfill_target": "docs/html/probe-sdl.html §04 Plaintext reference",
                "rows": base["sdl"]["rows"]},
            "shared_spectral_capacity_plaintext_per_layer_kind": {
                "status": "EMITTED (CPU from clean Qwen3 k/v capture; no GPU needed)",
                "source": "refine-logs/utility-tradeoff/plaintext_baselines.json -> shared_spectral_capacity",
                "backfill_target": "docs/html/probe-shared-spectral-capacity.html §04 Plaintext reference",
                "rows": base["shared_spectral_capacity"]["rows"]},
            "bnn_h_v_given_y_full_precision": {
                "status": "BACKFILL FROM DISK (full-precision h_cond_bits already stored; the 0.00 was a "
                          "2dp DISPLAY rounding in the MD, not a missing value — no GPU needed)",
                "source": "results/bnn_error_bounds_validation_dense.json",
                "backfill_target": "docs/html/bnn-attack.html / synthesis.html H(V|Y) column (millibit render)",
                "h_cond_bits_by_epsilon": hvy}},
    }
    (UT / "queue_results.json").write_text(json.dumps(qr, indent=2))
    print(f"[assemble] wrote queue_results.json")


if __name__ == "__main__":
    main()
