# Auto-review — Task 7 (leakage–utility, surface=utility-tradeoff)

Reviewer: codex / gpt-5.5 (xhigh). 3 rounds. Score 7 → 8.5 → 9, verdict **ready**.

## Round 1 (7/10, almost)
Core mechanics verified sound: perplexity is real next-token CE through the model with the defense hook active; invertible recon is a genuine inverse round-trip; no fabricated numbers; SDL + shared-spectral-capacity baselines genuinely emitted from cached clean captures; BNN H(V|Y) full-precision claim TRUE (values already on disk; 0.00 was 2dp display rounding). Five defects: (H) synthesis §04 limitations stale + GELO/KV-Cloak missing recon/overhead disclosure; (H) Vec2Text utility column showed release-cos not canonical nDCG@10; (M-H) capacity-PVI synthesis table not backed by dataset; (M) PriPert utility σ ≠ sweep σ; (M) Shredder hook used fixed prompt_index (not fresh per prompt).

## Round 2 (8.5/10, almost)
All five resolved. Remaining: capacity-PVI provenance cited synthesis.html (circular); GELO overhead text "≈3 ms" wrong; no machine-readable row status.

## Round 3 (9/10, ready)
All three confirmed fixed: provenance cites refine-logs/resid-capacity-pvi/RESULTS_STANDARDIZED.md; GELO overhead "≈6–9 ms"; row_status added (aligned:39, invertible_context:6, utility_only:2, anchor:1). No regression. Remaining limitations (single-seed, heterogeneous metrics by defense class, Pareto plot deferred to viz phase) are not blockers for a measurement+backfill phase.

## Method
Defense-class-aware utility measurement + report backfill. Lossy defenses (input-DP, PriPert, Shredder, Vec2Text-DP) → real task metric (perplexity via forward-hook with the defense active / retrieval nDCG@10) aligned to the recovery sweep; SGT → release-cosine fidelity. Invertible-in-TEE (KV-Cloak, GELO, AloePri keymat) → reconstruction rel-err (~1e-7, certified) + apply overhead. Piggyback baselines (SDL, shared-spectral-capacity) computed CPU-only from clean Qwen3 captures already on disk; BNN H(V|Y) backfilled from disk. Assembled into refine-logs/utility-tradeoff/leakage_utility.json (48 rows). Drivers: scripts/spikes/{utility_perplexity,plaintext_baselines,invertible_overhead,assemble_leakage_utility}.py.
