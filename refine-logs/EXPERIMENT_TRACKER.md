# Experiment Tracker — Union-Bhattacharyya & Fano error bounds

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | Implement union_bhattacharyya + fano_equivocation | src/talens/measures/channel_error_bounds.py | synthetic d=32,K=8 | — | MUST | DONE | No Y/X arg (T3). Gram cached over σ; fresh-noise MC for Ĥ_M. GPU+NumPy paths. |
| R002 | M0 | Unit tests | pytest tests/test_channel_error_bounds.py | synthetic | T-a..T-f + bracketing | MUST | DONE | 8/8 pass on host .venv. Ĥ_M matches brute-force H(V|Y); end-to-end bracketing holds. |
| R002b | M0 | Codex code review | gpt-5.5 xhigh thread 019eefe7 | — | no CRITICAL | MUST | DONE | MAJOR fix applied: bracketing uses certified LCB on H(V|Y), not raw Ĥ_M. 3 MINOR applied. |
| R003 | M1 | Bounds vs BNN ε-sweep (coarse) | bnn_error_bounds_validation.py | gemma-2-2b, pool=2048, ε∈{∞,1024,512,256,64} | C1 bracketing | MUST | DONE | 5/5 bracketed. BNN~0 until r=3.63 → degenerate range; ρ tie-dragged 0.707 → dense grid. |
| R004 | M1 | Bounds vs BNN ε-sweep (dense) | bnn_error_bounds_validation.py --epsilons 128..16 | gemma-2-2b, pool=2048, r∈[1.8,14.5] | C1+C4 | MUST | DONE | **10/10 bracketed; ρ(Fano-lb,BNN)=+0.937, ρ(union-ub,BNN)=+0.888.** Complementary regimes. |
| R005 | M1 | Morphological-floor attribution | top confusable pairs @ε=16 | pool=2048 | C2 | NICE | DONE | Top pairs = case/space/number neighbours ('Hardware'~' Hardware', '3'~'5', ' six'~' seven'). |
| R006 | M2 | Results + wiki + objective assessment | EXPERIMENT_RESULTS.md | — | C1-C4, objective | MUST | DONE | OBJECTIVE MET: formal MI probe (equivocation) correlates ρ=+0.94 with BNN@L0, independent of attack. |
