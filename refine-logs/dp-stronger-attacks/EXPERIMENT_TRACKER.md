# Experiment Tracker — NNS-PVI probe

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | Implement nns_v_information() | src/talens/measures/vinfo_capacity.py | synthetic d=32 | T1–T5 unit tests pass | MUST | TODO | Add alongside existing v_information_capacity(); same interface |
| R002 | M0 | Unit tests | pytest tests/test_vinfo_capacity.py | synthetic | 5 tests pass | MUST | TODO | Depends on R001 |
| R003 | M1 | Full 5-pt ε-sweep with NNS-PVI col | unified_dp_sweep.py + nns_pvi in _run_probes | gemma-2-2b, 256 prompts, pool=2048 | ρ(NNS-PVI-sel, BNN) ≥ 0.90; NNS-PVI-bits ≥ CapPVI-bits at ε∈{256,64} | MUST | TODO | Extend _run_probes(); pass C_raw, sigma, pool. Also add B3 cols (PCA-τ, RandProj-τ) and B5 col (NNS-PVI@L20) as freeriders |
| R004 | M2 | Compute ρ matrix, write findings | analysis of R003 output | — | C1/C2/C3 confirmed or not | MUST | TODO | Depends on R003 |
| R005 | M3 | Pool size ablation | nns_v_information() with pool∈{256,2048,8192} | gemma-2-2b, ε=256 only | NNS-PVI-bits vs pool_size | NICE | TODO | Only if C1 confirmed in R003/R004 |
