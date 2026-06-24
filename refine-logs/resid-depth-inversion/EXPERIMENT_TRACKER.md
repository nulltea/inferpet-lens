# Experiment Tracker — resid-depth-inversion

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | impl + sanity | nn, ima_paper_like inverters | host | shape/unit | MUST | DONE | tests/test_inverters.py |
| R002 | M1 | pilot depth sweep | nn/ridge/ima @ every-8 | vocab-disjoint | TTRSR, sel | MUST | DONE | runs/pilot/depth_sweep_pilot.json |
| R003 | M2 | full depth sweep | nn/ridge/ima + cap-PVI + CLUB @ every-4 | vocab-disjoint+shuffle | bits + TTRSR/F1 | MUST | DONE | runs/full/depth_sweep.json, exit 0, 205s |
| R004 | M3 | correlation analysis | calibrate_records per layer | — | Spearman, indep ρ | MUST | DONE | cross_depth_correlation in JSON: cap +0.85, CLUB +0.78 |
