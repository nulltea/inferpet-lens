# Experiment Tracker — resid-rep2text

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | capture L10 last-token resid + build buckets | Qwen3-4B capture | aux corpus | shapes, bucket counts | MUST | TODO | run_step name `capture` |
| R002 | M1 | adapter overfit smoke (pipeline learns?) | Rep2Text adapter, tiny subset | train subset | train CE | MUST | TODO | run_step name `adapter-smoke` |
| R003 | M2 | train adapter + eval real vs 3 controls per bucket | Rep2Text + controls | train/val/test | token-F1, ROUGE-L | MUST | TODO | run_step name `adapter-train-eval`; >10 min → perf gate |
| R004 | M3 | spectral channel-MI probe + correlation | I_G geometry-only | test buckets + σ sweep | bits, Spearman ρ | MUST | TODO | run_step name `probe` |
