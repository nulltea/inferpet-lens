# Experiment Tracker — KV-CLOAK (Task B-2)

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | unit identities | KV-CLOAK channels | synthetic+dev24 | gram-resid, spectrum-resid, invertibility | MUST | TODO | B1 |
| R002 | M1 | attack sweep | Identity/M/SP̂/A/SCX/naive/full × b × mask | dev24(b≤8)+L32(b≤32) | recovery p95cos vs floor, gram_error | MUST | TODO | B2 |
| R003 | M2 | matched probe | negentropy + spectral-capacity | same cells | bits; ρ vs recovery; b-flatness | MUST | TODO | B3, folded into R002 |
| R004 | M3 | standardize | analysis | — | bits+readout | MUST | TODO | RESULTS.md |
