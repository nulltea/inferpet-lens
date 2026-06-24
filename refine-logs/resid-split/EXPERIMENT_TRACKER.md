# Experiment Tracker — resid-split

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| pilot  | M0 | B1: ρ-sweep@L8 β=0.5 + β-sweep@ρ=0.25 | PriPert Transform; ridge/nn/mlp2; I_G+CLUB | vocab-disjoint | TTRSR/F1/cos/sel+CI; I_G bits; Fano | MUST | TODO | wiring check; reuse cache L8 |
| sweep  | M1 | B2: joint ℓ×ρ @ β=0.5 + plaintext anchor | PriPert; ridge/nn/mlp2; I_G+CLUB | vocab-disjoint | as above per (ℓ,ρ) | MUST | TODO | layers {0,8,16,24}, ρ∈{1,.5,.25,.1,.05} |
| —      | M2 | B3: converse certificate | post-hoc over sweep JSON | — | accessible ceiling vs realized bits; Fano violations | MUST | TODO | in-driver, no new GPU |
