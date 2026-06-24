# Experiment Tracker — resid-gelo

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | sanity: Gram-invariance + un-mix identities | GELO orth/ill/shield | L12, 1 op | feat-Gram rel-err, row-Gram conj, un-mix err | MUST | TODO | B1 / C0 |
| R002 | M1 | pilot sweep @ L12 | jade/jd/gram_error + ridge + floor | L12, ≤96 prompts | p95 cosine, margin, bits, ridge | MUST | TODO | B2 |
| R003 | M2 | full sweep {0,12,20} + probe-indep | as R002 + probe-indep | 3 layers | margin, bits, Spearman | MUST | TODO | B2/B3 / C1,C2 |
