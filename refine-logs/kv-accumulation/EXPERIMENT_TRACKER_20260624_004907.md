# Experiment Tracker — KV/QKV accumulation / BSS (Task B-1)

| Run ID | Milestone | Purpose | System / Variant | Data | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|------|---------|----------|--------|-------|
| R001 | M0 | port + synthetic sanity | bss_gram / bss_jade / bss_jd + bss_separability probe | synthetic | recovery cosine / flatness / cos_norm vs analytic | MUST | TODO | faithful ports; assert vs aloepri toy |
| R002 | M1 | dev-24 pilot, all-layer profile | jade, gram_error, probe | dev-24 (cached) | bits + p95 cosine / cos_norm, 36 layers × {kq,kqv_out,resid_post} | MUST | TODO | no GPU |
| R003 | M1 | dev-24 jd T-sweep | jd T∈{1,2,4,8,16} | dev-24 (cached) | p95 cosine(T) @ L0/L12/L20 | MUST | TODO | no GPU; ≥1 stack at T=16 |
| R004 | M1 | Hungarian floor control | random/Gaussian rows | dev-24 shapes | p95 cosine floor, 3 seeds | MUST | TODO | recovery must clear this |
| R005 | M2 | 512 GPU capture | capture kq+kqv_out @ L0/L12/L20 | release-gate-512 | — (produces operands) | COND | TODO | only if R002/R003 GO; PERF GATE first |
| R006 | M2 | 512 jd T-sweep | jd T∈{1,2,4,8,16,32} | release-gate-512 | p95 cosine(T) + slope CI | COND | TODO | depends R005 |
| R007 | M3 | probe↔recovery correlation | spectral-capacity + negentropy vs recovery | R002/3(+R006) | Spearman ρ, Pearson r | MUST | TODO | |ρ|≥0.7 ⇒ C2 |
