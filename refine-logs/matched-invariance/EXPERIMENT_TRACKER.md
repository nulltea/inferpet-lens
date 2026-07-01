# Experiment Tracker — matched-vs-selfgen invariance

| Run ID | Milestone | Purpose | System / Variant | Surface × Config | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|------------------|-------|---------|----------|--------|-------|
| R000 | M0 | sanity: self_gen==matched, training valid | matched, self_gen, floor | all surfaces × plaintext | prompt-rowsplit | top1, in-set | MUST | **DONE** | PASS: matched==self_gen all surfaces (resid .993, kqv .547, kq .269); dev=cuda after torch fix |
| R001 | M1 | C1/C2 main matrix | matched, self_gen, floor | {residual,kqv_out,kq} × {plaintext,keymat,alg1,alg2} | prompt-rowsplit | top1, gap, floor, in-set; 3 seeds | MUST | **DONE** | **C1+C2 PROVEN**: gap=0 invariant (keymat kqv/kq); self_gen→floor rotated (alg2 .004/.022); residual incompat (768→1024). Bonus: alg1 noise = intermediate (.231/.193). matched_vs_selfgen.json |
| R002 | M2 | C3 per-head fingerprint | SVD Π_head match | Q/K/V/O × {keymat,alg1,Alg2} | — | head-perm recovery acc | MUST | **DEFERRED** | design flaw: shared non-orthogonal Q̂ᵀ breaks naive SVD match; no Π_head under keymat/alg1 (identity). Needs Q̂ᵀ-invariant fingerprint before implementing |
| R003 | M3 | noise-aware self_gen | matched, self_gen(+own αₑ) | kqv_out × αₑ{0,0.5,1,2} | prompt-rowsplit | top1, gap | NICE | TODO | test if alg1 intermediate gap closes when self_gen models the public αₑ noise |
| R004 | M3 | disjoint appendix | matched, self_gen, floor | main matrix | disjoint | top1, gap | NICE | TODO | generalization confound (orthogonal to C1) |
