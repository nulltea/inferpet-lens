# Auto Review — resid-rep2text (Rep2Text on Qwen3 L10 residual)

## Round 1 — Score 8/10 — Verdict: almost
Reviewer (codex gpt-5.5 xhigh, thread 019ef7b6). Scope honesty good; probe-≠-attack PASS;
claim↔evidence fidelity strong; register appropriate. Two precision fixes: (W1) separate plaintext
length-only Spearman (undefined, r≡1) from full-grid ρ=0.176; (W2) make max-length entropy basis
explicit (capped 64 tok ⇒ H_X≤1102 b); (W3) drop stale "prior-only" from code docstring.

## Round 2 — Score 9/10 — Verdict: ready
W1–W3 confirmed addressed in the changed files. One non-blocking polish (headline precision) applied
after. No critical findings. STOP — positive assessment reached.

## Method Description
Rep2Text inversion: a 3-layer adapter maps a single Qwen3-4B layer-10 last-token residual (d=2560)
to k=8 soft-prompt embeddings driving a FROZEN Qwen3-1.7B decoder (shared tokenizer); only the adapter
is trained (teacher-forced CE). Recovery = greedy generation scored by token-F1/ROUGE-L vs true input
tokens, swept over residual-noise σ. The attack-independent probe is the geometry-only spectral
channel-MI I_G of the residual covariance spectrum. Finding: capacity is non-binding at L10, so the
capacity probe is vacuous across length (verified capacity-slack lemma); recovery is extraction-limited.
