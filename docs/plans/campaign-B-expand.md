<!-- ralphex-native plan (parsed by `### Task N:` sections); exempt from docs frontmatter.
     BLOCK B — expansion. Run UNATTENDED via systemd ONLY after Block A is complete and clean:
       set PLAN=docs/plans/campaign-B-expand.md in ~/.config/talens-harness/campaign.env
       systemctl --user start talens-research
     (or foreground: scripts/harness/run_campaign.sh docs/plans/campaign-B-expand.md)
     Design: memory autonomous-campaign-decisions; scripts/harness/README.md. -->

# Plan: Block B — new attacks & defenses (surface-paired)

## Overview

New attacks/defenses as homogeneous experiment→theory→report phases (recipe `full`), surface by
surface — each surface gets its new attack(s) then the defense(s) targeting it, ending in a
complete attack×probe×defense sweep + HTML page. Run **unattended** once Block A
(`campaign-A-consolidate.md`) is complete. A phase whose probe does not correlate self-appends a
follow-up Task tagged `spawn-depth:` (capped at 2). Threat anchor: `WEIGHTS-PUB` (per-phase
deviations noted in-objective). Crypto (Euston, Fision) + TwinShield-full are the HTML
zero-leakage reference box (built in Block A Task 8), not phases.

---

### Task 1: KV/QKV — accumulation / BSS attacks
recipe: full
gpu: true
surface: kv-accumulation
run_id: b-kv1-accumulation
gate: review refine-logs/kv-accumulation/REVIEW_STATE.json
spawn-depth: 0
objective: port the accumulation/BSS family (gram_error Gram-fingerprint, jd joint-diagonalization, jade) onto the KV/activation surface; does recovery scale with #observations T, and does a matched probe track it? (document sda/tfma/ia-weight as not-applicable per threat model.)
- [x] run-phase: b-kv1-accumulation

### Task 2: KV/QKV — KV-CLOAK defense + sweep
recipe: full
gpu: true
surface: kv-cloak
run_id: b-kv2-cloak
gate: review refine-logs/kv-cloak/REVIEW_STATE.json
spawn-depth: 0
objective: implement KV-CLOAK as a Transform (perm+rotation+additive mask, arXiv 2508.09442); sweep block size b; chart attack recovery (Task 1 attacks) + matched probe bits vs b; cross-check the code-available SCX permutation sibling.
- [x] run-phase: b-kv2-cloak

### Task 3: residual — Rep2Text attack (2511.06571)
recipe: full
gpu: true
surface: resid-rep2text
run_id: b-r1-rep2text
gate: review refine-logs/resid-rep2text/REVIEW_STATE.json
spawn-depth: 0
objective: implement Rep2Text (adapter→frozen-decoder, last-token resid @ L10 → full text, arXiv 2511.06571) on Qwen3; recovery (ROUGE/token-F1) vs sequence length; does a matched residual-stream probe track it?
- [x] run-phase: b-r1-rep2text

### Task 4: residual — depth-inversion + baseline inverters
recipe: full
gpu: true
surface: resid-depth-inversion
run_id: b-r2-depth-inversion
gate: review refine-logs/resid-depth-inversion/REVIEW_STATE.json
spawn-depth: 0
objective: implement the depth hidden-state inversion (arXiv 2507.16372) and the aloepri learned/baseline inverters (nn cosine-NN, isa deep-layer ridge, ima_paper_like 2-layer transformer); recovery vs depth; probe-vs-recovery across layers.
- [ ] run-phase: b-r2-depth-inversion

### Task 5: residual — GELO defense + sweep
recipe: full
gpu: true
surface: resid-gelo
run_id: b-r3-gelo
gate: review refine-logs/resid-gelo/REVIEW_STATE.json
spawn-depth: 0
objective: implement GELO (fresh invertible mixing U=AH + shield vectors, arXiv 2603.05035, code github.com/noskill/gelo) as a Transform on the QKVO-feeding residual; sweep κ(A) and shield fraction/energy vs the BSS attacks (Task 1) and anchor ridge; confirm the orthogonal-A Gram-invariance leak; probe bits vs shield params.
- [ ] run-phase: b-r3-gelo

### Task 6: residual — split-inference / PriPert defense + sweep
recipe: full
gpu: true
surface: resid-split
run_id: b-r4-split
gate: review refine-logs/resid-split/REVIEW_STATE.json
spawn-depth: 0
objective: implement split-inference with the PriPert defense (activation sparsification + adversarial perturbation, arXiv 2605.23158) at split layer Q1; sweep split layer and sparsity ratio vs the ActInv/inversion attacks; relate the empirical error floor to the paper's formal converse (Thm 1); probe bits vs parameters.
- [ ] run-phase: b-r4-split

### Task 7: embedding — Stained-Glass (SGT) defense + sweep
recipe: full
gpu: true
surface: embed-sgt
run_id: b-e1-sgt
gate: review refine-logs/embed-sgt/REVIEW_STATE.json
spawn-depth: 0
objective: implement the Stained Glass Transform (learned heteroscedastic Gaussian with MI-budget loss, arXiv 2506.09452) on the input/pooled embedding; sweep the MI-budget α; chart Vec2Text/Rep2Text-embedding recovery vs the directly-comparable spectral channel-MI probe (the cleanest MI-probe-predicts-attack test).
- [ ] run-phase: b-e1-sgt

### Task 8: attention-scores — cover-invariance lemma + check
recipe: full
gpu: true
surface: attn-cover-invariance
run_id: b-a1-attn-invariance
gate: review refine-logs/attn-cover-invariance/REVIEW_STATE.json
spawn-depth: 0
objective: state+prove the unpublished cover-invariance lemma (an orthogonal/permutation cover leaves softmax(QKᵀ) and MI(tokens;QK) untouched) and confirm empirically: inject an orthogonal-rotation Transform and verify attn_score recovery and the MI probe are unchanged; HTML page.
- [ ] run-phase: b-a1-attn-invariance

### Task 9: matched V-information probe for resid-rep2text
recipe: full
gpu: true
surface: resid-rep2text-v2
run_id: resid-rep2text-followup-1
gate: review refine-logs/resid-rep2text-v2/REVIEW_STATE.json
spawn-depth: 1
objective: the capacity probe (spectral channel-MI) is vacuous on the plaintext L10 residual because capacity is slack (claim:rep2text-capacity-nonbinding-extraction-limited). Build a matched EXTRACTABLE-information probe (𝒱-information, Xu 2020) under the decoder's hypothesis class and test whether it tracks Rep2Text recovery across length where the capacity probe failed; add a stronger-attack capacity ablation (more epochs/data/k, larger decoder) + derangement null + empirical per-token entropy to separate attack-weakness from representation limits.
- [ ] run-phase: resid-rep2text-followup-1
