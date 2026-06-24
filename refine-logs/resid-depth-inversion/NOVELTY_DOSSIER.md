# Novelty Dossier — claim:depth-inversion-certificate

## Claim parts and novelty self-assessment
- C1: "deep LLM residual states are not irreversible to input tokens" — REPRODUCES Dong et al.
  arXiv 2507.16372 ("Depth Gives a False Sense of Privacy") on Qwen3-4B. NOT novel; scoped reproduction.
- DECISION: a learned 2-layer inverter beats linear ridge at the deepest layer (L32), disjoint
  bootstrap CIs. Modest; learned>linear inversion is generally known.
- NOV-1 (candidate): a Fano-type CERTIFICATE LEMMA converting a vocab-disjoint, shuffle-subtracted
  retrieval selectivity into a population lower bound I(token;resid_ℓ) ≥ Φ(a_ℓ) = H(t)−H_b(1−a)−(1−a)log(K−1),
  threshold a*, + conservative finite-sample corollary using a LOWER entropy bound underline_H.
- NOV-2 (candidate): POSITIVE measurement-loop result — an ATTACK-INDEPENDENT probe (capacity-matched
  token-id V-information reader accuracy + CLUB MI upper bound) tracks inversion recovery across
  network DEPTH (Spearman +0.85 / +0.78) on plaintext residuals.

## Phase-B prior art found (web search, June 2026)
- Fano→MI-from-classification-accuracy is WELL ESTABLISHED:
  - arXiv:1606.05229 "Estimating mutual information in high dimensions via classification error"
  - "Analyzing Privacy Leakage in ML via Multiple Hypothesis Testing: A Lesson From Fano" (2022)
  - Standard Fano lower-bounds MI from Bayes error / classifier accuracy (textbook).
- Hidden-state inversion / depth: arXiv:2507.16372 (the reproduced paper); embedding-inversion
  family (ALGEN 2025, Zero2Text 2026, arXiv:2405.11916).
- IT privacy frameworks: Inf2Guard (arXiv:2403.02116), NoPeek (arXiv:2008.09161) — leakage-reduction
  via distance correlation / IT objectives, NOT a probe-predicts-attack-across-depth finding.
- No direct hit for an attack-independent IT probe whose value TRACKS inversion recovery across depth.

## Questions for the reviewer
1. Does NOV-1's specific framing (vocab-disjoint + shuffle control selectivity as the accuracy
   plugged into Fano to certify per-depth residual token leakage) appear in prior work, or is the
   Fano-from-accuracy mechanism so standard that NOV-1 is at best an incremental application?
2. Does NOV-2 (attack-INDEPENDENT probe predicting inversion recovery across depth) have prior art?
3. Per-contribution novelty verdict (HIGH/MEDIUM/LOW) + closest prior work + honest positioning.
