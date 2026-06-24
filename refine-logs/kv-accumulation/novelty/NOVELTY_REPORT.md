# Novelty check — claim:kv-bss-subspace-floor-and-negentropy-probe

Reviewer: Codex gpt-5.5 (xhigh), thread 019ef6eb-0bc3-7f12-9c39-321cbbd3cbb2. + WebSearch (arXiv/SS).

## Score: 5.5/10 — PROCEED (register as negative baseline + evaluation correction, not new ICA theory)

### Core claims & novelty
1. Subspace-membership-floor correction / matched Haar-demixing null for grading BSS against KNOWN
   sources under identity mixing — MEDIUM. Genuine but modest; ingredients classical (whitening→
   orthogonal rotation, sign/perm/scale ambiguity, Gaussian rotational non-identifiability). Not a
   standard named evaluation correction; ICA eval usually uses gain-matrix/performance indices.
2. Applying ICA/JD to transformer KV/QKV at all — MEDIUM/HIGH. No found paper does this; KV-cache
   attack literature uses weight-based inversion/collision.
3. negentropy predicts corrected-not-raw margin; near-zero corrected margin, flat in T — finding-novel
   (negative result).

### Closest prior work
| Work | Overlap | Delta |
|---|---|---|
| Comon 1994; Cardoso–Souloumiac JADE 1993; Hyvärinen negentropy | ICA identifiability, whitening, negentropy | we add the matched-null grading correction |
| Ilmonen 2012 (arXiv:1212.3953); Mesters–Zwiernik (arXiv:2206.13668) | ICA performance indices; rotational non-identification | related, NOT the Haar-demixing subtraction |
| Shadow in the Cache / KV-Cloak (arXiv:2508.09442) | KV-cache privacy attacks/defense | uses inversion/collision, NOT ICA/BSS |

### Framing fix applied
"BSS structurally ill-posed" → "the BSS attack framing / recovery metric is ill-posed under plaintext
identity mixing" (classical ICA with A=I is not inherently ill-posed; there are simply no hidden
sources and the cosine metric is subspace-membership-confounded).
