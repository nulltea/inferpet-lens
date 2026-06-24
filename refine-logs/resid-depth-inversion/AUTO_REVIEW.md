# Auto Review — resid-depth-inversion (Task 4)

Reviewer: Codex gpt-5.5, xhigh. Thread 019ef81c-0325-7fa1-86a7-dafac6ab60ea. Difficulty: medium.

## Round 1 — Score 8/10, verdict ALMOST
Strong scoped measurement deliverable; numbers match depth_sweep.json; L32 mlp2>ridge gap CI-disjoint;
proof sound in corrected population/conditional form; no evidence cap-PVI is the attack reparameterized.
Weaknesses (all wording/metadata, no new experiment):
1. Front-facing text said "selectivity certifies information" — should distinguish raw accuracy (Fano bound) from selectivity (operational falsification).
2. Probe scope under-disclosed (cap-PVI row-split/shared capped classes, not vocab-disjoint; CLUB target-adjacent, rank-only).
3. HTML carried process wording (queued, cross-model review, claim node) and an over-broad H1.
4. CLUB bits easy to misread — label as upper-bound score, rank only.
5. p≈0.004 → permutation p for n=9 is ~0.006; say p<0.01.

## Round 2 — Score 9/10, verdict READY
All five addressed (raw-accuracy vs selectivity separated; probe scope disclosed in claim C2 + HTML §04;
process wording removed, H1 scoped; CLUB relabeled rank-only; p<0.01). Three trivial non-blockers
(CLUB column label, one stale "selectivity"→"raw accuracy" open-item, diagram caption) applied for polish.
Remaining follow-up (not blocking): emit num_classes/n_train/n_test probe metadata into the run JSON (needs a GPU re-run).

POSITIVE STOP: score 9 >= 6 AND verdict ∈ {ready, almost}.
