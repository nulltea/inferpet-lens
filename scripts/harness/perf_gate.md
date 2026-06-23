# Performance gate — standardized review prompt

Feed this to `/auto-review-loop` (reviewer: codex) **before launching any GPU run**. The run
plan PASSES only when the reviewer answers YES to every check below. Iterate the run plan
(scope, device placement, batching) until it passes; do not launch a run the gate rejects.

Context the reviewer needs: the exact command(s) to be run, the surface/layers/corpus size, the
sweep grid, and the expected wall-time estimate. Hardware: one AMD Strix Halo iGPU (gfx1151);
**one GPU process at a time**; heavy work runs via `scripts/run_in_rocm.sh`.

## Checks (all must be YES)

1. **Optimal scope.** Is this the *smallest* run that answers the phase question? Has a
   fast-iterate pass (single layer / `--every-n` / a small corpus like `dev-24`) been done before
   the full sweep? Are there redundant or non-informative sweep points that can be dropped?
2. **Max GPU utilization.** Does *every* component that can run on GPU run on GPU — capture,
   probe/CLUB net training, attack fits? Specifically: PCA via covariance-eigh on GPU (not full
   SVD); matmuls/probe nets on `device='cuda'`; no silent CPU fallback (sklearn LR may stay CPU
   only where it is genuinely cheap and validated against the GPU path).
3. **Batching/saturation.** Are batch sizes large enough to saturate the iGPU (not single-prompt
   loops)? Is captured data cast to f32 only for the small slices that need it (model stays bf16)?
4. **Memory safety.** Any `O(n²·d)` transient that could OOM at the planned N? If so, is the
   moment-based / chunked estimator used instead (cf. CLUB exact moment estimate)?
5. **Wall-time + serialization.** Is the wall-time estimate stated and reasonable? If it exceeds
   ~10 min, is saturation confirmed first? Is the plan strictly serial (no second GPU container,
   no `/experiment-queue` fan-out) per one-GPU-process-at-a-time?
6. **Reproducibility.** Seeds fixed; output path under `refine-logs/<surface>/`; command
   wrapped in `scripts/run_in_rocm.sh`.

## Verdict

Return `ready` (score ≥ 6) only when 1–6 are all YES. Otherwise return the concrete fixes
(device moves, scope cuts, batch/memory changes) for the next iteration.
