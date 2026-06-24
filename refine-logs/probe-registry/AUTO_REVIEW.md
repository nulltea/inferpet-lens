# Auto-Review Loop — campaign-C Task 2 (probe registry)

Target: docs/plans/campaign-C-report-hardening.md Task 2 ("probe registry — canonical
names + symbols, used verbatim everywhere"). Reviewer: Codex / gpt-5.5 (xhigh), medium,
thread 019efb27-a15c-7051-861f-20251859e725.
Deliverable: docs/html/probes-registry.html + a `probes` navgroup on all 15 pages +
site-wide canonicalization of probe names + synthesis glossary deferral to the registry.

## Round 1 (2026-06-24)
- Score: 7/10 — Verdict: almost
- Passed: registry page satisfies Task 2(a); I_G vs shared-spectral-capacity distinction
  correct vs src/talens/measures/; `probes` navgroup on all 15 pages; old drift strings
  (`spectral channel-MI`, `spectral channel mutual information`, `capPVI`, `cap-PVI`) gone.
- Findings: (1) residual non-canonical visible mentions; (2) synthesis table shorthand;
  (3) pre-existing prose em-dashes on edited pages.
- Fixes: #1 + #2 eliminated; #3 scoped to registry page + edited spans, rest deferred
  to per-page cleanup passes (Tasks 4/5/6).

## Round 2
- Score: 7.5/10 — Verdict: almost. **Ruled the em-dash scoping ACCEPTABLE for Task 2.**
- Findings: more verbatim-naming nits (synthesis:101, index:117, shared-spectral-capacity,
  negentropy headers, row-negentropy coinage, resid-dp V_cap/SDL prose).
- Fixes: canonical names at synthesis:101 + index:117; "shared spectral capacity"; J in
  table headers; row-negentropy → whitened-row negentropy; ρ(negentropy,…)→ρ(J,…); resid-dp
  V_cap/SDL descriptions canonical.

## Round 3
- Score: 8.5/10 — Verdict: almost. **Ruled bare post-definition `negentropy` shorthand ACCEPTABLE.**
- Findings: index:117 missing (V_cap)/(CLUB)/(SDL); three index probe cells naming the Gaussian
  channel-capacity MI ceiling without I_G; kv-cloak:217 "spectral-capacity spread".
- Fixes: index symbols added; kv-cloak:217 → "shared spectral capacity spread".

## Round 4 (final)
- Score: 8/10 — Verdict: almost  → **PASS** (score≥6 AND verdict∈{ready,almost}).
- Residual nit: three descriptive index probe `<dd>` cells (lines 130/221/234) phrased without
  registry symbols.
- Fix applied post-review (artifact strictly improved over reviewed state): the three cells now
  read `capacity-matched predictive V-information (V_cap) and variational MI upper bound (CLUB)`,
  `variational MI upper bound (CLUB) on the permutation-invariant signature`, and
  `whitened-row negentropy separability (J), computed without joint diagonalization`.

### Round 4 reviewer raw response
<details><summary>expand</summary>

Score: 8/10. Verdict: almost. Remaining blocker: index.html still had three visible probe labels
(lines 130/221/234) not using registry names + symbols; minimum fix = canonicalize those probe
`<dd>` cells. Everything else in good shape: old umbrella strings gone, `probes` nav on all 15
pages, registry page satisfies the row/module/distinction spec, synthesis defers to the registry,
no I_G / shared spectral capacity conflation.

</details>

## Termination
MAX_ROUNDS (4) reached; STOP CONDITION met at every round (score 7 → 7.5 → 8.5 → 8, verdict
"almost"). The reviewer's last-named blocker (three index probe cells) was then fixed, leaving
the artifact strictly better than the 8/almost reviewed state. No GPU. No new research claim
(Task 2 recipe-fit: edit-pages + cleanup + auto-review only).
