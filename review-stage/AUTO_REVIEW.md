# Auto Review — campaign-C Task 4 (per-probe pages)

Target: docs/plans/campaign-C-report-hardening.md Task 4. Reviewer: Codex gpt-5.5 (xhigh), medium difficulty.

## Round 1 — Score 6/10, Verdict: not ready
Blockers: (W1) probe-j.html §04 plaintext numbers mislabeled KV-Cloak values (1044.7/1.47) as kv-accumulation; (W2) V_cap diagram drew an attack-output arrow into the probe result (probe≠attack visual breach); (W3) several diagrams had unlabeled arrows; (W4) claim cites as <code> not <a>, synthesis rows lacked anchors; (W5) "iGPU" + a spikes path in visible text, masthead/inline-style em-dash flags.

## Round 2 — Score 9/10, Verdict: ready
All Round-1 blockers resolved (W1 corrected to refine-logs/kv-accumulation/c2_robustness.json family means 280.3/47.0/3.56 bits; W2 cross-arm arrow removed + independence prose note; W3 every arrow labeled; W4b synthesis row anchors added and deep-linked; W5 iGPU + spikes path removed). Rebuttals SUSTAINED: W4a (claim-as-<code> matches vec2text.html and avoids 404 under the docs-root server), W5a (masthead/FIG/SVG em-dashes not prose; zero em-dashes in <p>/<li> prose), W5b (inline h4 styles match vec2text.html), W5c (max_rows is the real CLUB estimator cap). No remaining critical weaknesses; two minor cosmetic nits left to house convention. POSITIVE ASSESSMENT — gate met.

## Method Description
Seven per-probe documentation pages, one per registry entry, each rendering the measurement loop as two independent arms from one released surface: an attack arm (→ recovery) and an attack-independent probe arm (→ bits). Numbers are sourced verbatim from on-disk results; the two probes with no clean-model plaintext-across-layers reading (SDL, shared spectral capacity) carry explicit Task-7 queue placeholders.
