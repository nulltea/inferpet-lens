# Gap Map

_Field gaps with stable IDs._

## G1 — Inversion/matching as MI-efficient Bayes estimator, PROVEN to track MI  (unresolved)
No published work frames a confidential-inference recovery attack as a Bayes/MMSE-optimal
estimator and PROVES its recovery tracks mutual information (via I-MMSE / Fano / the 2 log n
alignment threshold). BeamClean (2505.13758) shows the noise-aware attack empirically dominates
ridge/NN (77% vs 17%) but proves nothing; database-alignment thresholds (Dai-Cullina-Kiyavash
1903.01422) and I-MMSE (cs/0412108) exist but are unconnected to leakage-MI tracking; Stained-Glass
(2506.09452) explicitly flags its MI->reconstruction bound as loose and invites improvement.
Targeted by: idea:info-efficient-attacks; claims weak-domination, strict-improvement,
mi-monotone-gaussian, perm-llr-threshold, restore-correlation.
