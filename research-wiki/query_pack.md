# Research Wiki Query Pack

_Auto-generated. Do not edit._

## Open Gaps
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

## Key Papers (22 total)
- [paper:balle2018_improving_gaussian_mechanism] Improving the Gaussian Mechanism for Differential Privacy: Analytical Calibration and Optimal Denoising
- [paper:blackwell1953_equivalent_comparisons_experiments] Equivalent Comparisons of Experiments (Blackwell ordering; Bayes-risk monotonicity under coarsening)
- [paper:cherisey2019_best_information_most] Best Information is Most Successful: Mutual Information and Success Rate in Side-Channel Analysis
- [paper:chung2022_diffusion_posterior_sampling] Diffusion Posterior Sampling for General Noisy Inverse Problems
- [paper:cover2006_elements_information_theory] Elements of Information Theory (Fano's inequality, DPI, sufficient statistics)
- [paper:cuturi2013_sinkhorn_distances_lightspeed] Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances
- [paper:dai2019_database_alignment_gaussian] Database Alignment with Gaussian Features
- [paper:dai2023_gaussian_database_alignment] Gaussian Database Alignment and Gaussian Planted Matching
- [paper:ding2018_efficient_random_graph] Efficient random graph matching via degree profiles
- [paper:ding2021_planted_matching_problem] The planted matching problem: Sharp threshold and infinite-order phase transition
- [paper:dong2025_depth_gives_false] Depth Gives a False Sense of Privacy: LLM Internal States Inversion
- [paper:fan2019_spectral_graph_matching] Spectral Graph Matching and Regularized Quadratic Relaxations I: The Gaussian Model
## Recent Relationships (30 total)
  claim:restore-correlation --uses--> paper:kale2025_beamclean_language_aware
  claim:restore-correlation --uses--> paper:guo2004_mutual_information_minimum
  claim:restore-correlation --uses--> paper:dai2019_database_alignment_gaussian
  idea:info-efficient-attacks --inspired_by--> paper:kale2025_beamclean_language_aware
  idea:info-efficient-attacks --inspired_by--> paper:guo2004_mutual_information_minimum
  idea:info-efficient-attacks --inspired_by--> paper:dai2019_database_alignment_gaussian
  idea:info-efficient-attacks --inspired_by--> paper:blackwell1953_equivalent_comparisons_experiments
  idea:info-efficient-attacks --inspired_by--> paper:pananjady2017_denoising_linear_models
  idea:info-efficient-attacks --inspired_by--> paper:chung2022_diffusion_posterior_sampling
  idea:info-efficient-attacks --addresses_gap--> gap:G1
  paper:kale2025_beamclean_language_aware --addresses_gap-
