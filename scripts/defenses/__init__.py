"""Defense-eval implementations (Part 2) — kept OUT of ``src/talens`` core.

Per the repo's scheme-agnostic rule (``CLAUDE.md``): the library asserts nothing
about any defense; a defense is an external, pluggable ``Transform`` (or, for the
weight surface, a generator of an obfuscated :class:`~talens.weights.types.WeightPair`).
These modules are consumed by tests and the ``scripts/spikes`` runners.

* :mod:`aloepri`  — AloePri covariant obfuscation (`2603.01499`): the
  Algorithm-1 key-matrix generator, the faithful obfuscated-embedding-table
  generator (feeds the VMA τ-recovery family), and activation-space covers.
* :mod:`shredder` — Shredder learned-noise split inference (ASPLOS'20,
  `1905.11814`): a static-Laplace cover and the learned-noise trainer.
"""
