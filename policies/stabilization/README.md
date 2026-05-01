# Stabilization Seed

Origin dataset: near-solution / low-unsat regimes.

This module covers fragile late-stage states. It should repair the remaining
clauses while preserving satisfied structure, so its threshold is intentionally
higher and its memory prior is stronger.

Encourages:

- fine-grained repair
- low-damage operator mixtures
- preservation of stable assignments

Suppresses:

- broad perturbation
- aggressive loop escape near solved states

Failure mode targeted: blowing up near-solutions.
