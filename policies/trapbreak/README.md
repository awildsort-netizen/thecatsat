# Trapbreak Seed

Origin dataset: high revisit / repeated-state regimes.

This module covers solver states that are active but not discovering new
territory. It is designed to interrupt loops and route energy toward operators
that change the active neighborhood.

Encourages:

- perturbation
- loop escape
- inhibition of repeated local moves

Suppresses:

- repeated pressure moves that return to the same basin
- stale memory reinforcement

Failure mode targeted: cyclic attractors and revisit traps.
