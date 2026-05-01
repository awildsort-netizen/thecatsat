# Oscillation-Damping Seed

Origin dataset: oscillatory / non-progressing states.

This module covers reversible dynamics: the unsatisfied count moves, but the
trajectory keeps returning to the same band. Its role is to damp undo/redo
patterns and encourage asymmetric moves.

Encourages:

- phase-shift perturbation
- loop damping
- irreversible local progress

Suppresses:

- operator pairs that repeatedly undo each other
- high-temperature drift without novelty

Failure mode targeted: oscillation around a narrow unsatisfied band.
