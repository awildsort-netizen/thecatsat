# Entropy-Shaping Seed

Origin dataset: high-entropy exploration regimes.

This module covers states where the solver has motion and variety, but the
operator distribution is too flat to become directional. It sharpens exploration
without collapsing immediately into near-solution stabilization.

Encourages:

- medium-radius graph moves
- directional exploration
- moderate concentration sharpening

Suppresses:

- flat operator mixtures
- unproductive thrashing

Failure mode targeted: energetic but unfocused exploration.
