# Density Seed

Origin dataset: high unsatisfied-clause density states.

This module covers regimes where conflict pressure is broad and useful. The
solver should spend less energy wandering and more energy exploiting strong
unsatisfied-clause gradients.

Encourages:

- conflict-heavy descent
- high-impact pressure moves
- concentration around local gain

Suppresses:

- novelty-only moves when pressure is informative
- memory-heavy repair before the field thins out

Failure mode targeted: wasting steps while obvious gradients exist.
