# Bootstrap Seed

Origin dataset: early random / cold-start states.

This module covers the opening search regime, where the solver has not yet
collected enough memory, revisit, or operator-success signal to trust narrow
exploitation. Its job is broad coverage with just enough pressure sensitivity
to avoid pure noise.

Encourages:

- broad operator coverage
- initial structure discovery
- low-threshold exploratory spikes

Suppresses:

- premature stabilization
- early overcommitment to memory or loop escape

Failure mode targeted: cold-start blindness.
