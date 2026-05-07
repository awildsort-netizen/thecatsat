# Excitable Operators and Refractory Timing Design

Date: 2026-05-07

## Goal

Unify Cat SAT propagation around one rule:

```text
operators modulate operators
```

Everything that can shape propagation is an operator. Policies, seeds, heuristics, SAT moves, neighbor influence, refractory gates, observer traces, and temperature schedules all enter the same nervous system as excitable units with state, inputs, activation dynamics, emissions, couplings, and refractory behavior.

The first implementation should keep this unification local and testable. It should add operator-level refractory timing and typed emissions without flattening every behavior into an unbounded generic hook.

## Core Law

Every operator that fires must rest for at least one propagation tick.

```python
BASE_REFRACTORY_TICKS = 1
```

This is Cat SAT's minimum temporal quantum. Operator classes can extend recovery time, but no operator can remove it.

```python
effective_refractory_ticks = max(
    BASE_REFRACTORY_TICKS,
    round(
        BASE_REFRACTORY_TICKS
        * operator_class_multiplier
        * fatigue_multiplier
        * phase_context_multiplier
    ),
)
```

The design principle is simple: no timeless operators. No policy, seed, gate, or schedule gets to sit outside propagation as an exempt controller.

## Operator Model

Introduce a shared operator state shape:

```python
operator_state[operator_id] = {
    "potential": 0.0,
    "threshold": 1.0,
    "last_fired_step": None,
    "refractory_until": 0,
    "spike_count": 0,
    "recent_input": 0.0,
}
```

Each propagation step applies the same activation loop:

```text
decay potential
integrate excitatory drive
integrate inhibitory drive
integrate policy pressure
integrate neighbor activity
check refractory gate
fire if potential crosses threshold
emit typed effects
reset or damp potential
enter refractory period
record trace
```

The universal loop does not mean every operator can do anything. The sharp boundary is:

```text
what may this operator emit into the world?
```

## Operator Classes

### Actuator Operators

Fast operators that directly mutate SAT solve state.

Examples:

- flip variable
- walk
- repair clause
- perturb assignment

Allowed emissions:

- `sat_action`
- `assignment_delta`
- `clause_repair`
- `trajectory_step`

### Field Operators

Slow or medium-speed operators that bias propagation without directly mutating the SAT assignment.

Examples:

- policy pressure
- seed bias
- concentration update
- excitation
- inhibition

Allowed emissions:

- `field_pressure`
- `operator_potential_delta`
- `concentration_delta`
- `excitation_delta`
- `inhibition_delta`

### Gate Operators

Timing operators that shape when other operators can fire.

Examples:

- refractory modulation
- threshold modulation
- phase window
- oscillation damping

Allowed emissions:

- `refractory_delta`
- `threshold_delta`
- `phase_context_delta`
- `coupling_window`

### Observer Operators

Diagnostic operators that measure propagation and emit trace annotations. They may influence later learning through recorded data, but they do not directly mutate the current solve state.

Examples:

- domination monitor
- coalition diversity monitor
- loop detector
- phase-lock observer

Allowed emissions:

- `trace_annotation`
- `metric_sample`
- `diagnostic_event`

### Seed Operators

Initial-condition operators that inject early field conditions and priors. They are operators, not metadata.

Examples:

- bootstrap
- density
- trapbreak
- entropy shaping
- stabilization
- plateau

Allowed emissions:

- `initial_pressure`
- `operator_prior`
- `field_pressure`
- `coupling_bias`

## Runtime Flow

At each propagation step:

1. Gather emissions from the previous step.
2. Convert emissions into operator inputs through coupling weights.
3. Update each operator's potential.
4. Skip threshold checks for operators still inside refractory time.
5. Fire eligible operators whose potential crosses threshold.
6. Apply typed emissions in a deterministic order:
   - observer annotations
   - field updates
   - gate updates
   - actuator actions
7. Apply post-spike reset and refractory timing.
8. Emit trace rows for potentials, spikes, emissions, and recovery windows.

The deterministic emission order keeps runs reproducible while still allowing coupled operator dynamics.

## Trace Format

Each fired operator should emit a structured trace row:

```json
{
  "step": 42,
  "operator_id": "density.seed",
  "operator_class": "seed",
  "potential_before": 1.12,
  "threshold": 1.0,
  "fired": true,
  "emission_kind": "field_pressure",
  "emission": {
    "target": "repair_clause",
    "delta": 0.18
  },
  "refractory_until": 43,
  "effective_refractory_ticks": 1,
  "coupling_context": {
    "source": "density.seed",
    "target": "repair_clause",
    "weight": 0.6
  }
}
```

Non-firing trace rows can be sampled or gated behind a debug flag so the trace does not become too large during benchmarks.

## Diagnostics

The first benchmark should ask whether refractory timing improves propagation quality without harming solve behavior.

Track:

- **Domination rate**: how often one operator accounts for a large share of spikes.
- **Coalition diversity**: how many distinct helpful operator groups fire during successful runs.
- **Loop reduction**: repeated short cycles before and after refractory gating.
- **Solve rate**: solved instances and time-to-solve compared with current policies.
- **Phase grouping**: whether operators that help each other begin to fire in stable temporal neighborhoods.

The first target is not maximum performance. The target is proving that a dominant operator can become a drummer instead of the sun.

## Scope

The first slice should:

- Add shared operator state for existing SAT move operators.
- Add `BASE_REFRACTORY_TICKS = 1`.
- Treat existing curriculum seeds as field or seed operators.
- Keep current SAT actuator behavior mostly intact.
- Add typed emissions and trace rows around the existing composer path.
- Keep policy comparison in the benchmark path.

The first slice should not:

- Replace the solver with a new architecture.
- Add a learned controller.
- Make every system concept dynamically pluggable.
- Introduce policy-level controllers outside the operator dynamics.
- Require phase locking to drive behavior before it is measurable.

## Testing

Initial tests should cover:

- Every operator that fires receives at least one refractory tick.
- Operators inside refractory time cannot fire again.
- Operator classes can extend, but not remove, refractory time.
- Field operators emit pressure without directly mutating SAT assignment state.
- Actuator operators still emit SAT actions through the existing solve path.
- Trace rows include potential, threshold, emission kind, and refractory window.
- Benchmark summaries include domination and coalition-diversity metrics.

## Implementation Planning Boundary

After this design is accepted, write an implementation plan that plugs into the existing `sat_composer.py`, `sat_curriculum.py`, and `benchmark_calorimeter.py` paths. The implementation should reuse the current policy comparison and `unittest` style rather than creating a parallel runtime.
