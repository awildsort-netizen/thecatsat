#!/usr/bin/env python3
"""Operator registrations for SAT graph-field and solver composition targets."""

from __future__ import annotations

import math
from collections.abc import Mapping

import sat_field
import sat_curriculum
import sat_furnace
import sprite_detector
from composer import (
    Composer,
    FieldContext,
    FieldOperator,
    materialize_function_operator,
    operator_candidate,
    require_keys,
)


EXCITABLE_POLICY = "excitable_fiber"
CURRICULUM_SEED_POLICY = sat_curriculum.CURRICULUM_SEED_POLICY
EFFECT_BASIS = ("pressure", "bridge", "loop_escape", "memory")


def formula_graph(formula: list) -> sat_field.FormulaGraph:
    return sat_field.formula_graph(formula)


def graph_adjacency(formula_graph: sat_field.FormulaGraph) -> dict:
    return sat_field.formula_graph_to_adjacency(formula_graph)


def spatial_samples(spatial_rows: list | tuple) -> list:
    return sat_field.spatial_rows_to_samples(spatial_rows)


def graph_runners(
    spatial_samples: list,
    graph_adjacency: dict,
    runner_quantile: float = 0.90,
) -> list:
    return sprite_detector.detect_graph_runners(
        spatial_samples,
        graph_adjacency,
        quantile=float(runner_quantile),
    )


def _function_operator(
    function,
    outputs: tuple[str, ...] | None = None,
    name: str | None = None,
) -> FieldOperator:
    return materialize_function_operator(operator_candidate(function), outputs=outputs, name=name)


def _formula_graph_operator() -> FieldOperator:
    return _function_operator(formula_graph)


def _formula_adjacency_operator() -> FieldOperator:
    return _function_operator(graph_adjacency)


def _spatial_samples_operator() -> FieldOperator:
    return _function_operator(spatial_samples)


def _graph_runners_operator() -> FieldOperator:
    return _function_operator(graph_runners)


# ---------------------------------------------------------------------------
# Phase 2: SAT furnace step operators
# ---------------------------------------------------------------------------

def clause_pressure(formula: list, spins: list, temperature: float) -> dict:
    pressures, clause_frustrations, unsatisfied = sat_furnace._clause_pressures(
        formula, spins, float(temperature)
    )
    return {
        "pressures": pressures,
        "clause_frustrations": clause_frustrations,
        "unsatisfied": unsatisfied,
    }


def influence_lift(formula: list, clause_frustrations: list, variables: int) -> list:
    return sat_furnace._variable_influence_matrix(formula, clause_frustrations, int(variables))


def adaptive_gate(
    samples: list,
    unsatisfied: int,
    best_unsatisfied: int,
    adaptive: bool = False,
) -> dict:
    if not adaptive:
        return {"adaptive_active": False, "adaptive_reason": "inactive_disabled"}
    active, reason = sat_furnace._adaptive_activation_state(
        samples, int(unsatisfied), int(best_unsatisfied)
    )
    return {"adaptive_active": active, "adaptive_reason": reason}


def adaptive_strength(
    samples: list,
    heat: float,
    free_energy: float,
    integration: float,
    entropy: float,
    unsatisfied: int,
    best_unsatisfied: int,
    adaptive_active: bool = False,
) -> float:
    if not adaptive_active:
        return 0.0
    return sat_furnace._adaptive_strength(
        samples,
        float(heat),
        float(free_energy),
        float(integration),
        float(entropy),
        int(unsatisfied),
        int(best_unsatisfied),
    )


def adaptive_control(
    samples: list,
    heat: float,
    free_energy: float,
    integration: float,
    entropy: float,
    unsatisfied: int,
    best_unsatisfied: int,
    adaptive_active: bool = False,
    adaptive_gain: float = 1.0,
):
    if not adaptive_active:
        return sat_furnace._default_control_state()
    return sat_furnace._adaptive_control_state(
        samples,
        float(heat),
        float(free_energy),
        float(integration),
        float(entropy),
        int(unsatisfied),
        int(best_unsatisfied),
        float(adaptive_gain),
    )


def _clause_pressure_operator() -> FieldOperator:
    return _function_operator(
        clause_pressure,
        outputs=("pressures", "clause_frustrations", "unsatisfied"),
        name="solver.clause_pressure",
    )


def _influence_lift_operator() -> FieldOperator:
    return _function_operator(influence_lift, outputs=("influence_matrix",), name="solver.influence_lift")


def _adaptive_gate_operator() -> FieldOperator:
    return _function_operator(
        adaptive_gate,
        outputs=("adaptive_active", "adaptive_reason"),
        name="solver.adaptive_gate",
    )


def _adaptive_strength_operator() -> FieldOperator:
    return _function_operator(adaptive_strength, outputs=("adaptive_gain",), name="solver._adaptive_strength")


def _adaptive_control_operator() -> FieldOperator:
    return _function_operator(adaptive_control, outputs=("control_state",), name="solver.adaptive_control")


def bridge_bias(influence_matrix: list) -> list:
    return sat_furnace._graph_bridge_bias(influence_matrix)


def loop_escape_bias(formula: list, clause_frustrations: list, variables: int) -> list:
    return sat_furnace._loop_escape_bias(formula, clause_frustrations, int(variables))


def memory_bias(fiber_memory, formula: list) -> list:
    return sat_furnace._fiber_memory_bias(fiber_memory, formula)


def operator_effects(
    pressures: list,
    bridge_bias: list,
    _loop_escape_bias: list,
    memory_bias: list,
) -> dict:
    return {
        "pressure": list(pressures),
        "bridge": list(bridge_bias),
        "loop_escape": list(_loop_escape_bias),
        "memory": list(memory_bias),
    }


def _bias_operators() -> list[FieldOperator]:
    return [
        _function_operator(bridge_bias, name="solver.bridge_bias"),
        _function_operator(loop_escape_bias, outputs=("_loop_escape_bias",), name="solver._loop_escape_bias"),
        _function_operator(memory_bias, name="solver.memory_bias"),
    ]


def _operator_effects_operator() -> FieldOperator:
    return _function_operator(operator_effects, name="solver.operator_effects")


def _concentration_field_operator() -> FieldOperator:
    """Compute transported operator concentrations over the fixed effect basis."""
    def _run(ctx: FieldContext) -> Mapping[str, object]:
        effects = ctx["operator_effects"]
        memory = ctx["fiber_memory"]
        prev = list(ctx.get("prev_concentrations", [0.0, 0.0, 0.0, 0.0]))
        heat = float(ctx["heat"])
        entropy = float(ctx["entropy"])
        integration = float(ctx["integration"])
        unsatisfied = float(ctx["unsatisfied"])
        clause_count = max(1.0, float(len(ctx["formula"])))

        pressure_mag = _mean_abs(effects["pressure"])
        bridge_mag = _mean_abs(effects["bridge"])
        loop_mag = _mean_abs(effects["loop_escape"])
        memory_mag = _mean_abs(effects["memory"])

        variable_fibers = memory.variable_fibers
        memory_heat = _safe_mean([fiber.heat for fiber in variable_fibers])
        memory_visits = _safe_mean([fiber.visits for fiber in variable_fibers])

        raw = [
            pressure_mag * (0.6 + 0.4 * (1.0 - sat_furnace.clamp01(entropy))),
            bridge_mag * (0.5 + 0.5 * sat_furnace.clamp01(1.0 - integration)),
            loop_mag * (0.4 + 0.6 * sat_furnace.clamp01(unsatisfied / clause_count)),
            memory_mag * (0.45 + 0.55 * sat_furnace.clamp01(memory_heat + memory_visits)),
        ]
        current = _normalize_distribution(raw)
        decay = sat_furnace.clamp01(float(memory.decay))
        transported = [
            decay * prev_value + (1.0 - decay) * value
            for prev_value, value in zip(prev, current)
        ]
        concentrations = _normalize_distribution(transported)
        seed_features = sat_curriculum.extract_features(ctx)
        seed_routing = sat_curriculum.route_seeds(seed_features)
        if str(ctx.get("policy", "baseline")) == CURRICULUM_SEED_POLICY:
            concentrations = sat_curriculum.blend_concentrations(
                concentrations,
                seed_routing.concentration_prior,
            )
        return {
            "concentrations": concentrations,
            "curriculum_seed_names": seed_routing.seed_names,
            "curriculum_seed_routes": list(seed_routing.weights),
            "curriculum_seed_active": seed_routing.active_seed,
            "curriculum_seed_active_index": seed_routing.active_index,
            "curriculum_seed_threshold": seed_routing.threshold,
            "curriculum_seed_excitatory_bias": seed_routing.excitatory_bias,
            "curriculum_seed_inhibitory_bias": seed_routing.inhibitory_bias,
        }

    return FieldOperator(
        name="solver.concentration_field",
        inputs=("operator_effects", "fiber_memory", "prev_concentrations", "heat", "entropy", "integration", "unsatisfied", "formula"),
        outputs=(
            "concentrations",
            "curriculum_seed_names",
            "curriculum_seed_routes",
            "curriculum_seed_active",
            "curriculum_seed_active_index",
            "curriculum_seed_threshold",
            "curriculum_seed_excitatory_bias",
            "curriculum_seed_inhibitory_bias",
        ),
        validate_inputs=require_keys(("operator_effects", "fiber_memory", "heat", "entropy", "integration", "unsatisfied", "formula")),
        run=_run,
    )


def _excitation_inhibition_operator() -> FieldOperator:
    """Decompose concentration channels into excitatory and inhibitory fields."""
    def _run(ctx: FieldContext) -> Mapping[str, object]:
        concentrations = list(ctx["concentrations"])
        heat = sat_furnace.clamp01(float(ctx["heat"]))
        entropy = sat_furnace.clamp01(float(ctx["entropy"]))
        integration = sat_furnace.clamp01(float(ctx["integration"]))
        adaptive_gain = sat_furnace.clamp01(float(ctx.get("adaptive_gain", 0.0)))
        seed_excitatory_bias = 0.0
        seed_inhibitory_bias = 0.0
        if str(ctx.get("policy", "baseline")) == CURRICULUM_SEED_POLICY:
            seed_excitatory_bias = float(ctx.get("curriculum_seed_excitatory_bias", 0.0))
            seed_inhibitory_bias = float(ctx.get("curriculum_seed_inhibitory_bias", 0.0))

        excitatory = []
        inhibitory = []
        for concentration in concentrations:
            excitatory.append(
                concentration
                * (
                    0.35
                    + seed_excitatory_bias
                    + 0.50 * heat
                    + 0.30 * (1.0 - integration)
                    + 0.20 * adaptive_gain
                )
            )
            inhibitory.append(
                concentration
                * (
                    0.30
                    + seed_inhibitory_bias
                    + 0.55 * entropy
                    + 0.15 * (1.0 - heat)
                )
            )
        local_field = [exc - inh for exc, inh in zip(excitatory, inhibitory)]
        return {
            "excitatory_field": excitatory,
            "inhibitory_field": inhibitory,
            "local_field": local_field,
            "field_strength": sum(local_field),
        }

    return FieldOperator(
        name="solver.excitation_inhibition",
        inputs=(
            "concentrations",
            "heat",
            "entropy",
            "integration",
            "adaptive_gain",
            "curriculum_seed_excitatory_bias",
            "curriculum_seed_inhibitory_bias",
            "policy",
        ),
        outputs=("excitatory_field", "inhibitory_field", "local_field", "field_strength"),
        validate_inputs=require_keys(("concentrations", "heat", "entropy", "integration")),
        run=_run,
    )


def _spike_gate_operator() -> FieldOperator:
    """Apply smooth threshold activation to the local field strength."""
    def _run(ctx: FieldContext) -> Mapping[str, object]:
        policy = str(ctx.get("policy", "baseline"))
        if policy not in {EXCITABLE_POLICY, CURRICULUM_SEED_POLICY}:
            return {"spike_strength": 0.0}
        threshold = float(ctx.get("spike_threshold", 0.35))
        if policy == CURRICULUM_SEED_POLICY:
            threshold = float(ctx.get("curriculum_seed_threshold", threshold))
        slope = max(0.1, float(ctx.get("spike_slope", 8.0)))
        field_strength = float(ctx.get("field_strength", 0.0))
        spike_strength = 1.0 / (1.0 + math.exp(-slope * (field_strength - threshold)))
        return {"spike_strength": sat_furnace.clamp01(spike_strength)}

    return FieldOperator(
        name="solver.spike_gate",
        inputs=("policy", "field_strength", "spike_threshold", "spike_slope", "curriculum_seed_threshold"),
        outputs=("spike_strength",),
        validate_inputs=require_keys(("field_strength",)),
        run=_run,
    )


def _mixed_drive_operator() -> FieldOperator:
    """Blend operator effect channels into one spike-modulated drive vector."""
    def _run(ctx: FieldContext) -> Mapping[str, object]:
        policy = str(ctx.get("policy", "baseline"))
        effects = ctx["operator_effects"]
        basis_vectors = [effects[name] for name in EFFECT_BASIS]
        vector_length = len(basis_vectors[0]) if basis_vectors else 0
        if policy not in {EXCITABLE_POLICY, CURRICULUM_SEED_POLICY} or vector_length == 0:
            return {"mixed_drive": [0.0 for _ in range(vector_length)]}

        concentrations = list(ctx["concentrations"])
        local_field = list(ctx["local_field"])
        positive_bias = [
            concentration * max(0.0, field)
            for concentration, field in zip(concentrations, local_field)
        ]
        weights = _normalize_distribution(positive_bias)
        if sum(weights) <= 1e-12:
            weights = _normalize_distribution(concentrations)

        mixed_drive = [0.0 for _ in range(vector_length)]
        for weight, basis in zip(weights, basis_vectors):
            for index, value in enumerate(basis):
                mixed_drive[index] += weight * value
        return {"mixed_drive": sat_furnace._normalize_vector(mixed_drive)}

    return FieldOperator(
        name="solver.mixed_drive",
        inputs=("policy", "operator_effects", "concentrations", "local_field"),
        outputs=("mixed_drive",),
        validate_inputs=require_keys(("operator_effects", "concentrations", "local_field")),
        run=_run,
    )


def _spin_update_operator() -> FieldOperator:
    """Wraps sat_furnace._spin_update_step; outputs next_spins, next_velocity, drive_values."""
    def _run(ctx: FieldContext) -> Mapping[str, object]:
        spins = list(ctx["spins"])
        velocity = list(ctx["velocity"])
        next_spins, next_velocity, drive_values = sat_furnace._spin_update_step(
            spins,
            velocity,
            ctx["pressures"],
            ctx["bridge_bias"],
            ctx["_loop_escape_bias"],
            ctx["memory_bias"],
            ctx["control_state"],
            float(ctx["learning_rate"]),
            float(ctx["inertia"]),
            float(ctx["noise"]),
            float(ctx.get("memory_drive", 0.12)),
            float(ctx.get("adaptive_gain", 0.0)),
            float(ctx.get("memory_scale", 0.0)),
            bool(ctx.get("adaptive_active", False)),
            ctx["rng"],
            mixed_drive=ctx.get("mixed_drive"),
            mixed_scale=float(ctx.get("spike_strength", 0.0)),
            cooling=float(ctx.get("cooling", 1.0)),
            lock_assignment=ctx.get("lock_assignment"),
        )
        return {"next_spins": next_spins, "next_velocity": next_velocity, "drive_values": drive_values}

    return FieldOperator(
        name="solver.spin_update",
        inputs=("spins", "velocity", "pressures", "bridge_bias", "_loop_escape_bias", "memory_bias",
            "control_state", "learning_rate", "inertia", "noise", "rng", "mixed_drive", "spike_strength"),
        outputs=("next_spins", "next_velocity", "drive_values"),
        validate_inputs=require_keys(("spins", "velocity", "pressures", "bridge_bias",
                                      "_loop_escape_bias", "memory_bias", "control_state",
                          "learning_rate", "inertia", "noise", "rng")),
        run=_run,
    )


def build_solver_composer() -> Composer:
    """Returns a Composer registered with all furnace step operators."""
    bias_ops = _bias_operators()
    return Composer(
        operators=(
            _clause_pressure_operator(),
            _influence_lift_operator(),
            _adaptive_gate_operator(),
            _adaptive_strength_operator(),
            _adaptive_control_operator(),
            *bias_ops,
            _operator_effects_operator(),
            _concentration_field_operator(),
            _excitation_inhibition_operator(),
            _spike_gate_operator(),
            _mixed_drive_operator(),
            _spin_update_operator(),
            # Phase B operators
            _thermo_metrics_operator(),
            _best_tracker_operator(),
            _lock_assignment_operator(),
            _cooling_operator(),
            _memory_scale_operator(),
            _sample_append_operator(),
            _spatial_append_operator(),
            _trace_append_operator(),
            _final_assignment_operator(),
            _furnace_result_operator(),
        )
    )


# ---------------------------------------------------------------------------
# Phase B: inline epoch operators
# ---------------------------------------------------------------------------

def thermo_metrics(spins: list, clause_frustrations: list, influence_matrix: list) -> dict:
    heat = sum(f * f for f in clause_frustrations) / max(1, len(clause_frustrations))
    free_energy = (
        sat_furnace._statistics_like_variance(spins)
        + sat_furnace._statistics_like_variance(clause_frustrations)
    )
    integration = sat_furnace._integration_score(influence_matrix)
    entropy = sat_furnace._assignment_entropy(spins)
    return {
        "heat": heat,
        "free_energy": free_energy,
        "integration": integration,
        "entropy": entropy,
    }


def best_tracker(
    spins: list,
    unsatisfied: int,
    prev_best_spins: list,
    prev_best_unsatisfied: int,
) -> dict:
    if int(unsatisfied) < int(prev_best_unsatisfied):
        return {"best_spins": list(spins), "best_unsatisfied": int(unsatisfied)}
    return {"best_spins": prev_best_spins, "best_unsatisfied": int(prev_best_unsatisfied)}


def lock_assignment(unsatisfied: int, planted_assignment: list | None = None) -> list[float] | None:
    if int(unsatisfied) == 0 and planted_assignment is not None:
        return [1.0 if value else -1.0 for value in planted_assignment]
    return None


def cooling(t: int, steps: int) -> float:
    return 1.0 - (int(t) / max(1, int(steps) - 1))


def memory_scale(
    control_state,
    samples: list,
    unsatisfied: int,
    best_unsatisfied: int,
    adaptive_gain: float = 0.0,
) -> float:
    return sat_furnace._action_memory_scale(
        control_state.action,
        samples,
        int(unsatisfied),
        int(best_unsatisfied),
        float(adaptive_gain),
    )


def _thermo_metrics_operator() -> FieldOperator:
    return _function_operator(
        thermo_metrics,
        outputs=("heat", "free_energy", "integration", "entropy"),
        name="solver.thermo_metrics",
    )


def _best_tracker_operator() -> FieldOperator:
    return _function_operator(
        best_tracker,
        outputs=("best_spins", "best_unsatisfied"),
        name="solver.best_tracker",
    )


def _lock_assignment_operator() -> FieldOperator:
    return _function_operator(lock_assignment, name="solver.lock_assignment")


def _cooling_operator() -> FieldOperator:
    return _function_operator(cooling, name="solver.cooling")


def _memory_scale_operator() -> FieldOperator:
    return _function_operator(memory_scale, name="solver.memory_scale")


def _sample_append_operator() -> FieldOperator:
    """B7 — append a FurnaceSample for the current epoch and return updated list.

    Reads from ``prev_samples`` (carry-forward key set by the epoch driver) so
    the Composer does not confuse the operator's output key with its input.
    """
    def _run(ctx: FieldContext) -> Mapping[str, object]:
        samples = list(ctx["prev_samples"])  # shallow copy from carry-forward
        samples.append(
            sat_furnace.FurnaceSample(
                t=int(ctx["t"]),
                heat=float(ctx["heat"]),
                free_energy=float(ctx["free_energy"]),
                integration=float(ctx["integration"]),
                unsatisfied_clauses=int(ctx["unsatisfied"]),
                _assignment_entropy=float(ctx["entropy"]),
            )
        )
        return {"samples": samples}

    return FieldOperator(
        name="solver.sample_append",
        inputs=("t", "heat", "free_energy", "integration", "unsatisfied", "entropy", "prev_samples"),
        outputs=("samples",),
        validate_inputs=require_keys(("t", "heat", "free_energy", "integration",
                                      "unsatisfied", "entropy", "prev_samples")),
        run=_run,
    )


def _spatial_append_operator() -> FieldOperator:
    """B8 — extend spatial_samples list with current-epoch frame rows.

    Reads from ``prev_spatial_samples`` (carry-forward key set by the epoch
    driver) so the Composer does not confuse the operator's output with its
    input.
    """
    def _run(ctx: FieldContext) -> Mapping[str, object]:
        spatial_samples = list(ctx["prev_spatial_samples"])
        rows = sat_furnace._spatial_frame_rows(
            int(ctx["t"]),
            ctx["formula"],
            ctx["spins"],
            ctx["clause_frustrations"],
            ctx["influence_matrix"],
            ctx["pressures"],
            float(ctx["temperature"]),
        )
        spatial_samples.extend(rows)
        return {"spatial_samples": spatial_samples}

    return FieldOperator(
        name="solver.spatial_append",
        inputs=("t", "formula", "spins", "clause_frustrations", "influence_matrix",
                "pressures", "temperature", "prev_spatial_samples"),
        outputs=("spatial_samples",),
        validate_inputs=require_keys(("t", "formula", "spins", "clause_frustrations",
                                      "influence_matrix", "pressures", "temperature",
                                      "prev_spatial_samples")),
        run=_run,
    )


def _trace_append_operator() -> FieldOperator:
    """B9 — append OperatorTrace rows for this epoch to operator_traces.

    Reads from ``prev_operator_traces`` (carry-forward key set by the epoch
    driver) so the Composer does not confuse the operator's output with its
    input.
    """
    def _run(ctx: FieldContext) -> Mapping[str, object]:
        traces = list(ctx["prev_operator_traces"])
        t = int(ctx["t"])
        control = ctx["control_state"]
        adaptive_active = bool(ctx["adaptive_active"])
        adaptive_reason = str(ctx["adaptive_reason"])
        adaptive_gain = float(ctx.get("adaptive_gain", 0.0))
        heat = float(ctx["heat"])
        entropy = float(ctx["entropy"])
        integration = float(ctx["integration"])
        unsatisfied = int(ctx["unsatisfied"])
        pressures = ctx["pressures"]
        influence_matrix = ctx["influence_matrix"]
        memory_bias = ctx["memory_bias"]
        drive_values = ctx["drive_values"]
        concentrations = list(ctx.get("concentrations", []))
        excitatory_field = list(ctx.get("excitatory_field", []))
        inhibitory_field = list(ctx.get("inhibitory_field", []))
        mixed_drive = list(ctx.get("mixed_drive", []))
        spike_strength = float(ctx.get("spike_strength", 0.0))
        policy = str(ctx.get("policy", "baseline"))
        # Tests-as-AF: an operator trace is "active" when the current
        # concentration climate decompresses it, not merely because the
        # policy name licenses it. We threshold the normalized
        # concentration vector against a uniform-prior floor — any
        # channel above 1/len(channels) is enriched relative to uniform
        # and counts as activation. Fall back to the legacy policy gate
        # when no concentration field has been computed yet (early
        # epochs, older callers).
        policy_excitable = policy in {EXCITABLE_POLICY, CURRICULUM_SEED_POLICY}
        if concentrations:
            uniform = 1.0 / len(concentrations)
            climate_threshold = uniform * 1.05
            excitable_active = (
                max(concentrations) > climate_threshold or policy_excitable
            )
        else:
            excitable_active = policy_excitable
        curriculum_active = policy == CURRICULUM_SEED_POLICY
        memory_scale = float(ctx["memory_scale"])
        previous_unsatisfied = int(ctx["previous_unsatisfied"])
        previous_integration = float(ctx["previous_integration"])
        delta_unsat = previous_unsatisfied - unsatisfied
        delta_integ = integration - previous_integration

        for op_name, op_outputs, op_active, op_mem_scale in (
            ("clause_pressure", pressures, True, 0.0),
            ("influence_lift", [sum(row) for row in influence_matrix], True, 0.0),
            ("adaptive_gate", [1.0 if adaptive_active else 0.0, adaptive_gain], adaptive_active, 0.0),
            ("control_action", [
                control.learning_rate_scale, control.inertia_scale,
                control.noise_scale, control.bridge_scale, control.loop_escape_scale,
            ], adaptive_active, 0.0),
            ("_fiber_memory_bias", memory_bias, adaptive_active and memory_scale > 0.0, memory_scale),
            ("excitable_concentration", concentrations, excitable_active, 0.0),
            ("excitable_field", [
                _safe_mean(excitatory_field),
                _safe_mean(inhibitory_field),
                float(ctx.get("field_strength", 0.0)),
            ], excitable_active, 0.0),
            ("excitable_spike", [spike_strength], excitable_active and spike_strength > 0.01, spike_strength),
            ("excitable_mixture", mixed_drive, excitable_active and spike_strength > 0.01, spike_strength),
            ("curriculum_seed_route", list(ctx.get("curriculum_seed_routes", [])), curriculum_active, 0.0),
            ("curriculum_seed_active", [float(ctx.get("curriculum_seed_active_index", 0))], curriculum_active, 0.0),
            ("spin_update", drive_values, True, memory_scale),
        ):
            sat_furnace._trace_operator(
                traces, t, op_name, op_active,
                control.action, adaptive_reason,
                heat, entropy, integration, unsatisfied,
                op_outputs, op_mem_scale,
                delta_unsat, delta_integ,
            )
        return {"operator_traces": traces}

    return FieldOperator(
        name="solver.trace_append",
        inputs=("t", "prev_operator_traces", "control_state", "adaptive_active", "adaptive_reason",
                "adaptive_gain", "heat", "entropy", "integration", "unsatisfied",
                "pressures", "influence_matrix", "memory_bias", "drive_values",
                "memory_scale", "previous_unsatisfied", "previous_integration",
                "concentrations", "excitatory_field", "inhibitory_field",
                "field_strength", "spike_strength", "mixed_drive", "policy"),
        outputs=("operator_traces",),
        validate_inputs=require_keys(("t", "prev_operator_traces", "control_state", "adaptive_active",
                                      "adaptive_reason", "heat", "entropy", "integration",
                                      "unsatisfied", "pressures", "influence_matrix",
                                      "memory_bias", "drive_values", "memory_scale",
                                      "previous_unsatisfied", "previous_integration")),
        run=_run,
    )


def _safe_mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _mean_abs(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(abs(value) for value in values) / len(values)


def _normalize_distribution(values: list[float]) -> list[float]:
    if not values:
        return []
    clipped = [max(0.0, float(value)) for value in values]
    total = sum(clipped)
    if total <= 1e-12:
        return [1.0 / len(clipped) for _ in clipped]
    return [value / total for value in clipped]


def _final_assignment_operator() -> FieldOperator:
    """B10 — pick final assignment: spins >= 0, fall back to best_spins if better."""
    def _run(ctx: FieldContext) -> Mapping[str, object]:
        spins = ctx["spins"]
        best_spins = ctx["best_spins"]
        formula = ctx["formula"]
        final = [s >= 0 for s in spins]
        if not all(sat_furnace.clause_satisfied(c, final) for c in formula):
            best = [s >= 0 for s in best_spins]
            if (
                sum(not sat_furnace.clause_satisfied(c, best) for c in formula)
                < sum(not sat_furnace.clause_satisfied(c, final) for c in formula)
            ):
                final = best
        solved = all(sat_furnace.clause_satisfied(c, final) for c in formula)
        return {"final_assignment": final, "solved": solved}

    return FieldOperator(
        name="solver.final_assignment",
        inputs=("spins", "best_spins", "formula"),
        outputs=("final_assignment", "solved"),
        validate_inputs=require_keys(("spins", "best_spins", "formula")),
        run=_run,
    )


def _furnace_result_operator() -> FieldOperator:
    """B11 — assemble FurnaceResult from accumulated epoch state."""
    def _run(ctx: FieldContext) -> Mapping[str, object]:
        return {
            "furnace_result": sat_furnace.FurnaceResult(
                formula=ctx["formula"],
                planted_assignment=ctx.get("planted_assignment"),
                samples=ctx["samples"],
                spatial_samples=ctx["spatial_samples"],
                operator_traces=ctx["operator_traces"],
                final_assignment=ctx["final_assignment"],
                solved=bool(ctx["solved"]),
            )
        }

    return FieldOperator(
        name="solver.furnace_result",
        inputs=("formula", "samples", "spatial_samples", "operator_traces",
                "final_assignment", "solved"),
        outputs=("furnace_result",),
        validate_inputs=require_keys(("formula", "samples", "spatial_samples",
                                      "operator_traces", "final_assignment", "solved")),
        run=_run,
    )


# ---------------------------------------------------------------------------
# Phase E: trial-level operators for benchmark_calorimeter
# ---------------------------------------------------------------------------

def _trial_furnace_operator() -> FieldOperator:
    """E1 — run the furnace solver and return a FurnaceResult."""
    def _run(ctx: FieldContext) -> Mapping[str, object]:
        result = sat_furnace.run_furnace(
            formula=ctx["formula"],
            variables=int(ctx["variables"]),
            steps=int(ctx["steps"]),
            rng=ctx["rng"],
            temperature=float(ctx["temperature"]),
            learning_rate=float(ctx["learning_rate"]),
            inertia=float(ctx["inertia"]),
            noise=float(ctx["noise"]),
            planted_assignment=ctx.get("planted_assignment"),
            adaptive=bool(ctx.get("adaptive", False)),
            memory_decay=float(ctx.get("memory_decay", 0.92)),
            memory_drive=float(ctx.get("memory_drive", 0.12)),
            policy=str(ctx.get("policy", "baseline")),
            spike_threshold=float(ctx.get("spike_threshold", 0.35)),
            spike_slope=float(ctx.get("spike_slope", 8.0)),
        )
        return {"furnace_result": result}

    return FieldOperator(
        name="trial.furnace",
        inputs=("formula", "variables", "steps", "rng",
                "temperature", "learning_rate", "inertia", "noise"),
        outputs=("furnace_result",),
        validate_inputs=require_keys(("formula", "variables", "steps", "rng",
                                      "temperature", "learning_rate", "inertia", "noise")),
        run=_run,
    )


def _trial_spatial_rows_operator() -> FieldOperator:
    """E1b — bridge FurnaceResult.spatial_samples → spatial_rows for graph operators."""
    def _run(ctx: FieldContext) -> Mapping[str, object]:
        result = ctx["furnace_result"]
        return {"spatial_rows": result.spatial_samples}

    return FieldOperator(
        name="trial.spatial_rows",
        inputs=("furnace_result",),
        outputs=("spatial_rows",),
        validate_inputs=require_keys(("furnace_result",)),
        run=_run,
    )


def _trial_spectral_operator() -> FieldOperator:
    """E2 — run spectral calorimeter analysis on furnace samples."""
    def _run(ctx: FieldContext) -> Mapping[str, object]:
        import spectral_calorimeter
        result = ctx["furnace_result"]
        frames = [
            spectral_calorimeter.SignalFrame(
                t=float(s.t),
                heat=s.heat,
                free_energy=s.free_energy,
                integration=s.integration,
            )
            for s in result.samples
        ]
        report = spectral_calorimeter.analyze_frames(
            frames,
            window_size=int(ctx["window"]),
            step_size=int(ctx["step_size"]),
        )
        return {"calorimeter_report": report}

    return FieldOperator(
        name="trial.spectral",
        inputs=("furnace_result", "window", "step_size"),
        outputs=("calorimeter_report",),
        validate_inputs=require_keys(("furnace_result", "window", "step_size")),
        run=_run,
    )


def _trial_sprites_operator() -> FieldOperator:
    """E4 — detect sprites from spatial samples."""
    def _run(ctx: FieldContext) -> Mapping[str, object]:
        return {"sprites": sprite_detector.detect_sprites(ctx["spatial_samples"])}

    return FieldOperator(
        name="trial.sprites",
        inputs=("spatial_samples",),
        outputs=("sprites",),
        validate_inputs=require_keys(("spatial_samples",)),
        run=_run,
    )


def _trial_metrics_row_operator() -> FieldOperator:
    """E3 — assemble the benchmark metrics dict from all trial outputs."""
    def _run(ctx: FieldContext) -> Mapping[str, object]:
        import benchmark_calorimeter as _bc
        result = ctx["furnace_result"]
        report = ctx["calorimeter_report"]
        formula = ctx["formula"]
        spatial_samples = ctx["spatial_samples"]
        runners = ctx["graph_runners"]
        sprites = ctx["sprites"]
        kind = str(ctx["kind"])
        adaptive = bool(ctx.get("adaptive", False))
        policy = str(ctx.get("policy", "baseline"))
        spike_threshold = float(ctx.get("spike_threshold", 0.35))
        spike_slope = float(ctx.get("spike_slope", 8.0))
        window = int(ctx["window"])
        step_size = int(ctx["step_size"])

        furnace_best_unsatisfied = min(
            (s.unsatisfied_clauses for s in result.samples), default=len(formula)
        )
        furnace_final_assignment_unsatisfied = _bc.count_unsatisfied(
            formula, result.final_assignment
        )

        row: dict[str, _bc.Scalar] = {
            "kind": kind,
            "seed": int(ctx["seed"]),
            "variables": int(ctx["variables"]),
            "clauses": len(formula),
            "steps": int(ctx["steps"]),
            "adaptive": adaptive,
            "policy": policy,
            "spike_threshold": spike_threshold if policy in {EXCITABLE_POLICY, CURRICULUM_SEED_POLICY} else 0.0,
            "spike_slope": spike_slope if policy in {EXCITABLE_POLICY, CURRICULUM_SEED_POLICY} else 0.0,
            "memory_decay": float(ctx.get("memory_decay", 0.0)) if adaptive else 0.0,
            "memory_drive": float(ctx.get("memory_drive", 0.0)) if adaptive else 0.0,
            "random_restarts": int(ctx.get("baseline_restarts", 0)),
            "random_solved": bool(ctx["random_solved"]),
            "random_best_unsatisfied": int(ctx["random_best_unsatisfied"]),
            "walksat_flips": int(ctx["walksat_flips"]),
            "walksat_solved": bool(ctx["walksat_solved"]),
            "walksat_best_unsatisfied": int(ctx["walksat_best_unsatisfied"]),
            "solved": result.solved,
            "furnace_best_unsatisfied": furnace_best_unsatisfied,
            "furnace_final_assignment_unsatisfied": furnace_final_assignment_unsatisfied,
            "final_unsatisfied_clauses": result.samples[-1].unsatisfied_clauses,
            "prediction": report.prediction,
            "confidence": report.confidence,
            "redshift_rate": report.redshift_rate,
            "spectral_entropy_slope": report.entropy_slope,
            "integration_slope": report.integration_slope,
            "concentration_slope": report.concentration_slope,
            "fragmentation_slope": report.fragmentation_slope,
            "collapse_slope": report.collapse_slope,
            "sprite_count": len(sprites),
            "sprite_mean_lifetime": _bc.mean(s.lifetime for s in sprites),
            "sprite_runner_count": _bc.count_by_classification(sprites, "runner"),
            "sprite_standing_wave_count": _bc.count_by_classification(sprites, "standing_wave"),
            "sprite_collapse_front_count": _bc.count_by_classification(sprites, "collapse_front"),
            "graph_runner_count": len(runners),
            "runner_mean_lifetime": _bc.mean(r.lifetime for r in runners),
            "runner_mean_unique_nodes": _bc.mean(r.unique_nodes for r in runners),
            "runner_mean_path_novelty": _bc.mean(r.path_novelty for r in runners),
            "runner_mean_revisit_rate": _bc.mean(r.revisit_rate for r in runners),
            "runner_mean_mass_density": _bc.mean(r.mass_density for r in runners),
            "runner_mean_loop_score": _bc.mean(r.loop_score for r in runners),
            "runner_mean_escape_score": _bc.mean(r.escape_score for r in runners),
            "runner_mean_bridge_score": _bc.mean(r.bridge_score for r in runners),
            "runner_mean_collapse_contribution": _bc.mean(r.collapse_contribution for r in runners),
            "exploratory_runner_count": _bc.count_by_classification(runners, "exploratory_runner"),
            "reinforced_loop_count": _bc.count_by_classification(runners, "reinforced_loop"),
            "escaping_runner_count": _bc.count_by_classification(runners, "escaping_runner"),
            "unsat_trap_count": _bc.count_by_classification(runners, "unsat_trap"),
            "diffuse_runner_count": _bc.count_by_classification(runners, "diffuse_runner"),
        }
        row["global_collapse_score"] = (
            -float(row["redshift_rate"])
            - float(row["spectral_entropy_slope"])
            + float(row["integration_slope"])
            + float(row["collapse_slope"])
        )
        row["runner_ecology_score"] = (
            float(row["runner_mean_path_novelty"])
            + float(row["runner_mean_escape_score"])
            - float(row["runner_mean_mass_density"])
            - float(row["runner_mean_loop_score"])
        )
        row["unsat_trap_score"] = (
            float(row["runner_mean_mass_density"])
            + float(row["runner_mean_revisit_rate"])
            + float(row["runner_mean_loop_score"])
            - float(row["runner_mean_escape_score"])
        )
        row.update(_bc.puzzle_ecology_metrics(
            formula=formula,
            variables=int(ctx["variables"]),
            random_best_unsatisfied=int(ctx["random_best_unsatisfied"]),
            walksat_best_unsatisfied=int(ctx["walksat_best_unsatisfied"]),
            furnace_best_unsatisfied=int(furnace_best_unsatisfied),
            random_solved=bool(ctx["random_solved"]),
            walksat_solved=bool(ctx["walksat_solved"]),
            furnace_solved=bool(result.solved),
        ))
        genome = _bc.solver_composition_genome()
        row.update(_bc.composition_genome_metrics(
            genome=genome,
            puzzle_border_score=float(row["puzzle_border_score"]),
            puzzle_composition_pressure=float(row["puzzle_composition_pressure"]),
            solved=bool(result.solved),
            furnace_best_unsatisfied=int(furnace_best_unsatisfied),
            walksat_best_unsatisfied=int(ctx["walksat_best_unsatisfied"]),
        ))
        row.update(_bc.gene_border_mutation_metrics(
            genome=genome,
            traces=result.operator_traces,
            puzzle_border_score=float(row["puzzle_border_score"]),
            puzzle_composition_pressure=float(row["puzzle_composition_pressure"]),
            solved=bool(result.solved),
            furnace_best_unsatisfied=int(furnace_best_unsatisfied),
            walksat_best_unsatisfied=int(ctx["walksat_best_unsatisfied"]),
        ))
        row.update(_bc.coupling_metrics(
            report.windows, spatial_samples, runners,
            window_size=window, step_size=step_size,
        ))
        row.update(_bc.choice_policy_metrics(report.windows, runners))
        row.update(_bc.operator_trace_metrics(result.operator_traces))
        row.update(_bc.transition_motif_metrics(
            result.operator_traces,
            climate_metrics=row,
        ))
        row.update(_excitable_trace_metrics(result.operator_traces, policy))
        return {"metrics_row": row}

    return FieldOperator(
        name="trial.metrics_row",
        inputs=("kind", "seed", "variables", "steps", "formula", "furnace_result",
                "calorimeter_report", "spatial_samples", "graph_runners", "sprites",
                "random_solved", "random_best_unsatisfied",
                "walksat_flips", "walksat_solved", "walksat_best_unsatisfied",
            "window", "step_size"),
        outputs=("metrics_row",),
        validate_inputs=require_keys(("kind", "seed", "variables", "steps", "formula",
                                      "furnace_result", "calorimeter_report",
                                      "spatial_samples", "graph_runners", "sprites",
                                      "random_solved", "random_best_unsatisfied",
                                      "walksat_flips", "walksat_solved",
                          "walksat_best_unsatisfied", "window", "step_size")),
        run=_run,
    )


def _excitable_trace_metrics(
    traces: list[sat_furnace.OperatorTrace],
    policy: str,
) -> dict[str, float | bool]:
    concentration_traces = [
        trace for trace in traces if trace.operator == "excitable_concentration"
    ]
    field_traces = [trace for trace in traces if trace.operator == "excitable_field"]
    spike_traces = [trace for trace in traces if trace.operator == "excitable_spike"]
    mixture_traces = [trace for trace in traces if trace.operator == "excitable_mixture"]
    seed_route_traces = [
        trace for trace in traces if trace.operator == "curriculum_seed_route"
    ]
    seed_active_traces = [
        trace for trace in traces if trace.operator == "curriculum_seed_active"
    ]
    excitable_policy = policy in {EXCITABLE_POLICY, CURRICULUM_SEED_POLICY}
    steps = max(1, len(spike_traces))
    active_spikes = [trace for trace in spike_traces if trace.active]
    mean_spike = (
        sum(trace.output_mean for trace in spike_traces) / len(spike_traces)
        if spike_traces
        else 0.0
    )
    return {
        "excitable_trace_present": bool(spike_traces),
        "excitable_spike_activation_count": float(len(active_spikes)),
        "excitable_spike_activation_rate": float(len(active_spikes)) / steps,
        "excitable_spike_mean_strength": mean_spike,
        "excitable_concentration_mean": (
            sum(trace.output_mean for trace in concentration_traces)
            / max(1, len(concentration_traces))
        ),
        "excitable_field_mean": (
            sum(trace.output_mean for trace in field_traces) / max(1, len(field_traces))
        ),
        "excitable_mixed_action_mean": (
            sum(trace.output_mean for trace in mixture_traces)
            / max(1, len(mixture_traces))
        ),
        "curriculum_seed_route_present": bool(seed_route_traces),
        "curriculum_seed_route_activation_rate": (
            len([trace for trace in seed_route_traces if trace.active])
            / max(1, len(seed_route_traces))
        ),
        "curriculum_seed_active_mean_index": (
            sum(trace.output_mean for trace in seed_active_traces)
            / max(1, len(seed_active_traces))
        ),
        "excitable_trace_chain_ok": bool(
            not excitable_policy
            or (
                concentration_traces
                and field_traces
                and spike_traces
                and mixture_traces
            )
        ),
    }


def build_trial_composer() -> Composer:
    """Returns a Composer for a single benchmark trial (E1–E4 operators + graph)."""
    return Composer(
        operators=(
            _trial_furnace_operator(),
            _trial_spatial_rows_operator(),
            _trial_spectral_operator(),
            _trial_sprites_operator(),
            _trial_metrics_row_operator(),
            # graph / spatial resolution
            _formula_graph_operator(),
            _formula_adjacency_operator(),
            _spatial_samples_operator(),
            _graph_runners_operator(),
        )
    )

