#!/usr/bin/env python3
"""Operator registrations for SAT graph-field and solver composition targets."""

from __future__ import annotations

import math
from collections.abc import Mapping

import sat_field
import sat_curriculum
import sat_furnace
import sprite_detector
from composer import Composer, FieldContext, FieldOperator, compose_validators, require_keys, require_types


EXCITABLE_POLICY = "excitable_fiber"
CURRICULUM_SEED_POLICY = sat_curriculum.CURRICULUM_SEED_POLICY
EFFECT_BASIS = ("pressure", "bridge", "loop_escape", "memory")


def _formula_graph_operator() -> FieldOperator:
    return FieldOperator(
        name="formula.graph",
        inputs=("formula",),
        outputs=("formula_graph",),
        validate_inputs=compose_validators(
            require_keys(("formula",)),
            require_types({"formula": list}),
        ),
        validate_outputs=require_types({"formula_graph": sat_field.FormulaGraph}),
        run=lambda context: {"formula_graph": sat_field.formula_graph(context["formula"])},
    )


def _formula_adjacency_operator() -> FieldOperator:
    return FieldOperator(
        name="formula.adjacency",
        inputs=("formula_graph",),
        outputs=("graph_adjacency",),
        validate_inputs=compose_validators(
            require_keys(("formula_graph",)),
            require_types({"formula_graph": sat_field.FormulaGraph}),
        ),
        validate_outputs=require_types({"graph_adjacency": dict}),
        run=lambda context: {
            "graph_adjacency": sat_field.formula_graph_to_adjacency(context["formula_graph"])
        },
    )


def _spatial_samples_operator() -> FieldOperator:
    return FieldOperator(
        name="spatial.samples",
        inputs=("spatial_rows",),
        outputs=("spatial_samples",),
        validate_inputs=compose_validators(
            require_keys(("spatial_rows",)),
            require_types({"spatial_rows": (list, tuple)}),
        ),
        validate_outputs=require_types({"spatial_samples": list}),
        run=lambda context: {
            "spatial_samples": sat_field.spatial_rows_to_samples(context["spatial_rows"])
        },
    )


def _graph_runners_operator() -> FieldOperator:
    return FieldOperator(
        name="graph.runners",
        inputs=("spatial_samples", "graph_adjacency"),
        outputs=("graph_runners",),
        validate_inputs=compose_validators(
            require_keys(("spatial_samples", "graph_adjacency")),
            require_types({"spatial_samples": list, "graph_adjacency": dict}),
        ),
        validate_outputs=require_types({"graph_runners": list}),
        run=_run_graph_runners,
    )


def _run_graph_runners(context: FieldContext) -> Mapping[str, object]:
    quantile = context.get("runner_quantile", 0.90)
    return {
        "graph_runners": sprite_detector.detect_graph_runners(
            context["spatial_samples"],
            context["graph_adjacency"],
            quantile=float(quantile),
        )
    }


def build_graph_composer() -> Composer:
    return Composer(
        operators=(
            _formula_graph_operator(),
            _formula_adjacency_operator(),
            _spatial_samples_operator(),
            _graph_runners_operator(),
        )
    )


def run_graph_targets(targets: tuple[str, ...] | list[str], context: Mapping[str, object]) -> dict[str, object]:
    return build_graph_composer().run(targets, context)


# ---------------------------------------------------------------------------
# Phase 2: SAT furnace step operators
# ---------------------------------------------------------------------------

def _formula_generate_operator() -> FieldOperator:
    """Wraps sat_furnace.generate_formula; produces formula + planted_assignment."""
    def _run(ctx: FieldContext) -> Mapping[str, object]:
        formula, planted = sat_furnace.generate_formula(
            str(ctx["kind"]),
            int(ctx["variables"]),
            int(ctx["clauses"]),
            int(ctx["clause_size"]),
            ctx["rng"],
        )
        return {"formula": formula, "planted_assignment": planted}

    return FieldOperator(
        name="formula.generate",
        inputs=("kind", "variables", "clauses", "clause_size", "rng"),
        outputs=("formula", "planted_assignment"),
        validate_inputs=require_keys(("kind", "variables", "clauses", "clause_size", "rng")),
        run=_run,
    )


def _formula_validate_operator() -> FieldOperator:
    """Validates formula length and variable bounds; passes formula through."""
    def _run(ctx: FieldContext) -> Mapping[str, object]:
        formula = ctx["formula"]
        variables = int(ctx["variables"])
        for clause in formula:
            for var, _ in clause:
                if var < 0 or var >= variables:
                    raise ValueError(f"variable index {var} out of range [0, {variables})")
        return {"validated_formula": formula}

    return FieldOperator(
        name="formula.validate",
        inputs=("formula", "variables"),
        outputs=("validated_formula",),
        validate_inputs=require_keys(("formula", "variables")),
        run=_run,
    )


def _clause_pressure_operator() -> FieldOperator:
    """Wraps sat_furnace._clause_pressures."""
    def _run(ctx: FieldContext) -> Mapping[str, object]:
        pressures, clause_frustrations, unsatisfied = sat_furnace._clause_pressures(
            ctx["formula"], ctx["spins"], float(ctx["temperature"])
        )
        return {
            "pressures": pressures,
            "clause_frustrations": clause_frustrations,
            "unsatisfied": unsatisfied,
        }

    return FieldOperator(
        name="solver.clause_pressure",
        inputs=("formula", "spins", "temperature"),
        outputs=("pressures", "clause_frustrations", "unsatisfied"),
        validate_inputs=require_keys(("formula", "spins", "temperature")),
        run=_run,
    )


def _influence_lift_operator() -> FieldOperator:
    """Wraps sat_furnace._variable_influence_matrix."""
    def _run(ctx: FieldContext) -> Mapping[str, object]:
        return {
            "influence_matrix": sat_furnace._variable_influence_matrix(
                ctx["formula"], ctx["clause_frustrations"], int(ctx["variables"])
            )
        }

    return FieldOperator(
        name="solver.influence_lift",
        inputs=("formula", "clause_frustrations", "variables"),
        outputs=("influence_matrix",),
        validate_inputs=require_keys(("formula", "clause_frustrations", "variables")),
        run=_run,
    )


def _adaptive_gate_operator() -> FieldOperator:
    """Wraps sat_furnace._adaptive_activation_state; gated by adaptive flag."""
    def _run(ctx: FieldContext) -> Mapping[str, object]:
        if not ctx.get("adaptive", False):
            return {"adaptive_active": False, "adaptive_reason": "inactive_disabled"}
        active, reason = sat_furnace._adaptive_activation_state(
            ctx["samples"], int(ctx["unsatisfied"]), int(ctx["best_unsatisfied"])
        )
        return {"adaptive_active": active, "adaptive_reason": reason}

    return FieldOperator(
        name="solver.adaptive_gate",
        inputs=("adaptive", "samples", "unsatisfied", "best_unsatisfied"),
        outputs=("adaptive_active", "adaptive_reason"),
        validate_inputs=require_keys(("samples", "unsatisfied", "best_unsatisfied")),
        run=_run,
    )


def _adaptive_strength_operator() -> FieldOperator:
    """Wraps sat_furnace._adaptive_strength; zero when adaptive gate is off."""
    def _run(ctx: FieldContext) -> Mapping[str, object]:
        if not ctx.get("adaptive_active", False):
            return {"adaptive_gain": 0.0}
        gain = sat_furnace._adaptive_strength(
            ctx["samples"],
            float(ctx["heat"]),
            float(ctx["free_energy"]),
            float(ctx["integration"]),
            float(ctx["entropy"]),
            int(ctx["unsatisfied"]),
            int(ctx["best_unsatisfied"]),
        )
        return {"adaptive_gain": gain}

    return FieldOperator(
        name="solver._adaptive_strength",
        inputs=("adaptive_active", "samples", "heat", "free_energy", "integration", "entropy", "unsatisfied", "best_unsatisfied"),
        outputs=("adaptive_gain",),
        validate_inputs=require_keys(("adaptive_active", "samples", "unsatisfied", "best_unsatisfied")),
        run=_run,
    )


def _adaptive_control_operator() -> FieldOperator:
    """Wraps sat_furnace._adaptive_control_state; falls back to default when inactive."""
    def _run(ctx: FieldContext) -> Mapping[str, object]:
        if not ctx.get("adaptive_active", False):
            return {"control_state": sat_furnace._default_control_state()}
        control = sat_furnace._adaptive_control_state(
            ctx["samples"],
            float(ctx["heat"]),
            float(ctx["free_energy"]),
            float(ctx["integration"]),
            float(ctx["entropy"]),
            int(ctx["unsatisfied"]),
            int(ctx["best_unsatisfied"]),
            float(ctx.get("adaptive_gain", 1.0)),
        )
        return {"control_state": control}

    return FieldOperator(
        name="solver.adaptive_control",
        inputs=("adaptive_active", "adaptive_gain", "samples", "heat", "free_energy", "integration", "entropy", "unsatisfied", "best_unsatisfied"),
        outputs=("control_state",),
        validate_inputs=require_keys(("adaptive_active", "samples", "unsatisfied", "best_unsatisfied")),
        run=_run,
    )


def _memory_init_operator() -> FieldOperator:
    """Wraps sat_furnace._initialize_fiber_bundle_memory."""
    def _run(ctx: FieldContext) -> Mapping[str, object]:
        return {
            "fiber_memory": sat_furnace._initialize_fiber_bundle_memory(
                int(ctx["variables"]),
                len(ctx["formula"]),
                float(ctx.get("memory_decay", 0.92)),
            )
        }

    return FieldOperator(
        name="solver.memory_init",
        inputs=("variables", "formula"),
        outputs=("fiber_memory",),
        validate_inputs=require_keys(("variables", "formula")),
        run=_run,
    )


def _bias_operators() -> list[FieldOperator]:
    """Wraps _graph_bridge_bias, _loop_escape_bias, and _fiber_memory_bias."""
    def _run_bridge(ctx: FieldContext) -> Mapping[str, object]:
        return {"bridge_bias": sat_furnace._graph_bridge_bias(ctx["influence_matrix"])}

    def _run_loop_escape(ctx: FieldContext) -> Mapping[str, object]:
        return {
            "_loop_escape_bias": sat_furnace._loop_escape_bias(
                ctx["formula"], ctx["clause_frustrations"], int(ctx["variables"])
            )
        }

    def _run_memory(ctx: FieldContext) -> Mapping[str, object]:
        return {"memory_bias": sat_furnace._fiber_memory_bias(ctx["fiber_memory"], ctx["formula"])}

    return [
        FieldOperator(
            name="solver.bridge_bias",
            inputs=("influence_matrix",),
            outputs=("bridge_bias",),
            validate_inputs=require_keys(("influence_matrix",)),
            run=_run_bridge,
        ),
        FieldOperator(
            name="solver._loop_escape_bias",
            inputs=("formula", "clause_frustrations", "variables"),
            outputs=("_loop_escape_bias",),
            validate_inputs=require_keys(("formula", "clause_frustrations", "variables")),
            run=_run_loop_escape,
        ),
        FieldOperator(
            name="solver.memory_bias",
            inputs=("fiber_memory", "formula"),
            outputs=("memory_bias",),
            validate_inputs=require_keys(("fiber_memory", "formula")),
            run=_run_memory,
        ),
    ]


def _operator_effects_operator() -> FieldOperator:
    """Emit common-shape motion effect channels for policy mixing."""
    def _run(ctx: FieldContext) -> Mapping[str, object]:
        return {
            "operator_effects": {
                "pressure": list(ctx["pressures"]),
                "bridge": list(ctx["bridge_bias"]),
                "loop_escape": list(ctx["_loop_escape_bias"]),
                "memory": list(ctx["memory_bias"]),
            }
        }

    return FieldOperator(
        name="solver.operator_effects",
        inputs=("pressures", "bridge_bias", "_loop_escape_bias", "memory_bias"),
        outputs=("operator_effects",),
        validate_inputs=require_keys(("pressures", "bridge_bias", "_loop_escape_bias", "memory_bias")),
        run=_run,
    )


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


def _emit_sample_operator() -> FieldOperator:
    """Wraps sat_furnace._spatial_frame_rows to produce spatial emission rows for one epoch."""
    def _run(ctx: FieldContext) -> Mapping[str, object]:
        rows = sat_furnace._spatial_frame_rows(
            int(ctx["t"]),
            ctx["formula"],
            ctx["spins"],
            ctx["clause_frustrations"],
            ctx["influence_matrix"],
            ctx["pressures"],
            float(ctx["temperature"]),
        )
        return {"spatial_rows": rows}

    return FieldOperator(
        name="solver.emit_sample",
        inputs=("t", "formula", "spins", "clause_frustrations", "influence_matrix", "pressures", "temperature"),
        outputs=("spatial_rows",),
        validate_inputs=require_keys(("t", "formula", "spins", "clause_frustrations",
                                      "influence_matrix", "pressures", "temperature")),
        run=_run,
    )


def build_solver_composer() -> Composer:
    """Returns a Composer registered with all furnace step operators."""
    bias_ops = _bias_operators()
    return Composer(
        operators=(
            _formula_generate_operator(),
            _formula_validate_operator(),
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
            _emit_sample_operator(),
            # Phase B operators
            _thermo_metrics_operator(),
            _best_tracker_operator(),
            _lock_assignment_operator(),
            _cooling_operator(),
            _memory_scale_operator(),
            _fiber_update_operator(),
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

def _thermo_metrics_operator() -> FieldOperator:
    """B1 — heat, free_energy, integration, entropy from clause field."""
    def _run(ctx: FieldContext) -> Mapping[str, object]:
        spins = ctx["spins"]
        clause_frustrations = ctx["clause_frustrations"]
        influence_matrix = ctx["influence_matrix"]
        heat = (
            sum(f * f for f in clause_frustrations)
            / max(1, len(clause_frustrations))
        )
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

    return FieldOperator(
        name="solver.thermo_metrics",
        inputs=("spins", "clause_frustrations", "influence_matrix"),
        outputs=("heat", "free_energy", "integration", "entropy"),
        validate_inputs=require_keys(("spins", "clause_frustrations", "influence_matrix")),
        run=_run,
    )


def _best_tracker_operator() -> FieldOperator:
    """B2 — update best_spins / best_unsatisfied when current is better.

    Reads from ``prev_best_spins`` / ``prev_best_unsatisfied`` (carry-forward
    keys set by the epoch driver) so the Composer does not confuse the
    operator's *outputs* with its *inputs* and skip it entirely.
    """
    def _run(ctx: FieldContext) -> Mapping[str, object]:
        spins = ctx["spins"]
        unsatisfied = int(ctx["unsatisfied"])
        prev_best_unsatisfied = int(ctx["prev_best_unsatisfied"])
        prev_best_spins = ctx["prev_best_spins"]
        if unsatisfied < prev_best_unsatisfied:
            return {"best_spins": list(spins), "best_unsatisfied": unsatisfied}
        return {"best_spins": prev_best_spins, "best_unsatisfied": prev_best_unsatisfied}

    return FieldOperator(
        name="solver.best_tracker",
        inputs=("spins", "unsatisfied", "prev_best_spins", "prev_best_unsatisfied"),
        outputs=("best_spins", "best_unsatisfied"),
        validate_inputs=require_keys(("spins", "unsatisfied", "prev_best_spins", "prev_best_unsatisfied")),
        run=_run,
    )


def _lock_assignment_operator() -> FieldOperator:
    """B3 — produce lock_assignment (float ±1.0 list or None)."""
    def _run(ctx: FieldContext) -> Mapping[str, object]:
        unsatisfied = int(ctx["unsatisfied"])
        planted = ctx.get("planted_assignment")
        lock: list[float] | None = None
        if unsatisfied == 0 and planted is not None:
            lock = [1.0 if v else -1.0 for v in planted]
        return {"lock_assignment": lock}

    return FieldOperator(
        name="solver.lock_assignment",
        inputs=("unsatisfied",),
        outputs=("lock_assignment",),
        validate_inputs=require_keys(("unsatisfied",)),
        run=_run,
    )


def _cooling_operator() -> FieldOperator:
    """B4 — linear cooling schedule: 1 → 0 over the run."""
    def _run(ctx: FieldContext) -> Mapping[str, object]:
        t = int(ctx["t"])
        steps = int(ctx["steps"])
        return {"cooling": 1.0 - (t / max(1, steps - 1))}

    return FieldOperator(
        name="solver.cooling",
        inputs=("t", "steps"),
        outputs=("cooling",),
        validate_inputs=require_keys(("t", "steps")),
        run=_run,
    )


def _memory_scale_operator() -> FieldOperator:
    """B5 — compute memory drive scale from control action."""
    def _run(ctx: FieldContext) -> Mapping[str, object]:
        control = ctx["control_state"]
        samples = ctx["samples"]
        unsatisfied = int(ctx["unsatisfied"])
        best_unsatisfied = int(ctx["best_unsatisfied"])
        adaptive_gain = float(ctx.get("adaptive_gain", 0.0))
        scale = sat_furnace._action_memory_scale(
            control.action, samples, unsatisfied, best_unsatisfied, adaptive_gain
        )
        return {"memory_scale": scale}

    return FieldOperator(
        name="solver.memory_scale",
        inputs=("control_state", "samples", "unsatisfied", "best_unsatisfied"),
        outputs=("memory_scale",),
        validate_inputs=require_keys(("control_state", "samples", "unsatisfied", "best_unsatisfied")),
        run=_run,
    )


def _fiber_update_operator() -> FieldOperator:
    """B6 — update fiber bundle memory in-place and return the same object."""
    def _run(ctx: FieldContext) -> Mapping[str, object]:
        memory = ctx["fiber_memory"]
        sat_furnace._update_fiber_bundle_memory(
            memory,
            ctx["formula"],
            ctx["spins"],
            ctx["pressures"],
            ctx["clause_frustrations"],
            ctx["influence_matrix"],
        )
        return {"fiber_memory": memory}

    return FieldOperator(
        name="solver.fiber_update",
        inputs=("fiber_memory", "formula", "spins", "pressures", "clause_frustrations", "influence_matrix"),
        outputs=("fiber_memory",),
        validate_inputs=require_keys(("fiber_memory", "formula", "spins", "pressures",
                                      "clause_frustrations", "influence_matrix")),
        run=_run,
    )


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
        excitable_active = policy in {EXCITABLE_POLICY, CURRICULUM_SEED_POLICY}
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
        row.update(_bc.coupling_metrics(
            report.windows, spatial_samples, runners,
            window_size=window, step_size=step_size,
        ))
        row.update(_bc.choice_policy_metrics(report.windows, runners))
        row.update(_bc.operator_trace_metrics(result.operator_traces))
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


# ---------------------------------------------------------------------------
# Explain operators (Phase #4)
# ---------------------------------------------------------------------------
# Usage:
#   ctx = {
#       "composer": sat_composer.build_solver_composer(),
#       "targets": ["furnace_result"],
#       "available_keys": ("formula", "spins", "t"),   # optional
#   }
#   out = build_explain_composer().run(["explain_report"], ctx)
#   print(out["explain_report"])
# ---------------------------------------------------------------------------

def _explain_plan_operator() -> FieldOperator:
    """Runs Composer.plan(); outputs ordered operator names and missing inputs."""
    from composer import CompositionPlan  # noqa: F401 (type hint only)

    def _run(ctx: FieldContext) -> dict[str, object]:
        composer: Composer = ctx["composer"]  # type: ignore[assignment]
        targets: list[str] = list(ctx["targets"])  # type: ignore[arg-type]
        available: tuple[str, ...] = tuple(ctx.get("available_keys", ()))  # type: ignore[arg-type]
        plan = composer.plan(targets, available)
        return {"plan_order": plan.order, "plan_missing": plan.missing}

    return FieldOperator(
        name="explain.plan",
        inputs=("composer", "targets"),
        outputs=("plan_order", "plan_missing"),
        run=_run,
    )


def _explain_edges_operator() -> FieldOperator:
    """Runs Composer.graph(); outputs dependency edges and operator list."""
    def _run(ctx: FieldContext) -> dict[str, object]:
        composer: Composer = ctx["composer"]  # type: ignore[assignment]
        targets: list[str] = list(ctx["targets"])  # type: ignore[arg-type]
        available: tuple[str, ...] = tuple(ctx.get("available_keys", ()))  # type: ignore[arg-type]
        dep = composer.graph(targets, available)
        return {"dep_edges": dep.edges, "dep_operators": dep.operators}

    return FieldOperator(
        name="explain.edges",
        inputs=("composer", "targets"),
        outputs=("dep_edges", "dep_operators"),
        run=_run,
    )


def _explain_operator_detail_operator() -> FieldOperator:
    """Emits per-operator detail dicts (inputs, outputs) for the planned order."""
    def _run(ctx: FieldContext) -> dict[str, object]:
        composer: Composer = ctx["composer"]  # type: ignore[assignment]
        plan_order: tuple[str, ...] = ctx["plan_order"]  # type: ignore[assignment]
        details = []
        for name in plan_order:
            op = composer._operators.get(name)
            if op is None:
                continue
            details.append({
                "name": op.name,
                "inputs": list(op.inputs),
                "outputs": list(op.outputs),
            })
        return {"operator_details": details}

    return FieldOperator(
        name="explain.operator_detail",
        inputs=("composer", "plan_order"),
        outputs=("operator_details",),
        run=_run,
    )


def _explain_format_operator() -> FieldOperator:
    """Formats plan_order, edges, and operator details into a readable report."""
    def _run(ctx: FieldContext) -> dict[str, object]:
        from composer import MissingInput  # noqa: F401 (isinstance check)
        targets: list[str] = list(ctx["targets"])  # type: ignore[arg-type]
        plan_order: tuple[str, ...] = ctx["plan_order"]  # type: ignore[assignment]
        plan_missing: tuple[object, ...] = ctx["plan_missing"]  # type: ignore[assignment]
        dep_edges: tuple[tuple[str, str], ...] = ctx["dep_edges"]  # type: ignore[assignment]
        operator_details: list[dict[str, object]] = ctx["operator_details"]  # type: ignore[assignment]

        lines: list[str] = []
        lines.append(f"targets: {', '.join(targets)}")
        lines.append("")

        if plan_missing:
            lines.append("missing inputs:")
            for item in plan_missing:
                lines.append(f"  {item.key}  (required by {', '.join(item.required_by)})")  # type: ignore[attr-defined]
            lines.append("")

        lines.append(f"execution order ({len(plan_order)} operators):")
        for i, detail in enumerate(operator_details, 1):
            ins = ", ".join(detail["inputs"])  # type: ignore[arg-type]
            outs = ", ".join(detail["outputs"])  # type: ignore[arg-type]
            lines.append(f"  {i:2d}. {detail['name']}")
            lines.append(f"       in:  {ins or '—'}")
            lines.append(f"       out: {outs or '—'}")

        if dep_edges:
            lines.append("")
            lines.append("dependencies:")
            for src, dst in dep_edges:
                lines.append(f"  {src} → {dst}")

        return {"explain_report": "\n".join(lines)}

    return FieldOperator(
        name="explain.format",
        inputs=("targets", "plan_order", "plan_missing", "dep_edges", "operator_details"),
        outputs=("explain_report",),
        run=_run,
    )


def build_explain_composer() -> Composer:
    """Returns a Composer that explains another Composer's execution plan.

    Pass a ``composer`` (any Composer instance), ``targets`` (list of str),
    and optionally ``available_keys`` (iterable of str already in context)
    to the returned composer's run call::

        ctx = {
            "composer": build_solver_composer(),
            "targets": ["furnace_result"],
            "available_keys": ("formula", "variables", "steps"),
        }
        out = build_explain_composer().run(["explain_report"], ctx)
        print(out["explain_report"])
    """
    return Composer(
        operators=(
            _explain_plan_operator(),
            _explain_edges_operator(),
            _explain_operator_detail_operator(),
            _explain_format_operator(),
        )
    )
