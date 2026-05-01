#!/usr/bin/env python3
"""SAT furnace: generate H/F/I trajectories from CNF soft-assignment dynamics."""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import sat_field


Literal = tuple[int, bool]
Clause = tuple[Literal, ...]
CNF = list[Clause]
EPSILON = 1e-12


@dataclass(frozen=True)
class FurnaceSample:
    t: int
    heat: float
    free_energy: float
    integration: float
    unsatisfied_clauses: int
    _assignment_entropy: float


@dataclass(frozen=True)
class FurnaceResult:
    formula: CNF
    planted_assignment: list[bool] | None
    samples: list[FurnaceSample]
    spatial_samples: list[dict[str, float | int | str]]
    operator_traces: list["OperatorTrace"]
    final_assignment: list[bool]
    solved: bool


@dataclass(frozen=True)
class OperatorTrace:
    t: int
    operator: str
    active: bool
    action: str
    reason: str
    input_heat: float
    input_entropy: float
    input_integration: float
    input_unsatisfied: int
    output_mean: float
    output_peak: float
    memory_scale: float
    delta_unsatisfied: int
    delta_integration: float


@dataclass(frozen=True)
class ControlState:
    action: str
    learning_rate_scale: float
    inertia_scale: float
    noise_scale: float
    bridge_scale: float
    loop_escape_scale: float


@dataclass
class MemoryFiber:
    heat: float = 0.0
    pressure: float = 0.0
    influence: float = 0.0
    alignment: float = 0.0
    visits: float = 0.0


@dataclass
class FiberBundleMemory:
    variable_fibers: list[MemoryFiber]
    clause_fibers: list[MemoryFiber]
    decay: float


def generate_formula(kind: str, variables: int, clauses: int, clause_size: int, rng: random.Random) -> tuple[CNF, list[bool] | None]:
    if kind in {"sat", "hard_sat"}:
        assignment = [rng.choice([False, True]) for _ in range(variables)]
        return generate_planted_sat(assignment, clauses, clause_size, rng), assignment
    if kind == "unsat":
        return generate_structural_unsat(variables, clauses, clause_size, rng), None
    if kind == "random":
        return generate_random_cnf(variables, clauses, clause_size, rng), None
    raise ValueError(f"unknown formula kind: {kind}")


def generate_planted_sat(assignment: Sequence[bool], clauses: int, clause_size: int, rng: random.Random) -> CNF:
    formula: CNF = []
    variables = len(assignment)
    while len(formula) < clauses:
        indices = rng.sample(range(variables), min(clause_size, variables))
        clause = tuple((index, rng.choice([False, True])) for index in indices)
        if clause_satisfied(clause, assignment):
            formula.append(clause)
    return formula


def generate_random_cnf(variables: int, clauses: int, clause_size: int, rng: random.Random) -> CNF:
    return [
        tuple((index, rng.choice([False, True])) for index in rng.sample(range(variables), min(clause_size, variables)))
        for _ in range(clauses)
    ]


def generate_structural_unsat(variables: int, clauses: int, clause_size: int, rng: random.Random) -> CNF:
    formula: CNF = []
    core_variables = list(range(min(max(3, clause_size), variables)))
    for mask in range(2 ** len(core_variables)):
        clause: list[Literal] = []
        for bit, variable in enumerate(core_variables):
            clause.append((variable, bool((mask >> bit) & 1)))
        formula.append(tuple(clause))

    while len(formula) < clauses:
        formula.append(tuple((index, rng.choice([False, True])) for index in rng.sample(range(variables), min(clause_size, variables))))
    return formula[:clauses]


_EPOCH_TARGETS = (
    "pressures", "clause_frustrations", "unsatisfied", "influence_matrix",
    "heat", "free_energy", "integration", "entropy",
    "best_spins", "best_unsatisfied",
    "lock_assignment", "cooling",
    "adaptive_active", "adaptive_reason", "adaptive_gain", "control_state",
    "fiber_memory", "memory_scale",
    "bridge_bias", "_loop_escape_bias", "memory_bias",
    "operator_effects", "concentrations", "excitatory_field", "inhibitory_field",
    "local_field", "field_strength", "spike_strength", "mixed_drive",
    "next_spins", "next_velocity", "drive_values",
    "samples", "spatial_samples", "operator_traces",
)

_STALE_EPOCH_KEYS = frozenset({
    "pressures", "clause_frustrations", "unsatisfied", "influence_matrix",
    "heat", "free_energy", "integration", "entropy",
    "best_spins", "best_unsatisfied",
    "samples", "spatial_samples", "operator_traces",
    "lock_assignment", "cooling",
    "adaptive_active", "adaptive_reason", "adaptive_gain", "control_state",
    "memory_scale", "bridge_bias", "_loop_escape_bias", "memory_bias",
    "operator_effects", "concentrations", "excitatory_field", "inhibitory_field",
    "local_field", "field_strength", "spike_strength", "mixed_drive",
    "next_spins", "next_velocity", "drive_values",
})


def _init_epoch_context(
    *,
    formula: CNF,
    variables: int,
    steps: int,
    rng: random.Random,
    temperature: float,
    learning_rate: float,
    inertia: float,
    noise: float,
    planted_assignment: list[bool] | None,
    adaptive: bool,
    memory_decay: float,
    memory_drive: float,
    policy: str,
    spike_threshold: float,
    spike_slope: float,
) -> dict[str, object]:
    spins: list[float] = [rng.uniform(-0.25, 0.25) for _ in range(variables)]
    return {
        "formula": formula,
        "variables": variables,
        "steps": steps,
        "temperature": temperature,
        "learning_rate": learning_rate,
        "inertia": inertia,
        "noise": noise,
        "adaptive": adaptive,
        "policy": policy,
        "spike_threshold": spike_threshold,
        "spike_slope": spike_slope,
        "memory_decay": memory_decay,
        "memory_drive": memory_drive,
        "planted_assignment": planted_assignment,
        "rng": rng,
        "spins": spins,
        "velocity": [0.0] * variables,
        "fiber_memory": _initialize_fiber_bundle_memory(variables, len(formula), memory_decay),
        "adaptive_active": False,
        "adaptive_reason": "inactive_disabled",
        "adaptive_gain": 0.0,
        "control_state": _default_control_state(),
        "prev_samples": [],
        "prev_spatial_samples": [],
        "prev_operator_traces": [],
        "prev_best_spins": list(spins),
        "prev_best_unsatisfied": len(formula),
        "prev_concentrations": [0.0, 0.0, 0.0, 0.0],
        "previous_unsatisfied": len(formula),
        "previous_integration": 0.0,
    }


def run_furnace(
    formula: CNF,
    variables: int,
    steps: int,
    rng: random.Random,
    temperature: float,
    learning_rate: float,
    inertia: float,
    noise: float,
    planted_assignment: list[bool] | None = None,
    adaptive: bool = False,
    memory_decay: float = 0.92,
    memory_drive: float = 0.12,
    policy: str = "baseline",
    spike_threshold: float = 0.35,
    spike_slope: float = 8.0,
) -> FurnaceResult:
    import sat_composer  # local import avoids module-level circular dependency

    composer = sat_composer.build_solver_composer()
    ctx = _init_epoch_context(
        formula=formula,
        variables=variables,
        steps=steps,
        rng=rng,
        temperature=temperature,
        learning_rate=learning_rate,
        inertia=inertia,
        noise=noise,
        planted_assignment=planted_assignment,
        adaptive=adaptive,
        memory_decay=memory_decay,
        memory_drive=memory_drive,
        policy=policy,
        spike_threshold=spike_threshold,
        spike_slope=spike_slope,
    )

    for t in range(steps):
        ctx["t"] = t
        # Carry-forward prev_* keys consumed by accumulator operators.
        ctx["prev_samples"] = ctx.get("samples", [])
        ctx["prev_spatial_samples"] = ctx.get("spatial_samples", [])
        ctx["prev_operator_traces"] = ctx.get("operator_traces", [])
        ctx["prev_best_spins"] = ctx.get("best_spins", ctx["prev_best_spins"])
        ctx["prev_best_unsatisfied"] = ctx.get("best_unsatisfied", ctx["prev_best_unsatisfied"])
        ctx["prev_concentrations"] = ctx.get("concentrations", ctx["prev_concentrations"])
        ctx["previous_unsatisfied"] = (
            ctx["prev_samples"][-1].unsatisfied_clauses  # type: ignore[index]
            if ctx["prev_samples"]
            else ctx["prev_best_unsatisfied"]
        )
        ctx["previous_integration"] = (
            ctx["prev_samples"][-1].integration  # type: ignore[index]
            if ctx["prev_samples"]
            else 0.0
        )
        for key in _STALE_EPOCH_KEYS:
            ctx.pop(key, None)
        out = composer.run(_EPOCH_TARGETS, ctx)
        ctx.update(out)
        ctx["spins"] = out["next_spins"]
        ctx["velocity"] = out["next_velocity"]

    out = composer.run(("final_assignment", "solved", "furnace_result"), ctx)
    return out["furnace_result"]  # type: ignore[return-value]


def _spin_update_step(
    spins: list[float],
    velocity: list[float],
    pressures: Sequence[float],
    bridge_bias: Sequence[float],
    loop_escape: Sequence[float],
    memory_bias: Sequence[float],
    control: ControlState,
    learning_rate: float,
    inertia: float,
    noise: float,
    memory_drive: float,
    adaptive_gain: float,
    memory_scale: float,
    adaptive_active: bool,
    rng: random.Random,
    mixed_drive: Sequence[float] | None = None,
    mixed_scale: float = 0.0,
    cooling: float = 1.0,
    lock_assignment: Sequence[float] | None = None,
) -> tuple[list[float], list[float], list[float]]:
    effective_learning_rate = learning_rate * control.learning_rate_scale
    effective_inertia = _clamp(inertia * control.inertia_scale, 0.0, 0.985)
    effective_noise = noise * control.noise_scale
    drive_values: list[float] = []
    for index in range(len(spins)):
        drive = effective_learning_rate * pressures[index]
        drive += effective_learning_rate * control.bridge_scale * bridge_bias[index]
        drive += effective_learning_rate * control.loop_escape_scale * loop_escape[index]
        if adaptive_active and memory_scale > 0.0:
            drive += effective_learning_rate * memory_drive * adaptive_gain * memory_scale * memory_bias[index]
        if mixed_drive is not None and mixed_scale > 0.0:
            drive += effective_learning_rate * mixed_scale * mixed_drive[index]
        drive_values.append(drive)
        if lock_assignment is not None:
            drive += 0.025 * cooling * lock_assignment[index]
        jitter = rng.gauss(0.0, effective_noise * cooling)
        velocity[index] = effective_inertia * velocity[index] + drive + jitter
        spins[index] = math.tanh(spins[index] + velocity[index])
    return spins, velocity, drive_values


def _default_control_state() -> ControlState:
    return ControlState(
        action="baseline",
        learning_rate_scale=1.0,
        inertia_scale=1.0,
        noise_scale=1.0,
        bridge_scale=0.0,
        loop_escape_scale=0.0,
    )


def _initialize_fiber_bundle_memory(variables: int, clauses: int, decay: float) -> FiberBundleMemory:
    return FiberBundleMemory(
        variable_fibers=[MemoryFiber() for _ in range(variables)],
        clause_fibers=[MemoryFiber() for _ in range(clauses)],
        decay=_clamp(decay, 0.0, 0.999),
    )


def _update_fiber_bundle_memory(
    memory: FiberBundleMemory,
    formula: CNF,
    spins: Sequence[float],
    pressures: Sequence[float],
    clause_frustrations: Sequence[float],
    influence: Sequence[Sequence[float]],
) -> None:
    variable_heat = [0.0 for _ in spins]
    variable_counts = [0 for _ in spins]
    for clause, frustration in zip(formula, clause_frustrations):
        heat = frustration * frustration
        for variable, _ in clause:
            variable_heat[variable] += heat
            variable_counts[variable] += 1

    for variable, fiber in enumerate(memory.variable_fibers):
        heat = variable_heat[variable] / max(1, variable_counts[variable])
        fiber.heat = _exponential_decay(fiber.heat, heat, memory.decay)
        fiber.pressure = _exponential_decay(fiber.pressure, abs(pressures[variable]), memory.decay)
        fiber.influence = _exponential_decay(fiber.influence, sum(influence[variable]), memory.decay)
        fiber.alignment = _exponential_decay(fiber.alignment, spins[variable], memory.decay)
        fiber.visits = _exponential_decay(fiber.visits, 1.0 if heat > 0.02 or abs(pressures[variable]) > 0.05 else 0.0, memory.decay)

    for clause_id, (fiber, frustration) in enumerate(zip(memory.clause_fibers, clause_frustrations)):
        fiber.heat = _exponential_decay(fiber.heat, frustration * frustration, memory.decay)
        fiber.pressure = _exponential_decay(fiber.pressure, frustration, memory.decay)
        fiber.influence = _exponential_decay(fiber.influence, _clause_memory_influence(formula[clause_id], influence), memory.decay)
        fiber.alignment = _exponential_decay(fiber.alignment, _clause_memory_alignment(formula[clause_id], spins), memory.decay)
        fiber.visits = _exponential_decay(fiber.visits, 1.0 if frustration > 0.12 else 0.0, memory.decay)


def _exponential_decay(previous: float, current: float, decay: float) -> float:
    return decay * previous + (1.0 - decay) * current


def _clause_memory_influence(clause: Clause, influence: Sequence[Sequence[float]]) -> float:
    variables = [variable for variable, _ in clause]
    total = 0.0
    for left in variables:
        for right in variables:
            if left != right:
                total += influence[left][right]
    return total / max(1, len(variables) * max(1, len(variables) - 1))


def _clause_memory_alignment(clause: Clause, spins: Sequence[float]) -> float:
    return sum((-spins[variable] if is_negated else spins[variable]) for variable, is_negated in clause) / max(1, len(clause))


def _fiber_memory_bias(memory: FiberBundleMemory, formula: CNF) -> list[float]:
    bias = [0.0 for _ in memory.variable_fibers]
    for clause, clause_fiber in zip(formula, memory.clause_fibers):
        trapped_mass = clause_fiber.heat * clause_fiber.visits * (1.0 - clamp01(abs(clause_fiber.alignment)))
        bridge_need = trapped_mass * (1.0 + clause_fiber.pressure) / max(1, len(clause))
        for variable, is_negated in clause:
            sign = -1.0 if is_negated else 1.0
            variable_fiber = memory.variable_fibers[variable]
            variable_trap = variable_fiber.heat * variable_fiber.visits
            escape_pressure = bridge_need + variable_trap * (1.0 - clamp01(abs(variable_fiber.alignment)))
            bias[variable] += sign * escape_pressure
    return _normalize_vector(bias)


def _adaptive_activation_state(samples: Sequence[FurnaceSample], unsatisfied: int, best_unsatisfied: int) -> tuple[bool, str]:
    if unsatisfied <= 1:
        return False, "inactive_near_solved"
    if len(samples) < 28:
        return False, "inactive_warmup"
    recent = samples[-24:]
    recent_best = min(sample.unsatisfied_clauses for sample in recent)
    recent_worst = max(sample.unsatisfied_clauses for sample in recent)
    recent_start = recent[0].unsatisfied_clauses
    recent_end = recent[-1].unsatisfied_clauses
    recently_improved = recent_end < recent_start or (unsatisfied < recent_worst and unsatisfied <= best_unsatisfied + 1)
    stalled_at_best = recent_best <= best_unsatisfied and not recently_improved and recent_end >= recent_best
    oscillating_without_gain = recent_worst - recent_best >= 3 and recent_best >= best_unsatisfied and not recently_improved
    if stalled_at_best:
        return True, "active_stalled_at_best"
    if oscillating_without_gain:
        return True, "active_oscillating_without_gain"
    return False, "inactive_progressing"


def _adaptive_should_activate(samples: Sequence[FurnaceSample], unsatisfied: int, best_unsatisfied: int) -> bool:
    active, _ = _adaptive_activation_state(samples, unsatisfied, best_unsatisfied)
    return active


def _trace_operator(
    traces: list[OperatorTrace],
    t: int,
    operator: str,
    active: bool,
    action: str,
    reason: str,
    heat: float,
    entropy: float,
    integration: float,
    unsatisfied: int,
    outputs: Sequence[float],
    memory_scale: float,
    delta_unsatisfied: int,
    delta_integration: float,
) -> None:
    traces.append(
        OperatorTrace(
            t=t,
            operator=operator,
            active=active,
            action=action,
            reason=reason,
            input_heat=heat,
            input_entropy=entropy,
            input_integration=integration,
            input_unsatisfied=unsatisfied,
            output_mean=_mean_abs(outputs),
            output_peak=_max_abs(outputs),
            memory_scale=memory_scale,
            delta_unsatisfied=delta_unsatisfied,
            delta_integration=delta_integration,
        )
    )


def _mean_abs(values: Sequence[float]) -> float:
    return sum(abs(value) for value in values) / max(1, len(values))


def _max_abs(values: Sequence[float]) -> float:
    return max((abs(value) for value in values), default=0.0)


def _action_memory_scale(action: str, samples: Sequence[FurnaceSample], unsatisfied: int, best_unsatisfied: int, adaptive_gain: float = 1.0) -> float:
    if action == "stabilize" or unsatisfied == 0:
        return 0.0
    recently_improved = bool(samples) and unsatisfied < min(sample.unsatisfied_clauses for sample in samples[-min(16, len(samples)) :])
    stalled = len(samples) >= 20 and min(sample.unsatisfied_clauses for sample in samples[-20:]) <= best_unsatisfied and not recently_improved
    gain = clamp01(adaptive_gain)
    if action == "perturb":
        return (0.65 if stalled else 0.10) * gain
    if action == "bridge":
        return 0.70 * (0.45 + 0.55 * gain)
    if action == "explore":
        return (0.35 if stalled else 0.08) * gain
    if action == "cool":
        return 0.0
    return 0.0


def _adaptive_strength(
    samples: Sequence[FurnaceSample],
    heat: float,
    free_energy: float,
    integration: float,
    entropy: float,
    unsatisfied: int,
    best_unsatisfied: int,
) -> float:
    if unsatisfied <= 1 or len(samples) < 8:
        return 0.0
    recent = samples[-min(24, len(samples)) :]
    older = samples[-min(48, len(samples)) : -min(24, len(samples))] if len(samples) >= 32 else []
    recent_best = min(sample.unsatisfied_clauses for sample in recent)
    recent_worst = max(sample.unsatisfied_clauses for sample in recent)
    recent_start = recent[0].unsatisfied_clauses
    recent_end = recent[-1].unsatisfied_clauses
    progress = max(0.0, float(recent_start - recent_end)) / max(1.0, float(recent_start))
    oscillation = clamp01(float(recent_worst - recent_best) / max(1.0, float(max(recent_worst, unsatisfied))))
    stagnation = 1.0 if recent_best <= best_unsatisfied and progress <= 0.05 else 0.0
    entropy_excess = clamp01(entropy)
    heat_excess = clamp01(heat + free_energy)
    integration_deficit = clamp01(1.0 - integration)
    escape_need = _mean_values([stagnation, oscillation, integration_deficit])
    collapse_readiness = _mean_values([max(0.0, 1.0 - entropy_excess), max(0.0, 1.0 - heat_excess), integration])
    if older:
        previous_heat = _mean_values([sample.heat for sample in older])
        current_heat = _mean_values([sample.heat for sample in recent])
        heat_release = clamp01((previous_heat - current_heat) / max(EPSILON, previous_heat)) if previous_heat > EPSILON else 0.0
    else:
        heat_release = 0.0
    benefit = _mean_values([escape_need, collapse_readiness, heat_release])
    risk = _mean_values([entropy_excess, heat_excess, oscillation]) * (0.55 + 0.45 * (1.0 - collapse_readiness))
    return clamp01(0.25 + 0.85 * benefit - 0.45 * risk)


def _adaptive_control_state(
    samples: Sequence[FurnaceSample],
    heat: float,
    free_energy: float,
    integration: float,
    entropy: float,
    unsatisfied: int,
    best_unsatisfied: int,
    adaptive_gain: float = 1.0,
) -> ControlState:
    previous_best = min((sample.unsatisfied_clauses for sample in samples), default=len(samples) + unsatisfied)
    current_improved = unsatisfied < previous_best
    recent_window = samples[-min(16, len(samples)) :]
    recent_best = min((sample.unsatisfied_clauses for sample in recent_window), default=unsatisfied)
    recent_worst = max((sample.unsatisfied_clauses for sample in recent_window), default=unsatisfied)
    improving = current_improved or (bool(recent_window) and unsatisfied < recent_worst and unsatisfied <= previous_best + 1)
    near_solved = unsatisfied <= 1
    stalled = len(samples) >= 16 and not improving and recent_best <= best_unsatisfied
    entropy_excess = clamp01(entropy)
    heat_excess = clamp01(heat + free_energy)
    integration_deficit = clamp01(1.0 - integration)
    stagnation_excess = 1.0 if stalled and unsatisfied > 0 else 0.0
    progress_credit = 1.0 if improving or near_solved or unsatisfied == 0 else 0.0
    explore_score = _mean_values([entropy_excess, integration_deficit, stagnation_excess]) * (0.35 + 0.65 * stagnation_excess) * (1.0 - 0.75 * progress_credit)
    bridge_score = _mean_values([integration_deficit, heat_excess, clamp01(unsatisfied / 8.0)]) * (0.65 + 0.35 * (1.0 - progress_credit))
    cool_score = _mean_values([heat_excess, entropy_excess, 1.0 - stagnation_excess, progress_credit])
    perturb_score = _mean_values([stagnation_excess, heat_excess, entropy_excess]) * stagnation_excess
    stabilize_score = _mean_values([integration, 1.0 - entropy_excess, 1.0 - heat_excess, progress_credit]) + (0.35 if near_solved else 0.0)
    actions = {
        "explore": explore_score,
        "bridge": bridge_score,
        "cool": cool_score,
        "perturb": perturb_score,
        "stabilize": stabilize_score,
    }
    action = max(actions, key=actions.get)
    gain = clamp01(adaptive_gain)
    if improving and action in {"explore", "perturb"}:
        action = "stabilize" if near_solved else "cool"
    if near_solved and action != "stabilize":
        action = "stabilize"
    if action == "explore":
        return ControlState(action, learning_rate_scale=1.0 - 0.06 * gain, inertia_scale=1.0 - 0.04 * gain, noise_scale=1.0 + 0.08 * gain, bridge_scale=0.14 * gain, loop_escape_scale=0.08 * gain)
    if action == "bridge":
        return ControlState(action, learning_rate_scale=1.0 + 0.08 * gain, inertia_scale=1.0 + 0.01 * gain, noise_scale=1.0 - 0.18 * gain, bridge_scale=0.45 * gain, loop_escape_scale=0.02 * gain)
    if action == "perturb":
        return ControlState(action, learning_rate_scale=1.0 - 0.28 * gain, inertia_scale=1.0 - 0.28 * gain, noise_scale=1.0 + 0.35 * gain, bridge_scale=0.04 * gain, loop_escape_scale=0.28 * gain)
    if action == "stabilize":
        return ControlState(action, learning_rate_scale=1.0 - 0.07 * gain, inertia_scale=1.0 + 0.06 * gain, noise_scale=1.0 - 0.68 * gain, bridge_scale=0.20 * gain, loop_escape_scale=0.0)
    return ControlState(action, learning_rate_scale=1.0 - 0.22 * gain, inertia_scale=1.0 + 0.03 * gain, noise_scale=1.0 - 0.52 * gain, bridge_scale=0.12 * gain, loop_escape_scale=0.0)


def _graph_bridge_bias(influence: Sequence[Sequence[float]]) -> list[float]:
    strengths = [sum(row) for row in influence]
    if not strengths:
        return []
    center = sum(strengths) / len(strengths)
    return _normalize_vector([center - strength for strength in strengths])


def _loop_escape_bias(formula: CNF, clause_frustrations: Sequence[float], variables: int) -> list[float]:
    bias = [0.0 for _ in range(variables)]
    if not clause_frustrations:
        return bias
    threshold = sum(clause_frustrations) / len(clause_frustrations)
    for clause, frustration in zip(formula, clause_frustrations):
        if frustration < threshold:
            continue
        for variable, is_negated in clause:
            bias[variable] += -1.0 if is_negated else 1.0
    return _normalize_vector(bias)


def _normalize_vector(values: Sequence[float]) -> list[float]:
    scale = max([abs(value) for value in values], default=0.0)
    if scale <= EPSILON:
        return [0.0 for _ in values]
    return [value / scale for value in values]


def _mean_values(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _clause_pressures(formula: CNF, spins: Sequence[float], temperature: float) -> tuple[list[float], list[float], int]:
    pressures = [0.0 for _ in spins]
    frustrations: list[float] = []
    unsatisfied = 0

    for clause in formula:
        literal_values = [_literal_value(spins[variable], is_negated, temperature) for variable, is_negated in clause]
        unsatisfied_products = [1.0 - value for value in literal_values]
        frustration = _product(unsatisfied_products)
        frustrations.append(frustration)
        if all(_discrete_literal_value(spins[variable] >= 0, is_negated) is False for variable, is_negated in clause):
            unsatisfied += 1

        for literal_index, (variable, is_negated) in enumerate(clause):
            sign = -1.0 if is_negated else 1.0
            value = literal_values[literal_index]
            other_unsatisfied = _product(
                unsatisfied_products[index]
                for index in range(len(unsatisfied_products))
                if index != literal_index
            )
            sensitivity = value * (1.0 - value) / max(EPSILON, temperature)
            pressures[variable] += sign * other_unsatisfied * sensitivity * (0.25 + frustration)

    scale = max(1.0, max(abs(value) for value in pressures))
    return [value / scale for value in pressures], frustrations, unsatisfied


def _variable_influence_matrix(formula: CNF, clause_frustrations: Sequence[float], variables: int) -> list[list[float]]:
    matrix = [[0.0 for _ in range(variables)] for _ in range(variables)]
    for clause, frustration in zip(formula, clause_frustrations):
        weight = frustration * frustration
        clause_variables = [variable for variable, _ in clause]
        for i, left in enumerate(clause_variables):
            for right in clause_variables[i + 1 :]:
                matrix[left][right] += weight
                matrix[right][left] += weight
    return matrix


def _spatial_frame_rows(
    t: int,
    formula: CNF,
    spins: Sequence[float],
    clause_frustrations: Sequence[float],
    influence: Sequence[Sequence[float]],
    pressures: Sequence[float],
    temperature: float,
) -> list[dict[str, float | int | str]]:
    variable_heat = [0.0 for _ in spins]
    variable_clause_count = [0 for _ in spins]
    for clause, frustration in zip(formula, clause_frustrations):
        for variable, _ in clause:
            variable_heat[variable] += frustration * frustration
            variable_clause_count[variable] += 1

    rows: list[dict[str, float | int | str]] = []
    for variable, spin in enumerate(spins):
        local_probability = _literal_value(spin, False, temperature)
        local_entropy = _binary_entropy(local_probability)
        rows.append(
            {
                "t": t,
                "kind": "variable",
                "id": variable,
                "heat": variable_heat[variable] / max(1, variable_clause_count[variable]),
                "influence": sum(influence[variable]),
                "entropy": local_entropy,
                "spin": spin,
                "pressure": pressures[variable],
            }
        )
    for clause_id, (clause, frustration) in enumerate(zip(formula, clause_frustrations)):
        clause_spin = sum((-spins[variable] if is_negated else spins[variable]) for variable, is_negated in clause) / max(1, len(clause))
        rows.append(
            {
                "t": t,
                "kind": "clause",
                "id": clause_id,
                "heat": frustration * frustration,
                "influence": sum(sum(influence[variable][other] for other, _ in clause) for variable, _ in clause),
                "entropy": _binary_entropy(1.0 - frustration),
                "spin": clause_spin,
                "pressure": frustration,
            }
        )
    return rows


def _binary_entropy(probability: float) -> float:
    probability = clamp01(probability)
    if probability <= EPSILON or probability >= 1.0 - EPSILON:
        return 0.0
    return -(probability * math.log(probability, 2) + (1.0 - probability) * math.log(1.0 - probability, 2))


def _integration_score(matrix: Sequence[Sequence[float]]) -> float:
    n = len(matrix)
    if n <= 1:
        return 1.0
    strengths = [sum(row) for row in matrix]
    total = sum(strengths)
    if total <= EPSILON:
        return 0.0
    normalized = [value / total for value in strengths]
    participation = 1.0 / max(sum(value * value for value in normalized), EPSILON)
    evenness = (participation - 1.0) / max(1.0, n - 1.0)
    connected = sum(1 for value in strengths if value > total / (n * n)) / n
    return clamp01(0.55 * evenness + 0.45 * connected)


def _literal_value(spin: float, is_negated: bool, temperature: float) -> float:
    value = 1.0 / (1.0 + math.exp(-spin / max(EPSILON, temperature)))
    return 1.0 - value if is_negated else value


def _discrete_literal_value(value: bool, is_negated: bool) -> bool:
    return not value if is_negated else value


def clause_satisfied(clause: Clause, assignment: Sequence[bool]) -> bool:
    return any(_discrete_literal_value(assignment[variable], is_negated) for variable, is_negated in clause)


def _assignment_entropy(spins: Sequence[float]) -> float:
    entropies = []
    for spin in spins:
        probability = clamp01((spin + 1.0) / 2.0)
        if probability <= EPSILON or probability >= 1.0 - EPSILON:
            entropies.append(0.0)
        else:
            entropies.append(-(probability * math.log(probability, 2) + (1.0 - probability) * math.log(1.0 - probability, 2)))
    return sum(entropies) / max(1, len(entropies))


def _statistics_like_variance(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _product(values) -> float:
    result = 1.0
    for value in values:
        result *= value
    return result


def clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def write_samples(path: Path, samples: Sequence[FurnaceSample]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["t", "H", "F", "I", "unsatisfied_clauses", "_assignment_entropy"])
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "t": sample.t,
                    "H": sample.heat,
                    "F": sample.free_energy,
                    "I": sample.integration,
                    "unsatisfied_clauses": sample.unsatisfied_clauses,
                    "_assignment_entropy": sample._assignment_entropy,
                }
            )


def write_spatial_samples(path: Path, spatial_samples: Sequence[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["t", "kind", "id", "heat", "influence", "entropy", "spin", "pressure"])
        writer.writeheader()
        writer.writerows(spatial_samples)


def write_graph(path: Path, formula: CNF) -> None:
    sat_field.write_formula_graph(path, sat_field.formula_graph(formula))


def write_operator_traces(path: Path, traces: Sequence[OperatorTrace]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "t",
                "operator",
                "active",
                "action",
                "reason",
                "input_heat",
                "input_entropy",
                "input_integration",
                "input_unsatisfied",
                "output_mean",
                "output_peak",
                "memory_scale",
                "delta_unsatisfied",
                "delta_integration",
            ],
        )
        writer.writeheader()
        for trace in traces:
            writer.writerow(
                {
                    "t": trace.t,
                    "operator": trace.operator,
                    "active": trace.active,
                    "action": trace.action,
                    "reason": trace.reason,
                    "input_heat": trace.input_heat,
                    "input_entropy": trace.input_entropy,
                    "input_integration": trace.input_integration,
                    "input_unsatisfied": trace.input_unsatisfied,
                    "output_mean": trace.output_mean,
                    "output_peak": trace.output_peak,
                    "memory_scale": trace.memory_scale,
                    "delta_unsatisfied": trace.delta_unsatisfied,
                    "delta_integration": trace.delta_integration,
                }
            )


def default_output_path(kind: str) -> Path:
    return Path(f"{kind}_trajectory.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SAT furnace H/F/I trajectories.")
    parser.add_argument("--kind", choices=["sat", "unsat", "hard_sat", "random"], default="sat")
    parser.add_argument("--variables", type=int, default=32)
    parser.add_argument("--clauses", type=int, default=136)
    parser.add_argument("--clause-size", type=int, default=3)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--learning-rate", type=float, default=0.055)
    parser.add_argument("--inertia", type=float, default=0.82)
    parser.add_argument("--noise", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--spatial-out", type=Path, help="Optional CSV path for per-variable/per-clause spatial sprite fields.")
    parser.add_argument("--graph-out", type=Path, help="Optional CSV path for clause-variable adjacency graph.")
    parser.add_argument("--trace-out", type=Path, help="Optional CSV path for Riordan operator trace rows.")
    parser.add_argument("--adaptive", action="store_true", help="Enable surplus/deficit adaptive choice controls.")
    parser.add_argument("--memory-decay", type=float, default=0.92, help="Exponential decay for adaptive fiber-bundle memory.")
    parser.add_argument("--memory-drive", type=float, default=0.12, help="Strength of adaptive fiber-bundle memory bias.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    learning_rate = args.learning_rate * (0.55 if args.kind == "hard_sat" else 1.0)
    inertia = max(args.inertia, 0.92) if args.kind == "hard_sat" else args.inertia
    noise = max(args.noise, 0.025) if args.kind == "hard_sat" else args.noise

    formula, planted = generate_formula(args.kind, args.variables, args.clauses, args.clause_size, rng)
    result = run_furnace(
        formula=formula,
        variables=args.variables,
        steps=args.steps,
        rng=rng,
        temperature=args.temperature,
        learning_rate=learning_rate,
        inertia=inertia,
        noise=noise,
        planted_assignment=planted,
        adaptive=args.adaptive,
        memory_decay=args.memory_decay,
        memory_drive=args.memory_drive,
    )
    output = args.out or default_output_path(args.kind)
    write_samples(output, result.samples)
    if args.spatial_out is not None:
        write_spatial_samples(args.spatial_out, result.spatial_samples)
    if args.graph_out is not None:
        write_graph(args.graph_out, formula)
    if args.trace_out is not None:
        write_operator_traces(args.trace_out, result.operator_traces)

    if planted is not None:
        planted_ok = all(clause_satisfied(clause, planted) for clause in formula)
        print(f"planted_satisfies_formula: {planted_ok}")
    print(f"kind: {args.kind}")
    print(f"variables: {args.variables}")
    print(f"clauses: {len(formula)}")
    print(f"steps: {len(result.samples)}")
    print(f"solved_by_furnace: {result.solved}")
    print(f"adaptive: {args.adaptive}")
    print(f"final_unsatisfied_clauses: {result.samples[-1].unsatisfied_clauses}")
    print(f"wrote: {output}")
    if args.spatial_out is not None:
        print(f"wrote_spatial: {args.spatial_out}")
    if args.graph_out is not None:
        print(f"wrote_graph: {args.graph_out}")
    if args.trace_out is not None:
        print(f"wrote_trace: {args.trace_out}")


if __name__ == "__main__":
    main()
