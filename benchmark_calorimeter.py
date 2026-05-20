#!/usr/bin/env python3
"""Benchmark SAT furnace trajectories with spectral and sprite diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import sat_composer
import sat_furnace
import spectral_calorimeter
import sprite_detector
from composer import Composer, FieldOperator

KINDS = ("sat", "unsat", "hard_sat")
DEFAULT_TRACE_CHECKPOINTS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00)
Scalar = float | int | str | bool


@dataclass(frozen=True)
class CompositionGene:
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    arity: int
    fanout: int
    role: str


@dataclass(frozen=True)
class CompositionGenome:
    target: str
    genes: tuple[CompositionGene, ...]
    edges: tuple[tuple[str, str], ...]
    missing_inputs: tuple[str, ...]


@dataclass(frozen=True)
class GeneMutationCandidate:
    gene: str
    role: str
    mutation: str
    score: float
    reason: str


@dataclass(frozen=True)
class TransitionMotif:
    source: str
    target: str
    count: int
    activation_rate: float
    mean_delta_unsatisfied: float
    mean_delta_integration: float
    entropy_shift: float
    persistence: float
    role: str


@dataclass(frozen=True)
class MotifEffect:
    motif: str
    role: str
    requires: tuple[str, ...]
    provides: tuple[str, ...]
    confidence: float
    evidence: float


@dataclass(frozen=True)
class MotifBootstrapPlan:
    targets: tuple[str, ...]
    order: tuple[str, ...]
    missing: tuple[str, ...]
    provided_effects: tuple[str, ...]
    provider_count: int


@dataclass(frozen=True)
class MotifRoleRule:
    role: str
    released_tension: float = 0.0
    min_persistence: float = 0.0
    entropy_min: float | None = None
    entropy_max: float | None = None


@dataclass(frozen=True)
class MotifRoleEffect:
    role: str
    requires: tuple[str, ...]
    provides: tuple[str, ...]


@dataclass(frozen=True)
class MotifFallbackNeedRule:
    need: str
    absent_role: str
    signal: str
    threshold: float
    polarity: str = "below"


@dataclass(frozen=True)
class MotifNeedRule:
    need: str
    signals: tuple[str, ...]
    threshold: float
    polarity: str = "above"


@dataclass(frozen=True)
class MotifHintRule:
    effect: str
    source: str
    hint: str


MOTIF_PRESSURE_EFFECTS = (
    "entropy_release",
    "bridge_opportunity",
    "puncture_repair_cycle",
    "stabilization_window",
)

MOTIF_HINT_RULES = (
    MotifHintRule("puncture_repair_cycle", "target", "explore_puncture_repair"),
    MotifHintRule("bridge_opportunity", "target", "explore_bridge_building"),
    MotifHintRule("entropy_release", "target", "explore_entropy_release"),
    MotifHintRule("stabilization_window", "target", "explore_stabilization"),
    MotifHintRule("entropy_release", "prerequisite", "prepare_entropy_release"),
    MotifHintRule("bridge_opportunity", "prerequisite", "prepare_bridge_building"),
    MotifHintRule("puncture_repair_cycle", "any", "explore_puncture_repair"),
    MotifHintRule("bridge_opportunity", "any", "explore_bridge_building"),
    MotifHintRule("entropy_release", "any", "explore_entropy_release"),
    MotifHintRule("stabilization_window", "any", "explore_stabilization"),
)

MOTIF_ROLE_RULES = (
    MotifRoleRule("puncture_and_seal", released_tension=0.35, min_persistence=0.20),
    MotifRoleRule("thermal_softening", entropy_min=0.04),
    MotifRoleRule("crystallization", entropy_max=-0.04, min_persistence=0.10),
    MotifRoleRule("bridge_building", min_persistence=0.25),
)

MOTIF_ROLE_EFFECTS = (
    MotifRoleEffect("thermal_softening", (), ("entropy_release",)),
    MotifRoleEffect("bridge_building", ("entropy_release",), ("bridge_opportunity",)),
    MotifRoleEffect("puncture_and_seal", ("entropy_release", "bridge_opportunity"), ("puncture_repair_cycle", "stabilization_window")),
    MotifRoleEffect("crystallization", ("bridge_opportunity",), ("stabilization_window",)),
    MotifRoleEffect("drift", (), ("motif_observation",)),
)

MOTIF_ROLE_EFFECT_BY_ROLE = {
    effect.role: effect
    for effect in MOTIF_ROLE_EFFECTS
}

CLIMATE_NEED_RULES = (
    MotifNeedRule("bridge_opportunity", ("is_border_niche", "puzzle_border_score"), 0.45),
    MotifNeedRule("puncture_repair_cycle", ("puzzle_composition_pressure", "trace_trap_contribution"), 0.35),
    MotifNeedRule("entropy_release", ("trace_trap_contribution", "loop_stagnation"), 0.20),
    MotifNeedRule("stabilization_window", ("unsolved_collapse",), 0.50),
    MotifNeedRule("stabilization_window", ("is_gradient_niche",), 1.0),
)

MOTIF_NEED_RULES = (
    MotifFallbackNeedRule("bridge_opportunity", "bridge_building", "mean_persistence", 0.15),
    MotifFallbackNeedRule("entropy_release", "thermal_softening", "mean_entropy_shift", 0.02),
    MotifFallbackNeedRule("puncture_repair_cycle", "puncture_and_seal", "mean_persistence", 0.25),
)


@dataclass(frozen=True)
class MutationControls:
    enabled: bool
    mutation: str
    source_gene: str
    adaptive: bool
    policy: str
    spike_threshold: float
    spike_slope: float
    memory_decay: float
    memory_drive: float
    learning_rate_scale: float
    inertia_delta: float
    noise_delta: float


def run_trial(
    kind: str,
    seed: int,
    variables: int,
    clauses: int,
    clause_size: int,
    steps: int,
    temperature: float,
    learning_rate: float,
    inertia: float,
    noise: float,
    window: int,
    step_size: int,
    runner_quantile: float,
    adaptive: bool = False,
    policy: str = "baseline",
    spike_threshold: float = 0.35,
    spike_slope: float = 8.0,
    memory_decay: float = 0.92,
    memory_drive: float = 0.12,
    baseline_restarts: int = 256,
    trace_rows: list[dict[str, Scalar]] | None = None,
    trace_checkpoints: Sequence[float] = DEFAULT_TRACE_CHECKPOINTS,
) -> dict[str, Scalar]:
    rng = random.Random(seed)
    adjusted_learning_rate = learning_rate * (0.55 if kind == "hard_sat" else 1.0)
    adjusted_inertia = max(inertia, 0.92) if kind == "hard_sat" else inertia
    adjusted_noise = max(noise, 0.025) if kind == "hard_sat" else noise
    formula, planted = sat_furnace.generate_formula(
        kind, variables, clauses, clause_size, rng
    )
    baseline_rng = random.Random(seed + 1_000_003)
    random_solved, random_best_unsatisfied = random_assignment_baseline(
        formula, variables, baseline_rng, baseline_restarts
    )
    walksat_solved, walksat_best_unsatisfied, walksat_flips, walksat_snapshots = (
        walksat_baseline_with_trace(
            formula,
            variables,
            baseline_rng,
            steps,
            trace_checkpoints=trace_checkpoints,
        )
    )
    trial_ctx = {
        "kind": kind,
        "seed": seed,
        "formula": formula,
        "planted_assignment": planted,
        "variables": variables,
        "steps": steps,
        "rng": rng,
        "temperature": temperature,
        "learning_rate": adjusted_learning_rate,
        "inertia": adjusted_inertia,
        "noise": adjusted_noise,
        "adaptive": adaptive,
        "policy": policy,
        "spike_threshold": spike_threshold,
        "spike_slope": spike_slope,
        "memory_decay": memory_decay,
        "memory_drive": memory_drive,
        "window": window,
        "step_size": step_size,
        "runner_quantile": runner_quantile,
        "baseline_restarts": baseline_restarts,
        "random_solved": random_solved,
        "random_best_unsatisfied": random_best_unsatisfied,
        "walksat_solved": walksat_solved,
        "walksat_best_unsatisfied": walksat_best_unsatisfied,
        "walksat_flips": walksat_flips,
    }
    trial_targets = ["metrics_row", "furnace_result", "calorimeter_report"]
    trial_out = sat_composer.build_trial_composer().run(trial_targets, trial_ctx)
    row = trial_out["metrics_row"]
    selected_candidate = GeneMutationCandidate(
        gene=str(row.get("gene_border_selected_gene", "none")),
        role=str(row.get("gene_border_selected_role", "none")),
        mutation=str(row.get("gene_border_selected_mutation", "none")),
        score=float(row.get("gene_border_selected_score", 0.0)),
        reason=str(row.get("gene_border_selected_reason", "")),
    )
    mutation_controls = mutation_controls_from_candidate(
        selected_candidate,
        adaptive=adaptive,
        policy=policy,
        spike_threshold=spike_threshold,
        spike_slope=spike_slope,
        memory_decay=memory_decay,
        memory_drive=memory_drive,
    )
    row.update(mutation_control_metrics(mutation_controls))
    row.update(run_mutant_replay(
        formula=formula,
        variables=variables,
        steps=steps,
        seed=seed,
        temperature=temperature,
        learning_rate=adjusted_learning_rate,
        inertia=adjusted_inertia,
        noise=adjusted_noise,
        planted_assignment=planted,
        baseline_best_unsatisfied=int(row["furnace_best_unsatisfied"]),
        controls=mutation_controls,
    ))

    if trace_rows is not None:
        result = trial_out["furnace_result"]
        report = trial_out["calorimeter_report"]
        trace_context = {
            "kind": kind,
            "seed": seed,
            "variables": variables,
            "clauses": len(formula),
            "steps": steps,
            "adaptive": adaptive,
            "policy": policy,
            "spike_threshold": spike_threshold,
            "spike_slope": spike_slope,
            "random_solved": random_solved,
            "random_best_unsatisfied": random_best_unsatisfied,
            "walksat_solved": walksat_solved,
            "walksat_best_unsatisfied": walksat_best_unsatisfied,
            "walksat_flips": walksat_flips,
            "furnace_solved": result.solved,
            "furnace_best_unsatisfied": int(row["furnace_best_unsatisfied"]),
            "furnace_final_assignment_unsatisfied": int(row["furnace_final_assignment_unsatisfied"]),
        }
        trace_rows.extend(
            furnace_trace_rows(trace_context, result, report, trace_checkpoints)
        )
        trace_rows.extend(walksat_trace_rows(trace_context, walksat_snapshots))
    return row


def random_assignment_baseline(
    formula: sat_furnace.CNF,
    variables: int,
    rng: random.Random,
    restarts: int,
) -> tuple[bool, int]:
    best_unsatisfied = len(formula)
    for _ in range(max(1, restarts)):
        assignment = [rng.choice([False, True]) for _ in range(variables)]
        unsatisfied = count_unsatisfied(formula, assignment)
        best_unsatisfied = min(best_unsatisfied, unsatisfied)
        if unsatisfied == 0:
            return True, 0
    return False, best_unsatisfied


def walksat_baseline(
    formula: sat_furnace.CNF,
    variables: int,
    rng: random.Random,
    max_flips: int,
    random_flip_probability: float = 0.35,
) -> tuple[bool, int, int]:
    solved, best_unsatisfied, flips, _snapshots = walksat_baseline_with_trace(
        formula,
        variables,
        rng,
        max_flips,
        random_flip_probability=random_flip_probability,
        trace_checkpoints=(1.0,),
    )
    return solved, best_unsatisfied, flips


def walksat_baseline_with_trace(
    formula: sat_furnace.CNF,
    variables: int,
    rng: random.Random,
    max_flips: int,
    random_flip_probability: float = 0.35,
    trace_checkpoints: Sequence[float] = DEFAULT_TRACE_CHECKPOINTS,
) -> tuple[bool, int, int, list[dict[str, Scalar]]]:
    flip_limit = max(0, max_flips)
    assignment = [rng.choice([False, True]) for _ in range(variables)]
    current_unsatisfied = count_unsatisfied(formula, assignment)
    best_unsatisfied = current_unsatisfied
    solved = current_unsatisfied == 0
    solved_flip = 0 if solved else flip_limit
    last_improvement_flip = 0

    current_history = [current_unsatisfied]
    best_history = [best_unsatisfied]
    steps_since_improvement_history = [0]
    random_flip_count_history = [0]
    greedy_flip_count_history = [0]
    repeated_variable_flip_count_history = [0]
    clause_revisit_count_history = [0]
    immediate_gain_history = [0.0]
    mean_gain_history = [0.0]

    random_flip_count = 0
    greedy_flip_count = 0
    repeated_variable_flip_count = 0
    clause_revisit_count = 0
    cumulative_gain = 0.0
    variable_flip_counts = [0 for _ in range(variables)]
    selected_clause_counts = [0 for _ in formula]

    for flip in range(1, flip_limit + 1):
        unsatisfied_clause_ids = [
            clause_id
            for clause_id, clause in enumerate(formula)
            if not sat_furnace.clause_satisfied(clause, assignment)
        ]
        if not unsatisfied_clause_ids:
            solved = True
            solved_flip = flip - 1
            break

        previous_unsatisfied = current_unsatisfied
        clause_id = rng.choice(unsatisfied_clause_ids)
        clause = formula[clause_id]
        if selected_clause_counts[clause_id] > 0:
            clause_revisit_count += 1
        selected_clause_counts[clause_id] += 1

        if rng.random() < random_flip_probability:
            variable = rng.choice(clause)[0]
            random_flip_count += 1
        else:
            candidate_scores = [
                (literal[0], flipped_unsatisfied_count(formula, assignment, literal[0]))
                for literal in clause
            ]
            variable, _candidate_unsatisfied = min(
                candidate_scores, key=lambda item: item[1]
            )
            greedy_flip_count += 1

        if variable_flip_counts[variable] > 0:
            repeated_variable_flip_count += 1
        variable_flip_counts[variable] += 1

        assignment[variable] = not assignment[variable]
        current_unsatisfied = count_unsatisfied(formula, assignment)
        immediate_gain = float(previous_unsatisfied - current_unsatisfied)
        cumulative_gain += immediate_gain
        if current_unsatisfied < best_unsatisfied:
            best_unsatisfied = current_unsatisfied
            last_improvement_flip = flip

        current_history.append(current_unsatisfied)
        best_history.append(best_unsatisfied)
        steps_since_improvement_history.append(flip - last_improvement_flip)
        random_flip_count_history.append(random_flip_count)
        greedy_flip_count_history.append(greedy_flip_count)
        repeated_variable_flip_count_history.append(repeated_variable_flip_count)
        clause_revisit_count_history.append(clause_revisit_count)
        immediate_gain_history.append(immediate_gain)
        mean_gain_history.append(cumulative_gain / max(1, flip))

        if current_unsatisfied == 0:
            solved = True
            solved_flip = flip
            break

    executed_flips = len(current_history) - 1
    snapshots = []
    for fraction, target_flip in checkpoint_targets(flip_limit, trace_checkpoints):
        index = min(target_flip, executed_flips)
        snapshots.append(
            {
                "checkpoint_fraction": fraction,
                "checkpoint_step": target_flip,
                "observed_step": index,
                "current_unsatisfied": current_history[index],
                "best_unsatisfied_so_far": best_history[index],
                "steps_since_improvement": steps_since_improvement_history[index],
                "improvement_rate_recent": recent_best_improvement_rate(
                    best_history, index
                ),
                "unsatisfied_slope_recent": recent_slope(current_history, index),
                "solved_so_far": current_history[index] == 0
                or (solved and solved_flip <= target_flip),
                "walksat_random_flip_rate": random_flip_count_history[index]
                / max(1, index),
                "walksat_greedy_flip_rate": greedy_flip_count_history[index]
                / max(1, index),
                "walksat_repeated_variable_flip_rate": repeated_variable_flip_count_history[
                    index
                ]
                / max(1, index),
                "walksat_clause_revisit_rate": clause_revisit_count_history[index]
                / max(1, index),
                "walksat_last_flip_gain": immediate_gain_history[index],
                "walksat_mean_flip_gain": mean_gain_history[index],
            }
        )
    return (
        solved,
        best_unsatisfied,
        solved_flip if solved else executed_flips,
        snapshots,
    )


def checkpoint_targets(
    budget_steps: int,
    checkpoints: Sequence[float],
) -> list[tuple[float, int]]:
    limit = max(0, budget_steps)
    normalized = normalized_trace_checkpoints(checkpoints)
    return [
        (fraction, min(limit, max(0, int(round(fraction * limit)))))
        for fraction in normalized
    ]


def normalized_trace_checkpoints(checkpoints: Sequence[float]) -> tuple[float, ...]:
    values = sorted({clamp01(float(value)) for value in checkpoints})
    return tuple(values) if values else (1.0,)


def furnace_trace_rows(
    context: dict[str, Scalar],
    result: sat_furnace.FurnaceResult,
    report: spectral_calorimeter.CalorimeterReport,
    checkpoints: Sequence[float],
) -> list[dict[str, Scalar]]:
    samples = result.samples
    if not samples:
        return []
    current_history = [sample.unsatisfied_clauses for sample in samples]
    best_history = cumulative_min(current_history)
    steps_since_history = steps_since_best_improvement(current_history)
    rows: list[dict[str, Scalar]] = []
    max_index = len(samples) - 1
    for fraction, target_index in checkpoint_targets(max_index, checkpoints):
        index = min(target_index, max_index)
        sample = samples[index]
        eligible_windows = [
            window for window in report.windows if window.end_t <= float(sample.t)
        ]
        latest_window = eligible_windows[-1] if eligible_windows else None
        row = dict(context)
        row.update(
            {
                "method": "furnace",
                "method_final_solved": result.solved,
                "method_final_best_unsatisfied": context["furnace_best_unsatisfied"],
                "checkpoint_fraction": fraction,
                "checkpoint_step": target_index,
                "observed_step": sample.t,
                "current_unsatisfied": sample.unsatisfied_clauses,
                "best_unsatisfied_so_far": best_history[index],
                "steps_since_improvement": steps_since_history[index],
                "improvement_rate_recent": recent_best_improvement_rate(
                    best_history, index
                ),
                "unsatisfied_slope_recent": recent_slope(current_history, index),
                "solved_so_far": sample.unsatisfied_clauses == 0
                or result.solved
                and index >= first_zero_index(current_history),
                "heat": sample.heat,
                "free_energy": sample.free_energy,
                "integration": sample.integration,
                "assignment_entropy": sample.assignment_entropy,
                "spectral_centroid": latest_window.spectral_centroid
                if latest_window
                else 0.0,
                "dominant_frequency": latest_window.dominant_frequency
                if latest_window
                else 0.0,
                "spectral_entropy": latest_window.spectral_entropy
                if latest_window
                else 0.0,
                "concentration_index": latest_window.concentration_index
                if latest_window
                else 0.0,
                "fragmentation_index": latest_window.fragmentation_index
                if latest_window
                else 0.0,
                "recycling_score": latest_window.recycling_score
                if latest_window
                else 0.0,
                "collapse_index": latest_window.collapse_index
                if latest_window
                else 0.0,
                "redshift_rate_so_far": spectral_slope(
                    eligible_windows, "spectral_centroid"
                ),
                "spectral_entropy_slope_so_far": spectral_slope(
                    eligible_windows, "spectral_entropy"
                ),
                "integration_slope_so_far": spectral_slope(
                    eligible_windows, "mean_integration"
                ),
                "collapse_slope_so_far": spectral_slope(
                    eligible_windows, "collapse_index"
                ),
            }
        )
        row.update(
            operator_trace_metrics(prefix_traces(result.operator_traces, sample.t))
        )
        row.update(
            excitable_trace_chain_snapshot(
                prefix_traces(result.operator_traces, sample.t)
            )
        )
        rows.append(row)
    return rows


def walksat_trace_rows(
    context: dict[str, Scalar], snapshots: Sequence[dict[str, Scalar]]
) -> list[dict[str, Scalar]]:
    rows: list[dict[str, Scalar]] = []
    for snapshot in snapshots:
        row = dict(context)
        row.update(
            {
                "method": "walksat",
                "method_final_solved": context["walksat_solved"],
                "method_final_best_unsatisfied": context["walksat_best_unsatisfied"],
            }
        )
        row.update(snapshot)
        rows.append(row)
    return rows


def prefix_traces(
    traces: Sequence[sat_furnace.OperatorTrace], t: int
) -> list[sat_furnace.OperatorTrace]:
    return [trace for trace in traces if trace.t <= t]


def spectral_slope(
    windows: Sequence[spectral_calorimeter.WindowSpectrum], attribute: str
) -> float:
    if len(windows) < 2:
        return 0.0
    return spectral_calorimeter.slope(
        [window.center_t for window in windows],
        [float(getattr(window, attribute)) for window in windows],
    )


def cumulative_min(values: Sequence[int]) -> list[int]:
    best_values: list[int] = []
    best = math.inf
    for value in values:
        best = min(best, value)
        best_values.append(int(best))
    return best_values


def steps_since_best_improvement(values: Sequence[int]) -> list[int]:
    best = math.inf
    last_improvement = 0
    result: list[int] = []
    for index, value in enumerate(values):
        if value < best:
            best = value
            last_improvement = index
        result.append(index - last_improvement)
    return result


def first_zero_index(values: Sequence[int]) -> int:
    for index, value in enumerate(values):
        if value == 0:
            return index
    return math.inf  # type: ignore[return-value]


def recent_slope(values: Sequence[int | float], index: int, window: int = 32) -> float:
    start = max(0, index - window)
    segment = [float(value) for value in values[start : index + 1]]
    if len(segment) < 2:
        return 0.0
    return spectral_calorimeter.slope(list(range(len(segment))), segment)


def recent_best_improvement_rate(
    best_history: Sequence[int | float], index: int, window: int = 32
) -> float:
    start = max(0, index - window)
    elapsed = max(1, index - start)
    return (float(best_history[start]) - float(best_history[index])) / elapsed


def flipped_unsatisfied_count(
    formula: sat_furnace.CNF, assignment: list[bool], variable: int
) -> int:
    assignment[variable] = not assignment[variable]
    unsatisfied = count_unsatisfied(formula, assignment)
    assignment[variable] = not assignment[variable]
    return unsatisfied


def count_unsatisfied(formula: sat_furnace.CNF, assignment: Sequence[bool]) -> int:
    return sum(
        1 for clause in formula if not sat_furnace.clause_satisfied(clause, assignment)
    )


def composition_genome_for_targets(
    composer: object,
    targets: Sequence[str],
    available_keys: Sequence[str],
) -> CompositionGenome:
    plan = composer.plan(targets, available_keys)  # type: ignore[attr-defined]
    dependency_graph = composer.graph(targets, available_keys)  # type: ignore[attr-defined]
    operators = getattr(composer, "_operators")
    output_consumer_counts: dict[str, int] = {}
    for name in plan.order:
        operator = operators[name]
        for input_key in operator.inputs:
            output_consumer_counts[input_key] = output_consumer_counts.get(input_key, 0) + 1

    genes: list[CompositionGene] = []
    for name in plan.order:
        operator = operators[name]
        outputs = tuple(operator.outputs)
        fanout = sum(output_consumer_counts.get(output, 0) for output in outputs)
        genes.append(
            CompositionGene(
                name=name,
                inputs=tuple(operator.inputs),
                outputs=outputs,
                arity=len(operator.inputs),
                fanout=fanout,
                role=composition_gene_role(name, tuple(operator.inputs), outputs),
            )
        )

    return CompositionGenome(
        target="+".join(targets),
        genes=tuple(genes),
        edges=tuple(dependency_graph.edges),
        missing_inputs=tuple(item.key for item in plan.missing),
    )


def solver_composition_genome() -> CompositionGenome:
    composer = sat_composer.build_solver_composer()
    available = (
        "formula", "variables", "steps", "temperature", "learning_rate", "inertia",
        "noise", "adaptive", "policy", "spike_threshold", "spike_slope",
        "memory_decay", "memory_drive", "planted_assignment", "rng", "spins",
        "velocity", "fiber_memory", "prev_samples", "prev_spatial_samples",
        "prev_operator_traces", "prev_best_spins", "prev_best_unsatisfied",
        "prev_concentrations", "previous_unsatisfied", "previous_integration", "t",
    )
    return composition_genome_for_targets(
        composer,
        sat_furnace._EPOCH_TARGETS,  # type: ignore[attr-defined]
        available,
    )


def composition_gene_role(
    name: str,
    inputs: Sequence[str],
    outputs: Sequence[str],
) -> str:
    joined = " ".join((name, *inputs, *outputs))
    if "validate" in name:
        return "validator"
    if "trace" in joined or "metrics" in joined or "report" in joined:
        return "observer"
    if "adaptive" in joined or "control" in joined or "policy" in joined:
        return "control"
    if "memory" in joined or "best" in joined or "sample_append" in name:
        return "memory"
    if "bias" in joined or "field" in joined or "drive" in joined or "spin" in joined:
        return "transform"
    if "formula" in joined or "graph" in joined or "spatial" in joined:
        return "adapter"
    return "transform"


def composition_genome_metrics(
    genome: CompositionGenome,
    puzzle_border_score: float,
    puzzle_composition_pressure: float,
    solved: bool,
    furnace_best_unsatisfied: int,
    walksat_best_unsatisfied: int,
) -> dict[str, Scalar]:
    gene_count = len(genome.genes)
    edge_count = len(genome.edges)
    max_possible_edges = max(1, gene_count * max(1, gene_count - 1))
    role_counts: dict[str, int] = {}
    for gene in genome.genes:
        role_counts[gene.role] = role_counts.get(gene.role, 0) + 1

    observer_count = role_counts.get("observer", 0)
    control_count = role_counts.get("control", 0)
    memory_count = role_counts.get("memory", 0)
    transform_count = role_counts.get("transform", 0)
    adapter_count = role_counts.get("adapter", 0)
    validator_count = role_counts.get("validator", 0)
    arity_mean = mean(gene.arity for gene in genome.genes)
    fanout_mean = mean(gene.fanout for gene in genome.genes)
    fanout_peak = max((gene.fanout for gene in genome.genes), default=0)
    role_diversity = len(role_counts) / 6.0
    edge_density = edge_count / max_possible_edges
    missing_penalty = min(1.0, len(genome.missing_inputs) / max(1, gene_count))
    parsimony = 1.0 / (1.0 + gene_count / 32.0)
    composition_complexity = clamp01(
        0.35 * min(1.0, gene_count / 32.0)
        + 0.20 * min(1.0, arity_mean / 8.0)
        + 0.20 * min(1.0, fanout_mean / 4.0)
        + 0.15 * edge_density
        + 0.10 * role_diversity
    )
    border_fit = clamp01(
        0.45 * puzzle_border_score
        + 0.25 * puzzle_composition_pressure
        + 0.15 * role_diversity
        + 0.15 * min(1.0, (control_count + memory_count + observer_count) / max(1, gene_count))
    )
    solve_reward = 1.0 if solved else 0.0
    walksat_gap = clamp01((walksat_best_unsatisfied - furnace_best_unsatisfied) / max(1, walksat_best_unsatisfied + 1))
    genome_fitness = clamp01(
        0.35 * border_fit
        + 0.25 * solve_reward
        + 0.20 * walksat_gap
        + 0.10 * parsimony
        + 0.10 * (1.0 - missing_penalty)
    )

    return {
        "composition_gene_count": float(gene_count),
        "composition_edge_count": float(edge_count),
        "composition_edge_density": edge_density,
        "composition_mean_arity": arity_mean,
        "composition_mean_fanout": fanout_mean,
        "composition_peak_fanout": float(fanout_peak),
        "composition_role_diversity": role_diversity,
        "composition_adapter_genes": float(adapter_count),
        "composition_transform_genes": float(transform_count),
        "composition_control_genes": float(control_count),
        "composition_memory_genes": float(memory_count),
        "composition_observer_genes": float(observer_count),
        "composition_validator_genes": float(validator_count),
        "composition_missing_inputs": float(len(genome.missing_inputs)),
        "composition_complexity": composition_complexity,
        "composition_parsimony": parsimony,
        "composition_border_fit": border_fit,
        "composition_genome_fitness": genome_fitness,
        "composition_gene_sequence": ">".join(gene.name for gene in genome.genes),
    }


def _prefer_policy(state: dict, fallback: str) -> None:
    if state["policy"] == "baseline":
        state["policy"] = fallback


def _mutation_retune_gate(state: dict, strength: float) -> None:
    state["adaptive"] = True
    _prefer_policy(state, sat_composer.CURRICULUM_SEED_POLICY)
    state["spike_threshold"] = max(0.12, state["spike_threshold"] - 0.10 * strength)
    state["spike_slope"] = min(14.0, state["spike_slope"] + 2.0 * strength)


def _mutation_lower_activation_border(state: dict, strength: float) -> None:
    state["adaptive"] = True
    _prefer_policy(state, sat_composer.EXCITABLE_POLICY)
    state["spike_threshold"] = max(0.10, state["spike_threshold"] - 0.16 * strength)
    state["noise_delta"] = 0.004 * strength


def _mutation_mutate_memory_decay(state: dict, strength: float) -> None:
    state["adaptive"] = True
    state["memory_decay"] = max(0.72, state["memory_decay"] - 0.10 * strength)
    state["memory_drive"] = min(0.28, state["memory_drive"] + 0.08 * strength)


def _mutation_reweight_transform(state: dict, strength: float) -> None:
    _prefer_policy(state, sat_composer.EXCITABLE_POLICY)
    state["spike_threshold"] = max(0.14, state["spike_threshold"] - 0.07 * strength)
    state["learning_rate_scale"] = 1.0 + 0.12 * strength


def _mutation_inhibit_or_rescale(state: dict, strength: float) -> None:
    state["learning_rate_scale"] = 1.0 - 0.18 * strength
    state["inertia_delta"] = -0.03 * strength
    state["noise_delta"] = -0.003 * strength


def _mutation_instrument_or_expose(state: dict, strength: float) -> None:
    state["adaptive"] = True
    _prefer_policy(state, sat_composer.CURRICULUM_SEED_POLICY)
    state["memory_drive"] = min(0.24, state["memory_drive"] + 0.04 * strength)


def _mutation_increase_resolution(state: dict, strength: float) -> None:
    _prefer_policy(state, sat_composer.CURRICULUM_SEED_POLICY)
    state["spike_slope"] = min(14.0, state["spike_slope"] + 1.5 * strength)


def _mutation_recombine_provider(state: dict, strength: float) -> None:
    state["policy"] = sat_composer.CURRICULUM_SEED_POLICY
    state["spike_threshold"] = max(0.16, state["spike_threshold"] - 0.05 * strength)


def _mutation_tighten_constraint(state: dict, strength: float) -> None:
    state["learning_rate_scale"] = 0.96
    state["noise_delta"] = -0.002 * strength


def _mutation_default(state: dict, strength: float) -> None:
    _prefer_policy(state, sat_composer.EXCITABLE_POLICY)
    state["spike_threshold"] = max(0.15, state["spike_threshold"] - 0.05 * strength)


_MUTATION_HANDLERS: dict[str, Callable[[dict, float], None]] = {
    "retune_gate": _mutation_retune_gate,
    "lower_activation_border": _mutation_lower_activation_border,
    "mutate_memory_decay": _mutation_mutate_memory_decay,
    "reweight_transform": _mutation_reweight_transform,
    "inhibit_or_rescale": _mutation_inhibit_or_rescale,
    "instrument_or_expose": _mutation_instrument_or_expose,
    "increase_resolution": _mutation_increase_resolution,
    "recombine_provider": _mutation_recombine_provider,
    "tighten_constraint": _mutation_tighten_constraint,
}


def mutation_controls_from_candidate(
    candidate: GeneMutationCandidate,
    *,
    adaptive: bool,
    policy: str,
    spike_threshold: float,
    spike_slope: float,
    memory_decay: float,
    memory_drive: float,
) -> MutationControls:
    if candidate.gene == "none" or candidate.score <= 0.0:
        return MutationControls(
            enabled=False,
            mutation="none",
            source_gene="none",
            adaptive=adaptive,
            policy=policy,
            spike_threshold=spike_threshold,
            spike_slope=spike_slope,
            memory_decay=memory_decay,
            memory_drive=memory_drive,
            learning_rate_scale=1.0,
            inertia_delta=0.0,
            noise_delta=0.0,
        )

    strength = clamp01(candidate.score)
    state = {
        "adaptive": adaptive,
        "policy": policy,
        "spike_threshold": spike_threshold,
        "spike_slope": spike_slope,
        "memory_decay": memory_decay,
        "memory_drive": memory_drive,
        "learning_rate_scale": 1.0,
        "inertia_delta": 0.0,
        "noise_delta": 0.0,
    }

    handler = _MUTATION_HANDLERS.get(candidate.mutation, _mutation_default)
    handler(state, strength)

    return MutationControls(
        enabled=True,
        mutation=candidate.mutation,
        source_gene=candidate.gene,
        adaptive=state["adaptive"],
        policy=state["policy"],
        spike_threshold=state["spike_threshold"],
        spike_slope=state["spike_slope"],
        memory_decay=state["memory_decay"],
        memory_drive=state["memory_drive"],
        learning_rate_scale=max(0.70, min(1.25, state["learning_rate_scale"])),
        inertia_delta=max(-0.08, min(0.04, state["inertia_delta"])),
        noise_delta=max(-0.008, min(0.010, state["noise_delta"])),
    )


def mutation_control_metrics(controls: MutationControls) -> dict[str, Scalar]:
    return {
        "mutation_enabled": controls.enabled,
        "mutation_source_gene": controls.source_gene,
        "mutation_action": controls.mutation,
        "mutation_adaptive": controls.adaptive,
        "mutation_policy": controls.policy,
        "mutation_spike_threshold": controls.spike_threshold,
        "mutation_spike_slope": controls.spike_slope,
        "mutation_memory_decay": controls.memory_decay,
        "mutation_memory_drive": controls.memory_drive,
        "mutation_learning_rate_scale": controls.learning_rate_scale,
        "mutation_inertia_delta": controls.inertia_delta,
        "mutation_noise_delta": controls.noise_delta,
    }


def run_mutant_replay(
    *,
    formula: sat_furnace.CNF,
    variables: int,
    steps: int,
    seed: int,
    temperature: float,
    learning_rate: float,
    inertia: float,
    noise: float,
    planted_assignment: list[bool] | None,
    baseline_best_unsatisfied: int,
    controls: MutationControls,
) -> dict[str, Scalar]:
    if not controls.enabled:
        return empty_mutant_metrics()

    result = sat_furnace.run_furnace(
        formula=formula,
        variables=variables,
        steps=steps,
        rng=random.Random(seed + 2_000_003),
        temperature=temperature,
        learning_rate=learning_rate * controls.learning_rate_scale,
        inertia=max(0.0, min(0.985, inertia + controls.inertia_delta)),
        noise=max(0.0, noise + controls.noise_delta),
        planted_assignment=planted_assignment,
        adaptive=controls.adaptive,
        memory_decay=controls.memory_decay,
        memory_drive=controls.memory_drive,
        policy=controls.policy,
        spike_threshold=controls.spike_threshold,
        spike_slope=controls.spike_slope,
    )
    mutant_best = min((sample.unsatisfied_clauses for sample in result.samples), default=len(formula))
    improvement = baseline_best_unsatisfied - mutant_best
    return {
        "mutation_replay_run": True,
        "mutation_replay_solved": result.solved,
        "mutation_replay_best_unsatisfied": int(mutant_best),
        "mutation_replay_delta_best_unsatisfied": int(improvement),
        "mutation_replay_improved": improvement > 0,
        "mutation_replay_trace_count": float(len(result.operator_traces)),
    }


def empty_mutant_metrics() -> dict[str, Scalar]:
    return {
        "mutation_replay_run": False,
        "mutation_replay_solved": False,
        "mutation_replay_best_unsatisfied": -1,
        "mutation_replay_delta_best_unsatisfied": 0,
        "mutation_replay_improved": False,
        "mutation_replay_trace_count": 0.0,
    }


def gene_border_mutation_metrics(
    genome: CompositionGenome,
    traces: Sequence[sat_furnace.OperatorTrace],
    puzzle_border_score: float,
    puzzle_composition_pressure: float,
    solved: bool,
    furnace_best_unsatisfied: int,
    walksat_best_unsatisfied: int,
    top_k: int = 3,
) -> dict[str, Scalar]:
    candidates = select_gene_border_mutations(
        genome=genome,
        traces=traces,
        puzzle_border_score=puzzle_border_score,
        puzzle_composition_pressure=puzzle_composition_pressure,
        solved=solved,
        furnace_best_unsatisfied=furnace_best_unsatisfied,
        walksat_best_unsatisfied=walksat_best_unsatisfied,
        top_k=top_k,
    )
    selected = candidates[0] if candidates else GeneMutationCandidate("none", "none", "none", 0.0, "no_candidate")
    return {
        "gene_border_candidate_count": float(len(candidates)),
        "gene_border_selected_gene": selected.gene,
        "gene_border_selected_role": selected.role,
        "gene_border_selected_mutation": selected.mutation,
        "gene_border_selected_score": selected.score,
        "gene_border_selected_reason": selected.reason,
        "gene_border_top_genes": ">".join(candidate.gene for candidate in candidates),
        "gene_border_top_mutations": ">".join(candidate.mutation for candidate in candidates),
    }


def select_gene_border_mutations(
    genome: CompositionGenome,
    traces: Sequence[sat_furnace.OperatorTrace],
    puzzle_border_score: float,
    puzzle_composition_pressure: float,
    solved: bool,
    furnace_best_unsatisfied: int,
    walksat_best_unsatisfied: int,
    top_k: int = 3,
) -> tuple[GeneMutationCandidate, ...]:
    trace_stats = operator_trace_stats_by_gene(traces)
    outcome_gap = clamp01((furnace_best_unsatisfied - walksat_best_unsatisfied) / max(1, furnace_best_unsatisfied + walksat_best_unsatisfied + 1))
    unsolved_pressure = 0.0 if solved else 1.0
    ecology_pressure = clamp01(
        0.45 * puzzle_border_score
        + 0.30 * puzzle_composition_pressure
        + 0.15 * outcome_gap
        + 0.10 * unsolved_pressure
    )
    if ecology_pressure <= 0.0:
        return ()

    candidates: list[GeneMutationCandidate] = []
    for gene in genome.genes:
        short_name = gene.name.split(".")[-1]
        stats = trace_stats.get(normalize_trace_operator_name(short_name), {})
        activation_rate = float(stats.get("activation_rate", 0.0))
        mean_delta_unsat = float(stats.get("mean_delta_unsat", 0.0))
        mean_peak = float(stats.get("mean_peak", 0.0))
        observed = float(stats.get("count", 0.0)) > 0.0
        role_prior = gene_border_role_prior(gene.role)
        fanout_pressure = min(1.0, gene.fanout / 4.0)
        arity_pressure = min(1.0, gene.arity / 8.0)
        silent_pressure = 0.35 if not observed and gene.role in {"control", "memory", "transform"} else 0.0
        underactive_pressure = max(0.0, 0.35 - activation_rate) if observed else 0.0
        harmful_pressure = clamp01(-mean_delta_unsat / 3.0)
        saturation_pressure = clamp01(mean_peak / 4.0)
        gene_border_score = clamp01(
            ecology_pressure
            * (
                0.24 * role_prior
                + 0.18 * fanout_pressure
                + 0.12 * arity_pressure
                + 0.16 * underactive_pressure
                + 0.16 * harmful_pressure
                + 0.08 * saturation_pressure
                + 0.06 * silent_pressure
            )
        )
        if gene_border_score <= 0.0:
            continue
        candidates.append(
            GeneMutationCandidate(
                gene=gene.name,
                role=gene.role,
                mutation=mutation_action_for_gene(gene, observed, activation_rate, mean_delta_unsat),
                score=gene_border_score,
                reason=gene_border_reason(
                    ecology_pressure, observed, activation_rate, mean_delta_unsat,
                    puzzle_composition_pressure, outcome_gap,
                ),
            )
        )

    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return tuple(candidates[:max(0, top_k)])


def operator_trace_stats_by_gene(
    traces: Sequence[sat_furnace.OperatorTrace],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[sat_furnace.OperatorTrace]] = {}
    for trace in traces:
        grouped.setdefault(normalize_trace_operator_name(trace.operator), []).append(trace)
    stats: dict[str, dict[str, float]] = {}
    for name, group in grouped.items():
        stats[name] = {
            "count": float(len(group)),
            "activation_rate": mean(1.0 if trace.active else 0.0 for trace in group),
            "mean_delta_unsat": mean(trace.delta_unsatisfied for trace in group),
            "mean_peak": mean(abs(trace.output_peak) for trace in group),
        }
    return stats


def normalize_trace_operator_name(name: str) -> str:
    return name.removeprefix("solver.").removeprefix("_")


def gene_border_role_prior(role: str) -> float:
    return {
        "control": 1.00,
        "memory": 0.92,
        "transform": 0.84,
        "observer": 0.58,
        "adapter": 0.46,
        "validator": 0.34,
    }.get(role, 0.50)


def mutation_action_for_gene(
    gene: CompositionGene,
    observed: bool,
    activation_rate: float,
    mean_delta_unsat: float,
) -> str:
    if not observed:
        return "instrument_or_expose"
    if mean_delta_unsat < -0.25:
        return "inhibit_or_rescale"
    if activation_rate < 0.10 and gene.role in {"control", "memory", "transform"}:
        return "lower_activation_border"
    if gene.role == "control":
        return "retune_gate"
    if gene.role == "memory":
        return "mutate_memory_decay"
    if gene.role == "transform":
        return "reweight_transform"
    if gene.role == "observer":
        return "increase_resolution"
    if gene.role == "adapter":
        return "recombine_provider"
    if gene.role == "validator":
        return "tighten_constraint"
    return "mutate_gene"


def gene_border_reason(
    ecology_pressure: float,
    observed: bool,
    activation_rate: float,
    mean_delta_unsat: float,
    puzzle_composition_pressure: float,
    outcome_gap: float,
) -> str:
    reasons: list[str] = [f"ecology_pressure={ecology_pressure:.3f}"]
    if not observed:
        reasons.append("silent_gene")
    elif activation_rate < 0.10:
        reasons.append(f"underactive={activation_rate:.3f}")
    if mean_delta_unsat < -0.25:
        reasons.append(f"harmful_delta={mean_delta_unsat:.3f}")
    if puzzle_composition_pressure > 0.25:
        reasons.append(f"composition_pressure={puzzle_composition_pressure:.3f}")
    if outcome_gap > 0.0:
        reasons.append(f"walksat_gap={outcome_gap:.3f}")
    return ";".join(reasons)


def puzzle_ecology_metrics(
    formula: sat_furnace.CNF,
    variables: int,
    random_best_unsatisfied: int,
    walksat_best_unsatisfied: int,
    furnace_best_unsatisfied: int,
    random_solved: bool,
    walksat_solved: bool,
    furnace_solved: bool,
) -> dict[str, Scalar]:
    """Passive ecology traits for locating border puzzles and composition pressure."""
    clause_count = max(1, len(formula))
    variable_count = max(1, variables)
    literal_count = sum(len(clause) for clause in formula)
    density = clause_count / variable_count
    clause_size_mean = literal_count / clause_count
    positive_literals = sum(
        1 for clause in formula for _variable, is_negated in clause if not is_negated
    )
    negative_literals = literal_count - positive_literals
    polarity_balance = (
        abs(positive_literals - negative_literals) / max(1, literal_count)
    )

    variable_degrees = [0 for _ in range(variable_count)]
    positive_degrees = [0 for _ in range(variable_count)]
    negative_degrees = [0 for _ in range(variable_count)]
    pair_counts: dict[tuple[int, int], int] = {}
    for clause in formula:
        variables_in_clause = sorted({variable for variable, _is_negated in clause})
        for variable, is_negated in clause:
            if 0 <= variable < variable_count:
                variable_degrees[variable] += 1
                if is_negated:
                    negative_degrees[variable] += 1
                else:
                    positive_degrees[variable] += 1
        for left_index, left in enumerate(variables_in_clause):
            for right in variables_in_clause[left_index + 1 :]:
                pair_counts[(left, right)] = pair_counts.get((left, right), 0) + 1

    degree_mean = mean(variable_degrees)
    degree_std = math.sqrt(mean((degree - degree_mean) ** 2 for degree in variable_degrees))
    active_variables = sum(1 for degree in variable_degrees if degree > 0)
    variable_coverage = active_variables / variable_count
    mixed_polarity_variables = sum(
        1 for pos, neg in zip(positive_degrees, negative_degrees) if pos > 0 and neg > 0
    )
    contradiction_pressure = mixed_polarity_variables / variable_count
    repeated_pair_edges = sum(count - 1 for count in pair_counts.values() if count > 1)
    loop_proxy = repeated_pair_edges / max(1, len(pair_counts))

    random_ratio = clamp01(random_best_unsatisfied / clause_count)
    walksat_ratio = clamp01(walksat_best_unsatisfied / clause_count)
    furnace_ratio = clamp01(furnace_best_unsatisfied / clause_count)
    solver_successes = int(random_solved) + int(walksat_solved) + int(furnace_solved)
    frontier_balance = 1.0 - abs((solver_successes / 3.0) - 0.5) * 2.0
    solver_disagreement = (max(random_ratio, walksat_ratio, furnace_ratio) - min(random_ratio, walksat_ratio, furnace_ratio))
    near_miss = 1.0 - min(1.0, min(random_ratio, walksat_ratio, furnace_ratio) * 8.0)
    composition_pressure = clamp01(abs(walksat_ratio - furnace_ratio) * 4.0)
    border_score = clamp01(
        0.40 * frontier_balance
        + 0.25 * solver_disagreement
        + 0.20 * near_miss
        + 0.15 * composition_pressure
    )

    if border_score >= 0.62:
        niche = "border"
    elif solver_successes == 0:
        niche = "resistant"
    elif solver_successes == 3:
        niche = "settled"
    elif composition_pressure >= 0.35:
        niche = "composition_gap"
    else:
        niche = "gradient"

    return {
        "puzzle_density": density,
        "puzzle_clause_size_mean": clause_size_mean,
        "puzzle_polarity_balance": polarity_balance,
        "puzzle_variable_degree_mean": degree_mean,
        "puzzle_variable_degree_std": degree_std,
        "puzzle_variable_coverage": variable_coverage,
        "puzzle_contradiction_pressure": contradiction_pressure,
        "puzzle_loop_proxy": loop_proxy,
        "puzzle_random_unsat_ratio": random_ratio,
        "puzzle_walksat_unsat_ratio": walksat_ratio,
        "puzzle_furnace_unsat_ratio": furnace_ratio,
        "puzzle_solver_disagreement": solver_disagreement,
        "puzzle_frontier_balance": frontier_balance,
        "puzzle_near_miss_signal": near_miss,
        "puzzle_composition_pressure": composition_pressure,
        "puzzle_border_score": border_score,
        "puzzle_ecology_niche": niche,
    }


def trace_operators(traces: Sequence[sat_furnace.OperatorTrace]) -> tuple[str, ...]:
    return tuple(sorted({trace.operator for trace in traces}))


def operator_trace_metrics(
    traces: Sequence[sat_furnace.OperatorTrace],
) -> dict[str, float | str]:
    operators = trace_operators(traces)
    metrics: dict[str, float | str] = {
        "trace_row_count": float(len(traces)),
        "trace_adaptive_activation_rate": 0.0,
        "trace_action_mode": "none",
        "trace_collapse_contribution": 0.0,
        "trace_trap_contribution": 0.0,
        "trace_operator_best": "none",
        "trace_operator_worst": "none",
    }
    if not traces:
        for operator in operators:
            metrics.update(empty_operator_metrics(operator))
        return metrics

    adaptive_gate_traces = [
        trace for trace in traces if trace.operator == "adaptive_gate"
    ]
    active_control_traces = [
        trace for trace in traces if trace.operator == "control_action" and trace.active
    ]
    metrics["trace_adaptive_activation_rate"] = mean(
        1.0 if trace.active else 0.0 for trace in adaptive_gate_traces
    )
    metrics["trace_action_mode"] = (
        mode(trace.action for trace in active_control_traces)
        if active_control_traces
        else "baseline"
    )

    operator_scores: dict[str, float] = {}
    for operator in operators:
        subset = [trace for trace in traces if trace.operator == operator]
        if not subset:
            metrics.update(empty_operator_metrics(operator))
            operator_scores[operator] = 0.0
            continue
        active_subset = [trace for trace in subset if trace.active]
        activation_rate = len(active_subset) / max(1, len(subset))
        mean_delta_unsatisfied = mean(trace.delta_unsatisfied for trace in subset)
        mean_delta_integration = mean(trace.delta_integration for trace in subset)
        mean_output_peak = mean(trace.output_peak for trace in subset)
        mean_memory_scale = mean(trace.memory_scale for trace in subset)
        collapse_contribution = mean(
            positive_part(trace.delta_unsatisfied)
            + positive_part(trace.delta_integration)
            for trace in subset
        )
        trap_contribution = mean(
            positive_part(-trace.delta_unsatisfied)
            + positive_part(-trace.delta_integration)
            for trace in subset
        )
        key = operator_metric_prefix(operator)
        metrics[f"{key}_activation_rate"] = activation_rate
        metrics[f"{key}_mean_delta_unsatisfied"] = mean_delta_unsatisfied
        metrics[f"{key}_mean_delta_integration"] = mean_delta_integration
        metrics[f"{key}_mean_output_peak"] = mean_output_peak
        metrics[f"{key}_mean_memory_scale"] = mean_memory_scale
        metrics[f"{key}_collapse_contribution"] = collapse_contribution
        metrics[f"{key}_trap_contribution"] = trap_contribution
        operator_scores[operator] = collapse_contribution - trap_contribution

    if operator_scores:
        metrics["trace_collapse_contribution"] = mean(
            float(metrics[f"{operator_metric_prefix(operator)}_collapse_contribution"])
            for operator in operators
        )
        metrics["trace_trap_contribution"] = mean(
            float(metrics[f"{operator_metric_prefix(operator)}_trap_contribution"])
            for operator in operators
        )
        metrics["trace_operator_best"] = max(operator_scores, key=operator_scores.get)
        metrics["trace_operator_worst"] = min(operator_scores, key=operator_scores.get)
    return metrics


def empty_operator_metrics(operator: str) -> dict[str, float]:
    key = operator_metric_prefix(operator)
    return {
        f"{key}_activation_rate": 0.0,
        f"{key}_mean_delta_unsatisfied": 0.0,
        f"{key}_mean_delta_integration": 0.0,
        f"{key}_mean_output_peak": 0.0,
        f"{key}_mean_memory_scale": 0.0,
        f"{key}_collapse_contribution": 0.0,
        f"{key}_trap_contribution": 0.0,
    }


def transition_motif_metrics(
    traces: Sequence[sat_furnace.OperatorTrace],
    max_gap: int = 1,
    climate_metrics: Mapping[str, object] | None = None,
) -> dict[str, float | str]:
    """Observe ordered operator motifs without choosing winners.

    This keeps motif data descriptive. Selection belongs to a bootstrapping
    composition layer that can ask what the current landscape needs.
    """
    motifs = transition_motifs(traces, max_gap=max_gap)
    bootstrap_plan = bootstrap_motif_plan(motifs, climate_metrics=climate_metrics)
    pressure_metrics = motif_bootstrap_pressure(bootstrap_plan)
    metrics: dict[str, float | str] = {
        "transition_motif_count": float(len(motifs)),
        "transition_motif_observation_count": 0.0,
        "transition_motif_roles": "none",
        "transition_motif_entropy_shift": 0.0,
        "transition_motif_persistence": 0.0,
        "transition_motif_role_diversity": 0.0,
        "motif_bootstrap_targets": ">".join(bootstrap_plan.targets) or "none",
        "motif_bootstrap_plan": ">".join(bootstrap_plan.order) or "none",
        "motif_bootstrap_missing": ">".join(bootstrap_plan.missing) or "none",
        "motif_bootstrap_provider_count": float(bootstrap_plan.provider_count),
        "motif_bootstrap_provided_effects": ">".join(bootstrap_plan.provided_effects) or "none",
        **pressure_metrics,
    }
    if not motifs:
        return metrics

    role_counts: dict[str, int] = {}
    for motif in motifs:
        role_counts[motif.role] = role_counts.get(motif.role, 0) + 1

    metrics["transition_motif_observation_count"] = float(
        sum(motif.count for motif in motifs)
    )
    metrics["transition_motif_roles"] = ">".join(sorted(role_counts))
    metrics["transition_motif_entropy_shift"] = mean(
        abs(motif.entropy_shift) for motif in motifs
    )
    metrics["transition_motif_persistence"] = mean(
        motif.persistence for motif in motifs
    )
    metrics["transition_motif_role_diversity"] = len(role_counts) / max(1, len(motifs))
    return metrics


def transition_motifs(
    traces: Sequence[sat_furnace.OperatorTrace],
    max_gap: int = 1,
) -> tuple[TransitionMotif, ...]:
    indexed = sorted(enumerate(traces), key=lambda item: (item[1].t, item[0]))
    ordered = [trace for _index, trace in indexed]
    grouped: dict[tuple[str, str], list[tuple[sat_furnace.OperatorTrace, sat_furnace.OperatorTrace]]] = {}
    for left_index, source in enumerate(ordered):
        for target in ordered[left_index + 1 :]:
            gap = target.t - source.t
            if gap < 0:
                continue
            if gap > max_gap:
                break
            if source.operator == target.operator:
                continue
            key = (
                normalize_trace_operator_name(source.operator),
                normalize_trace_operator_name(target.operator),
            )
            grouped.setdefault(key, []).append((source, target))

    motifs: list[TransitionMotif] = []
    for (source_name, target_name), pairs in grouped.items():
        target_active_rate = mean(1.0 if target.active else 0.0 for _source, target in pairs)
        source_delta = mean(source.delta_unsatisfied for source, _target in pairs)
        target_delta = mean(target.delta_unsatisfied for _source, target in pairs)
        source_integration = mean(source.delta_integration for source, _target in pairs)
        target_integration = mean(target.delta_integration for _source, target in pairs)
        entropy_shift = mean(
            target.input_entropy - source.input_entropy for source, target in pairs
        )
        persistence = clamp01(
            mean(
                positive_part(target.delta_unsatisfied)
                + 0.5 * positive_part(target.delta_integration)
                for _source, target in pairs
            )
            / 2.0
        )
        released_tension = clamp01(
            mean(
                positive_part(source.input_unsatisfied - target.input_unsatisfied)
                for source, target in pairs
            )
            / 4.0
        )
        motifs.append(
            TransitionMotif(
                source=source_name,
                target=target_name,
                count=len(pairs),
                activation_rate=target_active_rate,
                mean_delta_unsatisfied=target_delta,
                mean_delta_integration=target_integration,
                entropy_shift=entropy_shift,
                persistence=persistence,
                role=transition_motif_role(entropy_shift, persistence, released_tension),
            )
        )
    return tuple(motifs)


def transition_motif_role(
    entropy_shift: float,
    persistence: float,
    released_tension: float,
) -> str:
    signals = {
        "entropy_shift": entropy_shift,
        "persistence": persistence,
        "released_tension": released_tension,
    }
    return next(
        (
            rule.role
            for rule in MOTIF_ROLE_RULES
            if motif_role_rule_satisfied(rule, signals)
        ),
        "drift",
    )


def motif_role_rule_satisfied(
    rule: MotifRoleRule,
    signals: Mapping[str, float],
) -> bool:
    entropy_shift = signals.get("entropy_shift", 0.0)
    thresholds = (
        signals.get("released_tension", 0.0) >= rule.released_tension,
        signals.get("persistence", 0.0) >= rule.min_persistence,
        rule.entropy_min is None or entropy_shift > rule.entropy_min,
        rule.entropy_max is None or entropy_shift < rule.entropy_max,
    )
    return all(thresholds)


def bootstrap_motif_plan(
    motifs: Sequence[TransitionMotif],
    climate_metrics: Mapping[str, object] | None = None,
) -> MotifBootstrapPlan:
    """Compose motif effects toward landscape needs without ranking motifs."""
    targets = infer_motif_needs_from_climate(motifs, climate_metrics)
    effects = transition_motif_effects(motifs)
    operators = motif_effect_operators(effects)
    composer = Composer(operators)
    plan = composer.plan(targets)
    missing = tuple(item.key for item in plan.missing)
    provider_names = set(plan.order)
    provided_effects = tuple(
        sorted(
            effect
            for operator in operators
            if operator.name in provider_names
            for effect in operator.outputs
        )
    )
    return MotifBootstrapPlan(
        targets=targets,
        order=plan.order,
        missing=missing,
        provided_effects=provided_effects,
        provider_count=len(plan.order),
    )


def motif_bootstrap_pressure(plan: MotifBootstrapPlan) -> dict[str, float | str]:
    missing = set(plan.missing)
    targets = set(plan.targets)
    target_missing = missing & targets
    prerequisite_missing = missing - targets
    pressure_values = {
        f"motif_pressure_{effect}": float(effect in missing)
        for effect in MOTIF_PRESSURE_EFFECTS
    }
    total = clamp01(sum(pressure_values.values()) / max(1, len(pressure_values)))
    return {
        **pressure_values,
        "motif_pressure_total": total,
        "motif_pressure_target_missing_count": float(len(target_missing)),
        "motif_pressure_prerequisite_missing_count": float(len(prerequisite_missing)),
        "motif_pressure_target_missing": ">".join(sorted(target_missing)) or "none",
        "motif_pressure_prerequisite_missing": ">".join(sorted(prerequisite_missing)) or "none",
        "motif_pressure_action_hint": motif_pressure_action_hint(
            missing=missing,
            target_missing=target_missing,
            prerequisite_missing=prerequisite_missing,
        ),
    }


def motif_pressure_action_hint(
    missing: set[str],
    target_missing: set[str] | None = None,
    prerequisite_missing: set[str] | None = None,
) -> str:
    source_sets = {
        "target": target_missing or set(),
        "prerequisite": prerequisite_missing or set(),
        "any": missing,
    }
    return next(
        (
            rule.hint
            for rule in MOTIF_HINT_RULES
            if rule.effect in source_sets.get(rule.source, set())
        ),
        "none",
    )


def infer_motif_needs_from_climate(
    motifs: Sequence[TransitionMotif],
    climate_metrics: Mapping[str, object] | None = None,
) -> tuple[str, ...]:
    if climate_metrics is None:
        return infer_motif_needs(motifs)

    climate = motif_climate_signals(climate_metrics)
    needs = needs_from_rules(CLIMATE_NEED_RULES, climate)
    if not needs:
        needs.extend(infer_motif_needs(motifs))
    return tuple(dict.fromkeys(needs))


def motif_climate_signals(climate_metrics: Mapping[str, object]) -> dict[str, float]:
    niche = str(climate_metrics.get("puzzle_ecology_niche", ""))
    border_score = metric_float(climate_metrics, "puzzle_border_score")
    loop_score = metric_float(climate_metrics, "runner_mean_loop_score")
    path_novelty = metric_float(climate_metrics, "runner_mean_path_novelty")
    collapse_contribution = metric_float(climate_metrics, "trace_collapse_contribution")
    solved = metric_bool(climate_metrics, "solved")
    return {
        "is_border_niche": float(niche == "border"),
        "is_gradient_niche": float(niche == "gradient" and border_score < 0.45),
        "puzzle_border_score": border_score,
        "puzzle_composition_pressure": metric_float(climate_metrics, "puzzle_composition_pressure"),
        "trace_trap_contribution": metric_float(climate_metrics, "trace_trap_contribution"),
        "loop_stagnation": loop_score * positive_part(0.25 - path_novelty) / 0.25,
        "unsolved_collapse": collapse_contribution * float(not solved),
    }


def needs_from_rules(
    rules: Sequence[MotifNeedRule],
    signals: Mapping[str, float],
) -> list[str]:
    return [
        rule.need
        for rule in rules
        if rule_satisfied(rule, signals)
    ]


def rule_satisfied(rule: MotifNeedRule, signals: Mapping[str, float]) -> bool:
    score = max((signals.get(signal, 0.0) for signal in rule.signals), default=0.0)
    return (
        score >= rule.threshold
        if rule.polarity == "above"
        else score <= rule.threshold
    )


def metric_float(metrics: Mapping[str, object], key: str) -> float:
    try:
        return float(metrics.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def metric_bool(metrics: Mapping[str, object], key: str) -> bool:
    value = metrics.get(key, False)
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def infer_motif_needs(motifs: Sequence[TransitionMotif]) -> tuple[str, ...]:
    if not motifs:
        return ("motif_observation",)

    role_set = {motif.role for motif in motifs}
    signals = {
        "mean_persistence": mean(motif.persistence for motif in motifs),
        "mean_entropy_shift": mean(motif.entropy_shift for motif in motifs),
    }
    needs = [
        rule.need
        for rule in MOTIF_NEED_RULES
        if rule.absent_role not in role_set
        and fallback_need_rule_satisfied(rule, signals)
    ]
    return tuple(dict.fromkeys(needs or ["stabilization_window"]))


def fallback_need_rule_satisfied(
    rule: MotifFallbackNeedRule,
    signals: Mapping[str, float],
) -> bool:
    signal = signals.get(rule.signal, 0.0)
    return (
        signal <= rule.threshold
        if rule.polarity == "below"
        else signal >= rule.threshold
    )


def transition_motif_effects(
    motifs: Sequence[TransitionMotif],
) -> tuple[MotifEffect, ...]:
    return tuple(
        MotifEffect(
            motif=f"{motif.source}->{motif.target}",
            role=motif.role,
            requires=motif_effect_requirements(motif.role),
            provides=motif_effect_outputs(motif.role),
            confidence=clamp01(
                0.25
                + 0.25 * motif.activation_rate
                + 0.25 * motif.persistence
                + 0.05 * min(5, motif.count)
            ),
            evidence=float(motif.count),
        )
        for motif in motifs
    )


def motif_effect_requirements(role: str) -> tuple[str, ...]:
    return MOTIF_ROLE_EFFECT_BY_ROLE.get(
        role,
        MOTIF_ROLE_EFFECT_BY_ROLE["drift"],
    ).requires


def motif_effect_outputs(role: str) -> tuple[str, ...]:
    return MOTIF_ROLE_EFFECT_BY_ROLE.get(
        role,
        MOTIF_ROLE_EFFECT_BY_ROLE["drift"],
    ).provides


def motif_effect_operators(
    effects: Sequence[MotifEffect],
) -> tuple[FieldOperator, ...]:
    operators: list[FieldOperator] = []
    provided_outputs: set[str] = set()
    for index, effect in enumerate(effects):
        outputs = tuple(output for output in effect.provides if output not in provided_outputs)
        if not outputs:
            continue
        provided_outputs.update(outputs)
        name = f"motif.{index}.{effect.role}.{effect.motif}"
        operators.append(
            FieldOperator(
                name=name,
                inputs=effect.requires,
                outputs=outputs,
                run=lambda _ctx, motif_effect=effect, outputs=outputs: {
                    output: motif_effect.motif for output in outputs
                },
            )
        )
    return tuple(operators)




def operator_metric_prefix(operator: str) -> str:
    return f"trace_{operator.lstrip('_')}"


def excitable_trace_chain_snapshot(
    traces: Sequence[sat_furnace.OperatorTrace],
) -> dict[str, float | bool]:
    latest = latest_traces_by_name(
        traces,
        (
            "excitable_concentration",
            "excitable_field",
            "excitable_spike",
            "excitable_mixture",
            "spin_update",
        ),
    )
    concentration = latest.get("excitable_concentration")
    field = latest.get("excitable_field")
    spike = latest.get("excitable_spike")
    mixture = latest.get("excitable_mixture")
    spin = latest.get("spin_update")
    return {
        "chain_state_heat": float(spin.input_heat) if spin else 0.0,
        "chain_state_entropy": float(spin.input_entropy) if spin else 0.0,
        "chain_state_integration": float(spin.input_integration) if spin else 0.0,
        "chain_state_unsatisfied": float(spin.input_unsatisfied) if spin else 0.0,
        "chain_concentration": float(concentration.output_mean)
        if concentration
        else 0.0,
        "chain_excitation_inhibition": float(field.output_mean) if field else 0.0,
        "chain_spike_strength": float(spike.output_mean) if spike else 0.0,
        "chain_blended_action": float(mixture.output_mean) if mixture else 0.0,
        "chain_delta_unsat": float(spin.delta_unsatisfied) if spin else 0.0,
        "chain_complete": bool(
            concentration and field and spike and mixture and spin
        ),
    }


def latest_traces_by_name(
    traces: Sequence[sat_furnace.OperatorTrace],
    names: Sequence[str],
) -> dict[str, sat_furnace.OperatorTrace]:
    wanted = set(names)
    latest: dict[str, sat_furnace.OperatorTrace] = {}
    for trace in traces:
        if trace.operator in wanted:
            latest[trace.operator] = trace
    return latest


def choice_policy_metrics(
    windows: Sequence[spectral_calorimeter.WindowSpectrum],
    runners: Sequence[sprite_detector.GraphRunner],
) -> dict[str, float | str]:
    """Summarize choices implied by surplus/deficit imbalance."""
    if not windows:
        return empty_choice_policy_metrics()
    latest = windows[-1]
    recent = windows[-max(1, min(5, len(windows))) :]
    latest_runners = sorted(runners, key=lambda runner: runner.death_t, reverse=True)[
        : max(1, len(runners) // 3)
    ]
    entropy_excess = clamp01(latest.spectral_entropy)
    fragmentation_excess = clamp01(latest.fragmentation_index)
    recycling_excess = clamp01(latest.recycling_score)
    heat_excess = clamp01(mean(window.mean_heat for window in recent))
    loop_mass_excess = (
        clamp01(
            mean(runner.loop_score * runner.mass_density for runner in latest_runners)
        )
        if latest_runners
        else 0.0
    )
    integration_deficit = clamp01(1.0 - latest.mean_integration)
    redshift_deficit = clamp01(
        slope_deficit(
            [window.spectral_centroid for window in recent], wants_negative=True
        )
    )
    novelty_deficit = (
        clamp01(1.0 - mean(runner.path_novelty for runner in latest_runners))
        if latest_runners
        else 1.0
    )
    bridge_deficit = (
        clamp01(1.0 - mean(runner.bridge_score for runner in latest_runners))
        if latest_runners
        else 1.0
    )
    collapse_deficit = clamp01(1.0 - latest.collapse_index)
    stabilize_need = clamp01(
        latest.collapse_index
        * max(
            0.0,
            -spectral_calorimeter.slope(
                [window.center_t for window in recent],
                [window.spectral_entropy for window in recent],
            ),
        )
    )
    actions = {
        "explore": mean([loop_mass_excess, novelty_deficit, recycling_excess]),
        "bridge": mean([fragmentation_excess, bridge_deficit, integration_deficit]),
        "cool": mean([entropy_excess, heat_excess, redshift_deficit]),
        "stabilize": mean(
            [stabilize_need, latest.collapse_index, 1.0 - entropy_excess]
        ),
        "perturb": mean([loop_mass_excess, recycling_excess, redshift_deficit]),
    }
    action, action_score = max(actions.items(), key=lambda item: item[1])
    surplus_pressure = mean(
        [
            entropy_excess,
            fragmentation_excess,
            recycling_excess,
            heat_excess,
            loop_mass_excess,
        ]
    )
    deficit_pressure = mean(
        [
            integration_deficit,
            redshift_deficit,
            novelty_deficit,
            bridge_deficit,
            collapse_deficit,
        ]
    )
    imbalance_pressure = surplus_pressure + deficit_pressure
    policy_confidence = action_score / max(1e-12, sum(actions.values()))
    return {
        "choice_entropy_excess": entropy_excess,
        "choice_fragmentation_excess": fragmentation_excess,
        "choice_recycling_excess": recycling_excess,
        "choice_heat_excess": heat_excess,
        "choice_loop_mass_excess": loop_mass_excess,
        "choice_integration_deficit": integration_deficit,
        "choice_redshift_deficit": redshift_deficit,
        "choice_novelty_deficit": novelty_deficit,
        "choice_bridge_deficit": bridge_deficit,
        "choice_collapse_deficit": collapse_deficit,
        "choice_surplus_pressure": surplus_pressure,
        "choice_deficit_pressure": deficit_pressure,
        "choice_imbalance_pressure": imbalance_pressure,
        "choice_explore_score": actions["explore"],
        "choice_bridge_score": actions["bridge"],
        "choice_cool_score": actions["cool"],
        "choice_stabilize_score": actions["stabilize"],
        "choice_perturb_score": actions["perturb"],
        "choice_action": action,
        "choice_confidence": policy_confidence,
    }


def empty_choice_policy_metrics() -> dict[str, float | str]:
    return {
        "choice_entropy_excess": 0.0,
        "choice_fragmentation_excess": 0.0,
        "choice_recycling_excess": 0.0,
        "choice_heat_excess": 0.0,
        "choice_loop_mass_excess": 0.0,
        "choice_integration_deficit": 0.0,
        "choice_redshift_deficit": 0.0,
        "choice_novelty_deficit": 0.0,
        "choice_bridge_deficit": 0.0,
        "choice_collapse_deficit": 0.0,
        "choice_surplus_pressure": 0.0,
        "choice_deficit_pressure": 0.0,
        "choice_imbalance_pressure": 0.0,
        "choice_explore_score": 0.0,
        "choice_bridge_score": 0.0,
        "choice_cool_score": 0.0,
        "choice_stabilize_score": 0.0,
        "choice_perturb_score": 0.0,
        "choice_action": "none",
        "choice_confidence": 0.0,
    }


def slope_deficit(values: Sequence[float], wants_negative: bool) -> float:
    trend = spectral_calorimeter.slope(list(range(len(values))), values)
    return trend if wants_negative else -trend


def clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def coupling_metrics(
    windows: Sequence[spectral_calorimeter.WindowSpectrum],
    spatial_samples: Sequence[sprite_detector.SpatialSample],
    runners: Sequence[sprite_detector.GraphRunner],
    window_size: int,
    step_size: int,
) -> dict[str, float]:
    """Measure bidirectional coupling between local motion and global renormalization."""
    if not windows:
        return empty_coupling_metrics()
    local_windows = local_window_series(
        spatial_samples, runners, windows, window_size=window_size, step_size=step_size
    )
    global_collapse = [window.collapse_index for window in windows]
    global_redshift = signed_deltas(
        [window.spectral_centroid for window in windows], invert=True
    )
    global_entropy_drop = signed_deltas(
        [window.spectral_entropy for window in windows], invert=True
    )
    global_integration_gain = signed_deltas(
        [window.mean_integration for window in windows], invert=False
    )
    local_escape = [window["escape"] for window in local_windows]
    local_novelty = [window["novelty"] for window in local_windows]
    local_loop = [window["loop"] for window in local_windows]
    local_mass = [window["mass"] for window in local_windows]

    forward_escape_to_collapse = lagged_correlation(
        local_escape, signed_deltas(global_collapse), lag=1
    )
    forward_novelty_to_redshift = lagged_correlation(
        local_novelty, global_redshift, lag=1
    )
    forward_loop_to_trap = lagged_correlation(local_loop, local_mass, lag=1)
    renormalization_gain = lagged_correlation(global_collapse, local_escape, lag=1)
    global_bias_to_novelty = lagged_correlation(
        global_entropy_drop, local_novelty, lag=1
    )
    integration_feedback = lagged_correlation(
        global_integration_gain, local_escape, lag=1
    )
    bidirectional_coupling = mean(
        value
        for value in [
            positive_part(forward_escape_to_collapse),
            positive_part(forward_novelty_to_redshift),
            positive_part(renormalization_gain),
            positive_part(global_bias_to_novelty),
            positive_part(integration_feedback),
        ]
    )
    trap_reinforcement = mean(
        value
        for value in [
            positive_part(forward_loop_to_trap),
            positive_part(
                lagged_correlation(
                    local_loop, [1.0 - value for value in local_novelty], lag=1
                )
            ),
        ]
    )
    metrics = {
        "forward_escape_to_collapse": forward_escape_to_collapse,
        "forward_novelty_to_redshift": forward_novelty_to_redshift,
        "forward_loop_to_trap": forward_loop_to_trap,
        "renormalization_gain": renormalization_gain,
        "global_bias_to_novelty": global_bias_to_novelty,
        "integration_feedback": integration_feedback,
        "bidirectional_coupling_score": bidirectional_coupling,
        "trap_reinforcement_score": trap_reinforcement,
    }
    metrics.update(phase_transition_metrics(local_windows, windows))
    return metrics


def empty_coupling_metrics() -> dict[str, float]:
    return {
        "forward_escape_to_collapse": 0.0,
        "forward_novelty_to_redshift": 0.0,
        "forward_loop_to_trap": 0.0,
        "renormalization_gain": 0.0,
        "global_bias_to_novelty": 0.0,
        "integration_feedback": 0.0,
        "bidirectional_coupling_score": 0.0,
        "trap_reinforcement_score": 0.0,
        "phase_chain_score": 0.0,
        "phase_novelty_onset_t": -1.0,
        "phase_redshift_onset_t": -1.0,
        "phase_integration_onset_t": -1.0,
        "phase_collapse_onset_t": -1.0,
        "phase_ordered": 0.0,
        "phase_span": 0.0,
    }


def phase_transition_metrics(
    local_windows: Sequence[dict[str, float]],
    windows: Sequence[spectral_calorimeter.WindowSpectrum],
) -> dict[str, float]:
    novelty = [window["novelty"] for window in local_windows]
    redshift = signed_deltas(
        [window.spectral_centroid for window in windows], invert=True
    )
    integration_gain = signed_deltas(
        [window.mean_integration for window in windows], invert=False
    )
    collapse_gain = signed_deltas(
        [window.collapse_index for window in windows], invert=False
    )
    novelty_onset = onset_index(novelty, high=True)
    redshift_onset = onset_index(redshift, high=True)
    integration_onset = onset_index(integration_gain, high=True)
    collapse_onset = onset_index(collapse_gain, high=True)
    ordered = ordered_onsets(
        [novelty_onset, redshift_onset, integration_onset, collapse_onset]
    )
    phase_span = (
        float(
            max([novelty_onset, redshift_onset, integration_onset, collapse_onset])
            - min([novelty_onset, redshift_onset, integration_onset, collapse_onset])
        )
        if ordered
        else 0.0
    )
    novelty_strength = onset_strength(novelty, novelty_onset)
    redshift_strength = onset_strength(redshift, redshift_onset)
    integration_strength = onset_strength(integration_gain, integration_onset)
    collapse_strength = onset_strength(collapse_gain, collapse_onset)
    phase_chain_score = mean(
        value
        for value in [
            float(ordered),
            positive_part(novelty_strength),
            positive_part(redshift_strength),
            positive_part(integration_strength),
            positive_part(collapse_strength),
        ]
    )
    return {
        "phase_chain_score": phase_chain_score,
        "phase_novelty_onset_t": onset_time(windows, novelty_onset),
        "phase_redshift_onset_t": onset_time(windows, redshift_onset),
        "phase_integration_onset_t": onset_time(windows, integration_onset),
        "phase_collapse_onset_t": onset_time(windows, collapse_onset),
        "phase_ordered": float(ordered),
        "phase_span": phase_span,
    }


def onset_index(values: Sequence[float], high: bool) -> int:
    if not values:
        return -1
    threshold = robust_threshold(values, high=high)
    candidates = [
        index
        for index, value in enumerate(values)
        if (value >= threshold if high else value <= threshold)
    ]
    if not candidates:
        return -1
    return candidates[0]


def onset_time(
    windows: Sequence[spectral_calorimeter.WindowSpectrum], index: int
) -> float:
    if index < 0 or index >= len(windows):
        return -1.0
    return float(windows[index].center_t)


def ordered_onsets(indices: Sequence[int]) -> bool:
    if any(index < 0 for index in indices):
        return False
    return all(before <= after for before, after in zip(indices, indices[1:]))


def onset_strength(values: Sequence[float], index: int) -> float:
    if index < 0 or index >= len(values):
        return 0.0
    baseline = (
        statistics.fmean(values[: max(1, index)])
        if index > 0
        else statistics.fmean(values)
    )
    spread = statistics.pstdev(values) if len(values) > 1 else 0.0
    if spread <= 1e-12:
        return 0.0
    return (values[index] - baseline) / spread


def robust_threshold(values: Sequence[float], high: bool) -> float:
    if not values:
        return 0.0
    center = statistics.median(values)
    deviations = [abs(value - center) for value in values]
    mad = statistics.median(deviations) if deviations else 0.0
    spread = max(
        mad * 1.4826, statistics.pstdev(values) if len(values) > 1 else 0.0, 1e-12
    )
    return center + spread if high else center - spread


def local_window_series(
    spatial_samples: Sequence[sprite_detector.SpatialSample],
    runners: Sequence[sprite_detector.GraphRunner],
    windows: Sequence[spectral_calorimeter.WindowSpectrum],
    window_size: int,
    step_size: int,
) -> list[dict[str, float]]:
    samples_by_time: dict[int, list[sprite_detector.SpatialSample]] = {}
    for sample in spatial_samples:
        samples_by_time.setdefault(sample.t, []).append(sample)
    all_times = sorted(samples_by_time)
    if not all_times:
        return [
            {"escape": 0.0, "novelty": 0.0, "loop": 0.0, "mass": 0.0} for _ in windows
        ]
    series: list[dict[str, float]] = []
    for index, global_window in enumerate(windows):
        start_t = int(round(global_window.start_t))
        end_t = int(round(global_window.end_t))
        midpoint_t = int(round(global_window.center_t))
        frame_samples = [
            sample
            for t in range(start_t, end_t + 1)
            for sample in samples_by_time.get(t, [])
        ]
        active_runners = [
            runner
            for runner in runners
            if runner.birth_t <= end_t and runner.death_t >= start_t
        ]
        if active_runners:
            escape = mean(runner.escape_score for runner in active_runners)
            novelty = mean(runner.path_novelty for runner in active_runners)
            loop = mean(runner.loop_score for runner in active_runners)
            mass = mean(runner.mass_density for runner in active_runners)
        elif frame_samples:
            escape = 0.0
            novelty = mean(abs(sample.pressure) for sample in frame_samples)
            loop = mean(sample.entropy for sample in frame_samples)
            mass = mean(
                sprite_detector.sprite_intensity(sample) for sample in frame_samples
            )
        else:
            escape = novelty = loop = mass = 0.0
        series.append(
            {
                "escape": escape,
                "novelty": novelty,
                "loop": loop,
                "mass": mass,
                "t": midpoint_t + index * 0.0 + step_size * 0.0 + window_size * 0.0,
            }
        )
    return series


def signed_deltas(values: Sequence[float], invert: bool = False) -> list[float]:
    if not values:
        return []
    deltas = [0.0]
    for before, after in zip(values, values[1:]):
        delta = after - before
        deltas.append(-delta if invert else delta)
    return deltas


def lagged_correlation(
    source: Sequence[float], target: Sequence[float], lag: int = 1
) -> float:
    if lag < 0:
        raise ValueError("lag must be non-negative")
    if len(source) <= lag or len(target) <= lag:
        return 0.0
    return correlation(
        source[:-lag] if lag else source, target[lag:] if lag else target
    )


def correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 3:
        return 0.0
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denominator = math.sqrt(var_x * var_y)
    if denominator <= 1e-12:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator


def positive_part(value: float) -> float:
    return max(0.0, value)


def count_by_classification(items: Sequence[object], classification: str) -> int:
    return sum(1 for item in items if getattr(item, "classification") == classification)


def mean(values) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def mode(values) -> str:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts, key=counts.get) if counts else "none"


def summary_value(values) -> str:
    values = list(values)
    if not values:
        return "n/a"
    numeric_values = [float(value) for value in values]
    if all(value == numeric_values[0] for value in numeric_values):
        return format_number(numeric_values[0])
    return f"{format_number(min(numeric_values))}-{format_number(max(numeric_values))}"


def format_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.6g}"


def write_rows(path: Path, rows: Sequence[dict[str, Scalar]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: Sequence[dict[str, Scalar]]) -> None:
    print(f"trial_count: {len(rows)}")
    for kind in KINDS:
        subset = [row for row in rows if row["kind"] == kind]
        if not subset:
            continue
        print(f"kind: {kind}")
        print(f"  trials: {len(subset)}")
        print(f"  n: {summary_value(row['variables'] for row in subset)}")
        print(f"  clauses: {summary_value(row['clauses'] for row in subset)}")
        print(f"  steps: {summary_value(row['steps'] for row in subset)}")
        print(
            f"  solved_rate: {mean(1.0 if row['solved'] else 0.0 for row in subset):.3f}"
        )
        print(
            f"  furnace_best_unsatisfied: {mean(float(row['furnace_best_unsatisfied']) for row in subset):.3f}"
        )
        print(
            f"  furnace_final_assignment_unsatisfied: {mean(float(row['furnace_final_assignment_unsatisfied']) for row in subset):.3f}"
        )
        print(
            f"  random_solved_rate: {mean(1.0 if row['random_solved'] else 0.0 for row in subset):.3f}"
        )
        print(
            f"  random_best_unsatisfied: {mean(float(row['random_best_unsatisfied']) for row in subset):.3f}"
        )
        print(
            f"  walksat_solved_rate: {mean(1.0 if row['walksat_solved'] else 0.0 for row in subset):.3f}"
        )
        print(
            f"  walksat_best_unsatisfied: {mean(float(row['walksat_best_unsatisfied']) for row in subset):.3f}"
        )
        print(
            f"  furnace_and_walksat_solved: {sum(1 for row in subset if row['solved'] and row['walksat_solved'])}"
        )
        print(
            f"  furnace_only_solved: {sum(1 for row in subset if row['solved'] and not row['walksat_solved'])}"
        )
        print(
            f"  walksat_only_solved: {sum(1 for row in subset if row['walksat_solved'] and not row['solved'])}"
        )
        print(
            f"  neither_solved: {sum(1 for row in subset if not row['solved'] and not row['walksat_solved'])}"
        )
        print(
            f"  mean_furnace_minus_walksat_best_unsatisfied: {mean(float(row['furnace_best_unsatisfied']) - float(row['walksat_best_unsatisfied']) for row in subset):.3f}"
        )
        print(
            f"  adaptive_rate: {mean(1.0 if row.get('adaptive') else 0.0 for row in subset):.3f}"
        )
        print(
            f"  calorimeter_accuracy: {mean(1.0 if normalized_prediction(row['prediction']) == kind else 0.0 for row in subset):.3f}"
        )
        print(
            f"  global_collapse_score: {mean(float(row['global_collapse_score']) for row in subset):.6f}"
        )
        print(
            f"  runner_ecology_score: {mean(float(row['runner_ecology_score']) for row in subset):.6f}"
        )
        print(
            f"  unsat_trap_score: {mean(float(row['unsat_trap_score']) for row in subset):.6f}"
        )
        print(
            f"  runner_count: {mean(float(row['graph_runner_count']) for row in subset):.3f}"
        )
        print(
            f"  path_novelty: {mean(float(row['runner_mean_path_novelty']) for row in subset):.6f}"
        )
        print(
            f"  mass_density: {mean(float(row['runner_mean_mass_density']) for row in subset):.6f}"
        )
        print(
            f"  loop_score: {mean(float(row['runner_mean_loop_score']) for row in subset):.6f}"
        )
        print(
            f"  escape_score: {mean(float(row['runner_mean_escape_score']) for row in subset):.6f}"
        )
        print(
            f"  bidirectional_coupling_score: {mean(float(row['bidirectional_coupling_score']) for row in subset):.6f}"
        )
        print(
            f"  trap_reinforcement_score: {mean(float(row['trap_reinforcement_score']) for row in subset):.6f}"
        )
        print(
            f"  phase_chain_score: {mean(float(row['phase_chain_score']) for row in subset):.6f}"
        )
        print(
            f"  phase_ordered_rate: {mean(float(row['phase_ordered']) for row in subset):.3f}"
        )
        print(f"  phase_span: {mean(float(row['phase_span']) for row in subset):.3f}")
        print(f"  choice_action: {mode(str(row['choice_action']) for row in subset)}")
        print(
            f"  choice_imbalance_pressure: {mean(float(row['choice_imbalance_pressure']) for row in subset):.6f}"
        )
        print(
            f"  choice_surplus_pressure: {mean(float(row['choice_surplus_pressure']) for row in subset):.6f}"
        )
        print(
            f"  choice_deficit_pressure: {mean(float(row['choice_deficit_pressure']) for row in subset):.6f}"
        )
        print(
            f"  choice_explore_score: {mean(float(row['choice_explore_score']) for row in subset):.6f}"
        )
        print(
            f"  choice_bridge_score: {mean(float(row['choice_bridge_score']) for row in subset):.6f}"
        )
        print(
            f"  choice_perturb_score: {mean(float(row['choice_perturb_score']) for row in subset):.6f}"
        )
        print(
            f"  forward_escape_to_collapse: {mean(float(row['forward_escape_to_collapse']) for row in subset):.6f}"
        )
        print(
            f"  renormalization_gain: {mean(float(row['renormalization_gain']) for row in subset):.6f}"
        )
        print(
            f"  trace_adaptive_activation_rate: {mean(float(row['trace_adaptive_activation_rate']) for row in subset):.6f}"
        )
        print(
            f"  trace_action_mode: {mode(str(row['trace_action_mode']) for row in subset)}"
        )
        print(
            f"  trace_collapse_contribution: {mean(float(row['trace_collapse_contribution']) for row in subset):.6f}"
        )
        print(
            f"  trace_trap_contribution: {mean(float(row['trace_trap_contribution']) for row in subset):.6f}"
        )
        print(
            f"  trace_operator_best: {mode(str(row['trace_operator_best']) for row in subset)}"
        )
        print(
            f"  trace_operator_worst: {mode(str(row['trace_operator_worst']) for row in subset)}"
        )
        print(
            f"  trace_fiber_memory_activation_rate: {mean(float(row['trace_fiber_memory_bias_activation_rate']) for row in subset):.6f}"
        )
        print(
            f"  trace_spin_update_delta_unsatisfied: {mean(float(row['trace_spin_update_mean_delta_unsatisfied']) for row in subset):.6f}"
        )
        print(
            f"  reinforced_loop_count: {mean(float(row['reinforced_loop_count']) for row in subset):.3f}"
        )
        print(
            f"  exploratory_runner_count: {mean(float(row['exploratory_runner_count']) for row in subset):.3f}"
        )


def normalized_prediction(prediction: object) -> str:
    if prediction == "Hard SAT":
        return "hard_sat"
    return str(prediction).lower()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark SAT furnace diagnostics over many seeds."
    )
    parser.add_argument("--trials", type=int, default=5, help="Trials per kind.")
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--variables", type=int, default=32)
    parser.add_argument("--clauses", type=int, default=136)
    parser.add_argument("--clause-size", type=int, default=3)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--learning-rate", type=float, default=0.055)
    parser.add_argument("--inertia", type=float, default=0.82)
    parser.add_argument("--noise", type=float, default=0.015)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument("--runner-quantile", type=float, default=0.92)
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Run furnace with surplus/deficit adaptive controls.",
    )
    parser.add_argument(
        "--compare-adaptive",
        action="store_true",
        help="Run each seed twice: baseline and adaptive.",
    )
    parser.add_argument(
        "--policy",
        choices=("baseline", "excitable_fiber", "curriculum_seeds"),
        default="baseline",
        help="Motion policy membrane: baseline, excitable fiber, or curriculum seeds.",
    )
    parser.add_argument(
        "--compare-policies",
        action="store_true",
        help="Run each seed across baseline, excitable_fiber, and curriculum_seeds.",
    )
    parser.add_argument(
        "--spike-threshold",
        type=float,
        default=0.35,
        help="Excitable fiber threshold theta for spike activation.",
    )
    parser.add_argument(
        "--spike-slope",
        type=float,
        default=8.0,
        help="Excitable fiber sigmoid slope for spike activation.",
    )
    parser.add_argument(
        "--memory-decay",
        type=float,
        default=0.92,
        help="Exponential decay for adaptive fiber-bundle memory.",
    )
    parser.add_argument(
        "--memory-drive",
        type=float,
        default=0.12,
        help="Strength of adaptive fiber-bundle memory bias.",
    )
    parser.add_argument(
        "--baseline-restarts",
        type=int,
        default=256,
        help="Random-assignment baseline restarts per trial.",
    )
    parser.add_argument("--out", type=Path, default=Path("benchmark.csv"))
    parser.add_argument(
        "--trace-out",
        type=Path,
        default=None,
        help="Optional CSV path for per-checkpoint furnace and WalkSAT traces.",
    )
    parser.add_argument(
        "--trace-checkpoints",
        type=float,
        nargs="*",
        default=list(DEFAULT_TRACE_CHECKPOINTS),
        help="Budget fractions to emit when --trace-out is set.",
    )
    args = parser.parse_args()

    rows: list[dict[str, Scalar]] = []
    trace_rows: list[dict[str, Scalar]] = []
    trace_sink = trace_rows if args.trace_out is not None else None
    trace_checkpoints = normalized_trace_checkpoints(args.trace_checkpoints)
    adaptive_modes = [False, True] if args.compare_adaptive else [args.adaptive]
    policies = (
        ["baseline", "excitable_fiber", "curriculum_seeds"]
        if args.compare_policies
        else [args.policy]
    )
    for kind in args.kinds:
        for trial in range(args.trials):
            seed = args.seed_start + trial
            for adaptive in adaptive_modes:
                for policy in policies:
                    rows.append(
                        run_trial(
                            kind=kind,
                            seed=seed,
                            variables=args.variables,
                            clauses=args.clauses,
                            clause_size=args.clause_size,
                            steps=args.steps,
                            temperature=args.temperature,
                            learning_rate=args.learning_rate,
                            inertia=args.inertia,
                            noise=args.noise,
                            window=args.window,
                            step_size=args.step,
                            runner_quantile=args.runner_quantile,
                            adaptive=adaptive,
                            policy=policy,
                            spike_threshold=args.spike_threshold,
                            spike_slope=args.spike_slope,
                            memory_decay=args.memory_decay,
                            memory_drive=args.memory_drive,
                            baseline_restarts=args.baseline_restarts,
                            trace_rows=trace_sink,
                            trace_checkpoints=trace_checkpoints,
                        )
                    )
                    print(
                        f"completed kind={kind} seed={seed} adaptive={adaptive} policy={policy}"
                    )
    write_rows(args.out, rows)
    print(f"wrote: {args.out}")
    if args.trace_out is not None:
        write_rows(args.trace_out, trace_rows)
        print(f"wrote trace: {args.trace_out}")
    print_summary(rows)


if __name__ == "__main__":
    main()
