#!/usr/bin/env python3
"""Benchmark SAT furnace trajectories with spectral and sprite diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from pathlib import Path
from typing import Sequence

import sat_composer
import sat_furnace
import spectral_calorimeter
import sprite_detector

KINDS = ("sat", "unsat", "hard_sat")
DEFAULT_TRACE_CHECKPOINTS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00)
Scalar = float | int | str | bool


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


def operator_metric_prefix(operator: str) -> str:
    return f"trace_{operator}"


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
        choices=("baseline", "excitable_fiber"),
        default="baseline",
        help="Motion policy membrane: baseline or excitable fiber.",
    )
    parser.add_argument(
        "--compare-policies",
        action="store_true",
        help="Run each seed twice: baseline and excitable_fiber.",
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
    policies = ["baseline", "excitable_fiber"] if args.compare_policies else [args.policy]
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
