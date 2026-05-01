#!/usr/bin/env python3
"""Dataset-born curriculum seeds for excitable SAT operator mixing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


EFFECT_BASIS = ("pressure", "bridge", "loop_escape", "memory")
CURRICULUM_SEED_POLICY = "curriculum_seeds"


@dataclass(frozen=True)
class SolverSeedFeatures:
    progress: float
    unsat_ratio: float
    entropy: float
    heat: float
    integration: float
    stagnation: float
    revisit: float
    loop_pressure: float
    memory_pressure: float


@dataclass(frozen=True)
class CurriculumSeed:
    name: str
    dataset_slice: str
    centroid: SolverSeedFeatures
    concentration_prior: tuple[float, float, float, float]
    excitatory_bias: float
    inhibitory_bias: float
    threshold: float


@dataclass(frozen=True)
class SeedRouting:
    active_seed: str
    active_index: int
    seed_names: tuple[str, ...]
    weights: tuple[float, ...]
    concentration_prior: tuple[float, float, float, float]
    excitatory_bias: float
    inhibitory_bias: float
    threshold: float


SEEDS: tuple[CurriculumSeed, ...] = (
    CurriculumSeed(
        name="bootstrap",
        dataset_slice="early random / cold-start states",
        centroid=SolverSeedFeatures(0.05, 0.45, 0.90, 0.45, 0.25, 0.05, 0.05, 0.20, 0.05),
        concentration_prior=(0.30, 0.25, 0.25, 0.20),
        excitatory_bias=0.08,
        inhibitory_bias=-0.03,
        threshold=0.28,
    ),
    CurriculumSeed(
        name="density",
        dataset_slice="high unsatisfied-clause density states",
        centroid=SolverSeedFeatures(0.30, 0.70, 0.55, 0.75, 0.35, 0.25, 0.20, 0.40, 0.15),
        concentration_prior=(0.58, 0.18, 0.14, 0.10),
        excitatory_bias=0.10,
        inhibitory_bias=0.00,
        threshold=0.34,
    ),
    CurriculumSeed(
        name="trapbreak",
        dataset_slice="high revisit / repeated-state regimes",
        centroid=SolverSeedFeatures(0.45, 0.35, 0.45, 0.45, 0.45, 0.85, 0.85, 0.75, 0.55),
        concentration_prior=(0.12, 0.18, 0.50, 0.20),
        excitatory_bias=0.14,
        inhibitory_bias=-0.05,
        threshold=0.30,
    ),
    CurriculumSeed(
        name="entropy_shaping",
        dataset_slice="high-entropy exploration regimes",
        centroid=SolverSeedFeatures(0.40, 0.40, 0.88, 0.50, 0.35, 0.35, 0.25, 0.35, 0.20),
        concentration_prior=(0.24, 0.36, 0.24, 0.16),
        excitatory_bias=0.06,
        inhibitory_bias=0.02,
        threshold=0.36,
    ),
    CurriculumSeed(
        name="oscillation_damping",
        dataset_slice="oscillatory / non-progressing states",
        centroid=SolverSeedFeatures(0.55, 0.30, 0.62, 0.40, 0.50, 0.70, 0.65, 0.65, 0.35),
        concentration_prior=(0.18, 0.25, 0.35, 0.22),
        excitatory_bias=0.04,
        inhibitory_bias=0.08,
        threshold=0.40,
    ),
    CurriculumSeed(
        name="stabilization",
        dataset_slice="near-solution / low-unsat regimes",
        centroid=SolverSeedFeatures(0.82, 0.06, 0.28, 0.15, 0.72, 0.20, 0.20, 0.16, 0.40),
        concentration_prior=(0.40, 0.16, 0.08, 0.36),
        excitatory_bias=-0.01,
        inhibitory_bias=0.10,
        threshold=0.46,
    ),
    CurriculumSeed(
        name="plateau",
        dataset_slice="UNSAT-like long-stagnation regimes",
        centroid=SolverSeedFeatures(0.72, 0.22, 0.38, 0.35, 0.55, 0.95, 0.70, 0.58, 0.75),
        concentration_prior=(0.16, 0.20, 0.30, 0.34),
        excitatory_bias=0.11,
        inhibitory_bias=-0.02,
        threshold=0.32,
    ),
)


def extract_features(context: Mapping[str, object]) -> SolverSeedFeatures:
    formula = context.get("formula", [])
    clause_count = max(1, len(formula))  # type: ignore[arg-type]
    steps = max(1, int(context.get("steps", 1)))
    t = max(0, int(context.get("t", 0)))
    unsatisfied = float(context.get("unsatisfied", clause_count))
    samples = list(context.get("samples", []))
    best_unsatisfied = float(context.get("best_unsatisfied", unsatisfied))
    effects = context.get("operator_effects", {})

    stagnation = 0.0
    if samples:
        horizon = min(32, len(samples))
        recent = samples[-horizon:]
        recent_best = min(float(sample.unsatisfied_clauses) for sample in recent)
        stagnation = 1.0 if recent_best <= best_unsatisfied else 0.0
        if horizon > 1 and recent[-1].unsatisfied_clauses >= recent[0].unsatisfied_clauses:
            stagnation = max(stagnation, 0.65)

    return SolverSeedFeatures(
        progress=clamp01(t / max(1, steps - 1)),
        unsat_ratio=clamp01(unsatisfied / clause_count),
        entropy=clamp01(float(context.get("entropy", 0.0))),
        heat=clamp01(float(context.get("heat", 0.0))),
        integration=clamp01(float(context.get("integration", 0.0))),
        stagnation=clamp01(stagnation),
        revisit=clamp01(memory_visit_pressure(context)),
        loop_pressure=clamp01(mean_abs(effects.get("loop_escape", []))),  # type: ignore[union-attr]
        memory_pressure=clamp01(mean_abs(effects.get("memory", []))),  # type: ignore[union-attr]
    )


def route_seeds(features: SolverSeedFeatures, temperature: float = 0.18) -> SeedRouting:
    distances = [feature_distance(features, seed.centroid) for seed in SEEDS]
    scale = max(0.01, temperature)
    logits = [-distance / scale for distance in distances]
    weights = softmax(logits)
    active_index = max(range(len(SEEDS)), key=lambda index: weights[index])

    concentration = weighted_vector(
        [seed.concentration_prior for seed in SEEDS], weights
    )
    excitatory = sum(seed.excitatory_bias * weight for seed, weight in zip(SEEDS, weights))
    inhibitory = sum(seed.inhibitory_bias * weight for seed, weight in zip(SEEDS, weights))
    threshold = sum(seed.threshold * weight for seed, weight in zip(SEEDS, weights))
    return SeedRouting(
        active_seed=SEEDS[active_index].name,
        active_index=active_index,
        seed_names=tuple(seed.name for seed in SEEDS),
        weights=tuple(weights),
        concentration_prior=tuple(concentration),  # type: ignore[arg-type]
        excitatory_bias=excitatory,
        inhibitory_bias=inhibitory,
        threshold=threshold,
    )


def blend_concentrations(
    base: Sequence[float],
    seed_prior: Sequence[float],
    seed_weight: float = 0.45,
) -> list[float]:
    if not base:
        return normalize(seed_prior)
    alpha = clamp01(seed_weight)
    blended = [
        (1.0 - alpha) * float(base_value) + alpha * float(seed_value)
        for base_value, seed_value in zip(base, seed_prior)
    ]
    return normalize(blended)


def feature_distance(left: SolverSeedFeatures, right: SolverSeedFeatures) -> float:
    weights = SolverSeedFeatures(0.8, 1.2, 0.9, 0.7, 0.7, 1.3, 1.1, 1.0, 0.8)
    total = 0.0
    for key in left.__dataclass_fields__:
        delta = float(getattr(left, key)) - float(getattr(right, key))
        total += float(getattr(weights, key)) * delta * delta
    return math.sqrt(total)


def memory_visit_pressure(context: Mapping[str, object]) -> float:
    memory = context.get("fiber_memory")
    variable_fibers = getattr(memory, "variable_fibers", ())
    if not variable_fibers:
        return 0.0
    return mean(float(getattr(fiber, "visits", 0.0)) for fiber in variable_fibers)


def softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    peak = max(values)
    exp_values = [math.exp(value - peak) for value in values]
    total = sum(exp_values)
    if total <= 1e-12:
        return [1.0 / len(values) for _ in values]
    return [value / total for value in exp_values]


def weighted_vector(
    vectors: Sequence[Sequence[float]],
    weights: Sequence[float],
) -> list[float]:
    if not vectors:
        return []
    result = [0.0 for _ in vectors[0]]
    for vector, weight in zip(vectors, weights):
        for index, value in enumerate(vector):
            result[index] += float(weight) * float(value)
    return normalize(result)


def normalize(values: Sequence[float]) -> list[float]:
    clipped = [max(0.0, float(value)) for value in values]
    total = sum(clipped)
    if total <= 1e-12:
        return [1.0 / len(clipped) for _ in clipped] if clipped else []
    return [value / total for value in clipped]


def mean_abs(values: Sequence[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(abs(float(value)) for value in values) / len(values)


def mean(values: Sequence[float]) -> float:
    values = list(values)
    return sum(values) / max(1, len(values))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
