#!/usr/bin/env python3
"""Sprite detector for spatial SAT furnace fields."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

EPSILON = 1e-12


@dataclass(frozen=True)
class SpatialSample:
    t: int
    kind: str
    id: int
    heat: float
    influence: float
    entropy: float
    spin: float
    pressure: float


@dataclass(frozen=True)
class Sprite:
    sprite_id: int
    kind: str
    component_id: int
    birth_t: int
    death_t: int
    lifetime: int
    peak_intensity: float
    mean_intensity: float
    mean_influence: float
    mean_entropy: float
    signed_motion: float
    spin_displacement: float
    bridge_score: float
    collapse_contribution: float
    classification: str


@dataclass(frozen=True)
class GraphRunner:
    runner_id: int
    birth_t: int
    death_t: int
    lifetime: int
    path_length: int
    unique_nodes: int
    path_novelty: float
    revisit_rate: float
    runner_mass: float
    mass_density: float
    loop_score: float
    escape_score: float
    mean_intensity: float
    peak_intensity: float
    bridge_score: float
    collapse_contribution: float
    classification: str
    path: str


def load_samples(path: Path) -> list[SpatialSample]:
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle)
        return [
            SpatialSample(
                t=int(float(row["t"])),
                kind=row["kind"],
                id=int(float(row["id"])),
                heat=float(row["heat"]),
                influence=float(row["influence"]),
                entropy=float(row["entropy"]),
                spin=float(row["spin"]),
                pressure=float(row["pressure"]),
            )
            for row in rows
        ]


def load_graph(path: Path | None) -> dict[tuple[str, int], set[tuple[str, int]]]:
    adjacency: dict[tuple[str, int], set[tuple[str, int]]] = defaultdict(set)
    if path is None:
        return adjacency
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle)
        for row in rows:
            source = (row["source_kind"], int(float(row["source_id"])))
            target = (row["target_kind"], int(float(row["target_id"])))
            adjacency[source].add(target)
            adjacency[target].add(source)
    return adjacency


def detect_graph_runners(
    samples: Sequence[SpatialSample],
    adjacency: dict[tuple[str, int], set[tuple[str, int]]],
    quantile: float = 0.90,
    max_gap: int = 2,
    min_lifetime: int = 5,
) -> list[GraphRunner]:
    if not adjacency:
        return []
    by_time: dict[int, list[tuple[tuple[str, int], SpatialSample, float]]] = defaultdict(list)
    intensities = [sprite_intensity(sample) for sample in samples]
    threshold = quantile_value(intensities, quantile)
    for sample, intensity in zip(samples, intensities):
        if intensity >= threshold and intensity > EPSILON:
            by_time[sample.t].append(((sample.kind, sample.id), sample, intensity))

    active: list[dict[str, object]] = []
    completed: list[dict[str, object]] = []
    for t in sorted(by_time):
        current = sorted(by_time[t], key=lambda item: item[2], reverse=True)
        used_current: set[int] = set()
        next_active: list[dict[str, object]] = []
        for runner in active:
            last_node = runner["last_node"]
            candidate_index = None
            candidate_distance = 999
            for index, (node, _sample, _intensity) in enumerate(current):
                if index in used_current:
                    continue
                distance = graph_distance(last_node, node, adjacency)
                if distance is not None and distance < candidate_distance:
                    candidate_index = index
                    candidate_distance = distance
            if candidate_index is not None and candidate_distance <= 2:
                node, sample, intensity = current[candidate_index]
                used_current.add(candidate_index)
                runner["last_node"] = node
                runner["last_t"] = t
                runner["nodes"].append(node)
                runner["samples"].append(sample)
                runner["intensities"].append(intensity)
                next_active.append(runner)
            elif t - int(runner["last_t"]) <= max_gap:
                next_active.append(runner)
            else:
                completed.append(runner)
        for index, (node, sample, intensity) in enumerate(current):
            if index in used_current:
                continue
            next_active.append(
                {
                    "first_t": t,
                    "last_t": t,
                    "last_node": node,
                    "nodes": [node],
                    "samples": [sample],
                    "intensities": [intensity],
                }
            )
        active = next_active
    completed.extend(active)

    runners: list[GraphRunner] = []
    for raw in completed:
        runner = build_graph_runner(len(runners) + 1, raw)
        if runner.lifetime >= min_lifetime and runner.unique_nodes >= 3:
            runners.append(runner)
    return runners


def graph_distance(
    source: tuple[str, int],
    target: tuple[str, int],
    adjacency: dict[tuple[str, int], set[tuple[str, int]]],
) -> int | None:
    if source == target:
        return 0
    if target in adjacency.get(source, set()):
        return 1
    for neighbor in adjacency.get(source, set()):
        if target in adjacency.get(neighbor, set()):
            return 2
    return None


def build_graph_runner(runner_id: int, raw: dict[str, object]) -> GraphRunner:
    samples: list[SpatialSample] = raw["samples"]
    intensities: list[float] = raw["intensities"]
    nodes: list[tuple[str, int]] = raw["nodes"]
    birth_t = int(raw["first_t"])
    death_t = int(raw["last_t"])
    lifetime = death_t - birth_t + 1
    path_length = len(nodes)
    unique_nodes = len(set(nodes))
    path_novelty = unique_nodes / max(1, path_length)
    revisit_rate = 1.0 - path_novelty
    runner_mass = sum(intensities)
    mass_density = runner_mass / max(1, unique_nodes)
    entropy_drop = samples[0].entropy - samples[-1].entropy
    influence_gain = samples[-1].influence - samples[0].influence
    bridge_score = clamp01(
        0.35 * normalize_positive(unique_nodes / 8.0)
        + 0.25 * normalize_positive(path_length / 24.0)
        + 0.25 * normalize_positive(abs(samples[-1].spin - samples[0].spin))
        + 0.15 * normalize_positive(influence_gain)
    )
    collapse_contribution = clamp01(
        0.40 * normalize_positive(entropy_drop)
        + 0.35 * normalize_positive(influence_gain)
        + 0.25 * normalize_positive(statistics.fmean(intensities))
    )
    loop_score = clamp01(revisit_rate * normalize_positive(mass_density))
    escape_score = clamp01(path_novelty * bridge_score * (0.5 + collapse_contribution))
    classification = classify_graph_runner(path_novelty, revisit_rate, mass_density, loop_score, escape_score, bridge_score, collapse_contribution)
    path = " -> ".join(f"{kind}:{node_id}" for kind, node_id in compress_path(nodes, limit=12))
    return GraphRunner(
        runner_id=runner_id,
        birth_t=birth_t,
        death_t=death_t,
        lifetime=lifetime,
        path_length=path_length,
        unique_nodes=unique_nodes,
        path_novelty=path_novelty,
        revisit_rate=revisit_rate,
        runner_mass=runner_mass,
        mass_density=mass_density,
        loop_score=loop_score,
        escape_score=escape_score,
        mean_intensity=statistics.fmean(intensities),
        peak_intensity=max(intensities),
        bridge_score=bridge_score,
        collapse_contribution=collapse_contribution,
        classification=classification,
        path=path,
    )


def classify_graph_runner(
    path_novelty: float,
    revisit_rate: float,
    mass_density: float,
    loop_score: float,
    escape_score: float,
    bridge_score: float,
    collapse_contribution: float,
) -> str:
    if escape_score > 0.12 and bridge_score > 0.35 and path_novelty > 0.18:
        return "escaping_runner"
    if loop_score > 0.35 and revisit_rate > 0.65:
        return "reinforced_loop"
    if mass_density > 1.0 and collapse_contribution < 0.08:
        return "unsat_trap"
    if path_novelty > 0.30 and bridge_score > 0.25:
        return "exploratory_runner"
    return "diffuse_runner"


def compress_path(nodes: Sequence[tuple[str, int]], limit: int) -> list[tuple[str, int]]:
    compressed: list[tuple[str, int]] = []
    for node in nodes:
        if not compressed or compressed[-1] != node:
            compressed.append(node)
    if len(compressed) <= limit:
        return compressed
    head = compressed[: limit // 2]
    tail = compressed[-(limit - len(head)) :]
    return head + [("...", -1)] + tail


def detect_sprites(samples: Sequence[SpatialSample], quantile: float = 0.82, min_lifetime: int = 4) -> list[Sprite]:
    grouped: dict[tuple[str, int], list[SpatialSample]] = defaultdict(list)
    for sample in samples:
        grouped[(sample.kind, sample.id)].append(sample)

    sprites: list[Sprite] = []
    next_id = 1
    for (kind, component_id), component_samples in sorted(grouped.items()):
        component_samples.sort(key=lambda sample: sample.t)
        intensities = [sprite_intensity(sample) for sample in component_samples]
        threshold = quantile_value(intensities, quantile)
        active_run: list[tuple[SpatialSample, float]] = []
        for sample, intensity in zip(component_samples, intensities):
            if intensity >= threshold and intensity > EPSILON:
                active_run.append((sample, intensity))
            else:
                if len(active_run) >= min_lifetime:
                    sprites.append(build_sprite(next_id, kind, component_id, active_run))
                    next_id += 1
                active_run = []
        if len(active_run) >= min_lifetime:
            sprites.append(build_sprite(next_id, kind, component_id, active_run))
            next_id += 1
    return sprites


def sprite_intensity(sample: SpatialSample) -> float:
    return sample.heat * (0.45 + 0.35 * sample.influence + 0.20 * sample.entropy) + 0.15 * abs(sample.pressure)


def build_sprite(sprite_id: int, kind: str, component_id: int, run: Sequence[tuple[SpatialSample, float]]) -> Sprite:
    samples = [sample for sample, _ in run]
    intensities = [intensity for _, intensity in run]
    birth_t = samples[0].t
    death_t = samples[-1].t
    lifetime = death_t - birth_t + 1
    mean_intensity = statistics.fmean(intensities)
    mean_influence = statistics.fmean(sample.influence for sample in samples)
    mean_entropy = statistics.fmean(sample.entropy for sample in samples)
    signed_motion = samples[-1].pressure - samples[0].pressure
    spin_displacement = samples[-1].spin - samples[0].spin
    entropy_drop = samples[0].entropy - samples[-1].entropy
    influence_gain = samples[-1].influence - samples[0].influence
    bridge_score = clamp01(0.45 * normalize_positive(influence_gain) + 0.35 * normalize_positive(abs(spin_displacement)) + 0.20 * normalize_positive(lifetime / 32.0))
    collapse_contribution = clamp01(0.40 * normalize_positive(entropy_drop) + 0.35 * normalize_positive(influence_gain) + 0.25 * normalize_positive(mean_intensity))
    classification = classify_sprite(lifetime, bridge_score, collapse_contribution, mean_entropy, abs(spin_displacement))
    return Sprite(
        sprite_id=sprite_id,
        kind=kind,
        component_id=component_id,
        birth_t=birth_t,
        death_t=death_t,
        lifetime=lifetime,
        peak_intensity=max(intensities),
        mean_intensity=mean_intensity,
        mean_influence=mean_influence,
        mean_entropy=mean_entropy,
        signed_motion=signed_motion,
        spin_displacement=spin_displacement,
        bridge_score=bridge_score,
        collapse_contribution=collapse_contribution,
        classification=classification,
    )


def classify_sprite(lifetime: int, bridge_score: float, collapse_contribution: float, mean_entropy: float, displacement: float) -> str:
    if bridge_score > 0.30 and collapse_contribution > 0.12 and lifetime >= 12:
        return "runner"
    if lifetime >= 24 and mean_entropy > 0.55 and displacement < 0.2:
        return "standing_wave"
    if collapse_contribution > 0.18:
        return "collapse_front"
    return "hotspot"


def summarize_sprites(sprites: Sequence[Sprite]) -> dict[str, float | int | str]:
    if not sprites:
        return {
            "sprite_count": 0,
            "runner_count": 0,
            "standing_wave_count": 0,
            "collapse_front_count": 0,
            "mean_lifetime": 0.0,
            "max_lifetime": 0,
            "mean_bridge_score": 0.0,
            "mean_collapse_contribution": 0.0,
            "prediction_hint": "No persistent sprites detected",
        }
    runner_count = sum(1 for sprite in sprites if sprite.classification == "runner")
    standing_wave_count = sum(1 for sprite in sprites if sprite.classification == "standing_wave")
    collapse_front_count = sum(1 for sprite in sprites if sprite.classification == "collapse_front")
    mean_bridge = statistics.fmean(sprite.bridge_score for sprite in sprites)
    mean_collapse = statistics.fmean(sprite.collapse_contribution for sprite in sprites)
    if runner_count and mean_collapse > 0.02:
        prediction_hint = "Hard SAT / bridge-forming runner ecology"
    elif standing_wave_count > runner_count:
        prediction_hint = "UNSAT-like recurrent standing-wave ecology"
    elif collapse_front_count or mean_collapse > 0.06:
        prediction_hint = "SAT-like collapse-front ecology"
    else:
        prediction_hint = "Diffuse hotspot ecology"
    return {
        "sprite_count": len(sprites),
        "runner_count": runner_count,
        "standing_wave_count": standing_wave_count,
        "collapse_front_count": collapse_front_count,
        "mean_lifetime": statistics.fmean(sprite.lifetime for sprite in sprites),
        "max_lifetime": max(sprite.lifetime for sprite in sprites),
        "mean_bridge_score": mean_bridge,
        "mean_collapse_contribution": mean_collapse,
        "prediction_hint": prediction_hint,
    }


def write_sprite_log(path: Path, sprites: Sequence[Sprite]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Sprite.__dataclass_fields__.keys()))
        writer.writeheader()
        for sprite in sprites:
            writer.writerow(sprite.__dict__)


def write_runner_log(path: Path, runners: Sequence[GraphRunner]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(GraphRunner.__dataclass_fields__.keys()))
        writer.writeheader()
        for runner in runners:
            writer.writerow(runner.__dict__)


def print_summary(summary: dict[str, float | int | str], sprites: Sequence[Sprite], runners: Sequence[GraphRunner], limit: int) -> None:
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    print(f"graph_runner_count: {len(runners)}")
    if runners:
        print(f"mean_graph_runner_lifetime: {statistics.fmean(runner.lifetime for runner in runners):.6f}")
        print(f"mean_graph_runner_unique_nodes: {statistics.fmean(runner.unique_nodes for runner in runners):.6f}")
        print(f"mean_graph_runner_path_novelty: {statistics.fmean(runner.path_novelty for runner in runners):.6f}")
        print(f"mean_graph_runner_revisit_rate: {statistics.fmean(runner.revisit_rate for runner in runners):.6f}")
        print(f"mean_graph_runner_mass_density: {statistics.fmean(runner.mass_density for runner in runners):.6f}")
        print(f"mean_graph_runner_loop_score: {statistics.fmean(runner.loop_score for runner in runners):.6f}")
        print(f"mean_graph_runner_escape_score: {statistics.fmean(runner.escape_score for runner in runners):.6f}")
        print(f"mean_graph_runner_bridge_score: {statistics.fmean(runner.bridge_score for runner in runners):.6f}")
        print(f"mean_graph_runner_collapse_contribution: {statistics.fmean(runner.collapse_contribution for runner in runners):.6f}")
        for classification in ["exploratory_runner", "reinforced_loop", "escaping_runner", "unsat_trap", "diffuse_runner"]:
            print(f"{classification}_count: {sum(1 for runner in runners if runner.classification == classification)}")
    print("top_graph_runners:")
    ranked_runners = sorted(runners, key=lambda runner: (runner.bridge_score + runner.collapse_contribution, runner.lifetime), reverse=True)
    for runner in ranked_runners[:limit]:
        print(
            f"  - id={runner.runner_id} type={runner.classification} t={runner.birth_t}-{runner.death_t} life={runner.lifetime} "
            f"nodes={runner.unique_nodes} novelty={runner.path_novelty:.3f} revisit={runner.revisit_rate:.3f} "
            f"mass_density={runner.mass_density:.3f} loop={runner.loop_score:.3f} escape={runner.escape_score:.3f} "
            f"bridge={runner.bridge_score:.3f} collapse={runner.collapse_contribution:.3f} path={runner.path}"
        )
    print("top_sprites:")
    ranked = sorted(sprites, key=lambda sprite: (sprite.bridge_score + sprite.collapse_contribution, sprite.lifetime), reverse=True)
    for sprite in ranked[:limit]:
        print(
            f"  - id={sprite.sprite_id} kind={sprite.kind}:{sprite.component_id} "
            f"type={sprite.classification} t={sprite.birth_t}-{sprite.death_t} "
            f"life={sprite.lifetime} bridge={sprite.bridge_score:.3f} collapse={sprite.collapse_contribution:.3f}"
        )


def quantile_value(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return ordered[index]


def normalize_positive(value: float) -> float:
    return clamp01(value / (1.0 + abs(value))) if value > 0.0 else 0.0


def clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect long-lived sprites in SAT furnace spatial fields.")
    parser.add_argument("spatial_csv", type=Path)
    parser.add_argument("--graph", type=Path, help="Optional clause-variable adjacency CSV from sat_furnace.py --graph-out.")
    parser.add_argument("--quantile", type=float, default=0.82)
    parser.add_argument("--runner-quantile", type=float, default=0.92)
    parser.add_argument("--min-lifetime", type=int, default=4)
    parser.add_argument("--out", type=Path, help="Optional sprite event log CSV path.")
    parser.add_argument("--runner-out", type=Path, help="Optional graph runner event log CSV path.")
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    samples = load_samples(args.spatial_csv)
    sprites = detect_sprites(samples, quantile=args.quantile, min_lifetime=args.min_lifetime)
    runners = detect_graph_runners(samples, load_graph(args.graph), quantile=args.runner_quantile, min_lifetime=args.min_lifetime)
    summary = summarize_sprites(sprites)
    if args.out is not None:
        write_sprite_log(args.out, sprites)
        print(f"wrote: {args.out}")
    if args.runner_out is not None:
        write_runner_log(args.runner_out, runners)
        print(f"wrote_runners: {args.runner_out}")
    print_summary(summary, sprites, runners, args.top)


if __name__ == "__main__":
    main()
