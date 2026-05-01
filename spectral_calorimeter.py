#!/usr/bin/env python3
"""Spectral calorimeter for SAT/UNSAT trajectory diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


# Spectral Calorimeter:
# read heat not as magnitude, but as music.
# solving = redshifting conflict.

EPSILON = 1e-12


@dataclass(frozen=True)
class SignalFrame:
    t: float
    heat: float
    free_energy: float
    integration: float


@dataclass(frozen=True)
class WindowSpectrum:
    start_t: float
    end_t: float
    center_t: float
    spectral_centroid: float
    dominant_frequency: float
    spectral_entropy: float
    autocorrelation_half_life: float | None
    concentration_index: float
    fragmentation_index: float
    recycling_score: float
    collapse_index: float
    mean_heat: float
    mean_free_energy: float
    mean_integration: float


@dataclass(frozen=True)
class CalorimeterReport:
    windows: list[WindowSpectrum]
    redshift_rate: float
    entropy_slope: float
    integration_slope: float
    concentration_slope: float
    fragmentation_slope: float
    collapse_slope: float
    prediction: str
    confidence: float
    rationale: list[str]


def load_frames(path: Path) -> list[SignalFrame]:
    if path.suffix.lower() == ".json":
        return _load_json_frames(path)
    return _load_csv_frames(path)


def _load_csv_frames(path: Path) -> list[SignalFrame]:
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle)
        return [
            SignalFrame(
                t=float(row.get("t", row.get("time", index))),
                heat=float(row.get("H", row.get("heat", row.get("shadow_heat", 0.0)))),
                free_energy=float(row.get("F", row.get("free_energy", 0.0))),
                integration=float(row.get("I", row.get("integration", row.get("integration_score", 0.0)))),
            )
            for index, row in enumerate(rows)
        ]


def _load_json_frames(path: Path) -> list[SignalFrame]:
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        raw = raw.get("frames", raw.get("samples", []))
    return [
        SignalFrame(
            t=float(row.get("t", row.get("time", index))),
            heat=float(row.get("H", row.get("heat", row.get("shadow_heat", 0.0)))),
            free_energy=float(row.get("F", row.get("free_energy", 0.0))),
            integration=float(row.get("I", row.get("integration", row.get("integration_score", 0.0)))),
        )
        for index, row in enumerate(raw)
    ]


def analyze_frames(
    frames: Sequence[SignalFrame],
    window_size: int = 64,
    step_size: int = 8,
) -> CalorimeterReport:
    if len(frames) < window_size:
        raise ValueError(f"need at least {window_size} frames, got {len(frames)}")
    if window_size < 8:
        raise ValueError("window_size must be at least 8")
    if step_size < 1:
        raise ValueError("step_size must be at least 1")

    windows = [
        _analyze_window(frames[start : start + window_size])
        for start in range(0, len(frames) - window_size + 1, step_size)
    ]

    redshift_rate = slope([window.center_t for window in windows], [window.spectral_centroid for window in windows])
    entropy_slope = slope([window.center_t for window in windows], [window.spectral_entropy for window in windows])
    integration_slope = slope([window.center_t for window in windows], [window.mean_integration for window in windows])
    concentration_slope = slope([window.center_t for window in windows], [window.concentration_index for window in windows])
    fragmentation_slope = slope([window.center_t for window in windows], [window.fragmentation_index for window in windows])
    collapse_slope = slope([window.center_t for window in windows], [window.collapse_index for window in windows])
    prediction, confidence, rationale = classify_trajectory(
        windows,
        redshift_rate,
        entropy_slope,
        integration_slope,
        concentration_slope,
        fragmentation_slope,
        collapse_slope,
    )

    return CalorimeterReport(
        windows=windows,
        redshift_rate=redshift_rate,
        entropy_slope=entropy_slope,
        integration_slope=integration_slope,
        concentration_slope=concentration_slope,
        fragmentation_slope=fragmentation_slope,
        collapse_slope=collapse_slope,
        prediction=prediction,
        confidence=confidence,
        rationale=rationale,
    )


def _analyze_window(frames: Sequence[SignalFrame]) -> WindowSpectrum:
    times = [frame.t for frame in frames]
    heat = normalize([frame.heat for frame in frames])
    free_energy = normalize([frame.free_energy for frame in frames])
    integration = normalize([frame.integration for frame in frames])
    contradiction_music = [heat_i + free_i - int_i for heat_i, free_i, int_i in zip(heat, free_energy, integration)]
    frequencies, power = power_spectrum(contradiction_music, sampling_interval=median_delta(times))

    shadow_mass = positive_distribution([abs(frame.heat) + abs(frame.free_energy) - abs(frame.integration) for frame in frames])
    concentration = concentration_index(shadow_mass)
    fragmentation = fragmentation_index(shadow_mass)
    half_life = autocorrelation_half_life(contradiction_music, median_delta(times))
    recycling = recycling_score(half_life, times[-1] - times[0])

    return WindowSpectrum(
        start_t=times[0],
        end_t=times[-1],
        center_t=(times[0] + times[-1]) / 2.0,
        spectral_centroid=spectral_centroid(frequencies, power),
        dominant_frequency=dominant_frequency(frequencies, power),
        spectral_entropy=spectral_entropy(power),
        autocorrelation_half_life=half_life,
        concentration_index=concentration,
        fragmentation_index=fragmentation,
        recycling_score=recycling,
        collapse_index=collapse_index(concentration, fragmentation, recycling, statistics.fmean(frame.integration for frame in frames)),
        mean_heat=statistics.fmean(frame.heat for frame in frames),
        mean_free_energy=statistics.fmean(frame.free_energy for frame in frames),
        mean_integration=statistics.fmean(frame.integration for frame in frames),
    )


def power_spectrum(values: Sequence[float], sampling_interval: float = 1.0) -> tuple[list[float], list[float]]:
    centered = [value - statistics.fmean(values) for value in values]
    tapered = hann(centered)
    n = len(tapered)
    frequencies: list[float] = []
    power: list[float] = []
    for k in range(1, n // 2 + 1):
        real = 0.0
        imag = 0.0
        for sample_index, value in enumerate(tapered):
            angle = -2.0 * math.pi * k * sample_index / n
            real += value * math.cos(angle)
            imag += value * math.sin(angle)
        frequencies.append(k / (n * sampling_interval))
        power.append(real * real + imag * imag)
    return frequencies, power


def spectral_centroid(frequencies: Sequence[float], power: Sequence[float]) -> float:
    total_power = sum(power)
    if total_power <= EPSILON:
        return 0.0
    return sum(freq * weight for freq, weight in zip(frequencies, power)) / total_power


def dominant_frequency(frequencies: Sequence[float], power: Sequence[float]) -> float:
    if not power or max(power) <= EPSILON:
        return 0.0
    return frequencies[max(range(len(power)), key=power.__getitem__)]


def spectral_entropy(power: Sequence[float]) -> float:
    total_power = sum(power)
    if total_power <= EPSILON or len(power) <= 1:
        return 0.0
    probabilities = [value / total_power for value in power if value > EPSILON]
    entropy = -sum(probability * math.log(probability, 2) for probability in probabilities)
    return entropy / math.log(len(power), 2)


def autocorrelation_half_life(values: Sequence[float], sampling_interval: float = 1.0) -> float | None:
    centered = [value - statistics.fmean(values) for value in values]
    denominator = sum(value * value for value in centered)
    if denominator <= EPSILON:
        return None
    for lag in range(1, len(centered)):
        numerator = sum(centered[index] * centered[index + lag] for index in range(len(centered) - lag))
        if numerator / denominator <= 0.5:
            return lag * sampling_interval
    return None


def classify_trajectory(
    windows: Sequence[WindowSpectrum],
    redshift_rate: float,
    entropy_slope: float,
    integration_slope: float,
    concentration_slope: float,
    fragmentation_slope: float,
    collapse_slope: float,
) -> tuple[str, float, list[str]]:
    rationale: list[str] = []
    centroid_span = max(window.spectral_centroid for window in windows) - min(window.spectral_centroid for window in windows)
    dominant_stability = mode_fraction(round(window.dominant_frequency, 6) for window in windows)
    late_integration_jump = windows[-1].mean_integration - statistics.fmean(window.mean_integration for window in windows[: max(1, len(windows) // 2)])
    late_redshift = slope(
        [window.center_t for window in windows[len(windows) // 2 :]],
        [window.spectral_centroid for window in windows[len(windows) // 2 :]],
    )

    if redshift_rate < -1e-5:
        rationale.append("spectral centroid redshifts over time")
    if entropy_slope < -1e-5:
        rationale.append("spectral entropy decreases")
    if integration_slope > 1e-5:
        rationale.append("influence integration rises")
    if concentration_slope > 1e-5:
        rationale.append("shadow distribution concentrates")
    if fragmentation_slope > 1e-5:
        rationale.append("shadow distribution fragments")
    if collapse_slope > 1e-5:
        rationale.append("collapse index rises")
    if dominant_stability > 0.55:
        rationale.append("dominant frequency band persists")
    if centroid_span <= max(abs(statistics.fmean(window.spectral_centroid for window in windows)) * 0.05, EPSILON):
        rationale.append("centroid remains nearly stationary")
    if late_redshift < redshift_rate and late_integration_jump > 0:
        rationale.append("late redshift coincides with integration spike")

    sat_score = score(redshift_rate < -1e-5, entropy_slope < -1e-5, integration_slope > 1e-5, concentration_slope > 1e-5, collapse_slope > 1e-5)
    unsat_score = score(abs(redshift_rate) <= max(centroid_span * 0.01, 1e-5), dominant_stability > 0.55, integration_slope <= 1e-5, fragmentation_slope >= -1e-5, collapse_slope <= 1e-5)
    hard_sat_score = score(dominant_stability > 0.45, late_redshift < -1e-5, late_integration_jump > 0.1, concentration_slope > 1e-5, collapse_slope > 1e-5)

    scores = {"SAT": sat_score, "UNSAT": unsat_score, "Hard SAT": hard_sat_score}
    prediction = max(scores, key=scores.get)
    confidence = scores[prediction] / max(1, sum(scores.values()))
    return prediction, confidence, rationale


def score(*signals: bool) -> int:
    return sum(1 for signal in signals if signal)


def positive_distribution(values: Sequence[float]) -> list[float]:
    shifted = [max(0.0, value) for value in values]
    total = sum(shifted)
    if total <= EPSILON:
        return [1.0 / len(values) for _ in values] if values else []
    return [value / total for value in shifted]


def concentration_index(distribution: Sequence[float]) -> float:
    if not distribution:
        return 0.0
    n = len(distribution)
    participation = 1.0 / max(sum(value * value for value in distribution), EPSILON)
    if n <= 1:
        return 1.0
    return clamp01(1.0 - (participation - 1.0) / (n - 1.0))


def fragmentation_index(distribution: Sequence[float]) -> float:
    if not distribution:
        return 0.0
    threshold = statistics.fmean(distribution)
    active_regions = 0
    in_region = False
    for value in distribution:
        if value > threshold and not in_region:
            active_regions += 1
            in_region = True
        elif value <= threshold:
            in_region = False
    if distribution[0] > threshold and distribution[-1] > threshold and active_regions > 1:
        active_regions -= 1
    return clamp01((active_regions - 1.0) / max(1.0, len(distribution) / 8.0))


def recycling_score(half_life: float | None, window_span: float) -> float:
    if half_life is None:
        return 1.0
    if window_span <= EPSILON:
        return 0.0
    return clamp01(half_life / window_span)


def collapse_index(concentration: float, fragmentation: float, recycling: float, integration: float) -> float:
    normalized_integration = clamp01(integration)
    return clamp01(0.35 * concentration + 0.35 * normalized_integration + 0.15 * (1.0 - fragmentation) + 0.15 * (1.0 - recycling))


def clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def mode_fraction(values: Iterable[float]) -> float:
    counts: dict[float, int] = {}
    total = 0
    for value in values:
        counts[value] = counts.get(value, 0) + 1
        total += 1
    return max(counts.values()) / total if total else 0.0


def slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator <= EPSILON:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator


def normalize(values: Sequence[float]) -> list[float]:
    mean = statistics.fmean(values)
    variance = statistics.fmean((value - mean) ** 2 for value in values)
    deviation = math.sqrt(variance)
    if deviation <= EPSILON:
        return [0.0 for _ in values]
    return [(value - mean) / deviation for value in values]


def hann(values: Sequence[float]) -> list[float]:
    if len(values) == 1:
        return list(values)
    return [value * 0.5 * (1.0 - math.cos(2.0 * math.pi * index / (len(values) - 1))) for index, value in enumerate(values)]


def median_delta(times: Sequence[float]) -> float:
    deltas = [b - a for a, b in zip(times, times[1:]) if b > a]
    return statistics.median(deltas) if deltas else 1.0


def synthetic_frames(kind: str, count: int = 320) -> list[SignalFrame]:
    frames: list[SignalFrame] = []
    for index in range(count):
        t = float(index)
        progress = index / max(1, count - 1)
        if kind == "sat":
            frequency = 0.18 - 0.13 * progress
            integration = sigmoid((progress - 0.58) * 14.0)
            heat = (1.2 - progress) * math.sin(2 * math.pi * frequency * t) + 0.15 * math.sin(2 * math.pi * 0.03 * t)
            free_energy = (1.0 - 0.5 * progress) * math.cos(2 * math.pi * frequency * t)
        elif kind == "unsat":
            frequency = 0.16
            integration = 0.35 + 0.06 * math.sin(2 * math.pi * 0.025 * t)
            heat = 1.1 * math.sin(2 * math.pi * frequency * t) + 0.25 * math.sin(2 * math.pi * 0.31 * t)
            free_energy = 0.9 * math.cos(2 * math.pi * frequency * t)
        elif kind == "hard_sat":
            if progress < 0.68:
                frequency = 0.16
                integration = 0.28 + 0.03 * math.sin(2 * math.pi * 0.02 * t)
            else:
                frequency = 0.08 - 0.05 * ((progress - 0.68) / 0.32)
                integration = 0.28 + sigmoid((progress - 0.72) * 24.0)
            heat = (1.1 - 0.45 * max(0.0, progress - 0.68)) * math.sin(2 * math.pi * frequency * t)
            free_energy = math.cos(2 * math.pi * frequency * t)
        else:
            raise ValueError(f"unknown synthetic kind: {kind}")
        frames.append(SignalFrame(t=t, heat=heat, free_energy=free_energy, integration=integration))
    return frames


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def report_to_dict(report: CalorimeterReport) -> dict[str, object]:
    return {
        "prediction": report.prediction,
        "confidence": report.confidence,
        "redshift_rate": report.redshift_rate,
        "entropy_slope": report.entropy_slope,
        "integration_slope": report.integration_slope,
        "concentration_slope": report.concentration_slope,
        "fragmentation_slope": report.fragmentation_slope,
        "collapse_slope": report.collapse_slope,
        "rationale": report.rationale,
        "windows": [window.__dict__ for window in report.windows],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sliding-window spectral calorimeter for SAT trajectories.")
    parser.add_argument("input", nargs="?", type=Path, help="CSV/JSON file with t,H,F,I columns.")
    parser.add_argument("--window", type=int, default=64, help="Sliding FFT window size.")
    parser.add_argument("--step", type=int, default=8, help="Sliding window step size.")
    parser.add_argument("--synthetic", choices=["sat", "unsat", "hard_sat"], help="Run a built-in synthetic trajectory.")
    parser.add_argument("--json", action="store_true", help="Emit full JSON report.")
    args = parser.parse_args()

    if args.synthetic:
        frames = synthetic_frames(args.synthetic)
    elif args.input:
        frames = load_frames(args.input)
    else:
        parser.error("provide an input file or --synthetic")

    report = analyze_frames(frames, window_size=args.window, step_size=args.step)
    if args.json:
        print(json.dumps(report_to_dict(report), indent=2))
    else:
        print(f"prediction: {report.prediction}")
        print(f"confidence: {report.confidence:.3f}")
        print(f"redshift_rate: {report.redshift_rate:.8f}")
        print(f"spectral_entropy_slope: {report.entropy_slope:.8f}")
        print(f"integration_slope: {report.integration_slope:.8f}")
        print(f"concentration_slope: {report.concentration_slope:.8f}")
        print(f"fragmentation_slope: {report.fragmentation_slope:.8f}")
        print(f"collapse_slope: {report.collapse_slope:.8f}")
        print("rationale:")
        for item in report.rationale:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
