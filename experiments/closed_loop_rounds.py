#!/usr/bin/env python3
"""Close the loop: sampled paths warm the next field, round after round.

Up to now the loop was open: a gene trace warms a field, and a sampler
runs one batch under it. This script closes it. Each round samples paths
under the current field, pools their chosen literals into a gene trace,
and warms the next field from that trace. Repeat. Watch what happens.

Three small panels:

  1. Uniform start with no decay: weak biases get amplified into fixation.
  2. Uniform start with decay 0.6: evidence cools each round, drift instead
     of lock-in.
  3. Activation climate: same eligible set, but a ``W:dev`` climate window
     with ``window_scale={"dev": 4.0}`` makes a normally-quiet operator
     dominate the dev rounds. Eligibility unchanged either way.

Run with: ``python experiments/closed_loop_rounds.py``
"""

from __future__ import annotations

import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concentration import RoundResult, run_rounds


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def fmt_field(field: dict[str, float]) -> str:
    if not field:
        return "(empty)"
    rows = sorted(field.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{name}={weight:.2f}" for name, weight in rows)


def fmt_dist(distribution: Counter, total: int) -> str:
    rows = sorted(distribution.items(), key=lambda kv: (-kv[1], kv[0]))
    return "  ".join(
        f"{'->'.join(sig)}={100.0 * count / total:.1f}%"
        for sig, count in rows
    )


def show_rounds(rounds: tuple[RoundResult, ...], total: int) -> None:
    for i, r in enumerate(rounds):
        print(f"  round {i}: field={{{fmt_field(r.field_in)}}}")
        print(f"           dist : {fmt_dist(r.distribution, total)}")
        print(f"           fixation_index={r.fixation_index:.3f}")


def experiment_fixation_under_no_decay() -> None:
    banner("1. No decay: small early bias compounds into fixation")
    eligible = ("alpha", "beta", "gamma")
    steps = [("decision", eligible)]
    # Tiny initial bias toward alpha. Watch it grow.
    initial = {"alpha": 1.1, "beta": 1.0, "gamma": 1.0}
    rounds = run_rounds(
        n_rounds=6,
        trials_per_round=300,
        steps=steps,
        initial_field=initial,
        rng=random.Random(20260519),
        decay=1.0,
        bump=1.0,
    )
    show_rounds(rounds, 300)
    print()
    print("  Observation: a 10% nudge becomes near-fixation in a few rounds.")
    print("  No decay -> evidence only accumulates. The system locks in.")


def experiment_drift_under_decay() -> None:
    banner("2. Decay 0.6: old evidence fades, drift instead of lock-in")
    eligible = ("alpha", "beta", "gamma")
    steps = [("decision", eligible)]
    initial = {"alpha": 1.1, "beta": 1.0, "gamma": 1.0}
    rounds = run_rounds(
        n_rounds=8,
        trials_per_round=300,
        steps=steps,
        initial_field=initial,
        rng=random.Random(20260519),
        decay=0.6,
        bump=1.0,
    )
    show_rounds(rounds, 300)
    print()
    print("  Observation: each round forgets 40% of the previous prior.")
    print("  Lock-in is softer; the distribution wanders rather than fixates.")


def experiment_activation_climate() -> None:
    banner("3. Activation climate: dev tokens wake a dormant operator")
    # Eligibility carries an operator named 'reflect_operator' that is
    # dormant-but-allowed. Production climate gives it no special help;
    # it stays a quiet minority. The dev climate injects activation-factor
    # tokens each round — literals naming the dormant operator under a
    # W:dev window — so the field is re-warmed toward it independently of
    # what was actually sampled. The eligible set is identical either way.
    eligible = ("solve", "score", "reflect_operator")
    steps = [("op", eligible)]
    initial = {"solve": 3.0, "score": 3.0, "reflect_operator": 1.0}

    prod = run_rounds(
        n_rounds=5,
        trials_per_round=400,
        steps=steps,
        initial_field=initial,
        rng=random.Random(20260519),
        decay=0.7,
        bump=1.0,
    )
    # Activation factors: explicit climate tokens for dev mode that warm
    # the dormant operator. W:dev scopes them so window_scale can amplify.
    dev_climate = (
        "W:dev",
        "L:reflect_operator",
        "L:reflect_operator",
        "L:reflect_operator",
        "L:reflect_operator",
        "L:reflect_operator",
        "R",
    )
    dev = run_rounds(
        n_rounds=5,
        trials_per_round=400,
        steps=steps,
        initial_field=initial,
        rng=random.Random(20260519),
        decay=0.7,
        bump=1.0,
        climate_tokens=dev_climate,
        window_scale={"dev": 8.0},
    )

    print("  --- production climate (no activation tokens) ---")
    show_rounds(prod, 400)
    print()
    print("  --- development climate (W:dev L:reflect_operator x5 R, dev=8x) ---")
    show_rounds(dev, 400)
    print()
    print("  Observation: 'reflect_operator' was always eligible. Under the")
    print("  production climate it stays a quiet minority across rounds.")
    print("  Under the dev climate, activation-factor tokens re-warm it each")
    print("  round; combined with W:dev's window_scale=8x the dormant")
    print("  operator becomes the dominant choice by round 4. Eligibility")
    print("  unchanged in both runs.")


def experiment_climate_does_not_override_eligibility() -> None:
    banner("4. Even a roaring dev climate cannot summon an ineligible name")
    # 'reflect_operator' would dominate — but it isn't in the eligible set.
    eligible = ("solve", "score")
    steps = [("op", eligible)]
    initial = {"solve": 1.0, "score": 1.0, "reflect_operator": 1000.0}

    dev = run_rounds(
        n_rounds=4,
        trials_per_round=200,
        steps=steps,
        initial_field=initial,
        rng=random.Random(20260519),
        decay=0.5,
        bump=1.0,
        climate_tokens=("W:dev",),
        window_scale={"dev": 100.0},
    )
    chosen = {step.chosen for r in dev for path in r.paths for step in path.steps}
    print(f"  initial field: {fmt_field(initial)}")
    print(f"  eligible set : {eligible}")
    print(f"  climate      : W:dev, window_scale dev=100x")
    print(f"  unique chosen across all rounds: {sorted(chosen)}")
    print()
    show_rounds(dev, 200)
    print()
    print("  Observation: a maximally aggressive climate still picks only")
    print("  from {solve, score}. Concentration biases visibility and")
    print("  weight; eligibility is the hard gate.")


def main() -> None:
    experiment_fixation_under_no_decay()
    experiment_drift_under_decay()
    experiment_activation_climate()
    experiment_climate_does_not_override_eligibility()
    print()
    print("done.")


if __name__ == "__main__":
    main()
