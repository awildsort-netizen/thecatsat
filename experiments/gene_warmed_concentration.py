#!/usr/bin/env python3
"""Gene-warmed concentration fields: traces become atmospheric priors.

A gene string is a record of what actually happened on a previous run —
literal operator emissions, type windows, motif reuse. A concentration
field is a bias dictionary read by the sampler. This experiment connects
the two: feed a gene stream in, get a concentration field out, then run
trials under that field and observe how the path distribution shifts.

Three sections:

  1. Two different gene traces (A and B) seed two different fields. Same
     eligible scaffold; visibly different path distributions.
  2. A windowed trace under ``W:hot`` boosts literals inside the window
     via ``window_scale``, showing local climates modulate the prior.
  3. Same gene-warmed field, but the heavily-warmed name is dropped from
     the eligible set — sampler still refuses to pick it. Concentration
     remains bias, not command.

Run with: ``python experiments/gene_warmed_concentration.py``
"""

from __future__ import annotations

import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concentration import concentration_from_gene_tokens, run_many


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def fmt_dist(distribution: Counter, total: int) -> str:
    rows = sorted(distribution.items(), key=lambda kv: (-kv[1], kv[0]))
    return "\n".join(
        f"    {count:>4} ({100.0 * count / total:>5.1f}%)  {' -> '.join(sig)}"
        for sig, count in rows
    )


def fmt_field(field: dict[str, float]) -> str:
    rows = sorted(field.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{name}={weight:g}" for name, weight in rows)


def experiment_two_traces_two_fields() -> None:
    banner("1. Two gene traces -> two concentration fields -> two distributions")
    trace_a = ["L:alpha", "L:alpha", "L:alpha", "L:beta", "E"]
    trace_b = ["L:beta", "L:beta", "L:beta", "L:alpha", "E"]
    eligible = ("alpha", "beta", "gamma")
    steps = [("decision", eligible)]
    trials = 1200

    field_a = concentration_from_gene_tokens(trace_a)
    field_b = concentration_from_gene_tokens(trace_b)

    print(f"  trace A: {trace_a}")
    print(f"  field A: {fmt_field(field_a)}")
    print(f"  trace B: {trace_b}")
    print(f"  field B: {fmt_field(field_b)}")
    print(f"  eligible: {eligible}  trials={trials}")

    rng_a = random.Random(20260519)
    rng_b = random.Random(20260519)
    _, dist_a = run_many(trials, steps, field_a, rng_a)
    _, dist_b = run_many(trials, steps, field_b, rng_b)

    print()
    print("  distribution under field A (alpha-heavy trace):")
    print(fmt_dist(dist_a, trials))
    print()
    print("  distribution under field B (beta-heavy trace):")
    print(fmt_dist(dist_b, trials))
    print()
    print("  Observation: the same eligible scaffold yields different path")
    print("  distributions because the *trace* — what already happened —")
    print("  warmed different operators into the prior.")


def experiment_windowed_trace_local_climate() -> None:
    banner("2. Windowed trace + window_scale -> local climate as bias")
    # Inside W:hot, alpha was emitted; outside, beta and gamma.
    trace = ["W:hot", "L:alpha", "L:alpha", "R", "L:beta", "L:gamma", "E"]
    eligible = ("alpha", "beta", "gamma")
    steps = [("decision", eligible)]
    trials = 1200

    field_no_window_bias = concentration_from_gene_tokens(trace)
    field_hot_amplified = concentration_from_gene_tokens(
        trace, window_scale={"hot": 6.0}
    )

    print(f"  trace: {trace}")
    print(f"  field (no window scaling)  : {fmt_field(field_no_window_bias)}")
    print(f"  field (W:hot scaled x6)    : {fmt_field(field_hot_amplified)}")

    rng_flat = random.Random(20260519)
    rng_hot = random.Random(20260519)
    _, dist_flat = run_many(trials, steps, field_no_window_bias, rng_flat)
    _, dist_hot = run_many(trials, steps, field_hot_amplified, rng_hot)

    print()
    print("  distribution under unweighted-window field:")
    print(fmt_dist(dist_flat, trials))
    print()
    print("  distribution under amplified-window field (alpha lives in hot):")
    print(fmt_dist(dist_hot, trials))
    print()
    print("  Observation: the same trace with a louder local climate biases")
    print("  the sampler toward operators that lived inside that window.")
    print("  No new operators are invented; only the air pressure changes.")


def experiment_warmed_field_still_obeys_eligibility() -> None:
    banner("3. Heavy gene-warmed weights still bow to eligibility")
    # A trace that overwhelmingly warms 'ghost' — but ghost will never be
    # in the eligible set the sampler is given.
    trace = ["L:ghost"] * 10 + ["L:alpha", "E"]
    pruned_eligible = ("alpha", "beta")
    steps = [("decision", pruned_eligible)]
    trials = 400

    field = concentration_from_gene_tokens(trace)
    print(f"  trace (ghost emitted 10x, alpha 1x): {trace}")
    print(f"  resulting field: {fmt_field(field)}")
    print(f"  eligible set (ghost intentionally absent): {pruned_eligible}")

    rng = random.Random(20260519)
    paths, distribution = run_many(trials, steps, field, rng)
    chosen_names = {step.chosen for path in paths for step in path.steps}

    print()
    print(f"  trials={trials}")
    print(f"  unique chosen providers: {sorted(chosen_names)}")
    print(f"  was 'ghost' ever chosen?  {'ghost' in chosen_names}")
    print()
    print("  distribution:")
    print(fmt_dist(distribution, trials))
    print()
    print("  Observation: a gene-warmed field that screams 'ghost!' still")
    print("  picks only from {alpha, beta}. The gene trace shapes the prior,")
    print("  but the eligibility rule is the final gate.")


def main() -> None:
    experiment_two_traces_two_fields()
    experiment_windowed_trace_local_climate()
    experiment_warmed_field_still_obeys_eligibility()
    print()
    print("done.")


if __name__ == "__main__":
    main()
