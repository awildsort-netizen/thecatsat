#!/usr/bin/env python3
"""Concentration-field sampling over eligible composer providers.

Smallest useful experiment: take a target that has *multiple* plausible
providers in the reflected candidate set, define a concentration field
biasing the sampler toward one of them, and run many trials to see the
distribution of paths the field actually produces.

Two concentration fields are compared:

  * ``flat``  — empty bias dict; sampling should be ~uniform over eligibles.
  * ``warm`` — heavy weight on one specific provider; sampling should
                concentrate toward that provider without ever picking an
                ineligible one.

The eligibility-over-concentration claim is then probed directly: we drop
a heavily-weighted provider from the eligible set and confirm that no
trial selects it, even though its concentration is huge.

Each sampled path emits a gene-string-like trace (``L:<provider>`` tokens),
showing what *actually happened*, not what was intended.

Run with: ``python experiments/concentration_field.py``
"""

from __future__ import annotations

import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from composer import OperatorCandidate, discover_operator_candidates, rank_provider_candidates
from concentration import run_many


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


def reflected_eligible_for(target: str, nearby: tuple[str, ...]) -> tuple[str, ...]:
    """Use the composer's own reflection + ranking as the eligibility gate.

    Eligibility here = ``provider_fit`` score > 0 over the reflected public
    surface of a few project modules. This deliberately reuses the existing
    discovery path so the concentration field operates on a real candidate
    set, not invented strings.
    """

    import importlib

    pool: list[OperatorCandidate] = []
    for module_name in ("sat_field", "sat_composer", "streamable_genes", "attention_policies"):
        module = importlib.import_module(module_name)
        pool.extend(
            c for c in discover_operator_candidates(module) if not c.name.startswith("_")
        )
    fits = rank_provider_candidates(target, pool, nearby_terms=nearby)
    return tuple(fit.candidate.name for fit in fits)


def experiment_single_target_distribution() -> None:
    banner("1. Single target, two concentration fields")
    target = "formula_graph"
    nearby = ("sat", "formula", "graph")
    eligible = reflected_eligible_for(target, nearby)
    print(f"  target={target!r}  nearby={nearby}")
    print(f"  eligible providers ({len(eligible)}):")
    for name in eligible:
        print(f"    - {name}")
    if len(eligible) < 2:
        print()
        print("  (need >=2 eligible providers to make the experiment meaningful)")
        return

    steps = [(target, eligible)]
    trials = 400

    flat: dict[str, float] = {}
    warm: dict[str, float] = {eligible[0]: 8.0}

    rng_flat = random.Random(20260519)
    rng_warm = random.Random(20260519)

    _, dist_flat = run_many(trials, steps, flat, rng_flat)
    _, dist_warm = run_many(trials, steps, warm, rng_warm)

    print()
    print(f"  trials={trials}, seed=20260519")
    print()
    print(f"  field=flat (no bias)  -> uniform-ish distribution:")
    print(fmt_dist(dist_flat, trials))
    print()
    print(f"  field=warm (8x on {eligible[0]!r})  -> mass shifts:")
    print(fmt_dist(dist_warm, trials))
    print()
    print("  Observation: the same eligible set under two concentration")
    print("  fields produces visibly different path distributions, while no")
    print("  ineligible provider appears in either.")


def experiment_multistep_path_distribution() -> None:
    banner("2. Multi-step paths, two concentration fields, gene traces")
    target_a = "formula_graph"
    target_b = "clause_pressure"
    nearby_a = ("sat", "formula", "graph")
    nearby_b = ("sat", "solver")
    elig_a = reflected_eligible_for(target_a, nearby_a)
    elig_b = reflected_eligible_for(target_b, nearby_b)
    if not elig_a or not elig_b:
        print("  (could not assemble a two-step eligible chain; skipping)")
        return
    print(f"  step 1 target={target_a!r}  eligible({len(elig_a)})={elig_a}")
    print(f"  step 2 target={target_b!r}  eligible({len(elig_b)})={elig_b}")

    steps = [(target_a, elig_a), (target_b, elig_b)]
    trials = 600

    flat: dict[str, float] = {}
    warm: dict[str, float] = {elig_a[0]: 6.0, elig_b[0]: 6.0}

    rng_flat = random.Random(20260519)
    rng_warm = random.Random(20260519)

    paths_flat, dist_flat = run_many(trials, steps, flat, rng_flat)
    paths_warm, dist_warm = run_many(trials, steps, warm, rng_warm)

    print()
    print(f"  trials={trials}, seed=20260519")
    print()
    print(f"  field=flat: top 5 path signatures")
    flat_top = sorted(dist_flat.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    for sig, count in flat_top:
        print(f"    {count:>4} ({100.0 * count / trials:>5.1f}%)  {' -> '.join(sig)}")
    print(f"  unique paths under flat : {len(dist_flat)}")
    print()
    print(f"  field=warm: top 5 path signatures")
    warm_top = sorted(dist_warm.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    for sig, count in warm_top:
        print(f"    {count:>4} ({100.0 * count / trials:>5.1f}%)  {' -> '.join(sig)}")
    print(f"  unique paths under warm: {len(dist_warm)}")
    print()
    print("  Example gene-string traces (first 3 trials of each field):")
    for label, paths in (("flat", paths_flat[:3]), ("warm", paths_warm[:3])):
        print(f"    {label}:")
        for path in paths:
            tokens = list(path.gene_tokens) + ["E"]
            print(f"      {tokens}")
    print()
    print("  Observation: the gene-token stream is exactly what happened in")
    print("  the trial, not what was wished for. Two concentration fields")
    print("  give two different *empirical* distributions over the same")
    print("  eligible scaffold.")


def experiment_eligibility_beats_concentration() -> None:
    banner("3. Eligibility wins: heavy concentration on an ineligible provider")
    target = "formula_graph"
    nearby = ("sat", "formula", "graph")
    eligible = reflected_eligible_for(target, nearby)
    if len(eligible) < 2:
        print("  (need >=2 eligibles to demonstrate the suppression cleanly)")
        return

    ghost = "totally_nonexistent_provider"
    suppressed_eligible = tuple(name for name in eligible if name != eligible[0])
    print(f"  full eligible set      : {eligible}")
    print(f"  pruned eligible set    : {suppressed_eligible}  (dropped {eligible[0]!r})")
    print(f"  ghost provider         : {ghost!r}  (never in any eligible set)")

    field = {eligible[0]: 100.0, ghost: 1000.0}
    print(f"  concentration field    : {field}")

    rng = random.Random(20260519)
    steps = [(target, suppressed_eligible)]
    trials = 200
    paths, distribution = run_many(trials, steps, field, rng)

    chosen_names = {step.chosen for path in paths for step in path.steps}
    print()
    print(f"  trials={trials}")
    print(f"  unique chosen providers: {sorted(chosen_names)}")
    print(f"  any dropped/ghost ever chosen? "
          f"{eligible[0] in chosen_names or ghost in chosen_names}")
    print()
    print("  distribution:")
    print(fmt_dist(distribution, trials))
    print()
    print("  Observation: concentrations only weight what is already")
    print("  eligible. A dropped provider with weight 100x and a")
    print("  nonexistent provider with weight 1000x both never appear —")
    print("  the field biases, it does not command.")


def main() -> None:
    experiment_single_target_distribution()
    experiment_multistep_path_distribution()
    experiment_eligibility_beats_concentration()
    print()
    print("done.")


if __name__ == "__main__":
    main()
