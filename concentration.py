#!/usr/bin/env python3
"""Concentration-field sampling over eligible operator providers.

A concentration field is a mapping ``{provider_name: weight}``. Weights bias
which provider gets chosen among an already-eligible set; they do **not**
override eligibility. If a provider is not in the eligible set, its
concentration is ignored entirely — concentrations are biases, not commands.

The public surface is intentionally three small functions:

  * ``sample_provider`` — one weighted choice from an eligible set.
  * ``sample_path``     — repeated single-choice picks across a sequence of
                          (target, eligible) steps, yielding a gene trace.
  * ``run_many``        — repeated ``sample_path`` trials with a seeded RNG,
                          returning the distribution of paths.

This module deliberately knows nothing about the SAT solver, the furnace, or
any domain semantics. It only knows: eligible names, a concentration dict,
and a ``random.Random``. Anything richer belongs in an experiment script.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from streamable_genes import stream


@dataclass(frozen=True)
class SampledStep:
    """One sampled choice in a path.

    ``target`` is what was being produced. ``chosen`` is the provider name.
    ``eligible`` is the eligible set seen at this step (frozen for trace
    inspection). ``gene_token`` is the streamable-gene-style ``L:<name>``
    token a downstream decoder could consume.
    """

    target: str
    chosen: str
    eligible: tuple[str, ...]
    gene_token: str


@dataclass(frozen=True)
class SampledPath:
    """A full sampled trajectory across a sequence of steps."""

    steps: tuple[SampledStep, ...]

    @property
    def gene_tokens(self) -> tuple[str, ...]:
        return tuple(step.gene_token for step in self.steps)

    @property
    def signature(self) -> tuple[str, ...]:
        """Path identity for distribution counting: just the chosen names."""

        return tuple(step.chosen for step in self.steps)


def sample_provider(
    eligible: Sequence[str],
    concentration: Mapping[str, float],
    rng: random.Random,
    *,
    default_weight: float = 1.0,
) -> str:
    """Pick one name from ``eligible`` weighted by ``concentration``.

    Providers absent from the concentration map fall back to ``default_weight``
    so that an unbiased field (empty dict) is uniform over the eligible set.
    A provider with weight ``0.0`` is treated as eligible-but-suppressed; if
    *all* eligible providers carry weight ``0.0`` the call falls back to a
    uniform pick over ``eligible`` (no silent failure, no exception — the
    concentration is a bias, not a veto).
    """

    if not eligible:
        raise ValueError("sample_provider requires a non-empty eligible set")
    weights = [max(0.0, float(concentration.get(name, default_weight))) for name in eligible]
    total = sum(weights)
    if total <= 0.0:
        return rng.choice(list(eligible))
    return rng.choices(list(eligible), weights=weights, k=1)[0]


def sample_path(
    steps: Iterable[tuple[str, Sequence[str]]],
    concentration: Mapping[str, float],
    rng: random.Random,
) -> SampledPath:
    """Sample one provider per step, return the full path with gene tokens.

    ``steps`` is an iterable of ``(target, eligible_providers)`` pairs. The
    eligible set is taken as given — if upstream eligibility checks already
    pruned a provider, this function never sees it.
    """

    sampled: list[SampledStep] = []
    for target, eligible in steps:
        elig_tuple = tuple(eligible)
        chosen = sample_provider(elig_tuple, concentration, rng)
        sampled.append(
            SampledStep(
                target=target,
                chosen=chosen,
                eligible=elig_tuple,
                gene_token=f"L:{chosen}",
            )
        )
    return SampledPath(steps=tuple(sampled))


def run_many(
    trials: int,
    steps: Sequence[tuple[str, Sequence[str]]],
    concentration: Mapping[str, float],
    rng: random.Random,
) -> tuple[tuple[SampledPath, ...], Counter]:
    """Run ``trials`` independent sampled paths over the same step shape.

    Returns ``(paths, distribution)`` where ``distribution`` counts how many
    times each full path-signature occurred. The same ``rng`` is used across
    trials, so seeding it once before the call makes the entire experiment
    reproducible.
    """

    if trials < 0:
        raise ValueError(f"trials must be >= 0, got {trials}")
    paths: list[SampledPath] = []
    distribution: Counter[tuple[str, ...]] = Counter()
    for _ in range(trials):
        path = sample_path(steps, concentration, rng)
        paths.append(path)
        distribution[path.signature] += 1
    return tuple(paths), distribution


@dataclass(frozen=True)
class RoundResult:
    """One round of the closed feedback loop.

    ``field_in`` is the concentration field used by this round's sampler.
    ``paths`` are the sampled trajectories; ``distribution`` counts their
    signatures. ``trace`` is the flat sequence of ``L:<chosen>`` tokens
    pooled from every sampled path — the gene record the next round
    warms its field from.
    """

    field_in: dict[str, float]
    paths: tuple[SampledPath, ...]
    distribution: Counter
    trace: tuple[str, ...]

    @property
    def fixation_index(self) -> float:
        """Share of trials taken by the most common path signature.

        ``1.0`` means every trial picked the same path — full lock-in.
        ``1/k`` with ``k`` equally-popular signatures means uniform spread.
        Empty rounds (zero trials) report ``0.0``.

        This is intentionally the distribution view, not the field view:
        what the loop *did* this round, not what it was *biased toward*.
        Fixation rising across rounds is the signature of the city waking
        only one neighborhood; staying flat is healthy ecology.
        """

        total = self.distribution.total()
        if total <= 0:
            return 0.0
        return max(self.distribution.values()) / total


def run_rounds(
    n_rounds: int,
    trials_per_round: int,
    steps: Sequence[tuple[str, Sequence[str]]],
    initial_field: Mapping[str, float],
    rng: random.Random,
    *,
    decay: float = 1.0,
    bump: float = 1.0,
    window_scale: Mapping[str, float] | None = None,
    climate_tokens: Sequence[str] = (),
) -> tuple[RoundResult, ...]:
    """Close the loop: sample, harvest gene trace, warm next field, repeat.

    Each round:
      1. Samples ``trials_per_round`` paths under the current field.
      2. Pools every chosen step as ``L:<name>`` tokens — the round's trace.
      3. Builds the next field by decaying current weights by ``decay``
         (``1.0`` = no decay, ``0.0`` = full forgetting) then adding fresh
         evidence from the trace via ``concentration_from_gene_tokens``.

    ``climate_tokens`` are prepended to each round's trace before warming —
    e.g. ``["W:dev"]`` opens a window so every literal in this round is
    amplified when ``window_scale={"dev": 3.0}``. The climate biases the
    *next* field; eligibility at every step is still the hard gate.

    A round with no eligible-and-chosen names (impossible for non-empty
    steps) would leave the field untouched aside from decay; this is by
    design — silence cools the prior.
    """

    if n_rounds < 0:
        raise ValueError(f"n_rounds must be >= 0, got {n_rounds}")
    if not 0.0 <= decay <= 1.0:
        raise ValueError(f"decay must be in [0,1], got {decay}")

    field: dict[str, float] = dict(initial_field)
    results: list[RoundResult] = []
    for _ in range(n_rounds):
        paths, distribution = run_many(trials_per_round, steps, field, rng)
        trace = tuple(step.gene_token for path in paths for step in path.steps)
        warming = concentration_from_gene_tokens(
            tuple(climate_tokens) + trace,
            base=0.0,
            bump=bump,
            window_scale=window_scale,
        )
        next_field: dict[str, float] = {
            name: weight * decay for name, weight in field.items()
        }
        for name, weight in warming.items():
            next_field[name] = next_field.get(name, 0.0) + weight
        results.append(
            RoundResult(
                field_in=dict(field),
                paths=paths,
                distribution=distribution,
                trace=trace,
            )
        )
        field = next_field
    return tuple(results)


def concentration_from_gene_tokens(
    tokens: Iterable[str],
    *,
    base: float = 1.0,
    bump: float = 1.0,
    window_scale: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Build a concentration field from a streamable gene-token stream.

    Each ``L:<name>`` token is evidence that ``<name>`` was emitted, and warms
    its weight. The first time we see a name we seed it at ``base``; each
    subsequent occurrence adds ``bump``, so repeated literals accumulate.

    Non-literal tokens (``E``, ``A:``, ``D:``, raw ``W:``/``R``) never create
    a provider entry — only literals warm concentrations. ``W:<label>`` /
    ``R`` still scope the climate around the literals inside them: if
    ``window_scale`` is provided, the per-occurrence bump for a literal
    inside window ``label`` is multiplied by ``window_scale[label]`` (missing
    labels default to 1.0). This is how local climates show up in the field
    without inventing a separate per-window schema.

    Motif backreferences (``M:i``) are expanded through the canonical decoder,
    so ``D:1:a,b`` followed by ``M:1`` warms both ``a`` and ``b`` once.

    The returned dict is suitable as-is for ``sample_provider`` / ``run_many``:
    ineligible providers are ignored at sample time, so this function never
    needs to know which names a downstream composer would accept.
    """

    if base < 0.0:
        raise ValueError(f"base must be >= 0, got {base}")
    if bump < 0.0:
        raise ValueError(f"bump must be >= 0, got {bump}")
    scale = dict(window_scale or {})
    state = stream(tokens)
    field: dict[str, float] = {}
    for gene in state.emitted:
        factor = scale.get(gene.window, 1.0) if gene.window is not None else 1.0
        if gene.name not in field:
            field[gene.name] = base + bump * factor
        else:
            field[gene.name] += bump * factor
    return field
