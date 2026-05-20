#!/usr/bin/env python3
"""Reflection-driven operator discovery over existing modules.

Hypothesis: the composer already does the right thing structurally — name,
signature, return annotation, and module locality are enough to surface
function-shaped operator candidates without verbose metadata. This script
asks the question concretely: if we reflect over ``sat_field``,
``sat_composer``, and ``streamable_genes``, which of their public
functions are picked up as operator candidates, and which sample targets
do they provider-fit for?

Safety guardrails applied:
  * skip names starting with ``_`` (private/by-convention internal)
  * skip functions whose ``__module__`` does not match the module name
    (i.e. re-exports / imports — already discarded by composer.discover)
  * never call any discovered function here; we only inspect signatures.

Run with: ``python experiments/reflection_discovery.py``
"""

from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType
from typing import Iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from composer import (
    OperatorCandidate,
    discover_operator_candidates,
    rank_provider_candidates,
)


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def public_candidates(module: ModuleType) -> tuple[OperatorCandidate, ...]:
    """Same as composer.discover_operator_candidates but drop private names."""

    return tuple(c for c in discover_operator_candidates(module) if not c.name.startswith("_"))


def fmt_outputs(candidate: OperatorCandidate) -> str:
    return ", ".join(candidate.inferred_outputs)


def fmt_params(candidate: OperatorCandidate) -> str:
    return ", ".join(candidate.parameters) or "—"


def experiment_module_surface(module_names: Iterable[str]) -> dict[str, tuple[OperatorCandidate, ...]]:
    banner("1. Module reflection surface (public functions only)")
    discovered: dict[str, tuple[OperatorCandidate, ...]] = {}
    for module_name in module_names:
        module = importlib.import_module(module_name)
        public = public_candidates(module)
        private = tuple(
            c for c in discover_operator_candidates(module) if c.name.startswith("_")
        )
        discovered[module_name] = public
        print(f"\n  {module_name}:  public={len(public)}  private(skipped)={len(private)}")
        for cand in public[:8]:
            print(
                f"    - {cand.name:<28} "
                f"params=({fmt_params(cand)})  outputs=({fmt_outputs(cand)})"
            )
        if len(public) > 8:
            print(f"    ... ({len(public) - 8} more)")
    print()
    print("  Observation: public-function reflection produces a meaningful")
    print("  candidate set with zero added metadata. Output names come from")
    print("  the function name (or dataclass-return field names). Private")
    print("  names with leading ``_`` are skipped as a coarse safety gate.")
    return discovered


def experiment_provider_fit(discovered: dict[str, tuple[OperatorCandidate, ...]]) -> None:
    banner("2. Provider-fit over reflected candidates")
    targets = (
        ("formula_graph", ("sat", "formula")),
        ("graph_adjacency", ("sat", "graph")),
        ("clause_pressure", ("sat", "solver")),
        ("composable_now", ("genes", "stream")),
        ("nonexistent_target", ("sat",)),
    )
    pool: list[OperatorCandidate] = []
    for cands in discovered.values():
        pool.extend(cands)
    for target, nearby in targets:
        fits = rank_provider_candidates(target, pool, nearby_terms=nearby)
        print(f"\n  target={target!r}  nearby={nearby}")
        if not fits:
            print("    (no candidates fit)")
            continue
        for fit in fits[:3]:
            print(
                f"    score={fit.score:>4.2f}  "
                f"{fit.candidate.module}.{fit.candidate.name:<24}  "
                f"reasons={fit.reasons}"
            )
        if len(fits) > 3:
            print(f"    ... ({len(fits) - 3} more)")
    print()
    print("  Observation: provider_fit composes 'coastline' reasons rather")
    print("  than a single scalar. An ecological reader can see *why* a")
    print("  candidate is in the ranking (exact_output / output_terms /")
    print("  return_type / locality / nearby_locality), not just a fitness")
    print("  number. nonexistent_target shows the dark side: silent zero.")


def experiment_safety_audit(discovered: dict[str, tuple[OperatorCandidate, ...]]) -> None:
    banner("3. Safety audit — what would resist being run blindly")
    flagged: list[tuple[str, OperatorCandidate, list[str]]] = []
    for module_name, cands in discovered.items():
        for cand in cands:
            reasons: list[str] = []
            # Side-effect smell: any param named like a sink or source.
            sink_terms = {"path", "writer", "file", "handle", "stream", "rng", "module"}
            if any(p in sink_terms for p in cand.parameters):
                reasons.append("side_effect_param")
            # No-output smell: return_type is None / not annotated.
            if cand.return_type is None:
                reasons.append("unannotated_return")
            # Untyped fan-in: many required positional params (>5) makes
            # auto-wiring risky.
            if len(cand.parameters) > 5:
                reasons.append("wide_signature")
            if reasons:
                flagged.append((module_name, cand, reasons))
    print(f"  flagged candidates: {len(flagged)}")
    for module_name, cand, reasons in flagged[:12]:
        print(f"    - {module_name}.{cand.name}  reasons={reasons}")
    if len(flagged) > 12:
        print(f"    ... ({len(flagged) - 12} more)")
    print()
    print("  Observation: signature-only inspection is enough to spot the")
    print("  obvious risk classes (file/path/rng params, unannotated")
    print("  returns, wide arg lists). These are cheap guardrails; the next")
    print("  level would be runtime sandboxing, which costs more.")


def experiment_coverage_against_registered(discovered: dict[str, tuple[OperatorCandidate, ...]]) -> None:
    banner("4. Coverage: what would reflection auto-register that the solver already has?")
    import sat_composer  # local import

    solver = sat_composer.build_solver_composer()
    registered = set(solver._operators.keys())
    # Strip 'solver.' prefix to compare bare function names.
    registered_bare = {name.split(".", 1)[-1] for name in registered}
    reflected_names: set[str] = set()
    for cands in discovered.values():
        reflected_names.update(c.name for c in cands)
    overlap = sorted(registered_bare & reflected_names)
    only_registered = sorted(registered_bare - reflected_names)
    only_reflected = sorted(reflected_names - registered_bare)
    print(f"  registered solver operators: {len(registered_bare)}")
    print(f"  public reflected candidates: {len(reflected_names)}")
    print(f"  overlap                    : {len(overlap)}")
    print()
    print(f"  in registered & reflected ({len(overlap)}):")
    for name in overlap:
        print(f"    + {name}")
    print()
    print(f"  in registered only ({len(only_registered)}):")
    for name in only_registered[:10]:
        print(f"    - {name}")
    if len(only_registered) > 10:
        print(f"    ... ({len(only_registered) - 10} more)")
    print()
    print(f"  in reflected only ({len(only_reflected)}):")
    for name in only_reflected[:10]:
        print(f"    ? {name}")
    if len(only_reflected) > 10:
        print(f"    ... ({len(only_reflected) - 10} more)")
    print()
    print("  Observation: there is partial overlap by design — many solver")
    print("  operators are wrappers over private helpers (``_loop_escape_bias``,")
    print("  ``adaptive_strength``) and thus don't surface via public-only")
    print("  reflection. The reflected-only set is the candidate frontier:")
    print("  functions that *could* become operators next, with no extra")
    print("  metadata work — just a registration call.")


def main() -> None:
    modules = ("sat_field", "sat_composer", "streamable_genes", "attention_policies")
    discovered = experiment_module_surface(modules)
    experiment_provider_fit(discovered)
    experiment_safety_audit(discovered)
    experiment_coverage_against_registered(discovered)
    print()
    print("done.")


if __name__ == "__main__":
    main()
