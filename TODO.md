# Operator-First Refactor - Complete

Status as of 2026-05-01: all planned operator-first refactor phases are complete.

## Completed

- Phase A: removed legacy monolith path; single composed run path is active.
- Phase B: epoch inline logic was factored into registered solver operators.
- Phase C: adaptive branching moved into operator bodies.
- Phase D: epoch driver reduced to thin Composer orchestration with centralized init context.
- Phase E: trial execution moved to trial-level Composer operators.
- Phase F: furnace math internals are private; obsolete post-loop trace delta pass removed.

## Current Architecture

- composer.py: generic DAG executor.
- sat_furnace.py: domain types, private math primitives, thin run_furnace driver.
- sat_composer.py: solver/graph/trial operator registrations and operator logic.
- benchmark_calorimeter.py: trial orchestration and metrics over composed outputs.
