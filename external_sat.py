#!/usr/bin/env python3
"""External SAT solver adapter — composer-shaped, dependency-optional.

Wraps a real SAT solver binary (CaDiCaL, MiniSat, Kissat, Glucose, PicoSAT)
behind the same ``SolveResult`` shape used by the brute / DPLL / furnace
operators in :mod:`sat_benchmarks`. If no supported binary is on PATH the
adapter still returns a ``SolveResult`` — ``solved=False`` with
``work_metric='unavailable'`` — so callers can render a clean
"skipped: no external SAT binary" row without having to special-case it.

DIMACS conversion is tiny: the repo stores clauses as
``[(var_index_0based, is_negated_bool), ...]``, which maps to DIMACS as
``+/-(var+1)`` literals, one clause per line, terminated with ``0``.

Output parsing supports two common conventions:

* SAT-competition style ("s SATISFIABLE" / "v 1 -2 3 0"), used by
  CaDiCaL, Kissat, Glucose, PicoSAT.
* MiniSat style (status to stdout, model written to an output file as
  ``SAT\\n<literals> 0``).

If both forms are present we prefer the result file (MiniSat). If
neither yields a model on a SAT verdict we still return ``solved=True``
with ``assignment=None``; verifying the model is the caller's job.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from typing import Iterable, Sequence

from composer import FieldOperator
from sat_benchmarks import SolveResult


# Order matters: prefer modern, faster solvers first.
SUPPORTED_BINARIES: tuple[str, ...] = (
    "cadical",
    "kissat",
    "glucose",
    "minisat",
    "picosat",
)


def discover_solver(preferred: Sequence[str] | None = None) -> str | None:
    """Return the first supported solver binary on PATH, or ``None``."""
    candidates = preferred if preferred is not None else SUPPORTED_BINARIES
    for name in candidates:
        path = shutil.which(name)
        if path:
            return path
    return None


def formula_to_dimacs(formula: Iterable[Sequence[tuple[int, bool]]],
                      variables: int) -> str:
    """Convert the repo's ``[(var, is_neg)]`` clause form to DIMACS CNF text."""
    clause_list = list(formula)
    lines = [f"p cnf {variables} {len(clause_list)}"]
    for clause in clause_list:
        literals = []
        for var, is_neg in clause:
            lit = var + 1
            literals.append(str(-lit if is_neg else lit))
        literals.append("0")
        lines.append(" ".join(literals))
    return "\n".join(lines) + "\n"


def _parse_competition_output(stdout: str,
                              variables: int) -> tuple[bool | None,
                                                       tuple[bool, ...] | None]:
    """Parse SAT-competition style ``s`` / ``v`` lines from stdout."""
    status: bool | None = None
    model_literals: list[int] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("s "):
            verdict = line[2:].strip().upper()
            if verdict.startswith("SATISFIABLE"):
                status = True
            elif verdict.startswith("UNSATISFIABLE"):
                status = False
            # "UNKNOWN" leaves status as None.
        elif line.startswith("v "):
            for tok in line[2:].split():
                try:
                    model_literals.append(int(tok))
                except ValueError:
                    pass
    return status, _literals_to_assignment(model_literals, variables)


def _parse_minisat_result_file(text: str,
                               variables: int) -> tuple[bool | None,
                                                        tuple[bool, ...] | None]:
    """Parse MiniSat's two-line result file (``SAT\\n<lits> 0`` or ``UNSAT``)."""
    stripped = text.strip()
    if not stripped:
        return None, None
    lines = stripped.splitlines()
    header = lines[0].strip().upper()
    if header.startswith("UNSAT"):
        return False, None
    if header.startswith("SAT"):
        literals: list[int] = []
        for line in lines[1:]:
            for tok in line.split():
                try:
                    literals.append(int(tok))
                except ValueError:
                    pass
        return True, _literals_to_assignment(literals, variables)
    return None, None


def _literals_to_assignment(literals: Iterable[int],
                            variables: int) -> tuple[bool, ...] | None:
    """Convert DIMACS literals ``[1, -2, 3, 0]`` to a boolean assignment."""
    assignment = [False] * variables
    saw_any = False
    for lit in literals:
        if lit == 0:
            continue
        var_index = abs(lit) - 1
        if 0 <= var_index < variables:
            assignment[var_index] = (lit > 0)
            saw_any = True
    return tuple(assignment) if saw_any else None


def external_solve(formula,
                   variables: int,
                   *,
                   solver_path: str | None = None,
                   timeout_s: float = 30.0) -> SolveResult:
    """Run an external SAT solver and return a ``SolveResult``.

    If ``solver_path`` is ``None`` we auto-discover; if discovery fails
    the result has ``solved=False`` and ``work_metric='unavailable'``.
    """
    binary = solver_path or discover_solver()
    if not binary:
        return SolveResult(
            solver_name="external_unavailable",
            solved=False,
            final_unsatisfied=len(list(formula)),
            wall_time_s=0.0,
            work_metric="unavailable",
            work_units=0,
            assignment=None,
            metabolism={"reason": "no supported SAT binary on PATH",
                        "candidates": list(SUPPORTED_BINARIES)},
        )

    binary_name = os.path.basename(binary).lower()
    dimacs = formula_to_dimacs(formula, variables)

    with tempfile.TemporaryDirectory(prefix="extsat_") as tmpdir:
        cnf_path = os.path.join(tmpdir, "instance.cnf")
        with open(cnf_path, "w", encoding="ascii") as f:
            f.write(dimacs)

        result_path = os.path.join(tmpdir, "result.txt")
        is_minisat_style = "minisat" in binary_name
        if is_minisat_style:
            cmd = [binary, cnf_path, result_path]
        else:
            cmd = [binary, cnf_path]

        start = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - start
            return SolveResult(
                solver_name=f"external:{binary_name}",
                solved=False,
                final_unsatisfied=len(list(formula)),
                wall_time_s=elapsed,
                work_metric="external_seconds",
                work_units=int(elapsed),
                assignment=None,
                metabolism={"reason": "timeout",
                            "timeout_s": timeout_s,
                            "binary": binary_name},
            )
        elapsed = time.perf_counter() - start

        status: bool | None = None
        assignment: tuple[bool, ...] | None = None
        if is_minisat_style and os.path.exists(result_path):
            with open(result_path, "r", encoding="ascii") as f:
                status, assignment = _parse_minisat_result_file(
                    f.read(), variables
                )
        if status is None:
            status, comp_asgn = _parse_competition_output(proc.stdout, variables)
            if assignment is None:
                assignment = comp_asgn

    # Exit codes: SAT competition convention is 10=SAT, 20=UNSAT. Trust
    # parsed status first; fall back to exit code if the parser saw nothing.
    if status is None:
        if proc.returncode == 10:
            status = True
        elif proc.returncode == 20:
            status = False

    solved = bool(status)
    if status is None:
        # Unknown verdict — treat as unsolved, surface the exit code.
        return SolveResult(
            solver_name=f"external:{binary_name}",
            solved=False,
            final_unsatisfied=len(list(formula)),
            wall_time_s=elapsed,
            work_metric="external_seconds",
            work_units=int(elapsed),
            assignment=None,
            metabolism={"reason": "unknown verdict",
                        "exit_code": proc.returncode,
                        "binary": binary_name},
        )

    final_unsat = 0 if solved else len(list(formula))
    return SolveResult(
        solver_name=f"external:{binary_name}",
        solved=solved,
        final_unsatisfied=final_unsat,
        wall_time_s=elapsed,
        work_metric="external_seconds",
        work_units=int(elapsed * 1000),  # ms; integer for the shared field
        assignment=assignment if solved else None,
        metabolism={"binary": binary_name,
                    "exit_code": proc.returncode,
                    "wall_time_s": elapsed},
    )


def external_solver_op(*,
                       solver_path: str | None = None,
                       timeout_s: float = 30.0) -> FieldOperator:
    """Composer operator producing ``external_result``."""
    def _run(ctx):
        return {
            "external_result": external_solve(
                ctx["formula"],
                ctx["variables"],
                solver_path=ctx.get("external_solver_path", solver_path),
                timeout_s=float(ctx.get("external_timeout_s", timeout_s)),
            )
        }

    return FieldOperator(
        name="external_solve",
        inputs=("formula", "variables"),
        outputs=("external_result",),
        run=_run,
    )


def is_external_solver_available() -> bool:
    """True iff a supported SAT binary is discoverable on PATH."""
    return discover_solver() is not None
