#!/usr/bin/env python3
"""Tests for the external SAT solver adapter.

No real solver binary is required: subprocess invocation is exercised
through monkeypatched fakes, and the no-binary skip path is covered
explicitly. DIMACS conversion and output parsing are unit-tested
directly on strings — no I/O.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import external_sat
import sat_furnace
from external_sat import (
    _literals_to_assignment,
    _parse_competition_output,
    _parse_minisat_result_file,
    discover_solver,
    external_solve,
    formula_to_dimacs,
    is_external_solver_available,
)
from sat_benchmarks import SolveResult, build_sat_benchmark_composer


def _planted(seed: int, variables: int, clauses: int):
    rng = random.Random(seed)
    formula, _ = sat_furnace.generate_formula("sat", variables, clauses, 3, rng)
    return formula


class DimacsConversionTests(unittest.TestCase):
    def test_header_records_variables_and_clause_count(self) -> None:
        formula = [[(0, False), (1, True)], [(2, False)]]
        text = formula_to_dimacs(formula, variables=3)
        self.assertIn("p cnf 3 2", text.splitlines()[0])

    def test_literal_sign_follows_is_negated_flag(self) -> None:
        # var 0 positive, var 1 negated, var 2 positive
        formula = [[(0, False), (1, True), (2, False)]]
        text = formula_to_dimacs(formula, variables=3)
        # 1-based DIMACS literals; negation flips the sign.
        self.assertIn("1 -2 3 0", text)

    def test_each_clause_terminated_with_zero(self) -> None:
        formula = [[(0, False)], [(1, True)]]
        text = formula_to_dimacs(formula, variables=2)
        lines = text.strip().splitlines()
        # Header + 2 clause lines, each ending in " 0".
        self.assertEqual(len(lines), 3)
        for clause_line in lines[1:]:
            self.assertTrue(clause_line.endswith(" 0"), clause_line)

    def test_empty_formula_still_well_formed(self) -> None:
        text = formula_to_dimacs([], variables=4)
        self.assertEqual(text.strip(), "p cnf 4 0")

    def test_handles_planted_instance_round_trip_shape(self) -> None:
        formula = _planted(7, 5, 12)
        text = formula_to_dimacs(formula, variables=5)
        lines = text.strip().splitlines()
        self.assertEqual(len(lines), 1 + len(formula))
        self.assertTrue(lines[0].startswith("p cnf 5 "))


class OutputParsingTests(unittest.TestCase):
    def test_competition_satisfiable_parses_status_and_model(self) -> None:
        stdout = "c some comment\ns SATISFIABLE\nv 1 -2 3 0\n"
        status, asgn = _parse_competition_output(stdout, variables=3)
        self.assertIs(status, True)
        self.assertEqual(asgn, (True, False, True))

    def test_competition_unsatisfiable_parses_status_no_model(self) -> None:
        stdout = "s UNSATISFIABLE\n"
        status, asgn = _parse_competition_output(stdout, variables=4)
        self.assertIs(status, False)
        self.assertIsNone(asgn)

    def test_competition_unknown_leaves_status_none(self) -> None:
        stdout = "s UNKNOWN\n"
        status, _ = _parse_competition_output(stdout, variables=2)
        self.assertIsNone(status)

    def test_competition_model_can_span_multiple_v_lines(self) -> None:
        stdout = "s SATISFIABLE\nv 1 -2\nv 3 -4 0\n"
        status, asgn = _parse_competition_output(stdout, variables=4)
        self.assertIs(status, True)
        self.assertEqual(asgn, (True, False, True, False))

    def test_minisat_result_file_sat(self) -> None:
        text = "SAT\n1 -2 3 0\n"
        status, asgn = _parse_minisat_result_file(text, variables=3)
        self.assertIs(status, True)
        self.assertEqual(asgn, (True, False, True))

    def test_minisat_result_file_unsat(self) -> None:
        text = "UNSAT\n"
        status, asgn = _parse_minisat_result_file(text, variables=3)
        self.assertIs(status, False)
        self.assertIsNone(asgn)

    def test_minisat_result_file_empty(self) -> None:
        status, asgn = _parse_minisat_result_file("", variables=3)
        self.assertIsNone(status)
        self.assertIsNone(asgn)

    def test_literals_to_assignment_ignores_out_of_range_and_zero(self) -> None:
        # Variable index 99 is out of range for variables=3 and should be ignored.
        asgn = _literals_to_assignment([1, -2, 0, 99, -100], variables=3)
        self.assertEqual(asgn, (True, False, False))


class NoBinarySkipTests(unittest.TestCase):
    def test_discover_solver_returns_none_when_nothing_on_path(self) -> None:
        with mock.patch("external_sat.shutil.which", return_value=None):
            self.assertIsNone(discover_solver())
            self.assertFalse(is_external_solver_available())

    def test_external_solve_returns_unavailable_result_when_no_binary(self) -> None:
        formula = _planted(7, 5, 12)
        with mock.patch("external_sat.shutil.which", return_value=None):
            result = external_solve(formula, variables=5)
        self.assertIsInstance(result, SolveResult)
        self.assertFalse(result.solved)
        self.assertEqual(result.solver_name, "external_unavailable")
        self.assertEqual(result.work_metric, "unavailable")
        self.assertEqual(result.work_units, 0)
        self.assertIsNone(result.assignment)
        self.assertIn("reason", result.metabolism)


class FakeSubprocessTests(unittest.TestCase):
    """Drive external_solve through a fake subprocess to cover the happy path."""

    def _fake_run_competition_sat(self, cmd, **kwargs):
        result = mock.Mock()
        result.stdout = "s SATISFIABLE\nv 1 -2 3 -4 5 0\n"
        result.stderr = ""
        result.returncode = 10
        return result

    def _fake_run_competition_unsat(self, cmd, **kwargs):
        result = mock.Mock()
        result.stdout = "s UNSATISFIABLE\n"
        result.stderr = ""
        result.returncode = 20
        return result

    def test_external_solve_competition_sat_returns_solved_with_model(self) -> None:
        formula = _planted(7, 5, 12)
        with mock.patch("external_sat.shutil.which",
                        return_value="/fake/bin/cadical"), \
             mock.patch("external_sat.subprocess.run",
                        side_effect=self._fake_run_competition_sat):
            result = external_solve(formula, variables=5)
        self.assertTrue(result.solved)
        self.assertEqual(result.solver_name, "external:cadical")
        self.assertEqual(result.work_metric, "external_seconds")
        self.assertEqual(result.assignment, (True, False, True, False, True))
        self.assertEqual(result.final_unsatisfied, 0)

    def test_external_solve_competition_unsat_returns_not_solved(self) -> None:
        formula = _planted(7, 5, 12)
        with mock.patch("external_sat.shutil.which",
                        return_value="/fake/bin/kissat"), \
             mock.patch("external_sat.subprocess.run",
                        side_effect=self._fake_run_competition_unsat):
            result = external_solve(formula, variables=5)
        self.assertFalse(result.solved)
        self.assertEqual(result.solver_name, "external:kissat")
        self.assertIsNone(result.assignment)

    def test_external_solve_timeout_returns_clean_skip_result(self) -> None:
        formula = _planted(7, 5, 12)
        def _raise_timeout(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=0.001)
        with mock.patch("external_sat.shutil.which",
                        return_value="/fake/bin/cadical"), \
             mock.patch("external_sat.subprocess.run", side_effect=_raise_timeout):
            result = external_solve(formula, variables=5, timeout_s=0.001)
        self.assertFalse(result.solved)
        self.assertEqual(result.solver_name, "external:cadical")
        self.assertEqual(result.metabolism.get("reason"), "timeout")

    def test_external_solve_falls_back_to_exit_code_when_no_status_lines(self) -> None:
        # Solver emits no parseable s-line; we should accept exit code 10 = SAT.
        formula = _planted(7, 5, 12)
        def _fake_run(cmd, **kwargs):
            result = mock.Mock()
            result.stdout = "garbage with no s line\n"
            result.stderr = ""
            result.returncode = 10
            return result
        with mock.patch("external_sat.shutil.which",
                        return_value="/fake/bin/cadical"), \
             mock.patch("external_sat.subprocess.run", side_effect=_fake_run):
            result = external_solve(formula, variables=5)
        self.assertTrue(result.solved)
        self.assertEqual(result.solver_name, "external:cadical")


class MinisatStyleFakeTests(unittest.TestCase):
    """MiniSat writes its result to a file passed as argv[2]; emulate that."""

    def test_external_solve_minisat_style_reads_result_file(self) -> None:
        formula = _planted(7, 5, 12)

        def _fake_run(cmd, **kwargs):
            # cmd = [binary, cnf_path, result_path]
            result_path = cmd[2]
            with open(result_path, "w", encoding="ascii") as f:
                f.write("SAT\n1 -2 3 -4 5 0\n")
            res = mock.Mock()
            res.stdout = "SATISFIABLE\n"
            res.stderr = ""
            res.returncode = 10
            return res

        with mock.patch("external_sat.shutil.which",
                        return_value="/usr/bin/minisat"), \
             mock.patch("external_sat.subprocess.run", side_effect=_fake_run):
            result = external_solve(formula, variables=5)
        self.assertTrue(result.solved)
        self.assertEqual(result.solver_name, "external:minisat")
        self.assertEqual(result.assignment, (True, False, True, False, True))


class ComposerIntegrationTests(unittest.TestCase):
    def test_external_excluded_by_default(self) -> None:
        composer = build_sat_benchmark_composer()
        plan = composer.plan(
            ("brute_result", "dpll_result", "furnace_benchmark_result"),
            available_keys=("formula", "variables"),
        )
        self.assertEqual(plan.missing, ())
        self.assertNotIn("external_solve", plan.order)

    def test_external_included_when_flag_set(self) -> None:
        composer = build_sat_benchmark_composer(include_external=True)
        plan = composer.plan(
            ("brute_result", "dpll_result", "furnace_benchmark_result",
             "external_result"),
            available_keys=("formula", "variables"),
        )
        self.assertEqual(plan.missing, ())
        self.assertIn("external_solve", plan.order)

    def test_composer_external_result_is_solveresult_even_without_binary(self) -> None:
        formula = _planted(7, 5, 12)
        composer = build_sat_benchmark_composer(include_external=True)
        with mock.patch("external_sat.shutil.which", return_value=None):
            out = composer.run(
                ("brute_result", "dpll_result", "furnace_benchmark_result",
                 "external_result"),
                {
                    "formula": formula,
                    "variables": 5,
                    "furnace_steps": 10,
                    "furnace_seed": 1,
                },
            )
        self.assertIsInstance(out["external_result"], SolveResult)
        self.assertEqual(out["external_result"].work_metric, "unavailable")
        # Other rows are untouched.
        self.assertEqual(out["brute_result"].solver_name, "brute_force")


if __name__ == "__main__":
    unittest.main()
