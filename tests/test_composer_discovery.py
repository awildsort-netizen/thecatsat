#!/usr/bin/env python3

from __future__ import annotations

import types
import unittest
from dataclasses import dataclass

from composer import (
    choose_provider,
    discover_operator_candidates,
    materialize_function_operator,
    operator_candidate,
    rank_provider_candidates,
)


@dataclass(frozen=True)
class FormulaGraph:
    nodes: tuple[str, ...]


@dataclass(frozen=True)
class GeneratedFormula:
    formula: list[str]
    planted_assignment: list[bool]


def formula_graph(formula: list[str]) -> FormulaGraph:
    return FormulaGraph(nodes=tuple(formula))


def graph(formula: list[str]) -> FormulaGraph:
    return FormulaGraph(nodes=tuple(formula))


def generated_formula(kind: str) -> GeneratedFormula:
    return GeneratedFormula(formula=[kind], planted_assignment=[True])


class ComposerDiscoveryTests(unittest.TestCase):
    def test_candidate_infers_requires_and_provides_from_function_shape(self) -> None:
        candidate = operator_candidate(formula_graph)

        self.assertEqual(candidate.parameters, ("formula",))
        self.assertEqual(candidate.inferred_outputs, ("formula_graph",))
        self.assertIs(candidate.return_type, FormulaGraph)

    def test_dataclass_return_fields_become_named_outputs(self) -> None:
        candidate = operator_candidate(generated_formula)

        self.assertEqual(candidate.parameters, ("kind",))
        self.assertEqual(candidate.inferred_outputs, ("formula", "planted_assignment"))

    def test_provider_ranking_uses_name_type_and_locality(self) -> None:
        candidates = (operator_candidate(graph), operator_candidate(formula_graph))

        fits = rank_provider_candidates(
            "formula_graph",
            candidates,
            nearby_terms=("formula", "sat"),
            required_type=FormulaGraph,
        )

        self.assertEqual(fits[0].candidate.name, "formula_graph")
        self.assertIn("exact_output", fits[0].reasons)
        self.assertIn("required_type", fits[0].reasons)

    def test_locality_can_choose_between_semantically_close_candidates(self) -> None:
        local = operator_candidate(graph)
        distant = operator_candidate(graph)
        local = type(local)(
            function=local.function,
            name=local.name,
            module="sat_formula_graph",
            parameters=local.parameters,
            return_type=local.return_type,
            inferred_outputs=local.inferred_outputs,
            locality_terms=("sat", "formula", "graph"),
        )
        distant = type(distant)(
            function=distant.function,
            name=distant.name,
            module="generic_graph",
            parameters=distant.parameters,
            return_type=distant.return_type,
            inferred_outputs=distant.inferred_outputs,
            locality_terms=("generic", "graph"),
        )

        fit = choose_provider("formula_graph", (distant, local), nearby_terms=("sat", "formula"))

        self.assertIsNotNone(fit)
        self.assertEqual(fit.candidate.module, "sat_formula_graph")
        self.assertIn("nearby_locality", fit.reasons)

    def test_materialized_function_operator_runs_from_context(self) -> None:
        operator = materialize_function_operator(operator_candidate(formula_graph))

        result = operator.run({"formula": ["a", "b"]})

        self.assertEqual(operator.inputs, ("formula",))
        self.assertEqual(operator.outputs, ("formula_graph",))
        self.assertEqual(result["formula_graph"], FormulaGraph(nodes=("a", "b")))

    def test_discover_operator_candidates_reads_module_functions(self) -> None:
        module = types.ModuleType("local_ops")

        def local_value() -> int:
            return 1

        local_value.__module__ = "local_ops"
        module.local_value = local_value

        candidates = discover_operator_candidates(module)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].name, "local_value")
        self.assertEqual(candidates[0].inferred_outputs, ("local_value",))


if __name__ == "__main__":
    unittest.main()
