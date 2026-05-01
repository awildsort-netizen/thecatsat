#!/usr/bin/env python3

from __future__ import annotations

import random
import unittest

import sat_curriculum
import sat_furnace


class CurriculumSeedTests(unittest.TestCase):
    def test_trap_features_route_to_trapbreak_seed(self) -> None:
        features = sat_curriculum.SolverSeedFeatures(
            progress=0.45,
            unsat_ratio=0.35,
            entropy=0.45,
            heat=0.45,
            integration=0.45,
            stagnation=0.85,
            revisit=0.85,
            loop_pressure=0.75,
            memory_pressure=0.55,
        )

        routing = sat_curriculum.route_seeds(features)

        self.assertEqual(routing.active_seed, "trapbreak")
        self.assertAlmostEqual(sum(routing.weights), 1.0)
        self.assertEqual(len(routing.concentration_prior), len(sat_curriculum.EFFECT_BASIS))

    def test_seed_blend_preserves_distribution(self) -> None:
        blended = sat_curriculum.blend_concentrations(
            base=(0.25, 0.25, 0.25, 0.25),
            seed_prior=(0.58, 0.18, 0.14, 0.10),
        )

        self.assertAlmostEqual(sum(blended), 1.0)
        self.assertGreater(blended[0], blended[-1])

    def test_curriculum_policy_emits_seed_route_trace(self) -> None:
        rng = random.Random(7)
        formula, planted = sat_furnace.generate_formula(
            "sat",
            variables=8,
            clauses=24,
            clause_size=3,
            rng=rng,
        )

        result = sat_furnace.run_furnace(
            formula=formula,
            variables=8,
            steps=12,
            rng=rng,
            temperature=0.8,
            learning_rate=0.08,
            inertia=0.9,
            noise=0.02,
            planted_assignment=planted,
            adaptive=True,
            policy=sat_curriculum.CURRICULUM_SEED_POLICY,
        )

        route_traces = [
            trace
            for trace in result.operator_traces
            if trace.operator == "curriculum_seed_route"
        ]
        active_traces = [
            trace
            for trace in result.operator_traces
            if trace.operator == "curriculum_seed_active"
        ]
        spike_traces = [
            trace for trace in result.operator_traces if trace.operator == "excitable_spike"
        ]

        self.assertEqual(len(route_traces), 12)
        self.assertEqual(len(active_traces), 12)
        self.assertEqual(len(spike_traces), 12)
        self.assertTrue(all(trace.active for trace in route_traces))
        self.assertTrue(all(trace.active for trace in active_traces))


if __name__ == "__main__":
    unittest.main()
