#!/usr/bin/env python3

from __future__ import annotations

import unittest

import benchmark_calorimeter
import sat_furnace


class BenchmarkMetricTests(unittest.TestCase):
    def test_operator_metric_prefix_strips_private_marker(self) -> None:
        self.assertEqual(
            benchmark_calorimeter.operator_metric_prefix("_fiber_memory_bias"),
            "trace_fiber_memory_bias",
        )

    def test_puzzle_ecology_metrics_detect_composition_border(self) -> None:
        formula = [
            ((0, False), (1, True), (2, False)),
            ((0, True), (1, False), (2, True)),
            ((0, False), (2, True)),
        ]

        metrics = benchmark_calorimeter.puzzle_ecology_metrics(
            formula=formula,
            variables=3,
            random_best_unsatisfied=2,
            walksat_best_unsatisfied=0,
            furnace_best_unsatisfied=1,
            random_solved=False,
            walksat_solved=True,
            furnace_solved=False,
        )

        self.assertEqual(metrics["puzzle_ecology_niche"], "border")
        self.assertGreater(float(metrics["puzzle_border_score"]), 0.62)
        self.assertGreater(float(metrics["puzzle_composition_pressure"]), 0.0)
        self.assertEqual(float(metrics["puzzle_variable_coverage"]), 1.0)

    def test_solver_composition_genome_exposes_operator_genes(self) -> None:
        genome = benchmark_calorimeter.solver_composition_genome()
        gene_names = {gene.name for gene in genome.genes}

        self.assertIn("solver.clause_pressure", gene_names)
        self.assertIn("solver.spin_update", gene_names)
        self.assertEqual(genome.missing_inputs, ())
        self.assertGreater(len(genome.genes), 10)
        self.assertGreater(len(genome.edges), 0)

    def test_composition_genome_metrics_score_border_fit(self) -> None:
        genome = benchmark_calorimeter.solver_composition_genome()
        metrics = benchmark_calorimeter.composition_genome_metrics(
            genome=genome,
            puzzle_border_score=0.8,
            puzzle_composition_pressure=0.5,
            solved=True,
            furnace_best_unsatisfied=0,
            walksat_best_unsatisfied=2,
        )

        self.assertEqual(float(metrics["composition_missing_inputs"]), 0.0)
        self.assertGreater(float(metrics["composition_gene_count"]), 10.0)
        self.assertGreater(float(metrics["composition_border_fit"]), 0.0)
        self.assertGreater(float(metrics["composition_genome_fitness"]), 0.0)
        self.assertIn("solver.clause_pressure", str(metrics["composition_gene_sequence"]))

    def test_gene_border_mutation_selects_active_gene_border(self) -> None:
        genome = benchmark_calorimeter.solver_composition_genome()
        traces = [
            sat_furnace.OperatorTrace(
                t=0,
                operator="spin_update",
                active=True,
                action="baseline",
                reason="test",
                input_heat=0.0,
                input_entropy=0.0,
                input_integration=0.0,
                input_unsatisfied=3,
                output_mean=0.0,
                output_peak=2.0,
                memory_scale=0.0,
                delta_unsatisfied=-1,
                delta_integration=0.0,
            ),
            sat_furnace.OperatorTrace(
                t=1,
                operator="adaptive_gate",
                active=False,
                action="baseline",
                reason="test",
                input_heat=0.0,
                input_entropy=0.0,
                input_integration=0.0,
                input_unsatisfied=3,
                output_mean=0.0,
                output_peak=0.0,
                memory_scale=0.0,
                delta_unsatisfied=0,
                delta_integration=0.0,
            ),
        ]

        metrics = benchmark_calorimeter.gene_border_mutation_metrics(
            genome=genome,
            traces=traces,
            puzzle_border_score=0.9,
            puzzle_composition_pressure=0.7,
            solved=False,
            furnace_best_unsatisfied=2,
            walksat_best_unsatisfied=0,
        )

        self.assertGreater(float(metrics["gene_border_candidate_count"]), 0.0)
        self.assertNotEqual(metrics["gene_border_selected_gene"], "none")
        self.assertGreater(float(metrics["gene_border_selected_score"]), 0.0)
        self.assertIn(
            str(metrics["gene_border_selected_mutation"]),
            {
                "instrument_or_expose",
                "inhibit_or_rescale",
                "lower_activation_border",
                "retune_gate",
                "mutate_memory_decay",
                "reweight_transform",
                "increase_resolution",
                "recombine_provider",
                "tighten_constraint",
                "mutate_gene",
            },
        )
    def test_transition_motif_metrics_detect_ordered_productive_edge(self) -> None:
        traces = [
            sat_furnace.OperatorTrace(
                t=0,
                operator="anneal",
                active=True,
                action="soften",
                reason="test",
                input_heat=0.4,
                input_entropy=0.6,
                input_integration=0.1,
                input_unsatisfied=4,
                output_mean=0.2,
                output_peak=0.4,
                memory_scale=0.0,
                delta_unsatisfied=1,
                delta_integration=0.1,
            ),
            sat_furnace.OperatorTrace(
                t=1,
                operator="flip",
                active=True,
                action="repair",
                reason="test",
                input_heat=0.3,
                input_entropy=0.5,
                input_integration=0.3,
                input_unsatisfied=2,
                output_mean=0.5,
                output_peak=0.8,
                memory_scale=0.0,
                delta_unsatisfied=2,
                delta_integration=0.2,
            ),
            sat_furnace.OperatorTrace(
                t=2,
                operator="anneal",
                active=True,
                action="soften",
                reason="test",
                input_heat=0.5,
                input_entropy=0.8,
                input_integration=0.2,
                input_unsatisfied=3,
                output_mean=0.1,
                output_peak=0.2,
                memory_scale=0.0,
                delta_unsatisfied=0,
                delta_integration=-0.1,
            ),
        ]

        metrics = benchmark_calorimeter.transition_motif_metrics(traces)
        motifs = benchmark_calorimeter.transition_motifs(traces)
        motif_edges = {(motif.source, motif.target) for motif in motifs}

        self.assertGreater(float(metrics["transition_motif_count"]), 0.0)
        self.assertIn(("anneal", "flip"), motif_edges)
        self.assertIn(("flip", "anneal"), motif_edges)
        self.assertNotIn("transition_motif_best", metrics)
        self.assertNotIn("transition_motif_best_score", metrics)
        self.assertGreater(float(metrics["transition_motif_persistence"]), 0.0)
        self.assertIn("transition_motif_roles", metrics)

    def test_transition_motif_bootstrap_composes_needed_effects(self) -> None:
        motifs = (
            benchmark_calorimeter.TransitionMotif(
                source="anneal",
                target="lift",
                count=2,
                activation_rate=1.0,
                mean_delta_unsatisfied=0.0,
                mean_delta_integration=0.0,
                entropy_shift=0.2,
                persistence=0.1,
                role="thermal_softening",
            ),
            benchmark_calorimeter.TransitionMotif(
                source="lift",
                target="repair",
                count=2,
                activation_rate=1.0,
                mean_delta_unsatisfied=1.0,
                mean_delta_integration=0.2,
                entropy_shift=0.0,
                persistence=0.4,
                role="bridge_building",
            ),
            benchmark_calorimeter.TransitionMotif(
                source="repair",
                target="settle",
                count=1,
                activation_rate=1.0,
                mean_delta_unsatisfied=1.0,
                mean_delta_integration=0.2,
                entropy_shift=-0.1,
                persistence=0.3,
                role="puncture_and_seal",
            ),
        )

        plan = benchmark_calorimeter.bootstrap_motif_plan(motifs)

        self.assertEqual(plan.targets, ("stabilization_window",))
        self.assertEqual(plan.missing, ())
        self.assertIn("entropy_release", plan.provided_effects)
        self.assertIn("bridge_opportunity", plan.provided_effects)
        self.assertIn("stabilization_window", plan.provided_effects)
        self.assertEqual(plan.provider_count, 3)

    def test_transition_motif_bootstrap_uses_climate_needs(self) -> None:
        motifs = (
            benchmark_calorimeter.TransitionMotif(
                source="observe",
                target="wander",
                count=1,
                activation_rate=1.0,
                mean_delta_unsatisfied=0.0,
                mean_delta_integration=0.0,
                entropy_shift=0.0,
                persistence=0.0,
                role="drift",
            ),
        )
        climate = {
            "puzzle_ecology_niche": "border",
            "puzzle_border_score": 0.8,
            "puzzle_composition_pressure": 0.7,
            "trace_trap_contribution": 0.3,
            "solved": False,
        }

        plan = benchmark_calorimeter.bootstrap_motif_plan(
            motifs,
            climate_metrics=climate,
        )

        self.assertEqual(
            plan.targets,
            ("bridge_opportunity", "puncture_repair_cycle", "entropy_release"),
        )
        self.assertIn("bridge_opportunity", plan.missing)
        self.assertIn("puncture_repair_cycle", plan.missing)
        self.assertIn("entropy_release", plan.missing)
        self.assertEqual(plan.provider_count, 0)

    def test_transition_motif_metrics_reports_climate_missing_providers(self) -> None:
        traces = [
            sat_furnace.OperatorTrace(
                t=0,
                operator="observe",
                active=True,
                action="test",
                reason="test",
                input_heat=0.0,
                input_entropy=0.0,
                input_integration=0.0,
                input_unsatisfied=1,
                output_mean=0.0,
                output_peak=0.0,
                memory_scale=0.0,
                delta_unsatisfied=0,
                delta_integration=0.0,
            ),
            sat_furnace.OperatorTrace(
                t=1,
                operator="wander",
                active=True,
                action="test",
                reason="test",
                input_heat=0.0,
                input_entropy=0.0,
                input_integration=0.0,
                input_unsatisfied=1,
                output_mean=0.0,
                output_peak=0.0,
                memory_scale=0.0,
                delta_unsatisfied=0,
                delta_integration=0.0,
            ),
        ]

        metrics = benchmark_calorimeter.transition_motif_metrics(
            traces,
            climate_metrics={
                "puzzle_ecology_niche": "border",
                "puzzle_border_score": 0.8,
                "puzzle_composition_pressure": 0.7,
                "trace_trap_contribution": 0.3,
                "solved": False,
            },
        )

        self.assertIn("bridge_opportunity", str(metrics["motif_bootstrap_targets"]))
        self.assertIn("puncture_repair_cycle", str(metrics["motif_bootstrap_missing"]))
        self.assertEqual(float(metrics["motif_bootstrap_provider_count"]), 0.0)

    def test_motif_rules_are_table_driven(self) -> None:
        self.assertEqual(
            benchmark_calorimeter.transition_motif_role(
                entropy_shift=0.05,
                persistence=0.0,
                released_tension=0.0,
            ),
            "thermal_softening",
        )
        self.assertEqual(
            benchmark_calorimeter.motif_effect_outputs("bridge_building"),
            ("bridge_opportunity",),
        )
        self.assertEqual(
            benchmark_calorimeter.motif_effect_requirements("bridge_building"),
            ("entropy_release",),
        )
        self.assertEqual(
            benchmark_calorimeter.motif_effect_outputs("unknown_role"),
            ("motif_observation",),
        )

    def test_motif_bootstrap_pressure_from_missing_providers(self) -> None:
        plan = benchmark_calorimeter.MotifBootstrapPlan(
            targets=("entropy_release", "bridge_opportunity", "puncture_repair_cycle"),
            order=(),
            missing=("entropy_release", "bridge_opportunity", "puncture_repair_cycle"),
            provided_effects=(),
            provider_count=0,
        )

        pressure = benchmark_calorimeter.motif_bootstrap_pressure(plan)

        self.assertEqual(float(pressure["motif_pressure_entropy_release"]), 1.0)
        self.assertEqual(float(pressure["motif_pressure_bridge_opportunity"]), 1.0)
        self.assertEqual(float(pressure["motif_pressure_puncture_repair_cycle"]), 1.0)
        self.assertEqual(float(pressure["motif_pressure_stabilization_window"]), 0.0)
        self.assertGreater(float(pressure["motif_pressure_total"]), 0.0)
        self.assertEqual(float(pressure["motif_pressure_target_missing_count"]), 3.0)
        self.assertEqual(float(pressure["motif_pressure_prerequisite_missing_count"]), 0.0)
        self.assertEqual(pressure["motif_pressure_action_hint"], "explore_puncture_repair")

    def test_motif_bootstrap_pressure_distinguishes_prerequisites(self) -> None:
        plan = benchmark_calorimeter.MotifBootstrapPlan(
            targets=("bridge_opportunity",),
            order=(),
            missing=("entropy_release",),
            provided_effects=(),
            provider_count=0,
        )

        pressure = benchmark_calorimeter.motif_bootstrap_pressure(plan)

        self.assertEqual(float(pressure["motif_pressure_target_missing_count"]), 0.0)
        self.assertEqual(float(pressure["motif_pressure_prerequisite_missing_count"]), 1.0)
        self.assertEqual(pressure["motif_pressure_target_missing"], "none")
        self.assertEqual(pressure["motif_pressure_prerequisite_missing"], "entropy_release")
        self.assertEqual(pressure["motif_pressure_action_hint"], "prepare_entropy_release")

    def test_motif_bootstrap_pressure_is_quiet_when_complete(self) -> None:
        plan = benchmark_calorimeter.MotifBootstrapPlan(
            targets=("stabilization_window",),
            order=("motif.0.puncture_and_seal.repair->settle",),
            missing=(),
            provided_effects=("stabilization_window",),
            provider_count=1,
        )

        pressure = benchmark_calorimeter.motif_bootstrap_pressure(plan)

        self.assertEqual(float(pressure["motif_pressure_total"]), 0.0)
        self.assertEqual(pressure["motif_pressure_action_hint"], "none")

    def test_mutation_controls_retune_selected_gene_border(self) -> None:
        candidate = benchmark_calorimeter.GeneMutationCandidate(
            gene="solver.adaptive_gate",
            role="control",
            mutation="retune_gate",
            score=0.75,
            reason="test",
        )

        controls = benchmark_calorimeter.mutation_controls_from_candidate(
            candidate,
            adaptive=False,
            policy="baseline",
            spike_threshold=0.35,
            spike_slope=8.0,
            memory_decay=0.92,
            memory_drive=0.12,
        )
        metrics = benchmark_calorimeter.mutation_control_metrics(controls)

        self.assertTrue(controls.enabled)
        self.assertTrue(controls.adaptive)
        self.assertEqual(controls.policy, "curriculum_seeds")
        self.assertLess(controls.spike_threshold, 0.35)
        self.assertGreater(controls.spike_slope, 8.0)
        self.assertEqual(metrics["mutation_source_gene"], "solver.adaptive_gate")
        self.assertEqual(metrics["mutation_action"], "retune_gate")

    def test_mutant_replay_can_be_disabled(self) -> None:
        controls = benchmark_calorimeter.MutationControls(
            enabled=False,
            mutation="none",
            source_gene="none",
            adaptive=False,
            policy="baseline",
            spike_threshold=0.35,
            spike_slope=8.0,
            memory_decay=0.92,
            memory_drive=0.12,
            learning_rate_scale=1.0,
            inertia_delta=0.0,
            noise_delta=0.0,
        )

        metrics = benchmark_calorimeter.run_mutant_replay(
            formula=[((0, False),)],
            variables=1,
            steps=2,
            seed=0,
            temperature=0.35,
            learning_rate=0.055,
            inertia=0.82,
            noise=0.015,
            planted_assignment=None,
            baseline_best_unsatisfied=1,
            controls=controls,
        )

        self.assertFalse(metrics["mutation_replay_run"])
        self.assertEqual(metrics["mutation_replay_best_unsatisfied"], -1)


if __name__ == "__main__":
    unittest.main()
