#!/usr/bin/env python3

from __future__ import annotations

import csv
from pathlib import Path
import unittest

import sat_curriculum


ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = ROOT / "policy_modules"


class PolicyModuleTests(unittest.TestCase):
    def test_each_curriculum_seed_has_a_module_directory(self) -> None:
        expected = {seed.name for seed in sat_curriculum.SEEDS}
        actual = {
            path.name
            for path in MODULE_ROOT.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        }

        self.assertEqual(actual, expected)

    def test_modules_have_required_files(self) -> None:
        for seed in sat_curriculum.SEEDS:
            with self.subTest(seed=seed.name):
                module = MODULE_ROOT / seed.name
                self.assertTrue((module / "README.md").is_file())
                self.assertTrue((module / "dataset_slice.csv").is_file())
                self.assertTrue((module / "operator_priors.csv").is_file())

    def test_operator_priors_match_runtime_seed_priors(self) -> None:
        for seed in sat_curriculum.SEEDS:
            with self.subTest(seed=seed.name):
                with (MODULE_ROOT / seed.name / "operator_priors.csv").open() as handle:
                    rows = list(csv.DictReader(handle))

                operators = tuple(row["operator"] for row in rows)
                priors = tuple(float(row["prior"]) for row in rows)

                self.assertEqual(operators, sat_curriculum.EFFECT_BASIS)
                self.assertEqual(priors, seed.concentration_prior)
                self.assertAlmostEqual(sum(priors), 1.0)

    def test_dataset_slice_files_define_selection_criteria(self) -> None:
        for seed in sat_curriculum.SEEDS:
            with self.subTest(seed=seed.name):
                with (MODULE_ROOT / seed.name / "dataset_slice.csv").open() as handle:
                    rows = list(csv.DictReader(handle))

                self.assertGreaterEqual(len(rows), 3)
                self.assertEqual(
                    set(rows[0].keys()),
                    {"metric", "operator", "threshold", "description"},
                )


if __name__ == "__main__":
    unittest.main()
