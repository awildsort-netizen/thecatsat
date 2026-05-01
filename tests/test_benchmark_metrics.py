#!/usr/bin/env python3

from __future__ import annotations

import unittest

import benchmark_calorimeter


class BenchmarkMetricTests(unittest.TestCase):
    def test_operator_metric_prefix_strips_private_marker(self) -> None:
        self.assertEqual(
            benchmark_calorimeter.operator_metric_prefix("_fiber_memory_bias"),
            "trace_fiber_memory_bias",
        )


if __name__ == "__main__":
    unittest.main()
