# Policy Modules

Each directory is a dataset-born curriculum seed module. A module is not a
fully trained policy yet; it is the benchmarkable seed of one.

Required files:

- `README.md`: human explanation of the solver-state regime.
- `dataset_slice.csv`: measurable selection criteria for collecting examples.
- `operator_priors.csv`: seed priors over the current operator effect basis.

Current operator effect basis:

```text
pressure, bridge, loop_escape, memory
```

Runtime policy name:

```text
curriculum_seeds
```

The live seed catalog is defined in `sat_curriculum.py`. These modules are the
documentation and data-contract side of that catalog.
