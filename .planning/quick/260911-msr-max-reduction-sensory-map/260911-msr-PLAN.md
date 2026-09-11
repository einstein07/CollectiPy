---
phase: quick-260911-msr
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/models/mean_field_systems.py
  - src/models/movement/mean_field_model.py
  - src/dataHandling.py
  - tests/test_sensory_map_reduction.py
  - tests/fixtures/sensory_map_sum_pre_change.npz
  - scripts/compare_map_reduction.py
  - scripts/map_reduction/{__init__,factors,config_patch,metrics,run_cell,aggregate,plots}.py
  - scripts/map_reduction/README.md
  - scripts/map_reduction/RESULTS.md
  - slurm/map_reduction.sbatch
autonomous: true
requirements: [QUICK-260911-msr]
spec: max-sensory-map-spec.md

must_haves:
  truths:
    - "MeanFieldSystem.compute_sensory_map combines the per-target von Mises bumps by a configurable reduction over targets: sum (default, the original `vm @ q` expression on its original code path), max, or pnorm with exponent p >= 1"
    - "Every existing config in config/ that exercises the ring map produces a map np.array_equal to the pre-change output, with the `sensory_map` block absent and with `{reduction: sum}` present (fixture generated from commit cb4a50d before the edit)"
    - "The reduction sits exactly where the sum sat: sensory noise on the qualities is applied before it, the guard term is added after it, the 1/sqrt(n) scaling and everything else (detector, readout, kernel, kappa, u, beta, v, integration) are untouched; the DDM/LCA paths do not build the ring map and are unaffected"
    - "mean_field_model.sensory_map = {reduction, p} is validated at construction, defaults to sum, is passed through MeanFieldMovementModel.reset(), and is stamped into get_spin_system_data(), the run's config.json and the map_reduction column of <agent>_sensory_noise.csv"
    - "The same trial seed gives the same sigma_s draws, the same shared percept and the same internal-noise generator state under sum and max (4.5)"
    - "Section 5 tests 5.1-5.7 pass on a 3600-node ring (5.1 on n = 30); the full suite shows no failures beyond the pre-change baseline"
    - "scripts/compare_map_reduction.py plots the maps (a), runs/collects the 6.3 design (b), writes per-trial and per-cell tables (c), plots P(mean_of_pair) and P(arrive at C) vs Delta_close (d); slurm/map_reduction.sbatch runs the same tasks as a bwUniCluster array; the 6.6 two-target check is a subcommand"
  artifacts:
    - path: "src/models/mean_field_systems.py"
      provides: "SENSORY_MAP_REDUCTIONS, normalize_sensory_map_config, reduce_contributions, von_mises_contributions, sensory_map; MeanFieldSystem(sensory_map_reduction, sensory_map_p)"
      contains: "reduce_contributions"
    - path: "tests/test_sensory_map_reduction.py"
      provides: "5.1 regression against the pre-change fixture, 5.2-5.7 analytic checks, config validation, metadata stamping, 4.5 seed pairing"
      contains: "test_5_1_sum_is_bit_identical_to_pre_change_snapshot"
    - path: "scripts/compare_map_reduction.py"
      provides: "maps | manifest | run | aggregate | plot | sanity | all"
      contains: "cmd_sanity"
    - path: "slurm/map_reduction.sbatch"
      provides: "submission + execution mode array script, one task per (cell, chunk)"
      contains: "SLURM_ARRAY_TASK_ID"
---

# Quick task 260911-msr: max-reduction sensory map

Implements `max-sensory-map-spec.md` end to end: the configurable reduction in the
ring-attractor sensory map (default `sum`, bit-identical), the Section 5 tests with a
pre-change fixture, and the Section 6 validation experiment (scripts, SLURM array,
results note).

## Tasks

1. Snapshot the pre-change map on every ring config (`tests/fixtures/sensory_map_sum_pre_change.npz`) BEFORE touching the model.
2. Add the reduction to `MeanFieldSystem` (module-level helpers + constructor kwargs), keeping the `sum` path on its original `vm @ q` line.
3. Parse `mean_field_model.sensory_map` in `MeanFieldMovementModel`, pass it through `reset()`, stamp it into the per-tick record; append `map_reduction` to `_sensory_noise.csv`.
4. Tests 5.1-5.7 + Section 4 plumbing + 4.5 seed pairing; full suite vs the pre-change baseline.
5. Experiment package `scripts/map_reduction/` + CLI `scripts/compare_map_reduction.py` + `slurm/map_reduction.sbatch`; smoke run (n = 20), local full run (n = 200), 6.6 check; `RESULTS.md`.
