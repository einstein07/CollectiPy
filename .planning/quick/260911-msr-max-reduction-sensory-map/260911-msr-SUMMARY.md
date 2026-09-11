---
phase: quick-260911-msr
plan: 01
subsystem: simulation
tags: [ring-attractor, sensory-map, von-mises, max-reduction, close-targets, sweep, slurm]

# Dependency graph
requires:
  - phase: shared sensory stream (src/models/percept_stream.py) and the arena-seeded RA generators
    provides: paired trials across swept parameters (the same seed -> the same percept and internal noise)
  - phase: (u_hat, v) sweep scaffolding (scripts/uhat_v_sweep, slurm/uhat_v_sweep.sbatch)
    provides: one-file-per-task array runner, idempotent tasks, manifest-driven geometry
provides:
  - "mean_field_model.sensory_map = {reduction: sum|max|pnorm, p}: the reduction over targets when the von Mises bumps are combined into the ring input; default sum, bit-identical to before"
  - "MeanFieldSystem(sensory_map_reduction, sensory_map_p) + module-level reduce_contributions / von_mises_contributions / sensory_map (the spec's reference map)"
  - "tests/test_sensory_map_reduction.py (Section 5 battery + Section 4 plumbing + 4.5 seed pairing) and the pre-change fixture tests/fixtures/sensory_map_sum_pre_change.npz"
  - "scripts/compare_map_reduction.py + scripts/map_reduction/ + slurm/map_reduction.sbatch: the Section 6 experiment (maps figure, 42-cell paired design, per-trial/per-cell tables, outcome plots, 6.6 sanity check, results note)"
affects: [ra-ddm-comparison, mean-field-configs, ring-attractor-readout]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Reduction dispatch AROUND the historical expression, never a rewrite of it: the sum path keeps `vm @ q` so a bit-identical regression fixture is meaningful"
    - "Pre-change fixture generated from the untouched tree BEFORE the edit (module docstring records how to regenerate from commit cb4a50d)"
    - "Local parallel driver = one subprocess per task (the simulator forks agent processes; a daemonic Pool worker cannot)"
    - "Post-hoc commitment from the per-tick logs (<agent>_neural.csv + <agent>_targets.csv), gated on a unimodal ring state, because the runtime detector fires on the agent heading and the circular mean of a two-bump state is the midpoint"

key-files:
  created:
    - tests/test_sensory_map_reduction.py
    - tests/test_map_reduction_metrics.py
    - tests/fixtures/sensory_map_sum_pre_change.npz
    - scripts/compare_map_reduction.py
    - scripts/map_reduction/{__init__,factors,config_patch,metrics,run_cell,aggregate,plots}.py
    - scripts/map_reduction/README.md
    - scripts/map_reduction/RESULTS.md (+ results/)
    - slurm/map_reduction.sbatch
  modified:
    - src/models/mean_field_systems.py
    - src/models/movement/mean_field_model.py
    - src/dataHandling.py (trailing `map_reduction` column of <agent>_sensory_noise.csv)

key-decisions:
  - "Qualities on the campaign's strength scale (5.0 x relative q) so the ring sees the current campaign's amplitude and SNR; Section 5's unit tests use q = 1 as written"
  - "sigma = 0.1 as Section 6.2 fixes it (templates carry 1.5); the 6.6 check runs both sigmas"
  - "Arena side 2 for the three-target design: on the template's unit square, A at bearing 0 / range 0.5 sits ON the wall"
  - "T_max = 100 ticks; paired seeds 20260911 + i in every cell and both sanity arms, used for the arena seed AND the percept-stream seed"
  - "Commitment class read from the logs, gated on a unimodal ring state over 2 consecutive ticks (the first smoke run without the gate returned P(mean of pair) = 1 at every Delta, 60 degrees included)"

findings:
  - "At the RING level max does what the spec says: under sum the ring forms one bump at the pair's mean bearing for Delta_close <= 25 degrees (the 25.5-degree threshold), under max it keeps two bumps from 20 degrees up and merges only at 12-15 degrees (the 12-degree node spacing = the RA's own resolution, as 2.3 anticipates)"
  - "At the BEHAVIOUR level the two maps are indistinguishable in every cell: the runtime readout is the unthresholded circular mean (use_thresholding false), and the circular mean of a symmetric two-bump state is the midpoint, so heading, trajectory and the tick at which symmetry breaks are the same under both maps; P(arrive at C) does not track q_C under either map"
  - "6.6: at 60 degrees, dQ 1 %, u 6.0, sum and max make the identical choice in 200/200 paired trials at both sigma = 1.5 (campaign) and sigma = 0.1 (spec); accuracy shift 0, median arrival shift 1e-4 tick"
  - "Numbers: scripts/map_reduction/RESULTS.md and results/cells.csv (n = 200 paired trials per cell, run locally on 16 workers; the SLURM array is provided but was not submitted)"

deferred:
  - "The close-target failure mode survives the max map because of the CoM readout; a thresholded / peak readout would be the next lever, but readout changes are outside this spec (4.2, 8)"
  - "BifurcationDetector 'behavioral' mode keys on the agent heading (see deferred-items.md); untouched here"
---

# Quick task 260911-msr — max-reduction sensory map: summary

What was built, in one paragraph: the ring attractor's sensory map can now be built
by `sum` (default, unchanged bit for bit), `max` or `pnorm` over targets, from a
`sensory_map` block in `mean_field_model`, recorded in every run's config and
per-tick log; a 32-test battery (with a fixture snapshotted from the pre-change
tree) covers the spec's Section 5; and a self-contained experiment package runs
the Section 6 close-target design (sum vs max, 7 separations x 3 qualities of the
third target, 200 paired trials per cell) locally or as a bwUniCluster array, with
per-trial metrics read from the simulator's own logs.

The result is in `scripts/map_reduction/RESULTS.md`. Short form: `max` removes the
fictitious super-target at the ring level exactly as predicted (down to the ring's
node spacing), but the unthresholded circular-mean readout steers to the midpoint
of a two-bump state just as it does for a merged bump, so nothing behavioural
changes — and the existing 60-degree results are untouched (identical choices in
every paired trial).
