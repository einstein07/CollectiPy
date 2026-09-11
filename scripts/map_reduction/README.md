# Sensory-map reduction — sum vs max on close targets

Implements `max-sensory-map-spec.md` Sections 6 and 7. The model change itself
(the `sensory_map` block of `mean_field_model`) lives in
`src/models/mean_field_systems.py` and `src/models/movement/mean_field_model.py`;
its tests are `tests/test_sensory_map_reduction.py`. This directory is the
validation experiment: the three-target geometry with two close targets, run with
`reduction: sum` and `reduction: max` on paired seeds.

## What this measures

Two targets closer than the von Mises width sum into one input bump at their
mean bearing with amplitude approaching 2q (the merge threshold at κ = 20 is
Δ\* ≈ 25.5°). The ring is then handed a fictitious super-target: the decision
is mislocated to the mean bearing and the pair out-competes a third target by
amplitude alone. The `max` reduction keeps both peaks (a cusp of depth
r(Δ) = exp(κ(cos(Δ/2) − 1)) between them) and bounds the map by the best
quality. This experiment asks whether that removes the close-target failure
mode in the closed loop:

```
reduction     {sum, max}
Δ_close       {12, 15, 20, 25, 30, 40, 60}°     A at 0°, B at +Δ_close, C at −120°
q_C           {1.00, 1.01, 1.02}               q_A = q_B = 1.00 (×5.0 on the config scale)
trials        20 per cell (smoke), 200 per cell (full), the SAME seed list in every cell
```

What `max` cannot fix is the ring's own angular resolution (bump width set by
`v` and `u`): below it the network settles on the mean bearing whatever the
input looks like. That residual is reported, not tuned around (`κ`, the kernel,
`u` and the noise model are all out of scope).

## Files

| file | role |
|---|---|
| `factors.py` | THE single source: factor grids, geometry, seeds, locked 6.2 settings, decisions |
| `config_patch.py` | template → cell config → trial config (+ the 6.6 two-target configs), post-patch assertions, manifest |
| `metrics.py` | one run archive → the 6.4 metrics (midpoint-aware commitment, CoM, arrival) |
| `run_cell.py` | one array task: (cell × chunk) → one row file; scratch staging; idempotent; failures are data |
| `aggregate.py` | `raw/` → `trials.parquet`/`trials.csv` + `cells.csv` + `paired_contrasts.csv`; the 6.6 summary |
| `plots.py` | (a) the maps figure, (d) the outcome curves, plus secondary outcomes |
| `../compare_map_reduction.py` | the CLI front-end: `maps`, `manifest`, `run`, `aggregate`, `plot`, `sanity`, `report`, `all` |
| `results/` | the tracked copy of the summary tables and figures the results note refers to (`report` writes it) |
| `../../slurm/map_reduction.sbatch` | bwUniCluster job array (submission + execution mode), same conventions as `slurm/uhat_v_sweep.sbatch` |
| `../../config/qd_sweep_ra_template.json` | the RA-arm template of the current campaign, never modified in place |
| `RESULTS.md` | the results note (7.5): summary table, the two plots, the 6.6 check |

## Output layout

```
<results-root>/                       default results/map_reduction  (--smoke: .../smoke)
├── manifest.json                     42 cells, paired seeds, locked settings, config hashes
├── figures/maps_sum_vs_max.{png,pdf}         (a)
├── raw/cell<K>_chunk<J>.parquet      ONE file per task — never appended to
├── trials.parquet / trials.csv       every trial, one row each (aggregate)
├── cells.csv                         per-cell summary with Wilson intervals
├── paired_contrasts.csv              max − sum per (Δ_close, q_C), paired by seed
├── figures/outcomes_vs_delta.{png,pdf}       (d)  P(mean_of_pair), P(arrive at C)
├── figures/outcomes_extra_vs_delta.{png,pdf} ring shape on tick 2, P(timeout), P(no commitment), P(arrive A|B)
└── sanity/                           6.6: raw/, sanity_trials.csv, sanity_arms.csv, sanity_paired.csv
```

## Commands, in order

```bash
cd /home/sindiso/Documents/PhD/ring-attractor/CollectiPy
PY=.venv/bin/python
```

### 1. Smoke run (local, ~1 min on 16 workers)

```bash
$PY scripts/compare_map_reduction.py all --smoke --workers 16
```

Writes `results/map_reduction/smoke/` end to end (manifest with n = 20, the maps
figure, 840 trials, tables, plots). Statistics at n = 20 validate the pipeline
only.

### 2. Full run

Locally (~8 400 trials, ≈ 10 min on 16 workers):

```bash
$PY scripts/compare_map_reduction.py manifest
$PY scripts/compare_map_reduction.py run --all --workers 16
```

or as a SLURM array on bwUniCluster (**you submit, not the agent**):

```bash
$PY scripts/compare_map_reduction.py manifest --results-root <R>
bash slurm/map_reduction.sbatch                # plan only
TEST_ONLY=1 bash slurm/map_reduction.sbatch    # sbatch --test-only
SUBMIT=1 bash slurm/map_reduction.sbatch       # 42 tasks, one cell each
ONLY=7 SUBMIT=1 bash slurm/map_reduction.sbatch   # one cell
```

Re-running is free: a task whose output file is complete exits 0 immediately,
so a partial array is fixed by resubmitting the same command.

### 3. Aggregate and plot

```bash
$PY scripts/compare_map_reduction.py aggregate [--results-root <R>]
$PY scripts/compare_map_reduction.py plot      [--results-root <R>]
```

### 4. The 6.6 sanity check (local, 4 arms × 200 paired trials, ~2 min)

```bash
$PY scripts/compare_map_reduction.py sanity --workers 4
```

Two-target standard condition (Δ = 60°, δ_Q = 1 %, u = 6.0): `sum` vs `max` on
the paired seed list, at the template's σ (`campaign`, what the existing
results were produced with) and at the spec's σ = 0.1 (`spec`). Accuracy and
median arrival time must shift by less than the paired-trial standard error.

### 5. Results note

```bash
$PY scripts/compare_map_reduction.py report      # -> scripts/map_reduction/results/
```

copies `cells.csv`, `paired_contrasts.csv`, the sanity tables, the manifest and
the figures next to `RESULTS.md`, and prints the per-cell table as markdown.

## Per-trial metrics (6.4) — what the columns mean

| column | meaning |
|---|---|
| `t_commit_ticks`, `commit_class` | **midpoint-aware commitment**: the first logged tick from which the ring state is *unimodal* (one local maximum above half the peak activity) and its CoM is within 5° (the detector's own tolerance) of A, B, C or the A–B midpoint, with the same class on 2 consecutive ticks; class `mean_of_pair` if the CoM is closer to the midpoint than to either A or B, else the nearest target |
| `com_commit_deg`, `midpoint_commit_deg`, `bearing_{A,B,C}_commit_deg`, `n_peaks_commit` | the CoM, the A–B midpoint, the bearings and the bump count on that tick (egocentric, degrees) |
| `n_peaks_t1`, `n_peaks_t2`, `state_t2`, `t_first_unimodal` | the ring's shape on ticks 1 and 2 (`state_t2`: the gated class, or `bimodal`), and the first tick ≥ 2 with a single bump — the direct evidence of whether the map handed the ring one merged bump or two |
| `class_final`, `com_final_deg`, `n_peaks_final` | the same classification at the last logged tick, no tolerance, no gate |
| `arrived`, `reached`, `t_arrival_ticks`, `t_arrival_fine` | first tick inside the 0.05 m arrival radius; the sub-tick value intersects the logged segment with the circle |
| `timeout` | no arrival within `T_max` = 100 ticks |
| `t_bif_ticks`, `bif_target` | the `BifurcationDetector`'s own first event, kept for cross-reference only (see below) |

`compute_center_of_mass` is imported from the simulator, on the ring state the
run logged (`<agent>_neural.csv`); the bearings come from the same tick's
`<agent>_targets.csv`, i.e. the input that drove that tick's integration.

**Why the classification is gated on a unimodal ring state.** The runtime
readout is the unthresholded circular mean of the ring (`use_thresholding:
false`), and the circular mean of a *symmetric two-bump* state is the A–B
midpoint too. Classifying the raw CoM would therefore call every undecided ring
"mean of pair" — it did, in the first smoke run, at every Δ_close including 60°.
The gate separates "one merged bump at the midpoint" (the failure mode) from
"two bumps, not yet decided"; the two-tick streak absorbs the onset transient
(the ring starts from zero and its first tick is not yet its response shape).

**Why the detector's own event is not the commitment measure here.** In
`behavioral` mode the detector dispatches to `_update_behavioral_agent_angle`,
which fires when the agent *heading* (always 0 in the egocentric frame) is within
5° of a target bearing. With A placed at bearing 0° it therefore fires on tick 1
naming A in every trial — and, as the spec notes, it can only ever name a real
target. The midpoint-aware classification above is the quantity the detector
cannot see. The detector is untouched (4.2).

## Decisions (report-first)

- **Quality scale ×5.** The spec states qualities relative to A/B; the campaign
  this experiment sits beside expresses the same 1–2 % differences on a strength
  scale of 5.0 (5.0 vs 4.95), with the shared sensory noise (`white_rate`
  0.0707) calibrated on that scale. Config strengths are `5.0 × q_rel`, so the
  ring sees the campaign's input amplitude and signal-to-noise ratio. The Section
  5 unit tests use q = 1 exactly as written.
- **σ = 0.1** as 6.2 fixes it. Every RA template in `config/` carries 1.5
  (`scripts/uhat_v_sweep/RECON.md` D-02 records that spec/repo drift). The 6.6
  check runs both values so the existing 60° results are compared at their own
  σ.
- **Arena side 2** for the three-target design. With the template's unit
  square, A (bearing 0°, range 0.5) sits exactly on the wall and an agent parked
  between A and B is clamped by the wall rather than steered by the map. Side 2
  (the DDM campaign's choice for wide placements) takes the wall out of the
  dynamics; `config_patch` asserts every target is reachable inside the clamp
  box. The 6.6 check keeps the template's arena.
- **T_max = 100 ticks** (10× the direct travel time at 0.05 m/tick, the horizon
  of the (û, v) sweep). Timeouts are an outcome (6.4), so the horizon is part of
  the design and is recorded in the manifest.
- **Paired seeds** `seed_i = 20260911 + i`, the same list in every cell and in
  the sanity arms, used for both the arena `random_seed` and
  `sensory_stream.seed` (4.5: trial identity independent of the swept factors).
  The same seed produces the same percept stream and the same internal-noise
  draws under `sum` and `max` (tested in `tests/test_sensory_map_reduction.py`).
- **Bearing convention.** GPS bearing = atan2(−dy, dx) − heading (the plant is
  y-down), so a target at bearing b is placed at (r cos b, −r sin b). The logged
  bearings on tick 1 (A = 0°, B = +Δ, C = −120°) confirm the placement.
- **Where the reduction is recorded.** `mean_field_model.sensory_map` lands in
  the run's `config.json`, in `results.sweep_metadata`, and in every row of
  `<agent>_sensory_noise.csv` (`map_reduction` column).

## Out of scope

Changing κ, the kernel, u, the noise model, the detector, or any DDM/LCA code;
reinterpreting the existing 60° results; any smooth-max tuning beyond exposing
`p` (`reduction: pnorm` is implemented and tested but not part of the design).
