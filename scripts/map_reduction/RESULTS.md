# Results note — sum vs max sensory map on close targets

Spec: `max-sensory-map-spec.md` (Sections 6–7). Pipeline, metrics and the recorded
decisions: [`README.md`](README.md). Tables and figures referred to below are in
[`results/`](results/) (written by `compare_map_reduction.py report`); the full
per-cell table is `results/cells.csv` (one row per cell, Wilson intervals on every
proportion) and `results/summary_table.md`.

Run: 42 cells × 200 paired trials (8 400 trials, 0 numerical failures), locally on
16 workers, ~11 min; seeds 20260911 … 20261110 identical in every cell and used
for both the arena seed and the shared percept stream. Manifest git sha:
`cb4a50d…-dirty` (the working tree of commit cb4a50d plus this change set). The
SLURM array (`slurm/map_reduction.sbatch`) is provided and dry-run but was not
submitted — the local run *is* the full n = 200 design, not the smoke run.

## Headline

1. **At the ring level `max` does exactly what the spec predicts.** Under `sum`
   the ring forms one bump at the pair's mean bearing for Δ_close ≤ 25° (the
   25.5° merge threshold): P(mean of pair) = 1.00 / 1.00 / 0.99 / 0.82 at
   12 / 15 / 20 / 25°, 0.38 at 30°, 0 from 40°. Under `max` the ring keeps two
   bumps from Δ_close = 20° up (P(mean of pair) 0.065 at 20°, 0.000 from 25°;
   two bumps on tick 2 in 88–100 % of trials) and merges only at 12–15°, the
   ring's own resolution (12° node spacing) — the residual §2.3 anticipates.
2. **At the behavioural level the two maps are indistinguishable.** In every
   cell P(arrive at C), P(arrive at A or B), P(timeout) and the arrival-time
   medians agree to within ±0.01 / ±0.002 ticks; the paired difference in
   P(arrive at C) and P(timeout) is 0.000 (SE ≤ 0.005) in all 21 (Δ, q_C)
   pairs, and 4 189 of the 4 200 paired trials end at the same target. The
   reason is the runtime readout: with `use_thresholding: false` the heading
   is the circular mean of the whole ring, and the circular mean of a symmetric
   two-bump state is the A–B midpoint — the same command a merged bump gives.
   The "fictitious super-target" is re-created downstream of the map by the
   readout, so the mislocation to the mean bearing survives `max`.
3. **P(arrive at C) does not track q_C under either map**: pooled over
   Δ_close ≥ 30°, 0.003 / 0.003 / 0.008 (max) and 0.003 / 0.003 / 0.010 (sum)
   for q_C = 1.00 / 1.01 / 1.02. Under `sum` the pair wins by amplitude (2q);
   under `max` it wins by count — two activity bumps pull the circular mean
   harder than C's single bump — so C cannot compete on quality at these
   differences.
4. **§6.6 holds.** At Δ = 60°, δ_Q = 1 %, u = 6.0, `sum` and `max` make the
   identical choice in 200 / 200 paired trials at both σ = 1.5 (the template's,
   what the existing results were produced with) and σ = 0.1 (the spec's):
   accuracy shift 0.000, median arrival shift −0.0001 tick. The switch does not
   invalidate the existing RA–DDM results at 60°.

## Setup

- Geometry (6.1): A at bearing 0°, B at +Δ_close, C at −120°, all at range 0.5
  from the start pose; Δ_close ∈ {12, 15, 20, 25, 30, 40, 60}°; q_A = q_B =
  1.00, q_C ∈ {1.00, 1.01, 1.02} on the campaign's strength scale (×5.0).
- Runtime (6.2): n = 30, u = 6.0 (absolute), β = 1.0, v = 0.5, κ = 20,
  integration 50 / 0.1, σ = 0.1, linear velocity 0.05, angular velocity 120°/s,
  arrival radius 0.05, `use_thresholding` false, `scale_velocity` false, shared
  sensory stream (white_rate 0.0707, frozen_sd 0), T_max = 100 ticks, square
  arena of side 2 (see README "Decisions" for the last two).
- Template: `config/qd_sweep_ra_template.json` (the current campaign's RA arm),
  patched by `scripts/map_reduction/config_patch.py`, every 6.2 value asserted.

## Per-cell summary (q_C = 1.00; the q_C = 1.01 and 1.02 rows differ by ≤ 0.02)

| Δ_close | reduction | P(mean of pair) [95 % CI] | P(one bump at midpoint, t2) | P(two bumps, t2) | P(arrive C) | P(arrive A) | P(arrive B) | P(timeout) | P(no commit) | med. t_commit | med. t_arrival |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 12° | sum | 1.000 [0.981, 1.000] | 0.985 | 0.015 | 0.000 | 0.825 | 0.175 | 0.000 | 0.000 | 1 | 10.0 |
| 12° | max | 0.985 [0.957, 0.995] | 0.955 | 0.045 | 0.000 | 0.815 | 0.185 | 0.000 | 0.015 | 1 | 10.0 |
| 15° | sum | 1.000 [0.981, 1.000] | 0.970 | 0.030 | 0.000 | 0.540 | 0.460 | 0.000 | 0.000 | 1 | 9.6 |
| 15° | max | 0.750 [0.686, 0.805] | 0.715 | 0.285 | 0.000 | 0.530 | 0.470 | 0.000 | 0.250 | 1 | 9.6 |
| 20° | sum | 0.990 [0.964, 0.997] | 0.940 | 0.060 | 0.595 | 0.000 | 0.405 | 0.000 | 0.000 | 1 | 25.5 |
| 20° | max | 0.065 [0.038, 0.108] | 0.125 | 0.875 | 0.595 | 0.000 | 0.405 | 0.000 | 0.390 | 11 | 25.5 |
| 25° | sum | 0.820 [0.761, 0.867] | 0.825 | 0.175 | 0.570 | 0.055 | 0.375 | 0.000 | 0.030 | 1 | 25.5 |
| 25° | max | 0.000 [0.000, 0.019] | 0.010 | 0.990 | 0.570 | 0.055 | 0.375 | 0.000 | 0.165 | 11 | 25.5 |
| 30° | sum | 0.380 [0.316, 0.449] | 0.455 | 0.545 | 0.000 | 0.610 | 0.375 | 0.015 | 0.395 | 1 | 14.9 |
| 30° | max | 0.000 [0.000, 0.019] | 0.000 | 1.000 | 0.000 | 0.610 | 0.375 | 0.015 | 0.625 | 9 | 14.9 |
| 40° | sum | 0.000 [0.000, 0.019] | 0.000 | 1.000 | 0.010 | 0.615 | 0.345 | 0.030 | 0.390 | 9 | 16.4 |
| 40° | max | 0.000 [0.000, 0.019] | 0.000 | 1.000 | 0.010 | 0.615 | 0.345 | 0.030 | 0.390 | 9 | 16.4 |
| 60° | sum | 0.000 [0.000, 0.019] | 0.000 | 1.000 | 0.000 | 0.505 | 0.465 | 0.030 | 0.030 | 7 | 10.2 |
| 60° | max | 0.000 [0.000, 0.019] | 0.000 | 1.000 | 0.000 | 0.505 | 0.465 | 0.030 | 0.030 | 7 | 10.2 |

Paired contrast max − sum (`results/paired_contrasts.csv`, trials paired by seed):

| Δ_close | 12° | 15° | 20° | 25° | 30° | 40° | 60° |
|---|---|---|---|---|---|---|---|
| ΔP(mean of pair) ± SE | −0.015 ± 0.009 | −0.25 ± 0.03 | −0.925 ± 0.019 | −0.82 ± 0.027 | −0.38 ± 0.034 | 0.000 | 0.000 |
| ΔP(arrive at C) ± SE | 0.000 | 0.000 | 0.000 | 0.000 | ≤ 0.005 | 0.000 | 0.000 |
| ΔP(timeout) ± SE | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | ≤ 0.005 | 0.000 |

"No commitment" means the gated criterion (one bump, same class on two
consecutive ticks) never held before the run ended: the ring stayed two-bumped,
or was unimodal for a single tick, up to arrival. Those trials still arrive
(P(timeout) ≤ 0.03 in every such cell); they are undecided by the ring-state
yardstick, not by the plant.

## Figures

- [`results/maps_sum_vs_max.png`](results/maps_sum_vs_max.png) — (a) the sum
  and max maps for the 6.1 geometry at each Δ_close on the fine ring and at
  the 30 runtime nodes (q_C = 1.02): the sum peak at the pair (8.96 → 5.05 from
  12° to 40°) against the max dip at the midpoint (4.48 → 1.50).
- [`results/outcomes_vs_delta.png`](results/outcomes_vs_delta.png) — (d)
  P(mean of pair at commitment) and P(arrive at C) vs Δ_close, both reductions,
  one panel per q_C, 95 % Wilson bands.
- [`results/outcomes_extra_vs_delta.png`](results/outcomes_extra_vs_delta.png)
  — the ring shape on tick 2 (one bump at the midpoint / two bumps), P(timeout),
  P(no commitment), P(arrive at A or B).

## The 6.5 signature, item by item

| expected (6.5) | observed |
|---|---|
| `sum`: P(mean of pair) rising sharply as Δ_close drops below ≈ 25° | yes: 0.38 at 30°, 0.82 at 25°, 0.99–1.00 at ≤ 20° |
| `sum`: P(arrive at C) suppressed even at q_C = 1.02 | yes: ≤ 0.02 everywhere except the 20–25° geometry effect below, which is identical at q_C = 1.00 |
| `sum`: timeouts from midpoint heading | no: P(timeout) ≤ 0.085; with the wall out of the way (arena side 2) the agent walks the bisector to the chord, where the bearings have spread and the symmetry breaks |
| `max`: P(mean of pair) near zero down to Δ_close ≈ 20° | yes: 0.065 at 20°, 0.000 at ≥ 25°; the residual at 12–15° (0.985, 0.75) is the ring's node-spacing resolution limit, reported not tuned |
| `max`: P(arrive at C) tracking q_C | **no**: 0.003 / 0.003 / 0.008 pooled over Δ ≥ 30° — C never competes (see headline 3) |

The 20–25° cells arrive at C in ~60 % of trials under **both** maps, at q_C =
1.00 as much as at 1.02, after ~25 ticks: the agent walks the bisector to the
chord (tick ≈ 9–10), where A and B sit at ±60° and C at ≈ −150°, and the circular
mean of the three-bump state then points between A and C. This is a closed-loop
geometry effect of the CoM readout, not a quality effect; it vanishes at ≤ 15°
(the merged pair is reached before the bearings spread) and at ≥ 30° (the pair
separates by tick 7–9 and the agent turns to A or B; arrival then takes 15–16
ticks at 30–40° and 10 ticks at 60°). Among A and B, A (dead ahead at the start)
is favoured at 12° (0.82) and 30–40° (0.61) — identically under both maps.

## 6.6 — the two-target standard condition (`results/sanity_arms.csv`, `results/sanity_paired.csv`)

Δ = 60°, strengths 5.0 vs 4.95 (δ_Q = 1 %), u = 6.0, v = 0.5, the template's
unit-square arena, 200 paired trials per arm.

| variant | σ | reduction | accuracy (arrive at static_0) [95 % CI] | decided | median arrival (ticks) |
|---|---|---|---|---|---|
| campaign (template σ) | 1.5 | sum | 0.550 [0.481, 0.617] | 1.000 | 10.2439 |
| campaign (template σ) | 1.5 | max | 0.550 [0.481, 0.617] | 1.000 | 10.2438 |
| spec (6.2 σ) | 0.1 | sum | 0.690 [0.623, 0.750] | 1.000 | 10.2054 |
| spec (6.2 σ) | 0.1 | max | 0.690 [0.623, 0.750] | 1.000 | 10.2053 |

Paired: Δaccuracy = 0.000 (SE 0.000; the same target in 200 / 200 pairs at both
σ), Δmedian arrival = −0.0001 tick (paired-bootstrap SE 0.0006 at σ = 1.5, 0 at
σ = 0.1). Both shifts are inside the paired-trial standard error, as 6.6 requires.
At 60° the two maps differ by < 7 % of the peak in the tails only (§2.3), and the
ring's response to that is below anything the trajectory resolves.

## What this means for the spec's proposal

- The map-level claim is confirmed: `max` removes the amplitude doubling and keeps
  the pair resolvable at the ring level down to the ring's own resolution.
- The behavioural claim is not: with the unthresholded circular-mean readout, the
  pair's two bumps steer the agent to their mean bearing just as one merged bump
  does, and C loses to the pair on bump count instead of on amplitude. Changing
  the readout (a thresholded or peak readout would resolve two bumps; the
  `use_thresholding: true` path exists in the model) is the obvious next lever
  but is outside this spec (4.2, 8) and was not touched.
- Nothing in the existing 60° results moves (6.6), so `reduction: max` can be
  used in future campaigns without re-interpreting them; the default stays `sum`
  and every existing config is bit-identical (5.1).

## Provenance

- Code: `src/models/mean_field_systems.py` (`reduce_contributions`,
  `sensory_map`, `MeanFieldSystem(sensory_map_reduction=…)`),
  `src/models/movement/mean_field_model.py` (config block, metadata),
  `src/dataHandling.py` (`map_reduction` column). Tests:
  `tests/test_sensory_map_reduction.py` (32), `tests/test_map_reduction_metrics.py` (12).
- Experiment: `scripts/compare_map_reduction.py`, `scripts/map_reduction/`,
  `slurm/map_reduction.sbatch`. Per-cell config hashes are in
  `results/manifest.json`; every trial's exact config was written to its scratch
  directory and is reproducible from the manifest (cell → config, seed_i =
  20260911 + i).
- The per-trial table (`trials.parquet` / `trials.csv`, 8 400 rows) and the raw
  per-task files stay under `results/map_reduction/` (gitignored, ~5 MB).
