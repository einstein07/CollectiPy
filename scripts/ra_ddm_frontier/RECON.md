# RECON — RA slices + envelope vs the DDM frontier (1 %, Δθ = 60°)

Step 1 of `ra-ddm-frontier-slice-envelope-experiment.md`. Every interface both
campaigns depend on, located in the repo, plus **every point where the spec
document (or an implementation choice) departs from the repo or from the
researcher's instructions**. Repo root for all paths: `CollectiPy/`.

---

## The recon slots the spec asked for

### 1. "The previous DDM frontier" = the `campaign/` (ddm-characterisation) main matrix

| | |
|---|---|
| campaign code | [`campaign/`](../../campaign) (`CAMPAIGN_SPEC.md`) |
| the frontier slice | `main/q01_a60_ce*` — δ_Q = 1 %, Δθ = 60°, c_e swept |
| c_e grid | `[0.03, 0.1, 0.3, 1, 3, 8, 20, 50, 125, 300]` (`campaign/factors.py:C_E_GRID`) — **copied verbatim** into `ddm_manifest.csv` |
| n per point | 1000 (`campaign/factors.py:REPS`) |
| archived data | `seoul-data/beta-1/ddm-characterisation/tidy_trials.parquet` (the q01_a60 slice is the §12 regression reference) |
| quality construction | `QUALITY_BETTER = 5.0`, `q1 = q0·(1−dQ) = 4.95` — **confirmed identical** to the spec's static_0 = 5.0 / static_1 = 4.95 |

It is *not* `submit-bellman-ddm-sat-curve-bwunicluster.sh` (that sweep ran at
v = 0.001 m/s with independent seeds and a 5-point cost grid).

### 2. Config template lineage — **each model in ITS OWN established frame (D-01)**

`generate_manifest.py` derives both templates:

- `config/ra_ddm_frontier_ddm_template.json` — from
  [`config/campaign_ddm_base.json`](../../config/campaign_ddm_base.json) with the
  q01_a60 condition overrides `campaign/genconfig.condition_config` applies
  (positions, strengths, `bellman.N_t = 8661`, `T_max = null`,
  `T_max_check_factor = null`), gui and `_comment*` stripped, plus the
  researcher's tick-clock override (D-01 step 2: `ticks_per_second = 1` arena
  + agent, `snapshots_per_second = 1`). `cost_ratio` and `table_cache_dir`
  patched per point / at batch time. **DDM frame:** 1 tick/s,
  `time_limit = 60` s (61 ticks), arena side 2.
- `config/ra_ddm_frontier_ra_template.json` — **the archived
  `ra_ddm_frontier_sweep` environment, reproduced exactly**: the template that
  sweep patched ([`config/mean_field_2_targets_no_viz.json`](../../config/mean_field_2_targets_no_viz.json))
  plus precisely the patches its archived replicate configs carry
  (`linear_velocity 0.05`, `angular_velocity 120`, `sigma_s 0`, the shared
  sensory block, quiet logging). **RA frame:** 1 tick/s, `time_limit = 1000`
  ticks (≡ s), arena radius 1, targets at (0.433, ∓0.25). Only `u` and `v` are
  patched per cell. Verified field-by-field against the researcher-supplied
  archived replicate config (`u_5.199485/v_0.7/replicate_1`): identical except
  per-replicate fields, quiet logging, DataHandling output-form artifacts, and
  `white_rate` (0.07071068 here per the settled **D-09** calibration; the
  cluster copy's 0.1 was the per-target variant) — see D-07.

**Physically shared across both frames** (the quantities the comparison rests
on): `linear_velocity 0.05` m/s, `angular_velocity 120` deg/s, target range
0.5 m at Δθ = 60°, strengths 5.0 / 4.95, termination radius 0.05, start pose
(0,0)→+x, `sensory_stream = {shared, frozen_sd 0, white_rate 0.07071068}` —
a *rate*, so the integrated sensory noise per unit world time is
frame-invariant. The value is the researcher's calibration: evidence-channel
noise c = √2·η = 2 × ΔQ = 0.1 exactly (D-09).

### 3. Timeouts

RA: 1000 ticks ≡ 1000 s at 1 tick/s (the archived sweep's value — kept per the
researcher's instruction, NOT the spec's "copy the DDM timeout"). DDM: 60 s at
1 tick/s (`ticks_limit = 61`; arrival measured ~9–12 s, wide margin). u = 0
measured at ticks 16–129 in the RA frame — far inside the budget (D-05).

### 4. Model parameter field names

| | |
|---|---|
| RA coupling | `agents.movable_0.mean_field_model.u` |
| RA kernel shape | `agents.movable_0.mean_field_model.v` (the factorial's patch key, reused) |
| DDM criterion | `agents.movable_0.embodied_pure_ddm.cost_ratio` |

### 5. Seed-receiving fields, and where the spec's sketch is wrong

Exogenous-noise fields found in the templates (checked against
`src/entityManager.py`, `src/models/percept_stream.py`,
`src/models/egocentric_target_model.py`):

- `environment.sensory_stream.seed` → the shared percept stream. **The only
  truly exogenous seed field.** Receives `env_seed(60, 100, run_id, "sensory")`.
- arena `random_seed` → `EntityManager.initialize` →
  `Entity.set_random_generator` / `set_trial_seed` → **every model-private RNG**
  (the RA's internal σ noise via `MeanFieldSystem.rng`; any DDM-private draw via
  `TargetModel._make_rng`). With `sensory_stream.seed` set explicitly, the arena
  seed feeds *nothing* exogenous: all positions are fixed in the templates.

**Taken:** arena `random_seed = model_seed(model, 60, 100, run_id)`. This
deviates from the spec's sketch in two sanctioned ways: (i) the sketch's
`env_seed(…, "arena")` domain has **no receiving field** — arena randomness is
model-private by construction in this simulator — and (ii) the sketch's
`mf["<MODEL_SEED_KEY>"]` does not exist; the arena seed already *is* the
private-noise channel, delivered through a derivation path the code keeps
separate from the percept stream (`egocentric_target_model.py` lines 120–133).
This mirrors the campaign's own sensory/internal split (`campaign/seeds.py`),
re-keyed to the spec's scheme. (The archived frontier sweep fed ONE seed to
both roles; splitting them is what makes audit (d) provable.)

### 6. §3 rule 3 — exogenous noise IS time-indexed

`SharedPerceptStream` (`src/models/percept_stream.py`): every draw is a pure
function of `(trial_seed, kind, target_id, tick)` via blake2b sub-seeding —
order-independent, consumption-independent, reproducible across processes.
The white noise is a *rate*: per-draw SD = `white_rate/√dt`. No code change
needed. Neither model draws private noise from the sensory stream (audit item
d verifies empirically).

### 7. Log formats

One run archive per replicate: `<base_path>/config_folder_0/run_1.zip` with
`run_1/<agent>_position.csv`, `<agent>_percept.csv` (tick, target, q_hat,
white_rate, dt — the audit's env trace), `<agent>_sensory_noise.csv`,
`<agent>_neural.csv` (RA), `<agent>_ddm.csv` (DDM: x, z, committed,
committed_id, rt, …) and `run_1/events.json` (RA bifurcation events). Arrival
scoring: first tick inside `termination.radius` on the position log, sub-tick
refined by segment–circle intersection (ported from
`scripts/uhat_v_sweep/run_cell.py`).

### 8. Bellman tables

`bellman.N_t = ceil((r0/v)/1e-3) = 8661` at q01_a60. Tables depend only on
(A, c, c_e, geometry) — never on seeds — so the rerun uses the campaign's disk
cache (`src/models/bellman_table_cache.py`, atomic writes). The submission
script's DDM path populates the cache once before the array starts
(replicate 1 of each point, the run-the-model approach of
`campaign/precompute_tables.py`).

---

## Discrepancies and decisions

### D-01 (**researcher's decisions, two iterations**) one 1 s tick clock for both models; full §3 pairing restored

The spec (§2, §4) wanted the RA moved into the DDM frontier's frame
(10 ticks/s, 60 s). The researcher decided otherwise, in two steps:

1. **2026-08-29 — the RA keeps its own frame.** Every prior RA result — the
   factorial, the archived frontier sweep, the 0.765 / 11-tick anchor — lives
   at 1 tick/s; the RA campaign reproduces the archived
   `ra_ddm_frontier_sweep` environment exactly (§2 above).
2. **2026-08-30 — the DDM moves onto the RA's clock.** The DDM template
   overrides the campaign base to `ticks_per_second = 1` (arena AND agent —
   they alias otherwise) and `results.snapshots_per_second = 1`. The
   researcher edited `campaign_ddm_base.json`'s tick fields by hand the same
   way; the template builder sets them regardless, so base and builder agree.
   (That hand edit accidentally deleted the `termination` block — restored,
   since without proximity termination no run ever ends on arrival.)

With both models on one tick clock the shared stream hands them
**bitwise-identical percept realizations** at equal `(env_seed, target,
tick)` — the spec's full §3 pairing, verified as such by audit check (b) —
and per-run_id McNemar (§11) is licensed exactly as the spec intended.

Remaining frame differences, all deliberate or negligible: `time_limit`
(RA 1000 ticks vs DDM 60 s = 61 ticks — DDM arrival measured ~9–12 s, wide
margin), arena container (radius-1 gray vs side-2 white squares — inert:
trajectories stay well inside both and percepts are geometry-independent in
shared mode), and target x at 0.433 vs 0.4330127… (19 nm, far below the
0.05 m termination radius).

Consequence for the DDM's *dynamics* at the coarser tick: see D-11.

### D-02 the seeding scheme replaces the campaign's, by design

`campaign/seeds.py` (blake2b) and the archived sweep's replicate-indexed seeds
are superseded by `seeding.py`'s `frontier-v1` (md5, trial-identity-only) —
both campaigns rerun, so no old seed universe is mixed in. The regression gate
compares *distributions* against the archived frontier, which is correct
across seed universes.

### D-03 the spec's inline batch heredoc is replaced by `run_batch.py`

Config generation *and execution* live in one Python process per task instead
of a heredoc + bash loop over `main.py -c` (interpreter + numba JIT paid once
per task, not per replicate). `--subprocess` keeps the
one-`main.py`-per-replicate path; the two were verified **bit-identical** on
the same replicate. §7 holds verbatim: every replicate directory receives its
exact `config.json` + `run_meta.json` before the run, `.done` on success,
per-task failure files; any replicate re-executes with
`python src/main.py -c <dir>/config.json`.

### D-04 commitment-time supplement: the detector condition fails

§11 allows a commitment-plane supplement only if the identical bifurcation
detector ran on the DDM trajectories. It did not: `BifurcationDetector` is
RA-only (and its `behavioral` mode tracks the agent heading — see the
factorial's RECON D-04); the DDM's commitment is its own `rt`/`committed_id`.
The supplement is **not produced**; both commitment records are carried in the
trial tables, clearly labeled.

### D-05 u = 0 within the RA budget — **measured**

5 replicates of `U_v0.2_u0` in the RA frame: arrival at ticks 19–88 against
the 1000-tick budget (consistent with the spec's "~30+ ticks in earlier
sweeps"). Not censored.

### D-06 `CLAUDE.md` GSD workflow

As with the factorial (its RECON D-13): implemented directly from the spec
document at the researcher's explicit request. Flagged rather than assumed.

### D-07 RA template vs the researcher-supplied archived config — field-by-field

Diffed against the supplied `u_5.199485/v_0.7/replicate_1` config. Identical
except:

| field | archived | this campaign | why |
|---|---|---|---|
| `sensory_stream.white_rate` | 0.1 (env block; the stale metadata says 0.035) | **0.07071068** | the researcher's calibration, settled 2026-08-29: noise = 2 × ΔQ on the *evidence channel* (c = 0.1), not per target — see D-09 |
| `sensory_stream.seed` | null (derived from arena seed) | explicit `env_seed(…)` | frontier-v1; same mechanism, auditable |
| arena `random_seed` | one seed for both roles | `model_seed(model, …)` | stream separation (§3 rule 2); the sensory stream no longer reads it once `sensory_stream.seed` is explicit |
| `logging` | console INFO | console off, ERROR | operational only |
| `results.base_path` / metadata | sweep_metadata block | `run_meta.json` per replicate | §7 layout |
| `number: 1`, `arena` (singular), `post_bifurcation_swap: null`, `gui: {}` | present | input-schema form (`[1]`, `arenas.arena_0`, absent) | DataHandling's *output* form of the same input |

Everything else — tick rate 1, time_limit 1000, arena radius 1 gray square,
positions (0.433, ∓0.25), colors, strengths, plant, detection, the entire
`mean_field_model` block including σ = 1.5, `use_thresholding false`,
`g_adapt 0`, the bifurcation detector — **inherited unchanged**; only `u`, `v`
patched per cell.

### D-08 provenance

Templates and manifests carry the git SHA at generation time; every
`run_meta.json` records `seed_scheme = "frontier-v1"` and the SHA. Commit
before submitting, or the SHA carries `-dirty`.

### D-09 (**researcher's calibration**) noise = 2 × ΔQ on the EVIDENCE CHANNEL: `white_rate = √2·ΔQ ≈ 0.07071068`

The researcher's calibration (settled 2026-08-29, second iteration): "noise =
twice the signal" means the **evidence channel exactly** — the difference
percept q̂₀ − q̂₁, whose noise scale is the DDM's c = √2·η — so

    c = 2ΔQ = 0.1   ⇒   η = √2·ΔQ = 0.07071068…, with ΔQ = 5.0 − 4.95 = 0.05.

The two candidate definitions, for the record (the stream's per-draw SD is
η/√dt — a rate, frame-invariant per unit world time):

| definition of "noise = 2× signal" | exact η | status |
|---|---|---|
| **evidence-channel scale c = √2·η = 2ΔQ** | **η = √2·ΔQ ≈ 0.070711** | **taken** |
| per-target percept SD per √s = 2ΔQ | η = 2ΔQ = 0.1 | rejected (this is what the cluster copy of the archived RA sweep carried, with stale metadata; the beta-1 snapshot predates it at 0.035) |

Under the taken value the per-target percept SD is √2·ΔQ ≈ 1.41 ΔQ per √s.
`frontier.WHITE_RATE` is *derived* (`√2 × QUALITY_DELTA`, rounded to 8
decimals), not typed, and `frontier.NOISE_SCALE_C = 0.1` records the
calibrated quantity itself. Stamped into both templates — this
**deliberately overrides** `campaign_ddm_base.json`'s locked 0.035; the
Bellman tables are re-solved at the new c (different cache keys, handled
automatically).

### D-10 gates voided by the calibration change (consequence of D-09)

Both external reference points were measured at η = 0.035 and are therefore
**informational, not CI-gates**, under η = 0.1:

- the archived DDM frontier (`ddm-characterisation`, q01_a60): `aggregate.py
  --previous` still prints the side-by-side comparison but never fails on it —
  lower accuracy per c_e and shifted times are *expected* physics
  (SNR per unit time drops ~2.9×);
- the factorial anchor (0.765 / 11 ticks at v = 0.5, û = 1.00): reported with
  the calibration caveat; commit-time agreement (~11 ticks, travel-dominated)
  may persist, accuracy will sit lower.

The gates that remain sharp and blocking: the §3 audit battery, manifest
completeness, and the four u = 0 pure-replicate cells (v inert at u = 0 —
mutual CI overlap at any η). Environment/template drift on the DDM side is
covered by the template being derived from `campaign_ddm_base.json` with the
single documented override, plus determinism check (a).

### D-11 the DDM at a 1 s tick: coarser evidence steps, same flag set

At `ticks_per_second = 1` the DDM's per-substep evidence step grows to
c·√(dt/n_sub) = 0.1·√(1/16) = **0.025** (was 0.0079 at 10 ticks/s), and rt is
resolved at dt/n_sub = 62.5 ms. Quasi-static boundaries at the D-09
calibration (A = 0.05, c = 0.1, 60°): z\*(0.03) = 0.0028 and z\*(0.1) = 0.0093
sit below the step — the `DISCRETISATION_LIMITED_CE = {0.03, 0.1}` convention
carries over unchanged — while z\*(0.3) = 0.0278 clears it **marginally**
(1.1×): read the ce = 0.3 point with that in mind. Commitment *ticks* are also
10× coarser; arrival is still sub-tick-refined from the trajectory as before.

### D-12 wave 2 (spec v2, 2026-08-30): Set U-v2 top-up + DDM ceiling points

**Set U-v2** (§2): generated from wave-1 `cells.csv` by
`generate_manifest.py --topup-from`. Measured cliff windows (rule: û_hi = last
Set-R level with acc_all > 0.95 — 0.65 at every v; û_lo = first < 0.85 —
0.70 at v = 0.2, 0.75 elsewhere), sampled at Δû = 0.025, plus 0.3·u\* / 0.5·u\*
anchors, a 1.2·u\* committed point, and the û ∈ {1.75, 2.0, 2.4} tail:

| v | new u values |
|---|---|
| 0.2 | 4.0, 6.5, 8.5, 8.75, 9.25, 23.0, 26.25, 31.5 |
| 0.3 | 2.75, 4.5, 6.25, 6.5, 6.75, 16.0, 18.25, 21.75 |
| 0.4 | 2.25, 3.5, 4.75, 5.25, 5.5, 8.5, 17.25 |
| 0.5 | 1.75, 3.0, 4.0, 4.25, 4.5, 7.5 (tail all within 5 % of the existing 10–15 levels, skipped as the spec predicted) |

**29 cells × 1000 = 29 000 replicates.** Interpretation of the 5 % skip rule:
it applies to the anchors / committed / tail targets (the spec states it in
the tail paragraph); cliff-window samples deliberately interleave the integer
wave-1 grid and are dropped only on exact collision — the spec's own
indicative lists (8.25, 8.75, 9.25 at v = 0.2) confirm this reading. Cell ids
are `U2_…`, `sweep = absolute`, same directory layout — analysis pools waves
automatically; `manifest_full.csv` is the §9 completeness reference.

**DDM ceiling points** (§11): c_e ∈ {3000, 30000} (~10× and ~100× the previous
maximum), in `ddm_manifest_topup.csv`. The analytic fixed-bound
infinite-patience asymptote is also computed: Φ((A/c)·√(r₀/v)) =
Φ(0.5·√8.6603) = **0.9294** — no boundary policy can beat the full-horizon
ideal observer. `analyze_overlay.py` now emits `ceiling_check.json`: the
RA-beats-the-DDM-family claim is granted only if the cross-validated
envelope's peak (eval-half trials, so no winner's curse) clears both the
asymptote and the best measured DDM point with CI separation.

**Step-halving** (§2): `dt_check.py`, 200 trials/arm at dt 0.1 vs 0.05 on
identical frontier-v1 seeds, at the three stiffest new cells (u = 23, 26.25,
31.5 at v = 0.2) — result recorded below; blocking for the wave-2 submission.

### D-13 (spec v3, 2026-08-30): the DDM halt campaign — halt-at-midpoint, both families, NEW tree

§2b: both DDM families rerun under the **halt-at-midpoint** motion policy —
`bellman.terminal = "halt_sprt"` (wired in `embodied_pure_ddm_model.py` /
`bellman_boundary.py`: at arrival the undecided agent parks at the midpoint,
v = 0 verified exactly in smoke, and keeps integrating against the flat
plateau `z_halt`; runaway guard at `T_max + 10·D(z_halt)`, expected zero
hits), `halt_cost_rate = 1.0` (the physical value). Decisions and mechanics:

- **Exact parameterization** (the spec's recon slot): halt trigger =
  `past_horizon` of the Bellman table (t_evidence ≥ T_max = r0/v); the
  collapsing bound floors at b_min = z_halt = solve of `sinh(a)+a =
  c_e·k·A/2`, a = k·z_halt (k = 2A/c² = 10); collapse shape = the Bellman
  solve itself with the terminal slice V_T of the halted problem.
- **New results tree** `ra_ddm_frontier_ddm_halt/` with manifest
  `ddm_manifest_halt.csv` (schema `point_id, variant, bound, diff, n_runs,
  seed_scheme`, 26 points) and layout `points/<variant>/{ce_|b_}<bound>/`.
  The forced-choice wave-1/2 tree `ra_ddm_frontier_ddm/` is **frozen** and
  becomes the §9 regression reference (same calibration, same env seeds).
- **Model seeds re-keyed** per §3/§6: `model_seed("ddm-bellman", …)` /
  `model_seed("ddm-static", …)` (was plain `"ddm"` in waves 1/2). Audit (d)
  shows the DDM consumes **no private noise** in this configuration, so the
  re-key changes nothing physical — which is exactly what lets the zero-halt
  regression gate demand reproduction within CIs.
- **Regression gate re-scoped** (§9): `aggregate.py --previous-rerun
  <frozen ddm_trials.parquet>` — BLOCKING CI-overlap at every Bellman c_e
  with halt_frac ≤ 0.005; where the halt triggers, the expected signature is
  slower + more accurate (reported, not failed). The old `--previous`
  (archived 0.035 frontier) comparison stays informational (D-10).
- **time_limit stays 60 s** (§13 forbids timeout changes): `ddm_halt_budget()`
  (printed at manifest generation) flags censor risk where mean total +
  3·D(z_halt) > 60 s — the two ceiling points c_e ∈ {3000, 30000} (D ≈ 15/19 s).
  Their censored fraction is visible in `1 − decided_frac` and `halt_frac`;
  captions must carry it. Trials still halted at 60 s never arrive.
- Per-trial logging (§2b): `halted` (halt_event), `halt_duration`, `z_halt`,
  `halt_guard_hits` — new columns in `_ddm.csv` (dataHandling) and in
  `ddm_trials.parquet`; `ddm_points.csv` gains `halt_frac`,
  `median_halt_duration_s`, `halt_guard_hits`.
- The researcher's hand edit adding `terminal`/`halt_cost_rate` to
  `campaign_ddm_base.json` had (again) dropped the `termination` block —
  restored, same accident as in D-01 step 2; `build_ddm_template()` asserts
  the halt fields rather than setting them, so a hand-revert is caught.

### D-14 (spec v3): the static-bound family — one code path, two parameterizations

§2b Family 2, wired as the degenerate case as the spec demands:
`bellman.static_bound = b` short-circuits the PDE solve and installs the flat
table z(t) = b on [0, T_max] through `set_bellman_table` — the SAME
hold-past-the-horizon rule and halt machinery then apply with z_halt = b.
No parallel implementation; `cost_ratio` is pinned to 0.0 on static points
(diagnostic only — the boundary never reads it). Validation refuses
`static_bound` under any `threshold_policy` other than `bellman`.

- **b grid derived, not typed** (`frontier.static_b_grid()`): 14 log-spaced
  levels over [z*_quasistatic(c_e = 0.03) = 0.0028 — the fastest Bellman
  point's own boundary, same speed by construction —, b(acc = 0.995) =
  ln(199)/k = 0.5293]. Six levels sit below the 1 s-tick evidence substep
  0.025 (RECON D-11) and are marked discretisation-limited, the Bellman
  convention carried over.
- **b\* derived from the swept data** (`analyze_overlay.static_bstar` →
  `static_bstar.json`): b*_cost = argmin[P(err) + c·E[T_decision]], c = 2ΔQ
  = 0.1 (the §2b cost functional; censored trials charged the full 60 s);
  b*_RR = argmax[P(correct)/E[T_arrival]]. Bootstrap CIs over trials; Wald
  constant-drift analytic (ER = 1/(1+e^{kb}), DT = (b/A)tanh(kb/2)) as the
  sanity anchor with the discrepancy reported. Smoke (2 static points,
  n = 20): b*_cost = 0.1055 vs Wald 0.1124 — coherent.
- Smoke verification: 12/20 trials at b = 0.5293 halted; halt durations
  4.8–28.7 s around the predicted D(b) = 10.5 s; **zero movement** during
  every halted tick (the motor hold is exact).

### D-15 (spec v3): Set U-v3 — output-plane arc-length placement + kernel extension

`generate_manifest.py --wave3-from <cells.csv> --factorial <factorial
cells.csv>` (rule in `frontier.build_wave3_rows`): per v, monotone PCHIP
interpolants of (median arrival, acc_all) vs u, branches split at the
measured accuracy peak, ~9 levels/branch at equal arc-length increments in
per-v-normalized plane coordinates, rounded to 0.25, 5 % skip; plus a
gap-fill pass — realized chords between adjacent measured points > 2× the
arc-length budget get up to 3 equally spaced u midpoints. Generated
2026-08-30 from the pooled waves-1+2 `cells.csv`:

| v | map | new u |
|---|---|---|
| 0.2 | frontier | 1.25, 2.75, 4.25 |
| 0.3 | frontier | 1.25, 2.25, 3.25 |
| 0.4 | frontier | 0.75, 1.5, 1.75, 2.5 |
| 0.5 | frontier | 0.75, 1.5, 2.25 |
| 0.6 | factorial | 0, 1.75, 2.75–5.75 (cliff), 8.25, 9.75, 11.0, 13.25 |
| 0.8 | factorial | 0, 1.5, 2.5–4.25 (cliff), 5.25, 6.0, 7.5, 8.75, 10.0, 12.0 |

**45 cells × 1000 = 45 000 replicates** (`manifest_wave3.csv`;
`manifest_full.csv` now 174 rows). Decisions:

- Extension maps come from the factorial's n = 400 sweep (t there = commit
  ticks ≡ s at 1 tick/s — placement only, so the different η = 0.035
  calibration and t definition are acceptable); u*(v) is recomputed from the
  live kernel builder and gated against the factorial's recorded value
  (≤ 1 %; measured ≈ 2e-5). The factorial never sampled below û = 0.5, so
  extension v gets the U-v2-style {0.3, 0.5}·u* anchors, the û ∈
  {1.75, 2.0, 2.4} tails (13.25 / 12.0 ≈ 2.4·u*) and a u = 0 control —
  the §12 u = 0 gate becomes six-way (aggregate.py now pools ALL
  absolute-sweep u = 0 cells).
- Re-running the generator after wave-3 data lands switches v ∈ {0.6, 0.8}
  to their own frontier maps automatically (≥ 6 measured cells) and performs
  the spec's gap-fill pass; already-manifested cells are excluded by id, so
  it is a pure top-up in the same seed universe.
- Step-halving (blocking): the three highest new-u cells at their v —
  U3_v0.6_u11, U3_v0.8_u12, U3_v0.6_u13.25 (all far below the wave-2-checked
  u = 31.5, but new per-v maxima) — result in the pre-flight table.

---

## Pre-flight results (this workstation, 2026-08-30 — final configuration: η = 0.07071068 (c = 0.1), both models at 1 tick/s)

| gate | result |
|---|---|
| Anchor, u\*(0.5) = 6.157 | **PASS** — 6.156868, rel. err 2.14e-05; u\*(v) grid identical to the factorial's |
| §3 audit battery (a)–(d) | **ALL PASS** — see `AUDIT.md`; (b) is bitwise-identical cross-model percept realizations (full §3 pairing, D-01) |
| RA template vs the researcher-supplied archived config | field-by-field identical up to the D-07 table |
| Smoke §8 (2 RA cells + 3 DDM points × 20, real script path) | **PASS**, zero failures; aggregate → overlay end-to-end |
| u = 0 within RA budget | **PASS** — arrival at ticks 16–129 of 1000 |
| Factorial-anchor cell, smoke n = 20 (informational, D-10) | acc_all 0.700 [0.48, 0.86] (0.765 at the old η = 0.035 calibration), median commit **11 ticks** exactly |
| Archived-frontier comparison (informational, D-10/D-11) | ce = 8: 0.850 vs 0.982; ce = 300: 0.850 vs 0.998 at arrival 11.5 vs 9.7 s (higher noise: lower ceiling, longer deliberation); ce = 0.03: 0.700 vs 0.629 — at the 1 s tick the first substep (0.025) dwarfs z\* = 0.0028, so the "fast-guess" floor is set by single-substep SNR, not the boundary (D-11) |
| In-process vs `main.py -c` | bit-identical (position/percept/neural logs) |
| Measured wall time | RA 0.50–0.58 s/run, DDM 0.55–0.57 s/run (cache warm) → at `RUNS_PER_TASK=100`: ≲ 2 min/task; whole pair of campaigns ≈ 15–20 core-hours |
| SLURM dry run | RA 1000 tasks, DDM 100 + precompute; `sbatch --test-only` still needed on the login node |
| **Wave 2** step-halving (D-12; 200 trials/arm, dt 0.1 vs 0.05, identical seeds) | **PASS** at all three stiffest cells — u = 23: Δ = +0.010 [−0.035, +0.050]; u = 26.25: +0.010 [−0.030, +0.050]; u = 31.5: +0.015 [−0.025, +0.060]; zero failures, max\|z\| ≈ 1.85. `dt_check_report.json` |
| **Wave 2** ceiling point c_e = 30000, 3 local replicates | solver clean; rt 5.5–6.8 s of the 8.66 s horizon, all committed, arrival ~11 s |
| **Wave 2** TOPUP dry runs | RA 29 tasks, DDM 2 + precompute |
| Ceiling verdict on wave-1 data | claim **STANDS** vs the analytic asymptote (RA envelope eval-half peaks 0.984 / 0.994 vs 0.9294) and vs the measured DDM best (hi 0.9505) — empirical extreme-c_e confirmation pending the wave-2 DDM runs |
| **Spec v3** §3 audit rerun (halt templates, `ddm-bellman`/`ddm-static` seeds, static arm added to (c)) | **ALL PASS** — env traces identical across models AND variants; DDM consumes no private noise (audit d), so the D-13 seed re-key is physically inert |
| **Spec v3** static-bound smoke (b = 0.5293, n = 20; b = 0.1055, ce = 3, n = 3) | zero failures; 12/20 halted at b = 0.5293, halt durations 4.8–28.7 s around the predicted D = 10.5 s; **v = 0 exact** during every halted tick; flat table z ≡ b confirmed in `_ddm.csv`; RA 0.44–0.70 s/run |
| **Spec v3** halt budget (`ddm_halt_budget()`) | censor risk flagged ONLY at c_e ∈ {3000, 30000} (D(z_halt) = 14.6 / 19.2 s vs the 60 s budget) — expect a visible censored tail there, carried in `halt_frac` / `decided_frac` |
| **Spec v3** halt regression gate, smoke (ce = 3, n = 3) | PASS at halt_frac = 0 (machinery verified; the real gate runs on the full campaign) |
| **Spec v3** overlay pipeline end-to-end (real RA waves 1+2 × smoke halt DDM) | envelope + regret + McNemar + `static_bstar.json` (b*_cost 0.1055, Wald 0.1124) + `ceiling_check.json` + all three figures render, both families + b\* markers |
| **Wave 3** step-halving (200 trials/arm, dt 0.1 vs 0.05, identical seeds) | **PASS** at all three new per-v maxima — U3_v0.6_u11: Δ = +0.065 [−0.010, +0.140]; U3_v0.6_u13.25: +0.065 [−0.025, +0.150]; U3_v0.8_u12: +0.065 [−0.020, +0.145]; zero failures, max\|z\| ≈ 1.85. `dt_check_wave3_report.json` |
