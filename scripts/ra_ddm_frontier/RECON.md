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
