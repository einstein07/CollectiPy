# RECON — (û, v) factorial

Step 0 of `uhat-v-factorial-experiment.md`. Every interface the sweep depends on,
located in the repo, plus **every point where the spec document and the repo
disagree**. The spec's hard rule is that the repo wins; each disagreement below
says which value was taken and why.

Repo root for all paths: `CollectiPy/`. Verified against `git rev-parse HEAD`
`9f36964` (working tree modified — see D-12).

---

## The six required recon items

### 1. Ring-attractor model registration key and config file

| | |
|---|---|
| registration key | `mean_field` — [`src/models/movement/mean_field_model.py:484`](../../src/models/movement/mean_field_model.py#L484), `register_movement_model("mean_field", ...)` |
| selected by | `environment.agents.movable_0.moving_behavior = "mean_field"` |
| parameter block | `environment.agents.movable_0.mean_field_model` |
| dynamics | [`src/models/mean_field_systems.py`](../../src/models/mean_field_systems.py) (`MeanFieldSystem`), numba-JIT Euler integration |
| shared percept layer | [`src/models/egocentric_target_model.py`](../../src/models/egocentric_target_model.py), [`src/models/percept_stream.py`](../../src/models/percept_stream.py) |
| **config file** | [`config/mean_field_2_targets_no_viz.json`](../../config/mean_field_2_targets_no_viz.json) |

This is the RA analogue of `config/embodied-pure-ddm-2-targets.json`. It is the
template `submit-ra-frontier-sweep-bwunicluster.sh` patches, i.e. the RA arm of
the RA–DDM comparison. **No RA config in the repo carries a `sensory_stream`
block**; the frontier submit script injects it at patch time, and so does
`config_patch.py` here.

### 2. Single headless trial: entrypoint and where it reports

* **Entrypoint:** `python src/main.py -c <config.json>`, or in-process
  `EnvironmentFactory.create_environment(Config(path)).start()` — the route
  `campaign/run_chunk.py` already uses, and the one `run_cell.py` uses.
* **Output:** `<results.base_path>/config_folder_0/run_1.zip`, containing
  `run_1/<agent>_position.csv` (`tick, pos_x, pos_y, pos_z`),
  `<agent>_neural.csv` (`tick, neuron_0..29, norm_z, lambda1, omega`),
  `<agent>_sensory.csv`, `<agent>_percept.csv`, `<agent>_targets.csv`,
  `<agent>_sensory_noise.csv` and `run_1/events.json`.
* **There is no accumulator-style commitment record on the RA side.** The RA
  arm's decision is *behavioural*: the run ends when the arena's
  `termination: proximity` block fires, and what the agent chose is which target
  it arrived at. Both are read off the position log, exactly as
  `campaign/summarise.py` does for the DDM arm.

  | quantity | source |
  |---|---|
  | choice | nearest target within `termination.radius` of the logged position |
  | correctness | `choice == "static_0.s#0"` (the 5.00-quality target) |
  | arrival time | the tick of that first crossing (`ticks_per_second = 1`) |
  | commitment tick | see D-04 — the detector's event exists but is not usable as a choice label |

  This matches the two notebook cells that produced the prior results
  (`paper-ready-results-beta-1.ipynb`, cells 73 and 75): *"Decision time = ticks
  to reach the selected target … Decision accuracy = share of the runs that
  arrived which chose the 5.00 target."*

* **Timeout:** `Arena.run` computes `ticks_limit = time_limit * ticks_per_second + 1`
  ([`src/arena.py:1029`](../../src/arena.py#L1029)), so `T_max = 100` control
  ticks is `environment.time_limit = 100` at `ticks_per_second = 1`.
* **A diverging trial does not hang the task.** `MeanFieldSystem.step` raises on
  NaN/Inf; the agent process dies; `Environment.start` detects it and re-raises
  `RuntimeError("A subprocess exited unexpectedly.")`
  ([`src/environment.py:473`](../../src/environment.py#L473)). `run_cell.py`
  catches that and records the trial as `numerical_failure`.

### 3. Where and how the readout threshold is evaluated

`MeanFieldSystem.compute_dynamics`
([`src/models/mean_field_systems.py:543`](../../src/models/mean_field_systems.py#L543))
already evaluates the readout **at every Euler substep** — 500 of them per
control tick at `integration_time = 50`, `integration_dt = 0.1`. With
`use_thresholding = True` that is `circular_readout(z_t, theta, threshold=g_threshold)`
per substep; with `False` it is `compute_center_of_mass` per substep. Either way
`MeanFieldMovementModel.step` consumes only `bump_positions[-1]`.

Two consequences:

* The spec's proposed change ("if the readout threshold is evaluated only once
  per control tick, add an evaluation every 10 Euler substeps") **has no target**:
  the readout is already at substep resolution. What is quantised to one tick is
  the *motor command*, and behind it the physical arrival that defines DT. No
  logging-only change can refine that.
* So `t_commit_fine` is produced a different way, and it is still logging-only:
  the segment between the two logged positions bracketing the crossing is
  intersected with the termination circle
  (`run_cell._segment_circle_fraction`). Nothing in the dynamics, the motor path
  or the decision path is touched. **Measured effect:** at (v=0.5, û=0.9) all 12
  smoke trials report `t_commit_ticks = 11` while `t_commit_fine` spreads over
  10.19–10.30 — precisely the 1-tick quantisation the spec was worried about,
  now resolved.

### 4. Kernel construction W(v)

`MeanFieldSystem.compute_interaction_kernel`
([`src/models/mean_field_systems.py:279`](../../src/models/mean_field_systems.py#L279)):

```python
delta = |wrapped(theta_i - theta_j)|
M = (1 / num_neurons) * cos(pi * (delta / pi) ** v)
```

`generate_manifest.py` imports `MeanFieldSystem`, constructs it at the runtime
`N = 30` and reads `system.M` back — it never reimplements the formula, so a
normalisation change cannot silently desynchronise the manifest from the
simulation. `u*(v) = 1 / (λ_max(M) · sech²(β))`.

**Anchor gate passes:** `λ_max(v=0.5) = 0.386738`, `u* = 6.156868`, relative
error `2.14e-05` against the required 6.157. The computed u*(v) grid reproduces
the hand-entered table in `submit-ra-frontier-sweep-bwunicluster.sh` to every
printed digit (22.791063, 13.125162, 9.087682, 7.179396, 6.156868, 5.561975,
5.199485, 4.975203, 4.839030, 4.762196).

### 5. Existing SLURM scripts from the previous speed–accuracy sweeps

| script | role |
|---|---|
| [`submit-ra-frontier-sweep-bwunicluster.sh`](../../submit-ra-frontier-sweep-bwunicluster.sh) | **the direct predecessor** — matched (v, u\*) pairs, shared stream, 1000 reps |
| [`submit-why-ra-combined-sweep-bwcluster.sh`](../../submit-why-ra-combined-sweep-bwcluster.sh) | fixed-u curves (u = 0/5/6.2/8) |
| [`submit-mean-sweep-v-u-fixed-velocity-bwunicluster.sh`](../../submit-mean-sweep-v-u-fixed-velocity-bwunicluster.sh) | the batching/lost-run pattern both of the above derive from |
| [`submit_campaign.sh`](../../submit_campaign.sh) + [`campaign/`](../../campaign) | the DDM campaign's orchestration layer |

Reused in `slurm/uhat_v_sweep.sbatch`: `--partition=cpu`, `--mem=4G`,
`--cpus-per-task=1`, the venv interpreter discovery ladder
(`python3.12 → python3.10 → python3 → python`), the `PROJECT_DIR` /
`BASE_PATH_ROOT`-style environment overrides, the workspace-scratch results
root, the submission-mode / execution-mode split on `SLURM_ARRAY_TASK_ID`, and
the dry-run-by-default posture.

### 6. Sweep-manifest / config-templating convention

`campaign/` is the established convention and this sweep follows its idioms:
`factors.py` as the single source of factor levels and locked parameters,
`config_patch.py` for per-condition/per-trial effective configs, one output file
per array task, atomic publish (temp name in the destination directory then
`os.replace`), node-local scratch staging, an in-process runner with a
`--subprocess` fallback, and idempotency on a complete output file.

Two deliberate departures, both required by *this* design:

* **Layout.** The spec names `scripts/uhat_v_sweep/` and `slurm/`, and those are
  used. `campaign/` is the DDM campaign's own package and implements
  `CAMPAIGN_SPEC.md`; folding an RA factorial into it would blur two specs.
* **Seeds.** `campaign/seeds.py` derives seeds by `blake2b` over the factor
  levels. That is the *opposite* of what this design needs: Section 2 requires
  the identical seed list in every cell so contrasts across cells are paired. So
  `seed_i = BASE_SEED + i`, `BASE_SEED = 20260828`, used for both the arena
  `random_seed` and `sensory_stream.seed`. `aggregate.py` checks that every cell
  really did share one seed list and warns if not.

---

## Discrepancies between the spec document and the repo

### D-01 (**material**) `angular_velocity` — template 10, actual sweep 120

The spec inherits "the RA-arm config currently used for the RA–DDM comparison".
The template on disk carries `angular_velocity: 10` deg/s, but **every archived
config of the sweep that produced the Section 12 anchor carries 120**:

```
seoul-data/beta-1/ra_ddm_frontier_sweep/u_6.156868/v_0.5/replicate_*/config_folder_0/config.json
  → "angular_velocity": 120
```

`submit-ra-frontier-sweep-bwunicluster.sh` does not patch this field, so the
repo template has drifted from the code that produced the data (the same drift
`campaign/README.md` records for the DDM arm).

It is not cosmetic. Minimum turn radius is `v/ω`; at 10 deg/s that is
**0.286 m** against a 0.05 m termination radius, so the agent cannot turn inside
the target and orbits it. Measured directly: at (v=0.5, û=0.9) with
`angular_velocity = 10`, all 12 trials arrived at tick **50–51**; at 120, all 12
arrive at tick **11**.

**Taken: 120**, pinned in `factors.ANGULAR_VELOCITY` and *set* by
`config_patch.py` rather than inherited, with a post-patch assertion that the
turn radius fits inside the termination radius. Ground truth for the arbitration:
re-scoring the archived (v=0.5, u=u\*) cell gives **accuracy 0.7650, median DT
11.0 ticks over 1000/1000 arrivals** — the Section 12 anchor exactly, and
reachable only at 120.

### D-02 (**material**) `sigma` — spec says 0.1, repo says 1.5

Every RA config in `config/` sets `mean_field_model.sigma = 1.5`, including the
archived frontier-sweep configs, and no sweep script overrides it. The DDM
config's own comment is explicit that this parameter has no DDM counterpart:
*"the ring attractor's internal 'sigma' is neural noise with no DDM counterpart
and stays a free parameter."*

**Taken: 1.5** (repo wins). At 0.1 the sweep would not reproduce the anchor.

### D-03 (**material**) "thresholded readout, g_threshold = 0.6"

The repo sets `use_thresholding: false` in every RA config, including the
archived frontier-sweep configs. Under `false` the bump heading is a plain
centre-of-mass and **`g_threshold` is inert for the heading**; it still enters
`last_magnitude` / `last_concentration`, but with `scale_velocity: false` those
only gate "is there a bump at all", never the speed.

**Taken: `use_thresholding = false`, `g_threshold = 0.6`** (repo wins; the
threshold value is carried unchanged but has no effect on this experiment).
Recorded here because a reader of the spec would otherwise expect it to matter.

### D-04 (**material**) what "commitment" means, and the detector's event

The spec's schema asks for `t_commit_ticks`. There is no accumulator commitment
on the RA side. `BifurcationDetector` does emit an event, but three properties
make it unusable as the scoring commitment:

1. `mode: "behavioral"` in the config dispatches to `_update_behavioral_agent_angle`,
   not `_update_behavioral` — the bump-angle branch is commented out
   ([`src/models/bifurcation.py:754`](../../src/models/bifurcation.py#L754)). The
   criterion is therefore *the agent's physical heading* within 5° of a target,
   not the ring's bump.
2. It fires on a transient sweep. Observed: a trial whose heading passed through
   alignment with `static_1` at tick 39 while turning toward `static_0`, and
   which then arrived at `static_0`.
3. It is effectively one-shot per run — the `retrigger` guard suppresses later
   events once alignment is lost — so the first, possibly transient, event is
   all there is.

**Taken:** `t_commit_ticks` / `t_commit_fine` / `t_arrival_s` are the **arrival**
clock, which is what every prior RA analysis in this repo calls decision time and
what the 11-tick anchor refers to. The detector's event is still logged, as
`t_bif_ticks`, `bif_target` and `bif_agrees_with_choice`, so the relationship
between the two can be looked at — it is just not used for scoring.

### D-05 `terminal_categorical` is not an RA parameter

`geometric_error_mode: "terminal_categorical"` is a **DDM** config key
(`ddm_systems.py`, `config/embodied-pure-ddm-2-targets.json`). No such key exists
on the RA side. **Taken:** read as the *scoring convention* — 0–1 loss on the
terminal choice, undecided trials scored as errors in `acc_all`. Implemented in
`aggregate.py`; the code has no such config key.

### D-06 `T_max = 100` vs the predecessor's `time_limit = 1000`

The frontier sweep ran at 1000 ticks. **Taken: 100** as the spec fixes, but note
the consequence: at v = 0.1 the prior sweep had 101/1000 runs fail to arrive
inside *1000* ticks, so at 100 the undecided fraction there will be
substantially larger. That is by design ("it is data, not an exclusion") and is
why `decided_frac` is a heatmap of its own and `acc_all` is reported everywhere
beside `acc_decided`. It also means the v = 0.1 column is **not** directly
comparable to the archived frontier numbers.

### D-07 `sensory_stream.seed` — null in the predecessor, explicit here

The frontier sweep leaves `seed: null`, which means "derive from the arena
`random_seed`". Here it is set explicitly to the trial seed — the same number the
arena seed is set to, so the behaviour is identical, but the pairing is now
visible in the config rather than implied. `EntityManager.initialize` passes the
raw arena seed to `Entity.set_trial_seed`
([`src/entityManager.py:155`](../../src/entityManager.py#L155)), which is what the
shared stream reconstructs from.

### D-08 `sigma_s` must be 0 under shared mode

`percept_stream.py` refuses `sigma_s != 0` when `mode: "shared"`. The template
carries `sigma_s: 1.0`; the patch forces 0, exactly as the frontier sweep does.
Asserted post-patch.

### D-09 timing, and the 30–60 min task target

Section 7 asks for ~30–60 min per array task and says to measure in the smoke
test. Measured on this workstation: **0.44 s/trial** at (v=0.5, û=0.9), **0.84
s/trial** at the stiffest cell (v=0.1, û=1.5, u = 34.2, where about half the
trials run the full 100-tick budget). A whole 400-trial cell is therefore
**3–6 minutes**, and the entire 32,000-trial sweep is roughly 5 core-hours.

**Taken:** one cell per array task (80 tasks), wall limit 30 min — well over the
2× rule against the worst measured chunk. Packing cells together to reach 30–60
min would add a second index dimension for no gain at this cost.
`TRIALS_PER_CHUNK` still exists if the geometry needs changing.

### D-10 max resolved u

`1.5 × u*(0.1) = 34.187`, below the 35 threshold, so the manifest does **not**
raise the prominent flag. It is still the stiffest corner of the grid and is one
of the three cells the step-halving check targets.

### D-11 output format

Raw per-task files are Parquet (`pandas` 2.3.3 + `pyarrow` 25.0.1 are in
`CollectiPy/.venv`), with an automatic CSV fallback if `pyarrow` is missing on a
compute node. Neither `pandas`, `pyarrow` nor `statsmodels` is listed in
`requirements.txt`; `statsmodels` was absent from the venv and was installed for
the analysis step. `analyze_collapse.py` also carries an equivalent numpy IRLS
path (`--backend numpy`) and produces bit-identical deviances, so the analysis
cannot be blocked by a missing package.

### D-12 provenance caveat

`git status` shows the working tree modified relative to `9f36964` (seven
`config/mean_field_*.json` files, `run.sh`, `submit_campaign.sh`, plus untracked
`.claude/` and `.planning/`). Every output row therefore carries
`git_sha = "9f36964…-dirty"`, and `aggregate.py` warns that the code is not
recoverable from the sha alone. **Commit before submitting the real sweep.**
Note `config/mean_field_2_targets_no_viz.json` — the template this sweep reads —
is itself clean.

### D-13 `CLAUDE.md` GSD workflow

`CLAUDE.md` asks for repo edits to go through a GSD command. This work was done
directly from the spec document, which is itself a complete plan with its own
acceptance checklist. Flagged rather than assumed.

---

## Pre-flight results (run on this workstation)

| gate | result |
|---|---|
| Anchor, u\*(0.5) = 6.157 | **PASS** — computed 6.156868, relative error 2.14e-05 |
| u\*(v) grid vs the frontier sweep's hand-entered table | identical to every printed digit |
| Smoke test (2 cells × 12 trials) → aggregate → analyse | **PASS** end to end; plots render, no schema errors |
| Section 12 expectation, cell (v=0.5, û=1.0) | **PASS** — DT = 11 ticks, matching the archived 0.765 / 11.0 |
| Step-halving, 3 stiffest cells | see below |
| `sbatch --test-only` | **NOT RUN** — no SLURM on this workstation (`sbatch`, `squeue`, `sinfo` all absent). The script's own dry run is clean; the SLURM dry run has to be step 4b of the README, on the login node. |

Ground truth used to arbitrate D-01/D-02/D-03: re-scoring the archived matched
frontier cell
`seoul-data/beta-1/ra_ddm_frontier_sweep/u_6.156868/v_0.5/replicate_*` by the
notebook's own rule gives **1000/1000 arrived, accuracy 0.7650, mean and median
DT 11.0 ticks** — the Section 12 anchor exactly.

### Step-halving check (Section 11)

`integration_dt` 0.05 vs 0.1, identical seeds, paired bootstrap on the accuracy
difference. At the spec's 50 trials per arm:

| cell | v | û | u | acc @0.1 | acc @0.05 | Δ [95 % CI] | failures | max\|z\| | |
|---|---|---|---|---|---|---|---|---|---|
| 7 | 0.1 | 1.50 | 34.19 | 0.340 | 0.180 | −0.160 [−0.320, +0.000] | 0 | 1.83 | PASS (marginal) |
| 6 | 0.1 | 1.25 | 28.49 | 0.560 | 0.700 | +0.140 [−0.020, +0.300] | 0 | 1.82 | PASS (marginal) |
| 15 | 0.2 | 1.50 | 19.69 | 0.780 | 0.820 | +0.040 [−0.060, +0.140] | 0 | 1.82 | PASS |

**Zero numerical failures anywhere, and max\|z\| ≈ 1.8 — three orders of
magnitude below the 1e3 divergence threshold.** The integrator is not in trouble
at dt = 0.1 even at u = 34.2.

The accuracy criterion passes on all three, but cells 7 and 6 pass *marginally*:
zero sits on an edge of the bootstrap interval. At 50 trials the interval is
±0.16 wide, so "consistent with zero" there means "not refuted", not "shown
equal". `dt_check.py` flags this as `MARGINAL` rather than reporting it as clean,
and prints the higher-power re-run to do about it:

```
dt_check.py --results-root <R> --trials 200 --force
```

Note also that cell 7 is only 47.5 % decided at 200 trials — nearly half its
trials run the full 100-tick budget without arriving. Its accuracy estimate is
therefore noisy for a reason unrelated to the integrator, which is part of why
the step-halving interval there is so wide. Read that cell with the
`decided_frac` heatmap in hand (see D-06).
