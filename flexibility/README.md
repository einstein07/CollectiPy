# Flexibility campaign

Ring attractor at two gains vs. a collapsing-bound DDM, under a mid-trial world
change, swept over option quality difference, with the sensory noise realisation
**shared** across models.

Implements `FLEXIBILITY_RA_DDM_DESIGN.md`; section numbers below refer to it.

Self-contained and separate from `campaign/`, which belongs to the speed/accuracy
frontier campaign and locks a different operating point. The two share the codebase
and the config idioms, not their constants.

## Arms

| arm | model | decision rule |
|---|---|---|
| `ra_u6.2` | mean-field ring attractor | kernel `v = 0.5`, gain `u = 6.2` (near-critical) |
| `ra_u8` | mean-field ring attractor | kernel `v = 0.5`, gain `u = 8` (rigid) |
| `ddm_bellman` | embodied pure DDM | collapsing Bellman boundary, `terminal: halt_sprt` |

3 arms × 22 δ × 100 replicates = **6 600 runs, 660 array tasks**.

## Layout

| file | what it owns |
|---|---|
| `factors.py` | every locked value, the arms, the δ grid, the cost of error |
| `matrix.py` | the ordered condition list and the DDM's predicted operating point |
| `seeds.py` | the trial seed — arm-independent, so the arms are paired |
| `genconfig.py` | per-(condition, replicate) configs; the arms-matched assertion |
| `preflight.py` | the grid report and pre-launch checks |
| `precompute_tables.py` | one Bellman table per `ddm_bellman` condition |
| `run_chunk.py` | one array task |
| `u_resolution_check.py` | the §9 side experiment at δ = 0 |

Templates — never modified, deep-copied and overridden in memory:
`config/mean_field_2_targets_flexibility.json` (RA) and
`config/embodied_ddm_flexibility.json` (DDM).

## Running it

```bash
python3 -m flexibility.preflight                 # always first; login-node safe
bash submit-flexibility-sweep-bwunicluster.sh    # DRY_RUN=1 to stop before submission
```

The submit script does the rest: it submits the Bellman precompute as its own compute
job and then the array with `--dependency=afterok` on it. **Nothing is simulated on
the login node** — it runs only the preflight (a few seconds of arithmetic) and two
`sbatch` calls.

One task locally:

```bash
python3 -m flexibility.run_chunk --only ra_u8__d1.9296pct:0 --results-root <dir>
```

Output: `<root>/{arm}/diff_{pct}/replicate_{n}/`, the arm level replacing the
historical `u_{value}` level, so existing analysis tooling reads it unchanged. No
summariser runs in the array; the §6 measures are extracted downstream from the
snapshot tables and `events.json`, which carries `bifurcation_events` and
`swap_events`.

## What the design rests on, and how it is enforced

**The shared noise realisation.** The trial seed is derived from `(δ, replicate)`
with the **arm excluded from the key**, so all three arms replay the same realisation
at the same cell. Verified in real output: the three arms log bit-identical percepts
tick by tick until the swap fires, and the swap fires at different times only because
the arms commit at different times.

Three preconditions are fatal under `shared` mode and are checked at model
construction: arena and agent `ticks_per_second` must be equal; the RA's `sigma_s`
must be 0; the DDM's `eta_rate` must be `[0, 0]`. All three are written from
`factors.py`, not read from the templates.

**The strength swap, not a position swap.** `post_bifurcation_swap.attributes` must
be `["strength", "color"]`. The arena defaults it to `("position",)` when the key is
absent, and a position swap is a **no-op** here: both models sign their decision
variable by `target_ids` order, so exchanging coordinates leaves `q0 − q1` untouched.
That failure is silent — clean runs, a full dataset, and a 0% reversal rate that
reads as a rigidity result. `genconfig` writes the whole block unconditionally.

**Arms differ only in the decision rule.** `_patch_shared` writes every locked
parameter into every arm regardless of what the template held, and the arena block is
written wholesale rather than patched, so not even a cosmetic key can differ.
`assert_arms_matched` generates two configs and diffs them; the preflight runs it
across four probe δ values. `_patch_model` **requires** the model block it targets to
exist, so a mismatched template raises instead of silently no-opping.

## What the preflight will tell you, and why none of it is a problem

- **Reversal is physically possible at all 21 non-zero cells.** `T_rev` peaks at
  3.50 s against a 50 s travel budget, so a 0% reversal rate anywhere is a property
  of the model, never of the arena. This is the campaign's key precondition, and a
  failure here blocks submission.
- **One cell (δ = 1%) has a predicted initial accuracy below 75%** (70.6%). There a
  minority of trials commit to the worse option, where the swap makes the choice
  correct and there is nothing to reverse. §6 analyses those separately, so the cell
  costs effective replicates rather than being wrong.
- **Reversal latency is discretisation-limited above δ = 1.245%** at
  `ticks_per_second = 1`. Expected and accepted: reversal **rate** is the headline
  measurement and is resolved at every cell. Report latency only below that edge.
- **Pinned strengths mean the mean drive falls 5.00 → 3.00** across the sweep. The
  DDM is blind to it; the ring attractor is not, so at the top of the grid the RA
  arms are driven differently rather than merely discriminating a larger difference.
  Chosen for continuity with the earlier RA sweeps; `mean_drive` is recorded per cell
  so it is reported rather than hidden.

## Bellman tables

`|A|` changes with δ, so there are 22 distinct tables for the DDM arm. The swap
preserves `|A|` and flips only its sign, so **one table per condition stays valid on
both sides of the world change**. A cold solve is ~9 s; letting 100 replicates each
re-solve the same Crank–Nicolson PDE would waste ~100× the needed compute, so they
are solved once into a shared cache before the array runs.

The precompute runs **serially by default**. Each solve runs a full simulator, which
itself forks an arena, a manager and a detector process, so nesting a worker pool on
top multiplies processes for no real gain — 22 solves at ~9 s is about 3.5 minutes.
`precompute_tables` refuses to run without `SLURM_JOB_ID` unless given
`--allow-no-slurm`, because a partial cache from a killed login-node run looks
populated and would then be silently re-solved per replicate.

`N_t` carries a 5% margin: the model measures `r0` from the geometry at *evidence
onset*, one step in, so the realised horizon runs ~1.6% above the static estimate. A
stale `N_t` does not shorten the horizon — `T_max` is auto-derived — it coarsens the
solver grid.

## Measured costs

- RA replicate: **~1 s** (10 replicates in 9–10 s).
- DDM replicate with a warm cache: **~1 s**; cold table solve **~9 s**.
- 6 600 runs, 660 array tasks at `CHUNK = 10`, under the 1 000-task cap.
