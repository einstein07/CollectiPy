# Flexibility campaign

Ring attractor at two gains vs. a collapsing-bound DDM, under a mid-trial world
change, swept over option quality difference. Implements
`flexibility-experiment-design.md`. Section numbers below refer to that document.

Self-contained and separate from `campaign/`, which belongs to the speed/accuracy
frontier campaign and locks a different operating point (v = 0.05, white_rate =
0.035, ticks_per_second = 10). The two share the codebase and the config idioms,
not their constants.

## Layout

| file | what it owns |
|---|---|
| `factors.py` | every locked value, the arms, the δ grid, the cost of error |
| `matrix.py` | the ordered condition list, the realised operating point, the landmarks |
| `seeds.py` | the trial seed — arm-independent, so the arms are paired |
| `genconfig.py` | per-(condition, replicate) configs; the arms-matched assertion |
| `preflight.py` | grid report and pre-launch checks |
| `precompute_tables.py` | one Bellman table per `ddm_bellman` condition |
| `run_chunk.py` | one array task |
| `u_resolution_check.py` | the §2 side experiment at δ = 0 |

Templates: `config/mean_field_flexibility_base.json` (new, this campaign's own) and
`config/embodied_ddm_flexibility.json`. The shared
`config/mean_field_2_targets_no_viz.json` is **not** used — it carries no
`sensory_stream` and no `post_bifurcation_swap`, and other sweeps depend on it.

## Running it

```bash
python3 -m flexibility.preflight                       # always first
python3 -m flexibility.precompute_tables --cache-dir <dir>
bash submit-flexibility-sweep-bwunicluster.sh          # DRY_RUN=1 to stop before sbatch
```

Locally, one task: `python3 -m flexibility.run_chunk --only ra_uc__d0.5674pct:0
--results-root <dir>`.

Output: `<root>/{arm}/diff_{pct}/replicate_{n}/`, the arm level replacing the
historical `u_{value}` level. No summariser runs in the array; the §7 measures are
extracted downstream from the snapshot tables and `events.json`, which carries both
`bifurcation_events` and `swap_events` (verified: bifurcation at tick 28 → swap at
tick 29 → strengths exchange in the percept at tick 30).

## Where this departs from the design document

Five places. The first two are decisions taken since the document was written; the
rest are corrections to premises that did not survive contact with the code.

**1. The cost of error is fixed at c_e = 1.0, and §4.3's landmarks do not survive
it.** Under `terminal_categorical` the criterion is a 0–1 loss expressed in seconds,
so c_e = 1.0 says *being wrong costs as much as one second of delay*. Against a 50 s
travel budget an error is cheap — 1/50 of a trial — so the agent commits almost
immediately and lands near chance at small δ. That is the criterion doing its job,
not a defect.

§4.2 instead works from a fixed operating point (a 10% error rate) and derives every
landmark, and therefore the whole grid, from it. The boundary is not free to sit
there: it comes from the policy, given the cost of an error, and at a fixed cost it
moves with δ. So the design document's landmark values do not describe this campaign.
Measured by the preflight:

- **Reversal is always geometrically feasible.** T_rev tops out at 3.73 s against the
  50 s budget. At a large c_e most of the grid physically cannot reverse, so this is
  the main thing c_e = 1.0 buys.
- **Initial accuracy is at chance over the lower half of the grid**: 50.2% at
  δ = 0.1%, 57% at δ = 0.57%, reaching 90% only near δ = 1.9%. Twelve of the
  twenty-two non-zero cells commit at under 75%. There, about half the trials commit
  to the *worse* option, where the swap makes the choice correct and there is nothing
  to reverse; §7 requires those analysed separately, so the effective replicate count
  in those cells is roughly half of `REPS`.
- **t_c is non-monotonic in δ** and sub-tick everywhere (it peaks at 1.01 s near
  δ = 1.7%). §4.3's regime map orders cells by t_c assuming it falls as 1/δ², so that
  map does not describe this campaign. Commitment latency is not measurable at
  `ticks_per_second = 1`; **reversal** latency is, and reversal is the DV (§1).
- Three of §4.3's four landmarks — `no_commit`, `reversal`, `commit_three_tick` —
  **cease to exist**. The preflight prints them as `NOT REACHED` rather than
  omitting them, because "the reversal boundary is not on this grid" is a finding.

Landmarks are therefore solved numerically on the boundary the model's own solver
returns (`matrix.landmarks`, scan-then-bisect, no monotonicity assumed) rather than
inverted from §4.2's closed form, and cells are classified by what they yield for the
DV (`matrix.regime_of`) rather than by t_c.

**The grid is now mis-centered, and it is left that way deliberately.** The usable
band — reversal possible, first choice reliable enough for a reversal to be defined,
latency resolvable — is δ ∈ [1.14%, 1.46%], containing **one** of the twenty-two
non-zero grid points. The preflight reports this and names two independent levers,
applying neither, because both are design decisions:

| lever | effect |
|---|---|
| `ticks_per_second` 1 → 10 | band becomes [1.14%, 8.63%], ratio 7.6, **7** usable points |
| re-place the log leg (currently 0.10%–4.00%) | its lower edge sits below the band; 11 of 18 points are at near-chance accuracy |

The band's lower edge is set by the noise and the cost of error and does **not** move
with the tick rate; only the upper edge does, since that edge is pure sampling
resolution.

**2. The `ddm_static` arm is removed.** §2 specifies four arms; this campaign runs
three. The static-bound DDM and its calibration machinery are gone, along with
`STATIC_Z` and `calibrate_static_z.py`. The comparison is now the ring attractor at
two gains against a single collapsing-bound accumulator.

This retires §1's **prediction 2**, which contrasted DDM-bellman with DDM-static, so
the campaign makes no claim about it. Worth recording that it would have been a weak
test at this criterion anyway: the Bellman boundary falls 83.8% over the trial, but
commitment happens at t_c = 0.96 s of a 44 s horizon, so only 3.3% of the collapse
precedes it — a static control matched at t = 0 would have committed from nearly the
same boundary, which is the mechanism prediction 2 rests on. Predictions 1, 3 and 4
are unaffected: they concern reversal feasibility in closed form and the two ring
attractor arms.

**3. The mean-field template §8 describes does not exist.** §8 reads as though there
were an RA flexibility template carrying `post_bifurcation_swap` without
`attributes`, `white_rate = 0.035`, `radius: 1` and `linear_velocity: 0.05`. That
file is `config/mean_field_2_targets_flexibility.json`, which is a **dumped run
config** (wrapped in `"data"`, arena key singular), not a template. The actual
template, `mean_field_2_targets_no_viz.json`, has no `sensory_stream` and no
`post_bifurcation_swap` at all, and is shared with other sweeps. So the RA arm gets a
new template rather than edits to that one.

The §8 concerns still resolved, on the real files:

- *Swap attributes* — **confirmed a real bug.** `attributes` defaults to
  `("position",)` (`arena._normalize_post_bif_swap_config`), and a position swap is a
  no-op here: both models sign the decision variable by `target_ids` order, so
  exchanging coordinates leaves q0 − q1 untouched. Any past RA flexibility run
  without an explicit `attributes` swapped nothing that mattered.
- *Arena seeding key* — not a bug. Both templates use `"arenas"` (plural), which
  `config.py` requires; the singular key in the dumped config is the dump's own
  normalisation.
- *Arena shape, velocity, noise, tick comment, time limit* — all set from
  `factors.py` unconditionally in `_patch_shared`, so no template can drift.

**4. A stale `N_t` coarsens the grid; it does not shorten the horizon.** §6 warns
that the hard-coded 8660 would "silently solve the policy on a 5× too-short horizon".
It would not: `T_max` is `null` and the model derives `r0/v` itself. What a stale
`N_t` does is set dt = T_max/N_t, so 43.3 s over 8660 steps is dt = 5 ms — five times
the intended resolution, and the boundary converges as O(√dt). Still worth fixing,
for a different reason than stated.

`N_t` also carries a 5% margin (`BELLMAN_HORIZON_MARGIN`), because the model measures
`r0` from the geometry at *evidence onset*, one step in, so the realised horizon runs
~1.6% above the static estimate (0.4400 against 0.4330). Without the margin dt lands
at 1.016 ms rather than 1 ms.

**5. Reversal feasibility has two definitions and they disagree near the boundary.**
§4.3's landmark measures from t = 0 (`t_c + t_delay + T_rev ≤ T_b`); the model's
runtime check measures the *remaining* travel from the moment of commitment
(`d/v` vs `delay + 2z/A`). Both are reported per cell —
`pred_reversal_feasible` and `pred_reversal_feasible_post_commit` — since it is the
model's laxer version that produces the runtime warning.

## Measured costs

- RA replicate: **~1.2 s** (10 replicates in 12.1 s, δ = 8%, u = 8).
- Bellman cold table solve at N_t = 45467, N_x = 1601: **~8 s**. 23 tables, so the
  precompute is ~3 minutes at 8 workers — against ~2.4 hours per condition of pure
  duplication if 100 replicates each re-solved it. The shared `--cache-dir` is not
  optional.
- 6 900 runs, 690 array tasks at `CHUNK = 10`, under the 1 000-task cap.
