# Flexibility: ring attractor vs. collapsing-bound DDM, on matched noise

Post-commitment reversal under a mid-trial world change, swept over option quality
difference, with the sensory noise realisation **shared** across models.

This supersedes the earlier `flexibility-experiment-design.md` entirely. Nothing in
that document is assumed here.

---

## 1. Question

A single agent commits to one of two targets; the arena then exchanges the two target
strengths; the agent may or may not reverse before it arrives. The measured outcome is
**reversal**, not accuracy.

Three arms see an identical world, identical geometry, and — the point of the design —
**the identical noise realisation**:

| arm | model | decision rule |
|---|---|---|
| `ra_u62` | mean-field ring attractor | kernel `v = 0.5`, gain `u = 6.2` |
| `ra_u8` | mean-field ring attractor | kernel `v = 0.5`, gain `u = 8` |
| `ddm_bellman` | embodied pure DDM | collapsing Bellman boundary, `terminal: halt_sprt` |

`u = 6.2` sits just above the tabulated critical coupling for `v = 0.5`
(`u* = 6.156868`, from the matched (v, u\*) table used by the RA/DDM frontier sweep) —
0.7% supercritical, so *near*-critical rather than exactly critical. `u = 8` is deep
in the attractor regime. The contrast is flexible-but-slow against rigid-but-fast, with
the DDM as the normative reference.

The `δ = 0` cell is not filler. With equal strengths the exchange is a no-op, so that
cell measures spontaneous symmetry-breaking and the **spontaneous** reversal rate — the
null against which every other reversal count is scored.

---

## 2. Factors

**Arms** — 3, as above.

**Quality difference δ** — 22 values: one symmetric point plus 21 log-spaced from 1% to
80% (ratio 1.245):

```
0
1.000  1.245  1.550  1.930  2.402  2.991  3.723  4.635  5.771  7.184  8.944
11.135  13.863  17.259  21.486  26.750  33.302  41.460  51.616  64.259  80.000    (%)
```

**Strengths — pinned, not mean-preserving.** `static_0` is held at `Q̄ = 5.0` and
`static_1` is weakened:

```
strength_static_0 = 5.0
strength_static_1 = 5.0 · (1 − δ)
```

so `A = |ΔQ| = 5δ`. This preserves continuity with the earlier RA sweeps. It carries a
known consequence that belongs in the methods rather than in the discussion: the *mean*
drive falls from 5.0 at δ = 0 to 3.0 at δ = 0.8. The DDM is blind to this — it sees only
the difference — but the ring attractor is not, because total drive shifts the field's
operating point and therefore the effective distance from the critical coupling. At the
top of the grid the `ra_u62` arm is being driven differently, not just discriminating a
larger difference. Report δ and mean drive together for the RA arms.

**Replicates** — 100 per (arm, δ).

Total: 3 × 22 × 100 = **6 600 runs**, 660 array tasks at 10 replicates per task, under
the 1 000-task array cap.

---

## 3. What is held identical across arms

The comparison is only worth making if the arms differ in the decision rule and in
nothing else.

### 3.1 Noise realisation — the core of the design

`sensory_stream.mode: "shared"`, `frozen_sd: 0.0`, `white_rate: 0.07071068`,
`seed: null`. The seed resolves per trial from the arena `random_seed`, and that seed is
derived from `(δ, replicate)` **with the arm excluded from the key**:

```
trial_seed = H(campaign_seed, "trial", δ_token, replicate)
```

The historical generator keyed on `md5(f"{u}_{diff}_{run}")`, which put the gain in the
key, so every arm drew a *different* stream and the arms were unpaired. Dropping the arm
makes all three replay the same realisation at the same (δ, replicate), which is what
licenses paired statistics.

Sharing the seed is necessary but not sufficient — the models must also *consume* the
stream identically, and they do. `SharedPerceptStream` keys every draw by
`(trial_seed, kind, target_id, tick)` alone, so the realisation does not depend on a
model's internal sub-stepping: the DDM's `n_sub = 16` and the RA's 500 field steps per
tick both see one draw per arena tick. Both models reach it through the same
`TargetModel._apply_percept_stream` spine.

`c = √2 · white_rate = 0.100` exactly, which is the DDM's noise scale. The RA has no
`c`, but consumes the same percept `q̂ᵢ = qᵢ + βᵢ + εᵢ(t)`.

Three preconditions are fatal under `shared` mode and are checked at model construction
(`models/percept_stream.py`):

- arena `ticks_per_second` **must equal** the agent's — both are 1 here;
- the RA's `sigma_s` **must be 0** (its model-owned sensory noise has moved upstream;
  leaving the historical 1.0 would add a second, *unshared* noise source and silently
  unpair the arms while every seed still matched);
- the DDM's `eta_rate` **must be `[0.0, 0.0]`** for the same reason.

The RA's internal `sigma = 1.5` is neural noise with no DDM counterpart and is
deliberately **not** shared.

### 3.2 Geometry and kinematics

Targets at `(0.4330127018922193, ∓0.25)`, i.e. exactly `R = 0.5 m` at ±30° (60°
separation). `linear_velocity = 0.01` for both models, `angular_velocity = 120`,
termination radius `0.05`, square arena `side = 2`.

- travel budget `T_b = R/v = 50 s`
- minimum turn radius `v/ω = 0.0048 m`, well inside the termination radius, so a
  reversed agent can turn around rather than orbit
- Bellman arrival horizon `T_max = r0/v = 43.3 s`, with `r0 = R·cos30° = 0.433`

### 3.3 The world change

`environment.post_bifurcation_swap`, with

```json
{"pairs": [["static_0.s#0", "static_1.s#0"]], "delay_ticks": 1,
 "attributes": ["strength", "color"]}
```

**`attributes` is mandatory.** `arena._normalize_post_bif_swap_config` defaults it to
`("position",)` when absent, and a position swap is a *no-op* for this experiment: both
models sign their decision variable by `target_ids` order, so exchanging coordinates
leaves `q0 − q1` untouched and produces no drift reversal. Exchanging **strength** flips
the drift sign while preserving `|ΔQ|`. The RA template still omits this key (§7 item 4),
so a run made from it as-is swaps nothing that matters.

`delay_ticks = 1` at `ticks_per_second = 1` is a 1.0 s physical delay. The trigger is the
first behavioural bifurcation event, from whichever detector the arm carries; both model
families expose one, and both are configured with
`alignment_consecutive_ticks: 1` so the swap timing is identical across arms.

The model-internal `quality_swap` stays `enabled: false` in the DDM arm — leaving both
enabled would make the two swaps cancel to a net no-op.

---

## 4. Operating point

```
white_rate       = 0.07071068   →  c = √2 · white_rate = 0.100
ticks_per_second = 1            →  dt = 1.0 s  (arena and agent)
linear_velocity  = 0.01         →  T_b = 50 s
Q̄                = 5.0          →  A = 5δ
cost of error    = 1.0          (c_e, the DDM's criterion)
time_limit       = 1000 s
```

`c_e = 1.0`: under `geometric_error_mode: terminal_categorical` the criterion is a 0–1
loss expressed in seconds, so this says *being wrong costs as much as one second of
delay*. Against a 50 s budget an error is cheap, which is why commitment is fast.

### 4.1 What the DDM will do

From the model's own boundary solver (`myopic_z`, i.e. `sinh(a*) + a* = (A/c)²(c_e/c_τ)`,
`z = a*c²/2A`), with `c_τ = 1 − cos30° = 0.13397`:

| δ (%) | A | A/c | z | initial acc. | t_c (s) | T_rev (s) |
|---|---|---|---|---|---|---|
| 0 | 0 | 0 | — | 50.0% | — | — |
| 1.000 | 0.050 | 0.50 | 0.0875 | 70.6% | 0.72 | 3.50 |
| 1.245 | 0.062 | 0.62 | 0.1015 | 78.0% | 0.91 | 3.26 |
| 1.550 | 0.078 | 0.78 | 0.1120 | 85.0% | 1.01 | 2.89 |
| 1.930 | 0.097 | 0.97 | 0.1167 | 90.5% | 0.98 | 2.42 |
| 2.402 | 0.120 | 1.20 | 0.1155 | 94.1% | 0.85 | 1.92 |
| 2.991 | 0.150 | 1.50 | 0.1100 | 96.4% | 0.68 | 1.47 |
| 3.723 | 0.186 | 1.86 | 0.1018 | 97.8% | 0.52 | 1.09 |
| 4.635 | 0.232 | 2.32 | 0.0922 | 98.6% | 0.39 | 0.80 |
| 5.771 | 0.289 | 2.89 | 0.0822 | 99.1% | 0.28 | 0.57 |
| 7.184 | 0.359 | 3.59 | 0.0725 | 99.5% | 0.20 | 0.40 |
| 8.944 | 0.447 | 4.47 | 0.0633 | 99.7% | 0.14 | 0.28 |
| 11.14 | 0.557 | 5.57 | 0.0549 | 99.8% | 0.10 | 0.20 |
| 13.86 | 0.693 | 6.93 | 0.0473 | 99.9% | 0.07 | 0.14 |
| 17.26 | 0.863 | 8.63 | 0.0406 | 99.9% | 0.05 | 0.09 |
| 21.49 | 1.074 | 10.7 | 0.0346 | 99.9% | 0.03 | 0.06 |
| 26.75 | 1.338 | 13.4 | 0.0295 | ~100% | 0.02 | 0.04 |
| 33.30 | 1.665 | 16.7 | 0.0250 | ~100% | 0.02 | 0.03 |
| 41.46 | 2.073 | 20.7 | 0.0211 | ~100% | 0.01 | 0.02 |
| 51.62 | 2.581 | 25.8 | 0.0178 | ~100% | 0.01 | 0.01 |
| 64.26 | 3.213 | 32.1 | 0.0150 | ~100% | 0.00 | 0.01 |
| 80.00 | 4.000 | 40.0 | 0.0126 | ~100% | 0.00 | 0.01 |

Three facts follow, and they shape the analysis:

1. **Reversal is geometrically feasible at every δ.** `T_rev` peaks at 3.50 s against a
   50 s budget, so a 0% reversal rate anywhere is a property of the *model*, never of the
   arena. This is what makes the whole grid interpretable, and it is worth checking in
   the logs: `_check_reversal_feasibility` prints `d/v` against `delay + 2z/A` at runtime.
2. **The first choice is reliable across most of the grid** — 70.6% at δ = 1%, 90.5% by
   δ = 1.93%, ≥99% above δ = 5.8%. Trials that commit to the *worse* option have nothing
   to reverse (the swap makes them right) and are analysed separately, never pooled. At
   δ = 1% that is ~29% of trials, falling fast.
3. **Latency is resolution-limited; rate is not.** At `dt = 1 s`, commitment time is
   sub-tick everywhere (`t_c ≤ 1.01 s`) and reversal time drops below one tick above
   δ ≈ 3.7%. **Reversal rate is therefore the headline measurement** and is fully
   resolved at every cell. Reversal latency is reported only for δ ≲ 3.7% and flagged as
   discretisation-limited above it. The tick rate is deliberately left at 1.

No closed-form prediction is quoted for the RA arms. The DDM formulas above are a DDM
result; the RA's departure from them is the measurement.

---

## 5. Predictions

1. `ddm_bellman` reverses whenever the post-swap evidence can traverse `2z` inside the
   remaining travel time — which, per §4.1, is *always* on this grid. Its reversal rate
   should therefore be high and roughly flat above the point where the first choice
   becomes reliable, with the residual set by the fraction of trials that committed to
   the worse option and by the swap's arrival before commitment at large δ.
2. `ra_u62`, near-critical, is slow, noise-dominated and **flexible**: the bump is
   weakly pinned, so a small input asymmetry can depin it. Expect reversal, with high
   variance and long latencies, and possibly multiple reversals per trial.
3. `ra_u8` is fast, low-variance and **rigid**: the attractor is deep enough that the
   post-swap input may fail to depin the bump at any δ in the grid. **If `u = 8` fails to
   reverse where the DDM reverses deterministically, that is the result** — a hysteresis
   no accumulator model produces.
4. At δ = 0 all three arms measure spontaneous reversal only; any non-zero rate there is
   the noise floor for the rest of the grid.

The saturated top of the grid (δ ≳ 20%) is a **positive control for the reversal
machinery**, not a graded measurement: the DDM reversing there proves the swap fired, the
strength change propagated through `pack_objects_data` → GPS → percept, and the reversal
path works — so an RA arm that does *not* reverse there is showing genuine hysteresis
rather than a plumbing failure.

---

## 6. Measures

Per run, to `events.json` and the snapshot tables:

**Commitment** — whether the agent committed before arrival (a result in its own right,
not a data-quality filter: "never commits" is a plausible and interpretable
outcome for the near-critical arm); `t_commit`; which target; whether that target was the
better one *at commitment time*.

**Reversal** — binary reversal before arrival (**the headline measurement**); latency from the
swap event, not from commitment, where resolvable; **number of sign changes** of the
committed identity — multiple reversals are informative for the near-critical arm and
must not be collapsed to a binary.

**Outcome** — terminal target; censoring flag; realised swap time from
`collect_swap_events`. With `quality_swap` disabled the model-internal `pure_ddm_t_swap`
and dwell/release-latency columns stay null by design; the arena's `events.json` carries
`bifurcation_events` and `swap_events` instead.

**DDM arm** — `z` at commitment, realised crossing log-odds, and the boundary trajectory
`z(t)`.

**RA arms** — bump amplitude and width at commitment and at the swap, the order
parameter, and `λ_max` from the bifurcation detector. The depinning story needs the bump
geometry, not just the behavioural outcome — and near the critical point the bump may
never form, which the amplitude trace shows and the behavioural log does not.

Because seeds are shared, the primary statistics are **paired**: McNemar on reversal,
Wilcoxon signed-rank on latency, matched per (δ, replicate).

`g_adapt = 0.0` in both RA arms: with no spike-frequency adaptation the ring attractor
has no internal mechanism for escaping its own basin, so a reversal must be driven by the
**input alone**. This is a choice, and it makes the rigidity result cleaner. A
`g_adapt > 0` arm is the obvious follow-up if `u = 8` proves absolutely rigid.

---

## 7. Template fixes required before launching

`config/embodied_ddm_flexibility.json` is usable as-is apart from the noted items.
`config/mean_field_2_targets_flexibility.json` is **not a template** — it is a dumped run
config — and must be normalised.

Status as of the latest template edit — items 2, 3 and 6 have been fixed:

| # | file | issue | fix | status |
|---|---|---|---|---|
| 1 | RA | wrapped in a top-level `"data"` key. `Config.environment` reads `self.data["environment"]`, so **the entire environment is invisible** — `c.environment` comes back `{}` | unwrap to `{"environment": …}` | **OPEN — blocking** |
| 2 | RA | arena key was `"arena"` (singular) | `"arenas"` | fixed |
| 3 | RA | `"radius": 1` — `SquareArena` reads `side` and silently ignores `radius` | `"side": 2` | fixed |
| 4 | RA | `post_bifurcation_swap` has no `attributes` → defaults to `("position",)`, a **no-op swap** | `["strength", "color"]` | **OPEN — silent** |
| 5 | RA | `white_rate: 0.035` | `0.07071068` | **OPEN** |
| 6 | RA | `sigma_s` unset; fatal under `shared` mode | `0.0` | fixed |
| 7 | RA | `u: 6.156868`; positions `0.433`; `static_1: 4.95` | set per arm/condition by the generator | not required in the template |
| 8 | RA | `results.base_path` still points at a previous sweep's output | overwritten per replicate by the generator | not required in the template |
| 9 | DDM | `static_1.strength` currently `1.0` | overwritten per condition; base value cosmetic | not required |
| 10 | DDM | `bellman.N_t` sized for the old velocity | `T_max` stays `null` (model derives `r0/v`); size `N_t` so solver `dt ≤ 1 ms` at `T_max ≈ 43.3 s` | set by the generator |
| 11 | DDM | `cost_ratio` | `1.0` | set by the generator |
| 12 | both | `time_limit`, `ticks_per_second`, `linear_velocity`, arena `side`, `white_rate` | written from one constants module, never read from the templates | by construction |

Item 1 is blocking and item 4 is the dangerous one: a no-op swap produces a clean run,
a full dataset, and a 0% reversal rate everywhere that looks like a scientific result.

A stale `N_t` does **not** shorten the horizon — `T_max` is auto-derived — it coarsens
the solver grid. Note also that the model measures `r0` from the geometry at *evidence
onset*, one step in, so the realised horizon runs ~1.6% above the static estimate; size
`N_t` with a small margin so `dt` stays at 1 ms.

---

## 8. Execution

**Generator.** One constants module holds every locked value; the arms, the δ grid, and
the derived per-condition quantities follow from it. Configs are produced by patching a
per-arm template in memory — the template files are never modified. Two rules make a
violation loud rather than silent:

- every locked parameter is written into every arm unconditionally, so two arms at the
  same (δ, replicate) can differ **only** inside the model block; a generated-config diff
  asserts exactly this;
- the arm-specific patch **requires** the model block it targets to exist. The historical
  generator only knew how to reach `mean_field_model` and would have no-opped silently on
  a DDM template.

**Output layout** — `base_path/{arm}/diff_{pct}/replicate_{n}`, the arm level replacing
the historical `u_{value}` level, so existing analysis tooling reads it unchanged.

**Bellman tables.** `|A|` changes with δ, so there are 22 distinct tables for the DDM
arm. The swap preserves `|A|` and flips only its sign, so **one table per condition stays
valid on both sides of the world change**. A cold solve is ~9 s; letting 100 replicates
each re-solve the same Crank–Nicolson PDE would waste ~100× the needed compute. Solve
them once into a shared cache before the array runs.

**Nothing is simulated on the login node.** The precompute is its own compute job; the
array is submitted with `--dependency=afterok:<precompute_job>` so tasks cannot start
until the tables exist. `afterok` rather than `afterany`: a failed precompute leaves a
missing or partial cache, and every task would then re-solve in-process — the exact waste
the cache exists to prevent, discovered only from the wall-clock. Run the precompute
serially: each solve runs a full simulator, which itself forks an arena, a manager and a
detector process, so a nested worker pool multiplies processes for no real gain (22
solves × 9 s ≈ 3.5 minutes serially).

**Before submitting**, a preflight — pure arithmetic, login-node safe — should report the
grid with its predicted operating point, and assert that: `c_τ` matches the live model
code; every arm's template carries the block that arm patches; two arms at the same
(δ, replicate) carry the same `arena.random_seed`, `white_rate` and `linear_velocity` and
differ only in the model block; and reversal is feasible at every cell.

**Benchmark before sizing `--time`**: one RA run at `u = 6.2`, δ = 1% (slowest — near
critical *and* at the floor of the grid), and one DDM run at the same δ including a cold
table solve.

---

## 9. Open items

- **Censoring.** `time_limit = 1000 s` is 20× an uncontested traverse, so censoring
  should be rare and confined to the near-critical arm at small δ. Log the rate per cell
  and report it; do not filter on it.
- **Mean-drive confound.** Recorded in §2 and accepted for continuity with the earlier
  sweeps. If the `ra_u62` result turns out to depend on it, the mean-preserving
  parameterisation `5(1 ± δ/2)` is the control, at the cost of comparability with the
  older data.
- **`u = 6.2` is 0.7% supercritical**, not exactly critical, and finite-`N` effects at
  `N = 30` are largest near the critical point, so the *simulated* system's effective
  critical coupling may sit slightly off the analytical one. A short resolution check at
  δ = 0 across a few gains either side would pin it without growing the main campaign.
