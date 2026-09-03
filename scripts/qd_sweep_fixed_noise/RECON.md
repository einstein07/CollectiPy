# RECON — QD sweep at fixed noise (RA (u, v) surface + frozen-DDM misspecification)

Step 1 of `qd-sweep-fixed-noise-experiment.md` (§10.1). Every interface this
campaign inherits from the frontier implementation, located in the repo, plus
every point where the spec document (or an implementation choice) departs
from the repo or needed interpretation. Repo root for all paths:
`CollectiPy/`. The frontier campaign's own [`RECON.md`](../ra_ddm_frontier/RECON.md)
remains the authority on everything inherited unchanged (frames D-01,
calibration D-09, halt policy D-13/D-14, seeding D-02).

---

## §2 parameter block — reproduced with resolved values

```python
DTH_DEG          = 60
DIFF_BP          = [50, 100, 200]          # δ_Q in basis points; THE knob
WHITE_RATE       = 0.07071068              # PINNED LITERAL — see D-01
NOISE_SCALE_C    = 0.1                     # = √2·WHITE_RATE (evidence channel)
S0               = 5.0                     # static_1 = S0·(1−diff):
                                           #   4.975 / 4.95 / 4.9
U_GRID           = [0.0, 2.0, 2.5, …, 35.0]   # 68 levels (0, then 2–35 step 0.5)
V_GRID           = [0.1, 0.2, …, 1.0]      # 10 kernels — full range
N_RUNS_RA        = 1000
N_RUNS_DDM       = 1000
RA_SURFACE_DIFFS = [50, 100, 200]
FROZEN_CE        = [3, 20, 300]
T_MAX            = RA 1000 ticks (≡ s at 1 tick/s); DDM 60 s   # R-0 below
DTH_DEG          = 60
```

Design table (resolved; asserted in `qd.py` at import):

| δ_Q | A = S0·δ | A/c | k = 2A/c² | static_1 |
|---|---|---|---|---|
| 0.5 % | 0.025 | 0.25 | 5 | 4.975 |
| 1.0 % | 0.050 | 0.50 | 10 | 4.95 |
| 2.0 % | 0.100 | 1.00 | 20 | 4.9 |

The u = 0 → 2 gap is the user-specified design; u = 0 is the no-coupling
control (10 replicate cells per δ_Q). Standing conventions honored: absolute
u only — no û, no u\*(v), no relative coupling in any grid, manifest, label,
figure or this document; one knob (`DIFF_BP`) tunes difficulty.

## The recon slots the spec asked for

### R-0 — timeout inheritance (§2 `T_MAX`)

Read off the halted campaign's replicate configs directly
(`seoul-data/beta-1/ra_ddm_frontier_slices/…_1.0/cells/absolute/v_0.5/u_7.5/replicate_1`
and `…ddm_halt/…_1.0/points/bellman/ce_3/replicate_1`): **RA
`time_limit = 1000` ticks at 1 tick/s; DDM `time_limit = 60` s at 1 tick/s**
(61 ticks). Both frames otherwise exactly the frontier templates (D-01 there);
`qd.py` asserts both values on every emitted config.

### R-1 — the `white_rate → c` mapping, measured from logs (§2, BLOCKING)

`scripts/qd_sweep_fixed_noise/r1_noise_convention.py`, 200 probe replicates
per actual δ_Q through the real patcher + seed path (static bound b = 6.0,
far outside reach, so the accumulator runs uninterrupted), measuring the
per-unit-time increment variance of `x` in `_ddm.csv`:

| measurement (n = 5000 increments each) | result |
|---|---|
| ĉ at actual 50 bp | **0.0983 ± 0.0010** (travel 0.0967 / halt 0.0989) |
| ĉ at actual 100 bp | **0.1004 ± 0.0010** (travel 0.1023 / halt 0.0997) |
| ĉ at actual 200 bp | **0.1006 ± 0.0010** (travel 0.1006 / halt 0.1007) |
| pooled ĉ (weighted) | **0.0998** vs designed c = 0.1 — rel. err 0.2 % |
| drift ladder mean(Δx)/dt | **0.0255 / 0.0475 / 0.1009** vs A = 0.025 / 0.05 / 0.1 (all within 4 SE) |
| per-target percept SD per √s (η̂) | 0.0698–0.0711 vs white_rate 0.07071068 |
| evidence-diff SD from `_percept.csv` | 0.0992 / 0.1000 / 0.1004 |
| `_ddm.csv` `c` column | √2·0.07071068 = 0.10000000266… exactly, every actual δ_Q (the 8-decimal pin's own rounding — asserted to 1e-12 against that value and to 1e-6 against 0.1) |
| logged `white_rate` | 0.07071068, every actual δ_Q |

**PASS** (`results/qd_sweep_fixed_noise/r1_report.json`). The sim's
convention matches the design: per-draw percept SD = `white_rate/√dt` per
target, evidence channel c = √2·η = 0.1, invariant across δ_Q, while the
drift scales 1 : 2 : 4 — SNR genuinely moves. The old campaign's
constant-SNR pathology is measurably gone.

### R-2 — c_e ∈ FROZEN_CE presence in the halted campaign's results (gate §8.2)

`seoul-data/beta-1/ra_ddm_frontier_ddm_halt/ra_ddm_frontier_ddm_halt_1.0/ddm_points.csv`
carries the FULL Bellman grid at n = 1000 — D_ce0.03 … D_ce300 plus the
ceiling extremes D_ce3000/D_ce30000, e.g. `D_ce3` (acc 0.870, median
9.33 s), `D_ce20` (0.958, 9.84 s), `D_ce300` (0.996, 12.04 s) — so every
FROZEN_CE value (D-12: all twelve) has its reference — and
the static grid includes `S_b0.004189` (0.708) and `S_b0.1579` (0.907), the
two sweep-derived optima. Gate 2 references exist and are seed-paired with
this campaign at actual = 100 bp (verified: `env_seed(60, 100, 1)` =
2034332064 and `model_seed("ddm-bellman", 60, 100, 1)` = 1352414336 match
the halted campaign's `run_meta.json` bit-for-bit).

### R-3 — solver entry points for the frozen controllers (§4)

- **Frozen Bellman:** `embodied_pure_ddm.A_expected` (established config
  key; `src/models/movement/embodied_pure_ddm_model.py` `_resolve_A_expected`
  / `ddm_systems.update_A_hat`). Under `drift_knowledge 'known_magnitude'` +
  `A_source 'ensemble'` + explicit `A_expected`, the agent's believed |A| is
  a block constant = **A_design**; the Bellman table, z_halt and the halt
  guard are all solved with it (`_bellman_threshold`, line ~1046), while the
  percepts carry the ACTUAL strengths. The believed noise c = √2·white_rate
  comes from the pinned stream rate, so the noise belief is always correct —
  only the drift belief is misspecified, exactly §4. The model logs the
  design/actual disagreement (a WARNING per run — expected off-diagonal) and
  the table cache key includes A, so the 3 designs × 3 c_e give 9 distinct
  cached tables shared across actual conditions.
- **Frozen static:** `bellman.static_bound = b` (the §2b degenerate flat
  table through the same machinery, frontier D-14); `cost_ratio` pinned 0.0.
- **Analytic machinery:** `models/bellman_boundary.myopic_z`, `solve_z_halt`,
  `halt_exit_potential` — used for the per-point halt budget
  (`qd.ddm_budget`) and reporting, never for the freeze itself (D-03).

---

## Discrepancies and decisions

### D-01 — how the coupling died, structurally (§2's reason for existing)

The halted campaign swept δ_Q by editing `frontier.DIFF` per tree, and
`frontier.WHITE_RATE = round(√2 · QUALITY_DELTA, 8)` was **derived** from it
— so its 0.5 % / 2 % trees ran at white_rate 0.03535534 / 0.14142136
(verified in their replicate configs) and c = 2ΔQ at every δ_Q: SNR never
moved. In `qd.py`, `WHITE_RATE = 0.07071068` is a literal, `NOISE_SCALE_C =
0.1` a literal, and nothing in the module computes any noise quantity from
δ_Q. Enforcement is layered, all blocking:

1. import-time: `WHITE_RATE == 0.07071068` and `√2·WHITE_RATE == 0.1 ± 1e-7`;
2. per-config (`qd.assert_config`, run on every emitted config by the
   patchers): `white_rate == 0.07071068` exactly, DDM `eta_rate == [0, 0]`
   (no second noise path), RA `sigma_s == 0`, strengths ==
   (S0, S0·(1−actual)) exactly;
3. structural (`assert_noise_invariant_across_actuals`, run at manifest
   generation and by R-1): the entire `sensory_stream` block (seed excluded)
   must be byte-identical across probe configs at all three actual δ_Q;
4. empirical (R-1): ĉ measured from logs at every δ_Q; the drift ladder
   proves SNR moves.

Note the trap this closes: at δ_Q = 1 % the coupled and pinned values
coincide numerically (√2·0.05 = 0.07071068), which is exactly how the
coupling hid. The assertions check the literal and the cross-δ_Q invariance,
never the 1 % coincidence.

### D-02 — "the halted campaign" identified

`seoul-data/beta-1/ra_ddm_frontier_slices/ra_ddm_frontier_slices_{0.5,1.0,2.0}`
(RA) and `…/ra_ddm_frontier_ddm_halt/ra_ddm_frontier_ddm_halt_{0.5,1.0,2.0}`
(DDM, halt-at-midpoint, both families). Only the **1.0 trees** are valid at
this campaign's calibration (their white_rate is the pinned value); they are
the §8.2 / §8.4 references. The 0.5 / 2.0 trees are the coupling-defeated
runs — background evidence for D-01, never a gate.

### D-03 — the frozen static b\*: interpretation of §4, and why

§4: "static b\*_cost(design) and b\*_RR(design) — derived analytically via
the solver's own machinery (quasi-static/Wald at k_design), not by new
sweeps. Cross-check the design = 100 bp values against the sweep-derived
0.0042 and 0.1579…". Three facts pin the interpretation:

1. Gate §8.2 demands "the static b\* points must reproduce those results
   within CIs" against the halted campaign — only possible if the design-100
   frozen bounds ARE points of its swept grid, i.e. **0.004189 and 0.1579**
   (its `static_bstar.json` optima; 0.0042 in the spec is that grid level
   printed at 2 s.f.).
2. No pure Wald/quasi-static functional argmin yields 0.0042 at k = 10 (the
   Wald cost anchor is 0.1124 — the halted campaign's own analysis records
   the discrepancy as expected: the embodied task's concurrent travel and
   the substep overshoot shape the empirical optimum).
3. The other designs have no valid sweeps (D-02), so they need an analytic
   transport.

**Taken:** b\*(100 bp) = the sweep-derived optima, **re-derived at manifest
generation from the halted campaign's `ddm_trials.parquet` with the
identical cost/RR functionals** (err + 0.1·E[T_decision], censored trials
charged 60 s; acc/E[T_arrival]) and asserted to land on 0.004189 / 0.1579 —
a real regression of the derivation machinery, blocking. Other designs by
the **Wald log-odds transport b(design) = b\*(100)·k(100)/k(design)** — the
unique analytic map that preserves the controller's believed commitment
log-odds a = k·b (the Wald invariant that sets its believed accuracy
1/(1+e^(−a))) and reproduces the 100 bp values exactly. Resolved freeze
(recorded in `frozen_controllers.json`, `bound_param` in the manifest):

| design | b\*_cost | b\*_RR | (Wald analytic anchors, reported only) |
|---|---|---|---|
| 50 bp | 0.008378 | 0.3158 | 0.0620 / 0.0950 |
| 100 bp | 0.004189 | 0.1579 | 0.1124 / 0.1374 |
| 200 bp | 0.0020945 | 0.07895 | 0.1343 / 0.1372 |

The alternative reading (freeze the Wald-functional argmin at each
k_design) is recorded here for completeness and its anchors are computed and
carried in `frozen_controllers.json` and the §9 static panels — but it
cannot satisfy the spec's own §8.2 cross-check (fact 1), so it is not the
freeze. Note b\*_cost sits below the 1 s-tick evidence substep 0.025 at
every design (discretisation-limited, the frontier D-11 convention) — its
behaviour is dominated by single-substep overshoot, which is exactly the
§5.2 k_eff/b_eff quantification's job to expose.

### D-04 — seeding: inherited verbatim, keyed by ACTUAL δ_Q

`scripts/ra_ddm_frontier/seeding.py` (frontier-v1) is imported, not copied.
`env_seed(60, actual_bp, run_id, "sensory")` → the shared stream;
`model_seed(model, 60, actual_bp, run_id)` → the arena field (model-private
noise; the DDM consumes none — frontier audit d). Model tags `ra`,
`ddm-bellman`, `ddm-static` (§6; both static variants are one model). The
design δ_Q appears in NO seed — it only shapes the model block — so within
each actual δ_Q all 5 × 3 controllers and all 680 RA cells are fully
seed-paired (per-run_id regret CIs in §9 are licensed), and at actual =
100 bp the campaign reproduces the halted campaign's seed universe exactly
(R-2). Env realizations across different actual δ_Q are intentionally
different (distinct trial identities); the §9 pairing claims are always
within-actual. R-1's audit verifies both directions empirically.

### D-05 — RA arm: template inherited; strengths are the only δ_Q field

`qd.build_ra_template()` is the frontier's `build_ra_template()` (archived
`ra_ddm_frontier_sweep` frame: 1 tick/s, time_limit 1000, arena radius 1,
targets (0.433, ∓0.25), `mean_field_model` asserted at N = 30, β = 1.0,
κ = 20, integration_dt = 0.1, integration_time = 50, σ = 1.5, σ_s = 0,
g_threshold 0.6, no thresholding, g_adapt 0, fixed speed 0.05 m/s). The
spec's §3 lists "N = 30, β = 1.0, ?̶ = 0.1, κ = 20" with a mojibake glyph:
read as **τ (integration_dt) = 0.1**, since σ = 1.5 in the inherited
template and §3's governing clause is "unchanged from the frontier
campaign" (repo wins). Only `u`, `v` (per cell) and `static_1.strength`
(per actual δ_Q) are patched. v = 0.1 at N = 30 carries the standing
lattice-pathology caveat — interpret with the ring-size disclaimer until an
N-sweep says otherwise (§3).

### D-06 — Arm B diagonal is CI-equal, not bit-equal, to the halted campaign

On the design = actual = 100 bp diagonal, `A_expected = 0.05` (exact design
constant) while the halted campaign deduced |A| from the declared strengths
(5.0 − 4.95 = 0.050000000000000044 in binary). The Bellman solve inputs
differ at ~1e-15, so marginal trials can flip; gate §8.2 is CI-overlap (the
spec's own criterion), not bitwise. Everything else on the diagonal —
template, seeds, policy — is identical by construction.

### D-07 — manifest shipping (phasing since removed — see D-12)

§10.4's phase split (Arm B + the actual = 100 bp RA slice first) was
implemented as manifest slices and used for the first cluster deployment;
**D-12 removed the phasing** — both arms now submit all three actual δ_Q at
once. What stands from this decision: the submit script never generates
manifests (the §4 freeze needs seoul-data, which lives off-cluster) — they
are generated locally and shipped, the frontier's top-up discipline applied
to the whole campaign; and everything lives in ONE seed universe and one
results tree, so dropping or reordering cells never re-keys anything.

### D-08 — step-halving power (§3: "50 trials")

The spec pins 50 trials/arm; the frontier campaign's record shows 50/arm
cannot resolve a ~10-point accuracy step. `dt_check.py` defaults to the
spec's 50 and prints the caveat; `--trials 200` is the recommended sharper
setting (used for the pre-flight run below alongside the spec's 50).

### D-08b — provenance

Templates, manifests and every `run_meta.json` carry the git SHA at
generation time (frontier D-08 convention). **Commit the campaign before
submitting**, or every SHA carries `-dirty` (the pre-flight rows below do,
correctly — they were run from the working tree).

### D-09 — `CLAUDE.md` GSD workflow

As with both prior campaigns (frontier RECON D-06): implemented directly
from the researcher-supplied spec document, which defines its own §10
workflow, matching the established `scripts/<campaign>/` pattern. Flagged
rather than assumed.

### D-10 — u = 0 monotonicity gate operationalized

§8.3 "u = 0 accuracy must increase with δ_Q": with ~10 000 pooled u = 0
trials per δ_Q (Wilson half-width ≈ ±0.01), the gate demands strictly
increasing pooled accuracy AND CI separation between adjacent δ_Q. A real
SNR ladder (A/c 0.25 → 0.5 → 1.0) separates far beyond that; overlap would
mean SNR did not move — the exact failure this campaign exists to prevent.

### D-11 — halt guard under misspecification

The halt runaway guard is armed at T_max + 10·D(z_halt) with the CONTROLLER'S
believed parameters (design) — part of the frozen controller, deliberately
not corrected. Under actual < design a patient controller can out-wait its
own guard (e.g. d200_a50_ce300: D_design 3.2 s vs D_actual 8.5 s);
`halt_guard_hits` is logged per trial, summed per point, and printed by the
aggregator — data, not failure. Likewise censoring: `qd.ddm_budget` flags
censor risk from D at the ACTUAL SNR (worst: d50_a*_ce300, D_actual up to
25.7 s against the 60 s budget) — expected, reported, never hidden (§9).

### D-12 — researcher's revision, 2026-09-03 (supersedes the spec's §2 values)

Three directed changes, after inspecting the phase-1 tree on the cluster
(existing cluster data deleted; the campaign restarts from scratch — safe,
since frontier-v1 seeds are trial-identity keyed and nothing re-keys):

1. **`FROZEN_CE` = the full historic grid** {0.03, 0.1, 0.3, 1, 3, 8, 20,
   50, 125, 300, 3000, 30000} (campaign/factors.py verbatim + the two
   ceiling extremes), frozen at every design — Arm B grows to (12 + 2) × 9 =
   **126 points**. Consequences, all expected data: c_e ∈ {0.03…1} are
   discretisation-limited at the 1 s tick (z_halt < substep 0.025 — the
   fast-guess floor); the patient end designed at 50 bp censors heavily
   (d50 ce300: D ≈ 25.7 s; ce3000/30000 designed at 50 bp: halt exit alone
   exceeds the 60 s budget at every actual — near-total censoring, carried
   in `decided_frac`/`halt_frac`, never hidden). Gate §8.2's diagonal now
   covers all 12 Bellman c_e + both statics (all present in the halted 1 %
   reference).
2. **n = 100 runs/treatment** (both arms; was the spec's 1000). Wilson CIs
   widen to ≈ ±0.09 (the spec's ±0.03 target needed ≥600) — gates stay
   valid CI-overlap tests, just lower-powered; volumes drop to 204 k RA +
   12.6 k DDM runs (~30–90 core-h, ~21 GB). run_ids 1…100 are a prefix of
   the halted campaign's 1…1000, so seed pairing with the reference holds.
3. **No phasing** — both arms submit all three actual δ_Q at once (the
   §10.4 phase split and the manifest slices are removed); the across-δ_Q
   u = 0 monotonicity gate is evaluable from the first sync-back.

The §2 block above records the SPEC's pre-registered values; this decision
is the operative deviation record (hard rule: discrepancies to RECON.md).

---

## Pre-flight results (this workstation, 2026-09-02)

| gate | result |
|---|---|
| §4 b\* cross-check (blocking) | **PASS** — re-derived argmin/argmax from the halted campaign's swept trials = 0.004189 / 0.1579 exactly |
| §2 noise invariance (structural) | **PASS** — sensory block byte-identical across actual δ_Q, both templates |
| **R-1** (blocking) | **PASS** — table above; pooled ĉ = 0.0998 vs 0.1, drift ladder 0.0255 / 0.0475 / 0.1009 |
| §6 pairing audit | **PASS** — percept streams bitwise-identical across ra / ddm-bellman / ddm-static at equal (actual, run_id) (22 tick × target keys per run compared); 0/120 noise draws equal across actual δ_Q |
| Seed-universe reproduction at actual = 100 bp | **PASS** — `env_seed(60,100,1)` = 2034332064, `model_seed("ddm-bellman",60,100,1)` = 1352414336: bit-for-bit the halted campaign's run_meta values |
| **Trial-level reproduction** (the sharpest gate-2/4 evidence) | **PASS** — smoke `a100_v0.5_u6` runs 1–20 vs the halted campaign's `U_v0.5_u6` per-trial parquet: 20/20 identical choices, max arrival difference 2.1e-14 s |
| Smoke §10.2 (real script path, 20 reps/cell) | **PASS**, zero failures — RA `a100_v0.5_u6` acc 0.650 [0.43, 0.82]; `a200_v0.5_u6` acc 0.900 (2× signal → higher, SNR moves); DDM `d100_a200_ce20` (off-diagonal freeze) acc 1.000 [0.84, 1.00], median 9.55 s, decided 1.0 — §5.1's "actual > design → merely conservative, accuracy intact"; `A_hat = 0.05` (design) against the actual gap confirmed in `_ddm.csv`; z_halt 0.1861 = the budget table's value; aggregate → analyze end-to-end, all §9 figure types render |
| Gate machinery exercised on real references (n = 20) | **PASS** — §8.4 continuity u ∈ {4, 6, 8}: 1.000/0.650/0.750 vs 0.987/0.696/0.692 (CI overlap); §8.2 diagonal d100_a100_ce20: 0.950 vs 0.958, median 10.20 s vs 9.84 s (CI overlap); blocking exits verified |
| Step-halving u = 35 (§3, blocking; spec's 50 trials/arm, identical seeds) | **PASS** at all three kernels, zero numerical failures, ring state bounded (max\|z\| ≈ 1.84): v = 0.1 Δ = +0.04 [−0.16, +0.24]; v = 0.5 +0.04 [−0.06, +0.14]; v = 1.0 +0.18 [−0.04, +0.38]. `dt_check_report.json`. The v = 1.0 point estimate is large with the CI just touching zero — the D-08 power caveat in action — so a 200-trial confirmation at v ∈ {0.1, 1.0} was run in addition (`dt_check_report_200.json`, row below) |
| Step-halving, 200-trial confirmation (v = 0.1 and v = 1.0) | **PASS**, zero failures, max\|z\| ≈ 1.85 — v = 0.1: Δ = −0.010 [−0.105, +0.090] (the 50-trial +0.04 was noise); **v = 1.0: Δ = +0.100 [−0.005, +0.200]** — formally passes the pre-registered CI-contains-zero criterion at both 50 and 200 trials, but the point estimate is stable at ≈ +0.1 with the lower bound grazing zero. Read: the (v = 1.0, u ≈ 35) corner may carry a genuine dt sensitivity of order +0.1 accuracy that 200 trials cannot resolve. The gate does NOT exclude these cells (no failure occurred); the analysis must carry a dt caveat on v = 1.0 high-u cells, and a 500-trial arm (or running that v-slice at integration_dt 0.05) is the recommended tie-breaker before interpreting that corner — researcher's call at phase 2 |
| Measured wall time | (v = 0.5, u ∈ 4–8): 0.45–0.51 s/run, both arms. **Stiff/high-u corners are ~10× slower**: ≈ 5 s/run at (v = 0.1, u = 35) and ≈ 2.6 s at (v = 1.0, u = 35) — near-chance cells wander before arriving. Arm A therefore lands in the **~300–800 core-h** range depending on how much of the surface is slow (u = 0 rows ~25 s median arrival but 1 tick/s → still fast wall-clock); Arm B ≈ 6–10 core-h. Worst-case task length (1000 runs × ~5 s ≈ 85 min) stays far inside the 6 h SLURM limit |

## Acceptance status (§11)

- parameter block reproduced above with resolved values ✔
- §8 gates: R-1 recorded (PASS); gates 2–4 wired into `aggregate.py`
  (blocking exits) and resolvable after the phase-1 submission; gate 5
  (step-halving) recorded ✔
- every replicate directory self-contained (config + meta + logs + `.done`) ✔
- figures carry design SNR, actual SNR and the halt policy in captions ✔
- no û or u\* in any artifact of this campaign ✔ (grep-clean over
  `scripts/qd_sweep_fixed_noise/` and both manifests)

Out of scope, cited not implemented (§11): SFA/adaptation, u_WTA,
flexible-DDM variants, new motion policies, mixture-prior Bellman solves.
