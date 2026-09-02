# QD sweep at fixed noise — RA (u, v) surface + frozen-DDM misspecification

Implements `qd-sweep-fixed-noise-experiment.md`. **Read [`RECON.md`](RECON.md)
first** — especially R-1 (the noise-convention measurement every solve depends
on), D-01 (why `WHITE_RATE` is a pinned literal and what the assertions
protect), D-02 (what "the halted campaign" is and why its 1 % trees are the
gate references), and D-03 (the frozen static b\* derivation and its
cross-check).

## What this measures

Two questions, one campaign, one seed universe (`frontier-v1`, inherited):

- **Arm A (RA surface):** the RA over the full kernel range
  (`U_GRID` = 0, 2…35 × `V_GRID` = 0.1…1.0 — **absolute u only**, no û, no
  u\*(v) anywhere) as *actual* difficulty δ_Q ∈ {50, 100, 200} bp varies at
  **fixed noise** `white_rate = 0.07071068` (evidence-channel c = 0.1). With
  the old c = 2ΔQ coupling dead, SNR now genuinely scales with δ_Q
  (A/c = 0.25 / 0.5 / 1.0, k = 5 / 10 / 20).
- **Arm B (supervisor's test):** DDM controllers **frozen at a design δ_Q**
  — Bellman c_e ∈ {3, 20, 300} solved at (A_design, c = 0.1) via
  `A_expected`, plus static b\*_cost / b\*_RR — evaluated at every *actual*
  δ_Q: a 3 × 3 design × actual matrix per controller whose diagonal is the
  clairvoyant reference. Controllers are never re-tuned; freezing is the
  experiment. Predictions pre-registered in spec §5.

## Files

| file | role |
|---|---|
| `qd.py` | §2 parameter block (single source of truth), templates, patchers **with the blocking §2 assertions**, manifests, seed routing, the §4 controller freeze |
| `generate_manifest.py` | b\* cross-check (blocking) → manifests + templates + `frozen_controllers.json` |
| `run_batch.py` | one array task: configs + metadata per replicate, `.done` sentinels, per-task failure files |
| `r1_noise_convention.py` | **R-1 (blocking)**: measure c from the sim's own logs at every δ_Q; §6 pairing audit |
| `dt_check.py` | **§3 step-halving (blocking)**: u = 35 at v ∈ {0.1, 0.5, 1.0} |
| `submit-qd-sweep-fixed-noise-bwunicluster.sh` | SLURM submission, `CAMPAIGN=ra\|ddm`, auto-chunked arrays, DDM table precompute |
| `aggregate.py` | replicate tree → trials.parquet + cells.csv; completeness; **gates §8.2–8.4** |
| `analyze.py` | §9: heatmaps, tuning curves, peak track, design×actual matrices, regret, misspec curves + analytic overlay, censoring, SAT planes |

## Commands, in order (§10)

Run 1–4 locally / on a login node. **Only step 5 submits, and you run it
yourself.**

```bash
cd /home/sindiso/Documents/PhD/ring-attractor/CollectiPy
PY=.venv/bin/python
R=results/qd_sweep_fixed_noise
```

### 1. Recon — done; read `RECON.md`.

### 2. Manifests + templates + the frozen controllers (blocking checks inside)

```bash
$PY scripts/qd_sweep_fixed_noise/generate_manifest.py
```

Needs `../seoul-data/beta-1/ra_ddm_frontier_ddm_halt/…_1.0/ddm_trials.parquet`
(the §4 b\* re-derivation + cross-check against the sweep-derived
0.004189 / 0.1579 — a mismatch HALTS). Writes `ra_manifest.csv` (2040 cells),
the phased slices `ra_manifest_actual100.csv` / `ra_manifest_rest.csv`,
`ddm_manifest.csv` (45 points), `frozen_controllers.json`, and both templates.

### 3. R-1 — the noise-convention gate (BLOCKING, §2)

```bash
$PY scripts/qd_sweep_fixed_noise/r1_noise_convention.py        # ~10 min
```

Measures the accumulator's per-unit-time increment variance from `_ddm.csv`
at each actual δ_Q and asserts ĉ = 0.1 within ±5 %; asserts the drift ladder
0.025 / 0.05 / 0.1 (SNR moves, noise doesn't); §6 pairing audit (percept
streams identical across models, different across δ_Q). Writes
`$R/r1_report.json`. **If it fails: halt and report — never rescale.**

### 4. Smoke (§10.2) — the real script path, locally, no SLURM

2 RA cells (`a100_v0.5_u6` = row 961, `a200_v0.5_u6` = row 1641) and 1 DDM
point (`d100_a200_ce20`, an off-diagonal freeze = row 26), 20 replicates
each, end-to-end through aggregation and one figure:

```bash
S=$PWD/results/qd_sweep_fixed_noise/smoke
COMMON="PROJECT_DIR=$PWD RUNS_PER_CELL=20 RUNS_PER_TASK=20 BASE_PATH_ROOT=$S LOGS_DIR=$S"
for ROW in 961 1641; do
  env $COMMON CAMPAIGN=ra MANIFEST=$R/ra_manifest.csv TASK_OFFSET=0 SLURM_ARRAY_TASK_ID=$ROW \
      bash scripts/qd_sweep_fixed_noise/submit-qd-sweep-fixed-noise-bwunicluster.sh
done
env $COMMON CAMPAIGN=ddm MANIFEST=$R/ddm_manifest.csv TASK_OFFSET=0 SLURM_ARRAY_TASK_ID=26 \
    bash scripts/qd_sweep_fixed_noise/submit-qd-sweep-fixed-noise-bwunicluster.sh

$PY scripts/qd_sweep_fixed_noise/aggregate.py --arm ra  --base-root $S --manifest $R/ra_manifest.csv
$PY scripts/qd_sweep_fixed_noise/aggregate.py --arm ddm --base-root $S --manifest $R/ddm_manifest.csv
$PY scripts/qd_sweep_fixed_noise/analyze.py --base-root $S
```

(Smoke completeness reports the untouched cells as missing — expected; the
gates that need full data print their per-cell numbers only.)

### 5. Step-halving at u = 35 (BLOCKING, §3)

```bash
$PY scripts/qd_sweep_fixed_noise/dt_check.py               # 50 trials/arm (spec)
$PY scripts/qd_sweep_fixed_noise/dt_check.py --trials 200  # sharper (recommended)
```

u = 35 at v ∈ {0.1, 0.5, 1.0}, δ_Q = 100 bp, dt 0.1 vs 0.05, identical
seeds → `$R/dt_check_report.json`. A failing v EXCLUDES its high-u cells
(report which); never submit as-is.

### 6. Submit (cluster — **you run this, not the agent**)

Ship the manifests first (they are never generated on the cluster — the §4
freeze needs seoul-data):

```bash
scp $R/{ra_manifest.csv,ra_manifest_actual100.csv,ra_manifest_rest.csv,ddm_manifest.csv,frozen_controllers.json} \
    <cluster>:<LOGS_DIR>/
```

§10.4 order — Arm B and the actual = 100 bp slice of Arm A first (gates
2–4 resolvable early), remaining Arm A after those gates pass:

```bash
# plan only
DRY_RUN=1 CAMPAIGN=ddm bash scripts/qd_sweep_fixed_noise/submit-qd-sweep-fixed-noise-bwunicluster.sh
DRY_RUN=1 MANIFEST=<LOGS_DIR>/ra_manifest_actual100.csv \
    bash scripts/qd_sweep_fixed_noise/submit-qd-sweep-fixed-noise-bwunicluster.sh

# phase 1: Arm B (45 tasks) + Arm A actual=100 (680 tasks)
CAMPAIGN=ddm bash scripts/qd_sweep_fixed_noise/submit-qd-sweep-fixed-noise-bwunicluster.sh
MANIFEST=<LOGS_DIR>/ra_manifest_actual100.csv \
    bash scripts/qd_sweep_fixed_noise/submit-qd-sweep-fixed-noise-bwunicluster.sh

# phase 2 (after gates 2-4 pass): the remaining 1360 RA cells
MANIFEST=<LOGS_DIR>/ra_manifest_rest.csv \
    bash scripts/qd_sweep_fixed_noise/submit-qd-sweep-fixed-noise-bwunicluster.sh
```

One task = one cell (`RUNS_PER_TASK=1000`, ~10–17 min at ~0.5–0.7 s/run).
Arrays auto-chunk at `MAX_ARRAY=1000` (shrink it if sbatch reports "Resource
temporarily unavailable"). Re-running a command is free — `.done` replicates
are skipped, so a partial array is fixed by resubmitting. The DDM submission
first populates the Bellman table cache (replicate 1 of each point on the
login node; 9 distinct tables — cache keys include A_design; `PRECOMPUTE=0`
skips). Volume at defaults: **2.04 M RA + 45 k DDM runs ≈ 300–800
core-hours** (0.45 s/run at mid-surface cells, but ~2.5–5 s/run in the
stiff high-u corners at extreme v — see the RECON wall-time row; worst-case
task length ≈ 85 min, far inside the 6 h limit). Cost knobs, in order (§3):
`--n-runs-ra 600` at generation
(Wilson CIs ≈ ±0.03); then `--ra-diffs 100` with the other two δ_Q on a
reduced v set — **never thin `U_GRID`** (standing policy).

### 7. Aggregate + gates (§8 — all blocking)

```bash
$PY scripts/qd_sweep_fixed_noise/aggregate.py --arm ddm --base-root <root> \
    --previous-halted ../seoul-data/beta-1/ra_ddm_frontier_ddm_halt/ra_ddm_frontier_ddm_halt_1.0/ddm_trials.parquet
$PY scripts/qd_sweep_fixed_noise/aggregate.py --arm ra  --base-root <root> \
    --previous-ra ../seoul-data/beta-1/ra_ddm_frontier_slices/ra_ddm_frontier_slices_1.0/cells.csv
```

Gates printed and written as JSON next to the tables:

- **§8.2** `diagonal_regression_gate.json` — design = actual = 100 bp vs the
  halted campaign's D_ce{3,20,300} + S_b0.004189 + S_b0.1579 (same
  calibration, same env seeds): CI overlap on acc_all and median arrival.
- **§8.3** `u0_gate.json` — the 10 u = 0 cells per δ_Q all-pairs CI overlap
  (v inert at u = 0), and pooled u = 0 accuracy strictly increasing with
  CI separation across δ_Q (the cheap proof SNR moved).
- **§8.4** `ra_continuity_gate.json` — (v = 0.5, u ∈ {4, 6, 8}) at
  actual = 100 bp vs the halted campaign's absolute-sweep twins.

Missing replicates (both arms) land in `missing_replicates_{ra,ddm}.csv` —
resubmitting the campaign script fills exactly these.

### 8. Analysis (§9)

```bash
$PY scripts/qd_sweep_fixed_noise/analyze.py --base-root <root>
```

Everything lands in `<root>/analysis/` — RA heatmaps (acc_all, acc_decided,
median arrival, decided_frac) per δ_Q, tuning curves (linear u), the
empirical peak track u_peak(v; δ_Q) with bootstrap CIs, the 3 × 3
design × actual matrices, regret vs the same-actual diagonal (paired per
run_id), misspecification curves with the design point starred and the
analytic 1/(1+e^{−k·b}) overlay on the static panels, the censoring panel
(expected data at A/c = 0.25 under frozen bold thresholds, not failure), and
per-δ_Q SAT planes with design/actual/halt-policy in every caption.

## Traceability (§7)

Every replicate directory holds the exact `config.json` it executed (every
§2 assertion passed on it before writing), `run_meta.json` (identity,
actual_bp, design_bp, resolved A/c/k, both seeds, scheme `frontier-v1`, git
SHA), the simulator's native `config_folder_0/run_1.zip`, and `.done` on
success. Any replicate re-executes with `python src/main.py -c
<dir>/config.json`. Failures live in per-task files under `failures/`.
Expect **~200 GB** for the full RA tree at 2.04 M replicates
(~100 KB/run archive, the frontier's measured rate) — the §3 cost knobs
shrink this proportionally; check the scratch quota before phase 2.

## Out of scope (§11)

SFA/adaptation (`g_adapt = 0` asserted), u_WTA, flexible-DDM variants, any
new motion policy, mixture-prior (drift-marginalizing) Bellman solves — a
known extension, not implemented. Adding a quality difference later =
appending one integer to `DIFF_BP` in `qd.py`.
