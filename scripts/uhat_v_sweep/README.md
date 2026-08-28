# (û, v) factorial — criticality-normalized coupling × kernel shape

Implements `uhat-v-factorial-experiment.md`. **Read
[`RECON.md`](RECON.md) first** — it records six places where the spec document
and the repo disagree, three of them material (the plant's `angular_velocity`,
the ring's `sigma`, and what "commitment time" means for an embodied ring
attractor).

## What this measures

Is the ring attractor's speed–accuracy behaviour organised by *relative*
coupling û = u / u\*(v) alone, or does kernel shape v still matter once û is
held fixed? 8 û levels × 10 v levels, fully crossed, 400 trials per cell on one
shared seed list. The answer is a deviance decomposition, not a p-value.

```
u_hat = [0.50, 0.65, 0.80, 0.90, 1.00, 1.10, 1.25, 1.50]
v     = [0.1 … 1.0 step 0.1]
u     = u_hat * u_star(v),   u_star(v) = 1 / (lambda_max(W(v)) * sech^2(beta))
```

## Files

| file | role |
|---|---|
| `factors.py` | THE single source: factor grids, locked parameters, seeds, guardrail thresholds |
| `config_patch.py` | per-cell / per-trial effective configs, provenance, post-patch assertions |
| `generate_manifest.py` | u\*(v) from the simulator's own kernel builder; anchor gate; `manifest.json`, `u_star_table.csv` |
| `run_cell.py` | one array task: (cell × chunk) → one row file; scratch staging; idempotent |
| `dt_check.py` | Section 11 step-halving check on the three stiffest cells |
| `aggregate.py` | glob `raw/` → `trials.parquet` + `cells.csv`, with validation |
| `analyze_collapse.py` | deviance ladder, verdict, five plot families, optional paired contrasts |
| `../../slurm/uhat_v_sweep.sbatch` | bwUniCluster job array (submission mode + execution mode) |
| `../../config/mean_field_2_targets_no_viz.json` | the RA-arm template, never modified in place |

## Output layout

```
<results-root>/
├── manifest.json               80 cells: {cell_id, v, u_hat, u_star, u, n_trials, base_seed}
├── u_star_table.csv            fine grid v = 0.05 … 1.00 for overlays
├── raw/cell00_chunk000.parquet ONE file per array task — never appended to
├── trials.parquet              every trial (aggregate.py)
├── cells.csv                   per-cell summary with Wilson intervals
├── aggregate_report.json
├── dt_check/                   step-halving arms + step_halving_report.json
├── analysis/                   plots (PNG + PDF), collapse_summary.json, m3_residuals.csv
└── slurm_logs/
```

## Commands, in order

Run 1–3 and 5–6 locally / on a login node. **Only step 4 submits anything, and
you run it yourself.**

Local paths below; on the cluster set `PROJECT_DIR` and `RESULTS_ROOT` to the
values at the top of `slurm/uhat_v_sweep.sbatch` (or export your own).

```bash
cd /home/sindiso/Documents/PhD/ring-attractor/CollectiPy
PY=.venv/bin/python
R=results/uhat_v_sweep
```

### 1. Generate the manifest (local)

```bash
$PY scripts/uhat_v_sweep/generate_manifest.py --results-root $R
```

Halts if the u\*(0.5) = 6.157 anchor moves by more than 1 %. Prints u\*(v) on the
design grid and the maximum resolved u. Refuses to overwrite an existing
manifest without `--force` (re-keying a sweep mid-flight is not a thing you want
to do by accident).

### 2. Smoke test (local, ~15 s)

```bash
$PY scripts/uhat_v_sweep/run_cell.py --smoke --results-root $R
$PY scripts/uhat_v_sweep/aggregate.py  --results-root $R/smoke --manifest $R/manifest.json --smoke
$PY scripts/uhat_v_sweep/analyze_collapse.py --results-root $R/smoke
```

Two cells, (v=0.5, û=0.90) and (v=0.5, û=1.10), 12 trials each. This validates
the pipeline only — statistics are meaningless at n = 12.

Both smoke cells sit at one v, so the deviance ladder is **not identified** on
smoke output and `analyze_collapse.py` says so and draws the plots it can. To
exercise the ladder itself, run a 2 × 2 pilot instead:

```bash
T=$(mktemp -d); cp $R/manifest.json $R/u_star_table.csv $T/
for c in 35 37 75 77; do
  $PY scripts/uhat_v_sweep/run_cell.py --cell-id $c --trials-per-chunk 40 --results-root $T
done
$PY scripts/uhat_v_sweep/aggregate.py --results-root $T --allow-incomplete
$PY scripts/uhat_v_sweep/analyze_collapse.py --results-root $T --paired
```

### 3. Step-halving check (local, ~4 min)

```bash
$PY scripts/uhat_v_sweep/dt_check.py --results-root $R          # --dry-run to never touch the manifest
```

Section 11: (v=0.1, û=1.50), (v=0.1, û=1.25) and (v=0.2, û=1.50) at
`integration_dt` 0.05 vs 0.1, 50 trials each on identical seeds. Passes a cell
when it has zero numerical failures and the paired-bootstrap 95 % interval on
the accuracy difference contains zero. A failing cell is marked `excluded` in
`manifest.json`; the design is then unbalanced and both `aggregate.py` and
`analyze_collapse.py` say so.

### 4. Submit (cluster — **you run this, not the agent**)

```bash
# a. plan only, nothing submitted (this is also the default with no flags)
DRY_RUN=1 bash slurm/uhat_v_sweep.sbatch

# b. SLURM's own dry run
TEST_ONLY=1 bash slurm/uhat_v_sweep.sbatch

# c. for real
SUBMIT=1 bash slurm/uhat_v_sweep.sbatch
```

80 array tasks (one cell each), throttled to 40 concurrent, 30 min wall limit
per task — over 2× the worst measured chunk (see RECON D-09). Re-running is
free: a task whose output file is already complete exits 0 immediately, so a
partial array is fixed by resubmitting the same command.

Single cell or chunk:

```bash
ONLY=7   SUBMIT=1 bash slurm/uhat_v_sweep.sbatch     # cell 7
ONLY=7:2 SUBMIT=1 bash slurm/uhat_v_sweep.sbatch     # cell 7, chunk 2
```

### 5. Aggregate

```bash
$PY scripts/uhat_v_sweep/aggregate.py --results-root $R
```

Validates that all 80 cells are present with the expected trial counts, that no
(cell, seed) pair is duplicated, that every cell shares one seed list (the
paired design depends on it), and that a cell carries one `config_hash`.
Problems block the write unless you pass `--allow-incomplete`. Emits
`trials.parquet` and `cells.csv`, and prints the Section 12 expectation check
against the prior matched-critical result (accuracy 0.765, DT 11 ticks at
v = 0.5, û = 1.0).

### 6. Analyse

```bash
$PY scripts/uhat_v_sweep/analyze_collapse.py --results-root $R --paired
```

Prints the deviance ladder M0 → M4, the three headline shares, the largest M3
cell residual in probability points, and the pre-registered verdict; then the
same decomposition on log commitment time with its censoring caveat; then writes
the five plot families as PNG **and** PDF.

## Reading the result

| condition | verdict |
|---|---|
| v main effect + interaction together < 10 % of between-cell deviance | **collapse** onto û |
| interaction alone < 10 %, but v's main effect substantial | **additive**: f(û) + g(v) |
| otherwise | **interaction**: û is not sufficient; the M3 residual heatmap says where |

Accuracy is always reported both ways. `acc_decided` conditions on the agent
having arrived, which is a selection effect; `acc_all` scores an undecided trial
as an error, which is the 0–1 terminal loss the design specifies. The gap
between them is informative, so `decided_frac` gets its own heatmap and is never
folded into DT.

## Out of scope

Not implemented, by Section 13: the u_WTA continuation, readout-threshold
sweeps, any DDM-side change, and any change to targets, arena, motor parameters
or sensory-stream statistics.
