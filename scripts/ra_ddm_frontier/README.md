# RA slices + envelope vs the DDM frontier (1 %, Δθ = 60°)

Implements `ra-ddm-frontier-slice-envelope-experiment.md`. **Read
[`RECON.md`](RECON.md) first** — especially D-01 (both models run on ONE 1 s
tick clock: the RA in the archived `ra_ddm_frontier_sweep` environment with
`time_limit = 1000`, the DDM on the same clock with `time_limit = 60` s and
`snapshots_per_second = 1`; cross-model pairing is bitwise-identical percept
realizations per run_id — the spec's full §3), D-09 (`white_rate = √2·ΔQ ≈
0.07071068`, so the evidence-channel noise c = √2·η = 2×ΔQ = 0.1 exactly —
deliberately overriding the DDM campaign's 0.035), D-10 (the
archived-frontier and factorial-anchor gates are informational under the new
calibration) and D-11 (the DDM's coarser evidence step at the 1 s tick).

## What this measures

The RA family placed on the DDM's speed–accuracy plane at δ_Q = 1 %, Δθ = 60°:
four û-swept slices (Set R, 52 cells) + an absolute-u sweep (Set U, 48 cells,
incl. the u = 0 no-coupling control), and the cross-validated Pareto envelope,
against a seed-paired rerun of the DDM's c_e frontier (10 points). Both
campaigns share one seed universe (`frontier-v1`, `seeding.py`): for a given
run_id the RA and DDM trials see the identical exogenous noise realization, so
per-trial (McNemar) comparisons are licensed, not just means.

## Files

| file | role |
|---|---|
| `seeding.py` | §3's scheme, verbatim — the single source of truth for BOTH campaigns |
| `frontier.py` | grids, template derivation (§4), the two model patchers, seed routing, arrival scoring |
| `generate_manifest.py` | u\*(v) + anchor gate; `manifest.csv` (RA, 100 cells) and `ddm_manifest.csv` (10 points); derives both templates into `config/` |
| `run_batch.py` | one array task: configs + metadata persisted per replicate, `.done` sentinels, per-task failure files |
| `audit_seeding.py` | §3's blocking audit battery → `AUDIT.md` |
| `submit-ra-ddm-frontier-slices-bwunicluster.sh` | SLURM submission, `CAMPAIGN=ra\|ddm`, auto-chunked arrays, DDM table precompute |
| `aggregate.py` | replicate tree → `trials.parquet` + `cells.csv` (and DDM twins); completeness; the DDM regression gate |
| `analyze_overlay.py` | §10 envelope, §11 overlay figure + regret + McNemar |

## Commands, in order (§12)

Run 1–4 and 6–7 locally / on a login node. **Only step 5 submits, and you run
it yourself.**

```bash
cd /home/sindiso/Documents/PhD/ring-attractor/CollectiPy
PY=.venv/bin/python
R=results/ra_ddm_frontier
```

### 1. Recon — done; read `RECON.md`.

### 2. Seed audit (blocking)

```bash
$PY scripts/ra_ddm_frontier/audit_seeding.py
```

Writes `AUDIT.md`; non-zero exit blocks everything downstream.

### 3. Manifests + templates (anchor gate inside)

```bash
$PY scripts/ra_ddm_frontier/generate_manifest.py --campaign both --out-dir $R
```

### 4. Smoke test (§8) — the real script path, locally, no SLURM

2 RA cells (v=0.5, û ∈ {0.65, 1.00} = manifest rows 43 and 48) and 2 DDM
points (c_e 0.03 and 300 = rows 0 and 9), 20 replicates each. With
`RUNS_PER_TASK=20` the flat task index equals the manifest row index:

```bash
S=$PWD/results/ra_ddm_frontier/smoke
COMMON="PROJECT_DIR=$PWD RUNS_PER_CELL=20 RUNS_PER_TASK=20"
for ROW in 43 48; do
  env $COMMON CAMPAIGN=ra LOGS_DIR=$S/ra BASE_PATH_ROOT=$S/ra \
      MANIFEST=$S/ra/manifest.csv TASK_OFFSET=0 SLURM_ARRAY_TASK_ID=$ROW \
      bash scripts/ra_ddm_frontier/submit-ra-ddm-frontier-slices-bwunicluster.sh
done
for ROW in 0 9; do
  env $COMMON CAMPAIGN=ddm LOGS_DIR=$S/ddm BASE_PATH_ROOT=$S/ddm \
      MANIFEST=$S/ddm/ddm_manifest.csv TASK_OFFSET=0 SLURM_ARRAY_TASK_ID=$ROW \
      bash scripts/ra_ddm_frontier/submit-ra-ddm-frontier-slices-bwunicluster.sh
done
```

(Execution mode reads the manifest, so generate it into the smoke dirs first:
`$PY scripts/ra_ddm_frontier/generate_manifest.py --campaign ra --n-runs 20 --out $S/ra/manifest.csv`
and likewise `--campaign ddm --out $S/ddm/ddm_manifest.csv`.)

Then aggregation + overlay end-to-end on the smoke output:

```bash
$PY scripts/ra_ddm_frontier/aggregate.py --campaign ra  --base-root $S/ra
$PY scripts/ra_ddm_frontier/aggregate.py --campaign ddm --base-root $S/ddm \
    --previous ../seoul-data/beta-1/ddm-characterisation/tidy_trials.parquet
$PY scripts/ra_ddm_frontier/analyze_overlay.py --ra-root $S/ra --ddm-root $S/ddm
```

Measured on this workstation (see RECON pre-flight table): RA 0.44–0.47 s/run,
DDM 0.55–0.71 s/run (cache warm) — at `RUNS_PER_TASK=100` a task is ≲ 2 min
against the 24 h limit, and the whole pair of campaigns ≈ 15–20 core-hours.
In-process and `--subprocess` (`main.py -c`) runs verified bit-identical.
Smoke anchor cell at the final calibration (η = 0.07071068, c = 2ΔQ = 0.1):
acc 0.700 [0.48, 0.86], median commit 11 ticks (the factorial's 0.765 was at
η = 0.035 — RECON D-10).

### 5. Submit (cluster — **you run this, not the agent**)

```bash
# plan only
DRY_RUN=1 bash scripts/ra_ddm_frontier/submit-ra-ddm-frontier-slices-bwunicluster.sh
DRY_RUN=1 CAMPAIGN=ddm bash scripts/ra_ddm_frontier/submit-ra-ddm-frontier-slices-bwunicluster.sh

# for real (one task = one cell: RA 100 tasks, DDM 10 tasks, ~10-17 min each)
bash scripts/ra_ddm_frontier/submit-ra-ddm-frontier-slices-bwunicluster.sh
CAMPAIGN=ddm bash scripts/ra_ddm_frontier/submit-ra-ddm-frontier-slices-bwunicluster.sh
```

If sbatch says "Resource temporarily unavailable", the array is hitting a
per-user submit limit — check with
`sacctmgr show assoc user=$USER format=user,account,maxsubmitjobs,maxjobs`
and shrink the array chunks with e.g. `MAX_ARRAY=50` (the script auto-chunks;
each chunk is its own sbatch). The default geometry (100 elements) was chosen
to stay far below typical limits.

Re-running either command is free: replicates with `.done` are skipped, so a
partial array is fixed by resubmitting. The DDM submission first populates the
Bellman table cache (replicate 1 of each point, on the login node; set
`PRECOMPUTE=0` to skip). Cost knobs, in order (§2): `RUNS_PER_CELL=600` first;
then drop û ∈ {1.10, 1.25} and u ∈ {11, 13} from the grids in `frontier.py`
(bump the scheme? no — seeds are per-trial, dropping cells never re-keys);
never thin v.

### 6. Aggregate + gates

```bash
$PY scripts/ra_ddm_frontier/aggregate.py --campaign ra  --base-root <RA root>
$PY scripts/ra_ddm_frontier/aggregate.py --campaign ddm --base-root <DDM root> \
    --previous seoul-data/beta-1/ddm-characterisation/tidy_trials.parquet
```

Gates printed (§12.6): completeness (missing replicates listed → resubmit)
and the four u = 0 pure-replicate cells (mutual CI overlap; v is inert at
u = 0) are blocking. The archived-frontier comparison and the
factorial-anchor cell are **informational** — both references were measured
at white_rate 0.035 and this campaign is calibrated at 0.1 (RECON D-10) — so
systematic shifts there are expected, not drift. A blocking gate failing:
halt and diagnose before the overlay.

### 7. Envelope + overlay + regret + ceiling verification

```bash
$PY scripts/ra_ddm_frontier/analyze_overlay.py \
    --ra-root <RA root> --ddm-root <DDM root>
```

Outputs: `overlay_main` (§11 main figure — per-v absolute-u panels vs the DDM,
labels thinned in clusters), `tuning_curves` (accuracy vs u, log-x, per v),
`overlay_slices` (Set-R û-slices + cross-validated envelope, supplement),
`regret.json`, `mcnemar.csv`, and `ceiling_check.json` — the §11 requirement
that the "RA beats the DDM family" claim clears the DDM's infinite-patience
asymptote Φ((A/c)·√(r₀/v)) = 0.9294 with CI separation.

### 8. Wave 2 — Set U-v2 top-up + DDM ceiling points (§2, §11, §12.7)

Wave 2 is **derived from wave-1 data**, so it is generated locally where
`cells.csv` lives, then shipped to the cluster:

```bash
# a. derive the top-up grids from the measured cliff windows (local)
$PY scripts/ra_ddm_frontier/generate_manifest.py \
    --topup-from <RA root>/cells.csv --out-dir $R
#    -> $R/manifest_topup.csv, manifest_full.csv,
#       ddm_manifest_topup.csv, ddm_manifest_full.csv

# b. BLOCKING: step-halving check at the three stiffest new cells (local, ~10 min)
$PY scripts/ra_ddm_frontier/dt_check.py --manifest $R/manifest_topup.csv

# c. copy the four manifests into the campaigns' cluster directories
scp $R/manifest_topup.csv $R/manifest_full.csv  <cluster>:<RA LOGS_DIR>/
scp $R/ddm_manifest_*.csv                        <cluster>:<DDM LOGS_DIR>/

# d. submit the top-ups (cluster — you run this). TOPUP=1 skips manifest
#    regeneration and uses the shipped file; results land in the SAME tree,
#    and wave-1 replicates are untouched (.done idempotency).
TOPUP=1 MANIFEST=<RA LOGS_DIR>/manifest_topup.csv \
    bash scripts/ra_ddm_frontier/submit-ra-ddm-frontier-slices-bwunicluster.sh
TOPUP=1 CAMPAIGN=ddm MANIFEST=<DDM LOGS_DIR>/ddm_manifest_topup.csv \
    bash scripts/ra_ddm_frontier/submit-ra-ddm-frontier-slices-bwunicluster.sh

# e. after syncing back: re-aggregate (completeness is now judged against
#    manifest_full.csv automatically when it sits in the base root) + re-analyze
$PY scripts/ra_ddm_frontier/aggregate.py --campaign ra  --base-root <RA root>
$PY scripts/ra_ddm_frontier/aggregate.py --campaign ddm --base-root <DDM root>
$PY scripts/ra_ddm_frontier/analyze_overlay.py --ra-root <RA root> --ddm-root <DDM root>
```

Because frontier-v1 seeds are trial-identity keyed, wave 2 is a pure top-up:
wave-1 cells stay valid and paired, and the new cells are born paired with
everything else at every run_id.

## Traceability (§7)

Every replicate directory holds the exact `config.json` it executed,
`run_meta.json` (identity, both seeds, scheme `frontier-v1`, git SHA), the
simulator's native `config_folder_0/run_1.zip`, and `.done` on success. Any
replicate re-executes with `python src/main.py -c <dir>/config.json`. Failures
live in per-task files under `failures/`. Expect ~10 GB for the RA tree at
100 000 replicates (~100 KB/run archive).

## Out of scope (§13)

Other δ_Q panels / separations (`--diff` is refused ≠ 0.01 by design — a new
panel is a deliberate `frontier.py` edit plus a new manifest), u_WTA, any
change to the DDM's grid or calibration, SFA/adaptation (`g_adapt = 0`
asserted), any change to arena, speed, timeout, or detector.
