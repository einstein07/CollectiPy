# Embodied-DDM campaign — orchestration layer

Implements `CAMPAIGN_SPEC.md`. The DDM only; no ring-attractor conditions.

## Quick start

```bash
# 1. See the full plan, predictions and flags. Submits nothing.
./submit_campaign.sh --dry-run

# 2. Pre-flight gates (Section 9) — run locally or on a login/compute node:
python3 -m campaign.preflight stream-determinism          # 9.1
python3 -m campaign.preflight legacy                      # 9.2
python3 -m campaign.precompute_tables --results-root <R>  # tables + horizon check
python3 -m campaign.preflight small-matrix --results-root <R> --reps 20 --keep-raw   # 9.3

# 3. Submit (SLURM): precompute job + dependent array, throttled.
./submit_campaign.sh --results-root <R> --chunk 100 --max-concurrent 100 --reps 1000

# Re-run one condition or one chunk (idempotent; --force to overwrite):
./submit_campaign.sh --only main/q01_a60_ce20        --results-root <R>
./submit_campaign.sh --only quasi_static/q01_a60_ce20:3 --results-root <R>
```

## Files

| file | role |
|---|---|
| `factors.py` | THE single source: factor grids, locked params, campaign seed, Section 5.2 gate |
| `matrix.py` | ordered condition list (main + controls), derived geometry, closed-form predictions |
| `seeds.py` | Section 6 hash derivation (blake2b, criterion absent from the sensory seed) |
| `genconfig.py` | per-condition / per-replicate effective configs; manifests; startup log |
| `run_chunk.py` | one array task: condition x chunk, scratch staging, one CSV out, idempotent |
| `summarise.py` | run archive -> per-replicate summary row (incl. realised crossing log-odds) |
| `precompute_tables.py` | solves all main-matrix Bellman tables into the disk cache, horizon check on |
| `dry_run.py` | the Section 10 report |
| `preflight.py` | the Section 9 gates |
| `../submit_campaign.sh` | SLURM submission (precompute job + dependent throttled array) |
| `../config/campaign_ddm_base.json` | Section 1 locked parameters (never modified in place) |
| `../src/models/bellman_table_cache.py` | atomic npz cache for solved z(t) tables |

## Output layout (Section 8)

```
<results-root>/
├── table_cache/                        bellman_<sha1>.npz + precompute_report.json
├── slurm_logs/
├── main/q01_a60_ce0.03/ ... q05_a150_ce300/
│   ├── config.json                     the exact effective config (seeds null;
│   │                                   per-replicate seeds derived per manifest)
│   ├── manifest.json                   Section 8 fields, predictions, config hash
│   ├── chunks/chunk_0000.csv ...       ONE file per chunk on shared storage
│   └── raw/ ...                        only with --keep-raw (Section 9.3)
└── controls/quasi_static/q01_a60_ce*/  same structure
```

Chunk CSV columns: see `summarise.FIELDS`. `censored=1` rows never arrived inside
`time_limit` and must be excluded from accuracy, not scored. `a_realised =
2*A_hat*|x|/c^2` at the first committed tick is the Section 1.3 contamination
measurement (read at tick resolution, so it contains the threshold overshoot — that
overshoot is the thing being measured).

## The static control is blocked (Section 5.2)

`factors.STATIC_CONTROL_Z` is `None`: no static-boundary definition exists in the
code or configs (searched — `z_manual` only has a generic code default of 1.0). The
submit script refuses that arm and says so; main + quasi-static run without it. The
dry run prints `z_bellman(0)` per grid point (from the precompute report) — the
proposal on the table is `z_manual := z_bellman(0)` per `c_e`, which isolates the
collapse. Choose from that evidence, then set e.g.

```python
STATIC_CONTROL_Z = [
    {"z_manual": 0.054670, "from_c_e": 1},
    ...
]
```

## Decisions and deviations (report-first, per the spec's principle)

- **`z_min = 1e-4`** in the base config, not the historical 0.05. Section 1.3
  requires the chance-level end measured; a 0.05 floor would replace the policy's
  z ~ 0.003 at `c_e <= 0.1` with the floor itself. Not spec-locked; changeable in
  the base config.
- **`angular_velocity = 120` deg/s**, not the template's 10. The minimum turn
  radius is v/omega; at the locked v = 0.05 the template plant turns a 0.29 m
  circle and ORBITS the target outside the 0.05 m termination radius forever
  (caught by pre-flight 9.3: up to 19/20 censored at 150 degrees, stable closed
  orbit measured). Post-commitment motion only: rt and choice are bit-identical
  across the change (verified 20/20).
- **Arena `side = 2`.** The square arena spans [-side/2, side/2]; the default 1
  cannot hold the 150-degree placements (and re-samples an overlapping target to a
  RANDOM position rather than failing).
- **Per-tick logging stays on in scratch.** Section 7.4's aim (filesystem load) is
  met by shipping one CSV per chunk; the per-tick rows (<= 600/run, node-local,
  deleted with scratch) are what the crossing-row measurement needs.
- **In-process replicates.** Interpreter startup (~1.2 s) exceeds a cached
  replicate's sim time; the runner imports the simulator once per task. Verified
  bit-identical to one-process-per-replicate (`--subprocess`).
- **Model-side additions** (outside `campaign/`): the Bellman table disk cache
  (`bellman.table_cache_dir`, default null = historical behaviour), and a FIX to
  the ensemble |A| deduction, which under `sensory_stream: shared` read one noisy
  percept instead of the declared strengths — silently breaking
  `drift_knowledge: known_magnitude` (A_hat was N(A, ~0.157^2) per trial). Legacy
  mode is unaffected (pre-flight 9.2 passes).
- **The model's horizon is ~1.6% longer than the manifest's `T_max`**: the
  detection layer measures 3-D range (target centre sits at z = 0.1), so onset
  r0 = 0.440 not 0.433 at 60 degrees. Systematic across every condition and arm;
  manifests record the Section 3B planar values.
- **8 cells carry `CENSOR-RISK`** (predicted DT > 0.5 T_max): the Section 1.1
  calibration bounded `DT/T_max` at the 60-degree baseline only; at 120-150 degrees
  `T_max` shrinks faster than DT. Flagged in the dry run and manifests, as
  Section 10 requires; interpret those cells with the terminal collapse in mind.
