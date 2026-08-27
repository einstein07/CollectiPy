#!/bin/bash
# =============================================================================
# Embodied-DDM campaign — SLURM submission (CAMPAIGN_SPEC.md Section 7)
#
# Two jobs:
#   1. PRECOMPUTE — one task solving all main-matrix Bellman tables into the
#      shared disk cache (Section 7.3), with the 1.5x horizon check ON.
#   2. ARRAY     — one task per (condition-point x chunk), each running
#      `python3 -m campaign.run_chunk --index $SLURM_ARRAY_TASK_ID`, gated on
#      the precompute with --dependency=afterok. Concurrency is capped with the
#      array throttle (%N), never manual batching.
#
# The factors, the matrix and the task order live in campaign/factors.py and
# campaign/matrix.py — this script re-derives everything from them and hard-codes
# no counts.
#
# Usage:
#   ./submit_campaign.sh [--dry-run] [--chunk 100] [--max-concurrent 100]
#                        [--reps 1000] [--only <condition>[:<chunk>]] [--force]
#                        [--results-root DIR] [--keep-raw]
#
#   --dry-run         print the Section 10 report and the submission plan; submit
#                     nothing.
#   --chunk N         replicates per array task (default 100)
#   --max-concurrent N  array throttle (default 100; set from the partition limit)
#   --reps N          replicates per condition-point (default 1000)
#   --only C[:K]      submit only condition C (all its chunks, or just chunk K)
#   --force           re-run chunks whose output is already complete
#   --keep-raw        copy raw run archives to shared storage (Section 9.3 only;
#                     never for the bulk campaign)
#
# Environment overrides:
#   PROJECT_DIR       repo checkout on the cluster
#   RESULTS_ROOT      results root (Section 8 layout appears under it)
#   PARTITION         SLURM partition            (default: cpu)
#   TIME_LIMIT_TASK   per-array-task wall time   (default: 01:00:00)
#   TIME_LIMIT_PRE    precompute wall time       (default: 00:30:00)
#   MEM_PER_TASK      memory per task            (default: 4G)
#
# The static control (Section 5.2): while campaign/factors.py:STATIC_CONTROL_Z is
# unset the static arm DOES NOT EXIST in the matrix — this script will say so and
# submit the main matrix and the quasi-static control without it. No value is
# invented here.
# =============================================================================

#SBATCH --job-name=ddm_campaign

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults and argument parsing
# ---------------------------------------------------------------------------
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/data/campaign_ddm}"
PARTITION="${PARTITION:-cpu}"
TIME_LIMIT_TASK="${TIME_LIMIT_TASK:-01:00:00}"
TIME_LIMIT_PRE="${TIME_LIMIT_PRE:-00:30:00}"
MEM_PER_TASK="${MEM_PER_TASK:-4G}"

DRY_RUN=0; FORCE=0; KEEP_RAW=0; ONLY=""
CHUNK=100; MAX_CONCURRENT=100; REPS=1000

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)        DRY_RUN=1 ;;
        --force)          FORCE=1 ;;
        --keep-raw)       KEEP_RAW=1 ;;
        --only)           ONLY="$2"; shift ;;
        --only=*)         ONLY="${1#*=}" ;;
        --chunk)          CHUNK="$2"; shift ;;
        --chunk=*)        CHUNK="${1#*=}" ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift ;;
        --max-concurrent=*) MAX_CONCURRENT="${1#*=}" ;;
        --reps)           REPS="$2"; shift ;;
        --reps=*)         REPS="${1#*=}" ;;
        --results-root)   RESULTS_ROOT="$2"; shift ;;
        --results-root=*) RESULTS_ROOT="${1#*=}" ;;
        -h|--help)        sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

# ---------------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------------
PYTHON_BIN=""
for candidate in "$PROJECT_DIR/.venv/bin/python3" "$PROJECT_DIR/.venv/bin/python" \
                 python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then PYTHON_BIN="$candidate"; break; fi
done
[ -n "$PYTHON_BIN" ] || { echo "no python interpreter found" >&2; exit 1; }

cd "$PROJECT_DIR"

# ---------------------------------------------------------------------------
# EXECUTION MODE — inside a SLURM array task
# ---------------------------------------------------------------------------
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    EXTRA=()
    [ "${CAMPAIGN_FORCE:-0}" = "1" ] && EXTRA+=(--force)
    [ "${CAMPAIGN_KEEP_RAW:-0}" = "1" ] && EXTRA+=(--keep-raw)
    exec "$PYTHON_BIN" -m campaign.run_chunk \
        --index "$SLURM_ARRAY_TASK_ID" \
        --results-root "$RESULTS_ROOT" \
        --reps "$REPS" --chunk "$CHUNK" \
        "${EXTRA[@]}"
fi

# ---------------------------------------------------------------------------
# PRECOMPUTE MODE — inside the (non-array) precompute job
# ---------------------------------------------------------------------------
if [ "${CAMPAIGN_STAGE:-}" = "precompute" ]; then
    exec "$PYTHON_BIN" -m campaign.precompute_tables \
        --results-root "$RESULTS_ROOT" --workers "${SLURM_CPUS_PER_TASK:-4}"
fi

# ---------------------------------------------------------------------------
# SUBMISSION MODE
# ---------------------------------------------------------------------------
# Derive the plan (task count, --only index list, static gate) from the matrix.
PLAN="$("$PYTHON_BIN" - "$REPS" "$CHUNK" "$ONLY" <<'PY'
import sys
from campaign import factors, matrix

reps, chunk, only = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
table = matrix.task_table(reps, chunk)
if only:
    name, _, k = only.partition(":")
    target = matrix.find_condition(name)  # loud error on typos and ambiguity
    idx = [i for i, (c, ck) in enumerate(table)
           if (c.arm, c.name) == (target.arm, target.name)
           and (k == "" or ck == int(k))]
    if not idx:
        raise SystemExit(f"--only {only!r} matches no task")
    print("ARRAY_SPEC=" + ",".join(str(i) for i in idx))
else:
    print(f"ARRAY_SPEC=0-{len(table) - 1}")
print(f"TOTAL_TASKS={len(table)}")
print(f"N_CONDITIONS={len(matrix.build_conditions())}")
print(f"STATIC_GATED={int(factors.STATIC_CONTROL_Z is None)}")
PY
)"
eval "$PLAN"

echo "Embodied-DDM campaign"
echo "  project        : ${PROJECT_DIR}"
echo "  results root   : ${RESULTS_ROOT}"
echo "  reps / chunk   : ${REPS} / ${CHUNK}"
echo "  conditions     : ${N_CONDITIONS}"
echo "  array tasks    : ${TOTAL_TASKS}  (spec: ${ARRAY_SPEC}%${MAX_CONCURRENT})"
echo "  force          : ${FORCE}   keep-raw: ${KEEP_RAW}"
if [ "$STATIC_GATED" = "1" ]; then
    echo ""
    echo "  STATIC CONTROL: NOT SUBMITTED. controls.static.z_manual"
    echo "  (campaign/factors.py: STATIC_CONTROL_Z) is an unresolved methodological"
    echo "  parameter — no value is invented (CAMPAIGN_SPEC 5.2). The main matrix"
    echo "  and the quasi-static control are submitted without it; see the dry-run"
    echo "  z_bellman(0) table for the evidence to choose it from."
fi
echo ""

if [ "$DRY_RUN" = "1" ]; then
    "$PYTHON_BIN" -m campaign.dry_run --reps "$REPS" --chunk "$CHUNK" \
        --results-root "$RESULTS_ROOT" --max-concurrent "$MAX_CONCURRENT"
    echo ""
    echo "DRY RUN — nothing submitted."
    exit 0
fi

mkdir -p "$RESULTS_ROOT/slurm_logs"

# Job 1: the table precompute (Section 7.3) — the array depends on it.
PRE_ID="$(sbatch --parsable \
    --job-name=ddm_precompute \
    --partition="$PARTITION" --time="$TIME_LIMIT_PRE" \
    --mem="$MEM_PER_TASK" --cpus-per-task=4 \
    --output="$RESULTS_ROOT/slurm_logs/precompute_%j.out" \
    --error="$RESULTS_ROOT/slurm_logs/precompute_%j.err" \
    --export=ALL,CAMPAIGN_STAGE=precompute,PROJECT_DIR="$PROJECT_DIR",RESULTS_ROOT="$RESULTS_ROOT" \
    "$0")"
echo "precompute job : ${PRE_ID}"

# Job 2: the array, throttled, dependent on the precompute.
ARR_ID="$(sbatch --parsable \
    --job-name=ddm_campaign \
    --partition="$PARTITION" --time="$TIME_LIMIT_TASK" \
    --mem="$MEM_PER_TASK" --cpus-per-task=1 \
    --array="${ARRAY_SPEC}%${MAX_CONCURRENT}" \
    --dependency="afterok:${PRE_ID}" \
    --output="$RESULTS_ROOT/slurm_logs/task_%A_%a.out" \
    --error="$RESULTS_ROOT/slurm_logs/task_%A_%a.err" \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",RESULTS_ROOT="$RESULTS_ROOT",REPS="$REPS",CHUNK="$CHUNK",CAMPAIGN_FORCE="$FORCE",CAMPAIGN_KEEP_RAW="$KEEP_RAW" \
    "$0")"
echo "array job      : ${ARR_ID} (afterok:${PRE_ID})"
echo ""
echo "The campaign is idempotent: resubmitting this script after a partial failure"
echo "re-runs only incomplete chunks (Section 7.5)."
