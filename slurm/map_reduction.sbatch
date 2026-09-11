#!/bin/bash
# =============================================================================
# Sensory-map reduction experiment (max-sensory-map-spec.md Section 6)
# bwUniCluster 3.0 SLURM job array
#
# Design: reduction {sum, max} x Delta_close (7) x q_C (3) = 42 cells, 200 paired
# trials per cell (one seed list in every cell). One array task = one (cell,
# chunk); by default a chunk is the whole cell, so the array is 0-41.
#
# ONE output file per array task: <RESULTS_ROOT>/raw/cell<K>_chunk<J>.parquet.
# Nothing ever appends to a shared file. Tasks are idempotent: a complete file
# makes the task exit 0 at once, so a partial array is fixed by resubmitting.
#
# Boilerplate (partition, resources, venv discovery, layout) follows
# slurm/uhat_v_sweep.sbatch, the scaffolding this experiment reuses.
#
# Usage (login node):
#   $PY scripts/compare_map_reduction.py manifest --results-root <R>   # once
#   bash slurm/map_reduction.sbatch                 # print the plan, submit nothing
#   TEST_ONLY=1 bash slurm/map_reduction.sbatch     # sbatch --test-only
#   SUBMIT=1  bash slurm/map_reduction.sbatch       # actually sbatch it
#
# Afterwards, locally or on the login node:
#   $PY scripts/compare_map_reduction.py aggregate --results-root <R>
#   $PY scripts/compare_map_reduction.py plot      --results-root <R>
#
# Environment overrides:
#   PROJECT_DIR=path        repo root       (default: the cluster path below)
#   RESULTS_ROOT=path       results root    (default: workspace scratch)
#   TRIALS_PER_CHUNK=N      trials per task (default: the whole cell)
#   TIME_LIMIT=hh:mm:ss     per-task wall clock (default: 00:30:00)
#   ARRAY_THROTTLE=N        max concurrent tasks (default: 42)
#   ONLY=K[:J]              submit one cell (and optionally one chunk) only
# =============================================================================

# --- SLURM directives (only active when submitted via sbatch) ----------------
#SBATCH --job-name=map_reduction
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
# --output / --error / --array are set on the sbatch command line below.
# -----------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/kn/kn_kn/kn_pop547841/CollectiPy}"
VENV_BIN="$PROJECT_DIR/.venv/bin"
RESULTS_ROOT="${RESULTS_ROOT:-/pfs/work9/workspace/scratch/kn_pop547841-mySpace/collectipy-data/beta_1/map_reduction}"
LOGS_DIR="${LOGS_DIR:-${RESULTS_ROOT}/slurm_logs}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"
ARRAY_THROTTLE="${ARRAY_THROTTLE:-42}"
DRY_RUN="${DRY_RUN:-0}"
SUBMIT="${SUBMIT:-0}"
TEST_ONLY="${TEST_ONLY:-0}"
ONLY="${ONLY:-}"

PYTHON_BIN=""
for candidate in \
    "$VENV_BIN/python3.12" "$VENV_BIN/python3.10" "$VENV_BIN/python3" "$VENV_BIN/python" \
    python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON_BIN="$candidate"
        break
    fi
done
if [ -z "$PYTHON_BIN" ]; then
    echo "Python interpreter not found." >&2
    exit 1
fi

RUNNER="$PROJECT_DIR/scripts/compare_map_reduction.py"
MANIFEST="$RESULTS_ROOT/manifest.json"

# ---------------------------------------------------------------------------
# EXECUTION MODE — inside a SLURM array task
# ---------------------------------------------------------------------------
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    export TMPDIR="${TMPDIR:-/tmp}"
    echo "[array ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}] host=$(hostname) TMPDIR=$TMPDIR"
    exec "$PYTHON_BIN" "$RUNNER" run \
        --index "$SLURM_ARRAY_TASK_ID" \
        --results-root "$RESULTS_ROOT" \
        --manifest "$MANIFEST" \
        ${TRIALS_PER_CHUNK:+--trials-per-chunk "$TRIALS_PER_CHUNK"}
fi

# ---------------------------------------------------------------------------
# SUBMISSION MODE
# ---------------------------------------------------------------------------
if [ ! -f "$MANIFEST" ]; then
    echo "Manifest not found: $MANIFEST" >&2
    echo "Generate it first:" >&2
    echo "  $PYTHON_BIN $RUNNER manifest --results-root $RESULTS_ROOT" >&2
    exit 1
fi
if [ ! -f "$RUNNER" ]; then
    echo "Runner not found: $RUNNER" >&2
    exit 1
fi

read -r N_CELLS N_TRIALS PER_CHUNK N_CHUNKS N_TASKS <<EOF
$("$PYTHON_BIN" - "$MANIFEST" "${TRIALS_PER_CHUNK:-0}" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
n_trials = int(manifest["n_trials"])
per_chunk = int(sys.argv[2]) or n_trials
chunks = (n_trials + per_chunk - 1) // per_chunk
n_cells = int(manifest["n_cells"])
print(n_cells, n_trials, per_chunk, chunks, n_cells * chunks)
PY
)
EOF

echo "Sensory-map reduction experiment — SLURM plan"
echo "  project dir      : ${PROJECT_DIR}"
echo "  results root     : ${RESULTS_ROOT}"
echo "  manifest         : ${MANIFEST}"
echo "  python           : ${PYTHON_BIN}"
echo "  cells            : ${N_CELLS}"
echo "  trials per cell  : ${N_TRIALS}"
echo "  trials per task  : ${PER_CHUNK}  (chunks per cell: ${N_CHUNKS})"
echo "  array tasks      : ${N_TASKS}  (index = cell_id * ${N_CHUNKS} + chunk)"
echo "  total trials     : $(( N_CELLS * N_TRIALS ))"
echo "  wall limit/task  : ${TIME_LIMIT}"
echo "  throttle         : ${ARRAY_THROTTLE} concurrent"
echo "  logs             : ${LOGS_DIR}"
echo ""
echo "  Measured locally (22-core workstation, 16 workers): ~0.5 s per arriving"
echo "  trial and ~1.2 s per 100-tick timeout, so a 200-trial cell is 2-4 min."
echo ""

ARRAY_SPEC="0-$(( N_TASKS - 1 ))"
if [ -n "$ONLY" ]; then
    ONLY_CELL="${ONLY%%:*}"
    ONLY_CHUNK="${ONLY#*:}"
    [ "$ONLY_CHUNK" = "$ONLY" ] && ONLY_CHUNK=0
    ARRAY_SPEC="$(( ONLY_CELL * N_CHUNKS + ONLY_CHUNK ))"
    echo "  ONLY=${ONLY} -> array ${ARRAY_SPEC}"
    echo ""
fi

if [ "$DRY_RUN" = "1" ] || { [ "$SUBMIT" != "1" ] && [ "$TEST_ONLY" != "1" ]; }; then
    echo "Nothing submitted. To dry-run through SLURM:  TEST_ONLY=1 bash $0"
    echo "To submit for real:                           SUBMIT=1    bash $0"
    exit 0
fi

mkdir -p "$LOGS_DIR"

SBATCH_ARGS=(
    --array="${ARRAY_SPEC}%${ARRAY_THROTTLE}"
    --time="$TIME_LIMIT"
    --output="${LOGS_DIR}/map_reduction_%A_%a.out"
    --error="${LOGS_DIR}/map_reduction_%A_%a.err"
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",RESULTS_ROOT="$RESULTS_ROOT",TRIALS_PER_CHUNK="${TRIALS_PER_CHUNK:-}"
)

if [ "$TEST_ONLY" = "1" ]; then
    echo "sbatch --test-only:"
    sbatch --test-only "${SBATCH_ARGS[@]}" "$0"
    exit 0
fi

sbatch "${SBATCH_ARGS[@]}" "$0"
