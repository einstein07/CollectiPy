#!/bin/bash
# =============================================================================
# Flexibility campaign — ring attractor at two gains vs. a collapsing-bound DDM,
# under a mid-trial world change, swept over option quality difference.
# bwUniCluster3.0 SLURM job array.
#
# A single agent commits to one of two targets, the arena then EXCHANGES the two
# target strengths, and the agent may or may not reverse before it arrives. The
# dependent variable is REVERSAL, not accuracy.
#
#   ra_u6.2      ring attractor, near-critical gain u = 6.2 (kernel v = 0.5)
#   ra_u8        ring attractor at u = 8 (deep attractor; the rigidity arm)
#   ddm_bellman  collapsing boundary, terminal halt_sprt
#
# All three arms replay the SAME noise realisation at the same (delta, replicate):
# the trial seed excludes the arm, so the analysis is paired.
#
# Everything about the grid, the operating point and the arms lives in
# flexibility/factors.py — this script only decomposes the array and calls
# flexibility.run_chunk. It never patches a config itself.
#
# Usage (from the login node):
#     bash submit-flexibility-sweep-bwunicluster.sh
#
# NOTHING IS SIMULATED ON THE LOGIN NODE. The login node runs only the preflight
# (a few seconds of arithmetic) and two sbatch calls. The Bellman precompute is a
# COMPUTE JOB of its own, and the array is submitted with an afterok dependency on
# it, so the tables are in the cache before the first task starts.
#
#     login node   preflight, then submit both jobs
#     job 1        precompute — one Bellman table per ddm_bellman condition (§9.7)
#     job 2        the array — depends on job 1 completing successfully
#
# Running the precompute on the login node instead spawns a simulator (which itself
# forks an arena, a manager and a detector process) once per condition; that is what
# a login node's process and CPU budget is there to stop, and being killed mid-solve
# leaves a partially populated cache that the array would silently re-solve.
# =============================================================================

# Resource requests are given per stage on the sbatch command lines below, not as
# #SBATCH directives, because this one file is submitted twice with different needs.
# -----------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/kn/kn_kn/kn_pop547841/CollectiPy}"
VENV_BIN="$PROJECT_DIR/.venv/bin"
WORKSPACE="${WORKSPACE:-/pfs/work9/workspace/scratch/kn_pop547841-mySpace/collectipy-data}"
LOGS_DIR="${LOGS_DIR:-$WORKSPACE/flexibility_sweep}"
BASE_PATH_ROOT="${BASE_PATH_ROOT:-$LOGS_DIR}"
# The table cache MUST be shared across the array: at N_t ~ 45k a cold solve is
# ~8 s, so letting every replicate re-solve its condition's table would cost about
# 100x what the campaign needs.
CACHE_DIR="${CACHE_DIR:-$BASE_PATH_ROOT/table_cache}"
THROTTLE="${THROTTLE:-100}"

PARTITION="${PARTITION:-cpu}"
PRECOMPUTE_TIME="${PRECOMPUTE_TIME:-01:00:00}"
PRECOMPUTE_MEM="${PRECOMPUTE_MEM:-8G}"
PRECOMPUTE_CPUS="${PRECOMPUTE_CPUS:-1}"
# Serial by default: each solve runs a simulator that forks ~4 processes, so nesting
# a worker pool on top multiplies them. 23 solves at ~9 s is ~3.5 minutes serially.
PRECOMPUTE_WORKERS="${PRECOMPUTE_WORKERS:-1}"
ARRAY_TIME="${ARRAY_TIME:-24:00:00}"
ARRAY_MEM="${ARRAY_MEM:-4G}"

# ---------------------------------------------------------------------------
# Python interpreter (prefer venv, fall back to system)
# ---------------------------------------------------------------------------
PYTHON_BIN=""
for candidate in \
    "$VENV_BIN/python3.12" \
    "$VENV_BIN/python3" \
    "$VENV_BIN/python" \
    python3.10 \
    python3 \
    python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON_BIN="$candidate"
        break
    fi
done
if [ -z "$PYTHON_BIN" ]; then
    echo "Python interpreter not found." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# EXECUTION MODE 1 — the precompute job (submitted below, runs on a compute node)
# ---------------------------------------------------------------------------
if [ "${FLEX_STAGE:-}" = "precompute" ]; then
    cd "$PROJECT_DIR"
    exec "$PYTHON_BIN" -m flexibility.precompute_tables \
        --cache-dir "$CACHE_DIR" \
        --workers "$PRECOMPUTE_WORKERS"
fi

# ---------------------------------------------------------------------------
# EXECUTION MODE 2 — one array task
# ---------------------------------------------------------------------------
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    cd "$PROJECT_DIR"
    exec "$PYTHON_BIN" -m flexibility.run_chunk \
        --index "$SLURM_ARRAY_TASK_ID" \
        --results-root "$BASE_PATH_ROOT" \
        --cache-dir "$CACHE_DIR"
fi

# ---------------------------------------------------------------------------
# SUBMISSION MODE
# ---------------------------------------------------------------------------
cd "$PROJECT_DIR"

TOTAL_TASKS="$("$PYTHON_BIN" -c 'from flexibility import matrix; print(matrix.total_tasks())')"
TOTAL_RUNS="$("$PYTHON_BIN" -c 'from flexibility import matrix; print(matrix.total_runs())')"

echo "============================================================"
echo "Flexibility campaign"
echo "  results root : $BASE_PATH_ROOT"
echo "  table cache  : $CACHE_DIR"
echo "  total runs   : $TOTAL_RUNS"
echo "  array tasks  : $TOTAL_TASKS  (throttle %$THROTTLE)"
echo "============================================================"
echo

# --- 1. Preflight (login node: arithmetic only, no simulation) --------------
echo "[1/3] preflight"
if ! "$PYTHON_BIN" -m flexibility.preflight; then
    echo
    echo "PREFLIGHT FAILED — not submitting." >&2
    exit 1
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo
    echo "DRY_RUN=1 — stopping before submission."
    exit 0
fi

mkdir -p "$LOGS_DIR" "$CACHE_DIR"

# --- 2. Submit the precompute job ------------------------------------------
echo
echo "[2/3] submitting the Bellman precompute job"
PRECOMPUTE_JOB=$(sbatch --parsable \
    --job-name=flex_precompute \
    --partition="$PARTITION" \
    --time="$PRECOMPUTE_TIME" \
    --mem="$PRECOMPUTE_MEM" \
    --cpus-per-task="$PRECOMPUTE_CPUS" \
    --output="${LOGS_DIR}/precompute_%j.out" \
    --error="${LOGS_DIR}/precompute_%j.err" \
    --export=ALL,FLEX_STAGE=precompute,CACHE_DIR="$CACHE_DIR",PROJECT_DIR="$PROJECT_DIR",PRECOMPUTE_WORKERS="$PRECOMPUTE_WORKERS" \
    "$0")
echo "      job ${PRECOMPUTE_JOB}"

# --- 3. Submit the array, held until the tables exist -----------------------
# afterok, not afterany: if the precompute fails the tables are missing or partial,
# and every task would then re-solve its own in-process -- the ~100x waste the shared
# cache exists to prevent, discovered only from the wall-clock.
echo
echo "[3/3] submitting array 0-$((TOTAL_TASKS - 1))%${THROTTLE}, held on ${PRECOMPUTE_JOB}"
ARRAY_JOB=$(sbatch --parsable \
    --job-name=flexibility_sweep \
    --partition="$PARTITION" \
    --time="$ARRAY_TIME" \
    --mem="$ARRAY_MEM" \
    --cpus-per-task=1 \
    --dependency=afterok:"$PRECOMPUTE_JOB" \
    --array="0-$((TOTAL_TASKS - 1))%${THROTTLE}" \
    --output="${LOGS_DIR}/flexibility_%A_%a.out" \
    --error="${LOGS_DIR}/flexibility_%A_%a.err" \
    --export=ALL,BASE_PATH_ROOT="$BASE_PATH_ROOT",CACHE_DIR="$CACHE_DIR",PROJECT_DIR="$PROJECT_DIR" \
    "$0")
echo "      job ${ARRAY_JOB}"

echo
echo "Submitted. Watch with:  squeue -j ${PRECOMPUTE_JOB},${ARRAY_JOB}"
echo "If the precompute fails, the array stays held and can be cancelled with:"
echo "    scancel ${ARRAY_JOB}"
