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
#   ra_uc        ring attractor at the critical coupling u = 6.156868
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
# Order of operations, all of which this script performs or checks:
#     1. preflight   — grid, landmarks, arm-matching; refuses to submit on failure
#     2. precompute  — one Bellman table per ddm_bellman condition (Section 9.7)
#     3. sbatch      — the array
# =============================================================================

#SBATCH --job-name=flexibility_sweep
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
# --output/--error and --array are set on the sbatch command line below.
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
# EXECUTION MODE — inside a SLURM array task
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

# --- 1. Preflight ----------------------------------------------------------
# Blocks on a failure; warnings are printed and do not block.
echo "[1/3] preflight"
if ! "$PYTHON_BIN" -m flexibility.preflight; then
    echo
    echo "PREFLIGHT FAILED — not submitting." >&2
    exit 1
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo
    echo "DRY_RUN=1 — stopping before precompute and sbatch."
    exit 0
fi

# --- 2. Precompute the Bellman tables --------------------------------------
echo
echo "[2/3] precomputing Bellman tables into $CACHE_DIR"
mkdir -p "$CACHE_DIR"
"$PYTHON_BIN" -m flexibility.precompute_tables \
    --cache-dir "$CACHE_DIR" \
    --workers "${PRECOMPUTE_WORKERS:-8}"

# --- 3. Submit -------------------------------------------------------------
echo
echo "[3/3] submitting array 0-$((TOTAL_TASKS - 1))%$THROTTLE"
mkdir -p "$LOGS_DIR"
sbatch \
    --array="0-$((TOTAL_TASKS - 1))%${THROTTLE}" \
    --output="${LOGS_DIR}/flexibility_%A_%a.out" \
    --error="${LOGS_DIR}/flexibility_%A_%a.err" \
    --export=ALL,BASE_PATH_ROOT="$BASE_PATH_ROOT",CACHE_DIR="$CACHE_DIR",PROJECT_DIR="$PROJECT_DIR" \
    "$0"
