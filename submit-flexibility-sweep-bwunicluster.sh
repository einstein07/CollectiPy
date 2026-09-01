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
#     job 1        precompute — one Bellman table per ddm_bellman condition
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
WORKSPACE="${WORKSPACE:-/pfs/work9/workspace/scratch/kn_pop547841-mySpace/collectipy-data/beta_1}"
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
# Python interpreter — must be >= 3.10
#
# The simulator uses PEP 604 annotations (`X | Y`), so 3.9 fails at IMPORT time with
# a TypeError several frames deep in an unrelated module. Existence is therefore not
# enough: each candidate is version-checked, because a login node whose `python3` is
# 3.9 would otherwise be picked up silently and every array task would die after the
# job was already queued.
# ---------------------------------------------------------------------------
PYTHON_BIN=""
for candidate in \
    "$VENV_BIN/python3.12" \
    "$VENV_BIN/python3.10" \
    "$VENV_BIN/python3" \
    "$VENV_BIN/python" \
    python3.12 \
    python3.10 \
    python3 \
    python; do
    command -v "$candidate" >/dev/null 2>&1 || continue
    if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
        PYTHON_BIN="$candidate"
        break
    fi
done
if [ -z "$PYTHON_BIN" ]; then
    echo "No Python >= 3.10 found. The simulator uses PEP 604 ('X | Y') annotations," >&2
    echo "so 3.9 fails at import. Tried the venv at $VENV_BIN and the system path." >&2
    echo "On bwUniCluster:  module load devel/python/3.10.12_gnu_12.2" >&2
    echo "or create the venv:  python3.10 -m venv \"$PROJECT_DIR/.venv\"" >&2
    exit 1
fi
echo "python: $PYTHON_BIN ($("$PYTHON_BIN" -V 2>&1))"

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

# --- 2. Submit the precompute job -------------------------------------------
# Two jobs in one: it warms the shared Bellman table cache so replicates do not each
# solve the same PDE, AND it exercises every ddm_bellman condition once, which is what
# makes it a usable gate for step 3. There is no skip switch, because skipping it
# would silently remove the sanity check along with the optimisation.
echo
echo "[2/3] submitting the Bellman precompute job (warms the cache; gates the array)"
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

# --- 3. Submit the array, GATED on the precompute ---------------------------
# afterok, not afterany. The precompute solves every ddm_bellman condition once, which
# means it also EXERCISES every condition once -- so it doubles as the campaign's last
# and cheapest sanity check: a condition the model refuses to run fails here, in one
# job, instead of in 100 array tasks that each leave a partial result directory.
#
# That is not hypothetical. The first submission died exactly this way: at delta = 0
# the two strengths are identical, and the DDM's A_source 'ensemble' cannot deduce |A|
# from a zero gap, so it raised. One degenerate condition, caught before 6600 runs went
# out. Keep the gate.
#
# On failure: read the precompute's .err, fix the condition, resubmit. Releasing the
# array by hand (`scontrol update jobid=<id> dependency=`) is possible but means
# accepting that whatever the precompute caught is still in the grid.
echo
echo "[3/3] submitting array 0-$((TOTAL_TASKS - 1))%${THROTTLE}, gated on ${PRECOMPUTE_JOB}"
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
echo "Submitted. Watch with:  squeue -u \$USER"
echo
echo "The array runs whether or not the precompute succeeds: the table cache is"
echo "self-healing, so a failed precompute costs wall-clock, not results."
echo "To release the array immediately, ignoring the ordering:"
echo "    scontrol update jobid=${ARRAY_JOB} dependency="
