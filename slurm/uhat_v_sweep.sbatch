#!/bin/bash
# =============================================================================
# (u_hat, v) factorial — criticality-normalized coupling x kernel shape
# bwUniCluster 3.0 SLURM job array
#
# Design: uhat-v-factorial-experiment.md. In short:
#   cells   : 8 u_hat x 10 v = 80, FULLY CROSSED (not the matched (v, u*) pairs
#             of submit-ra-frontier-sweep-bwunicluster.sh — this sweep is the
#             factorial that separates u_hat from v)
#   u       : u = u_hat * u_star(v), read from manifest.json, never typed in
#   trials  : 400 per cell, on the SAME seed list in every cell (paired design)
#   config  : the RA arm of the RA-DDM comparison, unchanged except u and v;
#             see scripts/uhat_v_sweep/RECON.md for every discrepancy found
#
# ONE output file per array task: results/uhat_v_sweep/raw/cell<K>_chunk<J>.parquet.
# Nothing ever appends to a shared file.
#
# Boilerplate (partition, resources, venv discovery, layout) is taken from
# submit-ra-frontier-sweep-bwunicluster.sh, the sweep this one follows on from.
#
# Usage (from the login node):
#   bash slurm/uhat_v_sweep.sbatch                    # print the plan, submit nothing
#   DRY_RUN=1 bash slurm/uhat_v_sweep.sbatch          # same thing, explicitly
#   SUBMIT=1  bash slurm/uhat_v_sweep.sbatch          # actually sbatch it
#   TEST_ONLY=1 bash slurm/uhat_v_sweep.sbatch        # sbatch --test-only dry run
#
# Environment overrides:
#   PROJECT_DIR=path        repo root       (default: the cluster path below)
#   RESULTS_ROOT=path       results root    (default: workspace scratch)
#   TRIALS_PER_CHUNK=N      trials per task (default: the whole cell)
#   TIME_LIMIT=hh:mm:ss     per-task wall clock (default: 00:30:00)
#   ARRAY_THROTTLE=N        max concurrent tasks (default: 40)
#   ONLY=K[:J]              submit one cell (and optionally one chunk) only
# =============================================================================

# --- SLURM directives (only active when submitted via sbatch) ----------------
#SBATCH --job-name=uhat_v
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
RESULTS_ROOT="${RESULTS_ROOT:-/pfs/work9/workspace/scratch/kn_pop547841-mySpace/collectipy-data/beta_1/uhat_v_sweep}"
LOGS_DIR="${LOGS_DIR:-${RESULTS_ROOT}/slurm_logs}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"
ARRAY_THROTTLE="${ARRAY_THROTTLE:-40}"
DRY_RUN="${DRY_RUN:-0}"
SUBMIT="${SUBMIT:-0}"
TEST_ONLY="${TEST_ONLY:-0}"
ONLY="${ONLY:-}"

# ---------------------------------------------------------------------------
# Python interpreter — same discovery order as the existing sweep scripts.
# ---------------------------------------------------------------------------
PYTHON_BIN=""
for candidate in \
    "$VENV_BIN/python3.12" \
    "$VENV_BIN/python3.10" \
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

RUNNER="$PROJECT_DIR/scripts/uhat_v_sweep/run_cell.py"
MANIFEST="$RESULTS_ROOT/manifest.json"

# ---------------------------------------------------------------------------
# EXECUTION MODE — inside a SLURM array task
# ---------------------------------------------------------------------------
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    # Node-local scratch for the run archives; only the one row file per task
    # is published to shared storage, and run_cell.py does that atomically.
    export TMPDIR="${TMPDIR:-/tmp}"
    echo "[array ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}] host=$(hostname) TMPDIR=$TMPDIR"
    exec "$PYTHON_BIN" "$RUNNER" \
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
    echo "  $PYTHON_BIN $PROJECT_DIR/scripts/uhat_v_sweep/generate_manifest.py \\" >&2
    echo "      --results-root $RESULTS_ROOT" >&2
    exit 1
fi
if [ ! -f "$RUNNER" ]; then
    echo "Runner not found: $RUNNER" >&2
    exit 1
fi

# Task geometry comes from the manifest, so it cannot drift from the design.
read -r N_CELLS N_TRIALS PER_CHUNK N_CHUNKS N_TASKS N_EXCLUDED <<EOF
$("$PYTHON_BIN" - "$MANIFEST" "${TRIALS_PER_CHUNK:-0}" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
n_trials = int(manifest["n_trials"])
per_chunk = int(sys.argv[2]) or n_trials
chunks = (n_trials + per_chunk - 1) // per_chunk
n_cells = int(manifest["n_cells"])
excluded = sum(1 for c in manifest["cells"] if c.get("excluded"))
print(n_cells, n_trials, per_chunk, chunks, n_cells * chunks, excluded)
PY
)
EOF

echo "(u_hat, v) factorial — SLURM plan"
echo "  project dir      : ${PROJECT_DIR}"
echo "  results root     : ${RESULTS_ROOT}"
echo "  manifest         : ${MANIFEST}"
echo "  python           : ${PYTHON_BIN}"
echo "  cells            : ${N_CELLS}  (excluded: ${N_EXCLUDED} — those tasks exit 0 immediately)"
echo "  trials per cell  : ${N_TRIALS}"
echo "  trials per task  : ${PER_CHUNK}  (chunks per cell: ${N_CHUNKS})"
echo "  array tasks      : ${N_TASKS}  (index = cell_id * ${N_CHUNKS} + chunk)"
echo "  total trials     : $(( N_CELLS * N_TRIALS ))"
echo "  wall limit/task  : ${TIME_LIMIT}"
echo "  throttle         : ${ARRAY_THROTTLE} concurrent"
echo "  logs             : ${LOGS_DIR}"
echo ""
echo "  Measured locally: 0.44 s/trial at (v=0.5, u_hat=0.9), 0.84 s/trial at the"
echo "  stiffest cell (v=0.1, u_hat=1.5, u=34.2, where half the trials run the full"
echo "  100-tick budget). A 400-trial task is therefore ~3-6 min; the 30 min limit"
echo "  above is well over 2x the worst measured chunk."
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
    --output="${LOGS_DIR}/uhat_v_%A_%a.out"
    --error="${LOGS_DIR}/uhat_v_%A_%a.err"
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",RESULTS_ROOT="$RESULTS_ROOT",TRIALS_PER_CHUNK="${TRIALS_PER_CHUNK:-}"
)

if [ "$TEST_ONLY" = "1" ]; then
    echo "sbatch --test-only:"
    sbatch --test-only "${SBATCH_ARGS[@]}" "$0"
    exit 0
fi

sbatch "${SBATCH_ARGS[@]}" "$0"
