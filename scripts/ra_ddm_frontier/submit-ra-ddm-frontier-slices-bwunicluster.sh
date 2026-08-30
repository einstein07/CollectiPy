#!/bin/bash
# =============================================================================
# RA slices + DDM rerun for the RA–DDM frontier comparison (1 %, Δθ = 60°)
# bwUniCluster3.0 SLURM job array — manifest-driven, both campaigns, one script
#
# Usage (login node):
#   bash submit-ra-ddm-frontier-slices-bwunicluster.sh                # RA slices
#   CAMPAIGN=ddm bash submit-ra-ddm-frontier-slices-bwunicluster.sh  # DDM rerun
# Rerun failures:  resubmit the same command (done replicates are skipped)
# Plan only:       DRY_RUN=1 bash submit-ra-ddm-frontier-slices-bwunicluster.sh
#
# Smoke (§8, local — the real script path, no SLURM):
#   see scripts/ra_ddm_frontier/README.md
# =============================================================================
#SBATCH --job-name=ra_ddm_frontier
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

set -uo pipefail   # no -e in execution mode: one bad replicate must not kill the task

PROJECT_DIR="${PROJECT_DIR:-/home/kn/kn_kn/kn_pop547841/CollectiPy}"
VENV_BIN="$PROJECT_DIR/.venv/bin"
CAMPAIGN="${CAMPAIGN:-ra}"                # ra | ddm

case "$CAMPAIGN" in
    ra)  SWEEP_NAME="ra_ddm_frontier_slices"; MANIFEST_NAME="manifest.csv" ;;
    ddm) SWEEP_NAME="ra_ddm_frontier_ddm";    MANIFEST_NAME="ddm_manifest.csv" ;;
    *)   echo "CAMPAIGN must be 'ra' or 'ddm', got '$CAMPAIGN'" >&2; exit 1 ;;
esac

LOGS_DIR="${LOGS_DIR:-/pfs/work9/workspace/scratch/kn_pop547841-mySpace/collectipy-data/${SWEEP_NAME}}"
BASE_PATH_ROOT="${BASE_PATH_ROOT:-${LOGS_DIR}}"
MANIFEST="${MANIFEST:-${LOGS_DIR}/${MANIFEST_NAME}}"

DIFF=0.01                 # 1 % quality difference — the DDM panel being matched
RUNS_PER_CELL="${RUNS_PER_CELL:-1000}"   # run_id 1..N, both campaigns
RUNS_PER_TASK="${RUNS_PER_TASK:-100}"    # adjust after smoke timing
MAX_ARRAY="${MAX_ARRAY:-1000}"           # site array-size cap; submission auto-chunks
THROTTLE="${THROTTLE:-100}"              # concurrent tasks
DRY_RUN="${DRY_RUN:-0}"

PYTHON_BIN=""
for c in "$VENV_BIN/python3.12" "$VENV_BIN/python3.10" "$VENV_BIN/python3" \
         "$VENV_BIN/python" python3; do
    command -v "$c" >/dev/null 2>&1 && { PYTHON_BIN="$c"; break; }
done
[ -n "$PYTHON_BIN" ] || { echo "no python" >&2; exit 1; }

GEN="$PROJECT_DIR/scripts/ra_ddm_frontier/generate_manifest.py"
BATCH="$PROJECT_DIR/scripts/ra_ddm_frontier/run_batch.py"

# ---------------------------------------------------------------- SUBMISSION
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    set -e
    mkdir -p "$LOGS_DIR/failures"
    "$PYTHON_BIN" "$GEN" --campaign "$CAMPAIGN" --diff "$DIFF" \
        --n-runs "$RUNS_PER_CELL" --out "$MANIFEST"

    N_CELLS=$(( $(wc -l < "$MANIFEST") - 1 ))
    BATCHES=$(( (RUNS_PER_CELL + RUNS_PER_TASK - 1) / RUNS_PER_TASK ))
    TOTAL=$(( N_CELLS * BATCHES ))
    echo "campaign=${CAMPAIGN} cells=${N_CELLS} batches/cell=${BATCHES} total tasks=${TOTAL}"
    echo "manifest=${MANIFEST}"
    echo "results under ${BASE_PATH_ROOT}"

    if [ "$CAMPAIGN" = "ddm" ] && [ "${PRECOMPUTE:-1}" = "1" ]; then
        # Populate the Bellman table cache by running replicate 1 of every
        # point ONCE, into the real tree (cache key is computed inside the
        # solver, so running the model is the only mismatch-proof derivation —
        # same approach as campaign/precompute_tables.py). ~10 solves.
        echo "precomputing Bellman tables (replicate 1 of each point) ..."
        if [ "$DRY_RUN" = "1" ]; then
            echo "  [dry run] skipped"
        else
            for ROW_IDX in $(seq 0 $(( N_CELLS - 1 ))); do
                "$PYTHON_BIN" "$BATCH" --campaign ddm --manifest "$MANIFEST" \
                    --row "$ROW_IDX" --first-run 1 --last-run 1 \
                    --base-root "$BASE_PATH_ROOT" \
                    --table-cache-dir "$BASE_PATH_ROOT/table_cache" \
                    --failures-dir "$LOGS_DIR/failures" --task-tag "pre_$ROW_IDX"
            done
        fi
    fi

    OFFSET=0
    while [ "$OFFSET" -lt "$TOTAL" ]; do
        CHUNK=$(( TOTAL - OFFSET )); [ "$CHUNK" -gt "$MAX_ARRAY" ] && CHUNK=$MAX_ARRAY
        if [ "$DRY_RUN" = "1" ]; then
            echo "[dry run] sbatch --array=0-$((CHUNK - 1))%${THROTTLE} TASK_OFFSET=$OFFSET $0"
        else
            sbatch --array="0-$((CHUNK - 1))%${THROTTLE}" \
                --output="${LOGS_DIR}/%x_%A_%a.out" --error="${LOGS_DIR}/%x_%A_%a.err" \
                --export=ALL,CAMPAIGN="$CAMPAIGN",TASK_OFFSET="$OFFSET",MANIFEST="$MANIFEST",BASE_PATH_ROOT="$BASE_PATH_ROOT",LOGS_DIR="$LOGS_DIR",PROJECT_DIR="$PROJECT_DIR",RUNS_PER_CELL="$RUNS_PER_CELL",RUNS_PER_TASK="$RUNS_PER_TASK" \
                "$0"
        fi
        OFFSET=$(( OFFSET + CHUNK ))
    done
    exit 0
fi

# ----------------------------------------------------------------- EXECUTION
GLOBAL_TASK=$(( ${TASK_OFFSET:-0} + SLURM_ARRAY_TASK_ID ))
BATCHES=$(( (RUNS_PER_CELL + RUNS_PER_TASK - 1) / RUNS_PER_TASK ))
ROW_IDX=$(( GLOBAL_TASK / BATCHES ))
BATCH_IDX=$(( GLOBAL_TASK % BATCHES ))
FIRST_RUN=$(( BATCH_IDX * RUNS_PER_TASK + 1 ))
LAST_RUN=$(( FIRST_RUN + RUNS_PER_TASK - 1 ))
[ "$LAST_RUN" -gt "$RUNS_PER_CELL" ] && LAST_RUN=$RUNS_PER_CELL

echo "[task ${GLOBAL_TASK}] campaign=${CAMPAIGN} row=${ROW_IDX} runs=${FIRST_RUN}-${LAST_RUN}"

EXTRA=()
if [ "$CAMPAIGN" = "ddm" ]; then
    EXTRA+=(--table-cache-dir "$BASE_PATH_ROOT/table_cache")
fi

"$PYTHON_BIN" "$BATCH" --campaign "$CAMPAIGN" --manifest "$MANIFEST" \
    --row "$ROW_IDX" --first-run "$FIRST_RUN" --last-run "$LAST_RUN" \
    --base-root "$BASE_PATH_ROOT" --failures-dir "$LOGS_DIR/failures" \
    --task-tag "$GLOBAL_TASK" "${EXTRA[@]}"
exit $?
