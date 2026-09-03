#!/bin/bash
# =============================================================================
# QD sweep at fixed noise (qd-sweep-fixed-noise-experiment.md)
# bwUniCluster3.0 SLURM job array — manifest-driven, both arms, one script
#
# Usage (login node) — both arms carry ALL THREE actual δ_Q (RECON D-12):
#   CAMPAIGN=ddm bash submit-qd-sweep-fixed-noise-bwunicluster.sh   # Arm B (126 pts)
#   bash submit-qd-sweep-fixed-noise-bwunicluster.sh                # Arm A (2040 cells)
# Rerun failures:  resubmit the same command (done replicates are skipped)
# Plan only:       DRY_RUN=1 bash submit-qd-sweep-fixed-noise-bwunicluster.sh
#
# MANIFESTS ARE NEVER GENERATED HERE. The §4 controller freeze needs the
# halted campaign's swept trials (the b* cross-check is blocking), so
# generate_manifest.py runs LOCALLY where seoul-data lives, and the two CSVs
# are copied into $LOGS_DIR before submission — the frontier's top-up
# discipline, applied to the whole campaign.
# =============================================================================
#SBATCH --job-name=qd_sweep_fixed_noise
#SBATCH --partition=cpu
#SBATCH --time=06:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

set -uo pipefail   # no -e in execution mode: one bad replicate must not kill the task

PROJECT_DIR="${PROJECT_DIR:-/home/kn/kn_kn/kn_pop547841/CollectiPy}"
VENV_BIN="$PROJECT_DIR/.venv/bin"
CAMPAIGN="${CAMPAIGN:-ra}"                # ra | ddm  (Arm A | Arm B)

case "$CAMPAIGN" in
    ra)  MANIFEST_NAME="ra_manifest.csv" ;;
    ddm) MANIFEST_NAME="ddm_manifest.csv" ;;
    *)   echo "CAMPAIGN must be 'ra' or 'ddm', got '$CAMPAIGN'" >&2; exit 1 ;;
esac

SWEEP_NAME="qd_sweep_fixed_noise"
LOGS_DIR="${LOGS_DIR:-/pfs/work9/workspace/scratch/kn_pop547841-mySpace/collectipy-data/beta_1/${SWEEP_NAME}}"
BASE_PATH_ROOT="${BASE_PATH_ROOT:-${LOGS_DIR}}"
MANIFEST="${MANIFEST:-${LOGS_DIR}/${MANIFEST_NAME}}"

RUNS_PER_CELL="${RUNS_PER_CELL:-100}"    # run_id 1..N, both arms (RECON D-12)
# One task = one whole cell (~0.5-5 s/run -> ~1-9 min/task). The RA full
# manifest is 1050 rows -> auto-chunked into MAX_ARRAY-sized sbatch arrays.
RUNS_PER_TASK="${RUNS_PER_TASK:-100}"
MAX_ARRAY="${MAX_ARRAY:-1000}"           # site array-size cap; auto-chunks
THROTTLE="${THROTTLE:-500}"              # concurrent tasks (cpu nodes are
#                                          shared; check `sacctmgr show assoc
#                                          user=$USER` if jobs sit pending)
# Declared walltime per task. At n = 100 the worst measured task is ~10-15
# min, and a SHORT honest declaration makes tasks backfill into scheduler
# gaps — the main queue-time lever. Raise it if you raise RUNS_PER_TASK /
# RUNS_PER_CELL; a task killed at the limit is resumed by resubmitting
# (.done idempotency), so an underestimate degrades gracefully.
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"
PARTITION="${PARTITION:-cpu}"            # cpu | cpu_il (second shared pool)
# Cells bundled into ONE array task, consecutively by manifest row. Every
# array ELEMENT counts as a queued job toward the site's per-user submit
# cap ("sbatch: Resource temporarily unavailable" = you hit it), so raise
# this until N_CELLS/CELLS_PER_TASK fits under the cap (query it with
#   sacctmgr show assoc user=$USER format=user,account,maxsubmitjobs,maxjobs)
# and scale TIME_LIMIT with it: worst measured cell ~ 100 runs x 5 s
# ~ 9 min, so e.g. CELLS_PER_TASK=5 wants TIME_LIMIT=01:00:00.
CELLS_PER_TASK="${CELLS_PER_TASK:-1}"
DRY_RUN="${DRY_RUN:-0}"

PYTHON_BIN=""
for c in "$VENV_BIN/python3.12" "$VENV_BIN/python3.10" "$VENV_BIN/python3" \
         "$VENV_BIN/python" python3; do
    command -v "$c" >/dev/null 2>&1 && { PYTHON_BIN="$c"; break; }
done
[ -n "$PYTHON_BIN" ] || { echo "no python" >&2; exit 1; }

BATCH="$PROJECT_DIR/scripts/qd_sweep_fixed_noise/run_batch.py"

# ---------------------------------------------------------------- SUBMISSION
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    set -e
    mkdir -p "$LOGS_DIR/failures"
    if [ ! -f "$MANIFEST" ]; then
        echo "no manifest at $MANIFEST" >&2
        echo "generate locally (the §4 freeze + cross-check needs" \
             "seoul-data) and copy it here:" >&2
        echo "  python3 scripts/qd_sweep_fixed_noise/generate_manifest.py" >&2
        echo "  scp results/qd_sweep_fixed_noise/{ra_manifest.csv,ra_manifest_actual100.csv,ra_manifest_rest.csv,ddm_manifest.csv,frozen_controllers.json} <cluster>:$LOGS_DIR/" >&2
        exit 1
    fi

    N_CELLS=$(( $(wc -l < "$MANIFEST") - 1 ))
    BATCHES=$(( (RUNS_PER_CELL + RUNS_PER_TASK - 1) / RUNS_PER_TASK ))
    N_GROUPS=$(( (N_CELLS + CELLS_PER_TASK - 1) / CELLS_PER_TASK ))
    TOTAL=$(( N_GROUPS * BATCHES ))
    echo "campaign=${CAMPAIGN} cells=${N_CELLS} cells/task=${CELLS_PER_TASK} batches/cell=${BATCHES} total tasks=${TOTAL}"
    echo "manifest=${MANIFEST}"
    echo "results under ${BASE_PATH_ROOT}/{ra,ddm}/actual_<bp>/..."

    if [ "$CAMPAIGN" = "ddm" ] && [ "${PRECOMPUTE:-1}" = "1" ]; then
        # Populate the Bellman table cache by running replicate 1 of every
        # point ONCE (login node). The cache key includes the DESIGN drift
        # A_expected, so the 3 designs x 3 c_e give 9 distinct tables shared
        # across all actual conditions; static rows have no table and their
        # replicate 1 just runs (the array skips it via .done).
        echo "precomputing Bellman tables (replicate 1 of each point) ..."
        if [ "$DRY_RUN" = "1" ]; then
            echo "  [dry run] skipped"
        else
            for ROW_IDX in $(seq 0 $(( N_CELLS - 1 ))); do
                "$PYTHON_BIN" "$BATCH" --arm ddm --manifest "$MANIFEST" \
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
            echo "[dry run] sbatch --array=0-$((CHUNK - 1))%${THROTTLE} --time=${TIME_LIMIT} --partition=${PARTITION} TASK_OFFSET=$OFFSET $0"
        else
            sbatch --array="0-$((CHUNK - 1))%${THROTTLE}" \
                --time="$TIME_LIMIT" --partition="$PARTITION" \
                --output="${LOGS_DIR}/%x_%A_%a.out" --error="${LOGS_DIR}/%x_%A_%a.err" \
                --export=ALL,CAMPAIGN="$CAMPAIGN",TASK_OFFSET="$OFFSET",MANIFEST="$MANIFEST",BASE_PATH_ROOT="$BASE_PATH_ROOT",LOGS_DIR="$LOGS_DIR",PROJECT_DIR="$PROJECT_DIR",RUNS_PER_CELL="$RUNS_PER_CELL",RUNS_PER_TASK="$RUNS_PER_TASK",CELLS_PER_TASK="$CELLS_PER_TASK" \
                "$0"
        fi
        OFFSET=$(( OFFSET + CHUNK ))
    done
    exit 0
fi

# ----------------------------------------------------------------- EXECUTION
GLOBAL_TASK=$(( ${TASK_OFFSET:-0} + SLURM_ARRAY_TASK_ID ))
BATCHES=$(( (RUNS_PER_CELL + RUNS_PER_TASK - 1) / RUNS_PER_TASK ))
N_CELLS=$(( $(wc -l < "$MANIFEST") - 1 ))
GROUP_IDX=$(( GLOBAL_TASK / BATCHES ))
BATCH_IDX=$(( GLOBAL_TASK % BATCHES ))
FIRST_RUN=$(( BATCH_IDX * RUNS_PER_TASK + 1 ))
LAST_RUN=$(( FIRST_RUN + RUNS_PER_TASK - 1 ))
[ "$LAST_RUN" -gt "$RUNS_PER_CELL" ] && LAST_RUN=$RUNS_PER_CELL
ROW_FIRST=$(( GROUP_IDX * CELLS_PER_TASK ))
ROW_LAST=$(( ROW_FIRST + CELLS_PER_TASK - 1 ))
[ "$ROW_LAST" -ge "$N_CELLS" ] && ROW_LAST=$(( N_CELLS - 1 ))

echo "[task ${GLOBAL_TASK}] campaign=${CAMPAIGN} rows=${ROW_FIRST}-${ROW_LAST} runs=${FIRST_RUN}-${LAST_RUN}"

EXTRA=()
if [ "$CAMPAIGN" = "ddm" ]; then
    EXTRA+=(--table-cache-dir "$BASE_PATH_ROOT/table_cache")
fi

RC=0
for ROW_IDX in $(seq "$ROW_FIRST" "$ROW_LAST"); do
    "$PYTHON_BIN" "$BATCH" --arm "$CAMPAIGN" --manifest "$MANIFEST" \
        --row "$ROW_IDX" --first-run "$FIRST_RUN" --last-run "$LAST_RUN" \
        --base-root "$BASE_PATH_ROOT" --failures-dir "$LOGS_DIR/failures" \
        --task-tag "${GLOBAL_TASK}_r${ROW_IDX}" "${EXTRA[@]}" || RC=1
done
exit $RC
