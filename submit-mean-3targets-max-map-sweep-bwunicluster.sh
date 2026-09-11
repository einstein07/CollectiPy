#!/bin/bash
# =============================================================================
# Three equal targets, MAX sensory map — (u, v) sweep, 100 runs per cell
# bwUniCluster3.0 SLURM job array
#
# The `max` arm only: mean_field_model.sensory_map = {"reduction": "max"}.
# (The `sum` data for this geometry exists already; nothing here runs `sum`.)
#
# Config: the current campaign's RA template, config/qd_sweep_ra_template.json
# (sigma 1.5, shared sensory stream white_rate 0.07071068, 1 tick/s,
# time_limit 1000, linear velocity 0.05, angular velocity 120, arrival radius
# 0.05, unthresholded readout, constant speed), with the third target added — all three at the positions of
# config/mean_field_3_targets_no_viz.json:
#     static_0 [0.383, -0.321]   static_1 [0.5, 0.0]   static_2 [0.383, 0.321]
# (bearings +40 / 0 / -40 degrees at range 0.5), every strength 5.0, no
# quality difference, and that config's arena (square, side 2): the pasted
# template's unit square cannot place static_1 — its 0.05 m cylinder at
# x = 0.5 straddles the wall and the simulator refuses the run. u and v are
# set per cell; nothing else is touched.
#
# Grid (default): the qd sweep's RA surface, u = 0, 2, 3, ..., 35 (35 levels)
# x v = 0.1, ..., 1.0 (10 kernels) = 350 cells x 100 runs = 35 000 runs.
# Override with U_VALUES / V_VALUES (space-separated).
#
# Output tree (one replicate per directory, .done on verified success):
#   <BASE_PATH_ROOT>/v_<v>/u_<u>/replicate_<id>/{config.json, run_meta.json,
#                                                config_folder_0/run_1.zip, .done}
#   <LOGS_DIR>/failures/task_<tag>.log        one line per failed replicate
#
# Seeds: frontier-v1 (scripts/ra_ddm_frontier/seeding.py), keyed on
# (SEED_DTH_DEG=40, SEED_DIFF_BP=0, run_id) — the same run_id gives the same
# percept stream and the same internal noise in every cell. Export
# SEED_DTH_DEG / SEED_DIFF_BP to key them like an existing `sum` data set.
#
# Usage (login node):
#   bash submit-mean-3targets-max-map-sweep-bwunicluster.sh              # submit
#   DRY_RUN=1 bash submit-mean-3targets-max-map-sweep-bwunicluster.sh    # plan only
# Rerun failures: resubmit the same command (replicates with .done are skipped).
#
# Local smoke test (workstation, no SLURM), one cell, 2 runs:
#   PROJECT_DIR=$PWD BASE_PATH_ROOT=/tmp/t3max LOGS_DIR=/tmp/t3max \
#   U_VALUES="6" V_VALUES="0.5" RUNS_PER_CELL=2 RUNS_PER_TASK=2 \
#   SLURM_ARRAY_TASK_ID=0 bash submit-mean-3targets-max-map-sweep-bwunicluster.sh
#
# Environment overrides:
#   PROJECT_DIR, BASE_PATH_ROOT, LOGS_DIR   paths (defaults below)
#   U_VALUES, V_VALUES        the grid (default: run_batch.py --print-grid)
#   RUNS_PER_CELL=100         replicates per (u, v)
#   RUNS_PER_TASK=100         replicates per array task (one cell per task)
#   MAX_ARRAY=1000            site array-size cap; arrays are chunked to it
#   THROTTLE=200              concurrent tasks per array
#   TIME_LIMIT=01:00:00       per-task wall clock (measured 0.5-5 s/run on the
#                             qd sweep; stiff high-u cells are the slow ones)
#   PARTITION=cpu             cpu | cpu_il
#   ARENA_SIDE=2              square arena side (the 3-target config's)
#   SEED_DTH_DEG=40 SEED_DIFF_BP=0   seed keys (see above)
#   DRY_RUN=1                 print the plan and the sbatch commands only
# =============================================================================
#SBATCH --job-name=3targets_max_map
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

set -uo pipefail   # no -e in execution mode: one bad replicate must not kill the task

PROJECT_DIR="${PROJECT_DIR:-/home/kn/kn_kn/kn_pop547841/CollectiPy}"
VENV_BIN="$PROJECT_DIR/.venv/bin"
SWEEP_NAME="three_targets_max_map"
LOGS_DIR="${LOGS_DIR:-/pfs/work9/workspace/scratch/kn_pop547841-mySpace/collectipy-data/beta_1/${SWEEP_NAME}}"
BASE_PATH_ROOT="${BASE_PATH_ROOT:-${LOGS_DIR}}"

RUNS_PER_CELL="${RUNS_PER_CELL:-100}"
RUNS_PER_TASK="${RUNS_PER_TASK:-100}"
MAX_ARRAY="${MAX_ARRAY:-1000}"
THROTTLE="${THROTTLE:-200}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
PARTITION="${PARTITION:-cpu}"
ARENA_SIDE="${ARENA_SIDE:-2}"
DRY_RUN="${DRY_RUN:-0}"
export SEED_DTH_DEG="${SEED_DTH_DEG:-40}"
export SEED_DIFF_BP="${SEED_DIFF_BP:-0}"

PYTHON_BIN=""
for c in "$VENV_BIN/python3.12" "$VENV_BIN/python3.10" "$VENV_BIN/python3" \
         "$VENV_BIN/python" python3.10 python3; do
    command -v "$c" >/dev/null 2>&1 && { PYTHON_BIN="$c"; break; }
done
[ -n "$PYTHON_BIN" ] || { echo "no python" >&2; exit 1; }

RUNNER="$PROJECT_DIR/scripts/three_targets_max/run_batch.py"
[ -f "$RUNNER" ] || { echo "runner not found: $RUNNER" >&2; exit 1; }

# The grid: from the runner unless overridden, so the submit plan and the
# execution mapping can never disagree.
if [ -z "${U_VALUES:-}" ] || [ -z "${V_VALUES:-}" ]; then
    GRID="$("$PYTHON_BIN" "$RUNNER" --print-grid)"
    [ -n "${U_VALUES:-}" ] || U_VALUES="$(printf '%s\n' "$GRID" | sed -n '1p')"
    [ -n "${V_VALUES:-}" ] || V_VALUES="$(printf '%s\n' "$GRID" | sed -n '2p')"
fi
read -ra U_ARR <<< "$U_VALUES"
read -ra V_ARR <<< "$V_VALUES"
N_U="${#U_ARR[@]}"
N_V="${#V_ARR[@]}"
N_CELLS=$(( N_U * N_V ))
BATCHES=$(( (RUNS_PER_CELL + RUNS_PER_TASK - 1) / RUNS_PER_TASK ))
TOTAL=$(( N_CELLS * BATCHES ))
EXTRA=()
[ -n "$ARENA_SIDE" ] && EXTRA+=(--arena-side "$ARENA_SIDE")

# ---------------------------------------------------------------- SUBMISSION
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    set -e
    echo "Three equal targets, MAX sensory map — SLURM plan"
    echo "  project dir      : ${PROJECT_DIR}"
    echo "  results root     : ${BASE_PATH_ROOT}"
    echo "  logs             : ${LOGS_DIR}"
    echo "  python           : ${PYTHON_BIN}"
    echo "  u values (${N_U})  : ${U_VALUES}"
    echo "  v values (${N_V})  : ${V_VALUES}"
    echo "  cells            : ${N_CELLS}  (v-major, then u)"
    echo "  runs per cell    : ${RUNS_PER_CELL}  (runs per task ${RUNS_PER_TASK}, batches per cell ${BATCHES})"
    echo "  array tasks      : ${TOTAL}  (task = cell_index * ${BATCHES} + batch), chunked at ${MAX_ARRAY}"
    echo "  throttle / time  : ${THROTTLE} concurrent, ${TIME_LIMIT} per task, partition ${PARTITION}"
    echo "  seeds            : frontier-v1, dth ${SEED_DTH_DEG}, diff_bp ${SEED_DIFF_BP}, run_id 1..${RUNS_PER_CELL}"
    echo "  arena            : square, side ${ARENA_SIDE}"
    echo ""
    # Every emitted config passes the runner's assertions; probe one cell now
    # so a drifted template fails here, on the login node, not in 350 tasks.
    "$PYTHON_BIN" "$RUNNER" --v "${V_ARR[0]}" --u "${U_ARR[0]}" --first-run 1 --last-run 1 \
        --base-root "$(mktemp -d)" --configs-only "${EXTRA[@]}" >/dev/null
    echo "  template + patch : OK"
    mkdir -p "$LOGS_DIR/failures"
    OFFSET=0
    while [ "$OFFSET" -lt "$TOTAL" ]; do
        CHUNK=$(( TOTAL - OFFSET )); [ "$CHUNK" -gt "$MAX_ARRAY" ] && CHUNK=$MAX_ARRAY
        if [ "$DRY_RUN" = "1" ]; then
            echo "[dry run] sbatch --array=0-$((CHUNK - 1))%${THROTTLE} --time=${TIME_LIMIT} --partition=${PARTITION} TASK_OFFSET=${OFFSET} $0"
        else
            sbatch --array="0-$((CHUNK - 1))%${THROTTLE}" \
                --time="$TIME_LIMIT" --partition="$PARTITION" \
                --output="${LOGS_DIR}/%x_%A_%a.out" --error="${LOGS_DIR}/%x_%A_%a.err" \
                --export=ALL,TASK_OFFSET="$OFFSET",PROJECT_DIR="$PROJECT_DIR",BASE_PATH_ROOT="$BASE_PATH_ROOT",LOGS_DIR="$LOGS_DIR",U_VALUES="$U_VALUES",V_VALUES="$V_VALUES",RUNS_PER_CELL="$RUNS_PER_CELL",RUNS_PER_TASK="$RUNS_PER_TASK",ARENA_SIDE="$ARENA_SIDE",SEED_DTH_DEG="$SEED_DTH_DEG",SEED_DIFF_BP="$SEED_DIFF_BP" \
                "$0"
        fi
        OFFSET=$(( OFFSET + CHUNK ))
    done
    [ "$DRY_RUN" = "1" ] && echo "DRY RUN — nothing submitted."
    exit 0
fi

# ----------------------------------------------------------------- EXECUTION
GLOBAL_TASK=$(( ${TASK_OFFSET:-0} + SLURM_ARRAY_TASK_ID ))
if [ "$GLOBAL_TASK" -ge "$TOTAL" ]; then
    echo "[task ${GLOBAL_TASK}] beyond the task table (${TOTAL}); nothing to do"
    exit 0
fi
CELL_IDX=$(( GLOBAL_TASK / BATCHES ))
BATCH_IDX=$(( GLOBAL_TASK % BATCHES ))
V_IDX=$(( CELL_IDX / N_U ))
U_IDX=$(( CELL_IDX % N_U ))
V="${V_ARR[$V_IDX]}"
U="${U_ARR[$U_IDX]}"
FIRST_RUN=$(( BATCH_IDX * RUNS_PER_TASK + 1 ))
LAST_RUN=$(( FIRST_RUN + RUNS_PER_TASK - 1 ))
[ "$LAST_RUN" -gt "$RUNS_PER_CELL" ] && LAST_RUN=$RUNS_PER_CELL

echo "[task ${GLOBAL_TASK}] host=$(hostname) v=${V} u=${U} runs=${FIRST_RUN}-${LAST_RUN}"
cd "$PROJECT_DIR"
exec "$PYTHON_BIN" "$RUNNER" --v "$V" --u "$U" \
    --first-run "$FIRST_RUN" --last-run "$LAST_RUN" \
    --base-root "$BASE_PATH_ROOT" --failures-dir "$LOGS_DIR/failures" \
    --task-tag "${GLOBAL_TASK}_v${V}_u${U}" "${EXTRA[@]}"
