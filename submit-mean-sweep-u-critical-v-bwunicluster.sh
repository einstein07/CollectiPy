#!/bin/bash
# =============================================================================
# Critical-u sweep across v — bump formation threshold, no targets
# bwUniCluster3.0 SLURM job arrays
#
# Purpose: repeat the single-v critical-u sweep (see
# submit-mean-sweep-u-critical-bwunicluster.sh, run at v = 0.5) for every
# v in {0.1, ..., 1.0}, so the bump-magnitude-vs-u curves can be overlaid on
# one plot and u*(v) read off as a function of the recurrent gain v.
#
# Protocol is unchanged from the v = 0.5 sweep, so the v = 0.5 series produced
# here reproduces the curve already in seoul-data/beta-1/u_critical_sweep:
#   - both targets carry strength 0.0, i.e. the bump forms spontaneously with
#     no external drive; u is the only thing pushing the ring past threshold
#   - mean steady-state ||z|| over RUNS_PER_U replicates at each u
#   - 75 u values log-spaced from 1 to 100 (ratio ≈ 1.064, 6.4 %/step), which
#     puts ~10 sample points in the 4–10 window bracketing u* ≈ 6.16 at v = 0.5
#     while staying coarse in the flat saturation regime above ~20
#
# The config template (config/mean_field_u_critical_v_sweep.json) is a
# template-form copy of the config.json dumped by one of those v = 0.5 runs:
# same time_limit, num_neurons, kappa, sigma, beta, integration settings, and
# the same *absent* mean-field keys (use_thresholding, scale_velocity, sigma_s)
# so the model defaults that ran then still apply. Do not "helpfully" fill
# those in — it changes the curve.
#
# Output tree (one complete u-sweep per v subtree):
#
#   <base>/v_<v>/u_<u>/replicate_<n>/config_folder_0/run_1.zip
#
# Aggregate + plot it with:  ./analyse-u-critical-v-sweep.py <base>
#
# Usage (from a login node):
#   bash submit-mean-sweep-u-critical-v-bwunicluster.sh
#
# Environment overrides (all optional):
#   V_VALUES="0.1 0.2"     sweep only these v values
#   RUNS_PER_U=20          replicates per (v, u)          [default 100]
#   RUNS_PER_TASK=10       replicates packed per array task [default 50]
#   MAX_CONCURRENT=50      per-array throttle             [default 50]
#   SKIP_EXISTING=1        skip replicates whose run_1.zip already exists [default 1]
#   BASE_PATH_ROOT=/path   results root                   [default LOGS_DIR]
#   SUBMIT_DELAY=10        seconds between array submissions [default 10]
#   SUBMIT_RETRIES=5       sbatch attempts per array        [default 5]
#   CHAIN=1                run the v-arrays one after another via --dependency
#   DRY_RUN=1              print the sbatch commands instead of submitting
#
# One array is submitted per v value rather than one big array for the whole
# grid: 10 x 75 x 20 = 15000 tasks would exceed the cluster's MaxArraySize,
# whereas each per-v array is 150 tasks. Each task derives its (u, run_id)
# range from SLURM_ARRAY_TASK_ID and reads its v from the exported V_VALUE.
#
# Job-record pressure is the binding constraint here, not the array cap. At
# RUNS_PER_TASK=10 the ten arrays put 7500 job records on slurmctld in one
# burst, which it answers with
#     sbatch: error: Slurm temporarily unable to accept job, sleeping and retrying
# So the replicates are packed 50-per-task instead: 150 tasks per v, 1500
# records for the whole sweep, at ~19 min of wall time per task (~23 s/run,
# measured). SUBMIT_DELAY additionally spaces the ten sbatch calls out, and
# CHAIN=1 holds each v-array behind the previous one when the controller is
# busy enough that even that is too much at once.
# =============================================================================

# --- SLURM directives (only active when submitted via sbatch) ----------------
#SBATCH --job-name=u_critical_v_sweep
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
# --output, --error, --job-name and --array are set dynamically via sbatch flags below.
# -----------------------------------------------------------------------------

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR="${PROJECT_DIR:-/home/kn/kn_kn/kn_pop547841/CollectiPy}"
VENV_BIN="$PROJECT_DIR/.venv/bin"
CONFIG_TEMPLATE="$PROJECT_DIR/config/mean_field_u_critical_v_sweep.json"
LOGS_DIR="/pfs/work9/workspace/scratch/kn_pop547841-mySpace/collectipy-data/beta_1/u_critical_v_sweep"

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------
U_MIN=1.0
U_MAX=100.0
NUM_U_STEPS=75            # log-spaced → ratio ≈ 1.064 (6.4 %/step)
V_VALUES="${V_VALUES:-0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0}"
RUNS_PER_U="${RUNS_PER_U:-100}"       # replicates per (v, u)
# 50 replicates per task keeps the whole sweep to 1500 job records; at 10 it is
# 7500 and slurmctld starts refusing submissions (see the note in the header).
RUNS_PER_TASK="${RUNS_PER_TASK:-50}"
MAX_CONCURRENT="${MAX_CONCURRENT:-50}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SUBMIT_DELAY="${SUBMIT_DELAY:-10}"
SUBMIT_RETRIES="${SUBMIT_RETRIES:-5}"
CHAIN="${CHAIN:-0}"
DRY_RUN="${DRY_RUN:-0}"

BASE_PATH_ROOT="${BASE_PATH_ROOT:-${LOGS_DIR}}"

if (( RUNS_PER_U % RUNS_PER_TASK != 0 )); then
    echo "RUNS_PER_U (${RUNS_PER_U}) must be a multiple of RUNS_PER_TASK (${RUNS_PER_TASK})." >&2
    exit 1
fi

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
# Generate u values (deterministic — same result in every task)
# ---------------------------------------------------------------------------
read -ra U_VALUES <<< "$("$PYTHON_BIN" - <<PYEOF
import numpy as np
vals = np.round(np.logspace(
    np.log10(float("$U_MIN")),
    np.log10(float("$U_MAX")),
    int("$NUM_U_STEPS")
), 6)
print(" ".join(str(v) for v in vals))
PYEOF
)"

read -ra V_LIST <<< "$V_VALUES"

N_U="${#U_VALUES[@]}"
N_V="${#V_LIST[@]}"
BATCHES_PER_U=$(( RUNS_PER_U / RUNS_PER_TASK ))
TASKS_PER_V=$(( N_U * BATCHES_PER_U ))

# ---------------------------------------------------------------------------
# SUBMISSION MODE — one array per v value
# ---------------------------------------------------------------------------
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "Critical-u sweep across v — bump formation threshold (no targets)"
    echo "  u range           : ${U_MIN} → ${U_MAX}"
    echo "  num_u_steps       : ${N_U}"
    echo "  step ratio        : $("$PYTHON_BIN" -c "print(f'{(${U_MAX}/${U_MIN})**(1/(${N_U}-1)):.4f}')")"
    echo "  v values          : ${V_LIST[*]}  (${N_V} arrays)"
    echo "  runs_per_u        : ${RUNS_PER_U}"
    echo "  runs_per_task     : ${RUNS_PER_TASK}"
    echo "  tasks per v-array : ${TASKS_PER_V}"
    echo "  total array tasks : $(( TASKS_PER_V * N_V ))"
    echo "  total runs        : $(( N_U * N_V * RUNS_PER_U ))"
    echo "  est. per task     : ~$(( RUNS_PER_TASK * 23 / 60 )) min (wall limit is 24 h)"
    echo "  submit delay      : ${SUBMIT_DELAY}s between arrays, chain=${CHAIN}"
    echo "  skip existing     : ${SKIP_EXISTING}"
    echo "  results base path : ${BASE_PATH_ROOT}"
    echo ""
    echo "u values:"
    echo "  ${U_VALUES[*]}"
    echo ""

    if [ ! -f "$CONFIG_TEMPLATE" ]; then
        echo "Config template not found: ${CONFIG_TEMPLATE}" >&2
        exit 1
    fi

    if [ "$DRY_RUN" != "1" ]; then
        mkdir -p "$LOGS_DIR"
    fi

    SUBMITTED=()
    PREV_JOB=""
    for V_IDX in "${!V_LIST[@]}"; do
        V_VALUE="${V_LIST[$V_IDX]}"
        SBATCH_ARGS=(
            --parsable
            --job-name="u_crit_v${V_VALUE}"
            --array="0-$((TASKS_PER_V - 1))%${MAX_CONCURRENT}"
            --output="${LOGS_DIR}/u_critical_v${V_VALUE}_%A_%a.out"
            --error="${LOGS_DIR}/u_critical_v${V_VALUE}_%A_%a.err"
            --export="ALL,V_VALUE=${V_VALUE},BASE_PATH_ROOT=${BASE_PATH_ROOT},PROJECT_DIR=${PROJECT_DIR},RUNS_PER_U=${RUNS_PER_U},RUNS_PER_TASK=${RUNS_PER_TASK},SKIP_EXISTING=${SKIP_EXISTING}"
        )
        # Each v-array starts only once the previous one has left the queue, so
        # the controller never holds more than one array's worth of runnable work.
        if [ "$CHAIN" = "1" ] && [ -n "$PREV_JOB" ]; then
            SBATCH_ARGS+=( --dependency="afterany:${PREV_JOB}" )
        fi
        SBATCH_ARGS+=( "$0" )

        if [ "$DRY_RUN" = "1" ]; then
            echo "sbatch ${SBATCH_ARGS[*]}"
            PREV_JOB="DRYRUN$((V_IDX + 1))"
            continue
        fi

        # sbatch retries the "temporarily unable to accept job" case itself, but
        # gives up eventually; back off and try again rather than losing this v.
        JOB_ID=""
        for ATTEMPT in $(seq 1 "$SUBMIT_RETRIES"); do
            if OUT="$(sbatch "${SBATCH_ARGS[@]}" 2>&1)"; then
                JOB_ID="${OUT%%;*}"
                break
            fi
            echo "  v=${V_VALUE}: sbatch failed (attempt ${ATTEMPT}/${SUBMIT_RETRIES}): ${OUT}" >&2
            sleep $(( SUBMIT_DELAY * ATTEMPT * 2 ))
        done

        if [ -z "$JOB_ID" ]; then
            echo "" >&2
            echo "giving up on v=${V_VALUE} after ${SUBMIT_RETRIES} attempts." >&2
            echo "submitted so far: ${SUBMITTED[*]:-none}" >&2
            REMAINING=("${V_LIST[@]:$V_IDX}")
            echo "once the controller settles, submit the rest with:" >&2
            echo "  V_VALUES=\"${REMAINING[*]}\" bash $0" >&2
            echo "(SKIP_EXISTING=1 makes re-running a partially finished v harmless.)" >&2
            exit 1
        fi

        SUBMITTED+=("v=${V_VALUE}:${JOB_ID}")
        echo "  submitted v=${V_VALUE} as job ${JOB_ID}"
        if [ "$V_IDX" -lt "$(( N_V - 1 ))" ]; then
            sleep "$SUBMIT_DELAY"
        fi
    done

    if [ "$DRY_RUN" != "1" ]; then
        echo ""
        echo "submitted ${#SUBMITTED[@]} arrays: ${SUBMITTED[*]:-none}"
    fi
    exit 0
fi

# ---------------------------------------------------------------------------
# EXECUTION MODE — running inside a SLURM array task
# ---------------------------------------------------------------------------

# Load bwUniCluster3 Python module if the venv Python is not directly available.
# module load devel/python/3.10.12_gnu_12.2

if [ -z "${V_VALUE:-}" ]; then
    echo "V_VALUE not set — the array task was not submitted by this script." >&2
    exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID}"

U_IDX=$(( TASK_ID / BATCHES_PER_U ))
BATCH_IDX=$(( TASK_ID % BATCHES_PER_U ))
FIRST_RUN=$(( BATCH_IDX * RUNS_PER_TASK + 1 ))
LAST_RUN=$(( FIRST_RUN + RUNS_PER_TASK - 1 ))
U_VALUE="${U_VALUES[$U_IDX]}"

echo "[task ${TASK_ID}] v=${V_VALUE}  u=${U_VALUE}  replicates=${FIRST_RUN}–${LAST_RUN}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

for RUN_ID in $(seq "$FIRST_RUN" "$LAST_RUN"); do
    OUT_DIR="${BASE_PATH_ROOT}/v_${V_VALUE}/u_${U_VALUE}/replicate_${RUN_ID}"

    if [ "$SKIP_EXISTING" = "1" ] && [ -f "${OUT_DIR}/config_folder_0/run_1.zip" ]; then
        echo "  [task ${TASK_ID}] v=${V_VALUE} u=${U_VALUE} replicate=${RUN_ID} — already done, skipping"
        continue
    fi

    CONFIG_OUT="$TMP_DIR/u_critical_v${V_VALUE}_u${U_VALUE}_run${RUN_ID}.json"

    "$PYTHON_BIN" - <<PYEOF
import json, os, hashlib

template_path  = "$CONFIG_TEMPLATE"
out_path       = "$CONFIG_OUT"
u_value        = "$U_VALUE"
v_value        = "$V_VALUE"
run_id         = "$RUN_ID"
base_path_root = "$BASE_PATH_ROOT"

with open(template_path, "r", encoding="utf-8") as f:
    data = json.load(f)

try:
    u_float = float(u_value)
    v_float = float(v_value)
    run_int = int(run_id)
except ValueError as exc:
    raise SystemExit(f"Invalid values: v={v_value}, u={u_value}, run={run_id}") from exc

# v is part of the seed key, so the same replicate index at different v draws a
# different noise realisation — the v series stay statistically independent.
seed_key    = f"{v_value}_{u_value}_{run_id}"
random_seed = int(hashlib.md5(seed_key.encode()).hexdigest(), 16) % (2**31)

env = data.setdefault("environment", {})

for arena in env.get("arenas", {}).values():
    if isinstance(arena, dict):
        arena["random_seed"] = random_seed

# Set u and v for all agents with a mean_field_model block.
for agent_cfg in env.get("agents", {}).values():
    if isinstance(agent_cfg, dict):
        mf = agent_cfg.get("mean_field_model")
        if isinstance(mf, dict):
            mf["u"] = u_float
            mf["v"] = v_float

env["num_runs"] = 1

results = env.setdefault("results", {})
if isinstance(results, dict):
    results["base_path"] = os.path.join(
        base_path_root,
        f"v_{v_value}",
        f"u_{u_value}",
        f"replicate_{run_int}",
    )

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)
PYEOF

    echo "  [task ${TASK_ID}] v=${V_VALUE}  u=${U_VALUE}  replicate=${RUN_ID}"
    "$PYTHON_BIN" "$PROJECT_DIR/src/main.py" -c "$CONFIG_OUT" > /dev/null
done
