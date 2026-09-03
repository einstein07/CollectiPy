#!/bin/bash
# =============================================================================
# u-sensitivity sweep WITH SFA — asymmetric target quality × neural gain
# bwUniCluster3.0 SLURM job array
#
# Purpose: the standard two-target decision experiment (no target swap) run
# with spike-frequency adaptation enabled, i.e. the SFA counterpart of
# submit-mean-sweep-u-sensitivity-bwunicluster.sh.  For each neural gain u the
# arena is asymmetric: static_0 is held fixed at strength 5.0 while static_1 is
# weakened by a quality difference δ ∈ [0%, 80%].  The 22 δ values are:
#   • δ = 0   (symmetric arena, identical options)
#   • δ = 1% … 80%, 21 points log-spaced → equal fractional steps in strength
#     ratio, giving uniform resolution near the symmetric point where dynamics
#     are most sensitive to small asymmetries.
#
# SFA parameters come from the template and are NOT swept here:
#   g_adapt   = 0.8
#   tau_adapt = 3.2
# (they are the fixed operating point of the SFA campaign — the sweep over them
#  lives in submit-mean-flexibility-sfa-adapt-sweep-bwunicluster.sh).
#
# u values (7.5, 7.8, 10) match submit-mean-flexibility-sweep-sfa.sh, so this
# no-swap sweep is directly comparable to the SFA flexibility campaign.  They
# differ from the non-SFA sweep's (5, 6.156868, 8) because adaptation shifts
# the bifurcation point upwards.
#
# Derived config values:
#   strength_static_0 = 5.0  (fixed)
#   strength_static_1 = 5.0 × (1 − δ)
#
# Array sizing: 3 u × 22 δ × 10 batches (10 runs/batch) = 660 tasks, 6 600 runs
#
# Usage (from login node):
#   bash submit-mean-sweep-u-sensitivity-sfa-bwunicluster.sh
#
# The script submits itself as a SLURM array; no separate job file needed.
# Each array task derives its (u, diff, run_id) tuple from SLURM_ARRAY_TASK_ID.
# =============================================================================

# --- SLURM directives (only active when submitted via sbatch) ----------------
#SBATCH --job-name=u_sensitivity_sweep_sfa
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
# --output and --error are set dynamically via sbatch flags below.
# -----------------------------------------------------------------------------

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR="/home/kn/kn_kn/kn_pop547841/CollectiPy"
VENV_BIN="$PROJECT_DIR/.venv/bin"
CONFIG_TEMPLATE="$PROJECT_DIR/config/mean_field_2_targets_sfa.json"
LOGS_DIR="/pfs/work9/workspace/scratch/kn_pop547841-mySpace/collectipy-data/beta_1/u_sensitivity_sweep_sfa"
# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------
# Three neural gain values for sensitivity analysis under SFA
U_VALUES=(7.5 7.8 10)

FIXED_STRENGTH=5.0        # static_0 strength — never changes
NUM_DIFF_STEPS=22         # 1 symmetric point + 21 log-spaced asymmetric points
DIFF_MIN=0.01             # 1%  — smallest non-zero quality difference
DIFF_MAX=0.80             # 80% — largest quality difference
RUNS_PER_DIFF=100         # replicates per (u, diff) configuration
RUNS_PER_TASK=10          # replicates packed into each array task (keeps array ≤ 1000)

BASE_PATH_ROOT="${BASE_PATH_ROOT:-${LOGS_DIR}}"

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
# Generate quality-difference values (deterministic — same result every task)
# diff=0.0 is the symmetric point; the remaining 21 are log-spaced 1%→80%.
# ---------------------------------------------------------------------------
read -ra DIFF_VALUES <<< "$("$PYTHON_BIN" - <<PYEOF
import numpy as np
log_pts = np.round(np.logspace(
    np.log10(float("$DIFF_MIN")),
    np.log10(float("$DIFF_MAX")),
    int("$NUM_DIFF_STEPS") - 1
), 6)
all_pts = [0.0] + log_pts.tolist()
print(" ".join(str(v) for v in all_pts))
PYEOF
)"

N_U="${#U_VALUES[@]}"
N_DIFFS="${#DIFF_VALUES[@]}"
BATCHES_PER_DIFF=$(( RUNS_PER_DIFF / RUNS_PER_TASK ))
TASKS_PER_U=$(( N_DIFFS * BATCHES_PER_DIFF ))
TOTAL_TASKS=$(( N_U * TASKS_PER_U ))

# ---------------------------------------------------------------------------
# SUBMISSION MODE
# ---------------------------------------------------------------------------
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "u-sensitivity sweep WITH SFA — asymmetric arena, varying quality difference, no swap"
    echo "  u values          : ${U_VALUES[*]}"
    echo "  SFA               : g_adapt / tau_adapt fixed by template (0.8 / 3.2)"
    echo "  fixed strength    : ${FIXED_STRENGTH} (static_0)"
    echo "  quality diff range: 0% (symmetric) → $(echo "${DIFF_MAX}*100" | bc)%"
    echo "  num diff steps    : ${N_DIFFS}  (0% + 21 log-spaced $(echo "${DIFF_MIN}*100"|bc)%→$(echo "${DIFF_MAX}*100"|bc)%)"
    echo "  runs_per_diff     : ${RUNS_PER_DIFF}"
    echo "  runs_per_task     : ${RUNS_PER_TASK}"
    echo "  tasks per u value : ${TASKS_PER_U}"
    echo "  total array tasks : ${TOTAL_TASKS}"
    echo "  total runs        : $(( N_U * N_DIFFS * RUNS_PER_DIFF ))"
    echo "  config template   : ${CONFIG_TEMPLATE}"
    echo "  results base path : ${BASE_PATH_ROOT}"
    echo ""
    echo "Quality difference values (δ):"
    echo "  ${DIFF_VALUES[*]}"
    echo ""

    mkdir -p "$LOGS_DIR"

    sbatch \
        --array="0-$((TOTAL_TASKS - 1))%100" \
        --output="${LOGS_DIR}/u_sensitivity_sfa_%A_%a.out" \
        --error="${LOGS_DIR}/u_sensitivity_sfa_%A_%a.err" \
        --export=ALL,BASE_PATH_ROOT="$BASE_PATH_ROOT",PROJECT_DIR="$PROJECT_DIR" \
        "$0"
    exit 0
fi

# ---------------------------------------------------------------------------
# EXECUTION MODE — running inside a SLURM array task
# ---------------------------------------------------------------------------

# Load bwUniCluster3 Python module if the venv Python is not directly available.
# module load devel/python/3.10.12_gnu_12.2

# Decompose TASK_ID into (u_idx, diff_idx, batch_idx).
TASK_ID="${SLURM_ARRAY_TASK_ID}"
BATCHES_PER_DIFF=$(( RUNS_PER_DIFF / RUNS_PER_TASK ))
TASKS_PER_U=$(( N_DIFFS * BATCHES_PER_DIFF ))

U_IDX=$(( TASK_ID / TASKS_PER_U ))
REMAINDER=$(( TASK_ID % TASKS_PER_U ))
DIFF_IDX=$(( REMAINDER / BATCHES_PER_DIFF ))
BATCH_IDX=$(( REMAINDER % BATCHES_PER_DIFF ))

FIRST_RUN=$(( BATCH_IDX * RUNS_PER_TASK + 1 ))
LAST_RUN=$(( FIRST_RUN + RUNS_PER_TASK - 1 ))

U_VALUE="${U_VALUES[$U_IDX]}"
DIFF="${DIFF_VALUES[$DIFF_IDX]}"

echo "[task ${TASK_ID}] u=${U_VALUE}  diff=${DIFF}  replicates=${FIRST_RUN}–${LAST_RUN}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

for RUN_ID in $(seq "$FIRST_RUN" "$LAST_RUN"); do
    CONFIG_OUT="$TMP_DIR/u_sensitivity_sfa_u${U_VALUE}_diff${DIFF}_run${RUN_ID}.json"

    "$PYTHON_BIN" - <<PYEOF
import json, os, hashlib

template_path  = "$CONFIG_TEMPLATE"
out_path       = "$CONFIG_OUT"
u_value        = "$U_VALUE"
diff_value     = "$DIFF"
fixed_strength = "$FIXED_STRENGTH"
run_id         = "$RUN_ID"
base_path_root = "$BASE_PATH_ROOT"

with open(template_path, "r", encoding="utf-8") as f:
    data = json.load(f)

try:
    u_float      = float(u_value)
    diff_float   = float(diff_value)
    fixed_float  = float(fixed_strength)
    run_int      = int(run_id)
except ValueError as exc:
    raise SystemExit(f"Invalid parameter values: u={u_value}, diff={diff_value}, run={run_id}") from exc

# Deterministic per-run seed derived from all sweep parameters.
seed_key    = f"{u_value}_{diff_value}_{run_id}"
random_seed = int(hashlib.md5(seed_key.encode()).hexdigest(), 16) % (2**31)

env = data.setdefault("environment", {})

# This is the plain two-target experiment: no target swap of any kind, whatever
# the template happens to carry over from the flexibility configs.
env.pop("target_position_swaps", None)
env.pop("post_bifurcation_swap", None)

# Seed all arenas.
for arena in env.get("arenas", {}).values():
    if isinstance(arena, dict):
        arena["random_seed"] = random_seed

# static_0 fixed at FIXED_STRENGTH; static_1 weakened by quality difference δ.
objects = env.get("objects", {})

target_0 = objects.get("static_0")
if not isinstance(target_0, dict):
    raise SystemExit("Object 'static_0' not found in config.")
target_0["strength"] = [fixed_float]

target_1 = objects.get("static_1")
if not isinstance(target_1, dict):
    raise SystemExit("Object 'static_1' not found in config.")
target_1["strength"] = [round(fixed_float * (1.0 - diff_float), 8)]

# Set u for all agents that have a mean_field_model block. SFA parameters
# (g_adapt, tau_adapt) are left at their template values.
agents = env.get("agents", {})
for agent_cfg in agents.values():
    if isinstance(agent_cfg, dict):
        mf = agent_cfg.get("mean_field_model")
        if isinstance(mf, dict):
            mf["u"] = u_float

env["num_runs"] = 1

results = env.setdefault("results", {})
if isinstance(results, dict):
    diff_pct_label = f"{round(diff_float * 100, 4):.4f}pct"
    results["base_path"] = os.path.join(
        base_path_root,
        f"u_{u_value}",
        f"diff_{diff_pct_label}",
        f"replicate_{run_int}",
    )

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)
PYEOF

    echo "  [task ${TASK_ID}] u=${U_VALUE}  diff=${DIFF}  replicate=${RUN_ID}"
    "$PYTHON_BIN" "$PROJECT_DIR/src/main.py" -c "$CONFIG_OUT" > /dev/null
done
