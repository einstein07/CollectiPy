#!/bin/bash
# =============================================================================
# QD sweep at fixed noise — one-shot preflight (README steps 2-4, §10.1-10.3)
#
#   bash scripts/qd_sweep_fixed_noise/preflight.sh              # steps 2-4
#   bash scripts/qd_sweep_fixed_noise/preflight.sh --dt-check   # + step 5
#
# Runs, in order, STOPPING AT THE FIRST FAILING GATE:
#   2. generate_manifest.py   manifests + templates + the frozen controllers
#                             (§4 b* cross-check + §2 noise-invariance inside)
#   3. r1_noise_convention.py the R-1 noise-convention gate (BLOCKING, §2)
#   4. smoke (§10.2)          2 RA cells + 1 DDM point x 20 replicates through
#                             the REAL submit-script path, then aggregation of
#                             both arms and the §9 figures
#   5. dt_check.py            only with --dt-check: §3 step-halving at u = 35
#                             (BLOCKING before submission; ~20 min at the
#                             spec's 50 trials/arm — reminded loudly if skipped)
#
# Run this WHERE seoul-data LIVES (the workstation): step 2 re-derives the
# frozen static b* from the halted campaign's swept trials. The cluster never
# generates anything — it receives the shipped manifests (README step 6).
# Everything is idempotent: smoke replicates with .done are skipped, manifests
# that already match are left alone (pass --force to overwrite changed ones).
#
# Flags: --dt-check          append step 5 (spec's 50 trials/arm)
#        --dt-trials N       step-5 trials/arm (200 = the sharper setting)
#        --r1-runs N         R-1 probe runs per actual δ_Q (default 200)
#        --force             forwarded to generate_manifest.py
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$HERE/../.." && pwd)"
cd "$PROJECT_DIR"

PY=""
for c in .venv/bin/python3.12 .venv/bin/python3.10 .venv/bin/python3 \
         .venv/bin/python python3; do
    command -v "$c" >/dev/null 2>&1 && { PY="$c"; break; }
done
[ -n "$PY" ] || { echo "no python" >&2; exit 1; }

R="$PROJECT_DIR/results/qd_sweep_fixed_noise"
S="$R/smoke"
SUBMIT="$HERE/submit-qd-sweep-fixed-noise-bwunicluster.sh"

DT_CHECK=0
DT_TRIALS=50
R1_RUNS=200
GEN_FLAGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --dt-check)  DT_CHECK=1 ;;
        --dt-trials) DT_TRIALS="$2"; shift ;;
        --r1-runs)   R1_RUNS="$2"; shift ;;
        --force)     GEN_FLAGS+=(--force) ;;
        *) echo "unknown flag $1" >&2; exit 1 ;;
    esac
    shift
done

step() { printf '\n=== preflight step %s — %s ===\n' "$1" "$2"; }

# --------------------------------------------------------------- 2. manifests
step 2 "manifests + templates + frozen controllers (blocking checks inside)"
"$PY" "$HERE/generate_manifest.py" "${GEN_FLAGS[@]}"

# --------------------------------------------------------------------- 3. R-1
step 3 "R-1 noise-convention gate (${R1_RUNS} probe runs per actual δ_Q)"
"$PY" "$HERE/r1_noise_convention.py" --runs "$R1_RUNS"

# ------------------------------------------------------------------- 4. smoke
step 4 "smoke: 2 RA cells + 1 DDM point x 20 through the real script path"
# Rows resolved from the manifests by cell id (never hardcoded).
row_of() { awk -F, -v id="$2" 'NR > 1 && $1 == id { print NR - 2; exit }' "$1"; }
RA_ROW_1="$(row_of "$R/ra_manifest.csv" a100_v0.5_u6)"
RA_ROW_2="$(row_of "$R/ra_manifest.csv" a200_v0.5_u6)"
DDM_ROW="$(row_of "$R/ddm_manifest.csv" d100_a200_ce20)"
[ -n "$RA_ROW_1" ] && [ -n "$RA_ROW_2" ] && [ -n "$DDM_ROW" ] || {
    echo "smoke cells not found in the manifests" >&2; exit 1; }

for ROW in "$RA_ROW_1" "$RA_ROW_2"; do
    env PROJECT_DIR="$PROJECT_DIR" RUNS_PER_CELL=20 RUNS_PER_TASK=20 \
        BASE_PATH_ROOT="$S" LOGS_DIR="$S" CAMPAIGN=ra \
        MANIFEST="$R/ra_manifest.csv" TASK_OFFSET=0 SLURM_ARRAY_TASK_ID="$ROW" \
        bash "$SUBMIT"
done
env PROJECT_DIR="$PROJECT_DIR" RUNS_PER_CELL=20 RUNS_PER_TASK=20 \
    BASE_PATH_ROOT="$S" LOGS_DIR="$S" CAMPAIGN=ddm \
    MANIFEST="$R/ddm_manifest.csv" TASK_OFFSET=0 SLURM_ARRAY_TASK_ID="$DDM_ROW" \
    bash "$SUBMIT"

"$PY" "$HERE/aggregate.py" --arm ra  --base-root "$S" --manifest "$R/ra_manifest.csv"
"$PY" "$HERE/aggregate.py" --arm ddm --base-root "$S" --manifest "$R/ddm_manifest.csv"
"$PY" "$HERE/analyze.py" --base-root "$S"
echo "(smoke completeness lists the untouched cells as missing — expected)"

# ------------------------------------------------------------ 5. step-halving
if [ "$DT_CHECK" = "1" ]; then
    step 5 "step-halving at u = 35 (§3, ${DT_TRIALS} trials/arm)"
    "$PY" "$HERE/dt_check.py" --trials "$DT_TRIALS"
fi

# ------------------------------------------------------------------- verdict
printf '\n=== preflight PASSED (steps 2-4%s) ===\n' \
    "$([ "$DT_CHECK" = "1" ] && echo '+5')"
if [ "$DT_CHECK" != "1" ]; then
    if [ -f "$R/dt_check_report.json" ] && \
       grep -q '"check": "PASS"' "$R/dt_check_report.json"; then
        echo "step 5 (step-halving): PASS on record at $R/dt_check_report.json"
    else
        echo "*** STEP 5 NOT RUN AND NO PASSING REPORT ON RECORD ***"
        echo "*** step-halving is BLOCKING before submission (§3):  ***"
        echo "***   bash $HERE/preflight.sh --dt-check              ***"
    fi
fi
echo
echo "next (README step 6, cluster — you run it): ship the manifests, then"
echo "  scp $R/{ra_manifest.csv,ddm_manifest.csv,frozen_controllers.json} <cluster>:<LOGS_DIR>/"
echo "  CAMPAIGN=ddm bash scripts/qd_sweep_fixed_noise/submit-qd-sweep-fixed-noise-bwunicluster.sh"
echo "  bash scripts/qd_sweep_fixed_noise/submit-qd-sweep-fixed-noise-bwunicluster.sh"
