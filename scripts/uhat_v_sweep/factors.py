# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""The (u_hat, v) factorial's factors and locked parameters — defined in ONE place.

Implements Sections 2 and 3 of `uhat-v-factorial-experiment.md`. Everything
downstream (manifest generation, the runner, the SLURM geometry, aggregation and
analysis) imports from here; nothing re-declares a value.

The design is a FULL CROSSING of relative coupling u_hat = u / u*(v) with kernel
shape v. `u*(v)` is never hand-entered: it is computed in `generate_manifest.py`
from the simulator's own kernel builder, and only the manifest carries it.

Fixed configuration is INHERITED from the RA arm of the RA-DDM comparison, i.e.
`config/mean_field_2_targets_no_viz.json` patched exactly as
`submit-ra-frontier-sweep-bwunicluster.sh` patches it. The patch lives in
`config_patch.py`; the values it is allowed to touch are listed below. See
`RECON.md` for every point where this file departs from the spec document
(sigma, use_thresholding and the scoring vocabulary all do).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Factors (Section 2). The Cartesian product of these is the whole design.
# ---------------------------------------------------------------------------

#: Relative coupling. Denser near the knee at 1.0; 0.65 near the suspected
#: accuracy peak; 0.50 is the deepest previously observed sub-critical point;
#: 1.50 probes super-critical saturation. Do not change without asking.
U_HAT_GRID = [0.50, 0.65, 0.80, 0.90, 1.00, 1.10, 1.25, 1.50]

#: Kernel shape. Never thinned: it is the axis the residual effect lives on.
V_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

#: Trials per cell. Binomial 95 % CI half-width is +-0.035 at p = 0.85.
#: If compute-constrained: drop to 300 first, then drop u_hat in {0.90, 1.10}.
N_TRIALS = 400

#: Paired design (common random numbers): seed_i = BASE_SEED + i, the SAME list
#: in every cell. Used for BOTH the arena `random_seed` (which seeds the model's
#: internal sigma noise) and `sensory_stream.seed` (the shared percept stream), so
#: two cells differing only in (u, v) see the identical sensory realisation.
#:
#: This deliberately departs from `campaign/seeds.py`'s blake2b derivation: that
#: scheme keys the seed on the factor levels, which is the opposite of what a
#: paired contrast across cells needs. See RECON.md item 6.
BASE_SEED = 20260828

#: Section 2. A trial that has not arrived by then is `decided = False`; it is
#: data, not an exclusion. Maps to `environment.time_limit` because the arena
#: computes `ticks_limit = time_limit * ticks_per_second + 1` at 1 tick/second.
T_MAX_TICKS = 100

# ---------------------------------------------------------------------------
# Fixed configuration (Section 3) — inherited, never varied between cells.
# Only `u` and `v` differ from cell to cell.
# ---------------------------------------------------------------------------

#: The RA-arm template of the RA-DDM comparison.
BASE_CONFIG = "config/mean_field_2_targets_no_viz.json"

#: Ring geometry and dynamics, asserted against the patched config at run time.
NUM_NEURONS = 30
BETA = 1.0
KAPPA = 20
INTEGRATION_TIME = 50.0
INTEGRATION_DT = 0.1          # 500 Euler substeps per control tick
G_THRESHOLD = 0.6             # inert while use_thresholding is False; see RECON.md
USE_THRESHOLDING = False      # REPO VALUE. The spec says "thresholded readout".
SIGMA = 1.5                   # REPO VALUE. The spec says 0.1.
SCALE_VELOCITY = False        # constant forward speed; the readout only steers

#: Task geometry and plant. Out of scope to change (Section 13).
LINEAR_VELOCITY = 0.05
#: deg/s. NOT the template value (10) — see RECON.md item 7. The runs that
#: produced the Section 12 anchor (accuracy 0.765, DT 11 ticks) were made at 120,
#: as every archived config under
#: seoul-data/beta-1/ra_ddm_frontier_sweep/u_*/v_*/replicate_*/config_folder_0/
#: records. At 10 deg/s the minimum turn radius is v/omega = 0.286 m against a
#: 0.05 m termination radius, so the agent ORBITS: measured here, all 12 smoke
#: trials arrived at tick 50-51 instead of 11. `config_patch` therefore SETS this
#: value rather than inheriting it, and asserts it after patching.
ANGULAR_VELOCITY = 120
TICKS_PER_SECOND = 1
ARRIVAL_RADIUS = 0.05         # environment.termination.radius
QUALITY_CORRECT = 5.0         # static_0
QUALITY_DISTRACTOR = 4.95     # static_1  ->  d_gamma = 1 %
CORRECT_TARGET_ID = "static_0.s#0"
TARGET_IDS = ["static_0.s#0", "static_1.s#0"]

#: Campaign-wide matched sensory front end. Any drift is fatal in `config_patch`.
#: `seed` is the ONLY member set per trial (the frontier sweep left it null and
#: inherited the arena seed; setting it explicitly to the same value is
#: behaviourally identical and makes the pairing auditable). See RECON.md item 6.
SENSORY_STREAM_MODE = "shared"
SENSORY_STREAM_FROZEN_SD = 0.0
SENSORY_STREAM_WHITE_RATE = 0.035
SIGMA_S = 0.0                 # forced to 0 under shared mode by percept_stream.py

# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

#: Section 5 hard gate on the u* computation.
ANCHOR_V = 0.5
ANCHOR_U_STAR = 6.157
ANCHOR_TOL = 0.01             # relative

#: Section 5: flag prominently if the grid's largest u exceeds this.
MAX_U_WARN = 35.0

#: Section 6: a trial is a numerical failure at NaN/Inf, or beyond this.
MAX_ABS_STATE = 1.0e3

#: Section 11 — the three stiffest cells for the step-halving check.
STEP_HALVING_CELLS = [(0.1, 1.50), (0.1, 1.25), (0.2, 1.50)]
STEP_HALVING_TRIALS = 50
STEP_HALVING_DT = 0.05

#: Section 12 — the smoke cells. Statistics are meaningless at this n; the point
#: is that the pipeline runs end to end.
SMOKE_CELLS = [(0.5, 0.90), (0.5, 1.10)]
SMOKE_TRIALS = 12

#: Section 12 — expectation check for the FULL run, not the smoke run.
EXPECTED_ANCHOR_CELL = (0.5, 1.00)
EXPECTED_ACC = 0.765
EXPECTED_DT_TICKS = 11.0

#: Section 3 — DDM accumulator parameters that must never appear in an RA config.
PROHIBITED_KEYS = ("eta_rate", "lambda_t")


def cell_id(v: float, u_hat: float) -> int:
    """Stable cell index: v-major, u_hat-minor, matching `iter_cells()`."""
    return V_GRID.index(v) * len(U_HAT_GRID) + U_HAT_GRID.index(u_hat)


def iter_cells():
    """Yield (cell_id, v, u_hat) for all 80 cells in the canonical order."""
    k = 0
    for v in V_GRID:
        for u_hat in U_HAT_GRID:
            yield k, v, u_hat
            k += 1


N_CELLS = len(V_GRID) * len(U_HAT_GRID)
