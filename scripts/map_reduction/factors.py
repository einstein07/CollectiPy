# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Factors and locked parameters of the sensory-map reduction experiment — ONE place.

max-sensory-map-spec.md Section 6. Everything downstream (config patching, the
manifest, the runner, the SLURM geometry, aggregation, plots) imports from here and
nothing re-declares a value.

Design (6.3): reduction {sum, max} x Delta_close (7) x q_C (3) = 42 cells, paired
seeds across ALL factors (the same seed list in every cell, used for both the arena
seed and the shared percept stream, so trial i sees the same noise everywhere).

Geometry (6.1): three targets at range TARGET_RANGE from the start pose. Bearings
relative to A: A at 0, B at +Delta_close, C at -120 degrees. Positions follow the
simulator's bearing convention (GPS: bearing = atan2(-dy, dx) - heading, i.e. the
plant is y-down), so a target at bearing b sits at (r cos b, -r sin b).

Decisions recorded here rather than assumed (see README.md, "Decisions"):

  * QUALITY_SCALE. The spec states qualities relative to A/B (q_A = q_B = 1.00,
    q_C in {1.00, 1.01, 1.02}); the campaign this experiment sits next to expresses
    the same 1 % / 2 % differences on a strength scale of 5.0 (5.0 vs 4.95), and its
    sensory noise (white_rate) is calibrated on that scale. Config strengths are
    therefore QUALITY_SCALE * q_rel, so the ring sees the campaign's input amplitude
    and signal-to-noise ratio. Section 5's unit tests use q = 1 as written.
  * SIGMA = 0.1 as the spec's 6.2 fixes it. The RA templates in config/ carry 1.5
    (a documented repo/spec drift, scripts/uhat_v_sweep/RECON.md D-02); the 6.6
    sanity check runs BOTH values, the template's for comparability with the
    existing 60-degree results and the spec's for the design above.
  * T_MAX_TICKS = 100: ten times the direct travel time to a target at 0.05 m per
    tick, the horizon the (u_hat, v) sweep used. A trial that has not arrived by
    then is a timeout, which 6.4 counts as an outcome.
"""

from __future__ import annotations

import math

SWEEP = "map_reduction"
SPEC = "max-sensory-map-spec.md"

# ---------------------------------------------------------------------------
# Factors (6.3) — the Cartesian product of these is the whole design.
# ---------------------------------------------------------------------------
REDUCTIONS = ["sum", "max"]
DELTA_CLOSE_DEG = [12, 15, 20, 25, 30, 40, 60]
Q_C_REL = [1.00, 1.01, 1.02]

# ---------------------------------------------------------------------------
# Geometry (6.1)
# ---------------------------------------------------------------------------
Q_AB_REL = 1.00
QUALITY_SCALE = 5.0            # config strength = QUALITY_SCALE * relative quality
TARGET_RANGE = 0.5             # metres from the start pose, every target
#: Square arena side for the three-target design. The template's unit square puts
#: A (bearing 0, range 0.5) exactly on the wall, so an agent parked between A and
#: B would be clamped by the wall rather than by the map: side 2 (as the DDM
#: campaign used for wide placements) takes the wall out of the dynamics. The 6.6
#: check keeps the template's arena for comparability with the existing results.
ARENA_SIDE = 2.0
AGENT_DIAMETER = 0.033         # template value; the clamp box is side/2 - diameter/2
BEARING_A_DEG = 0.0
BEARING_C_DEG = -120.0
LABELS = ("A", "B", "C")
TARGET_IDS = {"A": "static_0.s#0", "B": "static_1.s#0", "C": "static_2.s#0"}
ID_TO_LABEL = {v: k for k, v in TARGET_IDS.items()}
TARGET_COLORS = {"A": "green", "B": "green", "C": "red"}   # GUI only

# ---------------------------------------------------------------------------
# Trials, seeds, horizon
# ---------------------------------------------------------------------------
N_TRIALS_SMOKE = 20            # 6.3: smoke run, locally
N_TRIALS_FULL = 200            # 6.3: full run, SLURM array
CHUNK = 200                    # trials per array task (default: a whole cell)
#: Paired design: seed_i = BASE_SEED + i, the SAME list in every cell, for both the
#: arena `random_seed` (the ring's internal sigma noise) and `sensory_stream.seed`
#: (the shared percept). 4.5: trial identity is independent of the swept factors.
BASE_SEED = 20260911
T_MAX_TICKS = 100

# ---------------------------------------------------------------------------
# Fixed settings (6.2) — asserted against the patched config at run time.
# ---------------------------------------------------------------------------
BASE_CONFIG = "config/qd_sweep_ra_template.json"   # the current campaign's RA arm
NUM_NEURONS = 30
U = 6.0                        # absolute u only
BETA = 1.0
V = 0.5
KAPPA = 20
INTEGRATION_TIME = 50.0
INTEGRATION_DT = 0.1
SIGMA = 0.1                    # spec 6.2 (template value is 1.5; see module docstring)
SIGMA_S = 0.0                  # shared stream owns the sensory noise
LINEAR_VELOCITY = 0.05
ANGULAR_VELOCITY = 120         # deg/s: the plant must turn inside the arrival radius
TICKS_PER_SECOND = 1
ARRIVAL_RADIUS = 0.05
USE_THRESHOLDING = False
SCALE_VELOCITY = False
SENSORY_STREAM = {"mode": "shared", "frozen_sd": 0.0, "white_rate": 0.07071068}
#: The behavioural detector's alignment tolerance, reused by the post-hoc
#: midpoint-aware commitment classification (6.4) so both use one yardstick.
ALIGNMENT_TOL_DEG = 5.0
ALIGNMENT_CONSECUTIVE_TICKS = 1

#: DDM accumulator parameters that must never appear in a ring-attractor config.
PROHIBITED_KEYS = ("eta_rate", "lambda_t")

# ---------------------------------------------------------------------------
# 6.6 sanity check on the prior campaigns: the two-target standard condition.
# ---------------------------------------------------------------------------
SANITY_N_TRIALS = 200
SANITY_DELTA_DEG = 60
SANITY_QUALITIES = (5.0, 4.95)         # static_0 (correct), static_1: dQ = 1 %
SANITY_CORRECT_ID = "static_0.s#0"
#: Two runtime variants: the template's own sigma (what the existing results were
#: produced with) and the spec's 6.2 sigma (what the design above runs at).
SANITY_VARIANTS = {"campaign": None, "spec": SIGMA}   # None = inherit the template

# ---------------------------------------------------------------------------
# Smoke cells (a 2x2 slice, still with all seeds paired)
# ---------------------------------------------------------------------------
SMOKE_TRIALS = N_TRIALS_SMOKE


def n_cells() -> int:
    return len(REDUCTIONS) * len(DELTA_CLOSE_DEG) * len(Q_C_REL)


N_CELLS = n_cells()


def cell_id(reduction: str, delta_deg: float, q_c_rel: float) -> int:
    """Stable cell index: reduction-major, then Delta_close, then q_C."""
    r = REDUCTIONS.index(reduction)
    d = DELTA_CLOSE_DEG.index(int(delta_deg))
    q = Q_C_REL.index(float(q_c_rel))
    return (r * len(DELTA_CLOSE_DEG) + d) * len(Q_C_REL) + q


def iter_cells():
    """Yield {cell_id, reduction, delta_close_deg, q_c_rel} in canonical order."""
    k = 0
    for reduction in REDUCTIONS:
        for delta in DELTA_CLOSE_DEG:
            for q_c in Q_C_REL:
                yield {"cell_id": k, "reduction": reduction,
                       "delta_close_deg": int(delta), "q_c_rel": float(q_c)}
                k += 1


def bearings_deg(delta_close_deg: float) -> dict[str, float]:
    """Egocentric bearings of A, B, C at the start pose, degrees."""
    return {"A": BEARING_A_DEG, "B": BEARING_A_DEG + float(delta_close_deg),
            "C": BEARING_C_DEG}


def position_for_bearing(bearing_deg: float, r: float = TARGET_RANGE) -> list[float]:
    """Arena position of a target seen at `bearing_deg` from the origin, heading 0.

    GPS bearing = atan2(-dy, dx): the y axis is flipped, so +bearing is -y.
    """
    b = math.radians(bearing_deg)
    return [round(r * math.cos(b), 6), round(-r * math.sin(b), 6), 0.0]


def strengths(q_c_rel: float) -> dict[str, float]:
    return {"A": QUALITY_SCALE * Q_AB_REL, "B": QUALITY_SCALE * Q_AB_REL,
            "C": QUALITY_SCALE * float(q_c_rel)}


def merge_threshold_deg(kappa: float = KAPPA) -> float:
    """Delta* of Section 2.2: kappa sin^2(D/2) = cos(D/2)  ->  ~25.5 deg at kappa 20."""
    c = (math.sqrt(1.0 + 4.0 * kappa * kappa) - 1.0) / (2.0 * kappa)
    return math.degrees(2.0 * math.acos(c))


def seed_for(trial_idx: int) -> int:
    """The paired seed list: identical in every cell and in the sanity check."""
    return int(BASE_SEED) + int(trial_idx)
