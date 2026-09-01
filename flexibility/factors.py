# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Flexibility campaign factors and locked parameters — defined in exactly ONE place.

Ring attractor at two gains vs. a collapsing-bound DDM, under a mid-trial world
change, swept over option quality difference, with the sensory noise realisation
SHARED across models. Implements FLEXIBILITY_RA_DDM_DESIGN.md; section numbers below
refer to that document.

Everything downstream (matrix, config generation, precompute, preflight, submission)
imports from here; nothing re-declares a value.

Deliberately SEPARATE from `campaign/factors.py`, which belongs to the speed/accuracy
frontier campaign and locks a different operating point. The two campaigns share the
codebase and the config idioms, not their constants.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Arms (Section 1). The ORDER IS THE CONTRACT: the SLURM array indexes into it.
# ---------------------------------------------------------------------------
#: Ring-attractor gains. u = 6.2 sits 0.7% above the tabulated critical coupling for
#: kernel v = 0.5 (u* = 6.156868, from the matched (v, u*) table the RA/DDM frontier
#: sweep uses), so it is NEAR-critical rather than exactly critical. Two things follow
#: that belong in the methods rather than in the data: a result this close to u*
#: depends on how u* was determined (continuum vs. finite-N, and on beta, sigma,
#: kappa), and finite-N effects at num_neurons = 30 are largest exactly here, so the
#: SIMULATED system's effective critical point may sit slightly off the analytical one.
U_NEAR_CRITICAL = 6.2
U_RIGID = 8.0

ARMS = ("ra_u6.2", "ra_u8", "ddm_bellman")

RA_ARMS = ("ra_u6.2", "ra_u8")
DDM_ARMS = ("ddm_bellman",)

#: Which base template each arm patches. Templates are NEVER modified; they are
#: deep-copied and overridden in memory.
TEMPLATES = {
    "ra_u6.2": "config/mean_field_2_targets_flexibility.json",
    "ra_u8": "config/mean_field_2_targets_flexibility.json",
    "ddm_bellman": "config/embodied_ddm_flexibility.json",
}

#: Gain per RA arm.
ARM_U = {"ra_u6.2": U_NEAR_CRITICAL, "ra_u8": U_RIGID}

# ---------------------------------------------------------------------------
# Locked operating point (Section 4). Decided; do not re-derive.
# ---------------------------------------------------------------------------
QUALITY_BETTER = 5.0          # static_0, PINNED; static_1 = QUALITY_BETTER*(1 - delta)
WHITE_RATE = 0.07071068       # sensory eta; c = sqrt(2)*eta = 0.100 exactly
FROZEN_SD = 0.0
TICKS_PER_SECOND = 1          # arena AND agent; equality is a fatal precondition
LINEAR_VELOCITY = 0.01        # m/s, both models
ANGULAR_VELOCITY = 120        # deg/s
TARGET_RANGE = 0.5            # R, metres
ANGULAR_SEP_DEG = 60          # targets at +/- 30 deg
TERMINATION_RADIUS = 0.05
ARENA_SIDE = 2
TIME_LIMIT = 1000             # seconds per run; hitting it = CENSORED, and the
                              # censoring rate is a reported quantity, not a filter
SWAP_DELAY_TICKS = 1          # 1.0 s physical delay at TICKS_PER_SECOND = 1
N_SUB = 16                    # DDM sub-steps per tick
SNAPSHOTS_PER_SECOND = 1

BELLMAN_DT = 1e-3             # solver time step; N_t sized so dt <= this
#: Safety margin on the horizon N_t is sized against. The model measures its own r0
#: at evidence onset, one step in, so the realised T_max sits ~1.6% above the static
#: geometric value; without this the solver dt would land just above 1 ms.
BELLMAN_HORIZON_MARGIN = 1.05

#: The cost of an error, in seconds. FIXED across the campaign.
#:
#: Under `geometric_error_mode: terminal_categorical` the criterion IS the 0-1 loss --
#: every error costs the same, whatever the margin -- expressed in seconds of delay
#: [READ FROM CODE: DriftDiffusionSystem.geometric_costs; _bellman_threshold sets
#: c_e = float(self.cost_ratio)]. So c_e = 1.0 says: being wrong costs exactly as much
#: as one second of delay. Against a 50 s travel budget an error is cheap -- 1/50 of a
#: trial -- which is WHY commitment is fast. That is the criterion doing its job.
COST_OF_ERROR = 1.0

#: Angular cost of delay, c_tau = 1 - cos(Delta/2) at Delta = 60 deg under
#: predecision_motion 'midpoint' [READ FROM CODE: DriftDiffusionSystem
#: .c_tau_linearised]. Restated here so the predicted boundary can be computed without
#: a live model; `flexibility.preflight` asserts it against the live code.
C_TAU = 1.0 - math.cos(math.radians(ANGULAR_SEP_DEG) / 2.0)

#: The DDM's noise scale under the shared stream: c^2 = 2*eta^2.
NOISE_C = math.sqrt(2.0) * WHITE_RATE

#: Must match the base configs' z_min, so predictions floor z as the model does.
Z_MIN = 1e-4

# ---------------------------------------------------------------------------
# Quality-difference grid (Section 2)
# ---------------------------------------------------------------------------
NUM_DIFF_STEPS = 22           # 1 symmetric point + 21 log-spaced, 1% -> 80%
DIFF_MIN = 0.01               # 1%  — the anchor of the original leg
DIFF_MAX = 0.80               # 80%

#: Extra points BELOW DIFF_MIN, continuing the same log ratio downward. The original
#: leg bottomed out at 1%, which is where both ring-attractor arms are still fully
#: rigid and the DDM still reverses on every trial, so the sub-1% region was
#: unresolved. Four more steps at the same 1.2450 ratio take the floor to 0.4163%,
#: keeping the WHOLE leg uniform in log delta rather than grafting on a second
#: spacing. delta = 0 is unaffected and stays RA-only: the DDM cannot deduce |A| from
#: a zero gap, so it has no symmetric cell to add points near.
EXTRA_LOW_STEPS = 4

REPS = 100                    # replicates per (arm, delta)

#: Replicates packed into one array task. This sets the ARRAY SIZE, not the degree of
#: parallelism -- concurrency is the submission throttle (`%N`), which is separate.
#:
#: 25 gives 308 array elements for 7700 runs, where 10 gave 770. Array elements each
#: count as a job toward the cluster's MaxJobCount and toward any per-association
#: MaxSubmitJobs, so a smaller array is easier for a loaded or limited controller to
#: accept -- and submission started failing when the sub-1% points took the array from
#: 650 to 770. Nothing about the science changes: same conditions, same seeds, same
#: replicate count.
#:
#: The cost is restart granularity. A task that dies loses 25 replicates instead of
#: 10, which at ~1 s per replicate (13.4 s worst case) is at most ~6 minutes of rework
#: against a 1 h walltime.
CHUNK = 25


def delta_grid() -> list[float]:
    """The ordered delta grid. ORDER IS THE CONTRACT — the array indexes into it.

    delta = 0 is NOT filler: with equal strengths the exchange is a no-op, so that
    cell measures spontaneous symmetry-breaking and the SPONTANEOUS reversal rate --
    the null distribution every other reversal count is scored against. It is present
    here and dropped per-arm for the DDM by `matrix.deltas_for`.

    Pure-math logspace rather than numpy's, so the grid does not depend on a numpy
    version, rounded to 6 dp exactly as the historical sweep script did so the delta
    tokens match the earlier output layout.
    """
    lo, hi = math.log10(DIFF_MIN), math.log10(DIFF_MAX)
    n = NUM_DIFF_STEPS - 1
    step = (hi - lo) / (n - 1)
    # i < 0 walks the same ratio below DIFF_MIN; i = 0 lands exactly on it.
    log_pts = [round(10.0 ** (lo + i * step), 6)
               for i in range(-EXTRA_LOW_STEPS, n)]
    return [0.0] + log_pts


def strengths_for(delta: float) -> tuple[float, float]:
    """(static_0, static_1) strengths at this delta — PINNED, not mean-preserving.

    static_0 is held at QUALITY_BETTER and static_1 weakened, so A = |dQ| = 5*delta.
    This preserves continuity with the earlier RA sweeps, and carries a known
    consequence recorded in Section 2 of the design: the MEAN drive falls from 5.0 at
    delta = 0 to 3.0 at delta = 0.8. The DDM is blind to that -- it sees only the
    difference -- but the ring attractor is not, because total drive shifts the
    field's operating point and hence the effective distance from the critical
    coupling. At the top of the grid the near-critical arm is being driven
    differently, not merely discriminating a larger difference. `mean_drive` is
    written into every condition's derived record so the confound is reported rather
    than hidden.
    """
    return QUALITY_BETTER, round(QUALITY_BETTER * (1.0 - delta), 8)


def mean_drive(delta: float) -> float:
    """Mean of the two target strengths — falls 5.0 -> 3.0 across the sweep."""
    q0, q1 = strengths_for(delta)
    return 0.5 * (q0 + q1)


# ---------------------------------------------------------------------------
# Seeding (Section 3.1)
# ---------------------------------------------------------------------------
#: Everything hash-derived descends from this one number.
CAMPAIGN_SEED = 20260901

#: TWO seeds per run, both written explicitly into every config -- see
#: `flexibility/seeds.py`, which follows the convention already used by
#: `ra_ddm_frontier_slices` and `ra_ddm_frontier_ddm*`:
#:
#:     sensory_stream.seed   H(seed, "sensory",  delta_token, replicate)   arm ABSENT
#:     arena.random_seed     H(seed, "internal", delta_token, arm, rep)    arm PRESENT
#:
#: The arm's absence from the sensory key is what pairs the arms; its presence in the
#: internal key keeps model-internal randomness independent, which is the honest
#: relationship between a ring attractor and a DDM -- they consume randomness
#: differently and at different rates, so a shared root would imply a coupling that
#: does not exist. The historical generator keyed the arena seed on
#: md5(f"{u}_{diff}_{run}"), putting the gain in the SENSORY path, so every arm drew a
#: different realisation and nothing was paired at all.
#:
#: Matched seeds are necessary but not sufficient; the models must also CONSUME the
#: stream identically, which they do: SharedPerceptStream keys each draw by
#: (trial_seed, kind, target_id, tick) alone, so neither the DDM's n_sub = 16 nor the
#: RA's 500 field steps per tick perturb it.

# ---------------------------------------------------------------------------
# Section 9: a short u-resolution check at delta = 0, run SEPARATELY from the main
# campaign. Pins where the SIMULATED system's critical point actually sits, which at
# N = 30 need not be the analytical one.
# ---------------------------------------------------------------------------
U_RESOLUTION_CHECK = (6.15, 6.2, 6.25)
U_RESOLUTION_CHECK_REPS = 200
