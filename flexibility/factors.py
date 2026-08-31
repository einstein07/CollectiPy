# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Flexibility campaign factors and locked parameters — defined in exactly ONE place.

Ring attractor at two gains vs. a collapsing-bound DDM, under a mid-trial world
change, swept over option quality difference. Everything downstream (matrix,
config generation, precompute, dry run, submission) imports from here; nothing
re-declares a value.

This is deliberately SEPARATE from `campaign/factors.py`, which belongs to the
speed/accuracy frontier campaign and locks a different operating point (v = 0.05,
white_rate = 0.035, ticks_per_second = 10). The two campaigns share the codebase
and the config idioms, not their constants.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Arms (Section 2). The ORDER IS THE CONTRACT: the SLURM array indexes into it.
# ---------------------------------------------------------------------------
#: Ring attractor at the critical coupling. NOT a supercritical offset -- this is
#: u_c itself, taken from the matched (v, u*) table at v = 0.5 that the RA/DDM
#: frontier sweep uses. Two consequences belong in the methods rather than in the
#: data: (a) a result AT u_c depends on how u_c was determined (continuum vs.
#: finite-N, and on beta, sigma, kappa), so the method and precision must be stated;
#: (b) finite-N effects at num_neurons = 30 are largest exactly here, so the
#: SIMULATED system's effective critical point may sit slightly off the analytical
#: one. `u_resolution_check` below pins the realised value without growing the
#: main campaign.
U_CRITICAL = 6.156868
U_RIGID = 8.0

ARMS = ("ra_uc", "ra_u8", "ddm_bellman")

#: Which base config each arm patches. The RA template is this campaign's own
#: (config/mean_field_flexibility_base.json); the shared
#: config/mean_field_2_targets_no_viz.json is NOT used, because it carries no
#: sensory_stream and no post_bifurcation_swap block and is depended on by other
#: sweeps that must not move.
TEMPLATES = {
    "ra_uc": "config/mean_field_flexibility_base.json",
    "ra_u8": "config/mean_field_flexibility_base.json",
    "ddm_bellman": "config/embodied_ddm_flexibility.json",
}

RA_ARMS = ("ra_uc", "ra_u8")
DDM_ARMS = ("ddm_bellman",)

# ---------------------------------------------------------------------------
# Locked operating point (Section 4.1). These are decided; do not re-derive.
# ---------------------------------------------------------------------------
QUALITY_MEAN = 5.0            # Qbar; strengths are Qbar*(1 +/- delta/2)
WHITE_RATE = 0.07071068       # sensory eta; c = sqrt(2)*eta = 0.100 exactly
FROZEN_SD = 0.0
TICKS_PER_SECOND = 1          # arena AND agent; equality is a fatal precondition
LINEAR_VELOCITY = 0.01        # m/s
ANGULAR_VELOCITY = 120        # deg/s
TARGET_RANGE = 0.5            # R, metres
ANGULAR_SEP_DEG = 60          # targets at +/- 30 deg
TERMINATION_RADIUS = 0.05
ARENA_SIDE = 2
TIME_LIMIT = 600              # seconds per run; hitting it = CENSORED, and the
                              # censoring rate is a reported quantity, not a filter
SWAP_DELAY_TICKS = 1          # 1.0 s physical delay at TICKS_PER_SECOND = 1
N_SUB = 16                    # DDM sub-steps per tick
BELLMAN_DT = 1e-3             # solver time step; N_t = ceil(T_max / this)
#: Safety margin on the horizon N_t is sized against. The model measures its own r0
#: at evidence onset, one step in, so the realised T_max sits slightly above the
#: static geometric value; without this the solver dt would land at 1.016 ms.
BELLMAN_HORIZON_MARGIN = 1.05
SNAPSHOTS_PER_SECOND = 1

# ---------------------------------------------------------------------------
# The criterion
# ---------------------------------------------------------------------------
#: The cost of an error, in seconds. FIXED across the whole campaign.
#:
#: Under `geometric_error_mode: terminal_categorical` the criterion IS the 0-1 loss
#: -- every error costs the same, whatever the margin -- expressed in seconds of
#: delay [READ FROM CODE: DriftDiffusionSystem.geometric_costs, and
#: _bellman_threshold sets c_e = float(self.cost_ratio)]. So c_e = 1.0 says: being
#: wrong costs exactly as much as one second of delay. Against a 50 s travel budget
#: an error is therefore cheap -- 1/50 of a trial -- which is WHY the agent commits
#: almost immediately and lands near chance at small delta. That is the criterion
#: doing its job, not a defect.
#:
#: Section 4.2 instead works from a fixed operating point (a 10% error rate) and
#: derives every landmark, and hence the Section 5 grid, from it. The boundary is not
#: free to sit there -- it follows from the cost of an error, and at a fixed cost it
#: moves with delta -- so the design document's landmark values do not describe this
#: campaign. The consequences are large and are measured by `flexibility.preflight`
#: rather than left to be discovered in the data:
#:
#:   * REVERSAL IS ALWAYS GEOMETRICALLY FEASIBLE. T_rev tops out at 3.73 s against a
#:     50 s travel budget. At a large c_e most of the grid cannot reverse at all, so
#:     this is the main thing c_e = 1.0 buys.
#:   * INITIAL ACCURACY IS AT CHANCE over the lower half of the grid -- 50.2% at
#:     delta = 0.1%, 57% at 0.57%, reaching 90% only near delta = 1.9%. Section 4.3
#:     assumed 96.2% at the reversal boundary. In those cells roughly half the trials
#:     commit to the WORSE option, where the swap makes the choice correct and there
#:     is nothing to reverse; Section 7 requires them analysed separately, so the
#:     effective replicate count there is about half of REPS.
#:   * t_c IS NON-MONOTONIC IN DELTA and sub-tick everywhere (it peaks at ~1.01 s
#:     near delta = 1.7%, falling away on both sides: at low delta z is small and
#:     cheap to reach, at high delta the drift is large). The Section 4.3 regime map
#:     orders cells by t_c assuming it falls as 1/delta^2, so that map does not
#:     describe this campaign. Commitment latency is not measurable at
#:     ticks_per_second = 1; REVERSAL latency is, and reversal is the dependent
#:     variable (Section 1).
#:   * Three of Section 4.3's four landmarks cease to exist. `matrix.landmarks`
#:     solves them numerically on the realised policy and reports the missing ones as
#:     NOT REACHED rather than omitting them.
COST_OF_ERROR = 1.0

#: Angular cost of delay, c_tau = 1 - cos(Delta/2) at Delta = 60 deg under
#: predecision_motion 'midpoint' [READ FROM CODE: DriftDiffusionSystem
#: .c_tau_linearised]. Restated here so the predicted boundary can be computed
#: without a live model; `flexibility.preflight` asserts it against the live code.
C_TAU = 1.0 - math.cos(math.radians(ANGULAR_SEP_DEG) / 2.0)

#: The DDM's noise scale under the shared stream: c^2 = 2*eta^2.
NOISE_C = math.sqrt(2.0) * WHITE_RATE

#: Must match the base configs' z_min, so predictions floor z as the model does.
Z_MIN = 1e-4

# ---------------------------------------------------------------------------
# Replication (Section 2, Section 5)
# ---------------------------------------------------------------------------
REPS = 100
#: The four saturated anchors have essentially zero variance (Section 5); they are a
#: positive control, not a graded measurement, so they may run at fewer replicates.
#: Set to REPS to disable the saving.
ANCHOR_REPS = 100
CHUNK = 10                    # replicates packed into one array task

# ---------------------------------------------------------------------------
# Quality-difference grid (Section 5)
# ---------------------------------------------------------------------------
#: delta = 0 is NOT filler: with equal strengths the exchange is a no-op, so that
#: cell measures spontaneous symmetry-breaking time and the SPONTANEOUS reversal
#: rate -- the null distribution every other reversal count is scored against.
DELTA_ZERO = 0.0

#: 18 log-spaced points, 0.10% -> 4.00%, ratio 1.2424. Straddles all four landmarks
#: of Section 4.3 with ~1.24x resolution through the transition.
LOG_LEG_MIN = 0.001
LOG_LEG_MAX = 0.04
LOG_LEG_N = 18

#: Ceiling / positive-control anchors (Section 4.4). Sub-tick for both DDM arms:
#: they reverse deterministically within one tick. Load-bearing as proof that the
#: swap fired and propagated, NOT a graded measurement -- do not fit a curve here.
ANCHORS = (0.08, 0.20, 0.45, 0.80)


def delta_grid() -> list[float]:
    """The ordered delta grid. ORDER IS THE CONTRACT — the array indexes into it."""
    ratio = (LOG_LEG_MAX / LOG_LEG_MIN) ** (1.0 / (LOG_LEG_N - 1))
    log_leg = [round(LOG_LEG_MIN * ratio ** i, 8) for i in range(LOG_LEG_N)]
    return [DELTA_ZERO] + log_leg + list(ANCHORS)


def is_anchor(delta: float) -> bool:
    """True for the saturated ceiling anchors of Section 4.4."""
    return any(math.isclose(delta, a, rel_tol=1e-9) for a in ANCHORS)


def reps_for(delta: float) -> int:
    """Replicate count for a delta level."""
    return ANCHOR_REPS if is_anchor(delta) else REPS


# ---------------------------------------------------------------------------
# Seeding (Section 3)
# ---------------------------------------------------------------------------
#: Everything hash-derived descends from this one number.
CAMPAIGN_SEED = 20260831

#: THE ARM IS DELIBERATELY ABSENT from the seed key. The historical generator keyed
#: the arena seed on md5(f"{u}_{diff}_{run}"), so every gain value drew a DIFFERENT
#: noise stream and the arms could not be paired. Dropping the arm makes all four
#: arms replay the same realisation at the same (delta, replicate), which is what
#: licenses the paired statistics (McNemar on reversal, Wilcoxon on latency) and is
#: worth a large factor in power. Sharing the seed is necessary but not sufficient;
#: the models must also CONSUME the stream identically, which they do -- the shared
#: stream keys each draw by (trial_seed, kind, target_id, tick) alone, so neither
#: the DDM's n_sub = 16 nor the RA's 500 field steps per tick perturb it.
SEED_KEY_INCLUDES_ARM = False


def reference_delta() -> float:
    """The campaign's representative cell — the grid delta reported as its headline.

    The grid point closest to the middle of the usable band: the range where a
    reversal is possible, the first choice is reliable enough for a reversal to be
    defined, and the reversal latency is still resolvable. Quoting a number from the
    near-chance region would describe cells where half the trials have nothing to
    reverse.

    Derived rather than hard-coded, because the band moves with the tick rate and
    with the cost of error. Falls back to the geometric middle of the grid if no
    delta satisfies all three conditions.
    """
    from flexibility import matrix

    band = matrix.usable_band()
    grid = [d for d in delta_grid() if d > 0.0]
    if band["lo"] is None:
        return grid[len(grid) // 2]
    target = math.sqrt(band["lo"] * band["hi"])
    return min(grid, key=lambda d: abs(math.log(d) - math.log(target)))

# ---------------------------------------------------------------------------
# Section 2: a short u-resolution check at delta = 0, run SEPARATELY from the main
# campaign. Pins where the simulated system's critical point actually sits, which
# at N = 30 need not be the analytical u_c.
# ---------------------------------------------------------------------------
U_RESOLUTION_CHECK = (6.14, 6.156868, 6.17)
U_RESOLUTION_CHECK_REPS = 200
