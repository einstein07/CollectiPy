# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""The campaign's factors and locked parameters — defined in exactly ONE place.

CAMPAIGN_SPEC.md Sections 1, 3, 4 and 5. Everything downstream (matrix, config
generation, submission, dry run) imports from here; nothing re-declares a value.
Change a factor here and the whole campaign follows, including the task count the
submission script asserts against.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Factors (Section 3) — the Cartesian product of these is the MAIN matrix.
# ---------------------------------------------------------------------------
QUALITY_DIFFS = [0.01, 0.02, 0.05]          # dQ, relative to the better target
ANGULAR_SEPS = [60, 90, 120, 150]           # dtheta, degrees, at fixed range d
C_E_GRID = [0.03, 0.1, 0.3, 1, 3, 8, 20, 50, 125, 300]
REPS = 1000                                 # replicates per condition-point
CHUNK = 100                                 # replicates per array task (Section 7.2)

# ---------------------------------------------------------------------------
# Locked parameters (Section 1). These are decided; do not re-derive.
# ---------------------------------------------------------------------------
QUALITY_BETTER = 5.0                        # static_0 strength; static_1 = q0*(1 - dQ)
TARGET_RANGE = 0.5                          # d, metres — fixed; dtheta varies placement
LINEAR_VELOCITY = 0.05                      # m/s
WHITE_RATE = 0.035                          # sensory eta (Section 1.1); c = sqrt(2)*eta
FROZEN_SD = 0.0                             # Section 1.2
TICKS_PER_SECOND = 10                       # arena AND agent
N_SUB = 16                                  # Section 1.3
TIME_LIMIT = 60                             # seconds per run; hitting it = censored
BELLMAN_DT = 1e-3                           # solver time step; N_t = ceil(T_max / this)

#: Criterion values whose boundary sits below the integration step (Section 1.3).
#: Flagged in every manifest and per-trial row; the realised crossing log-odds is
#: logged so the contamination is measured rather than assumed.
DISCRETISATION_LIMITED_C_E = {0.03, 0.1}

#: Everything hash-derived descends from this one number (Section 6).
CAMPAIGN_SEED = 20260827

#: Base config carrying the Section 1 locked parameters. Never modified in place.
BASE_CONFIG = "config/campaign_ddm_base.json"

# ---------------------------------------------------------------------------
# Controls (Section 5) — baseline cell only.
# ---------------------------------------------------------------------------
CONTROL_QUALITY_DIFF = 0.01
CONTROL_ANGULAR_SEP = 60

#: Section 5.2 — the static control's boundary. NO definition exists in the code or
#: configuration (searched: `z_manual` has only a generic code default of 1.0, which
#: is not a campaign value, and no config defines a static-boundary constant), so this
#: stays None and the static arm REFUSES to run until a value is chosen. The proposal
#: on the table — offered, not assumed — is z_manual := z_bellman(t=0) per c_e, which
#: isolates the collapse (same starting threshold, no time variation); the dry run
#: prints z_bellman(0) per grid point so the choice can be made from evidence.
#: To unblock, set this to a list of {"z_manual": float, "from_c_e": float} dicts
#: (from_c_e records provenance only; the manual policy itself reads no cost).
STATIC_CONTROL_Z: list[dict] | None = None
