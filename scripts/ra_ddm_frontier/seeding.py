# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2026 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""scripts/ra_ddm_frontier/seeding.py — single source of truth for BOTH campaigns.

Section 3 of `ra-ddm-frontier-slice-envelope-experiment.md`, verbatim. Seeds are
functions of trial identity only — (Δθ, δ_Q, run_id) — never of the model or a
swept parameter. Key fields are integers (δ_Q in basis points, Δθ in degrees,
run_id); the hash is md5 (stable across platforms and Python versions), reduced
mod 2^31 to fit the int32 seed fields the configs use. Any change to the
derivation bumps SCHEME — a new scheme is a new seed universe, never silent
drift.
"""

import hashlib

SCHEME = "frontier-v1"


def _derive(*fields: str) -> int:
    key = "::".join(fields)
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**31)


def _trial(dth_deg: int, diff_bp: int, run_id: int) -> str:
    # integers only: 1 % -> diff_bp = 100; dtheta = 60 deg -> dth_deg = 60
    return f"{SCHEME}::dth{int(dth_deg)}::dq{int(diff_bp)}::r{int(run_id)}"


def env_seed(dth_deg: int, diff_bp: int, run_id: int, domain: str) -> int:
    """Shared exogenous stream for one receiving field ('arena', 'sensory', ...).
    Identical across models by construction."""
    return _derive(_trial(dth_deg, diff_bp, run_id), "env", domain)


def model_seed(model: str, dth_deg: int, diff_bp: int, run_id: int) -> int:
    """Model-private noise ('ra' or 'ddm'). Never fed to an exogenous field."""
    return _derive(_trial(dth_deg, diff_bp, run_id), "model", model)
