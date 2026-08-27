# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Condition-point matrix (CAMPAIGN_SPEC.md Sections 3, 4, 5).

Builds the ordered list of condition-points — the main Cartesian product first, then
the controls appended separately — together with each point's derived geometry and its
closed-form predictions. The ORDER IS THE CONTRACT: the SLURM array indexes into this
list, so it must be deterministic and must never depend on anything but
`campaign/factors.py`.

Derived per condition (Section 3B, written into every manifest):

    r0 = d cos(dtheta/2)      distance to the bisector foot
    L  = 2 d sin(dtheta/2)    chord between the targets
    T_max = r0 / v            arrival horizon
    c_tau(0) = 1 - cos(dtheta/2)

Predictions use the QUASI-STATIC closed form (the analytic approximation to the
Bellman policy; src/models/ddm_systems.py):

    z* = solve_threshold_geometric(A, c, dtheta; c_e, terminal_categorical, midpoint)
    a = 2 A z* / c^2,   acc = 1 - 1/(1 + e^a),   DT = (c^2 / 2A^2) a tanh(a/2)
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from models.ddm_systems import DriftDiffusionSystem  # noqa: E402

from campaign import factors  # noqa: E402

#: The DDM's noise scale under the shared stream: c^2 = 2 * white_rate^2 (Section 2).
NOISE_C = math.sqrt(2.0) * factors.WHITE_RATE

#: Must match the base config's z_min: predictions floor z the same way the model does.
Z_MIN = 1e-4


def q_token(q_diff: float) -> str:
    """Canonical token for a quality-difference level: 0.01 -> 'q01'."""
    return f"q{round(q_diff * 100):02d}"


def a_token(ang_sep: int) -> str:
    """Canonical token for an angular-separation level: 60 -> 'a60'."""
    return f"a{int(ang_sep)}"


@dataclass
class Condition:
    """One condition-point: an arm, its factor levels, and everything derived."""

    arm: str                      # "main" | "quasi_static" | "static"
    q_diff: float
    ang_sep: int
    c_e: Optional[float]          # None only on the static arm
    z_manual: Optional[float] = None
    from_c_e: Optional[float] = None   # provenance of a static z, if any
    index: int = -1
    derived: dict = field(default_factory=dict)
    predicted: dict = field(default_factory=dict)

    # -- identity -----------------------------------------------------------
    @property
    def q_tok(self) -> str:
        return q_token(self.q_diff)

    @property
    def a_tok(self) -> str:
        return a_token(self.ang_sep)

    @property
    def crit_tok(self) -> str:
        """The criterion token: 'ce<c_e>' or, on the static arm, 'z<z_manual>'."""
        if self.arm == "static":
            return f"z{self.z_manual:g}"
        return f"ce{self.c_e:g}"

    @property
    def name(self) -> str:
        return f"{self.q_tok}_{self.a_tok}_{self.crit_tok}"

    @property
    def rel_dir(self) -> str:
        """Directory relative to the results root (Section 8 layout)."""
        if self.arm == "main":
            return f"main/{self.name}"
        return f"controls/{self.arm}/{self.name}"

    @property
    def threshold_policy(self) -> str:
        return {"main": "bellman", "quasi_static": "geometric",
                "static": "manual"}[self.arm]

    @property
    def discretisation_limited(self) -> bool:
        """Section 1.3: flagged by criterion value, exactly as specified."""
        return self.c_e in factors.DISCRETISATION_LIMITED_C_E

    # -- physics ------------------------------------------------------------
    def compute_derived(self) -> None:
        """Fill the Section 3B geometry and quality columns."""
        d = factors.TARGET_RANGE
        theta = math.radians(self.ang_sep)
        q0 = factors.QUALITY_BETTER
        q1 = q0 * (1.0 - self.q_diff)
        r0 = d * math.cos(theta / 2.0)
        self.derived = {
            "q0": q0,
            "q1": q1,
            "A": q0 - q1,
            "theta_rad": theta,
            "d": d,
            "r0": r0,
            "L": 2.0 * d * math.sin(theta / 2.0),
            "T_max": r0 / factors.LINEAR_VELOCITY,
            "c_tau0": 1.0 - math.cos(theta / 2.0),
            "v": factors.LINEAR_VELOCITY,
            "c": NOISE_C,
            "N_t": math.ceil((r0 / factors.LINEAR_VELOCITY) / factors.BELLMAN_DT),
            # static_0 (better) below the bisector, matching the base config.
            "pos_static_0": [d * math.cos(theta / 2.0), -d * math.sin(theta / 2.0), 0],
            "pos_static_1": [d * math.cos(theta / 2.0), d * math.sin(theta / 2.0), 0],
        }

    def compute_predictions(self) -> None:
        """Closed-form (quasi-static) predictions; exact for the static arm's z."""
        A, c = self.derived["A"], self.derived["c"]
        if self.arm == "static":
            z = max(float(self.z_manual), Z_MIN)
        else:
            z = max(
                DriftDiffusionSystem.solve_threshold_geometric(
                    A, c, self.derived["theta_rad"],
                    cost_ratio=float(self.c_e),
                    mode="terminal_categorical",
                    predecision_motion="midpoint",
                ),
                Z_MIN,
            )
        a = 2.0 * A * z / c ** 2
        dt = DriftDiffusionSystem.dt_from_a(a, A, c)
        self.predicted = {
            "z": z,
            "a": a,
            "accuracy": 1.0 - DriftDiffusionSystem.er_from_a(a),
            "DT": dt,
            "DT_over_T_max": dt / self.derived["T_max"],
            "evidence_step": NOISE_C * math.sqrt(
                (1.0 / factors.TICKS_PER_SECOND) / factors.N_SUB
            ),
        }


def build_conditions() -> list[Condition]:
    """The full ordered condition list: MAIN, then quasi_static, then static.

    The static arm is appended ONLY when `factors.STATIC_CONTROL_Z` is set
    (Section 5.2); when it is None the arm simply does not exist, and everything
    that would submit it refuses separately, by name.
    """
    conds: list[Condition] = []
    for q in factors.QUALITY_DIFFS:
        for ang in factors.ANGULAR_SEPS:
            for ce in factors.C_E_GRID:
                conds.append(Condition("main", q, ang, float(ce)))
    for ce in factors.C_E_GRID:
        conds.append(Condition(
            "quasi_static", factors.CONTROL_QUALITY_DIFF,
            factors.CONTROL_ANGULAR_SEP, float(ce),
        ))
    if factors.STATIC_CONTROL_Z is not None:
        for entry in factors.STATIC_CONTROL_Z:
            conds.append(Condition(
                "static", factors.CONTROL_QUALITY_DIFF,
                factors.CONTROL_ANGULAR_SEP, None,
                z_manual=float(entry["z_manual"]),
                from_c_e=entry.get("from_c_e"),
            ))
    for i, cond in enumerate(conds):
        cond.index = i
        cond.compute_derived()
        cond.compute_predictions()

    # Section 10: the counts are asserted, not assumed.
    n_main = len(factors.QUALITY_DIFFS) * len(factors.ANGULAR_SEPS) * len(factors.C_E_GRID)
    assert sum(c.arm == "main" for c in conds) == n_main, "main matrix count drifted"
    assert sum(c.arm == "quasi_static" for c in conds) == len(factors.C_E_GRID)
    names = [c.name + c.arm for c in conds]
    assert len(set(names)) == len(names), "duplicate condition names"
    return conds


def chunks_per_condition(reps: int, chunk: int) -> int:
    """Ceiling division: the last chunk of a condition may be partial."""
    return (int(reps) + int(chunk) - 1) // int(chunk)


def chunk_range(chunk_idx: int, reps: int, chunk: int) -> range:
    """Replicate indices [k*CHUNK, min((k+1)*CHUNK, reps)) — deterministic, so any
    chunk can be re-run in isolation (Section 7.2)."""
    lo = int(chunk_idx) * int(chunk)
    return range(lo, min(lo + int(chunk), int(reps)))


def task_table(reps: int, chunk: int) -> list[tuple[Condition, int]]:
    """Array-task index -> (condition, chunk index). One task per pair."""
    conds = build_conditions()
    n_chunks = chunks_per_condition(reps, chunk)
    return [(cond, k) for cond in conds for k in range(n_chunks)]


def match_condition(cond: Condition, spec: str) -> bool:
    """True when `spec` names this condition: a bare name ('q01_a60_ce20'), an
    arm-qualified name ('quasi_static/q01_a60_ce20'), or the full relative
    directory ('controls/quasi_static/q01_a60_ce20')."""
    return spec in (cond.name, f"{cond.arm}/{cond.name}", cond.rel_dir)


def find_condition(spec: str) -> Condition:
    """Look a condition up for --only. A bare name that exists in more than one
    arm (the controls reuse the baseline cell's names) is refused as ambiguous."""
    hits = [c for c in build_conditions() if match_condition(c, spec)]
    if len(hits) == 1:
        return hits[0]
    if hits:
        raise KeyError(
            f"{spec!r} is ambiguous across arms: "
            + ", ".join(f"{c.arm}/{c.name}" for c in hits)
            + " — qualify it with the arm."
        )
    raise KeyError(
        f"no condition named {spec!r}; names look like 'q01_a60_ce0.03', "
        "optionally arm-qualified ('quasi_static/q01_a60_ce0.03'); "
        "see the dry run for the full list"
    )
