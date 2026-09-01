# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Condition-point matrix and the per-cell predictions (Sections 2 and 4).

Builds the ordered list of condition-points -- arm x delta -- together with each
point's derived geometry and, for the DDM arm, what the agent will actually do there.
THE ORDER IS THE CONTRACT: the SLURM array indexes into this list, so it must be
deterministic and must never depend on anything but `flexibility/factors`.

The DDM's boundary is taken from the model's OWN solver rather than re-derived here:

    rho = (A/c)^2 (c_e/c_tau),   sinh(a*) + a* = rho,   z = a* c^2 / (2A)

and then, with zeta = A z / c^2 so the error rate is 1/(1 + e^{2 zeta}),

    initial accuracy = 1 - 1/(1 + e^{2 zeta})
    t_c    = (z/A) tanh(zeta)      time to commit
    T_rev  = 2z/A                  post-swap time to traverse 2z with the drift
                                   reversed against a non-absorbing boundary

No prediction is quoted for the RA arms. These are DDM results; the ring attractor's
departure from them is the measurement.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

# The realised boundary comes from the model's own solver: the point of a prediction
# is to be checkable against what the agent actually does, and a second implementation
# of sinh(a)+a=rho would only ever diverge from the first.
from models.bellman_boundary import myopic_z  # noqa: E402

from flexibility import factors  # noqa: E402


def delta_token(delta: float) -> str:
    """Canonical token for a delta level: 0.01 -> 'd1.0000pct'.

    Factor levels enter the seed derivation and the output paths as tokens, never as
    raw floats, so neither can drift with float formatting.
    """
    return f"d{delta * 100:.4f}pct"


# ---------------------------------------------------------------------------
# Geometry — shared by every condition; only the strengths vary
# ---------------------------------------------------------------------------
def geometry() -> dict:
    """Target placement and the two horizons, derived from the locked factors."""
    half = math.radians(factors.ANGULAR_SEP_DEG) / 2.0
    d = factors.TARGET_RANGE
    x, y = d * math.cos(half), d * math.sin(half)
    v = factors.LINEAR_VELOCITY
    r0 = x                                  # distance to the bisector foot
    return {
        "pos_static_0": (x, -y, 0),
        "pos_static_1": (x, +y, 0),
        "r0": r0,
        "L": 2.0 * y,                       # chord between the targets
        "T_max": r0 / v,                    # Bellman arrival horizon
        "T_b": d / v,                       # travel budget to a target
        # N_t sets the solver's dt as T_max/N_t, and the model derives its own T_max
        # from the geometry it MEASURES at evidence onset -- by which point the agent
        # has taken a step, so the realised r0 runs ~1.6% above this static estimate.
        # Sizing N_t on the static value alone leaves dt just above the 1 ms intended,
        # so it carries a margin. The boundary converges as O(sqrt(dt)).
        "N_t": int(math.ceil(
            (r0 / v) * factors.BELLMAN_HORIZON_MARGIN / factors.BELLMAN_DT
        )),
    }


def _phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# What the DDM will do at a given delta
# ---------------------------------------------------------------------------
def operating_point(delta: float) -> dict:
    """The DDM's realised boundary and timings at this delta, from its own solver."""
    c = factors.NOISE_C
    q0, q1 = factors.strengths_for(delta)
    A = q0 - q1
    g = geometry()
    t_b, dt = g["T_b"], 1.0 / factors.TICKS_PER_SECOND
    t_delay = factors.SWAP_DELAY_TICKS * dt

    if A <= 0.0:
        # Symmetric control: no drift. Every drift-dependent boundary degenerates and
        # is floored at z_min; the cell measures spontaneous symmetry breaking.
        return {
            "delta": delta, "A": 0.0, "a_over_c": 0.0, "c_e": factors.COST_OF_ERROR,
            "z": float("nan"), "zeta": float("nan"), "initial_accuracy": 0.5,
            "t_commit": math.inf, "t_reversal": math.inf,
            "d_prime": 0.0, "ceiling_accuracy": 0.5,
            "reversal_feasible": False, "commits_in_flight": False,
            "reversal_latency_resolvable": False,
        }

    z = myopic_z(A, c, factors.COST_OF_ERROR, factors.C_TAU)
    zeta = A * z / c ** 2
    t_commit = (z / A) * math.tanh(zeta)
    t_reversal = 2.0 * z / A
    d_prime = (A / c) * math.sqrt(t_b)
    return {
        "delta": delta,
        "A": A,
        "a_over_c": A / c,
        "c_e": factors.COST_OF_ERROR,
        "z": z,
        "zeta": zeta,
        "initial_accuracy": 1.0 - 1.0 / (1.0 + math.exp(2.0 * zeta)),
        "t_commit": t_commit,
        "t_reversal": t_reversal,
        "d_prime": d_prime,
        "ceiling_accuracy": _phi(d_prime),
        # Can the agent commit, wait out the swap delay, and cover 2z before arrival?
        "reversal_feasible": (t_commit + t_delay + t_reversal) <= t_b,
        "commits_in_flight": t_commit <= t_b,
        # Is the reversal LATENCY estimable at this tick rate? Reversal RATE is
        # observable regardless; only the timing needs the resolution.
        "reversal_latency_resolvable": t_reversal >= 3.0 * dt,
    }


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Condition:
    """One (arm, delta) cell of the campaign."""

    arm: str
    delta: float
    index: int
    derived: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Directory-safe identifier, and the manifest key."""
        return f"{self.arm}__{delta_token(self.delta)}"

    @property
    def is_ra(self) -> bool:
        """True for the two ring-attractor arms."""
        return self.arm in factors.RA_ARMS

    @property
    def reps(self) -> int:
        """Replicates for this cell."""
        return factors.REPS

    @property
    def chunks(self) -> int:
        """Array tasks this cell decomposes into."""
        return math.ceil(self.reps / factors.CHUNK)


def _derive(arm: str, delta: float, g: dict) -> dict:
    """Everything computable from (arm, delta) before the run."""
    q0, q1 = factors.strengths_for(delta)
    p = operating_point(delta)

    d = {
        "q0": q0,
        "q1": q1,
        "A": p["A"],
        "c": factors.NOISE_C,
        "a_over_c": p["a_over_c"],
        "c_e": p["c_e"],
        # The Section 2 confound, recorded per cell rather than left implicit.
        "mean_drive": factors.mean_drive(delta),
        "pos_static_0": g["pos_static_0"],
        "pos_static_1": g["pos_static_1"],
        "r0": g["r0"],
        "L": g["L"],
        "T_max": g["T_max"],
        "T_b": g["T_b"],
        "N_t": g["N_t"],
        "is_symmetric": delta <= 0.0,
    }
    if arm in factors.DDM_ARMS:
        d.update({
            "pred_z": p["z"],
            "pred_zeta": p["zeta"],
            "pred_initial_accuracy": p["initial_accuracy"],
            "pred_t_commit": p["t_commit"],
            "pred_t_reversal": p["t_reversal"],
            "pred_d_prime": p["d_prime"],
            "pred_ceiling_accuracy": p["ceiling_accuracy"],
            "pred_reversal_feasible": p["reversal_feasible"],
            "pred_commits_in_flight": p["commits_in_flight"],
            "pred_reversal_latency_resolvable": p["reversal_latency_resolvable"],
        })
    return d


def deltas_for(arm: str) -> list[float]:
    """The delta grid for one arm — NOT the same for every arm.

    The DDM omits delta = 0. With identical strengths the gap is exactly zero, and
    `A_source: 'ensemble'` deduces |A| from the declared target qualities, so there is
    nothing to deduce: `resolve_ensemble_A` raises rather than inventing a drift, and
    every known-|A| boundary would degenerate to z -> 0 anyway ("evidence is
    worthless"). Running it would mean handing the DDM an ASSUMED discriminability it
    has no basis for.

    The ring attractor has no |A| to deduce, so delta = 0 is a real cell there: it
    measures spontaneous symmetry-breaking, i.e. the rate at which the bump flips with
    no quality difference at all. That is the noise floor the other cells are read
    against, so it is kept.

    Consequence for the analysis: delta = 0 is the one cell that is NOT a three-way
    paired comparison. Every other cell has all three arms on the same seed.
    """
    grid = factors.delta_grid()
    if arm in factors.DDM_ARMS:
        return [d for d in grid if d > 0.0]
    return grid


def build() -> list[Condition]:
    """The ordered condition list. Deterministic; the array indexes into it."""
    g = geometry()
    conds, idx = [], 0
    for arm in factors.ARMS:
        for delta in deltas_for(arm):
            conds.append(Condition(arm, delta, idx, _derive(arm, delta, g)))
            idx += 1
    return conds


def find_condition(name: str) -> Condition:
    """Look a condition up by `Condition.name`."""
    for cond in build():
        if cond.name == name:
            return cond
    raise KeyError(f"no such condition: {name!r}")


def task_table() -> list[tuple[int, Condition, int]]:
    """(task_id, condition, first_replicate) for every array task, in array order.

    Built by walking the conditions rather than by an arithmetic decomposition of the
    task id, so it stays correct if replicate counts ever stop being uniform.
    """
    table, task_id = [], 0
    for cond in build():
        for chunk in range(cond.chunks):
            table.append((task_id, cond, chunk * factors.CHUNK + 1))
            task_id += 1
    return table


def total_tasks() -> int:
    """Number of SLURM array tasks."""
    return len(task_table())


def total_runs() -> int:
    """Number of individual simulation runs."""
    return sum(c.reps for c in build())
