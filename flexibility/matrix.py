# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Condition-point matrix and the per-cell predictions (Sections 4 and 5).

Builds the ordered list of condition-points -- arm x delta -- together with each
point's derived geometry and, for the DDM arms, what the agent will actually do
there. THE ORDER IS THE CONTRACT: the SLURM array indexes into this list, so it must
be deterministic and must never depend on anything but `flexibility/factors`.

Section 4.2 works from a FIXED zeta = A z / c^2 = 1.1, giving

    t_c    = zeta * tanh(zeta) * (c/A)^2      T_rev = 2 * zeta * (c/A)^2

with both times scaling as 1/delta^2 -- the fact its whole grid layout rests on. But
zeta follows from the criterion (`sinh(a*) + a* = (A/c)^2 (c_e/c_tau)`, zeta = a*/2),
and at the campaign's fixed cost of error it VARIES with delta. So `operating_point`
computes z, zeta, t_c and T_rev from the boundary the model's own solver returns,
and `landmarks` finds the regime boundaries numerically on that -- a scan and a
bisection, assuming no monotonicity, because t_c is not monotonic in delta here.
Landmarks whose condition is never met come back as None rather than being omitted:
"the reversal boundary is not on this grid" is a finding, not an absence.
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

# The REALISED boundary is taken from the model's own solver, never re-derived here:
# the point of the predictions is to be checkable against what the agent actually
# does, and a second implementation of sinh(a)+a=rho would only ever diverge.
from models.bellman_boundary import myopic_z  # noqa: E402

from flexibility import factors  # noqa: E402


def delta_token(delta: float) -> str:
    """Canonical token for a delta level: 0.005 -> 'd0.5000pct'.

    Factor levels enter the seed derivation and the output paths as tokens, never as
    raw floats, so neither can drift with float formatting.
    """
    return f"d{delta * 100:.4f}pct"


# ---------------------------------------------------------------------------
# Geometry (shared by every condition -- only the strengths vary)
# ---------------------------------------------------------------------------
def geometry() -> dict:
    """Target placement and the two horizons, derived from the locked factors."""
    half = math.radians(factors.ANGULAR_SEP_DEG) / 2.0
    d = factors.TARGET_RANGE
    x = d * math.cos(half)
    y = d * math.sin(half)
    r0 = x                      # distance to the bisector foot; the Bellman horizon
    v = factors.LINEAR_VELOCITY
    return {
        "pos_static_0": (x, -y, 0),
        "pos_static_1": (x, +y, 0),
        "r0": r0,
        "L": 2.0 * y,           # chord between the targets
        "T_max": r0 / v,        # arrival horizon the Bellman policy solves over
        "T_b": d / v,           # travel budget to a target (Section 4.1)
        # N_t sets the solver's dt as T_max/N_t, and the model derives its own T_max
        # from the geometry it MEASURES at evidence onset -- by which point the agent
        # has taken a step, so the realised r0 runs ~1.6% above this static estimate
        # (measured: 0.4400 against 0.4330). Sizing N_t on the static value alone
        # would leave dt at 1.016 ms rather than the 1 ms intended, so it carries a
        # margin. The boundary converges as O(sqrt(dt)); erring fine costs seconds.
        "N_t": int(math.ceil(
            (r0 / v) * factors.BELLMAN_HORIZON_MARGIN / factors.BELLMAN_DT
        )),
    }


# ---------------------------------------------------------------------------
# Closed form (Section 4.2)
# ---------------------------------------------------------------------------
def discriminability(a_over_c: float, t_b: float) -> float:
    """d' = (A/c) sqrt(T_b) — evidence available before arrival."""
    return a_over_c * math.sqrt(t_b)


def _phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def ceiling_accuracy(a_over_c: float, t_b: float) -> float:
    """Phi(d'): the best achievable before arrival, whatever the boundary."""
    return _phi(discriminability(a_over_c, t_b))


# ---------------------------------------------------------------------------
# The realised operating point, from the policy rather than from a fixed zeta
# ---------------------------------------------------------------------------
def operating_point(delta: float) -> dict:
    """What the agent will actually do at this delta, via the model's own solver.

    Section 4.2 derives everything from a FIXED zeta = 1.1, which is only true if
    the criterion is allowed to vary with delta. At a fixed cost of error zeta is
    whatever `sinh(a*) + a* = (A/c)^2 (c_e/c_tau)` makes it, so every quantity below
    is computed from the realised boundary rather than from that assumption.
    """
    c = factors.NOISE_C
    A = factors.QUALITY_MEAN * delta
    t_b = geometry()["T_b"]
    if A <= 0.0:
        return {"delta": delta, "A": 0.0, "a_over_c": 0.0, "c_e": factors.COST_OF_ERROR,
                "z": float("nan"), "zeta": float("nan"), "initial_accuracy": 0.5,
                "t_commit": math.inf, "t_reversal": math.inf,
                "d_prime": 0.0, "ceiling_accuracy": 0.5}
    c_e = factors.COST_OF_ERROR
    z = myopic_z(A, c, c_e, factors.C_TAU)
    zeta = A * z / c ** 2
    return {
        "delta": delta,
        "A": A,
        "a_over_c": A / c,
        "c_e": c_e,
        "z": z,
        "zeta": zeta,
        "initial_accuracy": 1.0 - 1.0 / (1.0 + math.exp(2.0 * zeta)),
        "t_commit": (z / A) * math.tanh(zeta),
        "t_reversal": 2.0 * z / A,
        "d_prime": discriminability(A / c, t_b),
        "ceiling_accuracy": ceiling_accuracy(A / c, t_b),
    }


def _solve_delta(predicate, lo: float = 1e-6, hi: float = 1.0,
                 steps: int = 4000) -> Optional[float]:
    """Smallest delta in [lo, hi] where `predicate(operating_point(delta))` turns True.

    A scan-then-bisect rather than a closed-form inversion: at a fixed cost of error
    the quantities below are NOT monotonic in delta (t_c rises then falls), so an
    analytic inversion would silently return one of two roots. The scan finds the
    first sign change and bisection refines it; None means the condition never holds
    on the grid's range, which is itself a result worth reporting.
    """
    prev_d, prev_v = lo, predicate(operating_point(lo))
    for i in range(1, steps + 1):
        d = lo * (hi / lo) ** (i / steps)
        v = predicate(operating_point(d))
        if v != prev_v:
            a, b = prev_d, d
            for _ in range(60):
                m = math.sqrt(a * b)
                if predicate(operating_point(m)) == prev_v:
                    a = m
                else:
                    b = m
            return b
        prev_d, prev_v = d, v
    return None


def landmarks() -> list[dict]:
    """The delta values that partition the grid, solved on the REALISED policy.

    The first four are Section 4.3's. The last two are added because at a fixed cost
    of error they, not Section 4.3's, are what actually bound the measurement: the
    dependent variable is REVERSAL (Section 1), so it is the reversal latency's
    resolvability and the reliability of the first choice that decide which cells
    carry information.

    A landmark whose condition never holds is returned with delta = None rather than
    silently omitted -- "the reversal boundary is not on this grid" is a finding.
    """
    g = geometry()
    t_b, dt = g["T_b"], 1.0 / factors.TICKS_PER_SECOND
    t_delay = factors.SWAP_DELAY_TICKS * dt

    specs = [
        ("no_commit", lambda p: p["t_commit"] <= t_b,
         "t_c = T_b; below this there is no commitment before arrival"),
        ("reversal", lambda p: p["t_commit"] + t_delay + p["t_reversal"] <= t_b,
         "t_c + t_delay + T_rev = T_b; THE reversal boundary"),
        ("commit_three_tick", lambda p: p["t_commit"] <= 3.0 * dt,
         "t_c = 3 ticks; commitment latency becomes discretisation-limited"),
        ("commit_one_tick", lambda p: p["t_commit"] <= dt,
         "t_c = 1 tick; above this commitment carries only a binary outcome"),
        ("reversal_three_tick", lambda p: p["t_reversal"] <= 3.0 * dt,
         "T_rev = 3 ticks; REVERSAL latency becomes discretisation-limited"),
        ("reversal_one_tick", lambda p: p["t_reversal"] <= dt,
         "T_rev = 1 tick; above this reversal carries only a binary outcome"),
        ("acc_75", lambda p: p["initial_accuracy"] >= 0.75,
         "initial accuracy = 75%; below this most first choices are near-chance"),
        ("acc_90", lambda p: p["initial_accuracy"] >= 0.90,
         "initial accuracy = 90%; the Section 4.2 operating point"),
    ]

    out = []
    for name, pred, meaning in specs:
        delta = _solve_delta(pred)
        entry = {"name": name, "meaning": meaning, "delta": delta}
        if delta is None:
            entry.update({"delta_pct": None, "a_over_c": None, "t_commit": None,
                          "t_reversal": None, "ceiling_accuracy": None,
                          "initial_accuracy": None})
        else:
            p = operating_point(delta)
            entry.update({
                "delta_pct": delta * 100.0,
                "a_over_c": p["a_over_c"],
                "t_commit": p["t_commit"],
                "t_reversal": p["t_reversal"],
                "ceiling_accuracy": p["ceiling_accuracy"],
                "initial_accuracy": p["initial_accuracy"],
            })
        out.append(entry)
    return out


def usable_band(ticks_per_second: Optional[float] = None,
                min_accuracy: float = 0.75,
                min_ticks: float = 3.0) -> dict:
    """The delta range where a reversal is both DEFINED and TIMEABLE.

    Two conditions pull in opposite directions, which is what makes the band narrow:
    initial accuracy rises with delta (below `min_accuracy` about half the trials
    commit to the worse option, where the swap leaves nothing to reverse), while the
    reversal latency FALLS with delta (below `min_ticks` only the occurrence of a
    reversal is observable, not its timing).

    The lower edge is set by the noise and the cost of error and does not move with
    the tick rate; the upper edge is pure sampling resolution and moves with it
    proportionally. Returns delta bounds, their ratio, and the grid points inside.
    """
    tps = factors.TICKS_PER_SECOND if ticks_per_second is None else ticks_per_second
    dt = 1.0 / tps

    t_b = geometry()["T_b"]
    t_delay = factors.SWAP_DELAY_TICKS * dt

    def ok(p: dict) -> bool:
        """Is a reversal possible, well-defined, and timeable at this delta?"""
        return (
            # possible: commit, wait out the delay, and cover 2z before arrival
            p["t_commit"] + t_delay + p["t_reversal"] <= t_b
            # well-defined: the first choice is reliable enough that most trials
            # commit to the option the swap then makes worse
            and p["initial_accuracy"] >= min_accuracy
            # timeable: the reversal latency spans enough ticks to be estimated
            and p["t_reversal"] >= min_ticks * dt
        )

    # A direct scan, not a root-find. `ok` is a conjunction of two conditions that
    # move in OPPOSITE directions with delta -- accuracy rises with delta while the
    # reversal latency falls -- so "the first delta where the predicate changes" is
    # not well defined. The scan makes no monotonicity assumption; bisection then
    # refines each edge.
    lo_scan, hi_scan = 1e-5, 1.0
    steps = 3000
    ds = [lo_scan * (hi_scan / lo_scan) ** (i / steps) for i in range(steps + 1)]
    good = [d for d in ds if ok(operating_point(d))]
    if not good:
        return {"ticks_per_second": tps, "lo": None, "hi": None, "ratio": None,
                "grid_points": [], "n_grid_points": 0}

    def refine(inside_d: float, outside_d: float) -> float:
        """Bisect in log-delta between a point in the band and one outside it."""
        a, b = inside_d, outside_d
        for _ in range(60):
            m = math.sqrt(a * b)
            if ok(operating_point(m)):
                a = m
            else:
                b = m
        return a

    lo, hi = min(good), max(good)
    below = [d for d in ds if d < lo]
    above = [d for d in ds if d > hi]
    if below:
        lo = refine(lo, max(below))
    if above:
        hi = refine(hi, min(above))

    inside = [d for d in factors.delta_grid()
              if d > 0.0 and ok(operating_point(d))]
    return {
        "ticks_per_second": tps,
        "lo": lo,
        "hi": hi,
        "ratio": (hi / lo) if (lo and hi) else None,
        "grid_points": inside,
        "n_grid_points": len(inside),
    }


def regime_of(delta: float, marks: Optional[list[dict]] = None) -> str:
    """What kind of measurement this cell yields, from its realised operating point.

    Classified on the DEPENDENT VARIABLE, which is reversal (Section 1), not on
    commitment latency: at a fixed cost of error t_c is sub-tick everywhere, so the
    Section 4.3 map -- which orders cells by t_c -- would put every cell in one bin
    and tell us nothing. What separates cells instead is (a) whether the first
    choice is reliable enough for a reversal to be defined for most trials, and
    (b) whether the reversal latency is resolvable at this tick rate.

    `marks` is accepted for signature compatibility and ignored; the classification
    reads the operating point directly, so it cannot go stale against the landmarks.
    """
    dt = 1.0 / factors.TICKS_PER_SECOND
    t_b = geometry()["T_b"]
    t_delay = factors.SWAP_DELAY_TICKS * dt

    if delta <= 0.0:
        return "symmetric_control"

    p = operating_point(delta)
    if p["t_commit"] > t_b:
        return "no_commitment_in_flight"
    if p["t_commit"] + t_delay + p["t_reversal"] > t_b:
        return "commits_cannot_return_in_time"
    if p["t_reversal"] < dt:
        # The reversal happens inside one tick: only its occurrence is observable.
        return "single_tick"
    if p["initial_accuracy"] < 0.75:
        # Reversal latency is resolvable, but ~half the trials commit to the worse
        # option, where the swap makes them right and there is nothing to reverse.
        # Usable, at roughly half the effective replicate count (Section 7's split).
        return "near_chance_commitment"
    if p["t_reversal"] < 3.0 * dt:
        return "reversal_discretisation_limited"
    return "graded"


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
        return factors.reps_for(self.delta)

    @property
    def chunks(self) -> int:
        """Array tasks this cell decomposes into."""
        return math.ceil(self.reps / factors.CHUNK)


def _derive(arm: str, delta: float, g: dict, marks: list[dict]) -> dict:
    """Everything computable from (arm, delta) before the run — written to the manifest."""
    qbar = factors.QUALITY_MEAN
    # Mean-preserving pair (Section 3). The historical generator pinned static_0 at
    # 5.0 and weakened static_1 to 5(1-delta), so the MEAN drive fell from 5.0 to 3.0
    # across the sweep. The DDM is blind to that (it sees only the difference) but the
    # ring attractor is not: total drive shifts the field's operating point and hence
    # the effective distance from u_c -- which matters most for the arm that sits
    # exactly AT u_c, where delta = 0.8 would leave it no longer critical. Qbar(1 +/-
    # delta/2) keeps dQ = Qbar*delta unchanged, so nothing in Section 4 moves, and the
    # strength exchange still preserves both the mean and |dQ|.
    q0 = qbar * (1.0 + delta / 2.0)
    q1 = qbar * (1.0 - delta / 2.0)
    A = q0 - q1                                  # == qbar * delta
    c = factors.NOISE_C
    a_over_c = A / c
    t_b = g["T_b"]
    dt = 1.0 / factors.TICKS_PER_SECOND
    t_delay = factors.SWAP_DELAY_TICKS * dt

    d = {
        "q0": q0,
        "q1": q1,
        "A": A,
        "c": c,
        "a_over_c": a_over_c,
        "c_e": factors.COST_OF_ERROR,
        "pos_static_0": g["pos_static_0"],
        "pos_static_1": g["pos_static_1"],
        "r0": g["r0"],
        "L": g["L"],
        "T_max": g["T_max"],
        "T_b": t_b,
        "N_t": g["N_t"],
        "regime": regime_of(delta, marks),
        "is_anchor": factors.is_anchor(delta),
    }
    # Predictions are a DDM closed form. Quoting them for the RA arms would invite
    # reading them as RA predictions, which is exactly the comparison under test.
    if arm in factors.DDM_ARMS:
        # The REALISED quasi-static boundary at this cell's criterion, from the
        # model's own solver -- not the idealised z = zeta c^2/A of Section 4.2.
        # Everything below is computed from the boundary the agent will ACTUALLY
        # use, which at a fixed cost of error is not the one Section 4.2 assumed.
        p = operating_point(delta)
        z, zeta = p["z"], p["zeta"]
        tc, trev = p["t_commit"], p["t_reversal"]
        d.update({
            "pred_z": z,
            "pred_zeta": zeta,
            "pred_error_rate": 1.0 - p["initial_accuracy"],
            "pred_t_commit": tc,
            "pred_t_reversal": trev,
            # Is the REVERSAL latency resolvable at this tick rate? The dependent
            # variable is reversal, so this is the resolution question that matters;
            # commitment latency is sub-tick at every cell here.
            "pred_reversal_latency_resolvable": trev >= 3.0 * dt,
            "pred_total": tc + t_delay + trev,
            # Section 4.3's landmark, measured from t = 0: commit, wait out the swap
            # delay, then cover 2z at drift A, all inside the travel budget.
            "pred_reversal_feasible": (tc + t_delay + trev) <= t_b,
            # The laxer inequality the model checks at runtime, measured from the
            # moment of commitment against the REMAINING travel
            # (_check_reversal_feasibility: d/v vs delay + 2z/A). Reported alongside
            # because the two disagree near the boundary, and it is the model's
            # version that produces the runtime warning.
            "pred_reversal_feasible_post_commit": (t_delay + trev) <= t_b,
            "pred_commits_in_flight": tc <= t_b,
            "pred_d_prime": p["d_prime"],
            "pred_ceiling_accuracy": p["ceiling_accuracy"],
            "pred_initial_accuracy": p["initial_accuracy"],
        })
    return d


def build() -> list[Condition]:
    """The ordered condition list. Deterministic; the array indexes into it."""
    g = geometry()
    marks = landmarks()
    conds: list[Condition] = []
    idx = 0
    for arm in factors.ARMS:
        for delta in factors.delta_grid():
            conds.append(Condition(arm, delta, idx, _derive(arm, delta, g, marks)))
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

    Cells may carry different replicate counts (the anchors can be thinned), so the
    array is built by walking this table rather than by an arithmetic decomposition
    of the task id -- an arithmetic split silently mis-assigns as soon as the counts
    stop being uniform.
    """
    table = []
    task_id = 0
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
