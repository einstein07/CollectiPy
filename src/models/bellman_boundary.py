# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Known-|A| coefficients for the optimal (Bellman) decision boundary.

FEATURE_BELLMAN_POLICY.md Section 5. This module assembles the DDM-specific pieces --
the subjective drift, the stopping cost, the grid -- and hands them to the generic
`obstacle_solver`. All the numerics live there; all the modelling lives here.

The boundary solves

    min { V_t + A tanh(k x / 2) V_x + 1/2 c^2 V_xx + c_tau(t),  c_e ER(x) - V } = 0
    V(x, T_max) = c_e ER(x),        k = 2A/c^2,   ER(x) = 1/(1 + exp(k|x|))

and `z(t)` is the edge of the contact set.

`geometric` is the QUASI-STATIC approximation to this: it solves the STATIC optimum
`sinh(a) + a = rho` and re-evaluates it each tick at the current geometry. The gap
between the two is the cost of that approximation, and is the point of the feature.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Optional

import numpy as np

from models.obstacle_solver import solve_obstacle_problem

logger = logging.getLogger("sim.bellman_boundary")

#: `exp(k * X_max)` overflows past ~709. Section 5.2 asks for a guard well inside that;
#: the obstacle itself is computed in an overflow-free form, so this only protects the
#: drift and the grid from being nonsense.
_MAX_K_XMAX = 500.0


def myopic_z(A: float, c: float, c_e: float, c_tau: float) -> float:
    """The static (quasi-static) optimum `z*` from `sinh(a) + a = rho`.

    Reuses the existing geometric-policy solver rather than re-deriving it: if these two
    ever disagree the whole comparison in Section 10 becomes meaningless.
    """
    # Imported lazily: ddm_systems dispatches to this module, so a module-level import
    # would close a cycle.
    from models.ddm_systems import DriftDiffusionSystem

    A, c, c_e, c_tau = abs(float(A)), float(c), float(c_e), float(c_tau)
    if A <= 0.0 or c <= 0.0 or c_tau <= 0.0:
        return 0.0
    rho = (A / c) ** 2 * (c_e / c_tau)
    a_star = float(DriftDiffusionSystem.solve_a_star(rho))
    return a_star * c ** 2 / (2.0 * A)


def halt_exit_potential(x, k: float, A: float):
    """`D(x) = (x/A) tanh(kx/2)` — the mean-exit-time potential of the halted problem
    (BELLMAN_KNOWN_A_TERMINAL_HALT Section 2.2).

    Even, smooth at 0 (`D ~ k x^2 / 2A`), in seconds. The mean time to exit `[-z, z]`
    from `x` under the subjective process is `D(z) - D(x)`; from a central start,
    `D(z) = (z/A) tanh(kz/2)` is Bogacz's decision time.
    """
    x = np.asarray(x, dtype=float)
    out = x * np.tanh(0.5 * float(k) * x) / float(A)
    return float(out) if out.ndim == 0 else out


def solve_z_halt(k: float, A: float, c_e: float, c_h: float) -> float:
    """The flat halt threshold: `sinh(a) + a = rho_h`, `a = k z_halt`,
    `rho_h = c_e k A / (2 c_h)` (Section 2.3).

    This IS the myopic equation with `c_tau -> c_h`: `rho_h = (A/c)^2 (c_e/c_h)`, so the
    existing `sinh(a)+a` solver is reused rather than re-derived — if the two ever
    disagree, test B9(i) is the arbiter.
    """
    from models.ddm_systems import DriftDiffusionSystem  # lazy: avoids an import cycle

    k, A, c_e, c_h = float(k), abs(float(A)), float(c_e), float(c_h)
    if c_h <= 0.0:
        raise ValueError("halt_cost_rate must be > 0")
    rho_h = c_e * k * A / (2.0 * c_h)
    a = float(DriftDiffusionSystem.solve_a_star(rho_h))
    return a / k


def terminal_slice(
    x: np.ndarray,
    obst: np.ndarray,
    k: float,
    A: float,
    c_e: float,
    terminal: str = "forced_choice",
    c_h: float = 1.0,
):
    """Build the terminal value `V(x, T_max)` for the backward solve.

    BELLMAN_KNOWN_A_TERMINAL_HALT Sections 2.4 / 4.2. Returns `(V_T, z_halt)`.

    forced_choice: `V_T = obst` — commit at arrival whatever you believe; the boundary
    collapses to `z(T_max) = 0`.

    halt_sprt: at arrival the agent may halt at the midpoint and keep integrating at
    delay rate `c_h` against the flat threshold `z_halt`, so

        V_T(x) = g_min - c_h D(x)   for |x| <= z_halt,      = c_e ER(x)   outside,
        g_min  = c_e ER(z_halt) + c_h D(z_halt).

    THE TRAP (Section 2.4): the interior branch continues BELOW the obstacle for
    |x| > z_halt (a threshold inside the current position is meaningless), so
    `np.minimum(obst, g_min - c_h D)` selects the wrong branch outside and silently
    corrupts the stopping-set extraction. Build piecewise, never by minimum — test
    B9(ii) is the gate for exactly this bug.
    """
    terminal = str(terminal).strip().lower()
    if terminal == "forced_choice":
        return obst.copy(), 0.0
    if terminal != "halt_sprt":
        raise ValueError(
            f"terminal must be 'forced_choice' or 'halt_sprt'; got '{terminal}'"
        )
    x = np.asarray(x, dtype=float)
    k, A, c_e, c_h = float(k), abs(float(A)), float(c_e), float(c_h)
    z_halt = solve_z_halt(k, A, c_e, c_h)
    D = halt_exit_potential(x, k, A)
    # ER(z_halt) in the overflow-free form (Section 6): only decaying exponentials.
    e = math.exp(-k * z_halt)
    g_min = c_e * e / (1.0 + e) + c_h * halt_exit_potential(z_halt, k, A)
    V_T = np.where(np.abs(x) <= z_halt, g_min - c_h * D, obst)
    return V_T, float(z_halt)


def stopping_cost_array(x: np.ndarray, k: float, c_e: float) -> np.ndarray:
    """`c_e * ER(x)` computed without a positive exponent (Section 5.2).

    `c_e / (1 + exp(k|x|))` overflows for large `k*X_max`; multiplying numerator and
    denominator by `exp(-k|x|)` gives the identical function with only decaying
    exponentials.
    """
    e = np.exp(-float(k) * np.abs(x))
    return float(c_e) * e / (1.0 + e)


def subjective_drift(x: np.ndarray, A: float, k: float) -> np.ndarray:
    """`A tanh(k x / 2)` -- the posterior mean drift (Section 3).

    THE TRAP. The world runs `dx = A dt + c dW`, but the agent does not know `sign(A)`,
    so under its own filtration the expected drift is the posterior mean: zero at x = 0,
    approaching +/-A as evidence accumulates. Passing the world's linear drift `A` here
    gives a systematically WRONG boundary, because the linear process appears to resolve
    the sign faster than the agent can actually know it.

    DIRECTION: Section 3 says the resulting boundary is "too low". It is too HIGH.
    Faster apparent resolution makes deliberation look CHEAP -- shorter DT and lower ER
    at any given z -- so the solver buys more of it. Measured at the calibrated parameter
    set: linear 0.2529 vs subjective 0.2182, a 16% over-estimate (test B5).

    Simulate with the objective linear drift; solve with this.
    """
    return float(A) * np.tanh(float(k) * x / 2.0)


def bellman_boundary(
    A: float,
    c: float,
    c_e: float,
    c_tau_of_t: Callable[[float], float],
    T_max: float,
    *,
    # Section 5 suggests N_x = 801. Measured: the mean relative boundary gap converges
    # from BELOW as the grid refines (0.062 -> 0.077 -> 0.083 -> 0.085 at N_x = 801,
    # 1601, 3201, 6401), because the extraction quantises against dx. 801 understates the
    # effect by ~27% and puts a visible sawtooth on z(t) as the contact edge steps from
    # node to node. 1601 halves the ripple at ~0.4 s; use 3201 for figures.
    N_x: int = 1601,
    # Section 5 suggests N_t = 2000. That is too coarse: the free boundary converges as
    # O(sqrt(dt)) (smooth pasting makes the contact tangential -- see
    # obstacle_solver.extract_boundary), so dt ~ 4e-3 leaves the boundary ~3 dx off and
    # test B1 fails. dt <~ 1e-3 brings it inside 2 dx, which at the r0/v horizon means
    # N_t of order 10^4. The solve is still well under a second.
    N_t: int = 10000,
    X_max: Optional[float] = None,
    X_max_factor: float = 4.0,
    z_myopic_onset: Optional[float] = None,
    scheme: str = "crank_nicolson",
    horizon_check_factor: Optional[float] = 1.5,
    terminal: str = "forced_choice",
    halt_cost_rate: float = 1.0,
):
    """Solve for `z(t)` on `[0, T_max]`.

    Returns `(t_grid, z, diagnostics)` where `t_grid[n] = n * T_max / N_t`. `diagnostics`
    carries the solver's own numbers plus `z_myopic` on the same grid, so the Section 10
    figure and test B6 need no extra machinery.

    `terminal` selects the condition at the horizon (BELLMAN_KNOWN_A_TERMINAL_HALT):
    'forced_choice' (default, unchanged behaviour) commits at arrival whatever the
    belief, so `z(T_max) = 0`; 'halt_sprt' lets the agent halt at the midpoint and keep
    integrating at delay rate `halt_cost_rate` against the flat threshold `z_halt`, so
    the boundary collapses onto that plateau instead. Under 'halt_sprt' the returned
    arrays gain an explicit final row at `t = T_max` with `z = z_halt` (recorded rather
    than left to be interpolated from the `N_t - 1` row), and `diagnostics` carries
    `z_halt`, `g_min` and `halt_mean_exit_time = D(z_halt)`.
    """
    A, c, c_e, T_max = abs(float(A)), float(c), float(c_e), float(T_max)
    terminal = str(terminal).strip().lower()
    if terminal not in {"forced_choice", "halt_sprt"}:
        raise ValueError(
            f"terminal must be 'forced_choice' or 'halt_sprt'; got '{terminal}'"
        )
    halt_cost_rate = float(halt_cost_rate)
    if terminal == "halt_sprt" and halt_cost_rate <= 0.0:
        raise ValueError("halt_cost_rate must be > 0 under terminal 'halt_sprt'")
    if A <= 0.0:
        raise ValueError(
            "bellman_boundary requires a known, non-zero |A|. Set A_expected with "
            "A_source 'ensemble' (the policy assumes the drift magnitude is known)."
        )
    if c <= 0.0:
        raise ValueError("bellman_boundary requires c > 0")
    if T_max <= 0.0:
        raise ValueError("bellman_boundary requires T_max > 0")

    k = 2.0 * A / c ** 2

    if X_max is None:
        if z_myopic_onset is None:
            z_myopic_onset = myopic_z(A, c, c_e, float(c_tau_of_t(0.0)))
        if not (z_myopic_onset > 0.0) or not math.isfinite(z_myopic_onset):
            raise ValueError(
                "cannot size the grid: z_myopic(0) is not positive and finite "
                f"(got {z_myopic_onset!r}). Pass X_max explicitly."
            )
        X_max = float(X_max_factor) * float(z_myopic_onset)
    X_max = float(X_max)

    if k * X_max >= _MAX_K_XMAX:
        raise ValueError(
            f"k * X_max = {k * X_max:.1f} exceeds {_MAX_K_XMAX:.0f}: the grid spans an "
            f"implausible number of log-odds (k = 2A/c^2 = {k:.4g}, X_max = {X_max:.4g}). "
            "This usually means eta_rate is far too small for the drift. Raise eta_rate, "
            "or pass a smaller X_max."
        )

    x = np.linspace(-X_max, X_max, int(N_x))
    drift = subjective_drift(x, A, k)
    obstacle = stopping_cost_array(x, k, c_e)

    # Terminal condition. Forced mode passes terminal_V=None so its path through the
    # solver is the historical one, bit for bit (Section 6 regression clause).
    z_halt = 0.0
    V_T = None
    if terminal == "halt_sprt":
        V_T, z_halt = terminal_slice(x, obstacle, k, A, c_e, terminal, halt_cost_rate)
        # Guaranteed by Section 2.3 when c_tau(0) < c_h (z_halt <= z_myopic(0) then);
        # an aggressive halt_cost_rate can break that ordering, so check rather than
        # trust it.
        if not (z_halt < X_max):
            raise ValueError(
                f"z_halt = {z_halt:.4g} does not fit inside X_max = {X_max:.4g}. "
                "The halt threshold exceeds the grid: raise X_max_factor (or "
                "halt_cost_rate is unphysically small)."
            )

    dt = T_max / float(N_t)
    V, z, diag = solve_obstacle_problem(
        x_grid=x, n_steps=int(N_t), dt=dt,
        drift=drift, diffusion=c,
        stopping_cost=obstacle, running_cost=c_tau_of_t, scheme=scheme,
        terminal_V=V_T,
    )
    t_grid = np.arange(int(N_t), dtype=float) * dt
    if terminal == "halt_sprt":
        # Record the T_max slice explicitly: z(T_max) = z_halt by construction, and
        # plots/lookups must not interpolate it from the N_t - 1 row (Section 4.2). The
        # runtime's hold-past-the-horizon rule then clamps z(t >= T_max) = z_halt for
        # free.
        t_grid = np.append(t_grid, T_max)
        z = np.append(z, z_halt)

    # The quasi-static comparison, on the same grid. One Newton solve per sample.
    z_my = np.array([myopic_z(A, c, c_e, float(c_tau_of_t(float(t)))) for t in t_grid])

    diag = dict(diag)
    diag.update({
        "A": A, "c": c, "c_e": c_e, "k": k, "X_max": X_max, "T_max": T_max,
        "k_X_max": k * X_max,
        "z_myopic": z_my,
        "z_myopic_onset": float(z_my[0]) if z_my.size else float("nan"),
        "unbounded_fraction": float(np.mean(~np.isfinite(z))),
        "terminal": terminal,
    })
    if terminal == "halt_sprt":
        e_h = math.exp(-k * z_halt)
        diag.update({
            "z_halt": float(z_halt),
            "halt_cost_rate": halt_cost_rate,
            "g_min": float(c_e * e_h / (1.0 + e_h)
                           + halt_cost_rate * halt_exit_potential(z_halt, k, A)),
            # Mean extra deliberation of a halt started at the midpoint; the runtime's
            # runaway guard caps episodes at T_max + 10x this (Section 4.3).
            "halt_mean_exit_time": float(halt_exit_potential(z_halt, k, A)),
        })

    if diag["unbounded_fraction"] > 0.0:
        logger.warning(
            "bellman_boundary: %.1f%% of the horizon has no contact set inside "
            "X_max = %.4g. The grid is too narrow; raise X_max_factor.",
            100.0 * diag["unbounded_fraction"], X_max,
        )

    # B3 as a startup check, not only a test: if the answer moves when the horizon moves,
    # the horizon is doing modelling work it should not be doing. A forced-mode
    # diagnostic only: under 'halt_sprt' T_max is the physical arrival time, not a
    # truncation, and the exact statement is test B11's fixed-point/extended-horizon
    # equivalence rather than this asymptotic probe.
    if terminal == "halt_sprt":
        if horizon_check_factor and horizon_check_factor > 1.0:
            diag["horizon_check"] = {
                "skipped": "terminal 'halt_sprt': T_max is the arrival time, not a "
                           "truncation; superseded by B11", "ok": True,
            }
    elif horizon_check_factor and horizon_check_factor > 1.0:
        diag["horizon_check"] = _horizon_check(
            A, c, c_e, c_tau_of_t, T_max, float(horizon_check_factor),
            N_x, N_t, X_max, scheme, z,
        )

    logger.info(
        "bellman: solved N_x=%d N_t=%d dx=%.4g dt=%.4g X_max=%.4g T_max=%.4g "
        "k*X_max=%.1f in %.2fs | z(0)=%.4g vs z_myopic(0)=%.4g",
        diag["n_x"], diag["n_t"], diag["dx"], diag["dt"], X_max, T_max,
        diag["k_X_max"], diag["wall_time_s"], z[0], diag["z_myopic_onset"],
    )
    if diag["wall_time_s"] > 2.0:
        logger.warning(
            "bellman: solve took %.1fs (> 2s). Check N_x/N_t -- the grid is probably "
            "far finer than the boundary needs.", diag["wall_time_s"],
        )
    if diag["dt"] > 2.0e-3:
        # The free boundary converges as O(sqrt(dt)), so a coarse dt degrades z(t) much
        # faster than it degrades the value function. Say so rather than letting a
        # plausible-looking but low boundary pass unremarked.
        logger.warning(
            "bellman: dt = %.2g is coarse for a free-boundary solve. z(t) converges as "
            "O(sqrt(dt)), so expect the boundary to sit LOW by roughly %.2g "
            "(~%.1f dx). Raise N_t to about %d for dt = 1e-3.",
            diag["dt"], 0.11 * math.sqrt(diag["dt"]),
            0.11 * math.sqrt(diag["dt"]) / diag["dx"], int(math.ceil(T_max / 1.0e-3)),
        )
    return t_grid, z, diag


def _horizon_check(A, c, c_e, c_tau_of_t, T_max, factor,
                   N_x, N_t, X_max, scheme, z_ref):
    """Re-solve at `factor * T_max` and compare over the middle of the shorter horizon.

    Away from the terminal region the boundary should not know where the horizon is. A
    material disagreement means `T_max` is truncating the problem rather than bounding it.
    """
    T2 = float(factor) * T_max
    N_t2 = int(round(int(N_t) * factor))
    x = np.linspace(-X_max, X_max, int(N_x))
    k = 2.0 * A / c ** 2
    _, z2, _ = solve_obstacle_problem(
        x_grid=x, n_steps=N_t2, dt=T2 / N_t2,
        drift=subjective_drift(x, A, k), diffusion=c,
        stopping_cost=stopping_cost_array(x, k, c_e),
        running_cost=c_tau_of_t, scheme=scheme,
    )
    # Compare on the middle 80% of the SHORT horizon, sampled from both solutions.
    t_probe = np.linspace(0.1 * T_max, 0.9 * T_max, 50)
    a = np.interp(t_probe, np.arange(int(N_t)) * (T_max / int(N_t)), z_ref)
    b = np.interp(t_probe, np.arange(N_t2) * (T2 / N_t2), z2)
    denom = np.maximum(np.abs(a), 1e-12)
    rel = float(np.max(np.abs(a - b) / denom))
    ok = rel < 0.01
    if not ok:
        logger.warning(
            "bellman: horizon check FAILED -- z(t) moves by %.1f%% when T_max is scaled "
            "by %.2g. The horizon is truncating the problem; raise T_max.",
            100.0 * rel, factor,
        )
    return {"factor": float(factor), "max_rel_diff": rel, "ok": bool(ok)}
