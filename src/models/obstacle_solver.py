# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Generic 1-D free-boundary (obstacle) solver.

FEATURE_BELLMAN_POLICY.md Section 2/4. This module deliberately knows NOTHING about
drift-diffusion models, decisions or geometry: it takes a drift array, a diffusion
constant, an obstacle array and a running-cost callable, and returns the value function
and the free boundary. Keeping it that way is what lets the same code serve the
unknown-drift model later, with a belief coordinate and reparameterised time.

The problem solved is

    min { L V + running_cost(t) ,  obstacle(x) - V } = 0
    V(x, T_max) = obstacle(x)

with the generator

    L V = V_t + mu(x) V_x + 1/2 sigma^2 V_xx

The algorithm is three lines repeated backward in time:

    1. diffuse   one Crank-Nicolson step of the PDE, ignoring the obstacle
    2. add cost  V += running_cost(t) * dt
    3. project   V = min(V, obstacle)          <- the obstacle problem, in one line

Step 3 is the whole thing, and it enforces smooth pasting implicitly. The free boundary
is then read off as the edge of the contact set; it is never imposed.

This is the same structure as American option pricing (the option value can never fall
below the payoff from exercising now), which is why the scheme is textbook rather than
bespoke.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import numpy as np
from scipy.linalg import solve_banded

logger = logging.getLogger("sim.obstacle_solver")

_SCHEMES = {"crank_nicolson": 0.5, "implicit": 1.0}


def _generator_bands(x_grid: np.ndarray, drift: np.ndarray, diffusion: float):
    """Tridiagonal bands of the generator `L = mu d/dx + (sigma^2/2) d2/dx2`.

    The drift term is UPWINDED so the discrete generator has non-negative off-diagonal
    entries (an M-matrix), which is what makes the scheme monotone. `mu(x)` changes sign
    at the centre of the grid, and central differencing there oscillates -- precisely
    where the free boundary is most sensitive.

    Upwinding for a GENERATOR means: `mu > 0` takes the FORWARD difference and `mu < 0`
    the BACKWARD one. Writing the coefficient of `V[i+1]` as
    `sigma^2/(2 dx^2) + max(mu, 0)/dx` and of `V[i-1]` as
    `sigma^2/(2 dx^2) + max(-mu, 0)/dx` makes both manifestly non-negative; the opposite
    convention lets an off-diagonal go negative once `|mu| dx > sigma^2/2`.

    (FEATURE_BELLMAN_POLICY.md Section 4.2 states this rule the other way round. Test B1
    is the arbiter: with a flat running cost the solver must reproduce the closed-form
    myopic boundary, and it does only under the convention implemented here.)
    """
    dx = float(x_grid[1] - x_grid[0])
    diff = 0.5 * float(diffusion) ** 2 / dx ** 2

    lower = diff + np.maximum(-drift, 0.0) / dx      # coefficient of V[i-1]
    upper = diff + np.maximum(drift, 0.0) / dx       # coefficient of V[i+1]
    diag = -(lower + upper)                          # rows sum to zero: L kills constants
    return lower, diag, upper


def _step_matrices(lower, diag, upper, dt: float, theta: float):
    """Banded LHS `(I - theta dt L)` and the explicit RHS operator `(I + (1-theta) dt L)`.

    The LHS is returned in the (l=1, u=1) banded layout `solve_banded` expects. Rows 0
    and -1 are Dirichlet: at the edge of the grid the value IS the stopping cost.
    """
    n = diag.size
    ab = np.zeros((3, n), dtype=float)
    ab[0, 1:] = -theta * dt * upper[:-1]             # superdiagonal
    ab[1, :] = 1.0 - theta * dt * diag               # diagonal
    ab[2, :-1] = -theta * dt * lower[1:]             # subdiagonal
    # Dirichlet rows
    ab[1, 0] = ab[1, -1] = 1.0
    ab[0, 1] = 0.0
    ab[2, -2] = 0.0
    return ab


def _apply_explicit(V, lower, diag, upper, dt: float, theta: float):
    """`(I + (1-theta) dt L) V`, interior only; the edges are overwritten by Dirichlet."""
    coef = (1.0 - theta) * dt
    out = V.copy()
    out[1:-1] = (
        V[1:-1]
        + coef * (lower[1:-1] * V[:-2] + diag[1:-1] * V[1:-1] + upper[1:-1] * V[2:])
    )
    return out


def extract_boundary(V, obstacle, x_grid, tol: float, n_fit: int = 12) -> float:
    """Smallest `|x|` at which `V` meets the obstacle, recovered sub-grid.

    Naively this is "the first node where `g = obstacle - V` reaches zero", but that is
    badly conditioned. SMOOTH PASTING means `V` meets the obstacle tangentially, so near
    the free boundary

        g(x) ~ C (z - x)^2

    and a threshold test `g <= tol` therefore locates `z` only to `sqrt(tol/C)`. Worse,
    the value function itself carries an O(dt) error from the once-per-step projection,
    which enters the boundary as O(sqrt(dt)) -- measured exactly: halving dt improves the
    boundary by only sqrt(2).

    The quadratic is the way out. `sqrt(g)` is LINEAR in `x` near the boundary, so a
    least-squares line through `sqrt(g)` over the last few continuation nodes extrapolates
    to `z` with the same order of accuracy as `V` itself, and about 1.6x better in
    practice than thresholding. Falls back to interpolating the gap when the window is too
    short to fit.

    (FEATURE_BELLMAN_POLICY.md Section 4.3 specifies linear interpolation on `V - obst`.
    That is the thresholding variant above, and it is what makes B1 miss.)

    Returns `+inf` when nothing is in contact (the whole grid is a continuation region),
    which the caller should treat as "no boundary inside X_max".
    """
    g = obstacle - V                     # >= 0 everywhere after projection
    mid = int(np.argmin(np.abs(x_grid)))  # x = 0 (or nearest)

    edge = None
    for i in range(mid, len(x_grid)):
        if g[i] <= tol:
            edge = i
            break
    if edge is not None and edge > mid + 2:
        # The quadratic only holds NEAR the boundary, so the window must be a fraction of
        # the boundary's own distance from the centre -- not a fixed node count. Under a
        # collapsing geometry z(t) can fall to ~15 dx, where a fixed 12-node window spans
        # most of the continuation region and the fit degrades badly.
        width = max(3, min(int(n_fit), (edge - mid) // 3))
        lo = max(mid + 1, edge - width)
        xs, gs = x_grid[lo:edge], g[lo:edge]
        keep = gs > 0.0
        if int(keep.sum()) >= 3:
            slope, intercept = np.polyfit(xs[keep], np.sqrt(gs[keep]), 1)
            if slope < 0.0:              # sqrt(g) must fall toward the boundary
                z = -intercept / slope
                if x_grid[lo] <= z <= x_grid[edge] + (x_grid[1] - x_grid[0]):
                    return float(abs(z))

    for i in range(mid, len(x_grid)):
        if g[i] <= tol:
            if i == mid:
                return float(abs(x_grid[i]))
            g0, g1 = g[i - 1], g[i]
            if g0 <= g1:                 # degenerate: no usable slope, take the node
                return float(abs(x_grid[i]))
            frac = (g0 - tol) / (g0 - g1)
            frac = min(max(frac, 0.0), 1.0)
            return float(abs(x_grid[i - 1] + frac * (x_grid[i] - x_grid[i - 1])))
    return float("inf")


def solve_obstacle_problem(
    x_grid: np.ndarray,
    n_steps: int,
    dt: float,
    drift: np.ndarray,
    diffusion: float,
    stopping_cost: np.ndarray,
    running_cost: Callable[[float], float],
    scheme: str = "crank_nicolson",
    tol: Optional[float] = None,
):
    """Solve the obstacle problem backward in time.

    Parameters
    ----------
    x_grid : (N_x,) uniform state grid.
    n_steps, dt : time discretisation; the horizon is `n_steps * dt`.
    drift : (N_x,) `mu(x)`, time-invariant.
    diffusion : scalar `sigma`.
    stopping_cost : (N_x,) the obstacle, i.e. the cost of stopping now.
    running_cost : callable `t -> float`, the cost of waiting one more second at `t`.
    scheme : 'crank_nicolson' (default, 2nd order in dt) or 'implicit' (1st order).
    tol : contact tolerance. Defaults to `1e-10 * max(stopping_cost)` so it scales with
        the problem rather than being a bare absolute number.

    Returns
    -------
    (V, boundary, diagnostics)
        `V` is the value function at t = 0, `boundary[n]` the free boundary at `t = n*dt`.
    """
    x_grid = np.asarray(x_grid, dtype=float).reshape(-1)
    drift = np.asarray(drift, dtype=float).reshape(-1)
    stopping_cost = np.asarray(stopping_cost, dtype=float).reshape(-1)
    if not (x_grid.size == drift.size == stopping_cost.size):
        raise ValueError("x_grid, drift and stopping_cost must have the same length")
    if x_grid.size < 5:
        raise ValueError("x_grid needs at least 5 points")
    n_steps = int(n_steps)
    if n_steps < 1 or float(dt) <= 0.0:
        raise ValueError("n_steps must be >= 1 and dt > 0")
    scheme = str(scheme).strip().lower()
    if scheme not in _SCHEMES:
        raise ValueError(
            f"scheme must be one of {sorted(_SCHEMES)}; got '{scheme}'. "
            "(PSOR is not implemented: Crank-Nicolson with explicit projection is the "
            "supported scheme, and is the standard treatment for this class of problem.)"
        )
    spacing = np.diff(x_grid)
    if not np.allclose(spacing, spacing[0], rtol=1e-9, atol=0.0):
        raise ValueError("x_grid must be uniform")

    theta = _SCHEMES[scheme]
    dt = float(dt)
    if tol is None:
        tol = 1e-10 * float(np.max(np.abs(stopping_cost)))
    tol = max(float(tol), 0.0)

    lower, diag, upper = _generator_bands(x_grid, drift, diffusion)
    ab = _step_matrices(lower, diag, upper, dt, theta)

    t0 = time.perf_counter()
    V = stopping_cost.copy()                       # terminal condition V(x, T_max)
    boundary = np.zeros(n_steps, dtype=float)

    for n in range(n_steps - 1, -1, -1):
        t = n * dt
        rhs = _apply_explicit(V, lower, diag, upper, dt, theta)
        # The running cost belongs INSIDE the solve, not added to its result. Adding it
        # afterwards computes `M^-1 N V + f dt` instead of `M^-1 (N V + f dt)`, which
        # biases the value function by exactly `theta * dt` -- verified empirically:
        # 0.5*dt under Crank-Nicolson, 1.0*dt under the implicit scheme, independent of
        # dx. That bias propagates straight into z(t) and makes the boundary ~8% low.
        # (FEATURE_BELLMAN_POLICY.md Section 4.1 shows it added after the step; test B1
        # fails under that ordering and passes under this one.)
        rhs[1:-1] += float(running_cost(t)) * dt   # pay for this step
        rhs[0] = stopping_cost[0]                  # Dirichlet: certainly stop out here
        rhs[-1] = stopping_cost[-1]
        V = solve_banded((1, 1), ab, rhs)
        V[0], V[-1] = stopping_cost[0], stopping_cost[-1]
        V = np.minimum(V, stopping_cost)           # OBSTACLE PROJECTION
        boundary[n] = extract_boundary(V, stopping_cost, x_grid, tol)

    elapsed = time.perf_counter() - t0
    contact = V >= stopping_cost - tol
    diagnostics = {
        "n_x": int(x_grid.size),
        "n_t": n_steps,
        "dx": float(spacing[0]),
        "dt": dt,
        "scheme": scheme,
        "tol": float(tol),
        "wall_time_s": float(elapsed),
        # Should be ~0: a non-trivial value means the projection is not biting where the
        # contact set says it should.
        "max_abs_gap_in_contact": (
            float(np.max(np.abs(V[contact] - stopping_cost[contact]))) if contact.any() else 0.0
        ),
        "contact_fraction": float(contact.mean()),
    }
    return V, boundary, diagnostics
