# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Discrete (sampled-evidence) Bellman boundary -- Variant D of
BELLMAN_KNOWN_A_DERIVATION Section 13.

`bellman_boundary` (Variant C) solves the diffusion LIMIT of the task: a PDE in
`(x, t)` with smooth pasting and closed forms at the midpoint. The simulator, however,
hands the accumulator ONE Gaussian observation per sampling interval `Delta_t` and
tests `|x| >= z` once per observation -- observation m, taken with the agent at node
m, against `b_m` (the runtime reads the table at `t_evidence - Delta_t`; see
`DriftDiffusionSystem.decision_time`). At the intervals actually simulated the two are
not the same problem: the per-step LLR increment has sd `sqrt(2 I Delta_t)`, which is of
order the threshold itself, so the continuous boundary over-waits at every step
(Section 13.7). This module solves the exact posterior-predictive Bellman recursion on
the step lattice instead:

    W_n(L) = min{ G(L),  c_n + E[ W_{n+1}(L') | L ] },      L' = L + kappa o,

with the expectation over the MIXTURE `p(L) N(+I dt, 2 I dt) + (1-p(L)) N(-I dt, 2 I dt)`
of Section 13.3 (never a single Gaussian with the posterior-mean drift -- that solves a
different, wrong, Bayesian problem), the exact per-step cost `c_n` of Section 13.4, and
the boundary read off as the transversal crossing of `G` and the continuation value
(no smooth pasting: Section 13.5).

Coordinates. The recursion is solved in LLR units `L = k x`, `k = 2A/c^2`, where the
grid is problem-independent, and the result is returned in the accumulator's EVIDENCE
units `z = b / k` so `DriftDiffusionSystem.set_bellman_table` takes it unchanged. The
information rate is `I = 2A^2/c^2 = kA`; the observation parameters of the derivation
enter only through `I dt` (mean LLR per step) and `2 I dt` (its variance).

Numerics (Section 13.8). The expectation is a convolution of the value function with
two fixed Gaussians, weighted by the posterior. It is evaluated on a symmetric uniform
LLR grid as a banded matrix of exact Gaussian densities with trapezoid weights, each
row rescaled to the exact (normal-CDF) mass inside the grid, and the immediate-stopping
value `G` integrated over the mass beyond the grid edge (never the edge value). This is
the "convolution on the grid" formulation the derivation names as equivalent to its
Gauss-Hermite-plus-interpolation scheme, and it is preferred because it is uniformly
second order in the grid spacing: Gauss-Hermite samples the kinked value function at
node positions that move with `n_quad`, which measurably wobbles the boundary (0.4% on
the worked example between 40 and 80 nodes) and fails outright on the first-passage
statistics, whose integrands jump at the boundary. The Gauss-Hermite operator is kept
as `quadrature='gauss_hermite'` for cross-checks. Either operator is assembled ONCE as
a sparse matrix, so one step of the recursion is a sparse matvec, and the stationary
midpoint fixed point is solved by policy (Howard) iteration -- exact for the
discretised chain in a handful of linear solves -- with value iteration kept as an
independent check and fallback.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.special import ndtr

logger = logging.getLogger("sim.bellman_discrete")

#: Interpolation nodes for the discrete quasi-static comparator (Section 13.6 (ii)):
#: `b_qs(cost)` is smooth and monotone in `log(cost)`, so this many log-spaced solves
#: reproduce it to well inside the tolerances anything downstream uses.
_QS_MAX_EXACT = 40
_QS_INTERP_NODES = 32

_METHODS = ("policy_iteration", "value_iteration")
_QUADRATURES = ("grid", "gauss_hermite")

#: Reach of the grid convolution in kernel standard deviations (1 - 1e-32 of the
#: mass) and the number of trapezoid points used on each tail beyond the grid edge.
_TAIL_SD = 12.0
_TAIL_POINTS = 600
_SQRT_2PI = math.sqrt(2.0 * math.pi)

#: Grid resolution floor: the value function is linearly interpolated at the quadrature
#: nodes, so the per-step kernel must be resolved by the grid. Measured on the frontier
#: calibration: ~1.4 cells per kernel sd biases b_mid by +0.8% (above the continuous
#: limit, which it must approach from below), 5.5 cells by 0.05%, 8+ cells by < 0.02%.
#: The requested N_L is raised to meet this floor when Delta_t is small.
_MIN_CELLS_PER_SD = 8.0
_N_L_CAP = 40001


# ---------------------------------------------------------------------------
# Elementary pieces
# ---------------------------------------------------------------------------
def posterior(L):
    """`p(L) = 1 / (1 + exp(-L))`, overflow-free on both tails."""
    L = np.asarray(L, dtype=float)
    out = np.empty_like(L)
    pos = L >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-L[pos]))
    e = np.exp(L[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def stopping_cost(L, c_e: float):
    """`G(L) = c_e / (1 + exp(|L|))` with decaying exponentials only."""
    e = np.exp(-np.abs(np.asarray(L, dtype=float)))
    return float(c_e) * e / (1.0 + e)


def gauss_hermite(n_quad: int):
    """Nodes `xi` and weights `w` with `E[F(m + s Z)] = sum_i w_i F(m + s sqrt(2) xi_i)`."""
    xi, w = np.polynomial.hermite.hermgauss(int(n_quad))
    return xi, w / math.sqrt(math.pi)


def mixture_nodes(L, I_dt: float, n_quad: int):
    """Quadrature points of the two components of the predictive mixture.

    Returns `(pts_plus, pts_minus, w)`: `pts_plus[i, j]` is `L_i + I dt + sqrt(2 I dt)
    sqrt(2) xi_j` (the observation drawn under H1), `pts_minus` its H2 counterpart with
    the shifted mean `-I dt`, and `w` the normalised weights.
    """
    L = np.asarray(L, dtype=float).reshape(-1)
    xi, w = gauss_hermite(n_quad)
    sd = math.sqrt(2.0 * float(I_dt))
    spread = sd * math.sqrt(2.0) * xi
    pts_plus = L[:, None] + float(I_dt) + spread[None, :]
    pts_minus = L[:, None] - float(I_dt) + spread[None, :]
    return pts_plus, pts_minus, w


def mixture_expectation(F: Callable, L, I_dt: float, n_quad: int = 40):
    """`E[F(L') | L]` under the predictive MIXTURE, `F` evaluated exactly at the nodes.

    This is the operator of Section 13.3 with no grid interpolation, for diagnostics and
    for test D3: with `F = posterior` it must return `posterior(L)` to quadrature
    precision -- the martingale identity that a single Gaussian with the posterior-mean
    drift fails at any finite interval.
    """
    L = np.asarray(L, dtype=float).reshape(-1)
    pts_plus, pts_minus, w = mixture_nodes(L, I_dt, n_quad)
    p = posterior(L)
    return p * (F(pts_plus) @ w) + (1.0 - p) * (F(pts_minus) @ w)


def gaussian_expectation(F: Callable, L, I_dt: float, n_quad: int = 40):
    """`E[F(L')]` under the SINGLE Gaussian `N(L + I dt tanh(L/2), 2 I dt)`.

    The Section 2 / Section 13.3 trap, kept callable so the negative control of test
    D3 can show it is NOT a martingale kernel at a finite interval. Never used by the
    solver.
    """
    L = np.asarray(L, dtype=float).reshape(-1)
    xi, w = gauss_hermite(n_quad)
    sd = math.sqrt(2.0 * float(I_dt))
    mean = L + float(I_dt) * np.tanh(0.5 * L)
    pts = mean[:, None] + (sd * math.sqrt(2.0) * xi)[None, :]
    return F(pts) @ w


def crossing(L, G, cont) -> float:
    """Smallest `L >= 0` with `G(L) <= cont(L)`, linearly interpolated (Section 13.8).

    The crossing is transversal (`W_n` has a kink there, Section 13.5), so linear
    interpolation between the two bracketing nodes is second-order accurate. Returns
    `0.0` when stopping is optimal already at the centre and `+inf` when nothing on the
    non-negative half-grid is in the stopping set.
    """
    L = np.asarray(L, dtype=float)
    f = np.asarray(G, dtype=float) - np.asarray(cont, dtype=float)
    pos = L >= 0.0
    Lp, fp = L[pos], f[pos]
    hit = np.flatnonzero(fp <= 0.0)
    if hit.size == 0:
        return float("inf")
    i = int(hit[0])
    if i == 0:
        return 0.0
    f0, f1 = fp[i - 1], fp[i]
    return float(Lp[i - 1] + (Lp[i] - Lp[i - 1]) * f0 / (f0 - f1))


# ---------------------------------------------------------------------------
# The one-step operator, assembled once
# ---------------------------------------------------------------------------
class MixtureKernel:
    """Sparse one-step expectation operator of the predictive mixture on a grid.

    `Q(W)[i] = E[ W~(L') | L_i ]`, where `W~` is `W` on the grid and equal to the
    immediate-stopping value `G` beyond its edges (Section 13.8). Every weight is
    non-negative and each row's weights sum to the mass that stays on the grid, so the
    assembled `P` is a genuine sub-stochastic transition matrix: the discretised
    problem is itself a well-defined Markov-chain optimal-stopping problem, which is
    what makes policy iteration exact for it.

    `quadrature='grid'` (default): exact Gaussian densities at the grid nodes with
    trapezoid weights, rows rescaled to the exact normal-CDF mass inside the grid, the
    off-grid remainder integrated against `G` on a fine tail grid. Uniformly O(h^2).
    `quadrature='gauss_hermite'`: the derivation's literal scheme -- `n_quad` nodes per
    component, `W` linearly interpolated at them. Kept as a cross-check.
    """

    def __init__(self, L, I_dt: float, c_e: float, n_quad: int = 40,
                 quadrature: str = "grid"):
        L = np.asarray(L, dtype=float).reshape(-1)
        if L.size < 5:
            raise ValueError("the LLR grid needs at least 5 points")
        h = np.diff(L)
        if not np.allclose(h, h[0], rtol=1e-9, atol=0.0):
            raise ValueError("the LLR grid must be uniform")
        if float(I_dt) <= 0.0:
            raise ValueError("I * Delta_t must be > 0")
        quadrature = str(quadrature).strip().lower()
        if quadrature not in _QUADRATURES:
            raise ValueError(f"quadrature must be one of {_QUADRATURES}; got '{quadrature}'")
        self.L = L
        self.h = float(h[0])
        self.I_dt = float(I_dt)
        self.sd = math.sqrt(2.0 * self.I_dt)
        self.c_e = float(c_e)
        self.n_quad = int(n_quad)
        self.quadrature = quadrature
        self.G = stopping_cost(L, c_e)
        self.p = posterior(L)
        self.i_zero = int(np.argmin(np.abs(L)))

        if quadrature == "grid":
            P_plus, off_plus, mass_plus = self._component_grid(+1.0)
            P_minus, off_minus, mass_minus = self._component_grid(-1.0)
        else:
            pts_plus, pts_minus, w = mixture_nodes(L, I_dt, n_quad)
            P_plus, off_plus, mass_plus = self._component_gh(pts_plus, w)
            P_minus, off_minus, mass_minus = self._component_gh(pts_minus, w)
        p = self.p
        self.P = (sp.diags(p) @ P_plus + sp.diags(1.0 - p) @ P_minus).tocsr()
        self.g_off = p * off_plus + (1.0 - p) * off_minus
        #: fraction of the kernel mass that falls outside the grid, per row
        self.off_mass = p * mass_plus + (1.0 - p) * mass_minus

    # -- grid convolution ----------------------------------------------------------
    def _component_grid(self, sign: float):
        """Banded trapezoid convolution with one Gaussian component, plus its tails."""
        L, h, n, sd = self.L, self.h, self.L.size, self.sd
        mu = L + sign * self.I_dt
        L_max = float(L[-1])
        half = int(math.ceil((_TAIL_SD * sd + self.I_dt) / h)) + 2
        offsets = np.arange(-half, half + 1)
        n_band = offsets.size
        if n * n_band > 5e7:
            logger.warning(
                "bellman_discrete: the grid convolution has %.1e entries (N_L = %d, band "
                "%d); this is far more grid than the kernel needs -- lower N_L.",
                n * n_band, n, n_band,
            )
        # Exact mass inside the grid, per row: what the on-grid weights must sum to.
        q = ndtr((L_max - mu) / sd) - ndtr((-L_max - mu) / sd)

        data, rows, cols = [], [], []
        step = max(1, int(2e6 // n_band))
        for r0 in range(0, n, step):
            r1 = min(n, r0 + step)
            ri = np.arange(r0, r1)
            J = ri[:, None] + offsets[None, :]
            valid = (J >= 0) & (J < n)
            Jc = np.clip(J, 0, n - 1)
            x = (L[Jc] - mu[ri, None]) / sd
            f = np.exp(-0.5 * x * x) * (h / (sd * _SQRT_2PI))
            f[(Jc == 0) | (Jc == n - 1)] *= 0.5          # trapezoid end weights
            f[~valid] = 0.0
            rowsum = f.sum(axis=1)
            scale = np.where(rowsum > 0.0, q[ri] / np.where(rowsum > 0.0, rowsum, 1.0), 0.0)
            f *= scale[:, None]
            R = np.broadcast_to(ri[:, None], J.shape)
            data.append(f[valid])
            rows.append(R[valid])
            cols.append(Jc[valid])
        P = sp.csr_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n),
        )
        P.sum_duplicates()

        # Tails: integral of G against the density beyond +/-L_max, for the rows whose
        # kernel actually reaches the edge (the others carry no off-grid mass).
        mass = np.clip(1.0 - q, 0.0, 1.0)
        off = np.zeros(n)
        idx = np.flatnonzero(mass > 1e-16)
        if idx.size:
            xs = L_max + np.linspace(0.0, _TAIL_SD * sd, _TAIL_POINTS)
            wt = np.full(xs.size, xs[1] - xs[0])
            wt[0] = wt[-1] = 0.5 * wt[0]
            Gt = stopping_cost(xs, self.c_e) * wt
            for r0 in range(0, idx.size, 2048):
                ri = idx[r0:r0 + 2048]
                xr = (xs[None, :] - mu[ri, None]) / sd
                xl = (-xs[None, :] - mu[ri, None]) / sd
                dens = (np.exp(-0.5 * xr * xr) + np.exp(-0.5 * xl * xl)) / (sd * _SQRT_2PI)
                off[ri] = dens @ Gt
        return P, off, mass

    # -- Gauss-Hermite + interpolation (the derivation's literal scheme) ------------
    def _component_gh(self, pts, w):
        """Interpolation matrix of one Gaussian component plus its off-grid remainder."""
        L, h, n = self.L, self.h, self.L.size
        n_rows, n_q = pts.shape
        u = (pts - L[0]) / h
        inside = (pts >= L[0]) & (pts <= L[-1])
        i0 = np.clip(np.floor(u).astype(int), 0, n - 2)
        frac = np.clip(u - i0, 0.0, 1.0)
        W = np.broadcast_to(w[None, :], pts.shape)
        rows = np.broadcast_to(np.arange(n_rows)[:, None], pts.shape)

        r = rows[inside]
        c0 = i0[inside]
        fr = frac[inside]
        ww = W[inside]
        data = np.concatenate([ww * (1.0 - fr), ww * fr])
        cols = np.concatenate([c0, c0 + 1])
        rows_all = np.concatenate([r, r])
        P = sp.csr_matrix((data, (rows_all, cols)), shape=(n_rows, n))
        P.sum_duplicates()

        outside = ~inside
        off = np.where(outside, W * stopping_cost(pts, self.c_e), 0.0).sum(axis=1)
        mass = np.where(outside, W, 0.0).sum(axis=1)
        return P, off, mass

    def Q(self, W: np.ndarray) -> np.ndarray:
        """`E[W(L') | L]` on the grid, the continuation expectation of Section 13.1."""
        return self.P @ W + self.g_off


# ---------------------------------------------------------------------------
# Stationary (midpoint) fixed point
# ---------------------------------------------------------------------------
def stationary_fixed_point(
    kernel: MixtureKernel,
    step_cost: float,
    *,
    tol: float = 1e-10,
    max_iter: Optional[int] = None,
    method: str = "policy_iteration",
    b_init: Optional[float] = None,
) -> dict:
    """Solve `W = min{ G, step_cost + Q(W) }` (Section 13.5) on the kernel's grid.

    `policy_iteration` (default) alternates an exact evaluation of the current
    continuation set -- the linear system `(I - P_CC) W_C = step_cost + g_C + P_CS G_S`
    -- with the improvement `C <- {step_cost + Q(W) < G}`; it terminates in finitely
    many rounds and returns the fixed point to linear-solve precision. `value_iteration`
    is the monotone contraction of Section 13.5 from `W = G`, exact in the limit and
    kept as an independent check (test D1); it needs `O(mean exit steps)` sweeps and is
    used as the fallback if the improvement ever cycles.

    `b_init` seeds policy iteration with the continuation set `|L| < b_init`. Policy
    iteration converges from any start, but from the one-step set `{cost + Q(G) < G}`
    it grows the region roughly one kernel width per round, which at a small
    `Delta_t` means dozens of rounds; the continuous closed form is within a few
    percent of the answer and brings it down to a handful.

    Returns a dict with `W`, `cont` (the continuation value `step_cost + Q(W)`), the
    boundary `b` (LLR), the continuation mask `C`, `iterations`, `residual` (the sup
    norm of `W - min{G, cont}`) and the `method` that produced the answer.
    """
    method = str(method).strip().lower()
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}; got '{method}'")
    G = kernel.G
    step_cost = float(step_cost)
    if step_cost < 0.0:
        raise ValueError("step_cost must be >= 0")

    if method == "value_iteration":
        return _value_iteration(kernel, step_cost, tol, max_iter)

    P, g_off = kernel.P, kernel.g_off
    W = G.copy()
    cont = step_cost + kernel.Q(W)
    C = cont < G
    if b_init is not None and math.isfinite(float(b_init)) and float(b_init) > 0.0:
        C = C | (np.abs(kernel.L) < float(b_init))
    cap = 200 if max_iter is None else int(max_iter)
    seen = set()
    for it in range(1, cap + 1):
        if not C.any():
            W = G.copy()
            cont = step_cost + kernel.Q(W)
            return _pack(kernel, W, cont, step_cost, it, "policy_iteration")
        key = C.tobytes()
        if key in seen:
            logger.warning(
                "bellman_discrete: policy iteration cycled after %d rounds; falling "
                "back to value iteration from the current iterate.", it,
            )
            return _value_iteration(kernel, step_cost, tol, None, W0=W)
        seen.add(key)
        idx = np.flatnonzero(C)
        S = ~C
        P_C = P[idx, :]
        A_CC = (sp.identity(idx.size, format="csc") - P_C[:, idx].tocsc())
        rhs = step_cost + g_off[idx] + P_C[:, np.flatnonzero(S)] @ G[S]
        W = G.copy()
        W[idx] = splu(A_CC).solve(rhs)
        cont = step_cost + kernel.Q(W)
        C_new = cont < G
        if np.array_equal(C_new, C):
            # The evaluation of the final policy is exact; project once so the returned
            # W is the Bellman update of itself to solve precision.
            W = np.minimum(G, cont)
            return _pack(kernel, W, cont, step_cost, it, "policy_iteration")
        C = C_new
    logger.warning(
        "bellman_discrete: policy iteration did not settle in %d rounds; falling back "
        "to value iteration.", cap,
    )
    return _value_iteration(kernel, step_cost, tol, None, W0=W)


def _value_iteration(kernel, step_cost, tol, max_iter, W0=None):
    G = kernel.G
    W = G.copy() if W0 is None else np.minimum(G, np.asarray(W0, dtype=float))
    cap = 2_000_000 if max_iter is None else int(max_iter)
    it = 0
    cont = step_cost + kernel.Q(W)
    for it in range(1, cap + 1):
        Wn = np.minimum(G, cont)
        delta = float(np.max(np.abs(Wn - W)))
        W = Wn
        cont = step_cost + kernel.Q(W)
        if delta < tol:
            break
    else:
        logger.warning(
            "bellman_discrete: value iteration hit max_iter=%d without reaching "
            "tol=%.1e (last change %.2e).", cap, tol, delta,
        )
    return _pack(kernel, W, cont, step_cost, it, "value_iteration")


def _pack(kernel, W, cont, step_cost, iterations, method):
    G = kernel.G
    return {
        "W": W,
        "cont": cont,
        "b": crossing(kernel.L, G, cont),
        "C": cont < G,
        "iterations": int(iterations),
        "residual": float(np.max(np.abs(W - np.minimum(G, cont)))),
        "step_cost": float(step_cost),
        "method": method,
    }


def stationary_statistics(kernel: MixtureKernel, fixed: dict, Delta_t: float) -> dict:
    """Mean exit time and realised error rate of the halted problem from `L = 0`.

    First-passage quantities of the SHARP-threshold process, not of the interpolated
    chain that solves the fixed point: on the continuation nodes `m = 1 + K m` gives
    the mean number of steps and `e = a + K e` the error probability, where `K` is the
    trapezoid Nystrom discretisation of the Gaussian mixture restricted to `(-b, b)`
    (each row rescaled to its exact normal-CDF continuing mass) and `a` integrates the
    posterior error `ER(L') = 1/(1 + e^{|L'|})` of a commitment at the exit point over
    the absorbed tails `|L'| >= b`. The integrands of these two quantities JUMP at the
    boundary, which is why they get their own quadrature rather than the kernel's:
    Gauss-Hermite nodes straddling a jump give errors of several percent. The realised
    error sits below `ER(b)` because the last observation carries the state past the
    boundary (Section 13.6 (iii)); reporting both makes the overshoot visible.
    """
    b = float(fixed["b"])
    L, h, sd, I_dt = kernel.L, kernel.h, kernel.sd, kernel.I_dt
    if not (math.isfinite(b) and b > 0.0):
        return {"halt_mean_exit_time": 0.0, "halt_error_rate": 0.5}
    inside = np.flatnonzero(np.abs(L) < b)
    if inside.size == 0:
        return {"halt_mean_exit_time": 0.0, "halt_error_rate": 0.5}
    Li = L[inside]
    n = Li.size
    p = kernel.p[inside]
    # trapezoid weights on the continuation nodes, the partial cells at +/-b included
    w = np.full(n, h)
    w[0] = 0.5 * h + (Li[0] + b)
    w[-1] = 0.5 * h + (b - Li[-1])
    if n == 1:
        w[0] = 2.0 * b
    xs = b + np.linspace(0.0, _TAIL_SD * sd, 4 * _TAIL_POINTS)      # absorbed tails
    wt = np.full(xs.size, xs[1] - xs[0])
    wt[0] = wt[-1] = 0.5 * wt[0]
    ERt = stopping_cost(xs, 1.0) * wt

    def component(sign):
        mu = Li + sign * I_dt
        x = (Li[None, :] - mu[:, None]) / sd
        K = np.exp(-0.5 * x * x) * (w[None, :] / (sd * _SQRT_2PI))
        q = ndtr((b - mu) / sd) - ndtr((-b - mu) / sd)          # exact continuing mass
        rowsum = K.sum(axis=1)
        K *= (q / np.where(rowsum > 0.0, rowsum, 1.0))[:, None]
        xr = (xs[None, :] - mu[:, None]) / sd
        xl = (-xs[None, :] - mu[:, None]) / sd
        a = ((np.exp(-0.5 * xr * xr) + np.exp(-0.5 * xl * xl)) / (sd * _SQRT_2PI)) @ ERt
        return K, a

    K_plus, a_plus = component(+1.0)
    K_minus, a_minus = component(-1.0)
    K = p[:, None] * K_plus + (1.0 - p)[:, None] * K_minus
    a = p * a_plus + (1.0 - p) * a_minus
    sol = np.linalg.solve(np.eye(n) - K, np.column_stack([np.ones(n), a]))
    j = int(np.argmin(np.abs(Li)))
    return {
        "halt_mean_exit_time": float(sol[j, 0] * Delta_t),
        "halt_error_rate": float(sol[j, 1]),
    }


# ---------------------------------------------------------------------------
# Geometry: the exact per-step cost
# ---------------------------------------------------------------------------
def approach_step_costs(
    r0: float,
    half_L: float,
    v: float,
    Delta_t: float,
    N: int,
    *,
    halt_cost_rate: float = 1.0,
    predecision_motion: str = "midpoint",
) -> np.ndarray:
    """`c_n`, n = 0..N-1, in seconds: the exact net cost of one more observation.

    Section 13.4: a full interval of delay less the travel credit earned by moving
    during it, `c_n = Delta_t + (d_{n+1} - d_n)/v` with `d_n = sqrt(a^2 + y_n^2)`,
    `y_n = max(0, h0 - v n Delta_t)`. This is the integral of the continuous rate
    `c_t (1 + d_dot / v) = c_t (1 - cos(alpha/2))` over the step, so it agrees with the
    Variant C running cost as `Delta_t -> 0` and is exact at any finite interval. A step
    that straddles arrival charges the moving part at the geometric rate and the halted
    remainder at `halt_cost_rate`; once `y_n = 0` every step costs
    `halt_cost_rate * Delta_t`, the halted running cost of Section 5.1.

    Under `predecision_motion: stationary` nothing is recovered by moving, so every
    step costs a full `Delta_t` regardless of the geometry (the continuous code's
    `c_tau = 1`).
    """
    N = int(N)
    Delta_t = float(Delta_t)
    if N < 1 or Delta_t <= 0.0:
        raise ValueError("N >= 1 and Delta_t > 0 are required")
    motion = str(predecision_motion).strip().lower()
    if motion == "stationary":
        return np.full(N, Delta_t)
    v = float(v)
    r0, half_L = float(r0), float(half_L)
    n = np.arange(N + 1, dtype=float)
    if v <= 1e-12:
        # No approach: the geometry is frozen at onset, the rate is 1 - cos(alpha_0/2).
        d0 = math.hypot(half_L, r0)
        rate = 1.0 - (r0 / d0 if d0 > 0.0 else 0.0)
        return np.full(N, Delta_t * rate)
    y = np.maximum(0.0, r0 - v * n * Delta_t)
    d = np.hypot(half_L, y)
    tau_move = np.clip(y[:-1] / v, 0.0, Delta_t)
    return tau_move + (d[1:] - d[:-1]) / v + float(halt_cost_rate) * (Delta_t - tau_move)


# ---------------------------------------------------------------------------
# The solve
# ---------------------------------------------------------------------------
def discrete_bellman_boundary(
    A: float,
    c: float,
    c_e: float,
    Delta_t: float,
    *,
    r0: float,
    half_L: float,
    v: float,
    terminal: str = "forced_choice",
    halt_cost_rate: float = 1.0,
    predecision_motion: str = "midpoint",
    T_max: Optional[float] = None,
    N_L: int = 2001,
    L_max_factor: float = 8.0,
    n_quad: int = 40,
    tol_fixed_point: float = 1e-10,
    method: str = "policy_iteration",
    quadrature: str = "grid",
):
    """Solve the discrete recursion of Section 13 and return `(t_grid, z, diagnostics)`.

    Same output contract as `bellman_boundary`: `t_grid[n] = n * Delta_t` for
    `n = 0..N`, `z[n]` the boundary in EVIDENCE units at that node, and the last row the
    terminal boundary -- `z_mid = b_mid / k` under `halt_sprt` (alias
    `continue_at_midpoint`), `0` under `forced_choice`. `N = ceil(T_arr / Delta_t)` is
    the first lattice step at or after arrival `T_arr = r0 / v` (or the supplied
    `T_max`).

    Inputs are the accumulator's own: `A` the known drift magnitude, `c` the noise
    scale, `c_e` the error cost in seconds, `Delta_t` the interval between the
    observations the runtime actually integrates (the DDM sub-step under `legacy`
    noise; the tick under a shared percept stream, which draws once per tick).

    The grid is sized as in Section 13.8: a provisional pass finds the midpoint
    fixed point, and the working grid spans `+/- L_max_factor * b_mid` in LLR units.
    `diagnostics` carries the LLR-side numbers (`b_mid`, `b_star_continuous`, the
    per-step sd against the threshold), the discrete quasi-static comparator
    `z_myopic` on the same lattice (the comparator the Section 13.6 ordering is exact
    against), its continuous counterpart, the halt statistics, and the solver's own
    numbers.
    """
    from models.bellman_boundary import myopic_z, normalise_terminal, solve_z_halt

    A, c, c_e, Delta_t = abs(float(A)), float(c), float(c_e), float(Delta_t)
    terminal = normalise_terminal(terminal)
    halt_cost_rate = float(halt_cost_rate)
    if A <= 0.0:
        raise ValueError(
            "discrete_bellman_boundary requires a known, non-zero |A| (the policy "
            "assumes the drift magnitude, hence the information rate, is known)."
        )
    if c <= 0.0:
        raise ValueError("discrete_bellman_boundary requires c > 0")
    if c_e < 0.0:
        raise ValueError("discrete_bellman_boundary requires c_e >= 0")
    if Delta_t <= 0.0:
        raise ValueError("discrete_bellman_boundary requires Delta_t > 0")
    if halt_cost_rate <= 0.0:
        raise ValueError("halt_cost_rate must be > 0")
    N_L = int(N_L)
    if N_L < 5:
        raise ValueError("N_L must be >= 5")
    if N_L % 2 == 0:
        N_L += 1          # keep L = 0 on the grid, so the crossing starts at the centre
    L_max_factor = float(L_max_factor)
    if L_max_factor <= 1.0:
        raise ValueError("L_max_factor must be > 1")

    t0 = time.perf_counter()
    k = 2.0 * A / c ** 2
    I = k * A                       # information rate, LLR per second
    I_dt = I * Delta_t
    v = float(v)

    # --- horizon on the step lattice -------------------------------------------------
    if T_max is None:
        T_arr = r0 / v if v > 1e-12 else 10.0
    else:
        T_arr = float(T_max)
        if T_arr <= 0.0:
            raise ValueError("T_max must be > 0")
    N = max(1, int(math.ceil(T_arr / Delta_t - 1e-9)))
    step_costs = approach_step_costs(
        r0, half_L, v, Delta_t, N,
        halt_cost_rate=halt_cost_rate, predecision_motion=predecision_motion,
    )
    halt_step_cost = halt_cost_rate * Delta_t

    # --- continuous closed forms, for sizing and for the discretisation gap -----------
    z_halt_cont = solve_z_halt(k, A, c_e, halt_cost_rate) if c_e > 0.0 else 0.0
    b_star = k * z_halt_cont
    z_qs0_cont = myopic_z(A, c, c_e, step_costs[0] / Delta_t)
    b_qs0_cont = k * z_qs0_cont

    # --- pass 1: provisional grid, midpoint fixed point -----------------------------
    L_max1 = max(8.0, 4.0 * max(b_star, b_qs0_cont))
    kern1 = MixtureKernel(np.linspace(-L_max1, L_max1, N_L), I_dt, c_e, n_quad, quadrature)
    fp1 = stationary_fixed_point(kern1, halt_step_cost, tol=tol_fixed_point, method=method,
                                 b_init=b_star)
    b_mid1 = fp1["b"] if math.isfinite(fp1["b"]) else b_star

    # --- pass 2: working grid ---------------------------------------------------------
    L_max = L_max_factor * max(b_mid1, 0.25 * b_qs0_cont, 0.5)
    sd_step = math.sqrt(2.0 * I_dt)
    N_L_eff = max(N_L, int(math.ceil(2.0 * L_max * _MIN_CELLS_PER_SD / sd_step)) + 1)
    if N_L_eff % 2 == 0:
        N_L_eff += 1
    if N_L_eff > _N_L_CAP:
        logger.warning(
            "bellman_discrete: resolving the per-step kernel (sd %.3g LLR) over "
            "|L| <= %.3g would need N_L = %d; capping at %d. Expect a small bias in "
            "b_n; lower L_max_factor or raise Delta_t.", sd_step, L_max, N_L_eff, _N_L_CAP,
        )
        N_L_eff = _N_L_CAP
    if N_L_eff != N_L:
        logger.info(
            "bellman_discrete: N_L raised from %d to %d so the per-step kernel "
            "(sd %.3g LLR) spans >= %.0f grid cells.", N_L, N_L_eff, sd_step, _MIN_CELLS_PER_SD,
        )
    L = np.linspace(-L_max, L_max, N_L_eff)
    kernel = MixtureKernel(L, I_dt, c_e, n_quad, quadrature)
    G = kernel.G
    fp = stationary_fixed_point(kernel, halt_step_cost, tol=tol_fixed_point, method=method,
                                b_init=b_mid1)
    b_mid = fp["b"]
    if not math.isfinite(b_mid):
        raise ValueError(
            f"no stopping set inside |L| <= {L_max:.3g} at the midpoint: raise "
            "L_max_factor (the halted continuation region does not fit the grid)."
        )
    W_mid = fp["W"]
    stats = stationary_statistics(kernel, fp, Delta_t)

    # --- backward recursion over the approach --------------------------------------
    if terminal == "halt_sprt":
        W = W_mid.copy()
        b_terminal = b_mid
    else:
        W = G.copy()
        b_terminal = 0.0
    b = np.empty(N + 1)
    b[N] = b_terminal
    for n in range(N - 1, -1, -1):
        cont = step_costs[n] + kernel.Q(W)
        b[n] = crossing(L, G, cont)
        W = np.minimum(G, cont)
    W0 = W

    # --- the discrete quasi-static comparator on the same lattice (Section 13.6 ii) --
    costs_needed = np.append(step_costs, halt_step_cost)
    b_qs = _quasi_static_sequence(kernel, costs_needed, halt_step_cost, b_mid,
                                  tol_fixed_point, method,
                                  b_guess=lambda cc: k * myopic_z(A, c, c_e, cc / Delta_t))
    # Its continuous counterpart, the b + sinh b = rho boundary at the frozen RATE.
    z_qs_cont = np.array([myopic_z(A, c, c_e, cc / Delta_t) for cc in costs_needed])

    t_grid = np.arange(N + 1, dtype=float) * Delta_t
    z = b / k
    elapsed = time.perf_counter() - t0
    finite = np.isfinite(b)
    diag = {
        "variant": "discrete",
        "A": A, "c": c, "c_e": c_e, "k": k, "I": I,
        "Delta_t": Delta_t, "N_steps": int(N), "T_arr": float(T_arr),
        "T_max": float(t_grid[-1]),
        "terminal": terminal, "halt_cost_rate": halt_cost_rate,
        "predecision_motion": str(predecision_motion),
        # LLR-side numbers
        "b_mid": float(b_mid), "z_mid": float(b_mid / k),
        "z_halt": float(b_mid / k) if terminal == "halt_sprt" else 0.0,
        "b_star_continuous": float(b_star), "z_star_continuous": float(z_halt_cont),
        "b_onset": float(b[0]) if finite[0] else float("inf"),
        "per_step_llr_sd": sd_step,
        "per_step_sd_over_b_mid": float(sd_step / b_mid) if b_mid > 0.0 else float("inf"),
        # comparators on the lattice, evidence units
        "z_myopic": b_qs / k,
        "z_myopic_onset": float(b_qs[0] / k),
        "z_myopic_continuous": z_qs_cont,
        "z_myopic_continuous_onset": float(z_qs_cont[0]),
        # halt statistics from the absorbed chain
        "halt_mean_exit_time": stats["halt_mean_exit_time"],
        "halt_error_rate": stats["halt_error_rate"],
        "halt_nominal_error_rate": float(1.0 / (1.0 + math.exp(b_mid))),
        # value at onset from a flat prior: the predicted Bayes risk (test D5)
        "W0_at_zero": float(W0[kernel.i_zero]),
        "step_costs": step_costs,
        # solver numbers
        "n_L": int(N_L_eff), "n_L_requested": int(N_L), "dL": kernel.h,
        "cells_per_sd": float(sd_step / kernel.h),
        "L_max": float(L_max), "quadrature": quadrature,
        "n_quad": int(n_quad) if quadrature == "gauss_hermite" else None,
        "off_grid_mass_max": float(np.max(kernel.off_mass)),
        "fixed_point_iterations": fp["iterations"],
        "fixed_point_residual": fp["residual"],
        "fixed_point_method": fp["method"],
        "tol_fixed_point": float(tol_fixed_point),
        "unbounded_fraction": float(np.mean(~finite)),
        "wall_time_s": float(elapsed),
    }
    if diag["unbounded_fraction"] > 0.0:
        logger.warning(
            "bellman_discrete: %.1f%% of the lattice has no stopping set inside "
            "L_max = %.3g; raise L_max_factor.", 100.0 * diag["unbounded_fraction"], L_max,
        )
    elif float(np.max(b[finite])) > 0.75 * L_max:
        logger.warning(
            "bellman_discrete: the boundary reaches %.3g of L_max = %.3g; the grid edge "
            "may be biasing the continuation value. Raise L_max_factor.",
            float(np.max(b[finite])), L_max,
        )
    if diag["fixed_point_residual"] > 100.0 * tol_fixed_point * max(c_e, 1.0):
        logger.warning(
            "bellman_discrete: midpoint fixed-point residual %.2e exceeds the tolerance "
            "%.1e.", diag["fixed_point_residual"], tol_fixed_point,
        )
    logger.info(
        "bellman_discrete: solved N=%d steps at Delta_t=%.4g s (I=%.4g LLR/s, per-step "
        "sd %.3g = %.0f%% of b_mid) in %.2fs | b_mid=%.4g vs continuous b*=%.4g "
        "(%.1f%% below) | z(0)=%.4g vs discrete quasi-static %.4g | halt: mean exit "
        "%.2fs, realised ER %.4f vs nominal %.4f",
        N, Delta_t, I, sd_step, 100.0 * diag["per_step_sd_over_b_mid"], elapsed,
        b_mid, b_star, 100.0 * (b_star - b_mid) / max(b_star, 1e-12),
        z[0], diag["z_myopic_onset"], diag["halt_mean_exit_time"],
        diag["halt_error_rate"], diag["halt_nominal_error_rate"],
    )
    return t_grid, z, diag


def _quasi_static_sequence(kernel, costs, halt_step_cost, b_mid, tol, method,
                           b_guess=None):
    """`b_qs` for each per-step cost: the stationary fixed point at that cost frozen
    forever. Exact at every distinct cost when there are few; otherwise solved on a
    log-spaced set of nodes (endpoints included) and interpolated in `log(cost)`.
    `b_guess(cost)` seeds each solve (the continuous closed form at that cost)."""
    costs = np.asarray(costs, dtype=float)
    uniq = np.unique(costs)
    cache = {float(halt_step_cost): float(b_mid)}

    def solve(cc):
        key = float(cc)
        if key not in cache:
            cache[key] = float(stationary_fixed_point(
                kernel, key, tol=tol, method=method,
                b_init=None if b_guess is None else b_guess(key),
            )["b"])
        return cache[key]

    if uniq.size <= _QS_MAX_EXACT:
        table = {float(cc): solve(cc) for cc in uniq}
        return np.array([table[float(cc)] for cc in costs])
    lo, hi = float(uniq[0]), float(uniq[-1])
    nodes = np.exp(np.linspace(math.log(max(lo, 1e-300)), math.log(hi), _QS_INTERP_NODES))
    nodes[0], nodes[-1] = lo, hi
    vals = np.array([solve(cc) for cc in nodes])
    return np.interp(np.log(costs), np.log(nodes), vals)
