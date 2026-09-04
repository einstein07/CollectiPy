# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Validation battery for the DISCRETE (sampled-evidence) Bellman boundary
(BELLMAN_KNOWN_A_DERIVATION Section 13.9, tests D1-D9), plus the worked example of
Section 13.6 and the wiring through the embodied pure-DDM model.

  D1  midpoint Bellman residual; policy iteration == value iteration     <- GATE
  D2  symmetry of W_n and of the extracted boundary
  D3  posterior martingale under the predictive MIXTURE (+ negative control)  <- GATE
  D4  LLR-grid and quadrature convergence (no Delta_t convergence: it is the task)
  D5  Monte Carlo agreement: realised Bayes risk = W_0(0); realised error =
      mean posterior error at the stopping LLR, NOT 1/(1+e^b)
  D6  high-time-cost limit: b_n -> 0
  D7  diffusion limit: b_mid -> b* from below, gap ~ sqrt(Delta_t); the approach
      boundaries converge to the continuous solve
  D8  orderings: b_n <= b_qs,n (discrete comparator) with equality at arrival,
      b_n >= b_mid, halt >= forced, forced b_N = 0 < b_{N-1}
  D9  local perturbation: regret minimised at gamma = 1

The worked example of Section 13.6 (calibrated arena, I = 0.6 LLR/s, C_e/c_t = 30 s,
Delta_t = 1 s) is the numerical regression against the derivation document itself:
b_mid = 1.93, b* = 2.56, forced-choice b_{N-1} = 1.17, per-step costs 0.145 / 0.946,
realised halt error 0.071 against a nominal 0.127. The document's figures come from
its literal Gauss-Hermite scheme (40 nodes); the grid convolution used here converges
to b_mid = 1.9253 and b_{N-1} = 1.160, and `quadrature='gauss_hermite'` reproduces the
document's 1.928 / 1.171 -- both are checked.

Run with:
    cd CollectiPy && env -u PYTHONPATH .venv/bin/python -m pytest \
        tests/test_bellman_discrete.py -q -p no:cacheprovider
"""

import math
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "src")
for p in (_SRC, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.bellman_boundary import (  # noqa: E402
    bellman_boundary, myopic_z, normalise_terminal,
)
from models.bellman_discrete import (  # noqa: E402
    MixtureKernel, approach_step_costs, crossing, discrete_bellman_boundary,
    gaussian_expectation, mixture_expectation, posterior, stationary_fixed_point,
    stationary_statistics, stopping_cost,
)

# --- Section 13.6 worked example ---------------------------------------------
# a = 0.25 m, h0 = 0.433 m, v = 0.04 m/s, I = 0.6 LLR/s, C_e/c_t = 30 s, Delta_t = 1 s.
# In evidence coordinates: A = 0.06 and c chosen so I = 2A^2/c^2 = 0.6, hence k = 10.
I_EX = 0.6
A_EX = 0.06
C_EX = math.sqrt(2.0 * A_EX ** 2 / I_EX)
K_EX = 2.0 * A_EX / C_EX ** 2
CE_EX = 30.0
R0, HALF_L, V_EX = 0.433, 0.25, 0.04
GEOM = dict(r0=R0, half_L=HALF_L, v=V_EX)


def _solve(dt, terminal="halt_sprt", **kw):
    args = dict(GEOM)
    args.update(kw)
    return discrete_bellman_boundary(A_EX, C_EX, CE_EX, dt, terminal=terminal, **args)


@pytest.fixture(scope="module")
def example():
    """The worked example, both terminal conditions, solved once."""
    th, zh, dh = _solve(1.0, "halt_sprt")
    tf, zf, df = _solve(1.0, "forced_choice")
    return {"t": th, "b_halt": zh * K_EX, "dh": dh, "b_forced": zf * K_EX, "df": df}


def _b_star(rho):
    """`b + sinh b = rho`, solved here independently of the code under test."""
    lo, hi = 0.0, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid + math.sinh(mid) < rho:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# The worked example -- regression against the derivation document
# ---------------------------------------------------------------------------
def test_worked_example_reproduces_section_13_6(example):
    dh, df = example["dh"], example["df"]
    assert dh["N_steps"] == 11                                # ceil(10.825 / 1)
    assert abs(dh["b_mid"] - 1.93) < 0.02, dh["b_mid"]
    # b* + sinh b* = C_e I / 2 = 9 -> 2.561, computed independently here
    assert abs(dh["b_star_continuous"] - _b_star(CE_EX * I_EX / 2.0)) < 1e-6
    assert abs(dh["b_star_continuous"] - 2.56) < 0.01
    # exact per-step cost, Section 13.4: 0.145 at the first step, 0.946 at the last
    assert abs(dh["step_costs"][0] - 0.145) < 0.002
    assert abs(dh["step_costs"][-1] - 0.946) < 0.002
    # forced choice: b_N = 0, and the last boundary before arrival is 1.17 (1.160 once
    # the quadrature is converged; the document's 1.171 is its 40-node Gauss-Hermite)
    assert example["b_forced"][-1] == 0.0
    assert abs(example["b_forced"][-2] - 1.17) < 0.02, example["b_forced"][-2]
    tg, zg, dg = _solve(1.0, "halt_sprt", quadrature="gauss_hermite", n_quad=40)
    tgf, zgf, _ = _solve(1.0, "forced_choice", quadrature="gauss_hermite", n_quad=40)
    assert abs(dg["b_mid"] - 1.928) < 0.002 and abs(zgf[-2] * K_EX - 1.171) < 0.002
    assert dg["quadrature"] == "gauss_hermite" and dh["quadrature"] == "grid"
    # the halt table ends on the plateau
    assert example["b_halt"][-1] == pytest.approx(dh["b_mid"])
    # nominal vs realised error at the midpoint (overshoot, Section 13.6 (iii))
    assert abs(dh["halt_nominal_error_rate"] - 0.127) < 0.002
    assert dh["halt_error_rate"] < dh["halt_nominal_error_rate"]
    assert abs(dh["halt_error_rate"] - 0.071) < 0.003
    # per-step LLR increment: mean 0.60, sd 1.10
    assert abs(dh["per_step_llr_sd"] - 1.10) < 0.01
    assert dh["variant"] == "discrete" and dh["terminal"] == "halt_sprt"


# ---------------------------------------------------------------------------
# D1 -- the midpoint fixed point
# ---------------------------------------------------------------------------
def test_d1_fixed_point_residual_and_method_agreement():
    kern = MixtureKernel(np.linspace(-16.0, 16.0, 2001), I_EX * 1.0, CE_EX, 40)
    pi = stationary_fixed_point(kern, 1.0, tol=1e-10)
    vi = stationary_fixed_point(kern, 1.0, tol=1e-13, method="value_iteration")
    assert pi["method"] == "policy_iteration" and vi["method"] == "value_iteration"
    # Bellman residual below the tolerance, both ways
    for fp in (pi, vi):
        W = fp["W"]
        again = np.minimum(kern.G, 1.0 + kern.Q(W))
        assert np.max(np.abs(W - again)) < 1e-9 * CE_EX
    # the two solvers agree, in value and in boundary
    assert np.max(np.abs(pi["W"] - vi["W"])) < 1e-8
    assert abs(pi["b"] - vi["b"]) < 1e-7
    # the boundary read from the converged slice IS b_mid
    assert crossing(kern.L, kern.G, 1.0 + kern.Q(pi["W"])) == pytest.approx(pi["b"], abs=1e-9)
    assert pi["iterations"] < 30 and vi["iterations"] > pi["iterations"]


def test_d1_warm_start_does_not_change_the_answer():
    kern = MixtureKernel(np.linspace(-16.0, 16.0, 2001), I_EX * 0.25, CE_EX, 40)
    cold = stationary_fixed_point(kern, 0.25)
    warm = stationary_fixed_point(kern, 0.25, b_init=2.4)
    assert abs(cold["b"] - warm["b"]) < 1e-9
    assert np.max(np.abs(cold["W"] - warm["W"])) < 1e-9
    assert warm["iterations"] <= cold["iterations"]


# ---------------------------------------------------------------------------
# D2 -- symmetry
# ---------------------------------------------------------------------------
def test_d2_symmetry_of_value_and_boundary(example):
    kern = MixtureKernel(np.linspace(-16.0, 16.0, 2001), I_EX * 1.0, CE_EX, 40)
    fp = stationary_fixed_point(kern, 1.0)
    W = fp["W"]
    assert np.max(np.abs(W - W[::-1])) < 1e-12 * CE_EX
    cont = 1.0 + kern.Q(W)
    b_pos = crossing(kern.L, kern.G, cont)
    b_neg = crossing(-kern.L[::-1], kern.G[::-1], cont[::-1])     # the L <= 0 side, mirrored
    assert b_neg == pytest.approx(b_pos, abs=1e-12)
    # and the recursion keeps the symmetry through the approach
    W = W.copy()
    for cost in example["dh"]["step_costs"][::-1]:
        W = np.minimum(kern.G, cost + kern.Q(W))
        assert np.max(np.abs(W - W[::-1])) < 1e-12 * CE_EX


# ---------------------------------------------------------------------------
# D3 -- the kernel is the mixture, and the posterior is its martingale
# ---------------------------------------------------------------------------
def test_d3_posterior_is_a_martingale_under_the_mixture_only():
    L = np.linspace(-8.0, 8.0, 321)
    for I_dt in (0.6, 0.06, 0.006):
        err_mix = np.max(np.abs(mixture_expectation(posterior, L, I_dt) - posterior(L)))
        assert err_mix < 1e-12, f"mixture identity fails at I dt = {I_dt}: {err_mix:.2e}"
    # Negative control: the single Gaussian with the posterior-mean drift and variance
    # 2 I dt is NOT a martingale kernel at a finite interval (7e-3 at Delta_t = 1 s in
    # the worked example, Section 13.3), and the failure shrinks as Delta_t -> 0.
    err_g1 = np.max(np.abs(gaussian_expectation(posterior, L, 0.6) - posterior(L)))
    err_g2 = np.max(np.abs(gaussian_expectation(posterior, L, 0.006) - posterior(L)))
    assert 3e-3 < err_g1 < 2e-2, err_g1
    assert err_g2 < err_g1 / 10.0
    # The assembled grid operator (Section 13.8's "convolution on the grid") must
    # reproduce the martingale identity wherever the grid edge -- beyond which G stands
    # in for the function -- is many kernel widths away: trapezoid on a smooth
    # integrand, so to ~1e-7 here, against ~1e-3 for the Gauss-Hermite operator whose
    # nodes sample the interpolant.
    kern = MixtureKernel(L, 0.6, CE_EX)
    inner = np.abs(L) < 2.0
    assert np.max(np.abs(kern.Q(posterior(L))[inner] - posterior(L)[inner])) < 1e-6
    gh = MixtureKernel(L, 0.6, CE_EX, n_quad=40, quadrature="gauss_hermite")
    err_gh = np.max(np.abs(gh.Q(posterior(L))[inner] - posterior(L)[inner]))
    assert 1e-6 < err_gh < 5e-3, err_gh
    assert np.max(np.abs(kern.Q(posterior(L)) - gh.Q(posterior(L)))[inner]) < 5e-3
    # rows sum to one: on-grid mass plus the off-grid remainder, both operators
    for k_ in (kern, gh):
        assert np.allclose(np.asarray(k_.P.sum(axis=1)).ravel() + k_.off_mass, 1.0, atol=1e-12)
        assert np.all(k_.P.data >= 0.0)


def test_d3_kernel_rejects_bad_inputs():
    with pytest.raises(ValueError, match="uniform"):
        MixtureKernel(np.array([0.0, 1.0, 3.0, 4.0, 5.0]), 0.1, 1.0)
    with pytest.raises(ValueError):
        MixtureKernel(np.linspace(-1, 1, 11), 0.0, 1.0)
    with pytest.raises(ValueError, match="quadrature"):
        MixtureKernel(np.linspace(-1, 1, 11), 0.1, 1.0, quadrature="simpson")
    kern = MixtureKernel(np.linspace(-1, 1, 11), 0.1, 1.0)
    with pytest.raises(ValueError, match="method"):
        stationary_fixed_point(kern, 0.1, method="nonsense")


# ---------------------------------------------------------------------------
# D4 -- grid and quadrature convergence
# ---------------------------------------------------------------------------
def test_d4_grid_and_quadrature_convergence():
    """b_mid stable under grid doubling (second order: the 1001 -> 2001 change is a
    few 1e-4 and 2001 -> 4001 a few 1e-6), and the two operators agree to well inside
    the 0.5% the derivation asks of n_quad 20 -> 40 -> 80."""
    T = dict(T_max=0.5)                                   # b_mid only; keep it quick
    b = {N_L: _solve(1.0, N_L=N_L, **T)[2]["b_mid"] for N_L in (1001, 2001, 4001)}
    assert abs(b[1001] - b[4001]) < 5e-4
    assert abs(b[2001] - b[4001]) < 5e-5
    assert abs(b[2001] - b[4001]) < abs(b[1001] - b[4001]) / 4.0     # ~O(h^2)
    for nq in (20, 40, 80):
        bg = _solve(1.0, quadrature="gauss_hermite", n_quad=nq, **T)[2]["b_mid"]
        assert abs(bg - b[4001]) < 5e-3 * b[4001], (nq, bg, b[4001])


def test_d4_grid_is_refined_automatically_for_a_narrow_kernel():
    # Delta_t = 2 ms: per-step sd 0.049 LLR against a +/-19 grid needs ~6k points.
    # T_max keeps the approach short; only the grid sizing is under test here.
    _, _, d = _solve(0.002, T_max=0.05)
    assert d["n_L_requested"] == 2001 and d["n_L"] > 2001
    assert d["cells_per_sd"] >= 8.0 - 1e-9
    assert d["b_mid"] < d["b_star_continuous"]      # from below, once resolved


# ---------------------------------------------------------------------------
# Monte Carlo helpers (D5, D9): the step process with the Section 13.5 event order
# ---------------------------------------------------------------------------
def _simulate(b_seq, step_costs, halt_step_cost, I_dt, c_e, n_trials, seed,
              b_plateau, max_extra=2000):
    """`L_0 = 0`; at step n compare `|L_n| >= b_n` (b_n = plateau past the table),
    commit to `sign(L_n)`; otherwise pay `c_n` and receive one observation
    `N(H I dt, 2 I dt)`, `H` drawn from the flat prior. Returns per-trial regret, error
    indicator, stopping LLR and stopping step."""
    rng = np.random.default_rng(seed)
    H = np.where(rng.random(n_trials) < 0.5, 1.0, -1.0)
    L = np.zeros(n_trials)
    cost = np.zeros(n_trials)
    live = np.ones(n_trials, dtype=bool)
    err = np.zeros(n_trials, dtype=bool)
    L_stop = np.zeros(n_trials)
    n_stop = np.zeros(n_trials, dtype=int)
    N = len(step_costs)
    sd = math.sqrt(2.0 * I_dt)
    for n in range(N + max_extra):
        if not live.any():
            break
        b_n = b_seq[n] if n < len(b_seq) else b_plateau
        hit = live & (np.abs(L) >= b_n)
        if hit.any():
            err[hit] = np.sign(L[hit]) != H[hit]
            L_stop[hit] = L[hit]
            n_stop[hit] = n
            live &= ~hit
        c_n = step_costs[n] if n < N else halt_step_cost
        cost[live] += c_n
        m = int(live.sum())
        L[live] += H[live] * I_dt + sd * rng.standard_normal(m)
    assert not live.any(), "trials still undecided at the end of the horizon"
    regret = cost + c_e * err
    return {"regret": regret, "err": err, "L_stop": L_stop, "n_stop": n_stop}


def _paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(d.size))


# ---------------------------------------------------------------------------
# D5 -- Monte Carlo agreement
# ---------------------------------------------------------------------------
def test_d5_realised_bayes_risk_matches_w0_and_error_is_the_posterior_mean(example):
    dh = example["dh"]
    r = _simulate(example["b_halt"], dh["step_costs"], dh["halt_cost_rate"] * dh["Delta_t"],
                  I_EX * dh["Delta_t"], CE_EX, n_trials=200_000, seed=3,
                  b_plateau=dh["b_mid"])
    # realised Bayes risk from a flat prior = W_0(0)
    se = r["regret"].std(ddof=1) / math.sqrt(r["regret"].size)
    assert abs(r["regret"].mean() - dh["W0_at_zero"]) < 3.5 * se, (
        f"regret {r['regret'].mean():.4f} +/- {se:.4f} vs W0 {dh['W0_at_zero']:.4f}"
    )
    # realised error = E[1/(1+e^{|L_tau|})] (the martingale identity), per trial paired
    post_err = 1.0 / (1.0 + np.exp(np.abs(r["L_stop"])))
    d, se_d = _paired(r["err"].astype(float), post_err)
    assert abs(d) < 3.5 * se_d, f"error {r['err'].mean():.4f} vs mean posterior error {post_err.mean():.4f}"
    # ... and NOT 1/(1+e^{b}) at the boundary actually crossed: overshoot buys accuracy
    b_at_stop = np.where(r["n_stop"] < len(example["b_halt"]),
                         example["b_halt"][np.minimum(r["n_stop"], len(example["b_halt"]) - 1)],
                         dh["b_mid"])
    nominal = 1.0 / (1.0 + np.exp(b_at_stop))
    d2, se2 = _paired(r["err"].astype(float), nominal)
    assert d2 < -4.0 * se2, "realised error should sit significantly below the nominal ER(b)"


def test_d5_halted_problem_statistics_match_the_absorbed_chain(example):
    dh = example["dh"]
    dt = dh["Delta_t"]
    # Flat plateau from L = 0: pure halt phase.
    r = _simulate(np.array([]), np.array([]), dh["halt_cost_rate"] * dt, I_EX * dt, CE_EX,
                  n_trials=200_000, seed=7, b_plateau=dh["b_mid"])
    t_exit = r["n_stop"] * dt
    se_t = t_exit.std(ddof=1) / math.sqrt(t_exit.size)
    assert abs(t_exit.mean() - dh["halt_mean_exit_time"]) < 3.5 * se_t, (
        f"mean exit {t_exit.mean():.3f} vs chain {dh['halt_mean_exit_time']:.3f} ({se_t:.3f} SE)"
    )
    e = float(r["err"].mean())
    se_e = math.sqrt(e * (1.0 - e) / r["err"].size)
    assert abs(e - dh["halt_error_rate"]) < 3.5 * se_e, (
        f"error {e:.4f} vs chain {dh['halt_error_rate']:.4f} ({se_e:.4f} SE)"
    )
    assert dh["halt_nominal_error_rate"] - e > 10.0 * se_e


# ---------------------------------------------------------------------------
# D6 -- high-time-cost limit
# ---------------------------------------------------------------------------
def test_d6_high_time_cost_collapses_the_boundary():
    _, z, d = discrete_bellman_boundary(A_EX, C_EX, 1e-3, 1.0, terminal="halt_sprt", **GEOM)
    assert np.all(z == 0.0), z
    assert d["b_mid"] == 0.0 and d["halt_mean_exit_time"] == 0.0
    assert d["halt_error_rate"] == 0.5


# ---------------------------------------------------------------------------
# D7 -- the diffusion limit
# ---------------------------------------------------------------------------
def test_d7_b_mid_converges_to_b_star_from_below_like_sqrt_dt():
    dts = (1.0, 0.5, 0.25, 0.125)
    out = [_solve(dt, T_max=0.5)[2] for dt in dts]          # short approach: b_mid only
    b_star = out[0]["b_star_continuous"]
    b_mid = np.array([d["b_mid"] for d in out])
    assert np.all(np.diff(b_mid) > 0.0), b_mid              # monotone increase
    assert np.all(b_mid < b_star)                            # from below
    ratio = (b_star - b_mid) / np.sqrt(dts)
    assert np.ptp(ratio) < 0.15 * ratio.mean(), ratio        # gap ~ sqrt(Delta_t)
    # the Section 13.7 figure: 1.93 at 1 s, ~2.11 at 0.5 s
    assert abs(b_mid[0] - 1.93) < 0.02 and abs(b_mid[1] - 2.11) < 0.02


def test_d7_approach_boundaries_converge_to_the_continuous_solve():
    T_arr = R0 / V_EX

    def c_tau(t):
        remaining = max(R0 - V_EX * float(t), 1e-6)
        return 1.0 - math.cos(math.atan2(HALF_L, remaining))

    tc, zc, dc = bellman_boundary(A_EX, C_EX, CE_EX, c_tau, T_arr, N_x=801, N_t=10000,
                                  horizon_check_factor=None, terminal="halt_sprt")
    gaps = []
    for dt in (0.4, 0.1, 0.025):
        t, z, d = _solve(dt)
        inside = t <= T_arr
        zc_on = np.interp(t[inside], tc, zc)
        gaps.append(float(np.max(np.abs(z[inside] - zc_on) / zc_on)))
        assert np.all(z[inside] <= zc_on * 1.02), "discrete boundary should not exceed the continuous one"
    assert gaps[1] < gaps[0] / 1.5 and gaps[2] < gaps[1] / 1.5, gaps
    assert gaps[2] < 0.06, gaps


# ---------------------------------------------------------------------------
# D8 -- orderings
# ---------------------------------------------------------------------------
def test_d8_orderings(example):
    dh = example["dh"]
    b_h, b_f = example["b_halt"], example["b_forced"]
    b_qs = dh["z_myopic"] * K_EX
    # Bellman <= discrete quasi-static comparator, equal at arrival
    assert np.all(b_h <= b_qs + 1e-9), (b_h, b_qs)
    assert b_h[-1] == pytest.approx(b_qs[-1], abs=1e-12)
    assert np.all(b_h[:-1] < b_qs[:-1])
    # the gap is anticipation only: the comparator itself is discrete, hence below the
    # continuous closed form (that gap would be discretisation, Section 13.6 (ii))
    b_qs_cont = dh["z_myopic_continuous"] * K_EX
    assert np.all(b_qs < b_qs_cont)
    # never below the plateau; halt mode nests forced mode; forced ends in a jump
    assert np.all(b_h >= dh["b_mid"] - 1e-9)
    assert np.all(b_f <= b_h + 1e-9)
    assert b_f[-1] == 0.0 and b_f[-2] > 0.5
    # per-step cost is increasing on the approach and the boundary falls with it
    assert np.all(np.diff(dh["step_costs"]) > 0.0)
    assert np.all(np.diff(b_h) < 0.0)


def test_d8_step_costs_are_the_exact_integral_of_the_continuous_rate():
    dt, N = 1.0, 11
    c = approach_step_costs(R0, HALF_L, V_EX, dt, N)
    # the continuous rate c_t (1 - cos(alpha/2)) integrated exactly over each step
    from scipy.integrate import quad

    def rate(t):
        y = max(R0 - V_EX * t, 0.0)
        return 1.0 - (y / math.hypot(HALF_L, y) if V_EX * t < R0 else 0.0)

    for n in range(N):
        exact, _ = quad(rate, n * dt, (n + 1) * dt, points=[R0 / V_EX] if n * dt < R0 / V_EX < (n + 1) * dt else None)
        assert abs(c[n] - exact) < 1e-9, (n, c[n], exact)
    # past arrival every step costs a full halt interval; stationary costs it always
    assert approach_step_costs(R0, HALF_L, V_EX, dt, 20)[-1] == pytest.approx(dt)
    assert np.allclose(approach_step_costs(R0, HALF_L, V_EX, dt, 5, predecision_motion="stationary"), dt)
    # a step straddling arrival splits: moving part at the geometric rate, rest at c_h
    c2 = approach_step_costs(R0, HALF_L, V_EX, dt, 20, halt_cost_rate=2.0)
    assert c2[-1] == pytest.approx(2.0 * dt)
    tau = (R0 - 10 * V_EX * dt) / V_EX        # the moving fraction of step 10
    d10 = math.hypot(HALF_L, R0 - 10 * V_EX * dt)
    assert c2[10] == pytest.approx(tau + (HALF_L - d10) / V_EX + 2.0 * (dt - tau))


# ---------------------------------------------------------------------------
# D9 -- local perturbation
# ---------------------------------------------------------------------------
def test_d9_regret_is_minimised_at_gamma_one(example):
    dh = example["dh"]
    gammas = (0.6, 0.8, 1.0, 1.25, 1.5)
    out = {
        g: _simulate(example["b_halt"] * g, dh["step_costs"],
                     dh["halt_cost_rate"] * dh["Delta_t"], I_EX * dh["Delta_t"], CE_EX,
                     n_trials=150_000, seed=11, b_plateau=dh["b_mid"] * g)["regret"]
        for g in gammas
    }
    detail = "  ".join(f"g={g:.2f}:{out[g].mean():.4f}" for g in gammas)
    means = np.array([out[g].mean() for g in gammas])
    assert gammas[int(np.argmin(means))] == 1.0, detail
    for g in gammas:
        if g == 1.0:
            continue
        d, se = _paired(out[g], out[1.0])
        assert d > -2.0 * se, f"gamma={g} beats the solved policy: {d:+.4f} +/- {se:.4f}. {detail}"
    for g in (0.6, 1.5):
        d, se = _paired(out[g], out[1.0])
        assert d > 3.0 * se, f"gamma={g} not significantly worse: {d:+.4f} +/- {se:.4f}. {detail}"


# ---------------------------------------------------------------------------
# Terminal alias, validation, solver contract
# ---------------------------------------------------------------------------
def test_continue_at_midpoint_is_an_alias_of_halt_sprt():
    assert normalise_terminal("continue_at_midpoint") == "halt_sprt"
    assert normalise_terminal(" Halt_SPRT ") == "halt_sprt"
    assert normalise_terminal("forced_choice") == "forced_choice"
    with pytest.raises(ValueError, match="terminal must be"):
        normalise_terminal("halt")
    t1, z1, d1 = _solve(1.0, "continue_at_midpoint")
    t2, z2, d2 = _solve(1.0, "halt_sprt")
    assert np.array_equal(z1, z2) and d1["terminal"] == "halt_sprt"


def test_solver_contract_and_validation():
    t, z, d = _solve(1.0)
    assert t[0] == 0.0 and np.allclose(np.diff(t), 1.0) and len(t) == d["N_steps"] + 1
    assert d["T_max"] == pytest.approx(t[-1]) and t[-1] >= R0 / V_EX
    assert z[-1] == pytest.approx(d["z_halt"]) and d["z_halt"] == pytest.approx(d["b_mid"] / K_EX)
    assert d["z_myopic"].shape == z.shape
    for bad in (dict(Delta_t=0.0), dict(c=0.0), dict(A=0.0), dict(c_e=-1.0)):
        kw = dict(A=A_EX, c=C_EX, c_e=CE_EX, Delta_t=1.0)
        kw.update(bad)
        with pytest.raises(ValueError):
            discrete_bellman_boundary(kw["A"], kw["c"], kw["c_e"], kw["Delta_t"], **GEOM)
    with pytest.raises(ValueError, match="terminal must be"):
        _solve(1.0, "nonsense")
    with pytest.raises(ValueError, match="method"):
        _solve(1.0, method="nonsense")
    with pytest.raises(ValueError, match="quadrature"):
        _solve(1.0, quadrature="nonsense")
    # T_max override shortens the lattice
    assert _solve(1.0, T_max=3.0)[2]["N_steps"] == 3


def test_stationary_statistics_degenerate_case():
    kern = MixtureKernel(np.linspace(-8.0, 8.0, 801), 0.6, 1e-6, 40)
    fp = stationary_fixed_point(kern, 1.0)
    assert fp["b"] == 0.0
    st = stationary_statistics(kern, fp, 1.0)
    assert st == {"halt_mean_exit_time": 0.0, "halt_error_rate": 0.5}
    # stopping cost is overflow-free far out
    assert stopping_cost(np.array([1e4, -1e4]), 1.0).tolist() == [0.0, 0.0]


# ---------------------------------------------------------------------------
# Cache key hygiene and round trip
# ---------------------------------------------------------------------------
def test_cache_key_separates_variants_and_legacy_keys_are_untouched(tmp_path):
    from models.bellman_table_cache import load_table, save_table, table_key
    base = dict(A=0.12, c=0.35, c_e=5.0, r0=0.4, L=0.5, v=0.05, T_max=8.0,
                N_x=801, N_t=8000, X_max_factor=4.0, scheme="crank_nicolson")
    legacy = table_key(**base)
    assert table_key(variant="continuous", **base) == legacy
    disc = table_key(variant="discrete", Delta_t=0.1, N_L=2001, L_max_factor=8.0,
                     n_quad=40, tol_fixed_point=1e-10, predecision_motion="midpoint", **base)
    assert disc != legacy
    assert table_key(variant="discrete", Delta_t=0.05, N_L=2001, **base) != disc
    assert table_key(variant="discrete", Delta_t=0.1, N_L=4001, **base) != disc
    assert table_key(variant="discrete", Delta_t=0.1, N_L=2001, L_max_factor=8.0,
                     n_quad=40, tol_fixed_point=1e-10, predecision_motion="midpoint",
                     quadrature="gauss_hermite", **base) != disc
    # round trip with the comparator array and the scalar extras
    t = np.arange(4) * 0.1
    z = np.array([0.5, 0.4, 0.3, 0.25])
    zqs = np.array([0.55, 0.45, 0.35, 0.25])
    path = save_table(tmp_path, disc, t, z, inputs={"A": 0.12}, z_myopic_onset=0.55,
                      wall_time_s=1.5, scheme="discrete", z_myopic_arr=zqs,
                      extras={"halt_mean_exit_time": 3.8, "b_mid": 1.93})
    assert path is not None
    t2, z2, meta = load_table(tmp_path, disc)
    assert np.array_equal(t2, t) and np.array_equal(z2, z)
    assert np.array_equal(meta["z_myopic_arr"], zqs)
    assert meta["extras"] == {"halt_mean_exit_time": 3.8, "b_mid": 1.93}
    assert meta["scheme"] == "discrete"
    # a table written without the optional fields loads as before
    save_table(tmp_path, legacy, t, z, inputs={"A": 0.12}, z_myopic_onset=0.5,
               wall_time_s=1.0, scheme="crank_nicolson")
    _, _, meta_old = load_table(tmp_path, legacy)
    assert meta_old["z_myopic_arr"] is None and meta_old["extras"] == {}


# ---------------------------------------------------------------------------
# Wiring through the embodied pure-DDM model
# ---------------------------------------------------------------------------
def _build_model(bellman=None, ticks=10, n_sub=4, sensory_stream=None, **cfg_overrides):
    """Arena-free stub agent running the full embodied_pure_ddm pipeline."""
    import models  # noqa: F401  (registers movement models)
    from geometry_utils.vector3D import Vector3D
    from plugin_registry import get_movement_model
    from test_midpoint_readout import _Shape, _StubAgent

    cfg = {
        "target_ids": ["static_0.s#0", "static_1.s#0"],
        "eta_rate": [0.25, 0.25],
        "n_sub": n_sub,
        "threshold_policy": "bellman",
        "boundary_mode": "static",
        "cost_ratio": 6.0,
        "z_min": 1e-6,
        "predecision_motion": "midpoint",
        "scaling_mode": "constant",
        "num_neurons": 30,
        "perception_width": 0.2,
        "bellman": {"variant": "discrete", "terminal": "continue_at_midpoint",
                    **(bellman or {})},
    }
    cfg.update(cfg_overrides)
    agent_cfg = {"embodied_pure_ddm": cfg, "detection": "GPS"}
    if sensory_stream is not None:
        agent_cfg["sensory_stream"] = sensory_stream
        agent_cfg["arena_ticks_per_second"] = ticks
    agent = _StubAgent(agent_cfg, ticks_per_second=ticks)
    agent.max_absolute_velocity = 0.1 / ticks       # 0.1 m/s: arrival at ~2.2 s
    model = get_movement_model("embodied_pure_ddm", agent)
    objs = {}
    for i, (sign, strength) in enumerate(((-1.0, 5.0), (1.0, 4.7))):
        ang = sign * 0.5
        p = Vector3D(0.25 * math.cos(ang), -0.25 * math.sin(ang), 0.0)
        eid = f"static_{i}.s#0"
        objs[eid] = ([_Shape(eid, p)], [p], [strength], [0.0])
    return agent, model, objs


def test_wired_model_solves_on_the_accumulators_own_lattice():
    agent, model, objs = _build_model(ticks=10, n_sub=4)
    assert model.bellman_variant == "discrete"
    assert model.bellman_terminal == "halt_sprt"            # alias normalised
    model.ddm.reset(seed=1)
    model.step(agent, 0, None, objs, {})                    # onset: the solve
    assert model._bellman_solved
    dt_obs = 1.0 / (10 * 4)                                 # legacy noise: dt / n_sub
    assert model._bellman_Delta_t == pytest.approx(dt_obs)
    t_tab, z_tab = model.ddm._z_table_t, model.ddm._z_table_z
    assert np.allclose(np.diff(t_tab), dt_obs)
    assert t_tab[-1] >= model._bellman_T_max - 1e-12 and t_tab[-1] == pytest.approx(model._bellman_T_max)
    assert model._bellman_z_halt == pytest.approx(z_tab[-1]) and z_tab[-1] > 0.0
    # exact at the nodes the runtime queries; held past the horizon
    for i in (0, 5, len(t_tab) - 1):
        assert model.ddm.boundary(float(t_tab[i])) == pytest.approx(float(z_tab[i]))
    assert model.ddm.boundary(float(t_tab[-1]) + 5.0) == pytest.approx(float(z_tab[-1]))
    # diagnostics reach the record
    data = model.get_spin_system_data()
    assert data["pure_ddm_bellman_variant"] == "discrete"
    assert data["pure_ddm_bellman_Delta_t"] == pytest.approx(dt_obs)
    assert data["pure_ddm_bellman_terminal"] == "halt_sprt"
    assert data["pure_ddm_z_myopic"] is not None
    assert data["pure_ddm_z_myopic"] >= data["pure_ddm_z_bellman"] - 1e-12
    assert data["pure_ddm_z_gap"] >= -1e-12
    assert model._bellman_diag["variant"] == "discrete"
    # the chain's own mean exit time sizes the runaway guard, not the continuous D(z)
    assert model._halt_mean_exit == pytest.approx(model._bellman_diag["halt_mean_exit_time"])
    assert model._geom_log["rho_branch"] == "bellman_discrete"


def test_wired_model_uses_the_tick_under_a_shared_stream():
    stream = {"mode": "shared", "frozen_sd": 0.0, "white_rate": 0.25, "seed": 4242}
    agent, model, objs = _build_model(ticks=10, n_sub=4, sensory_stream=stream,
                                      eta_rate=[0.0, 0.0])
    assert model.ddm.external_percept
    model.step(agent, 0, None, objs, {})
    assert model._bellman_Delta_t == pytest.approx(0.1)     # one percept per tick
    assert np.allclose(np.diff(model.ddm._z_table_t), 0.1)


def test_wired_model_runs_to_a_commitment_at_or_beyond_the_lattice_boundary():
    agent, model, objs = _build_model(ticks=10, n_sub=2)
    model.ddm.reset(seed=5)
    for tick in range(3000):
        model.step(agent, tick, None, objs, {})
        st = model._last_state
        if st is not None and st.committed is not None:
            break
    else:
        pytest.fail("never committed")
    assert abs(st.x) >= model.ddm.boundary(st.t_evidence) - 1e-12
    assert model.get_spin_system_data()["pure_ddm_committed"] is not None


def test_discrete_variant_config_validation():
    with pytest.raises(ValueError, match="variant must be"):
        _build_model(bellman={"variant": "nonsense"})
    with pytest.raises(ValueError, match="known_magnitude"):
        _build_model(drift_knowledge="estimated", A_source="online_evidence")
    with pytest.raises(ValueError, match="terminal must be"):
        _build_model(bellman={"terminal": "halt"})
    # the continuous default is untouched by the new keys
    agent, model, objs = _build_model(bellman={"variant": "continuous", "N_x": 401,
                                               "N_t": 2000})
    assert model.bellman_variant == "continuous"


def test_delta_t_override_is_honoured_and_flagged(caplog):
    import logging
    agent, model, objs = _build_model(bellman={"Delta_t": 0.5}, ticks=10, n_sub=4)
    with caplog.at_level(logging.WARNING, logger="sim.embodied_pure_ddm"):
        model.step(agent, 0, None, objs, {})
    assert model._bellman_Delta_t == 0.5
    assert np.allclose(np.diff(model.ddm._z_table_t), 0.5)
    assert any("OVERRIDES" in r.getMessage() for r in caplog.records)


def test_static_bound_still_short_circuits_the_discrete_variant():
    agent, model, objs = _build_model(bellman={"static_bound": 0.2})
    model.step(agent, 0, None, objs, {})
    assert model._bellman_diag["table_cache"] == "static"
    assert model.ddm.boundary(0.0) == pytest.approx(0.2)
    assert model._bellman_z_halt == pytest.approx(0.2)


def test_cache_round_trip_through_the_model(tmp_path):
    a1, m1, o1 = _build_model(bellman={"table_cache_dir": str(tmp_path)})
    m1.step(a1, 0, None, o1, {})
    assert m1._bellman_diag.get("table_cache") != "hit"
    files = list(tmp_path.glob("bellman_*.npz"))
    assert len(files) == 1
    a2, m2, o2 = _build_model(bellman={"table_cache_dir": str(tmp_path)})
    m2.step(a2, 0, None, o2, {})
    assert m2._bellman_diag["table_cache"] == "hit"
    assert np.array_equal(m1.ddm._z_table_z, m2.ddm._z_table_z)
    assert np.array_equal(m1._bellman_zqs_table[1], m2._bellman_zqs_table[1])
    assert m2._halt_mean_exit == pytest.approx(m1._halt_mean_exit)
    assert m2._bellman_z_halt == pytest.approx(m1._bellman_z_halt)
