# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
#  You may use, modify, and redistribute this file according to the terms of the
#  license. Attribution is required if this code is used in other works.
# ------------------------------------------------------------------------------

"""Pure two-alternative drift-diffusion decision substrate.

This is the decision substrate for the embodied *pure* DDM movement model — the
classical counterpart to both `MeanFieldSystem` (ring attractor) and
`AccumulatorSystem` (LCA). See EMBODIED_DDM_PURE_PLAN.md.

K = 2 ONLY
----------
The DDM's optimality claim (Bogacz et al. 2006, hereafter [B06]) is that it implements
the Sequential Probability Ratio Test, which is a *binary* result: the decision variable
`x` is a scalar log-likelihood ratio between exactly two hypotheses, and there is no
scalar LLR for three hypotheses. The K > 2 generalisation of the SPRT is the MSPRT,
which is a *race of K accumulators* (Bogacz & Gurney 2007; McMillen & Holmes 2006) —
structurally `accumulator_mode: race` in `AccumulatorSystem`, not a DDM. Do not bolt a
K > 2 mode onto this class.

Relationship to the LCA
-----------------------
[B06] Section 4 shows the linear LCA's difference variable `x = y0 - y1` obeys
`dx/dt = (w - k)x + (I0 - I1) + noise`, so the balanced condition `w = k`
(equivalently `beta_inh == lambda_leak`, the LCA's own bifurcation point) reduces the
LCA exactly to this pure DDM. That equivalence is the strongest available acceptance
test — see tests/test_ddm_systems.py::test_v4_lca_equivalence.

Timing
------
As with `AccumulatorSystem`, integration happens ONCE per control tick (optionally with
`n_sub` sub-steps), across ticks, in per-second units. This is deliberately not the ring
attractor's relax-to-attractor-every-tick regime.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger("sim.ddm")

# Dimensionless boundary a = 2Az/c^2 is capped here: sinh() overflows a double past
# a ~ 710, and a = 50 already implies ER ~ 1e-22.
A_STAR_MAX = 50.0
_EPS = 1e-12


@dataclass
class DDMState:
    """Snapshot of the decision variable after one control tick."""

    x: float = 0.0
    z: float = 0.0
    t_evidence: float = 0.0
    committed: Optional[int] = None
    p1: float = 0.5
    q_hat: np.ndarray = field(default_factory=lambda: np.zeros(2))
    A_inst: float = 0.0
    crossed_this_step: bool = False
    changed_mind: bool = False
    # --- post-commitment flexibility (FEATURE_POST_COMMITMENT_FLEXIBILITY) ---
    released_this_step: bool = False
    reversal_this_step: bool = False
    x_over_z: float = 0.0


class DriftDiffusionSystem:
    """Pure 2AFC drift-diffusion process with static or collapsing boundaries.

    Evidence (plan Sections 1.1-1.3): one noisy quality sample per target per sub-step,
    `q_hat_i ~ N(q_i, eta_i^2)` with `eta_i = eta_rate_i / sqrt(dt)` so that noise is a
    *rate* and results are tick-rate independent. The increment is then

        dx = (q_hat_0 - q_hat_1) * dt = A*dt + c*sqrt(dt)*xi,
        A = q_0 - q_1,   c^2 = eta_rate_0^2 + eta_rate_1^2

    `x > 0` favours target 0 (the first entry of the configured `target_ids`).
    """

    def __init__(
        self,
        eta_rate: Sequence[float] = (0.3, 0.3),
        evidence_mode: str = "difference",
        # --- across-trial variability ([B06] Section 3) ---
        extended: bool = False,
        s_A: float = 0.0,
        s_x: float = 0.0,
        x0: float = 0.0,
        # --- integration ---
        n_sub: int = 1,
        # --- boundary ---
        boundary_mode: str = "static",
        threshold_policy: str = "bayes_risk",
        z_manual: float = 1.0,
        cost_ratio: float = 10.0,
        D_iti: float = 1.0,
        D_penalty: float = 0.0,
        T0: float = 0.0,
        threshold_update_ticks: int = 0,
        geometric_error_mode: str = "terminal_graded",
        lambda_t: Optional[float] = None,
        delta_min: float = 1e-3,
        A_source: str = "ensemble",
        drift_knowledge: str = "known_magnitude",
        A_lognormal_s: float = 0.0,
        A_lognormal_debias: bool = True,
        A_lognormal_redraw: str = "onset",
        A_hat_min: float = 1e-9,
        A_rng: np.random.Generator | None = None,
        collapse_form: str = "weibull",
        z_min: float = 0.05,
        collapse_rate: float = 0.5,
        tau_c: float = 2.0,
        weibull_k: float = 3.0,
        weibull_a_inf: float = 0.2,
        deadline_T: Optional[float] = None,
        urgency_mode: str = "off",
        urgency_tau: float = 0.5,
        urgency_slope: float = 1.0,
        # --- readout / commitment ---
        readout_mode: str = "normalized",
        posterior_gain: Optional[float] = None,
        allow_changes_of_mind: bool = False,
        com_window: float = 0.3,
        # --- post-commitment flexibility ---
        flexibility: bool = False,
        post_commit_accumulation: str = "unbounded",
        A_expected: Optional[float] = None,
        A_expected_deferred: bool = False,
        rng: np.random.Generator | None = None,
    ):
        eta = np.asarray(eta_rate, dtype=float).reshape(-1)
        if eta.size == 1:
            eta = np.repeat(eta, 2)
        if eta.size != 2:
            raise ValueError("eta_rate must have 1 or 2 entries (K = 2 only)")
        if np.any(eta < 0.0):
            raise ValueError("eta_rate entries must be non-negative")
        self.eta_rate = eta

        self.evidence_mode = str(evidence_mode).strip().lower()
        if self.evidence_mode not in {"difference", "llr"}:
            raise ValueError("evidence_mode must be 'difference' or 'llr'")
        if self.evidence_mode == "llr":
            # Plan Section 1.2: precision-weighted LLR is only meaningful once per-target
            # sensory noise actually differs, which needs the `uncertainty` field
            # deferred in the LCA plan Section 2.2. Ship `difference` first rather than
            # shipping an unjustified weighting.
            raise NotImplementedError(
                "evidence_mode 'llr' requires per-target sensory noise (the deferred "
                "`uncertainty` field). Use 'difference', which is the SPRT when "
                "eta_rate[0] == eta_rate[1]."
            )

        self.extended = bool(extended)
        self.s_A = float(s_A)
        self.s_x = float(s_x)
        self.x0 = float(x0)

        self.n_sub = max(1, int(n_sub))

        self.boundary_mode = str(boundary_mode).strip().lower()
        if self.boundary_mode not in {"static", "collapsing"}:
            raise ValueError("boundary_mode must be 'static' or 'collapsing'")
        self.threshold_policy = str(threshold_policy).strip().lower()
        if self.threshold_policy not in {
            "manual", "bayes_risk", "reward_rate", "geometric", "bellman",
        }:
            raise ValueError(
                "threshold_policy must be 'manual', 'bayes_risk', 'reward_rate', "
                "'geometric' or 'bellman'"
            )
        self.z_manual = float(z_manual)
        self.cost_ratio = float(cost_ratio)
        self.D_iti = float(D_iti)
        self.D_penalty = float(D_penalty)
        self.T0 = float(T0)
        self.threshold_update_ticks = max(0, int(threshold_update_ticks))
        self.geometric_error_mode = str(geometric_error_mode).strip().lower()
        if self.geometric_error_mode not in {
            "terminal_graded", "correctable", "terminal_categorical"
        }:
            raise ValueError(
                "geometric_error_mode must be 'terminal_graded' (default), 'correctable' "
                "or 'terminal_categorical'"
            )
        self.lambda_t = None if lambda_t is None else float(lambda_t)
        if self.lambda_t is not None and self.lambda_t <= 0.0:
            raise ValueError("lambda_t must be > 0 (quality units per second)")
        self.delta_min = max(float(delta_min), _EPS)
        self.A_source = str(A_source).strip().lower()
        if self.A_source == "online":
            # Deprecated alias: kept for one release so existing configs keep working, but
            # ambiguous now that two "online" sources exist with different mechanisms.
            logger.warning(
                "A_source 'online' is deprecated and maps to 'online_evidence'; the other "
                "online source is 'online_lognormal' (a noisy percept of the magnitude). "
                "Update the config to the explicit name."
            )
            self.A_source = "online_evidence"
        if self.A_source not in {
            "oracle", "ensemble", "online_evidence", "online_lognormal"
        }:
            raise ValueError(
                "A_source must be 'oracle', 'ensemble', 'online_evidence' or "
                "'online_lognormal' ('online' is a deprecated alias for 'online_evidence')"
            )

        # Resolved here rather than further down because the known-|A| validation
        # immediately below needs it.
        self.A_expected = None if A_expected is None else float(A_expected)
        # DEFERRED RESOLUTION. `A_expected` is not free information the experimenter has
        # to supply twice: the scenario already declares the target qualities, and their
        # gap IS the ensemble magnitude. When the caller owns those declared qualities
        # (the movement model does; this class does not) it constructs with
        # A_expected_deferred=True and calls `resolve_ensemble_A` once, before the first
        # `update_A_hat`. Crucially this is NOT the deleted running-mean fallback: the
        # value comes from the scenario DEFINITION, not from the evidence, so it is a
        # block-level constant, defined before the first tick and independent of trial
        # history -- the two properties Section 3 removed the old fallback to protect.
        self.A_expected_deferred = bool(A_expected_deferred)

        # --- TASK_A_KNOWN_DRIFT.md: state the assumption, then enforce it -------
        # THE ASSUMPTION: the agent knows the MAGNITUDE |A| = |q0 - q1|; it does not know
        # the SIGN. Its task is sign discrimination only. This is load-bearing, not
        # cosmetic: `a = 2Az/c^2` IS the log-odds at commitment only if A is known, and
        # that identity is what makes a constant boundary optimal absent geometry change.
        # It follows that under this model the collapse has EXACTLY ONE SOURCE -- the
        # geometry -- which is what makes every collapse result attributable.
        self.drift_knowledge = str(drift_knowledge).strip().lower()
        if self.drift_knowledge not in {"known_magnitude", "estimated"}:
            raise ValueError(
                "drift_knowledge must be 'known_magnitude' (the implemented model, "
                "TASK_A_KNOWN_DRIFT.md) or 'estimated' (TASK_B_UNKNOWN_DRIFT.md)"
            )
        if self.drift_knowledge == "known_magnitude":
            if self.A_source not in {"ensemble", "oracle"}:  # noqa: E501
                raise ValueError(
                    f"A_source '{self.A_source}' is an ESTIMATION model: it infers |A| "
                    "from the evidence, which the known-|A| model does not permit (it "
                    "would make z* depend on trial history and give the collapse a "
                    "second source). Only 'ensemble' and 'oracle' are coherent here. "
                    "See TASK_B_UNKNOWN_DRIFT.md for the model where |A| is unknown; to "
                    "run this configuration as that model set "
                    "drift_knowledge: 'estimated'."
                )
            # Enforced only where A_hat actually DETERMINES the boundary. Under
            # threshold_policy 'manual' z is a fixed number and A_hat reaches nothing but
            # the logs, so there is no z*(Delta) whose invariance the requirement could
            # buy -- Section 3's stated reason for it. Demanding A_expected there would be
            # friction without a property.
            a_drives_z = self.threshold_policy in {
                "bayes_risk", "reward_rate", "geometric", "bellman",
            }
            # Deferral is only meaningful where A_hat DETERMINES the boundary. Under
            # 'manual' nothing reads it, so a symmetric-tie scenario -- identical target
            # qualities, a perfectly legitimate experiment -- must not fail on a
            # deduction it never needed.
            if not a_drives_z:
                self.A_expected_deferred = False
            if (
                a_drives_z
                and self.A_source == "ensemble"
                and self.A_expected is None
                and not self.A_expected_deferred
            ):
                # The null -> running-mean fallback is removed (TASK_A Section 3): it is
                # UNDEFINED at evidence onset, which is exactly when z* is first computed,
                # and it silently makes z* depend on trial history. Deferred resolution
                # from the DECLARED target qualities is exempt: see A_expected_deferred.
                raise ValueError(
                    "A_expected is REQUIRED under A_source 'ensemble': it is the "
                    "magnitude the agent is assumed to know. There is no null fallback -- "
                    "a running mean is undefined at evidence onset, which is when z* is "
                    "first computed. Set A_expected, use A_source 'oracle' to take the "
                    "true per-trial |q0 - q1|, or construct with A_expected_deferred=True "
                    "and call resolve_ensemble_A() with the declared target qualities."
                )
            if self.A_expected is not None and float(self.A_expected) <= 0.0:
                raise ValueError("A_expected must be > 0 under the known-|A| model")
        else:
            # Task B owns the history-dependent estimate; there is nothing to deduce.
            self.A_expected_deferred = False
            logger.info(
                "drift_knowledge 'estimated': A_source '%s' infers |A| from the evidence. "
                "This is NOT the known-|A| model -- `a` is no longer a confidence and the "
                "collapse has a second source. See TASK_B_UNKNOWN_DRIFT.md.",
                self.A_source,
            )
        self.A_lognormal_s = float(A_lognormal_s)
        if self.A_lognormal_s < 0.0:
            raise ValueError("A_lognormal_s must be >= 0")
        self.A_lognormal_debias = bool(A_lognormal_debias)
        self.A_lognormal_redraw = str(A_lognormal_redraw).strip().lower()
        if self.A_lognormal_redraw != "onset":
            raise ValueError(
                "A_lognormal_redraw supports only 'onset': redrawing per tick would turn a "
                "percept into a noise process and interact opaquely with "
                "threshold_update_ticks"
            )
        self.A_hat_min = float(A_hat_min)
        # Drift-estimate noise draws from a SEPARATE stream from the evidence noise, so
        # the threshold and the evidence path are not correlated across trials (which
        # would silently break paired-seed comparisons).
        self._A_rng = A_rng if A_rng is not None else np.random.default_rng()

        self.collapse_form = str(collapse_form).strip().lower()
        if self.collapse_form not in {
            "geometric", "deadline", "weibull", "hyperbolic", "exponential", "linear"
        }:
            raise ValueError(f"unknown collapse_form '{collapse_form}'")
        self.z_min = float(z_min)
        self.collapse_rate = float(collapse_rate)
        self.tau_c = max(float(tau_c), _EPS)
        self.weibull_k = float(weibull_k)
        self.weibull_a_inf = float(weibull_a_inf)
        self.deadline_T = None if deadline_T is None else float(deadline_T)

        self.urgency_mode = str(urgency_mode).strip().lower()
        if self.urgency_mode not in {"off", "boundary", "gating"}:
            raise ValueError("urgency_mode must be 'off', 'boundary' or 'gating'")
        self.urgency_tau = max(float(urgency_tau), _EPS)
        self.urgency_slope = float(urgency_slope)

        self.readout_mode = str(readout_mode).strip().lower()
        if self.readout_mode not in {"normalized", "posterior"}:
            raise ValueError("readout_mode must be 'normalized' or 'posterior'")
        self.posterior_gain = None if posterior_gain is None else float(posterior_gain)
        self.allow_changes_of_mind = bool(allow_changes_of_mind)
        self.com_window = float(com_window)

        # Post-commitment flexibility. The boundary stops being absorbing: the whole
        # state machine is `committed <=> |x| >= z(t)`, re-evaluated every step. There is
        # no hold counter and no second threshold.
        self.flexibility = bool(flexibility)
        self.post_commit_accumulation = str(post_commit_accumulation).strip().lower()
        if self.post_commit_accumulation not in {"unbounded", "clamp"}:
            raise ValueError(
                "post_commit_accumulation must be 'unbounded' or 'clamp'; got "
                f"'{post_commit_accumulation}'"
            )
        if self.flexibility and self.allow_changes_of_mind:
            # `allow_changes_of_mind` is a bounded window that permits ONE reversal;
            # flexibility supersedes it with an unbounded non-absorbing boundary.
            # Running both would apply two different reversal rules to the same state.
            logger.warning(
                "flexibility=True supersedes allow_changes_of_mind/com_window "
                "(a bounded one-shot reversal window); disabling the latter."
            )
            self.allow_changes_of_mind = False

        self.rng = rng if rng is not None else np.random.default_rng()

        # Boundary-policy coherence flag (plan Section 4.1). Set by the owner via
        # `flag_policy_incoherent()`; carried into the per-trial record so downstream
        # analysis cannot mistake the combination for an optimality result.
        self.boundary_policy_incoherent = False

        # Derived / state.
        self.c = float(math.sqrt(float(np.sum(self.eta_rate ** 2))))
        self._deadline_cache: dict = {}
        self.reset()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the decision variable and re-draw the per-trial variability terms."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.x = float(self.x0)
        self.t_evidence = 0.0
        self.committed: Optional[int] = None
        self.t_commit: Optional[float] = None
        self.rt: Optional[float] = None
        self.changed_mind = False
        self.crossed_this_step = False
        # --- flexibility bookkeeping ---
        self.released_this_step = False
        self.reversal_this_step = False
        self.n_commits = 0
        self.n_releases = 0
        self.n_reversals = 0
        self.t_first_commit: Optional[float] = None
        self.last_committed_target: Optional[int] = None
        # Transitions are appended AS THEY HAPPEN (per sub-step). Reconstructing them
        # per tick from the sticky flags loses any commit/release pair that opens and
        # closes inside one tick, which chatter (Section 3.2) produces constantly.
        # The owner drains this list each tick and enriches it with the pose.
        self.pending_transitions: list = []
        # Bellman policy: the whole z(t) trajectory is precomputed, so `boundary()`
        # becomes a table lookup rather than a formula (Section 6).
        self._z_table_t = None
        self._z_table_z = None
        self.past_horizon = False
        self.z_current = float(self.z_manual)
        self.z0 = float(self.z_manual)
        self.z_star_at_onset: Optional[float] = None
        self.A_hat = 0.0 if self.A_expected is None else float(self.A_expected)
        self.A_true = 0.0          # MAGNITUDE only; the signed difference is never stored
        self._A_sum = 0.0
        self._A_count = 0
        self._A_lognormal_hat = None   # drawn once, at evidence onset
        self._urgency_evidence = 0.0
        self.last_q = np.zeros(2, dtype=float)
        self.last_q_hat = np.zeros(2, dtype=float)
        self._A_offset = 0.0
        # Across-trial variability ([B06] Section 3): drift variability produces slow
        # errors, start-point variability produces fast errors. Drawn once per trial.
        if self.extended:
            if self.s_A > 0.0:
                self._A_offset = float(self.rng.normal(0.0, self.s_A))
            if self.s_x > 0.0:
                self.x = float(self.x0 + self.rng.uniform(-self.s_x, self.s_x))

    def flag_policy_incoherent(self, value: bool = True) -> None:
        """Mark that threshold_policy and boundary_mode are an incoherent pairing."""
        self.boundary_policy_incoherent = bool(value)

    def resolve_ensemble_A(self, value: float, *, source: str = "declared qualities") -> float:
        """Install the ensemble |A| deduced from the scenario (deferred resolution).

        Called once, before the first `update_A_hat`, by a caller that owns the DECLARED
        target qualities. `value` must be their gap, never a percept-derived or
        evidence-derived quantity: the point of `ensemble` is one threshold per block, so
        anything that varies within or across trials would reintroduce exactly the
        history dependence Section 3 removed the running-mean fallback to prevent.
        """
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"cannot deduce the ensemble |A| from the {source}: the gap is {value!r}. "
                "Identical target qualities mean A = 0, where every known-|A| boundary "
                "correctly degenerates to z -> 0 ('evidence is worthless') and there is "
                "nothing to solve. For a symmetric-tie experiment set A_expected "
                "explicitly to the discriminability the agent should ASSUME, or use "
                "threshold_policy 'manual'."
            )
        self.A_expected = value
        self.A_expected_deferred = False
        return value

    # ------------------------------------------------------------------
    # Drift-magnitude estimate (the A entering rho and z*)
    # ------------------------------------------------------------------
    def update_A_hat(self, q: Sequence[float]) -> float:
        """Resolve the drift MAGNITUDE estimate used by the threshold, and store it.

        `A` enters the threshold twice — as the SNR factor `rho ~ A^2 c_e/(c^2 c_tau)` and
        again in the conversion `z* = a* c^2/(2A)` — so where it comes from is the whole
        estimation question, especially under `terminal_categorical` where `c_e` is a
        constant and `A` is the only unknown left.

        MAGNITUDE, NEVER THE SIGNED DIFFERENCE. The agent is trying to determine the sign
        of `q0 - q1`; the threshold needs only its magnitude, and the two are independent
        (derivation Section 5.2). An estimate centred on the signed difference would leak
        the answer: the agent could take `sign(A_hat)` and skip the DDM entirely. The sign
        is discarded here, at the point of use, and no signed difference is ever stored on
        this object.

        Sources:
          oracle           - the true |q0 - q1|. Calibration and V-tests only.
          ensemble         - A_expected: one threshold per block, which is the correct
                             setting for a fixed environment. Supplied explicitly, or
                             deduced from the scenario's DECLARED target qualities via
                             resolve_ensemble_A(). Falls back to the running mean of the
                             true magnitude only under drift_knowledge 'estimated'.
          online_evidence  - |x|/t from the agent's own accumulated evidence (variance
                             c^2/t); falls back to the percept magnitude at t = 0.
          online_lognormal - |q0 - q1| * exp(eps), eps ~ N(0, s^2), drawn ONCE at evidence
                             onset and held for the trial. Multiplicative noise because
                             magnitude perception is roughly Weber-Fechner, and because it
                             keeps A_hat strictly positive (z* divides by it).
        """
        q = np.asarray(q, dtype=float).reshape(-1)
        gap = abs(float(q[0]) - float(q[1]))     # sign discarded HERE, at the point of use
        self.A_true = gap
        self._A_sum += gap
        self._A_count += 1
        running_mean = self._A_sum / max(self._A_count, 1)

        src = self.A_source
        if src == "oracle":
            estimate = gap
        elif src == "online_evidence":
            estimate = (
                abs(self.x) / self.t_evidence if self.t_evidence > _EPS else gap
            )
        elif src == "online_lognormal":
            if self._A_lognormal_hat is None:
                # Once per trial, at onset. A per-tick redraw would turn a percept into a
                # noise process; A_lognormal_redraw enforces this.
                eps = (
                    float(self._A_rng.normal(0.0, self.A_lognormal_s))
                    if self.A_lognormal_s > 0.0 else 0.0
                )
                # Jensen: rho ~ A_hat^2 is convex, so noise inflates rho in expectation by
                # exp(2 s^2). Subtracting s^2 from the exponent makes E[A_hat^2] = A^2
                # exactly, so noise adds variability without shifting the mean threshold.
                # With debias off, uncertainty RAISES the threshold - a claim about the
                # agent, not an artefact to ignore.
                adjust = self.A_lognormal_s ** 2 if self.A_lognormal_debias else 0.0
                self._A_lognormal_hat = gap * math.exp(eps - adjust)
            estimate = self._A_lognormal_hat
        else:  # "ensemble"
            # No null fallback under the known-|A| model -- __init__ has already refused
            # A_expected=None there. `running_mean` survives only for the 'estimated'
            # (Task B) arm, where a history-dependent estimate is the point rather than a
            # violation.
            if self.A_expected is None and self.A_expected_deferred:
                raise RuntimeError(
                    "A_expected was deferred but never resolved: call "
                    "resolve_ensemble_A() with the declared target qualities BEFORE the "
                    "first update_A_hat(), or the threshold would silently fall back to "
                    "a history-dependent running mean."
                )
            estimate = (
                float(self.A_expected) if self.A_expected is not None else running_mean
            )

        # Division guard only (z* = a* c^2 / (2 A_hat)), not a behavioural floor.
        self.A_hat = max(float(estimate), self.A_hat_min)
        return self.A_hat

    # ------------------------------------------------------------------
    # Closed-form performance (Appendix A.3)
    # ------------------------------------------------------------------
    @staticmethod
    def error_rate(A: float, c: float, z: float) -> float:
        """ER = 1 / (1 + exp(2Az/c^2))."""
        a = DriftDiffusionSystem.dimensionless_boundary(A, c, z)
        return 1.0 / (1.0 + math.exp(min(a, 700.0)))

    @staticmethod
    def decision_time(A: float, c: float, z: float) -> float:
        """DT = (z/A) * tanh(A z / c^2)."""
        A = abs(float(A))
        if A <= _EPS:
            return float("inf")
        return (z / A) * math.tanh(A * z / (c ** 2))

    @staticmethod
    def dimensionless_boundary(A: float, c: float, z: float) -> float:
        """a = 2Az/c^2 — the log-odds held at the moment of commitment (Appendix A.4)."""
        c = float(c)
        if c ** 2 <= _EPS:
            # Noise-free evidence (eta_rate = 0) is a legitimate deterministic
            # configuration; the log-odds it carries is unbounded, not undefined.
            return float("inf") if abs(float(A)) * float(z) > 0.0 else 0.0
        return 2.0 * abs(float(A)) * float(z) / (c ** 2)

    @staticmethod
    def er_from_a(a: float) -> float:
        """ER expressed in the dimensionless boundary: 1/(1+e^a)."""
        return 1.0 / (1.0 + math.exp(min(max(a, -700.0), 700.0)))

    @staticmethod
    def dt_from_a(a: float, A: float, c: float) -> float:
        """DT expressed in the dimensionless boundary: (c^2/2A^2)*a*tanh(a/2)."""
        A = abs(float(A))
        if A <= _EPS:
            return float("inf")
        return (c ** 2 / (2.0 * A ** 2)) * a * math.tanh(a / 2.0)

    # ------------------------------------------------------------------
    # Threshold solvers.
    #
    # Kept as three separate methods rather than one dispatch: they have different
    # signatures, different validity conditions, and only `geometric` is compatible
    # with a collapsing boundary (plan Section 4.1). Separate methods make that
    # restriction enforceable at the call site.
    # ------------------------------------------------------------------
    @staticmethod
    def solve_a_star(rho: float) -> float:
        """Solve `sinh(a) + a = rho` for a >= 0.

        `sinh(a)+a` is strictly increasing from 0 to infinity so the root is unique
        (Appendix A.5). Newton with f' = cosh(a)+1, initialised at ln(2*rho+1).
        """
        rho = float(rho)
        if not np.isfinite(rho) or rho <= 0.0:
            return 0.0 if rho <= 0.0 else A_STAR_MAX
        a = math.log(2.0 * rho + 1.0)
        if a >= A_STAR_MAX:
            return A_STAR_MAX
        a = max(a, 1e-9)
        for _ in range(100):
            f = math.sinh(a) + a - rho
            fp = math.cosh(a) + 1.0
            step = f / fp
            a_new = a - step
            if a_new <= 0.0:
                a_new = a * 0.5
            if a_new > A_STAR_MAX:
                return A_STAR_MAX
            if abs(a_new - a) <= 1e-13 * max(1.0, abs(a)):
                a = a_new
                break
            a = a_new
        return float(min(max(a, 0.0), A_STAR_MAX))

    @classmethod
    def solve_threshold_bayes_risk(cls, A: float, c: float, cost_ratio: float) -> float:
        """Bayes-risk optimal CONSTANT boundary (Appendix A.5).

            sinh(a*) + a* = rho = (A/c)^2 * (c_e/c_tau),      z* = a* c^2 / (2A)

        Only the cost *ratio* enters (plan Section 3.3): BR = c_tau*[DT + (c_e/c_tau)*ER],
        so argmin_z BR depends on c_e/c_tau alone.

        WARNING: this is the optimal boundary among *constant* boundaries. Under its own
        assumptions (A known and constant, infinite horizon, constant costs) the constant
        boundary is provably optimal and any collapse strictly increases Bayes risk
        (Appendix C). Do not use this to seed a collapsing profile and call it optimal.
        """
        A = abs(float(A))
        c = float(c)
        if c <= _EPS:
            return float("inf")
        if A <= _EPS:
            # Zero drift: the samples carry no information, ER = 1/2 for every z, and
            # DT -> z^2/c^2, so BR is minimised at z -> 0 (Appendix A.5's small-rho
            # asymptote z* -> A*c_e/(4 c_tau)). Do not deliberate over worthless
            # evidence. Callers floor this at z_min.
            return 0.0
        rho = (A / c) ** 2 * float(cost_ratio)
        a_star = cls.solve_a_star(rho)
        return a_star * c ** 2 / (2.0 * A)

    @classmethod
    def solve_threshold_reward_rate(
        cls, A: float, c: float, T0: float, D: float, Dp: float
    ) -> float:
        """Reward-rate optimal CONSTANT boundary ([B06] headline result).

            RR = (1 - ER) / (DT + T0 + D + ER*Dp)

        `z` appears in numerator and denominator so there is no clean closed form; this
        maximises numerically over the dimensionless boundary `a`. Same warning as
        `solve_threshold_bayes_risk` regarding collapsing boundaries.
        """
        A = abs(float(A))
        c = float(c)
        if c <= _EPS:
            return float("inf")
        if A <= _EPS:
            # Zero drift: accuracy is 1/2 whatever the boundary, so reward rate is
            # maximised by deciding immediately. Same degenerate limit as Bayes risk.
            return 0.0

        def neg_rr(a: float) -> float:
            er = cls.er_from_a(a)
            dt = cls.dt_from_a(a, A, c)
            denom = dt + float(T0) + float(D) + er * float(Dp)
            if denom <= _EPS:
                return float("inf")
            return -(1.0 - er) / denom

        # Dense grid then parabolic refinement — deterministic and robust without
        # assuming unimodality holds exactly at the grid edges.
        grid = np.linspace(1e-4, A_STAR_MAX, 2000)
        values = np.array([neg_rr(float(a)) for a in grid])
        i = int(np.argmin(values))
        lo = grid[max(i - 1, 0)]
        hi = grid[min(i + 1, grid.size - 1)]
        for _ in range(200):
            if hi - lo <= 1e-10:
                break
            m1 = lo + (hi - lo) / 3.0
            m2 = hi - (hi - lo) / 3.0
            if neg_rr(m1) < neg_rr(m2):
                hi = m2
            else:
                lo = m1
        a_star = 0.5 * (lo + hi)
        return a_star * c ** 2 / (2.0 * A)

    @classmethod
    def _z_from_rho(cls, rho: float, A: float, c: float) -> float:
        """Map `rho` to the boundary `z* = a* c^2 / (2A)`, with the small-rho branch.

        For rho << 1, `sinh(a)+a = rho` gives `a* -> rho/2` exactly to O(rho^3), so
        `z* -> rho c^2 / (4A)`. Using that closed form directly avoids relying on the
        Newton iteration in the regime where the boundary is vanishing.
        """
        A = abs(float(A))
        c = float(c)
        if c <= _EPS:
            return float("inf")
        if A <= _EPS:
            # Zero drift: the samples carry no information, so the optimal boundary
            # vanishes. Callers floor this at z_min. (See solve_threshold_bayes_risk.)
            return 0.0
        if not np.isfinite(rho):
            return A_STAR_MAX * c ** 2 / (2.0 * A)
        if rho < 1e-6:
            return float(rho) * c ** 2 / (4.0 * A)
        a_star = cls.solve_a_star(rho)
        return a_star * c ** 2 / (2.0 * A)

    @classmethod
    def geometric_rho(
        cls,
        A: float,
        c: float,
        delta: float,
        *,
        d: Optional[float] = None,
        v: Optional[float] = None,
        lambda_t: Optional[float] = None,
        cost_ratio: Optional[float] = None,
        mode: str = "terminal_graded",
        predecision_motion: str = "average",
        delta_min: float = 1e-3,
    ) -> float:
        """Return `rho` for the geometric threshold policy under one of three error models."""
        A = abs(float(A))
        c = float(c)
        if A <= _EPS or c <= _EPS:
            return 0.0
        # sin^2(delta/4) -> 0 sends rho -> inf (targets in the same direction means
        # deliberation is free, so demand arbitrarily strong evidence). Clamp so sinh()
        # cannot overflow.
        delta = max(abs(float(delta)), float(delta_min))
        mode = str(mode).strip().lower()

        if mode not in {"terminal_graded", "terminal_categorical", "correctable"}:
            raise ValueError(f"unknown geometric_error_mode '{mode}'")
        if mode == "terminal_graded" and (lambda_t is None or float(lambda_t) <= 0.0):
            raise ValueError(
                "geometric_error_mode 'terminal_graded' requires lambda_t > 0 "
                "(the opportunity cost of time, in quality units per second)"
            )
        if mode == "correctable" and (d is None or v is None or float(v) <= 0.0):
            raise ValueError("geometric_error_mode 'correctable' requires d and v > 0")

        # ONE code path for all three error models. The mode chooses c_e; c_tau is
        # always the same shared delay cost, and dividing by it is what makes the
        # boundary collapse. Do not re-introduce a per-mode rho expression: that
        # duplication is what previously let terminal_categorical drop c_tau entirely.
        c_tau, c_e = cls.geometric_costs(
            A=A, delta=delta, d=d, v=v, lambda_t=lambda_t, cost_ratio=cost_ratio,
            mode=mode, predecision_motion=predecision_motion,
        )
        # delta is clamped to delta_min above, so c_tau > 0 is guaranteed.
        assert c_tau > 0.0, f"c_tau must be positive (delta={delta}, mode={mode})"
        if c_tau <= _EPS:
            return float("inf")
        return (A / c) ** 2 * (c_e / c_tau)

    @classmethod
    def solve_threshold_geometric(
        cls,
        A: float,
        c: float,
        delta: float,
        *,
        d: Optional[float] = None,
        v: Optional[float] = None,
        lambda_t: Optional[float] = None,
        cost_ratio: Optional[float] = None,
        mode: str = "terminal_graded",
        predecision_motion: str = "average",
        delta_min: float = 1e-3,
    ) -> float:
        """Threshold with costs derived from the body's geometry (corrected Appendix B).

        The cost of delay is unchanged:

            c_tau = 1 - cos(Delta/2)          [dimensionless]

        The cost of an error depends on whether the agent can re-route:

          terminal_graded (DEFAULT) — the commitment is final, so the agent forfeits the
            quality difference `A` wherever it happens to be standing:
                c_e = A / lambda_t                              [seconds]
                rho = A^3 / (2 c^2 lambda_t sin^2(Delta/4))
            Absolute range `d` and speed `v` DROP OUT: the stakes of being wrong do not
            depend on position, only the angular configuration matters. rho scales as
            A^3, and as the agent approaches, rho tends to the finite floor
            A^3/(c^2 lambda_t) rather than to zero — so the collapse is BOUNDED and
            there is NO geometric deadline.

          correctable — the agent discovers the mistake and re-routes, paying a detour:
                c_e = 2 d sin(Delta/2) / v,  rho = (A/c)^2 (2d/v) cot(Delta/4)
            Here the detour shrinks as d -> 0, so rho -> 0 and the boundary collapses to
            zero: the geometry supplies its own deadline. That deadline is an artefact of
            correctability, which is why it is a separate mode and not the default.

          terminal_categorical — options differ in kind (safe/unsafe), 0-1 loss:
                rho = (A/c)^2 * cost_ratio

        `lambda_t` is the opportunity cost of time in quality units per second (Charnov's
        marginal value theorem currency). rho ~ 1/lambda_t, so a richer environment means
        a lower threshold and faster, sloppier decisions.
        """
        rho = cls.geometric_rho(
            A, c, delta, d=d, v=v, lambda_t=lambda_t, cost_ratio=cost_ratio,
            mode=mode, predecision_motion=predecision_motion, delta_min=delta_min,
        )
        return cls._z_from_rho(rho, A, c)

    @classmethod
    def geometric_z_floor(
        cls,
        A: float,
        c: float,
        lambda_t: Optional[float] = None,
        predecision_motion: str = "average",
        *,
        cost_ratio: Optional[float] = None,
        mode: str = "terminal_graded",
    ) -> float:
        """The finite floor the bounded collapse asymptotes to.

        As the agent approaches, `Delta -> pi` so `c_tau = 2 sin^2(Delta/4) -> 1` and
        rho stops falling:

            terminal_graded      : rho_floor = A^3 / (c^2 lambda_t)
            terminal_categorical : rho_floor = (A/c)^2 * cost_ratio

        The boundary converges to this floor and never reaches zero — the cost of a
        wrong choice never diminishes.
        """
        return cls.solve_threshold_geometric(
            A, c, math.pi, lambda_t=lambda_t, cost_ratio=cost_ratio, mode=mode,
            predecision_motion=predecision_motion,
        )

    @classmethod
    def calibrate_categorical(
        cls,
        A: float,
        delta: float,
        target_ER: float,
        target_DT: float,
        predecision_motion: str = "average",
    ) -> tuple[float, float]:
        """Return `(eta_rate, cost_ratio)` hitting a target (ER, DT) at a reference geometry.

        Inverts the frozen-geometry closed forms (derivation Section 8.1, with (3)
        solved for cost_ratio rather than lambda_t):

            a          = ln(1/ER - 1)
            c^2        = 2 A^2 DT / (a tanh(a/2))        ->  eta_rate = c / sqrt(2)
            cost_ratio = c^2 (sinh a + a) c_tau(Delta) / A^2

        The ER/DT pair is a *frozen-geometry reference*, not a prediction for a live run:
        under a collapsing boundary the realised error rate is higher and the decision
        time shorter.
        """
        A = abs(float(A))
        if A <= _EPS:
            raise ValueError("calibrate_categorical requires A > 0")
        if not 0.0 < float(target_ER) < 0.5:
            raise ValueError("target_ER must be in (0, 0.5)")
        if float(target_DT) <= 0.0:
            raise ValueError("target_DT must be > 0")

        a = math.log(1.0 / float(target_ER) - 1.0)
        c_sq = 2.0 * A ** 2 * float(target_DT) / (a * math.tanh(a / 2.0))
        c_tau = cls.c_tau_linearised(delta, predecision_motion)
        cost_ratio = c_sq * (math.sinh(a) + a) * c_tau / A ** 2
        return math.sqrt(c_sq / 2.0), cost_ratio

    @staticmethod
    def c_tau_linearised(delta: float, predecision_motion: str = "average") -> float:
        """Marginal cost of delay, `1 - cos(delta/2)` written as `2 sin^2(delta/4)`.

        Shared by ALL geometric_error_mode branches: the error mode sets c_e (the
        numerator of rho), the delay model sets c_tau (the denominator). All angular
        dependence — and therefore the entire collapse — lives here. A mode that omits
        it produces a boundary that cannot collapse.

        The half-angle form is used because it avoids the cancellation in `1 - cos` for
        small delta, where the collapse is steepest.

        Under `predecision_motion: stationary` no progress is made while deliberating,
        so every second costs a full second and c_tau == 1 identically, independent of
        the geometry (derivation Section 4.1).
        """
        if str(predecision_motion).strip().lower() == "stationary":
            return 1.0
        return 2.0 * math.sin(abs(float(delta)) / 4.0) ** 2

    @classmethod
    def geometric_costs(
        cls,
        A: float,
        delta: float,
        *,
        d: Optional[float] = None,
        v: Optional[float] = None,
        lambda_t: Optional[float] = None,
        cost_ratio: Optional[float] = None,
        mode: str = "terminal_graded",
        predecision_motion: str = "average",
    ) -> tuple[float, float]:
        """Return `(c_tau, c_e)` in seconds, as regret against an oracle.

        c_tau is shared by every error model (`c_tau_linearised`); only c_e differs:

            terminal_graded      : c_e = A / lambda_t     (forfeited quality, converted
                                                           to seconds by the opportunity
                                                           cost of time)
            terminal_categorical : c_e = cost_ratio       (0-1 loss: every error costs
                                                           the same, whatever the margin)
            correctable          : c_e = 2 d sin(Delta/2) / v   (the detour)
        """
        c_tau = cls.c_tau_linearised(delta, predecision_motion)
        delta = abs(float(delta))

        mode = str(mode).strip().lower()
        if mode == "terminal_graded":
            if lambda_t is None or float(lambda_t) <= 0.0:
                raise ValueError("terminal_graded requires lambda_t > 0")
            # The error is a pure loss of value: at a near-tie the agent is equidistant
            # from both targets, so the wrong choice costs no extra travel time.
            c_e = abs(float(A)) / float(lambda_t)
        elif mode == "terminal_categorical":
            if cost_ratio is None:
                raise ValueError("terminal_categorical requires cost_ratio")
            # 0-1 loss. cost_ratio is c_e alone, NOT c_e/c_tau: the delay cost still
            # applies and still carries the angular dependence.
            c_e = float(cost_ratio)
        elif mode == "correctable":
            if d is None or v is None:
                raise ValueError("correctable requires d and v")
            v = float(v)
            if v <= _EPS:
                return float(c_tau), float("inf")
            c_e = 2.0 * max(float(d), 0.0) * math.sin(delta / 2.0) / v
        else:
            raise ValueError(f"geometric_costs has no c_e for mode '{mode}'")
        return float(c_tau), float(c_e)

    @classmethod
    def calibrate_categorical_fixed_cost(
        cls,
        A: float,
        delta: float,
        target_ER: float,
        cost_ratio: float,
        predecision_motion: str = "average",
    ) -> tuple[float, float]:
        """Return `(eta_rate, implied_DT)` when cost_ratio is a FIXED constraint.

        `calibrate_categorical` treats (ER, DT) as the targets and solves for cost_ratio.
        When the error cost is instead fixed by the task, DT is no longer free.
        Eliminating c^2/A^2 between the two calibration identities

            (2) c^2        = 2 A^2 DT / (a tanh(a/2))
            (3) cost_ratio = c^2 (sinh a + a) c_tau(Delta) / A^2

        gives

            DT = cost_ratio * a tanh(a/2) / (2 (sinh a + a) c_tau(Delta))

        in which neither A nor c appears: with cost_ratio and the geometry fixed, the
        target error rate alone fixes the decision time. A and c then only set the scale
        c/A that realises it. DT scales as 1/c_tau, so the strong lever on deliberation
        time is the angular separation — i.e. how far away the agent starts.
        """
        A = abs(float(A))
        if A <= _EPS:
            raise ValueError("calibrate_categorical_fixed_cost requires A > 0")
        if not 0.0 < float(target_ER) < 0.5:
            raise ValueError("target_ER must be in (0, 0.5)")
        if float(cost_ratio) <= 0.0:
            raise ValueError("cost_ratio must be > 0")

        a = math.log(1.0 / float(target_ER) - 1.0)
        c_tau = cls.c_tau_linearised(delta, predecision_motion)
        implied_dt = float(cost_ratio) * a * math.tanh(a / 2.0) / (
            2.0 * (math.sinh(a) + a) * c_tau
        )
        c_sq = 2.0 * A ** 2 * implied_dt / (a * math.tanh(a / 2.0))
        return math.sqrt(c_sq / 2.0), implied_dt

    # ------------------------------------------------------------------
    # Deadline (Frazier & Yu 2008) optimal boundary — numerically solved
    # ------------------------------------------------------------------
    @staticmethod
    def solve_deadline_boundary(
        A: float,
        c: float,
        cost_ratio: float,
        T: float,
        n_t: int = 120,
        n_x: int = 401,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Backward-induct the optimal stopping boundary under a finite horizon `T`.

        Solves the free-boundary problem of Appendix C.2 with the terminal condition
        `V(x, T) = c_e * min(p, 1-p)` — at the deadline you must decide, whatever you
        believe. Just before `T` there is no time left to acquire information, so
        continuing is pure cost and the continuation region shrinks to nothing: the
        boundary collapses to zero. That is a proof of collapse from the terminal
        condition alone, with no free shape parameters.

        Costs enter only through their ratio (dividing the value function by c_tau), so
        this takes `cost_ratio = c_e/c_tau` exactly like the static solvers.

        Returns `(times, z_values)` for interpolation, or None if the inputs degenerate.
        """
        A = abs(float(A))
        c = float(c)
        T = float(T)
        if A <= _EPS or c <= _EPS or T <= _EPS:
            return None

        k = 2.0 * A / c ** 2
        z_static = DriftDiffusionSystem.solve_threshold_bayes_risk(A, c, cost_ratio)
        if not np.isfinite(z_static):
            z_static = c * math.sqrt(T)
        x_span = max(3.0 * z_static, 3.0 * c * math.sqrt(T), 1e-3)

        xs = np.linspace(-x_span, x_span, int(n_x))
        dx = float(xs[1] - xs[0])
        n_t = max(int(n_t), 4)
        dt = T / n_t
        # The Gaussian transition must be resolved by the spatial grid.
        while c * math.sqrt(dt) < 2.0 * dx and n_t > 4:
            n_t //= 2
            dt = T / n_t

        p = 1.0 / (1.0 + np.exp(-np.clip(k * xs, -500.0, 500.0)))
        stop = float(cost_ratio) * np.minimum(p, 1.0 - p)  # in units of c_tau

        sd = c * math.sqrt(dt)
        diff = xs[None, :] - (xs[:, None] + A * dt)
        P = np.exp(-0.5 * (diff / sd) ** 2)
        row = P.sum(axis=1, keepdims=True)
        row[row <= 0.0] = 1.0
        P = P / row

        times = np.linspace(0.0, T, n_t + 1)
        z = np.zeros(n_t + 1, dtype=float)
        V = stop.copy()
        z[-1] = 0.0  # terminal: decide now, whatever you believe
        for i in range(n_t - 1, -1, -1):
            cont = dt + P @ V
            V = np.minimum(stop, cont)
            stop_region = stop <= cont
            idx = np.flatnonzero(stop_region)
            z[i] = float(np.min(np.abs(xs[idx]))) if idx.size else x_span
        return times, z

    # ------------------------------------------------------------------
    # Boundary
    # ------------------------------------------------------------------
    def set_static_threshold(self, z: float) -> None:
        """Set the baseline boundary `z0` (the value a collapsing profile starts from)."""
        if not np.isfinite(z):
            z = self.z_manual
        self.z0 = float(max(z, self.z_min))
        if self.z_star_at_onset is None:
            self.z_star_at_onset = self.z0

    def set_bellman_table(self, t_grid, z_values) -> None:
        """Install a precomputed optimal boundary `z(t)` (FEATURE_BELLMAN_POLICY Section 6).

        Unlike every other policy this one is solved once for the whole trajectory, so
        `threshold_update_ticks` has nothing to do and is ignored by the caller.
        """
        t_grid = np.asarray(t_grid, dtype=float).reshape(-1)
        z_values = np.asarray(z_values, dtype=float).reshape(-1)
        if t_grid.size != z_values.size or t_grid.size < 2:
            raise ValueError("bellman table needs matching t and z arrays of length >= 2")
        self._z_table_t = t_grid
        self._z_table_z = z_values
        self.z0 = float(z_values[0])
        self.z_current = float(z_values[0])
        self.past_horizon = False

    def _bellman_z(self, t: float) -> float:
        """Table lookup with linear interpolation between samples.

        Past the horizon the table has no entry: HOLD the last value and flag it. Do not
        extrapolate -- the terminal collapse is an artefact of the horizon, not a
        prediction, and extrapolating it drives z to zero for the wrong reason.
        """
        tt = float(t)
        if tt >= self._z_table_t[-1]:
            self.past_horizon = True
            return float(self._z_table_z[-1])
        return float(np.interp(tt, self._z_table_t, self._z_table_z))

    def boundary(self, t: float) -> float:
        """Return the boundary `z(t)`; `t` is time since evidence onset (Section 4.3)."""
        if self._z_table_z is not None:
            return float(max(self._bellman_z(t), self.z_min))
        if self.boundary_mode == "static" or self.urgency_mode == "gating":
            return float(self.z0)
        t = max(float(t), 0.0)
        z0 = float(self.z0)
        form = self.collapse_form

        if form == "weibull":
            # Hawkins et al. (2015). Nests no-collapse exactly at a_inf = 1.
            frac = 1.0 - math.exp(-((t / self.tau_c) ** self.weibull_k))
            z = z0 * (1.0 - frac * (1.0 - self.weibull_a_inf))
        elif form == "hyperbolic":
            z = z0 * (1.0 - self.collapse_rate * t / (t + self.tau_c))
        elif form == "exponential":
            z = self.z_min + (z0 - self.z_min) * math.exp(-t / self.tau_c)
        elif form == "linear":
            z = z0 - self.collapse_rate * t
        elif form == "geometric":
            z = self._geometric_profile(t)
        elif form == "deadline":
            z = self._deadline_profile(t)
        else:
            z = z0

        if form == "weibull" and math.isclose(self.weibull_a_inf, 1.0, rel_tol=0.0, abs_tol=1e-12):
            # V7: the no-collapse limit must be *identical* to a static boundary, so do
            # not let the floor perturb it.
            return z0
        return float(max(z, self.z_min))

    def _geometric_profile(self, t: float) -> float:
        """Derived collapse from the embodied cost geometry (corrected Appendix B.4).

        No parametric decay is applied here. `rho` is re-evaluated from the CURRENT
        angular separation every tick by the owning movement model, which calls
        `set_static_threshold` — that is exact in both the far and near field and needs
        no case analysis, so the live `z0` already *is* the collapse profile.

        Under `terminal_graded` the resulting profile falls linearly in log-distance in
        the far field (`a* ~ const + 2 ln r`) and asymptotes to the finite floor
        `geometric_z_floor`. It does NOT reach zero: there is no geometric deadline,
        because the cost of a wrong choice never diminishes. Under `correctable` the
        detour does shrink with range, and the profile does collapse to zero.
        """
        return float(self.z0)

    def _deadline_profile(self, t: float) -> float:
        """Interpolate the numerically-solved Frazier-Yu profile.

        NOTE: under `terminal_graded` the horizon is NOT derivable from the geometry —
        the stakes of a wrong choice never shrink, so the geometry supplies no deadline
        (corrected Appendix B.4). An explicit `deadline_T` is required, and the result
        must not be described as geometrically derived. Only under `correctable` may the
        horizon fall back to the arrival time.
        """
        A = max(abs(self.A_hat), _EPS)
        T = self.deadline_T
        if T is None:
            if self.geometric_error_mode == "correctable":
                T = getattr(self, "_geom_T_arr", None)
            elif not getattr(self, "_warned_no_derived_deadline", False):
                self._warned_no_derived_deadline = True
                logger.warning(
                    "collapse_form 'deadline' under geometric_error_mode '%s' has no "
                    "geometrically derived horizon (the error cost does not shrink with "
                    "range, so there is no arrival deadline). Set deadline_T explicitly; "
                    "holding the boundary constant meanwhile.",
                    self.geometric_error_mode,
                )
        if T is None or T <= _EPS:
            return float(self.z0)
        key = (round(A, 6), round(self.c, 6), round(self.cost_ratio, 6), round(float(T), 4))
        profile = self._deadline_cache.get(key)
        if profile is None:
            profile = self.solve_deadline_boundary(A, self.c, self.cost_ratio, float(T))
            self._deadline_cache[key] = profile
        if profile is None:
            return float(self.z0)
        times, zs = profile
        return float(np.interp(min(t, float(T)), times, zs))

    def set_geometry(self, rho0: Optional[float], T_arr: Optional[float]) -> None:
        """Record the geometric collapse parameters (rho at onset, arrival time)."""
        if rho0 is not None:
            self._geom_rho0 = float(rho0)
        if T_arr is not None:
            self._geom_T_arr = float(T_arr)

    # ------------------------------------------------------------------
    # Readout (plan Section 2.1)
    # ------------------------------------------------------------------
    def weights(self, force_belief: bool = False) -> np.ndarray:
        """Return `(P_0, P_1)`, summing to 1 — the agent's belief, used as readout weights.

        `force_belief` computes the graded belief from `x` even after commitment; the
        movement model uses it during the non-decision (actuator lag) window, where the
        decision is latched internally but has not yet reached the motors.
        """
        if self.committed is not None and not force_belief:
            w = np.zeros(2, dtype=float)
            w[self.committed] = 1.0
            return w
        if self.readout_mode == "posterior":
            g = self.posterior_gain
            if g is None:
                g = 2.0 * abs(self.A_hat) / (self.c ** 2) if self.c > _EPS else 0.0
            if g > 0.0:
                p1 = 1.0 / (1.0 + math.exp(-max(min(g * self.x, 700.0), -700.0)))
                return np.array([p1, 1.0 - p1], dtype=float)
            # No usable drift estimate — fall through to the bounded readout.
        z = max(self.z_current, _EPS)
        p1 = min(max((1.0 + self.x / z) * 0.5, 0.0), 1.0)
        return np.array([p1, 1.0 - p1], dtype=float)

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------
    def step(self, q: Sequence[float], dt: float) -> DDMState:
        """Advance the decision variable by one control tick of `dt` seconds.

        `q` is the clean (pre-sampling) quality of each of the two targets, in the slot
        order fixed by the configured `target_ids`.
        """
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size != 2:
            raise ValueError(
                f"DriftDiffusionSystem is K = 2 only; got {q.size} targets. "
                "Use accumulator_mode 'race' (MSPRT) in embodied_ddm for K > 2."
            )
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        self.last_q = q.copy()
        A_inst = float(q[0] - q[1]) + self._A_offset
        # A_hat is resolved by update_A_hat(), which the movement model calls BEFORE the
        # threshold is computed. Calling it here too keeps a bare DriftDiffusionSystem
        # (unit tests, the LCA-equivalence harness) consistent without the caller having
        # to know about the ordering.
        self.update_A_hat(q)

        self.crossed_this_step = False
        self.released_this_step = False
        self.reversal_this_step = False

        # Decision already latched and outside any change-of-mind window: freeze the
        # decision variable but keep the evidence clock running for logging.
        # Under `flexibility` the boundary is NOT absorbing, so there is nothing to
        # freeze: integration continues and the state is re-derived every sub-step.
        if not self.flexibility and self.committed is not None and not self._in_com_window():
            self.t_evidence += dt
            self.z_current = self.boundary(self.t_evidence)
            return self._state(A_inst)

        dt_sub = dt / self.n_sub
        sqrt_dt_sub = math.sqrt(dt_sub)
        eta = self.eta_rate / sqrt_dt_sub  # noise as a RATE (plan Section 1.3)
        q_hat_acc = np.zeros(2, dtype=float)

        for _ in range(self.n_sub):
            xi = self.rng.standard_normal(2)
            q_hat = q + eta * xi
            q_hat_acc += q_hat
            delta = float(q_hat[0] - q_hat[1]) + self._A_offset

            if self.urgency_mode == "gating":
                # Cisek et al.: low-pass filter the evidence and MULTIPLY by a growing
                # urgency signal rather than integrating it. Genuinely a different
                # mechanism from a collapsing bound, not a reparameterisation.
                alpha = min(dt_sub / self.urgency_tau, 1.0)
                self._urgency_evidence += (delta - self._urgency_evidence) * alpha
                self.t_evidence += dt_sub
                self.x = self._urgency_evidence * (1.0 + self.urgency_slope * self.t_evidence)
            else:
                self.x += delta * dt_sub
                self.t_evidence += dt_sub

            self.z_current = self.boundary(self.t_evidence)
            if self.flexibility:
                if self.post_commit_accumulation == "clamp" and self.committed is not None:
                    # Hold |x| at the boundary while committed, so the reversal distance
                    # is a fixed 2z and release is immediate when the drift flips
                    # (Section 3.1). A CONTROL for the dwell confound, not a model.
                    if abs(self.x) > self.z_current:
                        self.x = math.copysign(self.z_current, self.x)
                self._update_flexible_state()
                # Never break: a non-absorbing boundary can be crossed again in the
                # same tick, and stopping early would hide the chatter of Section 3.2.
            elif self._check_commitment():
                break

        self.last_q_hat = q_hat_acc / float(self.n_sub)
        return self._state(A_inst)

    def _in_com_window(self) -> bool:
        """True while post-commitment evidence may still reverse the choice."""
        if not self.allow_changes_of_mind or self.t_commit is None:
            return False
        return (self.t_evidence - self.t_commit) <= self.com_window

    def _update_flexible_state(self) -> None:
        """Re-derive commitment from `|x| >= z(t)`. That is the entire state machine.

        Called every sub-step, never latching: the same call may commit, release or
        reverse. One threshold, no hold counters, no delays (Section 1).
        """
        z = self.z_current
        target = 0 if self.x > 0.0 else 1
        above = abs(self.x) >= z
        prev = self.committed

        if above and prev is None:
            self._flex_commit(target)
        elif above and prev != target:
            # Crossed clean through the dead zone inside a single sub-step. Record both
            # halves so the release/commit counts stay consistent with the trajectory.
            self._flex_release()
            self._flex_commit(target)
        elif not above and prev is not None:
            self._flex_release()

    def _flex_commit(self, target: int) -> None:
        """Latch a commitment (non-absorbing) and update the transition counters."""
        self.committed = int(target)
        self.t_commit = self.t_evidence
        self.rt = self.t_evidence + self.T0
        self.crossed_this_step = True
        self.n_commits += 1
        self.pending_transitions.append({
            "t": float(self.t_evidence), "type": "commit",
            "target": int(target), "x": float(self.x), "z": float(self.z_current),
        })
        if self.t_first_commit is None:
            self.t_first_commit = float(self.t_evidence)
        # A REVERSAL is a commitment to a DIFFERENT target than the previous one. That
        # is what makes it immune to boundary chatter (Section 3.2): a flicker
        # re-commits to the same target and correctly does not count.
        if self.last_committed_target is not None and self.last_committed_target != target:
            self.n_reversals += 1
            self.reversal_this_step = True
            self.changed_mind = True
        self.last_committed_target = int(target)

    def _flex_release(self) -> None:
        """Drop back below the boundary; the agent returns to pre-decision motion."""
        self.pending_transitions.append({
            "t": float(self.t_evidence), "type": "release",
            "target": self.committed, "x": float(self.x), "z": float(self.z_current),
        })
        self.committed = None
        self.t_commit = None
        self.released_this_step = True
        self.n_releases += 1

    def _check_commitment(self) -> bool:
        """Latch or reverse a decision. Returns True if integration should stop."""
        z = self.z_current
        if self.committed is None:
            if abs(self.x) >= z:
                self.committed = 0 if self.x > 0.0 else 1
                self.t_commit = self.t_evidence
                self.rt = self.t_evidence + self.T0
                self.crossed_this_step = True
                return not self.allow_changes_of_mind
            return False
        # Already committed and inside the change-of-mind window (Resulaj et al. 2009):
        # reverse only on crossing the *opposite* boundary.
        if self._in_com_window():
            if (self.committed == 0 and self.x <= -z) or (self.committed == 1 and self.x >= z):
                self.committed = 1 - self.committed
                self.changed_mind = True
                self.crossed_this_step = True
                self.rt = self.t_evidence + self.T0
                return True
            return False
        return True

    def _state(self, A_inst: float) -> DDMState:
        """Package the current state for the caller."""
        w = self.weights()
        return DDMState(
            x=float(self.x),
            z=float(self.z_current),
            t_evidence=float(self.t_evidence),
            committed=self.committed,
            p1=float(w[0]),
            q_hat=self.last_q_hat.copy(),
            A_inst=float(A_inst),
            crossed_this_step=bool(self.crossed_this_step),
            changed_mind=bool(self.changed_mind),
            released_this_step=bool(self.released_this_step),
            reversal_this_step=bool(self.reversal_this_step),
            x_over_z=(
                float(self.x / self.z_current) if self.z_current > _EPS else float("inf")
            ),
        )

    # ------------------------------------------------------------------
    def get_state(self) -> DDMState:
        """Return the current state without advancing."""
        return self._state(0.0)

    @property
    def implied_log_odds(self) -> float:
        """`a = 2A|x|/c^2` — the log-odds the agent currently holds (Appendix A.4)."""
        return self.dimensionless_boundary(self.A_hat, self.c, abs(self.x))
