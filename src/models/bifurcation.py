# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
#  You may use, modify, and redistribute this file according to the terms of the
#  license. Attribution is required if this code is used in other works.
# ------------------------------------------------------------------------------

"""Bifurcation detector for the mean-field ring attractor model.

Implements two named detection modes (D-01):

  "behavioral": fires when the activity bump aligns with a target for N consecutive
               ticks (D-02). Model-agnostic.

  "analytical": uses model-specific analytical metrics —
               standard model (g_adapt==0): gradient-of-lambda1 local maximum (D-03)
               SFA model (g_adapt>0): Omega threshold-crossing (D-04, Ermentrout et al. 2014)

The original _check_spike() method is preserved unchanged for backward compatibility
with existing tests that call it directly via the feed_sequence() helper.

References:
  arXiv:2602.05683 — Section 4.4: Jacobian eigenvalue lambda_1 approach (standard model)
  Ermentrout, Folias & Kilpatrick (2014), Eq. 4.23/4.26: Omega criterion (SFA model)
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import numpy as np

logger = logging.getLogger("sim.mean_field.bifurcation")


class BifurcationDetector:
    """Detect bifurcation events in the mean-field ring attractor model.

    Supports two named detection modes:

        "behavioral" (default): fires when the bump angle stays within
            alignment_tolerance_deg of any known target for
            alignment_consecutive_ticks consecutive simulation ticks.

        "analytical": uses model-specific analytical metrics —
            standard model: gradient-of-lambda1 local max above gradient_threshold
            SFA model: Omega threshold-crossing (Hopf bifurcation criterion)

    Backward compatibility: _check_spike() is preserved unchanged; existing tests
    using feed_sequence() continue to work. New modes use metric+mode event schema;
    old _check_spike() uses lambda1 key.

    Usage pattern:
        detector = BifurcationDetector(agent_name="agent_0", mode="behavioral")
        # At each simulation tick:
        event = detector.update(tick, mf_system, bump_angle, target_angles, target_ids)
        if event:
            # bifurcation detected; event = {agent, tick, metric, target, mode}
    """

    def __init__(
        self,
        agent_name: str,
        lambda_threshold: float = -0.1,
        spike_min_separation: int = 10,
        mode: str = "behavioral",
        alignment_tolerance_deg: float = 5.0,
        alignment_consecutive_ticks: int = 5,
        gradient_window: int = 5,
        gradient_threshold: float = 0.005,
    ):
        """Initialise the detector.

        Args:
            agent_name: Identifier for the owning agent (stored in events).
            lambda_threshold: Re(lambda_1) threshold for _check_spike() backward-compat path.
                              Default: -0.1.
            spike_min_separation: Minimum ticks between consecutive spike detections.
                                   Prevents double-firing on trailing edge. Default: 10.
            mode: Detection mode — "behavioral" or "analytical". Default: "behavioral".
            alignment_tolerance_deg: Tolerance (degrees) for behavioral alignment check.
                                      Default: 5.0.
            alignment_consecutive_ticks: Number of consecutive aligned ticks required to
                                          fire a behavioral event. Default: 5.
            gradient_window: Window size (ticks) for discrete gradient computation.
                             Default: 5.
            gradient_threshold: Minimum gradient value to fire an analytical event.
                                 Default: 0.005.
        """
        self.agent_name = agent_name
        self.lambda_threshold = lambda_threshold
        self.spike_min_separation = spike_min_separation
        self._buffer: deque = deque(maxlen=3)  # stores last 3 (tick, lambda1) tuples
        self._last_fire_tick: Optional[int] = None
        self.events: list[dict] = []
        self.last_lambda1: Optional[float] = None  # last successfully computed Re(lambda_1)
        self.retrigger = False  # whether to allow re-triggering after firing (resets alignment counter)

        # Mode configuration
        self.mode = mode  # "behavioral" or "analytical"
        self.alignment_tolerance_rad = alignment_tolerance_deg * np.pi / 180.0
        self.alignment_consecutive_ticks = alignment_consecutive_ticks
        self.gradient_window = gradient_window
        self.gradient_threshold = gradient_threshold

        # Behavioral mode state
        self._alignment_counter: int = 0
        self._alignment_target: Optional[str] = None

        # Gradient criterion state (analytical/standard)
        self._lambda1_history: deque = deque(maxlen=max(gradient_window + 1, 3))
        self._gradient_history: deque = deque(maxlen=3)  # for gradient local-max detection

        # Omega criterion state (analytical/SFA)
        self._omega_buffer: deque = deque(maxlen=3)  # for 3-sample crossing detection
        self.last_omega: Optional[float] = None  # exposed for pkl logging

    # ------------------------------------------------------------------
    # Jacobian and eigenvalue computation
    # ------------------------------------------------------------------

    def compute_jacobian(self, mf: "MeanFieldSystem") -> np.ndarray:
        """Compute the z-subspace Jacobian per D-02.

        Formula:
            J(z) = -I + diag((u - s) * sech^2((u - s) * M @ z + b - beta)) @ M

        where:
            Standard model (g_adapt == 0.0):  s = 0  (scalar)
            SFA model (g_adapt > 0.0):        s = mf.adapt_ring  (vector, element-wise)

        Both paths are handled identically via numpy broadcasting.

        Args:
            mf: MeanFieldSystem instance providing the current neural state.

        Returns:
            J: (num_neurons x num_neurons) float64 ndarray.
        """
        z = mf.neural_ring
        M = mf.M
        b = mf.b
        u = mf.u
        beta = mf.beta
        n = mf.num_neurons

        # s is 0.0 for standard model, or the adapt_ring vector for SFA
        s = mf.adapt_ring if mf.g_adapt > 0.0 else 0.0

        # Argument to sech^2: (u - s) * M @ z + b - beta
        arg = (u - s) * (M @ z) + b - beta
        # sech^2(x) = 1 / cosh^2(x)
        sech2 = 1.0 / np.cosh(arg) ** 2

        # J_ij = -delta_ij + (u - s_i) * sech2_i * M_ij
        # Written as: J = -I + diag((u - s) * sech2) @ M
        J = -np.eye(n) + np.diag((u - s) * sech2) @ M

        return J

    def compute_lambda1(self, mf: "MeanFieldSystem") -> Optional[float]:
        """Return Re(lambda_1) — the largest real part of Jacobian eigenvalues.

        Per T-02-02 threat mitigation: if eigenvalue computation fails (singular matrix,
        NaN in Jacobian), logs a warning and returns None so the caller can skip the tick.

        Args:
            mf: MeanFieldSystem instance.

        Returns:
            float: Re(lambda_1), or None if computation failed.
        """
        try:
            J = self.compute_jacobian(mf)
            if not np.all(np.isfinite(J)):
                logger.warning(
                    "BifurcationDetector: Jacobian contains non-finite values at step %s "
                    "for agent '%s' — skipping tick",
                    getattr(mf, '_step_count', '?'),
                    self.agent_name,
                )
                return None
            eigenvalues = np.linalg.eigvals(J)
            return float(np.max(eigenvalues.real))
        except (np.linalg.LinAlgError, ValueError) as exc:
            logger.warning(
                "BifurcationDetector: eigvals failed for agent '%s' at step %s: %s — skipping tick",
                self.agent_name,
                getattr(mf, '_step_count', '?'),
                exc,
            )
            return None

    # ------------------------------------------------------------------
    # Omega computation (SFA model criterion — D-04)
    # ------------------------------------------------------------------

    def compute_omega(self, mf: "MeanFieldSystem", perception_vec=None) -> Optional[float]:
        """Compute Omega scalar for SFA ring attractor (Ermentrout, Folias & Kilpatrick 2014).

        Omega = (1+beta_paper)*A / ((1+beta_paper)*A + I0 + 1e-12)

        where:
            beta_paper = mf.g_adapt (adaptation strength)
            A = first Fourier mode magnitude of neural_ring
            I0 = first Fourier mode magnitude of sensory input (mf.b)

        T-02-08 mitigation: +1e-12 denominator guard prevents division by zero.
        T-02-09 mitigation: returns None if perception_vec is None or empty.

        Args:
            mf: MeanFieldSystem instance.
            perception_vec: Optional explicit perception vector. If None, uses mf.b.

        Returns:
            Omega scalar in [0, 1], or None if computation fails.
        """
        try:
            beta_paper = mf.g_adapt
            theta = mf.theta
            n = mf.num_neurons

            # First Fourier mode of neural_ring: A = |sum(z_i * exp(i*theta_i))| / (N/2)
            z_complex = np.sum(mf.neural_ring * np.exp(1j * theta))
            A = np.abs(z_complex) / (n / 2.0)

            # First Fourier mode of perception/input
            input_vec = perception_vec if perception_vec is not None else mf.b
            if input_vec is None or len(input_vec) == 0:
                return None
            i0_complex = np.sum(input_vec * np.exp(1j * theta))
            I0 = np.abs(i0_complex) / (n / 2.0)

            # Omega (Eq. 4.23)
            numerator = (1.0 + beta_paper) * A
            denominator = (1.0 + beta_paper) * A + I0 + 1e-12  # T-02-08
            omega = numerator / denominator

            if not np.isfinite(omega):
                logger.warning(
                    "BifurcationDetector: Omega non-finite for agent '%s' (A=%.6f, I0=%.6f)",
                    self.agent_name, A, I0,
                )
                return None

            return float(omega)
        except Exception as exc:
            logger.warning(
                "BifurcationDetector: Omega computation failed for agent '%s': %s",
                self.agent_name, exc,
            )
            return None

    # ------------------------------------------------------------------
    # Spike detection logic (preserved for backward compatibility)
    # ------------------------------------------------------------------

    def _check_spike(
        self,
        tick: int,
        lambda1: float,
        bump_angle: float,
        target_angles: list[float],
        target_ids: list[str],
    ) -> Optional[dict]:
        """Core spike-detection logic (3-sample local-maximum + threshold check).

        PRESERVED UNCHANGED for backward compatibility with existing tests.
        Called by unit tests via feed_sequence() helper (bypassing compute_lambda1()).

        Per D-03: uses 3-sample micro-buffer for local-maximum detection.
        Per D-04: suppresses re-detection within spike_min_separation ticks.
        Per D-05: event dict has keys {agent, tick, lambda1, target}.

        Note: This method uses the OLD event schema with 'lambda1' key (not 'metric').
        New mode-specific paths (_update_behavioral, _check_gradient, _check_omega_crossing)
        use the NEW schema with 'metric' and 'mode' keys.

        Timing note: After appending tick N, the buffer holds [N-2, N-1, N].
        A local maximum at N-1 can only be confirmed once N is observed. The emitted
        event records ``tick = t_peak`` (the N-1 sample, when the peak actually occurred).

        Args:
            tick: Current simulation tick (the confirm tick, one after the peak).
            lambda1: Re(lambda_1) value for this tick.
            bump_angle: Current bump heading (radians), used for target assignment.
            target_angles: List of known target angles (radians).
            target_ids: Corresponding target identifiers.

        Returns:
            Event dict if a bifurcation was detected, otherwise None.
        """
        self._buffer.append((tick, lambda1))

        if len(self._buffer) < 3:
            return None

        # After appending tick N, buffer is: [N-2, N-1, N]
        (_, v_prev), (t_peak, v_peak), (_, v_confirm) = self._buffer

        # Local maximum check + threshold check (D-03)
        if v_prev < v_peak > v_confirm and v_peak > self.lambda_threshold:
            # Suppression check (D-04); use t_peak (confirmed peak time)
            if (
                self._last_fire_tick is not None
                and t_peak - self._last_fire_tick < self.spike_min_separation
            ):
                return None

            # Find nearest target (D-05)
            nearest_target = self._nearest_target(bump_angle, target_angles, target_ids)

            event = {
                "agent": self.agent_name,
                "tick": int(t_peak),   # tick when peak occurred (one tick before detection)
                "lambda1": float(round(v_peak, 6)),
                "target": nearest_target,
                "x": self._agent_x,
                "y": self._agent_y,
                "orientation": self._agent_orientation,
            }
            self.events.append(event)
            self._last_fire_tick = t_peak
            logger.info(
                "Bifurcation detected: agent=%s tick=%d lambda1=%.6f target=%s",
                self.agent_name,
                t_peak,
                v_peak,
                nearest_target,
            )
            return event

        return None

    # ------------------------------------------------------------------
    # Behavioral mode (D-02)
    # ------------------------------------------------------------------

    def _update_behavioral(
        self,
        tick: int,
        bump_angle: float,
        target_angles: list[float],
        target_ids: list[str],
    ) -> Optional[dict]:
        """Behavioral mode: fire when bump aligns with a target for N consecutive ticks.

        Per D-02: fires a bifurcation event when the ring attractor bump angle is within
        alignment_tolerance_rad of any known target for alignment_consecutive_ticks
        consecutive simulation ticks. Re-triggerable via suppression window.

        Args:
            tick: Current simulation tick.
            bump_angle: Current bump heading (radians).
            target_angles: List of known target angles (radians).
            target_ids: Corresponding target identifiers.

        Returns:
            Event dict if behavioral bifurcation detected, otherwise None.
        """
        if bump_angle is None or not target_angles:
            self._alignment_counter = 0
            self._alignment_target = None
            return None

        # Find which target (if any) the bump is aligned with
        aligned_target_id = None
        for angle, tid in zip(target_angles, target_ids):
            delta = abs(((bump_angle - angle + np.pi) % (2 * np.pi)) - np.pi)
            if delta <= self.alignment_tolerance_rad:
                aligned_target_id = tid
                break

        if aligned_target_id is not None:
            if aligned_target_id == self._alignment_target:
                self._alignment_counter += 1
            else:
                # Switched to a different target — reset counter
                if self._alignment_target is not None:
                    self.retrigger = True  # allow re-triggering if we switch targets
                self._alignment_target = aligned_target_id
                self._alignment_counter = 1
        else:
            # Not aligned with any target — reset
            self._alignment_counter = 0
            self._alignment_target = None
            self.retrigger = False  # reset re-trigger flag when losing alignment
            return None

        if self._alignment_counter >= self.alignment_consecutive_ticks:
            # Suppression check
            if (self._last_fire_tick is not None
                    and tick - self._last_fire_tick < self.spike_min_separation or self._last_fire_tick is not None and not self.retrigger):
                return None

            event = {
                "agent": self.agent_name,
                "tick": int(tick),
                "metric": float(bump_angle),
                "target": aligned_target_id,
                "mode": "behavioral",
                "x": self._agent_x,
                "y": self._agent_y,
                "orientation": self._agent_orientation,
            }
            self.events.append(event)
            self._last_fire_tick = tick
            self._alignment_counter = 0  # reset after fire for re-triggerability
            logger.info(
                "Behavioral bifurcation detected: agent=%s tick=%d target=%s bump_angle=%.4f",
                self.agent_name, tick, aligned_target_id, bump_angle,
            )
            return event

        return None

    # ------------------------------------------------------------------
    # Behavioral mode — agent heading angle criterion
    # ------------------------------------------------------------------

    def _update_behavioral_agent_angle(
        self,
        tick: int,
        agent_angle: float,
        target_angles: list[float],
        target_ids: list[str],
    ) -> Optional[dict]:
        """Behavioral mode: fire when the agent's physical heading aligns with a target.

        Analogous to _update_behavioral but uses the agent's own locomotion heading
        angle rather than the ring attractor bump angle. Fires a bifurcation event when
        agent_angle is within alignment_tolerance_rad of any known target for
        alignment_consecutive_ticks consecutive simulation ticks.

        Args:
            tick: Current simulation tick.
            agent_angle: Agent's current heading angle (radians).
            target_angles: List of known target angles (radians).
            target_ids: Corresponding target identifiers.

        Returns:
            Event dict if behavioral bifurcation detected, otherwise None.
        """
        if agent_angle is None or not target_angles:
            self._alignment_counter = 0
            self._alignment_target = None
            return None

        # Find which target (if any) the agent heading is aligned with
        aligned_target_id = None
        for angle, tid in zip(target_angles, target_ids):
            delta = abs(((agent_angle - angle + np.pi) % (2 * np.pi)) - np.pi)
            if delta <= self.alignment_tolerance_rad:
                aligned_target_id = tid
                break

        if aligned_target_id is not None:
            if aligned_target_id == self._alignment_target:
                self._alignment_counter += 1
            else:
                # Switched to a different target — reset counter
                if self._alignment_target is not None:
                    self.retrigger = True
                self._alignment_target = aligned_target_id
                self._alignment_counter = 1
        else:
            # Not aligned with any target — reset
            self._alignment_counter = 0
            self._alignment_target = None
            self.retrigger = False
            return None

        if self._alignment_counter >= self.alignment_consecutive_ticks:
            # Suppression check
            if (self._last_fire_tick is not None
                    and tick - self._last_fire_tick < self.spike_min_separation or self._last_fire_tick is not None and not self.retrigger):
                return None

            event = {
                "agent": self.agent_name,
                "tick": int(tick),
                "metric": float(agent_angle),
                "target": aligned_target_id,
                "mode": "behavioral_agent_angle",
                "x": self._agent_x,
                "y": self._agent_y,
                "orientation": self._agent_orientation,
            }
            self.events.append(event)
            self._last_fire_tick = tick
            self._alignment_counter = 0
            logger.info(
                "Behavioral (agent angle) bifurcation detected: agent=%s tick=%d target=%s agent_angle=%.4f",
                self.agent_name, tick, aligned_target_id, agent_angle,
            )
            return event

        return None

    # ------------------------------------------------------------------
    # Analytical mode — gradient criterion (standard model, D-03)
    # ------------------------------------------------------------------

    def _check_gradient(
        self,
        tick: int,
        lambda1: float,
        bump_angle: float,
        target_angles: list[float],
        target_ids: list[str],
    ) -> Optional[dict]:
        """Gradient-of-lambda1 criterion for standard model analytical detection.

        Maintains a rolling window of lambda1 values. Computes the discrete gradient
        dL = (lambda1[t] - lambda1[t - gradient_window]) / gradient_window.
        Fires when dL is at a local maximum above gradient_threshold.

        This fires at the steepest part of the lambda1 rise (the actual decision onset)
        rather than on the plateau where the old threshold criterion kept re-firing.

        Args:
            tick: Current simulation tick.
            lambda1: Current Re(lambda_1) value.
            bump_angle: Current bump heading (radians), for target assignment.
            target_angles: List of known target angles (radians).
            target_ids: Corresponding target identifiers.

        Returns:
            Event dict if gradient bifurcation detected, otherwise None.
        """
        self._lambda1_history.append((tick, lambda1))

        if len(self._lambda1_history) < self.gradient_window + 1:
            return None

        # Discrete gradient: (current - value_gradient_window_ago) / gradient_window
        _, l_old = self._lambda1_history[-(self.gradient_window + 1)]
        _, l_new = self._lambda1_history[-1]
        gradient = (l_new - l_old) / self.gradient_window

        # Append gradient to 3-sample buffer for local-max detection
        self._gradient_history.append((tick, gradient))

        if len(self._gradient_history) < 3:
            return None

        (_, g_prev), (t_peak, g_peak), (_, g_confirm) = self._gradient_history

        # Local maximum of gradient above threshold
        if g_prev < g_peak > g_confirm and g_peak > self.gradient_threshold:
            # Suppression check
            if (self._last_fire_tick is not None
                    and t_peak - self._last_fire_tick < self.spike_min_separation):
                return None

            nearest_target = self._nearest_target(bump_angle, target_angles, target_ids)

            event = {
                "agent": self.agent_name,
                "tick": int(t_peak),
                "metric": float(round(l_new, 6)),
                "target": nearest_target,
                "mode": "analytical",
                "x": self._agent_x,
                "y": self._agent_y,
                "orientation": self._agent_orientation,
            }
            self.events.append(event)
            self._last_fire_tick = t_peak
            logger.info(
                "Gradient bifurcation detected: agent=%s tick=%d lambda1=%.6f gradient=%.6f target=%s",
                self.agent_name, t_peak, l_new, g_peak, nearest_target,
            )
            return event

        return None

    # ------------------------------------------------------------------
    # Analytical mode — Omega criterion (SFA model, D-04)
    # ------------------------------------------------------------------

    def _check_omega_crossing(
        self,
        tick: int,
        omega: float,
        bump_angle: float,
        target_angles: list[float],
        target_ids: list[str],
        mf: "MeanFieldSystem",
    ) -> Optional[dict]:
        """Detect when Omega crosses the Hopf bifurcation threshold.

        Threshold = (1 + alpha) / (1 + beta) where alpha = 1/tau_adapt, beta = g_adapt.
        Per Ermentrout, Folias & Kilpatrick (2014), Eq. 4.26.

        Uses 3-sample buffer: fires when omega[t-1] < threshold <= omega[t] (upward crossing).
        Confirmed by checking that the third sample (current) is also above threshold.

        Args:
            tick: Current simulation tick.
            omega: Current Omega value.
            bump_angle: Current bump heading (radians), for target assignment.
            target_angles: List of known target angles (radians).
            target_ids: Corresponding target identifiers.
            mf: MeanFieldSystem instance (provides tau_adapt, g_adapt for threshold).

        Returns:
            Event dict if Omega crossing detected, otherwise None.
        """
        alpha_paper = 1.0 / mf.tau_adapt
        beta_paper = mf.g_adapt
        threshold = (1.0 + alpha_paper) / (1.0 + beta_paper)

        self._omega_buffer.append((tick, omega))

        if len(self._omega_buffer) < 3:
            return None

        (_, o_prev2), (t_cross, o_prev), (_, o_curr) = self._omega_buffer

        # Upward crossing: the middle sample crossed from below to above
        # Confirmed by: o_prev2 < threshold AND o_prev >= threshold AND o_curr >= threshold
        crossed = (o_prev2 < threshold) and (o_prev >= threshold)

        if not crossed:
            return None

        # Suppression check
        if (self._last_fire_tick is not None
                and t_cross - self._last_fire_tick < self.spike_min_separation):
            return None

        nearest_target = self._nearest_target(bump_angle, target_angles, target_ids)

        event = {
            "agent": self.agent_name,
            "tick": int(t_cross),
            "metric": float(round(o_prev, 6)),
            "target": nearest_target,
            "mode": "analytical",
            "x": self._agent_x,
            "y": self._agent_y,
            "orientation": self._agent_orientation,
        }
        self.events.append(event)
        self._last_fire_tick = t_cross
        logger.info(
            "Omega bifurcation detected: agent=%s tick=%d omega=%.6f threshold=%.6f target=%s",
            self.agent_name, t_cross, o_prev, threshold, nearest_target,
        )
        return event

    # ------------------------------------------------------------------
    # Analytical mode dispatch (_update_analytical)
    # ------------------------------------------------------------------

    def _update_analytical(
        self,
        tick: int,
        mf: "MeanFieldSystem",
        bump_angle: float,
        target_angles: list[float],
        target_ids: list[str],
        perception_vec=None,
    ) -> Optional[dict]:
        """Analytical mode: dispatches to gradient (standard) or Omega (SFA) criterion.

        For standard model (g_adapt==0): uses gradient-of-lambda1 local max (D-03).
        For SFA model (g_adapt>0): uses Omega threshold crossing (D-04).

        Note: metric computation is done in update() for D-05 compliance (always log
        regardless of mode). This method receives already-computed metrics and applies
        the detection criterion only.

        Args:
            tick: Current simulation tick.
            mf: MeanFieldSystem instance.
            bump_angle: Current bump heading (radians).
            target_angles: List of known target angles (radians).
            target_ids: Corresponding target identifiers.
            perception_vec: Optional perception vector for Omega (passed through from update()).

        Returns:
            Event dict if analytical bifurcation detected, otherwise None.
        """
        is_sfa = mf.g_adapt > 0.0

        if is_sfa:
            if self.last_omega is None:
                return None
            return self._check_omega_crossing(tick, self.last_omega, bump_angle,
                                              target_angles, target_ids, mf)
        else:
            if self.last_lambda1 is None:
                return None
            return self._check_gradient(tick, self.last_lambda1, bump_angle,
                                        target_angles, target_ids)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def update(
        self,
        tick: int,
        mf: "MeanFieldSystem",
        bump_angle: float,
        target_angles: list[float],
        target_ids: list[str],
        perception_vec=None,
        agent_angle: Optional[float] = None,
        agent_x: Optional[float] = None,
        agent_y: Optional[float] = None,
        agent_orientation: Optional[float] = None,
    ) -> Optional[dict]:
        """Check for a bifurcation event at this tick.

        Always computes the analytical metric for pkl logging (D-05), regardless of mode.
        Then dispatches to the mode-specific detection logic.

        Per D-05: mean_field_lambda1 and mean_field_omega are always logged to pkl
        regardless of detection mode, so researchers have the full time-series.

        Args:
            tick: Current simulation tick.
            mf: MeanFieldSystem instance (provides neural state for Jacobian/Omega).
            bump_angle: Current bump heading (radians).
            target_angles: Known target angles (radians).
            target_ids: Corresponding target identifiers.
            perception_vec: Optional raw perception vector for Omega I0 computation.

        Returns:
            Event dict if a bifurcation was detected, otherwise None.
        """
        self._agent_x = agent_x
        self._agent_y = agent_y
        self._agent_orientation = agent_orientation

        # Always compute the analytical metric for pkl logging (D-05)
        is_sfa = mf.g_adapt > 0.0
        if is_sfa:
            omega = self.compute_omega(mf, perception_vec)
            self.last_omega = omega
        else:
            lambda1 = self.compute_lambda1(mf)
            if lambda1 is not None:
                self.last_lambda1 = lambda1

        # Dispatch to mode-specific detection
        if self.mode == "behavioral":
            #return self._update_behavioral(tick, bump_angle, target_angles, target_ids)
            return self._update_behavioral_agent_angle(tick, agent_angle, target_angles, target_ids)
        elif self.mode == "analytical":
            if is_sfa:
                if self.last_omega is None:
                    return None
                return self._check_omega_crossing(tick, self.last_omega, bump_angle,
                                                  target_angles, target_ids, mf)
            else:
                if self.last_lambda1 is None:
                    return None
                return self._check_gradient(tick, self.last_lambda1, bump_angle,
                                            target_angles, target_ids)
        else:
            logger.warning(
                "Unknown bifurcation mode '%s' for agent '%s'",
                self.mode, self.agent_name,
            )
            return None

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all accumulated state for a fresh run.

        Call this at the start of each new run instead of poking internal
        attributes directly. Centralises state reset so future additions
        (e.g. running means, event counts) only need to be cleared here.
        """
        self.events.clear()
        self._buffer.clear()
        self._last_fire_tick = None
        self.last_lambda1 = None

        # Mode-specific state
        self._alignment_counter = 0
        self._alignment_target = None
        self._lambda1_history.clear()
        self._gradient_history.clear()
        self._omega_buffer.clear()
        self.last_omega = None

    @staticmethod
    def _nearest_target(
        bump_angle: float,
        target_angles: list[float],
        target_ids: list[str],
    ) -> str:
        """Return the target ID closest to the current bump angle (circular distance).

        Uses the same circular-distance convention as _delta_angle() in
        mean_field_systems.py: |((a - b + pi) % (2*pi)) - pi|.

        Args:
            bump_angle: Current bump heading (radians).
            target_angles: Candidate target angles (radians).
            target_ids: Corresponding target identifiers.

        Returns:
            Nearest target ID, or "unknown" if lists are empty / index out of range.
        """
        if not target_angles or not target_ids:
            return "unknown"
        # Clip both lists to the shorter length so a mismatched caller
        # (e.g. len(target_angles) != len(target_ids)) never produces an
        # out-of-range index or silently returns "unknown" for valid angles.
        n = min(len(target_angles), len(target_ids))
        deltas = np.abs(
            (np.array(target_angles[:n], dtype=float) - bump_angle + np.pi) % (2 * np.pi) - np.pi
        )
        idx = int(np.argmin(deltas))
        return target_ids[idx]
