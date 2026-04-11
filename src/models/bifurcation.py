# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
#  You may use, modify, and redistribute this file according to the terms of the
#  license. Attribution is required if this code is used in other works.
# ------------------------------------------------------------------------------

"""Bifurcation detector for the mean-field ring attractor model.

Implements the Jacobian eigenvalue lambda_1 spike detection approach (D-01 through D-05)
from Phase 2 design decisions. Detects when an agent's activity bump commits to a
target direction by monitoring Re(lambda_1) for local-maximum spikes above threshold.

Reference: arXiv:2602.05683 — Section 4.4 (Methods). CollectiPy derives its Jacobian
from the CollectiPy ODE (D-02), not from the paper's equations verbatim.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import numpy as np

logger = logging.getLogger("sim.mean_field.bifurcation")


class BifurcationDetector:
    """Detect bifurcation events via Jacobian eigenvalue lambda_1 spikes (per D-01).

    Usage pattern:
        detector = BifurcationDetector(agent_name="agent_0")
        # At each simulation tick:
        event = detector.update(tick, mf_system, bump_angle, target_angles, target_ids)
        if event:
            # bifurcation detected; event = {agent, tick, lambda1, target}
    """

    def __init__(
        self,
        agent_name: str,
        lambda_threshold: float = -0.1,
        spike_min_separation: int = 10,
    ):
        """Initialise the detector.

        Args:
            agent_name: Identifier for the owning agent (stored in events).
            lambda_threshold: Re(lambda_1) must exceed this value to register a spike.
                              Pre-bifurcation equilibrium: lambda_1 ~ -0.5 to -2.0.
                              At decision point: lambda_1 -> 0. Default: -0.1.
            spike_min_separation: Minimum ticks between consecutive spike detections.
                                   Prevents double-firing on trailing edge. Default: 10.
        """
        self.agent_name = agent_name
        self.lambda_threshold = lambda_threshold
        self.spike_min_separation = spike_min_separation
        self._buffer: deque = deque(maxlen=3)  # stores last 3 (tick, lambda1) tuples
        self._last_fire_tick: Optional[int] = None
        self.events: list[dict] = []

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
    # Spike detection logic (separated for testability)
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

        Called by update() after compute_lambda1(), and directly by unit tests
        with synthetic lambda1 values (bypassing the Jacobian computation).

        Per D-03: uses 3-sample micro-buffer for local-maximum detection.
        Per D-04: suppresses re-detection within spike_min_separation ticks.
        Per D-05: event dict has keys {agent, tick, lambda1, target}.

        Timing note: After appending tick N, the buffer holds [N-2, N-1, N].
        A local maximum at N-1 can only be confirmed once N is observed (so that
        v_confirm = lambda1[N] is available). The emitted event therefore records
        ``tick = t_peak`` (the N-1 sample, when the peak actually occurred) rather
        than the current tick N. Detection is fired one tick after the peak.

        Target-assignment caveat: ``bump_angle``, ``target_angles``, and
        ``target_ids`` are supplied by the caller at the *confirm* tick (N), not
        at the peak tick (N-1). Target assignment is therefore approximate — off
        by at most one tick of angular motion.

        Args:
            tick: Current simulation tick (the confirm tick, one after the peak).
            lambda1: Re(lambda_1) value for this tick.
            bump_angle: Current bump heading (radians), used for target assignment.
                        NOTE: evaluated at the confirm tick, not the peak tick.
            target_angles: List of known target angles (radians).
            target_ids: Corresponding target identifiers.

        Returns:
            Event dict if a bifurcation was detected, otherwise None.
        """
        self._buffer.append((tick, lambda1))

        if len(self._buffer) < 3:
            return None

        # After appending tick N, buffer is: [N-2, N-1, N]
        # t_peak / v_peak: the middle sample — the actual peak tick (N-1)
        # v_confirm: the newest sample — used only to confirm descent (N)
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
    # Main entry point
    # ------------------------------------------------------------------

    def update(
        self,
        tick: int,
        mf: "MeanFieldSystem",
        bump_angle: float,
        target_angles: list[float],
        target_ids: list[str],
    ) -> Optional[dict]:
        """Check for a bifurcation event at this tick.

        Computes Re(lambda_1) from the current MeanFieldSystem state, then applies
        the 3-sample local-maximum spike detector.

        Args:
            tick: Current simulation tick.
            mf: MeanFieldSystem instance (provides neural state for Jacobian).
            bump_angle: Current bump heading (radians).
            target_angles: Known target angles (radians).
            target_ids: Corresponding target identifiers.

        Returns:
            Event dict if a bifurcation was detected, otherwise None.
        """
        lambda1 = self.compute_lambda1(mf)
        if lambda1 is None:
            # Computation failed (T-02-02 mitigation); skip tick
            return None

        return self._check_spike(
            tick=tick,
            lambda1=lambda1,
            bump_angle=bump_angle,
            target_angles=target_angles,
            target_ids=target_ids,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

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
        deltas = np.abs(
            (np.array(target_angles, dtype=float) - bump_angle + np.pi) % (2 * np.pi) - np.pi
        )
        idx = int(np.argmin(deltas))
        return target_ids[idx] if idx < len(target_ids) else "unknown"
