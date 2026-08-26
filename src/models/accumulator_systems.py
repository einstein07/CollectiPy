# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
#  You may use, modify, and redistribute this file according to the terms of the
#  license. Attribution is required if this code is used in other works.
# ------------------------------------------------------------------------------

"""Classical linear accumulator (DDM / LCA) decision substrate.

This is the decision substrate for the embodied DDM movement model, the classical
counterpart to `MeanFieldSystem`. It is deliberately *linear* (the only nonlinearity is
optional rectification at `y_floor`) so that any behavioural difference from the ring
attractor is attributable to mechanism, not to a hidden nonlinearity.

Timing note (see plan Section 0.3 / Task 2.3): the ring attractor runs ~500 internal
Euler steps to relaxation every control tick; it does not integrate evidence over time,
its cross-tick memory is *which basin* the state fell into. This accumulator does the
opposite: it integrates ONCE per control tick (optionally with `n_sub` sub-steps),
across ticks, in per-second units. That asymmetry is the experiment and must not be
"fixed".

The system is index-based: it holds one accumulator per target *identity*, persisting
across ticks (object permanence), which the angle-indexed ring attractor gets for free.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable, Mapping, Optional

import numpy as np

logger = logging.getLogger("sim.accumulator")


class AccumulatorSystem:
    """Leaky competing accumulator / race / two-choice DDM over target identities.

    Dynamics (per accumulator i, in per-second units):

        dy_i = ( -lambda * y_i + mu_i - beta_inh * sum_{k != i} y_k ) * dt
               + sigma * sqrt(dt) * xi_i
        y_i  = max(y_i, y_floor)                 # rectification, if y_floor is finite

    where the drift `mu_i = gamma * evidence_tilde_i` is built by `compute_evidence`.
    """

    def __init__(
        self,
        max_targets: int,
        target_ids: Iterable[str] | None = None,
        masked_policy: str = "leak",
        # --- evidence ---
        dist_mode: str = "none",
        d_0: float = 1.0,
        target_radius: float = 0.05,
        loom_filter_ticks: float = 4.0,
        attention_mode: str = "none",
        kappa_a: float = 4.0,
        saccade_rate_hz: float = 2.0,
        normalize: str = "divisive",
        sigma_n: float = 0.1,
        gamma: float = 1.0,
        sigma_s: float = 0.0,
        target_quality_modulations: Mapping[str, Mapping[str, float]] | None = None,
        sensory_time_mode: str = "world_time",
        sensory_dt: float = 1.0,
        # --- accumulator ---
        accumulator_mode: str = "lca",
        lambda_leak: float = 1.0,
        beta_inh: float = 1.0,
        sigma: float = 0.1,
        y_floor: Optional[float] = 0.0,
        n_sub: int = 1,
        rng: np.random.Generator | None = None,
    ):
        if max_targets <= 0:
            raise ValueError("max_targets must be positive")
        self.N_max = int(max_targets)
        self.masked_policy = str(masked_policy).strip().lower()
        if self.masked_policy not in {"leak", "freeze"}:
            raise ValueError("masked_policy must be 'leak' or 'freeze'")

        # Evidence config
        self.dist_mode = str(dist_mode).strip().lower()
        self.d_0 = float(d_0)
        self.target_radius = float(target_radius)
        self.loom_filter_ticks = float(loom_filter_ticks)
        self.attention_mode = str(attention_mode).strip().lower()
        self.kappa_a = float(kappa_a)
        self.saccade_rate_hz = float(saccade_rate_hz)
        self.normalize = str(normalize).strip().lower()
        self.sigma_n = float(sigma_n)
        self.gamma = float(gamma)
        self.sigma_s = float(sigma_s)
        self.target_quality_modulations = self._normalize_modulations(target_quality_modulations)
        self.sensory_time_mode = str(sensory_time_mode).strip().lower()
        self.sensory_dt = float(sensory_dt)

        # Accumulator config
        self.accumulator_mode = str(accumulator_mode).strip().lower()
        if self.accumulator_mode not in {"race", "lca", "ddm2"}:
            raise ValueError("accumulator_mode must be 'race', 'lca', or 'ddm2'")
        self.lambda_leak = float(lambda_leak)
        self.beta_inh = float(beta_inh)
        self.sigma = float(sigma)
        self.y_floor = None if y_floor is None else float(y_floor)
        self.n_sub = max(1, int(n_sub))
        self.rng = rng if rng is not None else np.random.default_rng()

        # Preallocated deterministic slots (reproducible cross-run comparison).
        self._preassigned_ids = [str(t) for t in (target_ids or [])]
        if len(self._preassigned_ids) > self.N_max:
            raise ValueError(
                f"target_ids ({len(self._preassigned_ids)}) exceeds max_targets ({self.N_max})"
            )

        # State (allocated in reset()).
        self._slots: dict[str, int] = {}
        self._next_slot: int = 0
        self.y = np.zeros(self.N_max, dtype=float)
        self.mask = np.zeros(self.N_max, dtype=bool)
        self._in_play = np.zeros(self.N_max, dtype=bool)
        self.bias = np.zeros(self.N_max, dtype=float)
        self._alpha_prev = np.full(self.N_max, -1.0, dtype=float)   # loom: -1 = unset
        self._alpha_dot_filt = np.zeros(self.N_max, dtype=float)
        self.sensory_time = 0.0
        self._saccade_center = 0.0
        self._last_saccade_time = 0.0

        # Logging snapshots (per-slot, refreshed each step()).
        self.last_e = np.zeros(self.N_max, dtype=float)
        self.last_etilde = np.zeros(self.N_max, dtype=float)
        self.last_mu = np.zeros(self.N_max, dtype=float)
        self.last_ids: list[str] = []
        self.last_indices = np.array([], dtype=int)

        self.reset()

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_modulations(
        modulations: Mapping[str, Mapping[str, float]] | None,
    ) -> dict[str, dict[str, float]]:
        """Normalize per-target sinusoidal modulation parameters (epsilon, omega, psi)."""
        if not modulations:
            return {}
        out: dict[str, dict[str, float]] = {}
        for tid, params in modulations.items():
            out[str(tid)] = {
                "epsilon": float(params.get("epsilon", 0.0)),
                "omega": float(params.get("omega", 0.0)),
                "psi": float(params.get("psi", 0.0)),
            }
        return out

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Zero all accumulators and re-draw the frozen per-slot sensor bias."""
        self.y[:] = 0.0
        self.mask[:] = False
        self._in_play[:] = False
        self._alpha_prev[:] = -1.0
        self._alpha_dot_filt[:] = 0.0
        self.sensory_time = 0.0
        self._saccade_center = 0.0
        self._last_saccade_time = 0.0
        self.last_e[:] = 0.0
        self.last_etilde[:] = 0.0
        self.last_mu[:] = 0.0
        self.last_ids = []
        self.last_indices = np.array([], dtype=int)

        # Deterministic slot preassignment.
        self._slots = {}
        self._next_slot = 0
        for tid in self._preassigned_ids:
            self._slots[tid] = self._next_slot
            self._next_slot += 1

        # Frozen sensor bias, analogue of MeanFieldSystem.sigma_s (drawn once per reset).
        if self.sigma_s > 0.0:
            self.bias = self.rng.standard_normal(self.N_max) * self.sigma_s
        else:
            self.bias = np.zeros(self.N_max, dtype=float)

    # ------------------------------------------------------------------
    # Slot registry (Task 2.1)
    # ------------------------------------------------------------------
    def register_targets(self, ids: Iterable[str]) -> np.ndarray:
        """Map target identity strings to accumulator indices, assigning new ones.

        Never resizes the bank mid-trial: resizing would destroy accumulated evidence
        and make RT undefined. Raises if the number of distinct identities exceeds
        max_targets.
        """
        indices = []
        for raw in ids:
            tid = str(raw)
            slot = self._slots.get(tid)
            if slot is None:
                if self._next_slot >= self.N_max:
                    raise ValueError(
                        f"AccumulatorSystem: target '{tid}' would need slot "
                        f"{self._next_slot} but max_targets={self.N_max}. "
                        "Increase max_targets; never resize mid-trial."
                    )
                slot = self._next_slot
                self._slots[tid] = slot
                self._next_slot += 1
            self._in_play[slot] = True
            indices.append(slot)
        return np.array(indices, dtype=int)

    # ------------------------------------------------------------------
    # Evidence construction (Task 2.2)
    # ------------------------------------------------------------------
    def _apply_target_quality_modulation(self, ids: list[str], s: np.ndarray) -> np.ndarray:
        """Apply per-target sinusoidal quality modulation on the shared sensory clock.

        Mirrors MeanFieldSystem._apply_target_quality_modulation so that
        `target_quality_modulations` behaves identically in both models.
        """
        if not self.target_quality_modulations:
            return s
        out = s.astype(float).copy()
        for idx, tid in enumerate(ids):
            params = self.target_quality_modulations.get(str(tid))
            if params is None:
                continue
            out[idx] *= 1.0 + params["epsilon"] * math.sin(
                params["omega"] * self.sensory_time + params["psi"]
            )
        return out

    def compute_evidence(
        self,
        indices: np.ndarray,
        phi: np.ndarray,
        d: np.ndarray,
        s: np.ndarray,
        ids: list[str],
        heading: float,
        dt: float,
    ) -> np.ndarray:
        """Build the drift vector mu (shape (N_max,)) from a percept of K seen targets.

        Stages: (1) base salience + quality modulation, (2) distance modulation,
        (3) attentional gate, frozen bias, (4) divisive normalization, then
        mu = gamma * evidence_tilde. Slots not in `indices` get mu = 0.
        """
        eps = 1e-9
        phi = np.asarray(phi, dtype=float)
        d = np.asarray(d, dtype=float)
        s = np.asarray(s, dtype=float)

        # Stage 1: base salience + sinusoidal quality modulation.
        e = self._apply_target_quality_modulation(ids, s)

        # Stage 2: distance modulation.
        if self.dist_mode == "exp":
            e = e * np.exp(-d / max(self.d_0, eps))
        elif self.dist_mode == "subtense":
            e = e * (2.0 / math.pi) * np.arctan(self.target_radius / np.maximum(d, eps))
        elif self.dist_mode == "loom":
            alpha = 2.0 * np.arctan(self.target_radius / np.maximum(d, eps))
            prev = self._alpha_prev[indices]
            unset = prev < 0.0
            alpha_dot_raw = np.where(unset, 0.0, (alpha - prev) / max(dt, eps))
            tau = max(1.0, self.loom_filter_ticks)
            filt_prev = self._alpha_dot_filt[indices]
            alpha_dot = filt_prev + (alpha_dot_raw - filt_prev) / tau
            alpha_dot = np.where(unset, 0.0, alpha_dot)
            self._alpha_dot_filt[indices] = alpha_dot
            self._alpha_prev[indices] = alpha
            e = e * np.maximum(alpha_dot, 0.0)
        elif self.dist_mode != "none":
            raise ValueError(f"unknown dist_mode '{self.dist_mode}'")

        # Stage 3: attentional gate (aDDM analogue). vonmises centres on the readout
        # heading (closed-loop, prone to lock-in); saccade uses an exogenous centre.
        if self.attention_mode == "vonmises":
            e = e * np.exp(self.kappa_a * (np.cos(phi - heading) - 1.0))
        elif self.attention_mode == "saccade":
            e = e * np.exp(self.kappa_a * (np.cos(phi - self._saccade_center) - 1.0))
        elif self.attention_mode != "none":
            raise ValueError(f"unknown attention_mode '{self.attention_mode}'")

        # Frozen per-slot sensor bias.
        e = e + self.bias[indices]

        # Stage 4: divisive normalization (keeps commitment thresholds comparable
        # across option-count K and distance conditions).
        if self.normalize == "divisive":
            etilde = e / (self.sigma_n + float(np.sum(e)))
        elif self.normalize == "none":
            etilde = e
        else:
            raise ValueError(f"unknown normalize '{self.normalize}'")

        mu_seen = self.gamma * etilde

        # Scatter to full-length vectors; record snapshots for logging.
        mu = np.zeros(self.N_max, dtype=float)
        mu[indices] = mu_seen
        self.last_e[:] = 0.0
        self.last_etilde[:] = 0.0
        self.last_e[indices] = e
        self.last_etilde[indices] = etilde
        self.last_mu = mu
        return mu

    # ------------------------------------------------------------------
    # Saccade process (exogenous fixation switching for attention_mode="saccade")
    # ------------------------------------------------------------------
    def _advance_saccade(self, phi: np.ndarray, dt: float) -> None:
        """Advance the exogenous fixation process, decoupled from the readout."""
        if self.attention_mode != "saccade" or phi.size == 0:
            return
        # Periodic switch every 1/saccade_rate_hz seconds to a uniformly random target.
        period = 1.0 / max(self.saccade_rate_hz, 1e-6)
        if (self.sensory_time - self._last_saccade_time) >= period:
            self._saccade_center = float(phi[self.rng.integers(0, phi.size)])
            self._last_saccade_time = self.sensory_time

    # ------------------------------------------------------------------
    # Integration (Task 2.3)
    # ------------------------------------------------------------------
    def _effective_rates(self) -> tuple[float, float]:
        """Return (lambda, beta_inh) for the configured accumulator_mode."""
        if self.accumulator_mode == "race":
            return 0.0, 0.0
        if self.accumulator_mode == "ddm2":
            # Pure integrators; the decision variable is the difference y0 - y1.
            return 0.0, 0.0
        return self.lambda_leak, self.beta_inh  # lca

    def step(
        self,
        ids: Iterable[str],
        phi: np.ndarray,
        d: np.ndarray,
        s: np.ndarray,
        heading: float,
        dt: float,
        mu_external: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Advance all accumulators by one control tick of length `dt` seconds.

        `ids/phi/d/s` describe the K targets seen this tick. Evidence is zero-order-held
        across the `n_sub` Euler-Maruyama sub-steps. `mu_external` (shape (N_max,)) is an
        optional additive drift term used for guard->drift coupling (Phase 4), applied
        after the evidence stages and never smeared between options. Returns `y`.
        """
        ids = [str(t) for t in ids]
        indices = self.register_targets(ids)
        self.last_ids = ids
        self.last_indices = indices

        # mask = seen this tick.
        self.mask[:] = False
        if indices.size:
            self.mask[indices] = True

        if self.accumulator_mode == "ddm2" and self._next_slot > 2:
            raise ValueError("accumulator_mode 'ddm2' supports at most 2 targets")

        self._advance_saccade(np.asarray(phi, dtype=float), dt)
        mu = self.compute_evidence(indices, phi, d, s, ids, heading, dt)
        if mu_external is not None:
            mu = mu + np.asarray(mu_external, dtype=float)
            self.last_mu = mu

        lam, beta = self._effective_rates()
        in_play = self._in_play
        dt_sub = dt / self.n_sub
        sqrt_dt_sub = math.sqrt(dt_sub)

        for _ in range(self.n_sub):
            total = float(np.sum(self.y * in_play))
            inhib = beta * (total - self.y)
            drift = -lam * self.y + mu - inhib
            noise = self.sigma * sqrt_dt_sub * self.rng.standard_normal(self.N_max)
            increment = drift * dt_sub + noise
            # Only in-play accumulators evolve; never-registered slots stay silent.
            increment = np.where(in_play, increment, 0.0)
            if self.masked_policy == "freeze":
                # Unseen (but in-play) slots hold their value.
                increment = np.where(self.mask, increment, 0.0)
            self.y = self.y + increment
            if self.y_floor is not None:
                self.y = np.maximum(self.y, self.y_floor)
            self.y = np.where(in_play, self.y, 0.0)

        self._advance_sensory_time()
        return self.y

    def _advance_sensory_time(self) -> None:
        """Advance the modulation clock one control tick."""
        self.sensory_time += self.sensory_dt

    # ------------------------------------------------------------------
    # Readout helpers / diagnostics
    # ------------------------------------------------------------------
    @property
    def lambda1(self) -> float:
        """Leading Jacobian eigenvalue of the difference modes: beta_inh - lambda_leak.

        For the linear LCA the K-1 difference modes destabilize at exactly
        beta_inh == lambda_leak, independent of K (see plan Phase 5). Exposed so the
        shared BifurcationDetector can run in lambda_threshold mode on the DDM with a
        closed-form eigenvalue instead of a numerical spectrum.
        """
        return self.beta_inh - self.lambda_leak

    def two_choice_difference(self) -> float:
        """Decision variable x = y0 - y1 for the two-choice DDM reduction (ddm2)."""
        if self._next_slot < 2:
            return float(self.y[0]) if self._next_slot == 1 else 0.0
        return float(self.y[0] - self.y[1])

    def in_play_mask(self) -> np.ndarray:
        """Boolean mask of accumulators that have ever been registered."""
        return self._in_play.copy()

    def slot_of(self, target_id: str) -> Optional[int]:
        """Return the accumulator index for a target identity, if assigned."""
        return self._slots.get(str(target_id))

    def get_state(self) -> np.ndarray:
        """Return a copy of the current accumulator state."""
        return self.y.copy()
