# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
#  You may use, modify, and redistribute this file according to the terms of the
#  license. Attribution is required if this code is used in other works.
# ------------------------------------------------------------------------------

"""Embodied classical accumulator (DDM / LCA) movement model.

Shares the perception spine (`TargetModel`) and the actuation quirks of
`MeanFieldMovementModel` bit-for-bit, but replaces the ring-attractor decision
substrate with a linear leaky-competing-accumulator (`AccumulatorSystem`). See
EMBODIED_DDM_PLAN.md. The two models differ *only* in the decision substrate; the
difference is the experiment.

Timing (plan Section 0.3): the accumulator integrates ONCE per control tick (with
optional `n_sub` sub-steps), across ticks, in per-second units. This is deliberately
NOT the ring attractor's 500-steps-to-relaxation-per-tick regime. Do not "fix" it.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from typing import Optional

import numpy as np

from models.accumulator_systems import AccumulatorSystem
from models.bifurcation import BifurcationDetector
from models.egocentric_target_model import TargetModel
from models.readout import circular_readout
from models.utils import normalize_angle
from plugin_registry import register_movement_model

logger = logging.getLogger("sim.embodied_ddm")
logger.setLevel(logging.DEBUG)


class EmbodiedDDMMovementModel(TargetModel):
    """Movement model driven by a linear accumulator over target identities."""

    def __init__(self, agent):
        """Initialize the instance."""
        params = agent.config_elem.get("embodied_ddm", {}) or {}
        # Shared perception/detection spine (identical to the ring attractor).
        self._init_target_model(agent, params)

        # --- structure ---
        self.max_targets = int(params.get("max_targets", 8))
        self.masked_policy = str(params.get("masked_policy", "leak"))
        self.norm_scale = float(params.get("norm_scale", 1.0))
        self.pre_run_steps = max(0, int(params.get("pre_run_steps", 0)))

        # --- evidence ---
        self.dist_mode = str(params.get("dist_mode", "none"))
        self.d_0 = float(params.get("d_0", 1.0))
        self.target_radius = float(params.get("target_radius", 0.05))
        self.loom_filter_ticks = float(params.get("loom_filter_ticks", 4.0))
        self.attention_mode = str(params.get("attention_mode", "none"))
        self.kappa_a = float(params.get("kappa_a", 4.0))
        self.saccade_rate_hz = float(params.get("saccade_rate_hz", 2.0))
        self.normalize = str(params.get("normalize", "divisive"))
        self.sigma_n = float(params.get("sigma_n", 0.1))
        self.gamma = float(params.get("gamma", 1.0))
        self.sigma_s = float(params.get("sigma_s", 0.0))

        # --- accumulator ---
        self.accumulator_mode = str(params.get("accumulator_mode", "lca"))
        self.lambda_leak = float(params.get("lambda_leak", 1.0))
        self.beta_inh = float(params.get("beta_inh", 1.0))
        self.sigma = float(params.get("sigma", 0.1))
        y_floor = params.get("y_floor", 0.0)
        self.y_floor = None if y_floor is None else float(y_floor)
        self.n_sub = max(1, int(params.get("n_sub", 1)))

        # --- readout / commitment ---
        self.y_threshold = float(params.get("y_threshold", 0.0))
        self.scaling_mode = str(params.get("scaling_mode", "concentration")).strip().lower()
        if self.scaling_mode not in {"concentration", "constant", "magnitude", "norm"}:
            raise ValueError(
                "scaling_mode must be 'concentration', 'constant', 'magnitude' or 'norm'"
            )
        # Pre-commitment motion policy, shared with embodied_pure_ddm. `average` (default)
        # is the existing behaviour; `midpoint` cuts the decision->motion arm of the
        # embodiment loop while keeping motion->threshold.
        self.predecision_motion = str(
            params.get("predecision_motion", "average")
        ).strip().lower()
        if self.predecision_motion not in {"average", "midpoint", "forward", "stationary"}:
            raise ValueError(
                "predecision_motion must be 'average', 'midpoint', 'forward' or 'stationary'"
            )
        self.bisector_eps = float(params.get("bisector_eps", 1e-6))
        self.commitment_mode = str(params.get("commitment_mode", "soft"))
        self.R_commit = float(params.get("R_commit", 0.9))
        self.p_commit = float(params.get("p_commit", 0.7))
        self.R_release = float(params.get("R_release", 0.7))
        self.hold_ticks = int(params.get("hold_ticks", 5))
        self.allow_reversal = bool(params.get("allow_reversal", True))
        self.Z = float(params.get("Z", 1.0))

        # --- guards ---
        self.guard_mode = str(params.get("guard_mode", "off"))
        self.beta_g = float(params.get("beta_g", 1.0))
        self.guard_decay_rate = float(params.get("guard_decay_rate", 2.0))
        self.guard_kappa = float(params.get("guard_kappa", 20.0))

        # Display kernel for the GUI shim.
        self._display_kappa = float(params.get("display_kappa", self.guard_kappa))

        # Runtime state.
        self.accumulator: Optional[AccumulatorSystem] = None
        self._last_heading: Optional[float] = None
        self._last_concentration: float = 0.0
        self._last_magnitude: float = 0.0
        self._hold: int = 0
        self._committed: Optional[int] = None
        self._committed_bearing: float = 0.0
        self._slot_bearings: dict[int, float] = {}
        self._last_modulated_s = np.array([], dtype=float)

        # Bifurcation detection (behavioral mode works unchanged on readout quantities;
        # lambda_threshold mode can use the accumulator's closed-form lambda1).
        bif_cfg = params.get("bifurcation", {})
        self.bifurcation_detector = BifurcationDetector(
            agent_name=str(agent.get_name()),
            lambda_threshold=float(bif_cfg.get("lambda_threshold", -0.1)),
            spike_min_separation=int(bif_cfg.get("spike_min_separation", 10)),
            mode=str(bif_cfg.get("mode", "behavioral")),
            alignment_tolerance_deg=float(bif_cfg.get("alignment_tolerance_deg", 5.0)),
            alignment_consecutive_ticks=int(bif_cfg.get("alignment_consecutive_ticks", 5)),
            gradient_window=int(bif_cfg.get("gradient_window", 5)),
            gradient_threshold=float(bif_cfg.get("gradient_threshold", 0.005)),
        )
        # Sensory percept stream (FEATURE_SHARED_SENSORY_STREAM.md). Under `shared` the
        # frozen sensor bias moves upstream into the stream, so sigma_s must be 0; the
        # accumulator's own `sigma` is internal noise and is left alone.
        self._init_percept_stream(
            dist_mode=self.dist_mode,
            attention_mode=self.attention_mode,
            sigma_s=self.sigma_s,
        )
        self.reset()
        logger.info(
            "%s embodied-DDM model instantiated (max_targets=%d, mode=%s, lambda=%.3f, beta_inh=%.3f, dt=one-step-per-tick)",
            self.agent.get_name(),
            self.max_targets,
            self.accumulator_mode,
            self.lambda_leak,
            self.beta_inh,
        )

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset the accumulator state."""
        self.perception = None
        self._build_percept_stream()
        self._last_heading = None
        self._hold = 0
        self._committed = None
        self._committed_bearing = 0.0
        self._slot_bearings = {}
        self.accumulator = AccumulatorSystem(
            max_targets=self.max_targets,
            target_ids=self.target_ids or None,
            masked_policy=self.masked_policy,
            dist_mode=self.dist_mode,
            d_0=self.d_0,
            target_radius=self.target_radius,
            loom_filter_ticks=self.loom_filter_ticks,
            attention_mode=self.attention_mode,
            kappa_a=self.kappa_a,
            saccade_rate_hz=self.saccade_rate_hz,
            normalize=self.normalize,
            sigma_n=self.sigma_n,
            gamma=self.gamma,
            sigma_s=self.sigma_s,
            target_quality_modulations=self.target_quality_modulations,
            sensory_time_mode=self.sensory_time_mode,
            sensory_dt=self.sensory_dt,
            accumulator_mode=self.accumulator_mode,
            lambda_leak=self.lambda_leak,
            beta_inh=self.beta_inh,
            sigma=self.sigma,
            y_floor=self.y_floor,
            n_sub=self.n_sub,
            rng=self._make_rng("accumulator"),
        )
        if hasattr(self, "bifurcation_detector"):
            self.bifurcation_detector.reset()
        logger.debug("%s embodied-DDM system reset", self.agent.get_name())

    # ------------------------------------------------------------------
    def _zero_commands(self) -> None:
        """Stop the agent and clear transient readout state."""
        self.agent.linear_velocity_cmd = 0.0
        self.agent.angular_velocity_cmd = 0.0
        self._last_heading = None
        self._last_concentration = 0.0
        self._last_r_geom = 0.0
        self._bisector_guard_fired = False
        self._last_magnitude = 0.0

    def pre_run(self, objects: dict, agents: dict) -> None:
        """Let the accumulator settle before the main loop."""
        if self.pre_run_steps <= 0:
            return
        for _ in range(self.pre_run_steps):
            self._decide_and_actuate(objects, agents, tick=None, arena_shape=None, actuate=False)

    def step(self, agent, tick: int, arena_shape, objects: dict, agents: dict) -> None:
        """Execute a simulation step."""
        start_time = time.perf_counter()
        if self.accumulator is None:
            self.reset()
        logger.debug("--------------------%s embodied-DDM step tick=%s-------------------------", self.agent.get_name(), tick)
        try:
            self._decide_and_actuate(objects, agents, tick, arena_shape, actuate=True)
        finally:
            if logger.isEnabledFor(logging.DEBUG):
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                logger.debug("----------------------------%s embodied-DDM step duration = %.3f ms-----------------------------------", self.agent.get_name(), elapsed_ms)

    # ------------------------------------------------------------------
    def _decide_and_actuate(self, objects, agents, tick, arena_shape, actuate: bool) -> None:
        """Full per-tick pipeline: sense -> evidence -> integrate -> read out -> act."""
        self._update_perception(objects, agents, tick, arena_shape)

        # Gate on target identities being present (the DDM does not need the
        # ring-shaped perception vector; it consumes _mf_entities["targets"]).
        entities = (self._mf_entities or {}).get("targets") or []
        if not entities:
            if actuate:
                self._zero_commands()
            return

        percept = self._build_target_percept()
        ids = list(percept.ids)
        phi = np.asarray(percept.phi, dtype=float)
        d = np.asarray(percept.d, dtype=float)
        s = np.asarray(percept.s, dtype=float)

        # Optional social coupling: inject neighbours as weak pseudo-targets (same knob
        # and formula as MeanFieldMovementModel.step, so the collective extension stays
        # comparable model-for-model).
        if self.alpha > 0.0 and s.size > 0:
            neighbor_ids, neighbor_angles, n_neighbors = self._collect_neighbor_targets(agents)
            if n_neighbors > 0:
                neighbor_strength = (float(np.max(s)) * self.alpha) / n_neighbors
                ids = ids + neighbor_ids
                phi = np.concatenate([phi, neighbor_angles])
                d = np.concatenate([d, np.zeros(n_neighbors, dtype=float)])
                s = np.concatenate([s, np.full(n_neighbors, neighbor_strength, dtype=float)])

        dt = 1.0 / self._resolve_agent_tick_rate()
        heading_prev = self._last_heading if self._last_heading is not None else 0.0

        # Guard -> drift coupling (Phase 4). Guards are spatial, not options; they can
        # only bias existing accumulators via an angular kernel (guard->target), never
        # own an accumulator and never smear evidence target->target.
        mu_external = self._guard_drift_term(ids, phi)

        try:
            y = self.accumulator.step(ids, phi, d, s, heading_prev, dt, mu_external=mu_external)
        except ValueError as exc:
            logger.error("%s embodied-DDM accumulator error: %s", self.agent.get_name(), exc)
            if actuate:
                self._zero_commands()
            return

        indices = self.accumulator.last_indices
        self._last_modulated_s = self.accumulator.last_e.copy()

        # Current bearings per seen slot (for readout + latched steering + display).
        phi_full = np.zeros(self.accumulator.N_max, dtype=float)
        if indices.size:
            phi_full[indices] = phi[: indices.size]
        for local_i, slot in enumerate(indices):
            self._slot_bearings[int(slot)] = float(phi[local_i])

        mask = self.accumulator.mask
        if not np.any(mask):
            if actuate:
                self._zero_commands()
            return

        # Purely geometric concentration of the seen bearings: |sum of unit vectors|/K.
        # For K = 2 this is cos(Delta/2). Logged separately from `concentration` so the
        # geometric and evidence-derived values are never conflated across motion modes.
        seen_phi = phi_full[mask]
        r_geom = (
            float(np.hypot(np.sum(np.sin(seen_phi)), np.sum(np.cos(seen_phi))) / seen_phi.size)
            if seen_phi.size else 0.0
        )
        self._last_r_geom = r_geom
        self._bisector_guard_fired = False

        # `midpoint` cuts the decision->motion arm of the embodiment loop: the heading is
        # the equal-weight circular mean of the seen bearings (for K = 2, the angular
        # bisector), independent of the accumulator state. Evidence is still integrated
        # and logged; only the heading ignores it.
        if self.predecision_motion == "midpoint" and self._committed is None:
            readout_weights = np.full(seen_phi.size, 1.0 / max(seen_phi.size, 1), dtype=float)
            readout_threshold = 0.0
        else:
            readout_weights = y[mask]
            readout_threshold = self.y_threshold

        heading, magnitude, concentration = circular_readout(
            readout_weights, seen_phi, threshold=readout_threshold
        )
        if (
            self.predecision_motion == "midpoint"
            and self._committed is None
            and r_geom < self.bisector_eps
        ):
            # Bearings cancel (Delta -> pi): atan2(0, 0) is undefined, so hold.
            self._bisector_guard_fired = True
            heading = self._last_heading if self._last_heading is not None else 0.0
        self._last_heading = heading
        self._last_concentration = float(concentration)
        self._last_magnitude = float(magnitude)

        # Commitment (latch / un-latch) and, if latched, steer toward the winner.
        self._update_commitment(y, mask, indices, concentration)
        if self._committed is not None:
            heading = self._committed_bearing

        if not actuate:
            return

        # ---- Actuation: byte-identical structure to MeanFieldMovementModel.step ----
        if self.reference == "allocentric":
            heading = heading - math.radians(self.agent.orientation.z)
        angle_deg = normalize_angle(math.degrees(heading))

        if self.guard_mode == "actuation":
            angle_deg = self._apply_guard_actuation(angle_deg)

        angle_deg = max(min(angle_deg, self.agent.max_angular_velocity), -self.agent.max_angular_velocity)
        self.agent.angular_velocity_cmd = angle_deg

        if self._committed is None and self.predecision_motion == "stationary":
            # Non-embodied control: no translation AND no rotation, so bearings never
            # change and the feedback loop is fully cut.
            self._zero_commands()
            return
        if self._committed is None and self.predecision_motion == "forward":
            scaling = 1.0
        else:
            scaling = self._compute_scaling(y, mask, magnitude, concentration)
        self.agent.linear_velocity_cmd = self.agent.max_absolute_velocity * scaling

        # Bifurcation detection: identical signature to the ring attractor call site.
        target_angles_for_bif = [float(e.get("angle", 0.0)) for e in entities if "angle" in e]
        self.bifurcation_detector.update(
            tick=tick if tick is not None else 0,
            mf=_AccumulatorBifShim(self.accumulator),
            bump_angle=heading,
            target_angles=target_angles_for_bif,
            target_ids=self.target_ids,
            perception_vec=self.perception,
            agent_angle=0.0,
            agent_x=float(self.agent.position.x),
            agent_y=float(self.agent.position.y),
            agent_orientation=float(self.agent.orientation.z),
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "%s embodied-DDM -> heading=%.2fdeg conc=%.3f mag=%.3f scaling=%.3f committed=%s lin=%.5f",
                self.agent.get_name(), angle_deg, concentration, magnitude, scaling,
                self._committed, self.agent.linear_velocity_cmd,
            )

    # ------------------------------------------------------------------
    def _compute_scaling(self, y, mask, magnitude, concentration) -> float:
        """Map a readout order parameter to forward speed, matching the ring attractor."""
        if self.scaling_mode == "constant":
            # No gating at all: full speed until commitment. Recommended alongside
            # `midpoint` so that mode does exactly one thing.
            return float(np.clip(self.norm_scale, 0.0, 1.0))
        if self.scaling_mode == "magnitude":
            # Deliberately NOT clipped: kept unbounded for backward compatibility with
            # the ring attractor's legacy readout (LCA plan Task 0.4).
            return float(magnitude)
        if self.scaling_mode == "norm":
            l2 = float(np.linalg.norm(y[mask]))
            denom = max(1.0, math.sqrt(float(np.count_nonzero(mask))))
            return float(np.clip(self.norm_scale * l2 / denom, 0.0, 1.0))
        # "concentration" (default): bounded [0, 1].
        return float(np.clip(concentration, 0.0, 1.0))

    def _update_commitment(self, y, mask, indices, concentration) -> None:
        """Latch or release a committed choice per commitment_mode."""
        masked_slots = np.where(mask)[0]
        if masked_slots.size == 0:
            return

        if self.commitment_mode == "threshold":
            if self.accumulator_mode == "ddm2":
                x = self.accumulator.two_choice_difference()
                if abs(x) >= self.Z and self._committed is None:
                    self._committed = 0 if x > 0 else 1
            else:
                in_play = self.accumulator.in_play_mask()
                candidates = np.where(in_play)[0]
                if candidates.size and self._committed is None:
                    best = candidates[int(np.argmax(y[candidates]))]
                    if y[best] >= self.Z:
                        self._committed = int(best)
            if self._committed is not None:
                self._committed_bearing = self._slot_bearings.get(
                    int(self._committed), self._committed_bearing
                )
            return

        # soft commitment (default): two-part criterion — angular coherence AND a
        # dominant option — held for hold_ticks. concentration ~ 1 for collinear targets
        # regardless of coherence, hence the second (probability) criterion.
        w = np.maximum(y[mask], 0.0)
        sw = float(np.sum(w))
        p = float(np.max(w) / (sw + 1e-12)) if w.size else 0.0
        committed_now = (concentration >= self.R_commit) and (p >= self.p_commit)
        self._hold = self._hold + 1 if committed_now else 0

        if self._committed is None and self._hold >= self.hold_ticks:
            winner_local = int(np.argmax(w))
            self._committed = int(masked_slots[winner_local])

        if self._committed is not None:
            if self.allow_reversal and concentration < self.R_release:
                self._committed = None
                self._hold = 0
            else:
                self._committed_bearing = self._slot_bearings.get(
                    int(self._committed), self._committed_bearing
                )

    # ------------------------------------------------------------------
    def _guard_drift_term(self, ids: list[str], phi: np.ndarray) -> Optional[np.ndarray]:
        """Guard->drift coupling: bias each seen accumulator by nearby guards.

        Returns a full-length (N_max,) additive drift term aligned to accumulator slots,
        or None if guards are off / absent. Only guard->target coupling; never
        target->target, so evidence is never smeared between options.
        """
        if self.guard_mode != "drift":
            return None
        _, _, _, guard_angles, guard_qualities, guard_distances = self._convert_perception_to_targets()
        if guard_angles is None or guard_angles.size == 0:
            return None
        # register_targets is idempotent for already-seen ids; step() re-registers the
        # same ids in the same order, so these indices align with `phi`.
        indices = self.accumulator.register_targets(ids)
        weights_g = guard_qualities * np.exp(-self.guard_decay_rate * guard_distances)    # (G,)
        delta = phi[:, None] - guard_angles[None, :]                                       # (K, G)
        kernel = np.exp(self.guard_kappa * (np.cos(delta) - 1.0))                          # (K, G)
        contrib = kernel @ weights_g                                                       # (K,)
        mu_ext = np.zeros(self.accumulator.N_max, dtype=float)
        mu_ext[indices] = -self.beta_g * contrib[: indices.size]
        return mu_ext

    def _apply_guard_actuation(self, angle_deg: float) -> float:
        """Guard repulsion applied at the actuation level (outside the decision model)."""
        _, _, _, guard_angles, guard_qualities, guard_distances = self._convert_perception_to_targets()
        if guard_angles is None or guard_angles.size == 0:
            return angle_deg
        weights_g = guard_qualities * np.exp(-self.guard_decay_rate * guard_distances)
        gx = float(np.sum(weights_g * np.cos(guard_angles)))
        gy = float(np.sum(weights_g * np.sin(guard_angles)))
        if gx == 0.0 and gy == 0.0:
            return angle_deg
        guard_heading = math.degrees(math.atan2(gy, gx))
        strength = float(np.hypot(gx, gy))
        # Signed offset of the desired heading from the guard resultant; push further
        # away from the guard, scaled by strength and beta_g.
        diff = normalize_angle(angle_deg - guard_heading)
        push = self.beta_g * strength * math.copysign(1.0, diff if diff != 0.0 else 1.0)
        return normalize_angle(angle_deg + push)

    # ------------------------------------------------------------------
    # GUI / logging shim (Task 7.2). Emits the mean-field dict shape so the existing GUI
    # panel and mean_field pkl logging render/serialize it with no downstream changes;
    # `decision_model` distinguishes it in saved data.
    # ------------------------------------------------------------------
    def _display_ring(self) -> np.ndarray:
        """Scatter rectified accumulator activity onto a ring for display."""
        ring = np.zeros(self.num_neurons, dtype=float)
        if self.accumulator is None:
            return ring
        y = self.accumulator.y
        mask = self.accumulator.mask
        for slot in np.where(mask)[0]:
            bearing = self._slot_bearings.get(int(slot))
            if bearing is None:
                continue
            val = max(float(y[slot]), 0.0)
            if val <= 0.0:
                continue
            ring += val * np.exp(self._display_kappa * (np.cos(self.group_angles - bearing) - 1.0))
        return ring

    def _build_target_signal_snapshot(self, target_metadata: list[dict]) -> list[dict]:
        """Per-target panel snapshot including accumulator state."""
        snapshot: list[dict] = []
        for idx, entry in enumerate(target_metadata):
            target_id = str(entry.get("id", f"target_{idx}"))
            slot = self.accumulator.slot_of(target_id) if self.accumulator else None
            y_val = float(self.accumulator.y[slot]) if (slot is not None and self.accumulator) else 0.0
            mu_val = float(self.accumulator.last_mu[slot]) if (slot is not None and self.accumulator) else 0.0
            base_quality = float(entry.get("intensity", 0.0))
            snapshot.append(
                {
                    "id": target_id,
                    "label": target_id,
                    "base_quality": base_quality,
                    "modulated_quality": base_quality,
                    "angle": float(entry.get("angle", 0.0)),
                    "distance": float(entry.get("distance", 0.0)),
                    "accumulator": y_val,
                    "drift": mu_val,
                }
            )
        return snapshot

    def get_spin_system_data(self):
        """Return a mean-field-shaped dict so the GUI/logging render the DDM unchanged."""
        if self.accumulator is None:
            return None
        ring = self._display_ring()
        state_matrix = np.clip(ring.reshape(self.num_neurons, 1), 0.0, 1.0)
        num_groups = self.num_neurons
        num_spins_per_group = 1
        angles_flat = np.repeat(self.group_angles, num_spins_per_group)
        perception_vec = np.zeros(num_groups, dtype=float)
        if self.perception is not None:
            flat = np.asarray(self.perception, dtype=float).reshape(-1)
            perception_vec[: min(num_groups, flat.size)] = flat[: min(num_groups, flat.size)]
        entities_copy = copy.deepcopy(self._mf_entities) if self._mf_entities else {"targets": [], "guards": []}
        target_metadata = copy.deepcopy(entities_copy.get("targets", []))
        avg_angle = self._last_heading
        if avg_angle is not None and self.reference == "allocentric":
            avg_angle = math.atan2(math.sin(avg_angle + math.radians(self.agent.orientation.z)),
                                   math.cos(avg_angle + math.radians(self.agent.orientation.z)))
        committed_id = None
        if self._committed is not None and self.accumulator is not None:
            for tid, slot in self.accumulator._slots.items():
                if slot == self._committed:
                    committed_id = tid
                    break
        data = {
            # model == "mean_field" so the existing GUI panel and mean_field pkl logger
            # accept it; decision_model marks that the substrate is the accumulator.
            "model": "mean_field",
            "decision_model": "embodied_ddm",
            "states": state_matrix,
            "angles": (angles_flat, num_groups, num_spins_per_group),
            "external_field": perception_vec,
            "avg_direction_of_activity": avg_angle,
            "mean_field_state": ring.copy(),
            "mean_field_perception": None if self.perception is None else self.perception.copy(),
            "mean_field_perception_raw": None if self.perception is None else self.perception.copy(),
            "mean_field_sensory_map": ring.copy(),
            "mean_field_target_metadata": target_metadata,
            "mean_field_modulated_target_qualities": self._last_modulated_s.copy(),
            "mean_field_target_signals": self._build_target_signal_snapshot(target_metadata),
            "mean_field_sensory_time": float(getattr(self.accumulator, "sensory_time", 0.0)),
            "mean_field_entities": entities_copy,
            "mean_field_norm": float(self._last_concentration),
            "mean_field_beta": float(self.beta_inh),
            "mean_field_lambda1": float(self.accumulator.lambda1),
            "mean_field_omega": None,
            "channel": self._active_perception_channel,
            # Accumulator-specific fields (for analysis of DDM runs).
            "accumulator_state": self.accumulator.y.copy(),
            "accumulator_mask": self.accumulator.mask.copy(),
            "accumulator_committed": self._committed,
            "accumulator_committed_id": committed_id,
            "accumulator_commitment_mode": self.commitment_mode,
            "accumulator_Z": float(self.Z),
            "accumulator_hold": self._hold,
            "accumulator_concentration": float(self._last_concentration),
            "accumulator_magnitude": float(self._last_magnitude),
            # --- motor readout diagnostics (Task M4) ---
            "accumulator_R_geom": float(self._last_r_geom),
            "accumulator_heading_mode": self.predecision_motion,
            "accumulator_scaling_mode": self.scaling_mode,
            "accumulator_bisector_guard_fired": bool(self._bisector_guard_fired),
        }
        # Stamp which sensory protocol produced this record, so no result is ambiguous.
        data.update(self.percept_stream_record())
        return data


class _AccumulatorBifShim:
    """Duck-typed MeanFieldSystem stand-in so `BifurcationDetector` runs unchanged.

    `BifurcationDetector.update` always calls `compute_lambda1(mf)` (for pkl logging),
    which reads `mf.neural_ring/M/b/u/beta/num_neurons/adapt_ring/g_adapt` and returns
    the largest real Jacobian eigenvalue. We construct a K-dimensional stand-in whose
    Jacobian `J = -I + diag(u * sech^2(0)) @ M` equals the linear-LCA Jacobian
    `-(lambda - beta_inh) I - beta_inh * 11^T`, whose top eigenvalue is exactly the
    accumulator's closed-form `lambda1 = beta_inh - lambda_leak` (plan Phase 5). This
    lets both `behavioral` (ignores mf) and `lambda_threshold` (uses J) modes work with
    no edits to bifurcation.py.
    """

    def __init__(self, accumulator: AccumulatorSystem):
        lam = accumulator.lambda_leak
        beta = accumulator.beta_inh
        k = int(np.count_nonzero(accumulator.in_play_mask()))
        k = max(1, k)
        ones = np.ones((k, k), dtype=float)
        # With z=0, b=0, beta=0, u=1: J = -I + M. Choosing M = -((lam-beta-1) I + beta*11^T)
        # gives J = -(lam-beta) I - beta*11^T, top eigenvalue = beta - lam.
        self.M = -((lam - beta - 1.0) * np.eye(k) + beta * ones)
        self.neural_ring = np.zeros(k, dtype=float)
        self.b = np.zeros(k, dtype=float)
        self.u = 1.0
        self.beta = 0.0
        self.num_neurons = k
        self.adapt_ring = np.zeros(k, dtype=float)
        self.g_adapt = 0.0  # accumulator is never in the SFA (Omega) regime
        self._step_count = 0


register_movement_model("embodied_ddm", lambda agent: EmbodiedDDMMovementModel(agent))
