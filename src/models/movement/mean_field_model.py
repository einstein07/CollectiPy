# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
#  You may use, modify, and redistribute this file according to the terms of the
#  license. Attribution is required if this code is used in other works.
# ------------------------------------------------------------------------------

"""Movement model that wraps the phenomenological mean-field ring attractor."""

import copy
import logging
import math
import time
from typing import Optional

import numpy as np

from models.bifurcation import BifurcationDetector
from models.egocentric_target_model import TargetModel
from models.mean_field_systems import MeanFieldSystem
from models.utils import normalize_angle
from plugin_registry import register_movement_model

logger = logging.getLogger("sim.mean_field")
logger.setLevel(logging.DEBUG)


class MeanFieldMovementModel(TargetModel):
    """Movement model driven by the MeanFieldSystem.

    Inherits the shared perception/detection spine from `TargetModel`; the methods that
    used to live here (`_update_perception`, `_convert_perception_to_targets`, ...) are
    now inherited unchanged so the embodied DDM sees the world identically.
    """

    def __init__(self, agent):
        """Initialize the instance."""
        self.agent = agent
        self.params = agent.config_elem.get("mean_field_model", {}) or {}
        self.steps_per_tick = max(1, int(self.params.get("steps_per_tick", 1)))
        self.pre_run_steps = max(0, int(self.params.get("pre_run_steps", 0)))
        self.reference = self.params.get("reference", "egocentric")
        self.perception_width = float(self.params.get("perception_width", 0.3))
        self.perception_global_inhibition = float(self.params.get("perception_global_inhibition", 0.0))
        self.num_neurons = int(self.params.get("num_neurons", 100))
        self.integration_time = float(self.params.get("integration_time", 50.0))
        self.integration_dt = float(self.params.get("integration_dt", self.params.get("dt", 0.1)))
        self.sensory_time_mode = str(self.params.get("sensory_time_mode", "world_time"))
        self.sensory_dt = self._resolve_sensory_dt()
        self.g_adapt = float(self.params.get("g_adapt", 0.0))
        self.tau_adapt = float(self.params.get("tau_adapt", 0.0))
        self.norm_scale = float(self.params.get("norm_scale", 1.0))
        self.perception_range = self._resolve_detection_range()
        self.task = (agent.get_task() or self.params.get("task") or "selection").lower()
        if hasattr(agent, "set_task") and not agent.get_task():
            agent.set_task(self.task)
        self.group_angles = np.linspace(0, 2 * math.pi, self.num_neurons, endpoint=False)
        self.target_ids = [str(x) for x in self.params.get("target_ids", [])]
        self.guard_ids = [str(x) for x in self.params.get("guard_ids", [])]
        self.target_quality_modulations = self._normalize_target_quality_modulations(
            self.params.get("target_quality_modulations")
        )
        self.guard_decay_rate = float(self.params.get("guard_decay_rate", self.params.get("spatial_decay", 2.0)))
        self.alpha = float(self.params.get("alpha", 0.0))
        self.perception = None
        self._active_perception_channel = "objects"
        self._mf_entities = {"targets": [], "guards": []}
        self._last_bump_angle: Optional[float] = None
        self._last_norm: float = 0.0
        self.mean_field_system: Optional[MeanFieldSystem] = None
        self.detection_model = self._create_detection_model()
        # Bifurcation detection config (mean_field_model.bifurcation namespace)
        bif_cfg = self.params.get("bifurcation", {})
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
        self.use_thresholding = bool(self.params.get("use_thresholding", True))
        # Which readout order parameter drives forward speed (Task 0.4). Default is the
        # bounded "concentration"; "magnitude"/"norm" reproduce the legacy behaviour of
        # use_thresholding True/False respectively.
        self.scaling_mode = str(self.params.get("scaling_mode", "concentration"))
        self.reset()
        logger.info(
            "%s mean-field model instantiated (neurons=%d, steps_per_tick=%d, sensory_time_mode=%s, sensory_dt=%.6f)",
            self.agent.get_name(),
            self.num_neurons,
            self.steps_per_tick,
            self.sensory_time_mode,
            self.sensory_dt,
        )

    def reset(self) -> None:
        """Reset the mean-field state."""
        self.perception = None
        self._last_bump_angle = None
        self.mean_field_system = MeanFieldSystem(
            num_neurons=self.num_neurons,
            u=float(self.params.get("u", 6.0)),
            beta=float(self.params.get("beta", 1.0)),
            v=float(self.params.get("v", 0.5)),
            kappa=float(self.params.get("kappa", 20.0)),
            spatial_decay=float(self.params.get("spatial_decay", 2.0)),
            num_targets=int(self.params.get("num_targets", 0)),
            num_guards=int(self.params.get("num_guards", 0)),
            target_qualities=self.params.get("target_qualities"),
            guard_qualities=self.params.get("guard_qualities"),
            target_quality_modulations=self.target_quality_modulations,
            sigma=float(self.params.get("sigma", 0.01)),
            sigma_s=float(self.params.get("sigma_s", 0.0)),
            dt=self.integration_dt,
            integration_time=self.integration_time,
            sensory_time_mode=self.sensory_time_mode,
            sensory_dt=self.sensory_dt,
            g_adapt=self.g_adapt,
            tau_adapt=self.tau_adapt,
            g_threshold=float(self.params.get("g_threshold", 0.6)),
            use_thresholding=bool(self.params.get("use_thresholding", True)),
            scaling_mode=self.scaling_mode,
        )
        if hasattr(self, 'bifurcation_detector'):
            self.bifurcation_detector.reset()
        logger.debug("%s mean-field system reset", self.agent.get_name())

    def pre_run(self, objects: dict, agents: dict) -> None:
        """Let the system settle before the main loop."""
        if self.pre_run_steps <= 0:
            return
        self._update_perception(objects, agents, None, None)
        if self.perception is None:
            return
        for _ in range(self.pre_run_steps):
            target_ids, targets, qualities, guard_angles, guard_qualities, guard_distances = self._convert_perception_to_targets()
            self.mean_field_system.num_targets = len(targets)
            self.mean_field_system.num_guards = 0 if guard_angles is None else len(guard_angles)
            self.mean_field_system.step(
                target_ids=target_ids,
                target_angles=targets,
                target_qualities=qualities,
                guard_angles=guard_angles,
                guard_qualities=guard_qualities,
                guard_decay_rate=self.guard_decay_rate,
                guard_distances=guard_distances,
            )
        logger.debug("%s mean-field pre-run completed (%d steps)", self.agent.get_name(), self.pre_run_steps)

    def step(self, agent, tick: int, arena_shape, objects: dict, agents: dict) -> None:
        """Execute a simulation step."""
        start_time = time.perf_counter()
        if self.mean_field_system is None:
            self.reset()
            if self.mean_field_system is None:
                return
        logger.debug("--------------------%s mean-field step tick=%s-------------------------", self.agent.get_name(), tick)
        try:
            self._update_perception(objects, agents, tick, arena_shape)
            if self.perception is None or not np.any(self.perception):
                self.agent.linear_velocity_cmd = 0.0
                self.agent.angular_velocity_cmd = 0.0
                self._last_bump_angle = None
                self._last_norm = 0.0
                return
            target_ids, targets, qualities, guard_angles, guard_qualities, guard_distances = self._convert_perception_to_targets()
            if self.alpha > 0.0 and qualities.size > 0:
                neighbor_ids, neighbor_angles, n_neighbors = self._collect_neighbor_targets(agents)
                if n_neighbors > 0:
                    neighbor_strength = (float(np.max(qualities)) * self.alpha) / n_neighbors
                    target_ids = target_ids + neighbor_ids
                    targets = np.concatenate([targets, neighbor_angles])
                    qualities = np.concatenate([qualities, np.full(n_neighbors, neighbor_strength)])
                    for nid, nangle in zip(neighbor_ids, neighbor_angles):
                        self._mf_entities["targets"].append({
                            "id": nid,
                            "angle": float(nangle),
                            "distance": 0.0,
                            "intensity": neighbor_strength,
                        })
            self.mean_field_system.num_targets = len(targets)
            self.mean_field_system.num_guards = len(guard_angles) if guard_angles is not None else 0
            neural_field = None
            bump_positions = None
            final_norm = 0.0
            for _ in range(self.steps_per_tick):
                neural_field, bump_positions, final_norm = self.mean_field_system.step(
                    target_ids=target_ids,
                    target_angles=targets,
                    target_qualities=qualities,
                    guard_angles=guard_angles,
                    guard_qualities=guard_qualities,
                    guard_decay_rate=self.guard_decay_rate,
                    guard_distances=guard_distances,
                )
            angle_rad = None
            if bump_positions is not None and len(bump_positions) > 0:
                angle_rad = bump_positions[-1]
            if angle_rad is None:
                self.agent.linear_velocity_cmd = 0.0
                self.agent.angular_velocity_cmd = 0.0
                self._last_bump_angle = None
                self._last_norm = 0.0
                return
            if self.reference == "allocentric":
                angle_rad = angle_rad - math.radians(self.agent.orientation.z)
            angle_deg = normalize_angle(math.degrees(angle_rad))
            angle_deg = max(min(angle_deg, self.agent.max_angular_velocity), -self.agent.max_angular_velocity)
            # Forward speed is gated by the readout order parameter chosen by
            # scaling_mode (Task 0.4). "concentration" (default) is bounded in [0, 1] so
            # an undecided agent physically slows down; "norm" reproduces the legacy
            # use_thresholding=False clip; "magnitude" reproduces the legacy
            # use_thresholding=True (unbounded) behaviour.
            mf = self.mean_field_system
            if self.scaling_mode == "norm":
                norm = float(np.linalg.norm(neural_field)) if neural_field is not None else float(mf.last_l2)
                scaling = float(np.clip(self.norm_scale * norm / max(1.0, math.sqrt(self.num_neurons)), 0.0, 1.0))
            elif self.scaling_mode == "magnitude":
                norm = float(mf.last_magnitude)
                scaling = norm
            else:  # "concentration"
                norm = float(mf.last_concentration)
                scaling = float(np.clip(norm, 0.0, 1.0))
            self._last_norm = norm
            self.agent.linear_velocity_cmd = self.agent.max_absolute_velocity * scaling
            logger.debug("%s mean-field raw command -> angle=%.2f norm=%.3f scaling=%.3f", self.agent.get_name(), angle_deg, norm, scaling)
            self.agent.angular_velocity_cmd = angle_deg
            self._last_bump_angle = angle_rad
            # Bifurcation detection: check after this tick
            if self.mean_field_system is not None:
                target_angles_for_bif = []
                for t in self._mf_entities.get("targets", []):
                    if "angle" in t:
                        target_angles_for_bif.append(float(t["angle"]))
                self.bifurcation_detector.update(
                    tick=tick,
                    mf=self.mean_field_system,
                    bump_angle=angle_rad,
                    target_angles=target_angles_for_bif,
                    target_ids=self.target_ids,
                    perception_vec=self.perception,
                    agent_angle=0.0,  # egocentric frame: agent heading is always 0
                    agent_x=float(self.agent.position.x),
                    agent_y=float(self.agent.position.y),
                    agent_orientation=float(self.agent.orientation.z),
                )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "%s mean-field direction updated -> angle=%.2f norm=%.3f scaling=%.3f linear_vel_cmd=%.5f",
                    self.agent.get_name(),
                    angle_deg,
                    norm,
                    scaling,
                    self.agent.linear_velocity_cmd
                )
        finally:
            if logger.isEnabledFor(logging.DEBUG):
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                logger.debug("----------------------------%s mean-field step duration = %.3f ms-----------------------------------", self.agent.get_name(), elapsed_ms)

    def _build_target_signal_snapshot(
        self,
        target_metadata: list[dict],
        modulated_target_qualities: np.ndarray,
    ) -> list[dict]:
        """Return a GUI-friendly per-target input snapshot."""
        flattened = np.asarray(modulated_target_qualities, dtype=float).reshape(-1)
        snapshot: list[dict] = []
        for idx, entry in enumerate(target_metadata):
            target_id = str(entry.get("id", f"target_{idx}"))
            base_quality = float(entry.get("intensity", 0.0))
            modulated_quality = base_quality
            if idx < flattened.size:
                modulated_quality = float(flattened[idx])
            snapshot.append(
                {
                    "id": target_id,
                    "label": target_id,
                    "base_quality": base_quality,
                    "modulated_quality": modulated_quality,
                    "angle": float(entry.get("angle", 0.0)),
                    "distance": float(entry.get("distance", 0.0)),
                }
            )
        return snapshot

    def get_mean_field_data(self):
        """Return raw state for logging or visualisation."""
        if not self.mean_field_system:
            return None
        z = self.mean_field_system.get_state()
        target_metadata = copy.deepcopy((self._mf_entities or {}).get("targets", []))
        modulated_target_qualities = self.mean_field_system.get_modulated_target_qualities()
        sensory_time = float(getattr(self.mean_field_system, "sensory_time", 0.0))
        sensory_increment = float(getattr(self.mean_field_system, "sensory_dt", self.sensory_dt))
        last_sensory_time = max(0.0, sensory_time - sensory_increment)
        return {
            "state": z.copy(),
            "perception_raw": None if self.perception is None else self.perception.copy(),
            "sensory_map": self.mean_field_system.get_sensory_map(),
            "target_metadata": target_metadata,
            "modulated_target_qualities": modulated_target_qualities,
            "target_signals": self._build_target_signal_snapshot(
                target_metadata,
                modulated_target_qualities,
            ),
            "sensory_time": last_sensory_time,
            "channel": self._active_perception_channel,
            "angle": self._last_bump_angle,
        }

    def _normalize_state_for_display(self, values: np.ndarray) -> np.ndarray:
        """Map neural field values to [0, 1] for GUI coloring using absolute scale."""
        matrix = np.asarray(values, dtype=float).reshape(self.num_neurons, 1)
        if matrix.size == 0:
            return matrix
        return np.clip((matrix + 1.0) * 0.5, 0.0, 1.0)

    def _prepare_perception_vector(self, length: int) -> np.ndarray:
        """Flatten perception into a vector consumed by the GUI plot."""
        vector = np.zeros(length, dtype=float)
        if self.perception is not None:
            flat = np.asarray(self.perception, dtype=float).reshape(-1)
            count = min(length, flat.size)
            if count > 0:
                vector[:count] = flat[:count]
        return vector

    def get_spin_system_data(self):
        """
        Expose mean-field neural activity using the same structure as the spin model.

        The GUI expects:
            (state_matrix, (angles, num_groups, num_spins_per_group), perception_vector, avg_angle)
        """
        if not self.mean_field_system:
            return None
        snapshot = self.get_mean_field_data()
        if not snapshot:
            return None
        state_matrix = self._normalize_state_for_display(snapshot["state"])
        num_groups = self.num_neurons
        num_spins_per_group = 1
        perception_vec = self._prepare_perception_vector(num_groups * num_spins_per_group)
        theta = getattr(self.mean_field_system, "theta", None)
        use_theta = theta is not None and len(theta) == num_groups
        angles_source = theta if use_theta else self.group_angles
        angles_flat = np.repeat(angles_source, num_spins_per_group)
        if use_theta:
            shift = num_groups // 2
            perception_vec = np.roll(perception_vec, -shift)
        raw_state = snapshot["state"].copy()
        raw_perception = None if snapshot.get("perception_raw") is None else snapshot["perception_raw"].copy()
        raw_sensory_map = None if snapshot.get("sensory_map") is None else snapshot["sensory_map"].copy()
        entities_copy = copy.deepcopy(self._mf_entities) if self._mf_entities else {"targets": [], "guards": []}
        target_metadata = copy.deepcopy(snapshot.get("target_metadata") or [])
        modulated_target_qualities = (
            None
            if snapshot.get("modulated_target_qualities") is None
            else snapshot["modulated_target_qualities"].copy()
        )
        target_signals = copy.deepcopy(snapshot.get("target_signals") or [])
        avg_angle = snapshot.get("angle")
        if avg_angle is not None and self.reference == "allocentric":
            avg_angle = avg_angle + math.radians(self.agent.orientation.z)
            avg_angle = math.atan2(math.sin(avg_angle), math.cos(avg_angle))
        data = {
            "states": state_matrix,
            "angles": (angles_flat, num_groups, num_spins_per_group),
            "external_field": perception_vec,
            "avg_direction_of_activity": avg_angle,
            "model": "mean_field",
            "mean_field_state": raw_state,
            "mean_field_perception": raw_perception,
            "mean_field_perception_raw": raw_perception,
            "mean_field_sensory_map": raw_sensory_map,
            "mean_field_target_metadata": target_metadata,
            "mean_field_modulated_target_qualities": modulated_target_qualities,
            "mean_field_target_signals": target_signals,
            "mean_field_sensory_time": float(snapshot.get("sensory_time", 0.0)),
            "mean_field_entities": entities_copy,
            "mean_field_norm": self._last_norm,
            "mean_field_beta": float(self.mean_field_system.beta),
            "mean_field_lambda1": (
                self.bifurcation_detector.last_lambda1
                if hasattr(self, "bifurcation_detector")
                else None
            ),
            "mean_field_omega": (
                self.bifurcation_detector.last_omega
                if hasattr(self, "bifurcation_detector") and self.g_adapt > 0.0
                else None
            ),
            "channel": snapshot.get("channel"),
        }
        # Drain new bifurcation events detected this tick (Path A IPC: events flow
        # through per-tick spin data from agent process to Arena).
        new_bif = list(self.bifurcation_detector.events) if hasattr(self, 'bifurcation_detector') else []
        if new_bif:
            self.bifurcation_detector.events.clear()
        data["new_bifurcation_events"] = new_bif
        return data


register_movement_model("mean_field", lambda agent: MeanFieldMovementModel(agent))
