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
from typing import Optional, Tuple

import numpy as np

from models.mean_field_systems import MeanFieldSystem
from models.utils import normalize_angle
from plugin_base import MovementModel
from plugin_registry import (
    get_detection_model,
    register_movement_model,
)

logger = logging.getLogger("sim.mean_field")
logger.setLevel(logging.DEBUG)


class MeanFieldMovementModel(MovementModel):
    """Movement model driven by the MeanFieldSystem."""

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
        self.perception_range = self._resolve_detection_range()
        self.task = (agent.get_task() or self.params.get("task") or "selection").lower()
        if hasattr(agent, "set_task") and not agent.get_task():
            agent.set_task(self.task)
        self.group_angles = np.linspace(0, 2 * math.pi, self.num_neurons, endpoint=False)
        self.target_ids = [str(x) for x in self.params.get("target_ids", [])]
        self.guard_ids = [str(x) for x in self.params.get("guard_ids", [])]
        self.guard_decay_rate = float(self.params.get("guard_decay_rate", self.params.get("spatial_decay", 2.0)))
        self.perception = None
        self._active_perception_channel = "objects"
        self._mf_entities = {"targets": [], "guards": []}
        self._last_bump_angle: Optional[float] = None
        self._last_norm: float = 0.0
        self.mean_field_system: Optional[MeanFieldSystem] = None
        self.detection_model = self._create_detection_model()
        self.reset()
        logger.info(
            "%s mean-field model instantiated (neurons=%d, steps_per_tick=%d)",
            self.agent.get_name(),
            self.num_neurons,
            self.steps_per_tick,
        )

    def _create_detection_model(self):
        """Create detection model matching the mean-field layout."""
        context = {
            "num_groups": self.num_neurons,
            "num_spins_per_group": 1,
            "perception_width": self.perception_width,
            "group_angles": self.group_angles,
            "reference": self.reference,
            "perception_global_inhibition": self.perception_global_inhibition,
            "max_detection_distance": self.perception_range,
            "detection_config": getattr(self.agent, "detection_config", {}),
            "mean_field_target_ids": self.target_ids,
            "mean_field_guard_ids": self.guard_ids,
        }
        detection_name = getattr(self.agent, "detection", None) or self.agent.config_elem.get("detection", "GPS")
        return get_detection_model(detection_name, self.agent, context)

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
            num_targets=int(self.params.get("num_targets", self.num_neurons)),
            num_guards=int(self.params.get("num_guards", 0)),
            target_qualities=self.params.get("target_qualities"),
            guard_qualities=self.params.get("guard_qualities"),
            sigma=float(self.params.get("sigma", 0.01)),
            dt=self.integration_dt,
            integration_time=self.integration_time,
        )
        logger.debug("%s mean-field system reset", self.agent.get_name())

    def pre_run(self, objects: dict, agents: dict) -> None:
        """Let the system settle before the main loop."""
        if self.pre_run_steps <= 0:
            return
        self._update_perception(objects, agents, None, None)
        if self.perception is None:
            return
        for _ in range(self.pre_run_steps):
            targets, qualities, guard_angles, guard_qualities, guard_distances = self._convert_perception_to_targets()
            self.mean_field_system.num_targets = len(targets)
            self.mean_field_system.num_guards = 0 if guard_angles is None else len(guard_angles)
            self.mean_field_system.step(
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
            targets, qualities, guard_angles, guard_qualities, guard_distances = self._convert_perception_to_targets()
            self.mean_field_system.num_targets = len(targets)
            self.mean_field_system.num_guards = len(guard_angles) if guard_angles is not None else 0
            neural_field = None
            bump_positions = None
            final_norm = 0.0
            for _ in range(self.steps_per_tick):
                neural_field, bump_positions, final_norm = self.mean_field_system.step(
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
            norm = float(np.linalg.norm(neural_field)) if neural_field is not None else final_norm
            self._last_norm = norm
            scaling = np.clip(norm / max(1.0, math.sqrt(self.num_neurons)), 0.0, 1.0)
            self.agent.linear_velocity_cmd = self.agent.max_absolute_velocity * scaling
            self.agent.angular_velocity_cmd = angle_deg
            self._last_bump_angle = angle_rad
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "%s mean-field direction updated -> angle=%.2f norm=%.3f scaling=%.3f",
                    self.agent.get_name(),
                    angle_deg,
                    norm,
                    scaling,
                )
        finally:
            if logger.isEnabledFor(logging.DEBUG):
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                logger.debug("----------------------------%s mean-field step duration = %.3f ms-----------------------------------", self.agent.get_name(), elapsed_ms)

    def _update_perception(self, objects: dict, agents: dict, tick: int | None = None, arena_shape=None) -> None:
        """Update sensory perception from detections."""
        if self.detection_model is None:
            self.perception = None
            return
        if tick is not None and hasattr(self.agent, "should_sample_detection"):
            if not self.agent.should_sample_detection(tick):
                return
        snapshot = self.detection_model.sense(self.agent, objects, agents, arena_shape)
        if snapshot is None:
            self.perception = None
            return
        if isinstance(snapshot, dict):
            selected, channel_name = self._select_perception_channel(snapshot)
        else:
            selected, channel_name = snapshot, "raw"
        self.perception = selected
        self._active_perception_channel = channel_name
        self._mf_entities = snapshot.get("mean_field_entities") or {"targets": [], "guards": []}
        if logger.isEnabledFor(logging.DEBUG):
            max_val = float(np.max(self.perception)) if self.perception is not None else 0.0
            logger.debug(
                "%s perception channel=%s max=%.4f",
                self.agent.get_name(),
                channel_name,
                max_val,
            )
        logger.debug("%s mean-field entities=%r", self.agent.get_name(), self._mf_entities)

    def _select_perception_channel(self, snapshot: dict[str, np.ndarray]) -> tuple[np.ndarray, str]:
        """Select a perception channel depending on the configured task."""
        task_name = (self.agent.get_task() or self.task or "selection").lower()
        objects_channel = snapshot.get("objects")
        agents_channel = snapshot.get("agents")
        combined_channel = snapshot.get("combined")
        if task_name in ("selection", "objects"):
            return self._channel_with_fallback(
                (objects_channel, "objects"),
                (combined_channel, "combined"),
                (agents_channel, "agents"),
            )
        if task_name in ("flocking", "agents"):
            return self._channel_with_fallback(
                (agents_channel, "agents"),
                (combined_channel, "combined"),
                (objects_channel, "objects"),
            )
        return self._channel_with_fallback(
            (combined_channel, "combined"),
            (objects_channel, "objects"),
            (agents_channel, "agents"),
        )

    def _channel_with_fallback(
        self,
        primary: tuple[np.ndarray | None, str],
        secondary: tuple[np.ndarray | None, str],
        tertiary: tuple[np.ndarray | None, str],
    ) -> tuple[np.ndarray, str]:
        """Return the first available perception channel."""
        for channel, name in (primary, secondary, tertiary):
            if channel is not None:
                return channel, name
        raise ValueError("Detection model did not provide any perception channels")

    def _resolve_detection_range(self) -> float:
        """Resolve the maximum detection radius from the agent configuration."""
        if hasattr(self.agent, "get_detection_range"):
            try:
                return float(self.agent.get_detection_range())
            except (TypeError, ValueError):
                logger.warning(
                    "%s provided invalid detection range via accessor; falling back to legacy config",
                    self.agent.get_name()
                )
        config_elem = getattr(self.agent, "config_elem", {})
        settings = {}
        if isinstance(config_elem, dict):
            settings = config_elem.get("detection_settings", {}) or {}
        range_candidate = None
        if isinstance(settings, dict):
            range_candidate = settings.get("range", settings.get("distance"))
        if range_candidate is None and isinstance(config_elem, dict):
            range_candidate = config_elem.get("perception_distance")
        if range_candidate is None and hasattr(self.agent, "perception_distance"):
            range_candidate = self.agent.perception_distance
        if range_candidate is None:
            return 0.1
        try:
            value = float(range_candidate)
        except (TypeError, ValueError):
            logger.warning("%s invalid detection range '%s', using default 0.1", self.agent.get_name(), range_candidate)
            return 0.1
        if value <= 0:
            return 0.1
        return value

    def _convert_perception_to_targets(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Convert detection metadata into target and guard descriptors."""
        meta = self._mf_entities or {"targets": [], "guards": []}
        target_entries = meta.get("targets") or []
        guard_entries = meta.get("guards") or []
        if target_entries:
            target_angles = np.array([entry.get("angle", 0.0) for entry in target_entries], dtype=float)
            target_qualities = np.array([entry.get("intensity", 1.0) for entry in target_entries], dtype=float)
        else:
            target_angles = np.array([0.0])
            target_qualities = np.array([0.0])
        guard_angles = guard_qualities = guard_distances = None
        if guard_entries:
            guard_angles = np.array([entry.get("angle", 0.0) for entry in guard_entries], dtype=float)
            guard_qualities = np.array([entry.get("intensity", 1.0) for entry in guard_entries], dtype=float)
            guard_distances = np.array([entry.get("distance", 0.0) for entry in guard_entries], dtype=float)
        return target_angles, target_qualities, guard_angles, guard_qualities, guard_distances

    def get_mean_field_data(self):
        """Return raw state for logging or visualisation."""
        if not self.mean_field_system:
            return None
        z = self.mean_field_system.get_state()
        return {
            "state": z.copy(),
            "perception": None if self.perception is None else self.perception.copy(),
            "channel": self._active_perception_channel,
            "angle": self._last_bump_angle,
        }

    def _normalize_state_for_display(self, values: np.ndarray) -> np.ndarray:
        """Map neural field values to [0, 1] for GUI coloring."""
        matrix = np.asarray(values, dtype=float).reshape(self.num_neurons, 1)
        if matrix.size == 0:
            return matrix
        max_abs = float(np.max(np.abs(matrix))) if matrix.size else 0.0
        if not math.isfinite(max_abs) or max_abs <= 1e-9:
            normalized = np.zeros_like(matrix)
        else:
            normalized = matrix / max_abs
        normalized = np.clip((normalized + 1.0) * 0.5, 0.0, 1.0)
        return normalized

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
        angles_flat = np.repeat(self.group_angles, num_spins_per_group)
        raw_state = snapshot["state"].copy()
        raw_perception = None if snapshot.get("perception") is None else snapshot["perception"].copy()
        entities_copy = copy.deepcopy(self._mf_entities) if self._mf_entities else {"targets": [], "guards": []}
        return {
            "states": state_matrix,
            "angles": (angles_flat, num_groups, num_spins_per_group),
            "external_field": perception_vec,
            "avg_direction_of_activity": snapshot.get("angle"),
            "model": "mean_field",
            "mean_field_state": raw_state,
            "mean_field_perception": raw_perception,
            "mean_field_entities": entities_copy,
            "mean_field_norm": self._last_norm,
            "channel": snapshot.get("channel"),
        }


register_movement_model("mean_field", lambda agent: MeanFieldMovementModel(agent))
