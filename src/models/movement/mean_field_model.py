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

from models.bifurcation import BifurcationDetector
from models.mean_field_systems import MeanFieldSystem
from models.bifurcation import BifurcationDetector
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
        self.perception = None
        self._active_perception_channel = "objects"
        self._mf_entities = {"targets": [], "guards": []}
        self._last_bump_angle: Optional[float] = None
        self._last_norm: float = 0.0
        self.mean_field_system: Optional[MeanFieldSystem] = None
        self.detection_model = self._create_detection_model()
<<<<<<< HEAD
        # Bifurcation detection config (D-09: mean_field_model.bifurcation namespace)
=======
        # Bifurcation detection config (mean_field_model.bifurcation namespace)
>>>>>>> 4e59663 (feat(03-01): propagate bifurcation events through agent snapshots (Path A D-06))
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
        self.reset()
        logger.info(
            "%s mean-field model instantiated (neurons=%d, steps_per_tick=%d, sensory_time_mode=%s, sensory_dt=%.6f)",
            self.agent.get_name(),
            self.num_neurons,
            self.steps_per_tick,
            self.sensory_time_mode,
            self.sensory_dt,
        )

    def _resolve_agent_tick_rate(self) -> float:
        """Return the effective agent update rate used by the simulator."""
        if hasattr(self.agent, "ticks"):
            try:
                ticks = float(self.agent.ticks())
                if ticks > 0.0:
                    return ticks
            except (TypeError, ValueError):
                pass
        ticks = getattr(self.agent, "ticks_per_second", 1)
        try:
            ticks = float(ticks)
        except (TypeError, ValueError):
            ticks = 1.0
        return max(1.0, ticks)

    def _resolve_sensory_dt(self) -> float:
        """Resolve how much simulated time the modulation clock advances per internal update."""
        mode = str(self.sensory_time_mode or "world_time").strip().lower()
        if "sensory_dt" in self.params:
            return float(self.params.get("sensory_dt", 0.0))
        if mode in {"integration", "integration_time", "legacy"}:
            return self.integration_time
        return 1.0 / (self._resolve_agent_tick_rate() * self.steps_per_tick)

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
            target_quality_modulations=self.target_quality_modulations,
            sigma=float(self.params.get("sigma", 0.01)),
            dt=self.integration_dt,
            integration_time=self.integration_time,
            sensory_time_mode=self.sensory_time_mode,
            sensory_dt=self.sensory_dt,
            g_adapt=self.g_adapt,
            tau_adapt=self.tau_adapt,
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
            norm = float(np.linalg.norm(neural_field)) if neural_field is not None else final_norm
            self._last_norm = norm
            scaling = np.clip(self.norm_scale * norm / max(1.0, math.sqrt(self.num_neurons)), 0.0, 1.0)
            self.agent.linear_velocity_cmd = self.agent.max_absolute_velocity * scaling #self.agent.max_absolute_velocity   
            self.agent.angular_velocity_cmd = angle_deg
            self._last_bump_angle = angle_rad
<<<<<<< HEAD
            # Bifurcation detection: check after this tick (D-01, D-05)
=======
            # Bifurcation detection: check after this tick
>>>>>>> 4e59663 (feat(03-01): propagate bifurcation events through agent snapshots (Path A D-06))
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
<<<<<<< HEAD
=======
                    agent_angle=0.0,  # egocentric frame: agent heading is always 0
>>>>>>> 4e59663 (feat(03-01): propagate bifurcation events through agent snapshots (Path A D-06))
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

    def _normalize_target_quality_modulations(self, raw_config) -> dict[str, dict[str, float]]:
        """Normalize modulation settings keyed by target ID."""
        if not raw_config:
            return {}
        if not isinstance(raw_config, dict):
            raise ValueError("target_quality_modulations must be a mapping keyed by target ID")

        normalized: dict[str, dict[str, float]] = {}
        for target_id, params in raw_config.items():
            if not isinstance(params, dict):
                raise ValueError(
                    f"target_quality_modulations['{target_id}'] must be a mapping"
                )
            normalized[str(target_id)] = {
                "epsilon": float(params.get("epsilon", 0.0)),
                "omega": float(params.get("omega", 0.0)),
                "psi": float(params.get("psi", 0.0)),
            }
        return normalized

    def _convert_perception_to_targets(
        self,
    ) -> Tuple[list[str], np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Convert detection metadata into target and guard descriptors."""
        meta = self._mf_entities or {"targets": [], "guards": []}
        target_entries = meta.get("targets") or []
        guard_entries = meta.get("guards") or []
        if target_entries:
            target_ids = [str(entry.get("id", "")) for entry in target_entries]
            target_angles = np.array([entry.get("angle", 0.0) for entry in target_entries], dtype=float)
            target_qualities = np.array([entry.get("intensity", 1.0) for entry in target_entries], dtype=float)
        else:
            target_ids = []
            target_angles = np.array([], dtype=float)
            target_qualities = np.array([], dtype=float)
        guard_angles = guard_qualities = guard_distances = None
        if guard_entries:
            guard_angles = np.array([entry.get("angle", 0.0) for entry in guard_entries], dtype=float)
            guard_qualities = np.array([entry.get("intensity", 1.0) for entry in guard_entries], dtype=float)
            guard_distances = np.array([entry.get("distance", 0.0) for entry in guard_entries], dtype=float)
        return target_ids, target_angles, target_qualities, guard_angles, guard_qualities, guard_distances

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
        }
        # Drain new bifurcation events detected this tick (Path A IPC: events flow
        # through per-tick spin data from agent process to Arena).
        new_bif = list(self.bifurcation_detector.events) if hasattr(self, 'bifurcation_detector') else []
        if new_bif:
            self.bifurcation_detector.events.clear()
        data["new_bifurcation_events"] = new_bif
        return data


register_movement_model("mean_field", lambda agent: MeanFieldMovementModel(agent))
