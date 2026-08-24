# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
#  You may use, modify, and redistribute this file according to the terms of the
#  license. Attribution is required if this code is used in other works.
# ------------------------------------------------------------------------------

"""Shared perception/detection spine for egocentric target-driven movement models.

`TargetModel` holds the perception and detection code that the ring attractor
(`MeanFieldMovementModel`) and the embodied accumulator (`EmbodiedDDMMovementModel`)
must share *bit-for-bit* so that any behavioural difference between the two models is
attributable to the decision substrate and not to how the world was sensed.

The shared boundary between the two models is the output of
`_convert_perception_to_targets` / `_build_target_percept`: from egocentric bearing,
range and quality onward, the ring attractor scatters the percept onto its neurons
while the accumulator maps it to a per-identity `(K,)` vector. Nothing upstream of that
boundary may differ between the models.

`MeanFieldMovementModel` keeps its own `__init__` (unchanged) and simply inherits the
methods here. New models call `_init_target_model` to set up the shared attributes the
methods below depend on.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from models.utils import normalize_angle
from plugin_base import MovementModel
from plugin_registry import get_detection_model

logger = logging.getLogger("sim.target_model")


@dataclass
class TargetPercept:
    """Explicit carrier for the shared perception boundary.

    ids: stable identity strings, length K.
    phi: (K,) egocentric bearing, radians, wrapped to (-pi, pi].
    d:   (K,) range, metres.
    s:   (K,) intensity / quality.
    """

    ids: list[str] = field(default_factory=list)
    phi: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    d: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    s: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

    def __len__(self) -> int:
        return len(self.ids)


class TargetModel(MovementModel):
    """Base class carrying the shared detection/perception code path.

    Subclasses provide their own ``__init__`` and either set the attributes the methods
    below expect directly (as ``MeanFieldMovementModel`` does, unchanged) or call
    ``_init_target_model`` to set up the shared spine (as ``EmbodiedDDMMovementModel``
    does).
    """

    # ------------------------------------------------------------------
    # Shared setup for new models (MeanFieldMovementModel does not use this;
    # it keeps its own byte-identical __init__).
    # ------------------------------------------------------------------
    def _init_target_model(self, agent, params: dict) -> None:
        """Populate the attributes the shared detection/perception methods depend on."""
        self.agent = agent
        self.params = params or {}
        self.reference = self.params.get("reference", "egocentric")
        self.perception_width = float(self.params.get("perception_width", 0.3))
        self.perception_global_inhibition = float(
            self.params.get("perception_global_inhibition", 0.0)
        )
        self.num_neurons = int(self.params.get("num_neurons", 100))
        self.integration_time = float(self.params.get("integration_time", 50.0))
        self.steps_per_tick = max(1, int(self.params.get("steps_per_tick", 1)))
        self.sensory_time_mode = str(self.params.get("sensory_time_mode", "world_time"))
        self.sensory_dt = self._resolve_sensory_dt()
        self.perception_range = self._resolve_detection_range()
        self.group_angles = np.linspace(0, 2 * math.pi, self.num_neurons, endpoint=False)
        self.target_ids = [str(x) for x in self.params.get("target_ids", [])]
        self.guard_ids = [str(x) for x in self.params.get("guard_ids", [])]
        self.target_quality_modulations = self._normalize_target_quality_modulations(
            self.params.get("target_quality_modulations")
        )
        self.alpha = float(self.params.get("alpha", 0.0))
        self.task = (agent.get_task() or self.params.get("task") or "selection").lower()
        if hasattr(agent, "set_task") and not agent.get_task():
            agent.set_task(self.task)
        self.perception = None
        self._active_perception_channel = "objects"
        self._mf_entities = {"targets": [], "guards": []}
        self.detection_model = self._create_detection_model()

    # ------------------------------------------------------------------
    # Moved verbatim from MeanFieldMovementModel (Task 1.1).
    # ------------------------------------------------------------------
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
        """Create detection model matching the shared ring layout."""
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

    def _collect_neighbor_targets(
        self, agents: dict
    ) -> Tuple[list[str], np.ndarray, int]:
        """Return spatial angles and count of all neighbor agents (no range filter)."""
        neighbor_ids: list[str] = []
        neighbor_angles: list[float] = []
        my_name = self.agent.get_name()
        for club, agent_shapes in agents.items():
            for n, shape in enumerate(agent_shapes):
                meta = getattr(shape, "metadata", {}) if hasattr(shape, "metadata") else {}
                entity_name = meta.get("entity_name") or f"{club}_{n}"
                if entity_name == my_name:
                    continue
                pos = shape.center_of_mass()
                dx = pos.x - self.agent.position.x
                dy = pos.y - self.agent.position.y
                angle_deg = math.degrees(math.atan2(-dy, dx))
                if self.reference == "egocentric":
                    angle_deg -= self.agent.orientation.z
                angle_rad = math.radians(normalize_angle(angle_deg))
                neighbor_ids.append(entity_name)
                neighbor_angles.append(angle_rad)
        return neighbor_ids, np.array(neighbor_angles, dtype=float), len(neighbor_ids)

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

    # ------------------------------------------------------------------
    # New (Task 1.1): explicit percept carrier for the accumulator model.
    # ------------------------------------------------------------------
    def _build_target_percept(self) -> TargetPercept:
        """Build a `TargetPercept` from the current `_mf_entities` target list.

        Unlike `_convert_perception_to_targets`, this retains per-target distance so the
        accumulator's distance-modulation stages (`dist_mode`) have `d` available. The
        ring attractor never consumes distance for targets, so the two paths stay in
        parity for the `dist_mode: none` baseline.
        """
        meta = self._mf_entities or {"targets": [], "guards": []}
        entries = meta.get("targets") or []
        ids = [str(entry.get("id", "")) for entry in entries]
        phi = np.array([float(entry.get("angle", 0.0)) for entry in entries], dtype=float)
        d = np.array([float(entry.get("distance", 0.0)) for entry in entries], dtype=float)
        s = np.array([float(entry.get("intensity", 1.0)) for entry in entries], dtype=float)
        if phi.size:
            phi = (phi + np.pi) % (2 * np.pi) - np.pi
        return TargetPercept(ids=ids, phi=phi, d=d, s=s)
