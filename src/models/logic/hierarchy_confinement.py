# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Fabio Oddi
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
#  You may use, modify, and redistribute this file according to the terms of the
#  license. Attribution is required if this code is used in other works.
# ------------------------------------------------------------------------------

import logging
from typing import Optional
from geometry_utils.vector3D import Vector3D
from arena_hierarchy import ArenaHierarchy
from plugin_registry import register_logic_model

logger = logging.getLogger("sim.logic.hierarchy")


class HierarchyConfinementLogic:
    """
    Logic plugin that keeps an agent inside the bounds of its hierarchy node.

    Agents must declare the desired node via the `hierarchy_node` field inside
    their configuration. When that field is missing the model falls back to
    the hierarchy root ("0") so that the behaviour is backwards compatible.
    """

    def __init__(self, agent):
        """Initialize the instance."""
        self.agent = agent
        self.node_id = agent.get_hierarchy_node() or "0"
        self._radius: Optional[float] = None
        self.node_level: Optional[int] = None
        self.agent.set_hierarchy_node(self.node_id)

    def reset(self):
        """Reset the component state."""
        self._radius = None
        self.node_level = None

    def pre_run(self, _objects: dict, _agents: dict) -> None:
        """Pre run."""
        self._ensure_inside(None)

    def step(self, _agent, _tick: int, arena_shape, _objects: dict, _agents: dict) -> None:
        """Execute the simulation step."""
        if self._radius is None:
            self._radius = self._estimate_agent_radius()
        self._ensure_inside(arena_shape)

    def after_movement(self, _agent, _tick: int, arena_shape, _objects: dict, _agents: dict) -> None:
        """After movement."""
        if self._radius is None:
            self._radius = self._estimate_agent_radius()
        hierarchy = self._get_hierarchy(arena_shape)
        if not hierarchy:
            return
        if not self.node_id:
            self.node_id = hierarchy.get_root_id()
            self.agent.set_hierarchy_node(self.node_id)
        if not self.node_id:
            return
        next_position = self.agent.get_position() + self.agent.get_forward_vector()
        clamped_x, clamped_y = hierarchy.clamp_point(self.node_id, next_position.x, next_position.y, padding=self._radius or 0.0)
        if clamped_x == next_position.x and clamped_y == next_position.y:
            self._update_level_info(hierarchy)
            return
        corrected_vector = Vector3D(
            clamped_x - self.agent.get_position().x,
            clamped_y - self.agent.get_position().y,
            self.agent.get_forward_vector().z
        )
        self.agent.forward_vector = corrected_vector
        self._update_level_info(hierarchy)

    def _ensure_inside(self, arena_shape):
        """Ensure inside."""
        hierarchy = self._get_hierarchy(arena_shape)
        if not hierarchy:
            self.agent.set_hierarchy_level(None)
            return
        if not self.node_id:
            self.node_id = hierarchy.get_root_id()
            self.agent.set_hierarchy_node(self.node_id)
        self._update_level_info(hierarchy)
        pos = self.agent.get_position()
        clamped_x, clamped_y = hierarchy.clamp_point(self.node_id, pos.x, pos.y, padding=self._radius or 0.0)
        if clamped_x == pos.x and clamped_y == pos.y:
            return
        corrected = Vector3D(clamped_x, clamped_y, pos.z)
        self.agent.set_position(corrected)
        self.agent.get_shape().translate_attachments(self.agent.orientation.z)
        logger.debug(
            "%s clamped inside hierarchy node %s",
            self.agent.get_name(),
            self.node_id
        )

    def _get_hierarchy(self, arena_shape) -> Optional[ArenaHierarchy]:
        """Return the hierarchy."""
        if arena_shape is None:
            return None
        metadata = getattr(arena_shape, "metadata", None)
        if not metadata:
            return None
        hierarchy = metadata.get("hierarchy")
        if hierarchy is None and self.node_id:
            logger.debug("No hierarchy metadata available for %s", self.agent.get_name())
        return hierarchy

    def _update_level_info(self, hierarchy: Optional[ArenaHierarchy]):
        """Update level info."""
        if not hierarchy or not self.node_id:
            self.node_level = None
            self.agent.set_hierarchy_level(None)
            return
        level = hierarchy.level_of(self.node_id)
        self.node_level = level
        self.agent.set_hierarchy_level(level)

    def _estimate_agent_radius(self) -> float:
        """Estimate the agent radius."""
        shape = self.agent.get_shape()
        center = shape.center
        if not shape.vertices():
            return 0.01
        distances = []
        for vertex in shape.vertices():
            distances.append(max(abs(vertex.x - center.x), abs(vertex.y - center.y)))
        radius = max(distances) if distances else 0.01
        return max(radius, 0.01)


register_logic_model("hierarchy_confinement", lambda agent: HierarchyConfinementLogic(agent))
