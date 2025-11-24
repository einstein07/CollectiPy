# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Fabio Oddi
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
#  You may use, modify, and redistribute this file according to the terms of the
#  license. Attribution is required if this code is used in other works.
# ------------------------------------------------------------------------------

import logging
import math
from plugin_base import MovementModel
from plugin_registry import register_movement_model
from models.movement.common import apply_motion_state
from models.utils import normalize_angle

logger = logging.getLogger("sim.movement.random_way_point")

class RandomWayPointMovement(MovementModel):
    """Random way point movement."""
    def __init__(self, agent):
        """Initialize the instance."""
        self.agent = agent

    def step(self, agent, tick: int, arena_shape, objects: dict, agents: dict) -> None:
        """Execute the simulation step."""
        self._ensure_goal(arena_shape)
        self._steer_towards_goal()
        apply_motion_state(self.agent)

    def _ensure_goal(self, arena_shape) -> None:
        """Ensure goal."""
        agent = self.agent
        if agent.goal_position is None or self._distance_to_goal(agent) <= 0.001:
            agent.goal_position = self._random_goal(arena_shape)
            logger.debug(
                "%s new waypoint %s",
                agent.get_name(),
                (agent.goal_position.x, agent.goal_position.y, agent.goal_position.z)
            )

    def _steer_towards_goal(self) -> None:
        """Steer towards goal."""
        agent = self.agent
        dx, dy = self._wrapped_vector_to_goal(agent)
        angle_to_goal = math.degrees(math.atan2(-dy, dx))
        angle_to_goal = normalize_angle(angle_to_goal - agent.orientation.z)
        dist_mag = math.hypot(dx, dy)
        if abs(dist_mag) >= agent.prev_goal_distance:
            agent.last_motion_tick += 1
        agent.prev_goal_distance = dist_mag
        if agent.last_motion_tick > agent.ticks_per_second:
            agent.last_motion_tick = 0
            agent.goal_position = None
        if angle_to_goal >= agent.max_angular_velocity:
            agent.motion = agent.LEFT
        elif angle_to_goal <= -agent.max_angular_velocity:
            agent.motion = agent.RIGHT
        else:
            agent.motion = agent.FORWARD
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("%s steering %s (angle_to_goal=%.2f)", agent.get_name(), agent.motion, angle_to_goal)

    def _random_goal(self, arena_shape):
        """Return a new goal position."""
        return self.agent.shape._get_random_point_inside_shape(self.agent.random_generator, arena_shape)

    def _wrapped_vector_to_goal(self, agent):
        """Return the shortest vector towards the goal accounting for wrap."""
        return (
            agent.goal_position.x - agent.position.x,
            agent.goal_position.y - agent.position.y
        )

    def _distance_to_goal(self, agent):
        """Return the wrapped distance to the current goal."""
        if agent.goal_position is None:
            return 0.0
        dx, dy = self._wrapped_vector_to_goal(agent)
        return math.hypot(dx, dy)

register_movement_model("random_way_point", lambda agent: RandomWayPointMovement(agent))
