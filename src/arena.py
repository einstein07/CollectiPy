# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Fabio Oddi
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
#  You may use, modify, and redistribute this file according to the terms of the
#  license. Attribution is required if this code is used in other works.
# ------------------------------------------------------------------------------

import logging, time, math, random
from typing import Optional, Any
import multiprocessing as mp
from config import Config
from random import Random
from bodies.shapes3D import Shape3DFactory
from entity import EntityFactory
from geometry_utils.vector3D import Vector3D
from dataHandling import DataHandlingFactory
from arena_hierarchy import ArenaHierarchy, Bounds2D

class ArenaFactory():

    """Arena factory."""
    @staticmethod
    def create_arena(config_elem:Config):
        """Create arena."""
        if config_elem.arena.get("_id") in ("abstract", "none", None):
            return AbstractArena(config_elem)
        elif config_elem.arena.get("_id") == "circle":
            return CircularArena(config_elem)
        elif config_elem.arena.get("_id") == "rectangle":
            return RectangularArena(config_elem)
        elif config_elem.arena.get("_id") == "square":
            return SquareArena(config_elem)
        elif config_elem.arena.get("_id") == "unbounded":
            return UnboundedArena(config_elem)
        else:
            raise ValueError(f"Invalid shape type: {config_elem.arena['_id']} valid types are: none, abstract, circle, rectangle, square, unbounded")

class Arena():
    
    """Arena."""
    def __init__(self, config_elem:Config):
        """Initialize the instance."""
        self.random_generator = Random()
        self._seed_random = random.SystemRandom()
        self.ticks_per_second = int(config_elem.environment.get("ticks_per_second", 3))
        configured_seed = config_elem.arena.get("random_seed")
        if configured_seed is None:
            configured_seed = 0
        self._configured_seed = int(configured_seed)
        self.random_seed = self._configured_seed
        self._id = "none" if config_elem.arena.get("_id") == "abstract" else config_elem.arena.get("_id","none") 
        self.objects = {object_type: (config_elem.environment.get("objects",{}).get(object_type),[]) for object_type in config_elem.environment.get("objects",{}).keys()}
        self.agents_shapes = {}
        self.agents_spins = {}
        self.agents_metadata = {}
        self.data_handling = None
        results_enabled = bool(getattr(config_elem, "results", {}))
        if results_enabled:
            self.data_handling = DataHandlingFactory.create_data_handling(config_elem)
        self._hierarchy = None
        self._hierarchy_enabled = "hierarchy" in config_elem.arena
        self._hierarchy_config = config_elem.arena.get("hierarchy") if self._hierarchy_enabled else None
        gui_cfg = config_elem.gui if hasattr(config_elem, "gui") else {}
        throttle_cfg = gui_cfg.get("throttle", {})
        if isinstance(throttle_cfg, (int, float)):
            throttle_cfg = {"max_backlog": throttle_cfg}
        raw_threshold = throttle_cfg.get("max_backlog", gui_cfg.get("max_backlog", 6))
        try:
            threshold = int(raw_threshold)
        except (TypeError, ValueError):
            threshold = 6
        self._gui_backpressure_threshold = max(0, threshold)
        raw_interval = throttle_cfg.get("poll_interval_ms", gui_cfg.get("poll_interval_ms", 8))
        try:
            interval_ms = float(raw_interval)
        except (TypeError, ValueError):
            interval_ms = 8.0
        enabled_flag = throttle_cfg.get("enabled")
        if enabled_flag is None:
            enabled_flag = gui_cfg.get("adaptive_throttle", True)
        self._gui_backpressure_enabled = bool(enabled_flag) if enabled_flag is not None else True
        self._gui_backpressure_interval = max(0.001, interval_ms / 1000.0)
        self._gui_backpressure_active = False
        self.quiet = getattr(config_elem, "quiet", False) if config_elem else False
        self.termination_config = self._normalize_termination_config(
            config_elem.environment.get("termination", {}) if config_elem else {}
        )
        self._target_position_swap_specs = self._normalize_target_position_swaps(
            config_elem.environment.get("target_position_swaps", []) if config_elem else []
        )
        self._target_position_swap_events = []
        self._target_position_swap_index = 0
        # Post-bifurcation swap config (Phase 3)
        post_bif_cfg = config_elem.environment.get("post_bifurcation_swap") if config_elem else None
        self._post_bif_swap_cfg = self._normalize_post_bif_swap_config(post_bif_cfg)
        self._post_bif_swap_triggered = False
        self._post_bif_swap_event = None  # scheduled swap event dict or None
        self._latest_queue_bif_events: list = []  # cumulative events from EntityManager queue

    @staticmethod
    def _blocking_get(q, timeout: float = 0.01, sleep_s: float = 0.001):
        """Get from a queue/Pipe with tiny sleep to avoid busy-wait."""
        while True:
            if hasattr(q, "poll"):
                try:
                    if q.poll(timeout):
                        return q.get()
                except EOFError:
                    return None
            else:
                try:
                    return q.get(timeout=timeout)
                except EOFError:
                    return None
                except Exception:
                    pass
            time.sleep(sleep_s)

    @staticmethod
    def _maybe_get(q, timeout: float = 0.0):
        """Non-blocking get with optional timeout."""
        if hasattr(q, "poll"):
            try:
                if q.poll(timeout):
                    return q.get()
            except EOFError:
                return None
            return None
        try:
            return q.get(timeout=timeout)
        except EOFError:
            return None
        except Exception:
            return None

    def get_id(self):
        """Return the id."""
        return self._id
    
    def get_seed(self):
        """Return the seed."""
        return self.random_seed
    
    def get_random_generator(self):
        """Return the random generator."""
        return self.random_generator

    def increment_seed(self):
        """Increment seed."""
        self.random_seed += 1
        
    def reset_seed(self):
        """Reset the seed to a deterministic starting point."""
        base_seed = self._configured_seed if self._configured_seed is not None else 0
        if base_seed < 0:
            base_seed = 0
        self.random_seed = base_seed
        
    def randomize_seed(self):
        """Assign a random seed (used when GUI reset is requested)."""
        self.random_seed = self._seed_random.randrange(0, 2**32)
        
    def set_random_seed(self):
        """Set the random seed."""
        if self.random_seed > -1:
            self.random_generator.seed(self.random_seed)
        else:
            self.random_seed = self._seed_random.randrange(0, 2**32)
            self.random_generator.seed(self.random_seed)

    def initialize(self):
        """Initialize the component state."""
        self.reset()
        for key,(config,entities) in self.objects.items():
            for n in range(config["number"]):
                entities.append(EntityFactory.create_entity(entity_type="object_"+key,config_elem=config,_id=n))
                
    def run(self,num_runs,time_limit, arena_queue:mp.Queue, agents_queue:mp.Queue, gui_in_queue:mp.Queue, dec_arena_in:mp.Queue, gui_control_queue:mp.Queue, render:bool=False):
        """Run the simulation routine."""
        pass

    def reset(self):
        """Reset the component state."""
        self.set_random_seed()

    def _collect_bifurcation_events(self):
        """Transfer bifurcation events from all agents' detectors to DataHandling."""
        if self.data_handling is None:
            return
        for _config, entities in self.objects.values():
            for entity in entities:
                plugin = getattr(entity, '_movement_plugin', None)
                if plugin is None:
                    continue
                detector = getattr(plugin, 'bifurcation_detector', None)
                if detector is not None and detector.events:
                    self.data_handling.collect_bifurcation_events(detector.events)
                    detector.events.clear()  # prevent double-collection on repeated calls

    def close(self):
        """Close the component resources."""
        self._collect_bifurcation_events()
        # Flush queue-received bifurcation events before writing events.json (IPC path).
        if self.data_handling is not None and self._latest_queue_bif_events:
            self.data_handling.collect_bifurcation_events(self._latest_queue_bif_events)
            self._latest_queue_bif_events = []
        for (config,entities) in self.objects.values():
            for n in range(len(entities)):
                entities[n].close()
        if self.data_handling is not None: self.data_handling.close(self.agents_shapes)

    def get_wrap_config(self):
        """Optional metadata describing wrap-around projection (default: None)."""
        return None

    def get_hierarchy(self):
        """Return the hierarchy."""
        return self._hierarchy

    def _create_hierarchy(self, bounds: Optional[Bounds2D]):
        """Create hierarchy."""
        if not self._hierarchy_enabled or self._hierarchy_config is None:
            return None
        cfg = self._hierarchy_config or {}
        depth = int(cfg.get("depth", 0))
        branches = int(cfg.get("branches", 1))
        try:
            return ArenaHierarchy(bounds, depth=depth, branches=branches)
        except ValueError as exc:
            raise ValueError(f"Invalid hierarchy configuration: {exc}") from exc

    @staticmethod
    def _normalize_termination_config(cfg: Any) -> dict:
        """Normalize termination configuration."""
        if not isinstance(cfg, dict) or not cfg:
            return {}
        normalized = dict(cfg)
        term_type = str(normalized.get("type", "")).strip().lower()
        if not term_type:
            return {}
        normalized["type"] = term_type
        try:
            normalized["radius"] = float(normalized.get("radius", 0.0))
        except (TypeError, ValueError):
            normalized["radius"] = 0.0
        target_ids = normalized.get("target_ids", [])
        if isinstance(target_ids, (str, int, float)):
            target_ids = [target_ids]
        if isinstance(target_ids, (list, tuple, set)):
            normalized["target_ids"] = [str(x) for x in target_ids if str(x)]
        else:
            normalized["target_ids"] = []
        agent_ids = normalized.get("agent_ids", "any")
        if isinstance(agent_ids, str):
            agent_key = agent_ids.strip().lower()
            if agent_key in ("any", "all", "*"):
                normalized["agent_ids"] = None
            else:
                normalized["agent_ids"] = [agent_ids]
        elif isinstance(agent_ids, (list, tuple, set)):
            normalized["agent_ids"] = [str(x) for x in agent_ids if str(x)]
        else:
            normalized["agent_ids"] = None
        normalized["mode"] = str(normalized.get("mode", "any") or "any").strip().lower()
        return normalized

    @staticmethod
    def _normalize_target_position_swaps(cfg: Any) -> list[dict]:
        """Normalize target position swap events configured at environment scope."""
        if not cfg:
            return []
        if not isinstance(cfg, list):
            raise ValueError("environment.target_position_swaps must be a list")
        normalized = []
        for idx, entry in enumerate(cfg):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"target_position_swaps[{idx}] must be a mapping"
                )
            tick_value = entry.get("at_tick")
            seconds_value = None
            for key in ("at_seconds", "at_second", "at_s", "time_seconds", "time_s"):
                if key in entry:
                    seconds_value = entry.get(key)
                    break
            if tick_value is None and seconds_value is None:
                raise ValueError(
                    f"target_position_swaps[{idx}] must define at_tick or at_seconds"
                )
            parsed_tick = None
            if tick_value is not None:
                try:
                    parsed_tick = int(tick_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"target_position_swaps[{idx}].at_tick must be an integer"
                    ) from exc
                if parsed_tick < 0:
                    raise ValueError(
                        f"target_position_swaps[{idx}].at_tick must be >= 0"
                    )
            parsed_seconds = None
            if seconds_value is not None:
                try:
                    parsed_seconds = float(seconds_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"target_position_swaps[{idx}].at_seconds must be a float"
                    ) from exc
                if parsed_seconds < 0.0:
                    raise ValueError(
                        f"target_position_swaps[{idx}].at_seconds must be >= 0"
                    )

            pairs_cfg = entry.get("pairs", entry.get("swaps"))
            if pairs_cfg is None:
                pairs_cfg = entry.get("pair")
            if pairs_cfg is None and any(
                key in entry
                for key in ("target_a_id", "target_b_id", "target_a", "target_b", "from", "to")
            ):
                pairs_cfg = [entry]
            if pairs_cfg is None:
                raise ValueError(
                    f"target_position_swaps[{idx}] must define 'pairs'"
                )
            if not isinstance(pairs_cfg, (list, tuple)):
                pairs_cfg = [pairs_cfg]

            parsed_pairs = []
            for pair_idx, pair_cfg in enumerate(pairs_cfg):
                if isinstance(pair_cfg, (list, tuple)):
                    if len(pair_cfg) != 2:
                        raise ValueError(
                            f"target_position_swaps[{idx}].pairs[{pair_idx}] must contain exactly two IDs"
                        )
                    left_raw, right_raw = pair_cfg[0], pair_cfg[1]
                elif isinstance(pair_cfg, dict):
                    left_raw = pair_cfg.get(
                        "a",
                        pair_cfg.get(
                            "target_a",
                            pair_cfg.get(
                                "target_a_id",
                                pair_cfg.get("from", pair_cfg.get("left"))
                            ),
                        ),
                    )
                    right_raw = pair_cfg.get(
                        "b",
                        pair_cfg.get(
                            "target_b",
                            pair_cfg.get(
                                "target_b_id",
                                pair_cfg.get("to", pair_cfg.get("right"))
                            ),
                        ),
                    )
                else:
                    raise ValueError(
                        f"target_position_swaps[{idx}].pairs[{pair_idx}] must be a pair list or mapping"
                    )
                left_id = str(left_raw).strip() if left_raw is not None else ""
                right_id = str(right_raw).strip() if right_raw is not None else ""
                if not left_id or not right_id:
                    raise ValueError(
                        f"target_position_swaps[{idx}].pairs[{pair_idx}] contains empty target IDs"
                    )
                if left_id == right_id:
                    raise ValueError(
                        f"target_position_swaps[{idx}].pairs[{pair_idx}] must contain two distinct IDs"
                    )
                parsed_pairs.append((left_id, right_id))

            normalized.append(
                {
                    "tick": parsed_tick,
                    "seconds": parsed_seconds,
                    "pairs": parsed_pairs,
                    "label": str(entry.get("label", f"swap_{idx}")),
                }
            )
        return normalized

    @staticmethod
    def _normalize_post_bif_swap_config(cfg) -> dict | None:
        """Validate and normalise the post_bifurcation_swap environment config.

        Returns None if cfg is absent/empty (backward compatible — no swap).
        Raises ValueError for malformed input.
        """
        if not cfg:
            return None
        if not isinstance(cfg, dict):
            raise ValueError("environment.post_bifurcation_swap must be a mapping")
        pairs_cfg = cfg.get("pairs")
        if pairs_cfg is None:
            raise ValueError("environment.post_bifurcation_swap must define 'pairs'")
        if not isinstance(pairs_cfg, (list, tuple)):
            raise ValueError("environment.post_bifurcation_swap.pairs must be a list")
        parsed_pairs = []
        for pair_idx, pair in enumerate(pairs_cfg):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(
                    f"environment.post_bifurcation_swap.pairs[{pair_idx}] must be a 2-element list"
                )
            left_id = str(pair[0]).strip() if pair[0] is not None else ""
            right_id = str(pair[1]).strip() if pair[1] is not None else ""
            if not left_id or not right_id:
                raise ValueError(
                    f"environment.post_bifurcation_swap.pairs[{pair_idx}] contains empty target IDs"
                )
            if left_id == right_id:
                raise ValueError(
                    f"environment.post_bifurcation_swap.pairs[{pair_idx}] must contain two distinct IDs"
                )
            parsed_pairs.append((left_id, right_id))
        delay_raw = cfg.get("delay_ticks", 0)
        try:
            delay_ticks = int(delay_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "environment.post_bifurcation_swap.delay_ticks must be a non-negative integer"
            ) from exc
        if delay_ticks < 0:
            raise ValueError(
                "environment.post_bifurcation_swap.delay_ticks must be >= 0"
            )
        return {"pairs": parsed_pairs, "delay_ticks": delay_ticks}

    def _collect_bifurcation_events(self):
        """Transfer bifurcation events from all agents' detectors to DataHandling."""
        if self.data_handling is None:
            return
        for _config, entities in self.objects.values():
            for entity in entities:
                plugin = getattr(entity, '_movement_plugin', None)
                if plugin is None:
                    continue
                detector = getattr(plugin, 'bifurcation_detector', None)
                if detector is not None and detector.events:
                    self.data_handling.collect_bifurcation_events(detector.events)
                    detector.events.clear()  # prevent double-collection

    def _find_first_bifurcation_in_snapshots(self, agent_snapshots: list) -> dict | None:
        """Find the first (earliest-tick) bifurcation event across all agent snapshots."""
        earliest = None
        for snap in agent_snapshots:
            if not snap:
                continue
            for _grp, spin_list in snap.get("agents_spins", {}).items():
                if not isinstance(spin_list, list):
                    continue
                for spin_data in spin_list:
                    if not isinstance(spin_data, dict):
                        continue
                    for ev in spin_data.get("new_bifurcation_events", []):
                        if earliest is None or ev.get("tick", float("inf")) < earliest.get("tick", float("inf")):
                            earliest = ev
        return earliest

    @staticmethod
    def _rescue_bif_events_from_snapshot(snap: dict | None, dest: list) -> None:
        """Extract new_bifurcation_events from a snapshot into dest before it is discarded."""
        if not snap:
            return
        for spin_list in snap.get("agents_spins", {}).values():
            if isinstance(spin_list, list):
                for sd in spin_list:
                    if isinstance(sd, dict):
                        dest.extend(sd.get("new_bifurcation_events", []))

    def _check_post_bif_swap(self, tick: int, agent_snapshots: list,
                              rescued_bif_events: list | None = None) -> None:
        """Check for first bifurcation event and schedule/execute post-bifurcation swap.

        ``rescued_bif_events`` carries events that were extracted from snapshots
        replaced during the tick's IPC drain loop — they would otherwise be lost
        because the Arena always keeps only the *latest* snapshot per manager.
        """
        if self._post_bif_swap_cfg is None:
            return
        # Phase 1: If swap not yet triggered, scan for first bifurcation event
        if not self._post_bif_swap_triggered:
            bif_event = self._find_first_bifurcation_in_snapshots(agent_snapshots)
            # Also check rescued events (from snapshots that were overwritten mid-tick)
            if bif_event is None and rescued_bif_events:
                for ev in rescued_bif_events:
                    if isinstance(ev, dict):
                        if bif_event is None or ev.get("tick", float("inf")) < bif_event.get("tick", float("inf")):
                            bif_event = ev
            if bif_event is not None:
                self._post_bif_swap_triggered = True
                swap_tick = bif_event["tick"] + self._post_bif_swap_cfg["delay_ticks"]
                self._post_bif_swap_event = {
                    "tick": swap_tick,
                    "pairs": list(self._post_bif_swap_cfg["pairs"]),
                    "triggered_by_agent": bif_event.get("agent", "unknown"),
                    "bifurcation_tick": bif_event["tick"],
                }
                logging.info(
                    "Post-bifurcation swap scheduled at tick %d (bif_tick=%d + delay=%d, agent=%s)",
                    swap_tick, bif_event["tick"], self._post_bif_swap_cfg["delay_ticks"],
                    bif_event.get("agent", "unknown"),
                )
                # Forward all bifurcation events to DataHandling for events.json
                if self.data_handling is not None:
                    all_bif_events = list(rescued_bif_events) if rescued_bif_events else []
                    for snap in agent_snapshots:
                        if not snap:
                            continue
                        for _grp, spin_list in snap.get("agents_spins", {}).items():
                            if not isinstance(spin_list, list):
                                continue
                            for spin_data in spin_list:
                                if not isinstance(spin_data, dict):
                                    continue
                                evts = spin_data.get("new_bifurcation_events", [])
                                if evts:
                                    all_bif_events.extend(evts)
                    if all_bif_events:
                        self.data_handling.collect_bifurcation_events(all_bif_events)
        # Phase 2: If swap is scheduled and due, execute it
        if self._post_bif_swap_event is not None and tick >= self._post_bif_swap_event["tick"]:
            self._execute_post_bif_swap(tick)

    def _execute_post_bif_swap(self, tick: int) -> None:
        """Execute the scheduled post-bifurcation swap."""
        event = self._post_bif_swap_event
        if event is None:
            return
        objects_by_name = self._index_objects_by_name()
        for left_id, right_id in event.get("pairs", []):
            left_obj = objects_by_name.get(left_id)
            right_obj = objects_by_name.get(right_id)
            if left_obj is None or right_obj is None:
                logging.warning(
                    "Post-bif swap: missing target '%s' or '%s' at tick %s",
                    left_id, right_id, tick,
                )
                continue
            self._swap_object_xy_positions(left_obj, right_obj)
            logging.info(
                "Post-bifurcation swap executed at tick %s: %s <-> %s (position)",
                tick, left_id, right_id,
            )
        if self.data_handling is not None:
            self.data_handling.collect_swap_events([event])
        self._post_bif_swap_event = None

    def _prepare_target_position_swaps_for_run(self) -> None:
        """Resolve configured swap events to tick indices for the current run."""
        events = []
        for spec in self._target_position_swap_specs:
            tick = spec["tick"]
            if tick is None:
                seconds = 0.0 if spec["seconds"] is None else float(spec["seconds"])
                tick = int(math.ceil(seconds * self.ticks_per_second))
            tick = max(0, int(tick))
            events.append(
                {
                    "tick": tick,
                    "pairs": list(spec["pairs"]),
                    "label": spec["label"],
                }
            )
        events.sort(key=lambda item: item["tick"])
        self._target_position_swap_events = events
        self._target_position_swap_index = 0
        if events:
            logging.info(
                "Loaded %d target position swap events for this run",
                len(events),
            )

    def _index_objects_by_name(self) -> dict:
        """Return object entities keyed by runtime entity name."""
        indexed = {}
        for _, entities in self.objects.values():
            for entity in entities:
                name = entity.get_name() if hasattr(entity, "get_name") else None
                if name:
                    indexed[str(name)] = entity
        return indexed

    @staticmethod
    def _swap_object_xy_positions(first, second) -> None:
        """Swap x/y positions of two objects while preserving each own z."""
        pos_first = first.get_position()
        pos_second = second.get_position()
        first.set_position(Vector3D(pos_second.x, pos_second.y, pos_first.z))
        second.set_position(Vector3D(pos_first.x, pos_first.y, pos_second.z))

    def _apply_target_position_swap_event(self, event: dict, objects_by_name: dict, tick: int) -> None:
        """Apply a single configured target position swap event."""
        for left_id, right_id in event.get("pairs", []):
            left_obj = objects_by_name.get(left_id)
            right_obj = objects_by_name.get(right_id)
            if left_obj is None or right_obj is None:
                logging.warning(
                    "Skipping target swap '%s' at tick %s: missing target(s) '%s' or '%s'",
                    event.get("label", ""),
                    tick,
                    left_id,
                    right_id,
                )
                continue
            self._swap_object_xy_positions(left_obj, right_obj)
            logging.info(
                "Target position swap '%s' executed at tick %s: %s <-> %s",
                event.get("label", ""),
                tick,
                left_id,
                right_id,
            )

    def _apply_due_target_position_swaps(self, tick: int) -> None:
        """Apply all target position swap events scheduled up to `tick`."""
        if not self._target_position_swap_events:
            return
        objects_by_name = None
        while self._target_position_swap_index < len(self._target_position_swap_events):
            event = self._target_position_swap_events[self._target_position_swap_index]
            if int(event.get("tick", 0)) > tick:
                break
            if objects_by_name is None:
                objects_by_name = self._index_objects_by_name()
            self._apply_target_position_swap_event(event, objects_by_name, tick)
            self._target_position_swap_index += 1

    def _should_terminate_run(self, agents_shapes: dict, objects_data: dict) -> bool:
        """Return True if termination conditions are satisfied."""
        cfg = self.termination_config or {}
        if cfg.get("type") != "proximity":
            return False
        radius = float(cfg.get("radius", 0.0) or 0.0)
        if radius <= 0:
            return False
        target_ids = cfg.get("target_ids") or []
        if not target_ids:
            return False
        agent_filter = cfg.get("agent_ids")
        agent_positions = self._collect_agent_positions(agents_shapes, agent_filter)
        if not agent_positions:
            return False
        target_positions = self._collect_target_positions(objects_data, agents_shapes, target_ids)
        if not target_positions:
            return False
        for _, agent_pos in agent_positions.items():
            if agent_pos is None:
                continue
            for target_id in target_ids:
                target_pos = target_positions.get(target_id)
                if target_pos is None:
                    continue
                dx = float(target_pos.x - agent_pos.x)
                dy = float(target_pos.y - agent_pos.y)
                if math.hypot(dx, dy) <= radius:
                    return True
        return False

    def _collect_agent_positions(self, agents_shapes: dict, agent_filter) -> dict:
        """Collect agent positions keyed by entity name."""
        positions = {}
        if not isinstance(agents_shapes, dict):
            return positions
        allowed = None
        if isinstance(agent_filter, list):
            allowed = set(agent_filter)
        for shapes in agents_shapes.values():
            for shape in shapes:
                meta = getattr(shape, "metadata", None) if shape is not None else None
                entity_id = meta.get("entity_name") if isinstance(meta, dict) else None
                if not entity_id:
                    entity_id = meta.get("name") if isinstance(meta, dict) else None
                if not entity_id:
                    continue
                if allowed is not None and entity_id not in allowed:
                    continue
                try:
                    pos = shape.center_of_mass()
                except Exception:
                    pos = None
                positions[entity_id] = pos
        return positions

    def _collect_target_positions(self, objects_data: dict, agents_shapes: dict, target_ids: list) -> dict:
        """Collect target positions keyed by entity name."""
        targets = {}
        wanted = set(target_ids)
        # objects
        if isinstance(objects_data, dict):
            for _, payload in objects_data.items():
                if not payload or len(payload) < 2:
                    continue
                shapes, positions = payload[0], payload[1]
                for shape, position in zip(shapes, positions):
                    meta = getattr(shape, "metadata", None) if shape is not None else None
                    entity_id = meta.get("entity_name") if isinstance(meta, dict) else None
                    if not entity_id:
                        entity_id = meta.get("name") if isinstance(meta, dict) else None
                    if not entity_id or entity_id not in wanted:
                        continue
                    targets[entity_id] = position
        # agents can also be targets
        if isinstance(agents_shapes, dict):
            for shapes in agents_shapes.values():
                for shape in shapes:
                    meta = getattr(shape, "metadata", None) if shape is not None else None
                    entity_id = meta.get("entity_name") if isinstance(meta, dict) else None
                    if not entity_id:
                        entity_id = meta.get("name") if isinstance(meta, dict) else None
                    if not entity_id or entity_id not in wanted:
                        continue
                    try:
                        pos = shape.center_of_mass()
                    except Exception:
                        pos = None
                    targets[entity_id] = pos
        return targets


class AbstractArena(Arena):
    
    """Abstract arena."""
    def __init__(self, config_elem:Config):
        """Initialize the instance."""
        super().__init__(config_elem)
        logging.info("Abstract arena created successfully")
        self._hierarchy = self._create_hierarchy(None)
    
    def get_shape(self):
        """Return the shape."""
        pass
    
    def close(self):
        """Close the component resources."""
        super().close()

PLACEMENT_MAX_ATTEMPTS = 200
PLACEMENT_MARGIN_FACTOR = 1.3
PLACEMENT_MARGIN_EPS = 0.002


class SolidArena(Arena):
    
    """Solid arena."""
    def __init__(self, config_elem:Config):
        """Initialize the instance."""
        super().__init__(config_elem)
        self._grid_origin = None
        self._grid_cell_size = None
        self.shape = self._build_arena_shape(config_elem)
        self._update_hierarchy_from_shape()

    def _build_arena_shape(self, config_elem:Config):
        """Return the collision shape based on the arena configuration."""
        shape_type = self._arena_shape_type()
        shape_cfg = self._arena_shape_config(config_elem)
        return Shape3DFactory.create_shape("arena", shape_type, shape_cfg)

    def _arena_shape_type(self):
        """Return the default shape id used for the arena."""
        return self._id

    def _arena_shape_config(self, config_elem:Config):
        """Return the arena configuration passed to the shape factory."""
        return {key:val for key,val in config_elem.arena.items()}

    def get_shape(self):
        """Return the shape."""
        return self.shape
    
    def initialize(self):
        """Initialize the component state."""
        super().initialize()
        min_v = self.shape.min_vert()
        max_v = self.shape.max_vert()
        rng = self.random_generator
        self._grid_origin = Vector3D(min_v.x, min_v.y, 0)
        radii_map, max_radius = self._compute_entity_radii()
        self._grid_cell_size = max(max_radius * 2.0, 0.05)
        occupancy = {}
        for (config, entities) in self.objects.values():
            n_entities = len(entities)
            for entity in entities:
                entity.set_position(Vector3D(999, 0, 0), False)
            for n in range(n_entities):
                entity = entities[n]
                if not entity.get_orientation_from_dict():
                    rand_angle = Random.uniform(rng, 0.0, 360.0)
                    entity.set_start_orientation(Vector3D(0, 0, rand_angle))
                position = entity.get_start_position()
                if not entity.get_position_from_dict():
                    placed = self._place_entity_random(
                        entity,
                        radii_map[id(entity)],
                        occupancy,
                        rng,
                        min_v,
                        max_v
                    )
                    if not placed:
                        raise Exception(f"Impossible to place object {entity.entity()} in the arena")
                else:
                    entity.to_origin()
                    target = Vector3D(position.x, position.y, position.z + abs(entity.get_shape().min_vert().z))
                    entity.set_start_position(target)
                    shape = entity.get_shape()
                    if shape.check_overlap(self.shape)[0]:
                        logging.warning(
                            "Configured position for object %s overlaps arena walls; re-sampling position.",
                            entity.entity()
                        )
                        placed = self._place_entity_random(
                            entity,
                            radii_map[id(entity)],
                            occupancy,
                            rng,
                            min_v,
                            max_v
                        )
                        if not placed:
                            raise Exception(f"Impossible to place object {entity.entity()} in the arena")
                    else:
                        self._register_shape_in_grid(shape, target, radii_map[id(entity)], occupancy)

    def pack_objects_data(self) -> dict:
        """Pack objects data."""
        out = {}
        for _,entities in self.objects.values():
            shapes = []
            positions = []
            strengths = []
            uncertainties = []
            for n in range(len(entities)):
                shapes.append(entities[n].get_shape())
                positions.append(entities[n].get_position())
                strengths.append(entities[n].get_strength())
                uncertainties.append(entities[n].get_uncertainty())
            out.update({entities[0].entity():(shapes,positions,strengths,uncertainties)})
        return out

    def _compute_entity_radii(self):
        """Compute entity radii."""
        radii = {}
        max_radius = 0.0
        for (_, entities) in self.objects.values():
            for entity in entities:
                entity.to_origin()
                shape = entity.get_shape()
                radius = self._estimate_shape_radius(shape)
                radii[id(entity)] = radius
                max_radius = max(max_radius, radius)
        return radii, max_radius if max_radius > 0 else 0.1

    def _estimate_shape_radius(self, shape):
        """Estimate the shape radius."""
        radius_getter = getattr(shape, "get_radius", None)
        if callable(radius_getter):
            try:
                r = float(radius_getter())
                if r > 0:
                    return r
            except Exception:
                pass
        if shape.vertices_list:
            center = shape.center_of_mass()
            return max((Vector3D(v.x - center.x, v.y - center.y, v.z - center.z).magnitude() for v in shape.vertices_list), default=0.05)
        return 0.05

    def _place_entity_random(self, entity, radius, occupancy, rng, min_v, max_v):
        """Place entity random."""
        attempts = 0
        shape_n = entity.get_shape()
        min_vert_z = abs(shape_n.min_vert().z)
        effective_radius = radius * PLACEMENT_MARGIN_FACTOR + PLACEMENT_MARGIN_EPS
        min_x = min_v.x + effective_radius
        max_x = max_v.x - effective_radius
        min_y = min_v.y + effective_radius
        max_y = max_v.y - effective_radius
        if min_x >= max_x or min_y >= max_y:
            return False
        while attempts < PLACEMENT_MAX_ATTEMPTS:
            rand_pos = Vector3D(
                Random.uniform(rng, min_x, max_x),
                Random.uniform(rng, min_y, max_y),
                min_vert_z
            )
            entity.to_origin()
            entity.set_position(rand_pos)
            shape = entity.get_shape()
            if shape.check_overlap(self.shape)[0]:
                attempts += 1
                continue
            if self._shape_overlaps_grid(shape, rand_pos, radius, occupancy):
                attempts += 1
                continue
            entity.set_start_position(rand_pos)
            self._register_shape_in_grid(shape, rand_pos, radius, occupancy)
            return True
        return False

    def _shape_overlaps_grid(self, shape, position, radius, occupancy):
        """Shape overlaps grid."""
        if not occupancy:
            return False
        cells = self._cells_for_shape(position, radius, pad=1)
        checked = set()
        for cell in cells:
            if cell in checked:
                continue
            checked.add(cell)
            for other_shape, other_radius in occupancy.get(cell, []):
                center_delta = Vector3D(
                    shape.center.x - other_shape.center.x,
                    shape.center.y - other_shape.center.y,
                    0
                )
                if center_delta.magnitude() >= (radius + other_radius):
                    continue
                if shape.check_overlap(other_shape)[0]:
                    return True
        return False

    def _register_shape_in_grid(self, shape, position, radius, occupancy):
        """Register shape in grid."""
        cells = self._cells_for_shape(position, radius)
        for cell in cells:
            occupancy.setdefault(cell, []).append((shape, radius))

    def _cells_for_shape(self, position, radius, pad: int = 0):
        """Cells for shape."""
        if self._grid_cell_size is None or self._grid_cell_size <= 0:
            return [(0, 0)]
        origin = self._grid_origin or Vector3D()
        cell_size = self._grid_cell_size
        min_x = int(math.floor((position.x - radius - origin.x) / cell_size)) - pad
        max_x = int(math.floor((position.x + radius - origin.x) / cell_size)) + pad
        min_y = int(math.floor((position.y - radius - origin.y) / cell_size)) - pad
        max_y = int(math.floor((position.y + radius - origin.y) / cell_size)) + pad
        cells = []
        for cx in range(min_x, max_x + 1):
            for cy in range(min_y, max_y + 1):
                cells.append((cx, cy))
        return cells
    
    def pack_detector_data(self) -> dict:
        """Pack detector data."""
        out = {}
        for _,entities in self.objects.values():
            shapes = []
            positions = []
            for n in range(len(entities)):
                shapes.append(entities[n].get_shape())
                positions.append(entities[n].get_position())
            out.update({entities[0].entity():(shapes,positions)})
        return out
    
    def _apply_gui_backpressure(self, gui_in_queue: mp.Queue):
        """Pause the simulation loop when the GUI cannot keep up with rendering."""
        if not self._gui_backpressure_enabled or gui_in_queue is None:
            return
        threshold = self._gui_backpressure_threshold
        if threshold <= 0:
            return
        try:
            backlog = gui_in_queue.qsize()
        except (NotImplementedError, AttributeError, OSError):
            return
        if backlog < threshold:
            self._gui_backpressure_active = False
            return
        if not self._gui_backpressure_active:
            logging.warning("GUI rendering is %s frames behind; slowing down ticks", backlog)
            self._gui_backpressure_active = True
        while True:
            try:
                backlog = gui_in_queue.qsize()
            except (NotImplementedError, AttributeError, OSError):
                break
            if backlog < threshold:
                break
            time.sleep(self._gui_backpressure_interval)
        self._gui_backpressure_active = False
        
    def run(self,num_runs,time_limit, arena_queue: Any, agents_queue: Any, gui_in_queue: Any,dec_arena_in: Any, gui_control_queue: Any,render:bool=False):
        """Function to run the arena in a separate process (supports multiple agent queues)."""
        arena_queues = arena_queue if isinstance(arena_queue, list) else [arena_queue]
        agents_queues = agents_queue if isinstance(agents_queue, list) else [agents_queue]
        n_managers = len(agents_queues)

        def _combine_agent_snapshots(snapshots, cached_shapes, cached_spins, cached_metadata):
            """
            Merge per-manager snapshots; when the same group key appears in multiple
            managers (e.g., split of the same agent type), combine them for the
            current tick. Shapes/spins are rebuilt every merge to avoid
            duplicating entries across ticks; metadata is preserved from cache
            unless a snapshot provides an update.
            """
            shapes: dict = {}
            spins: dict = {}
            metadata = {k: list(v) for k, v in cached_metadata.items()}
            for snap in snapshots:
                if not snap:
                    continue
                for grp, vals in snap.get("agents_shapes", {}).items():
                    shapes.setdefault(grp, []).extend(vals)
                for grp, vals in snap.get("agents_spins", {}).items():
                    spins.setdefault(grp, []).extend(vals)
                for grp, vals in snap.get("agents_metadata", {}).items():
                    metadata[grp] = list(vals)
            # If no metadata arrived in this batch, keep cached metadata.
            return shapes, spins, metadata

        ticks_limit = time_limit*self.ticks_per_second + 1 if time_limit > 0 else 0
        run = 1
        while run < num_runs + 1:
            logging.info(f"Run number {run} started")
            self._prepare_target_position_swaps_for_run()
            # Reset post-bifurcation swap state for this run (D-04: once per run)
            self._post_bif_swap_triggered = False
            self._post_bif_swap_event = None
            self._apply_due_target_position_swaps(0)
            arena_data = {
                "status": [0,self.ticks_per_second],
                "objects": self.pack_objects_data()
            }
            if render:
                gui_in_queue.put({**arena_data, "agents_shapes": self.agents_shapes, "agents_spins": self.agents_spins, "agents_metadata": self.agents_metadata})
                self._apply_gui_backpressure(gui_in_queue)
            for q in arena_queues:
                q.put({**arena_data, "random_seed": self.random_seed})

            latest_agent_data = [None] * n_managers
            for idx, q in enumerate(agents_queues):
                latest_agent_data[idx] = self._maybe_get(q, timeout=1.0)
            if any(d is None for d in latest_agent_data):
                break
            self.agents_shapes, self.agents_spins, self.agents_metadata = _combine_agent_snapshots(
                latest_agent_data,
                self.agents_shapes,
                self.agents_spins,
                self.agents_metadata
            )
            initial_tick_rate = latest_agent_data[0].get("status", [0, self.ticks_per_second])[1]
            if self.data_handling is not None:
                self.data_handling.new_run(
                    run,
                    self.agents_shapes,
                    self.agents_spins,
                    self.agents_metadata,
                    initial_tick_rate
                )
            t = 1
            running = False if render else True
            step_mode = False
            reset = False
            last_snapshot_info = None
            termination_triggered = False
            while True:
                if ticks_limit > 0 and t >= ticks_limit: break
                if render:
                    cmd = self._maybe_get(gui_control_queue, timeout=0.0)
                    while cmd is not None:
                        if cmd == "start":
                            running = True
                        elif cmd == "stop":
                            running = False
                        elif cmd == "step":
                            running = False
                            step_mode = True
                        elif cmd == "reset":
                            running = False
                            reset = True
                        cmd = self._maybe_get(gui_control_queue, timeout=0.0)
                self._apply_due_target_position_swaps(t)
                arena_data = {
                    "status": [t,self.ticks_per_second],
                    "objects": self.pack_objects_data()
                }
                if running or step_mode:
                    if not render and not getattr(self, "quiet", False):
                        print(f"\rarena_ticks {t}", end='', flush=True)
                    for q in arena_queues:
                        q.put(arena_data)
                    ready = [False] * n_managers
                    # Accumulate bifurcation events from EVERY snapshot read this tick.
                    # The Arena keeps only the *latest* snapshot per manager for position
                    # data, so any snapshot that gets overwritten would silently drop its
                    # new_bifurcation_events.  We rescue them here and forward them to
                    # _check_post_bif_swap so the swap trigger is never missed.
                    _rescued_bif_events: list = []
                    while not all(ready):
                        for idx, q in enumerate(agents_queues):
                            candidate = self._maybe_get(q, timeout=0.01)
                            if candidate is not None:
                                # Rescue events from the snapshot we are about to discard
                                self._rescue_bif_events_from_snapshot(latest_agent_data[idx], _rescued_bif_events)
                                latest_agent_data[idx] = candidate
                        for idx, snap in enumerate(latest_agent_data):
                            ready[idx] = bool(snap and snap.get("run_complete") and snap["status"][0]/snap["status"][1] >= t/self.ticks_per_second)
                        detector_data = {
                            "objects": self.pack_detector_data()
                        }
                        if all(q.qsize()==0 for q in arena_queues):
                            for q in arena_queues:
                                q.put(arena_data)
                            if dec_arena_in is not None:
                                dec_arena_in.put(detector_data)
                        time.sleep(0.001)

                    for idx, q in enumerate(agents_queues):
                        latest = self._maybe_get(q, timeout=0.0)
                        if latest is not None:
                            # Rescue events before final overwrite too
                            self._rescue_bif_events_from_snapshot(latest_agent_data[idx], _rescued_bif_events)
                            latest_agent_data[idx] = latest
                    self.agents_shapes, self.agents_spins, self.agents_metadata = _combine_agent_snapshots(
                        latest_agent_data,
                        self.agents_shapes,
                        self.agents_spins,
                        self.agents_metadata
                    )
                    self._check_post_bif_swap(t, latest_agent_data, rescued_bif_events=_rescued_bif_events)
                    if self.data_handling is not None:
                        tick_stamp = arena_data.get("status", [t, self.ticks_per_second])[0]
                        tick_rate = arena_data.get("status", [tick_stamp, self.ticks_per_second])[1]
                        self.data_handling.save(
                            self.agents_shapes,
                            self.agents_spins,
                            self.agents_metadata,
                            tick_stamp,
                            tick_rate
                        )
                        last_snapshot_info = (tick_stamp, tick_rate)
                    if render:
                        gui_in_queue.put({**arena_data, "agents_shapes": self.agents_shapes, "agents_spins": self.agents_spins, "agents_metadata": self.agents_metadata})
                        self._apply_gui_backpressure(gui_in_queue)
                    if self._should_terminate_run(self.agents_shapes, arena_data.get("objects")):
                        termination_triggered = True
                        terminate_cmd = {"command": "terminate_run", "status": True, "reason": "proximity"}
                        for q in arena_queues:
                            q.put(terminate_cmd)
                        logging.info("Termination triggered (proximity) at tick %s", t)
                        break
                    step_mode = False
                    t += 1
                elif reset:
                    break
                else: time.sleep(0.0005)
            if self.data_handling is not None and last_snapshot_info:
                self.data_handling.save(
                    self.agents_shapes,
                    self.agents_spins,
                    self.agents_metadata,
                    last_snapshot_info[0],
                    last_snapshot_info[1],
                    force=True
                )
            if t < ticks_limit and not reset and not termination_triggered: break
            if run < num_runs:
                if not reset:
                    run += 1
                    self.increment_seed()
                else:
                    self.randomize_seed()
                self.reset()
                if reset:
                    arena_data = {
                                "status": "reset",
                                "objects": self.pack_objects_data()
                    }
                    for q in arena_queues:
                        q.put(arena_data)
                if not render: print("")
            elif not reset:
                run += 1
                self.close()
                if not render: print("")
            else:
                self.randomize_seed()
                self.reset()
                arena_data = {
                            "status": "reset",
                            "objects": self.pack_objects_data()
                }
                for q in arena_queues:
                    q.put(arena_data)
        
    def reset(self):
        """Reset the component state."""
        super().reset()
        min_v = self.shape.min_vert()
        max_v = self.shape.max_vert()
        rng = self.random_generator
        # Flush bifurcation events and write events.json before closing the run
        self._collect_bifurcation_events()
        if self.data_handling is not None and self._latest_queue_bif_events:
            self.data_handling.collect_bifurcation_events(self._latest_queue_bif_events)
            self._latest_queue_bif_events = []
        if self.data_handling is not None: self.data_handling.close(self.agents_shapes)
        # Reset post-bifurcation swap state for next run
        self._post_bif_swap_triggered = False
        self._post_bif_swap_event = None
        for (config, entities) in self.objects.values():
            n_entities = len(entities)
            for entity in entities:
                entity.set_position(Vector3D(999, 0, 0), False)
            for n in range(n_entities):
                entity = entities[n]
                entity.set_start_orientation(entity.get_start_orientation())
                if not entity.get_orientation_from_dict():
                    rand_angle = Random.uniform(rng, 0.0, 360.0)
                    entity.set_start_orientation(Vector3D(0, 0, rand_angle))
                position = entity.get_start_position()
                if not entity.get_position_from_dict():
                    count = 0
                    done = False
                    shape_n = entity.get_shape()
                    shape_type_n = entity.get_shape_type()
                    while not done and count < 500:
                        done = True
                        rand_pos = Vector3D(
                            Random.uniform(rng, min_v.x, max_v.x),
                            Random.uniform(rng, min_v.y, max_v.y),
                            position.z
                        )
                        entity.to_origin()
                        entity.set_position(rand_pos)
                        shape_n = entity.get_shape()
                        if shape_n.check_overlap(self.shape)[0]:
                            done = False
                        if done:
                            for m in range(n_entities):
                                if m == n:
                                    continue
                                other_entity = entities[m]
                                other_shape = other_entity.get_shape()
                                other_shape_type = other_entity.get_shape_type()
                                if shape_type_n == other_shape_type and shape_n.check_overlap(other_shape)[0]:
                                    done = False
                                    break
                        count += 1
                        if done:
                            entity.set_start_position(rand_pos)
                    if not done:
                        raise Exception(f"Impossible to place object {entity.entity()} in the arena")
                else:
                    entity.to_origin()
                    entity.set_start_position(Vector3D(position.x, position.y, position.z + abs(entity.get_shape().min_vert().z)))

    def close(self):
        """Close the component resources."""
        super().close()

    def _update_hierarchy_from_shape(self):
        """Update hierarchy from shape."""
        bounds = None
        if hasattr(self, "shape") and self.shape is not None:
            min_v = self.shape.min_vert()
            max_v = self.shape.max_vert()
            bounds = Bounds2D(min_v.x, min_v.y, max_v.x, max_v.y)
        self._hierarchy = self._create_hierarchy(bounds)
        if hasattr(self, "shape") and self.shape is not None and hasattr(self.shape, "metadata"):
            self.shape.metadata["hierarchy"] = self._hierarchy
            if self._hierarchy:
                self.shape.metadata["hierarchy_colors"] = getattr(self._hierarchy, "level_colors", {})
                self.shape.metadata["hierarchy_node_numbers"] = {
                    node_id: node.order for node_id, node in self._hierarchy.nodes.items()
                }

class UnboundedArena(SolidArena):
    
    """Unbounded arena rendered as a large square without wrap-around."""
    def __init__(self, config_elem:Config):
        """Initialize the instance."""
        raw = config_elem.arena.get("diameter", None)
        try:
            self.diameter = float(raw) if raw is not None else None
        except (TypeError, ValueError):
            self.diameter = None
        if not self.diameter or self.diameter <= 0:
            self.diameter = self._estimate_initial_diameter(config_elem)
        if self.diameter <= 0:
            raise ValueError("UnboundedArena could not derive a positive initial diameter")
        super().__init__(config_elem)
        logging.info(
            "Unbounded arena created (diameter=%.3f, square side=%.3f)",
            self.diameter,
            self.diameter
        )

    def _estimate_initial_diameter(self, config_elem: Config) -> float:
        """
        Heuristic initial span when the user does not provide a diameter.
        Uses agent count/size to choose a finite square for spawning/rendering.
        """
        agents_cfg = config_elem.environment.get("agents", {}) if hasattr(config_elem, "environment") else {}
        total = 0
        max_diam = 0.05
        for cfg in agents_cfg.values():
            if not isinstance(cfg, dict):
                continue
            num = cfg.get("number", 0)
            if isinstance(num, (list, tuple)) and num:
                try:
                    num = int(num[0])
                except Exception:
                    num = 0
            try:
                num_int = int(num)
            except Exception:
                num_int = 0
            total += max(num_int, 0)
            try:
                diam = float(cfg.get("diameter", max_diam))
                if diam > max_diam:
                    max_diam = diam
            except Exception:
                pass
        if total <= 0:
            return 2.0
        agent_radius = max_diam * 0.5
        # Spread agents on a disk with comfortable spacing; convert to square side.
        import math
        disk_radius = max(agent_radius * 4.0, agent_radius * math.sqrt(total) * 2.5, 0.5)
        return max(disk_radius * 2.0, 1.0)

    def _arena_shape_type(self):
        """Use a special unbounded footprint."""
        return "unbounded"

    def _arena_shape_config(self, config_elem:Config):
        """Adapt the configuration parameters for the unbounded factory."""
        return {
            "color": config_elem.arena.get("color", "gray"),
            "side": self.diameter
        }

    def get_wrap_config(self):
        """Return the wrap config."""
        half = self.diameter * 0.5
        origin = Vector3D(-half, -half, 0)
        return {
            "unbounded": True,
            "origin": origin,
            "width": self.diameter,
            "height": self.diameter,
            "initial_half": half
        }

class CircularArena(SolidArena):
    
    """Circular arena."""
    def __init__(self, config_elem:Config):
        """Initialize the instance."""
        super().__init__(config_elem)
        self.height = config_elem.arena.get("height", 1)
        self.radius = config_elem.arena.get("radius", 1)
        self.color = config_elem.arena.get("color", "white")
        logging.info("Circular arena created successfully")
    

class RectangularArena(SolidArena):
    
    """Rectangular arena."""
    def __init__(self, config_elem:Config):
        """Initialize the instance."""
        super().__init__(config_elem)
        self.height = config_elem.arena.get("height", 1)
        self.length = config_elem.arena.get("length", 1)
        self.width = config_elem.arena.get("width", 1)
        self.color = config_elem.arena.get("color", "white")
        logging.info("Rectangular arena created successfully")
    
class SquareArena(SolidArena):
    
    """Square arena."""
    def __init__(self, config_elem:Config):
        """Initialize the instance."""
        super().__init__(config_elem)
        self.height = config_elem.arena.get("height", 1)
        self.side = config_elem.arena.get("side", 1)
        self.color = config_elem.arena.get("color", "white")
        logging.info("Square arena created successfully")
    
