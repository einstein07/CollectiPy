# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Fabio Oddi
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
#  You may use, modify, and redistribute this file according to the terms of the
#  license. Attribution is required if this code is used in other works.
# ------------------------------------------------------------------------------

import csv
import logging
import math
import os, json, pickle, shutil, zipfile
from typing import Any

logger = logging.getLogger("sim.data_handling")

import numpy as np
from config import Config

logger = logging.getLogger("sim.data_handling")

class DataHandlingFactory():
    """Data handling factory."""
    @staticmethod
    def create_data_handling(config_elem: Config):
        """Create data handling."""
        if config_elem.arena.get("_id") in ("abstract", "none", None):
            return DataHandling(config_elem)
        else:
            return SpaceDataHandling(config_elem)

class DataHandling():
    """Data handling."""
    SPIN_FIELD_ORDER = ("states", "angles", "external_field", "avg_direction_of_activity")

    def __init__(self, config_elem: Config):
        """Initialize the instance."""
        results_cfg = config_elem.results or {}
        base_path = results_cfg.get("base_path") or "./data/"
        self.agent_specs = self._normalize_specs(results_cfg.get("agent_specs"))
        self.group_specs = self._normalize_specs(results_cfg.get("group_specs"))
        legacy_specs = self._normalize_specs(results_cfg.get("model_specs"))
        agent_specs_were_provided = "agent_specs" in results_cfg
        if not agent_specs_were_provided and not self.agent_specs:
            # Preserve the legacy behaviour: saving base agent states unless explicitly disabled.
            self.agent_specs = {"base"}
        if legacy_specs:
            if "spin_model" in legacy_specs:
                self.agent_specs.add("spin_model")
            if "graphs" in legacy_specs:
                self.group_specs.update({"graph_messages", "graph_detection", "graphs"})
            if "graph_messages" in legacy_specs:
                self.group_specs.add("graph_messages")
            if "graph_detection" in legacy_specs:
                self.group_specs.add("graph_detection")
        self.base_dump_enabled = "base" in self.agent_specs
        self.spin_dump_enabled = "spin_model" in self.agent_specs
        self.mean_field_logging_enabled = "mean_field" in self.agent_specs
        self.graph_messages_enabled = "graphs" in self.group_specs or "graph_messages" in self.group_specs
        self.graph_detection_enabled = "graphs" in self.group_specs or "graph_detection" in self.group_specs
        self.snapshots_per_second = self._parse_snapshot_rate(results_cfg.get("snapshots_per_second", 1))
        abs_base_path = os.path.join(os.path.abspath(""), base_path)
        os.makedirs(abs_base_path, exist_ok=True)
        folder_id = 0
        while True:
            candidate = os.path.join(abs_base_path, f"config_folder_{folder_id}")
            try:
                os.makedirs(candidate, exist_ok=False)
                self.config_folder = candidate
                break
            except FileExistsError:
                folder_id += 1
        with open(os.path.join(self.config_folder, "config.json"), "w") as f:
            json.dump(config_elem.__dict__, f, indent=4, default=str)
        self.agents_files = {}
        self.agent_spin_files = {}
        self.mean_field_files = {}
        self.agent_name_order = []
        self.agent_lookup = {}
        self.agents_metadata = {}
        self.run_folder: str | None = None
        self._bifurcation_events: list[dict] = []
        self._swap_events: list[dict] = []
        self._ticks_per_second = 1
        self._snapshot_offsets = [1]
        self._last_snapshot_tick = None
        self._graph_step_dirs = {}
        self._graphs_root = None
        self._bifurcation_events: list[dict] = []
        self.hierarchy_enabled = bool(getattr(config_elem, "arena", {}).get("hierarchy"))

    def _normalize_specs(self, value):
        """Return a normalized set of model spec tokens."""
        if value is None:
            return set()
        if isinstance(value, str):
            iterable = [value]
        elif isinstance(value, (list, tuple, set)):
            iterable = value
        else:
            iterable = []
        specs = {str(item).strip().lower() for item in iterable if str(item).strip()}
        return specs

    def _parse_snapshot_rate(self, value):
        """Return a valid snapshot count per simulated second.

        The cap used to be 2/s, which silently made any decision shorter than ~1 s
        unplottable: a 0.6 s first passage produced barely one logged sample and looked
        like instantaneous commitment. `_build_snapshot_offsets` dedupes offsets against
        the tick rate, so requesting more snapshots than there are ticks simply saturates
        at one sample per tick.
        """
        try:
            rate = int(value)
        except (TypeError, ValueError):
            rate = 1
        return max(1, min(1000, rate))

    def _sanitize_tick_rate(self, ticks_per_second):
        """Ensure ticks per second is a positive integer."""
        try:
            value = int(ticks_per_second)
        except (TypeError, ValueError):
            value = 1
        return max(1, value)

    def _build_snapshot_offsets(self, ticks_per_second: int):
        """Return the list of tick offsets (within a second) where snapshots are taken."""
        ticks_per_second = max(1, ticks_per_second)
        offsets = set()
        for slot in range(1, self.snapshots_per_second + 1):
            raw = round(slot * ticks_per_second / self.snapshots_per_second)
            offsets.add(max(1, min(ticks_per_second, raw)))
        offsets.add(ticks_per_second)
        return sorted(offsets)

    def _prepare_graph_dirs(self):
        """Initialize per-step graph folders if requested."""
        if not self.run_folder:
            return
        self._graph_step_dirs = {}
        self._graphs_root = None
        if not (self.graph_messages_enabled or self.graph_detection_enabled):
            return
        graphs_root = os.path.join(self.run_folder, "graphs")
        os.makedirs(graphs_root, exist_ok=True)
        self._graphs_root = graphs_root
        if self.graph_messages_enabled:
            msg_dir = os.path.join(graphs_root, "messages")
            os.makedirs(msg_dir, exist_ok=True)
            self._graph_step_dirs["messages"] = msg_dir
        if self.graph_detection_enabled:
            det_dir = os.path.join(graphs_root, "detection")
            os.makedirs(det_dir, exist_ok=True)
            self._graph_step_dirs["detection"] = det_dir

    def _update_tick_rate(self, ticks_per_second):
        """Update the cached tick rate and snapshot schedule when needed."""
        if ticks_per_second is None:
            return
        sanitized = self._sanitize_tick_rate(ticks_per_second)
        if sanitized != self._ticks_per_second:
            self._ticks_per_second = sanitized
            self._snapshot_offsets = self._build_snapshot_offsets(self._ticks_per_second)

    def _should_capture_tick(self, tick: int, force: bool = False) -> bool:
        """Return True if the current tick should be captured."""
        if force:
            return True
        if tick is None:
            return False
        if tick <= 0:
            return False
        offsets = self._snapshot_offsets or [self._ticks_per_second]
        position_in_second = ((tick - 1) % max(1, self._ticks_per_second)) + 1
        return position_in_second in offsets

    def _finalize_graph_archives(self):
        """Zip per-step graph files (if any) and clean temporary folders."""
        if not self.run_folder:
            return
        for mode, dir_path in list(self._graph_step_dirs.items()):
            if not os.path.isdir(dir_path):
                continue
            archive_path = os.path.join(self.run_folder, f"{mode}_graphs.zip")
            if os.path.exists(archive_path):
                os.remove(archive_path)
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
                for root, _, files in os.walk(dir_path):
                    for filename in sorted(files):
                        abs_path = os.path.join(root, filename)
                        arcname = os.path.relpath(abs_path, self.run_folder)
                        zf.write(abs_path, arcname)
            shutil.rmtree(dir_path)
        if self._graphs_root and os.path.isdir(self._graphs_root) and not os.listdir(self._graphs_root):
            os.rmdir(self._graphs_root)
        self._graph_step_dirs = {}
        self._graphs_root = None

    def collect_bifurcation_events(self, events: list[dict]) -> None:
        """Accumulate bifurcation events from an agent's detector."""
        self._bifurcation_events.extend(events)

    def _write_events_json(self) -> None:
        """Write bifurcation and swap events to events.json sidecar (D-06)."""
        if self.run_folder is None or not os.path.isdir(self.run_folder):
            return
        events_data = {
            "bifurcation_events": self._bifurcation_events,
            "swap_events": [],  # Reserved for Phase 3
        }
        events_path = os.path.join(self.run_folder, "events.json")
        try:
            with open(events_path, "w") as f:
                json.dump(events_data, f, indent=2)
        except OSError as exc:
            logger.error(
                "DataHandling: failed to write events.json to '%s': %s",
                events_path, exc
            )

    def new_run(self, run: int, shapes, spins, metadata, ticks_per_second: int | None = None):
        """Create a new run."""
        self.run_folder = os.path.join(self.config_folder, f"run_{run}")
        if os.path.exists(self.run_folder):
            raise Exception(f"Error run folder {self.run_folder} already present")
        os.mkdir(self.run_folder)
        self.agents_files = {}
        self.agent_spin_files = {}
        self.agent_name_order = []
        self.agent_lookup = {}
        self.agents_metadata = metadata or {}
        self._bifurcation_events = []
        self._ticks_per_second = self._sanitize_tick_rate(ticks_per_second)
        self._snapshot_offsets = self._build_snapshot_offsets(self._ticks_per_second)
        self._last_snapshot_tick = None
        self._bifurcation_events = []
        self._swap_events = []
        self._prepare_graph_dirs()

    def collect_bifurcation_events(self, events: list[dict]) -> None:
        """Accumulate bifurcation events from an agent's detector."""
        self._bifurcation_events.extend(events)

    def collect_swap_events(self, events: list[dict]) -> None:
        """Accumulate swap events for the current run."""
        self._swap_events.extend(events)

    def _write_events_json(self) -> None:
        """Write bifurcation and swap events to events.json sidecar."""
        if self.run_folder is None or not os.path.isdir(self.run_folder):
            return
        events_data = {
            "bifurcation_events": self._bifurcation_events,
            "swap_events": self._swap_events,
        }
        events_path = os.path.join(self.run_folder, "events.json")
        try:
            with open(events_path, "w") as f:
                json.dump(events_data, f, indent=2)
        except OSError as exc:
            logger.error(
                "DataHandling: failed to write events.json to '%s': %s",
                events_path, exc
            )

    def save(self, shapes, spins, metadata, tick: int, ticks_per_second: int | None = None, force: bool = False):
        """Save value (override in subclasses)."""
        _ = (shapes, spins, metadata, tick, ticks_per_second, force)

    def close(self, shapes):
        """Close the component resources."""
        self._write_events_json()
        self._archive_run_folder()

    def _archive_run_folder(self):
        """Compress the current run folder and remove its original contents."""
        if self.run_folder is None or not os.path.isdir(self.run_folder):
            return
        zip_path = f"{self.run_folder}.zip"
        if os.path.exists(zip_path):
            os.remove(zip_path)
        base_dir = os.path.dirname(self.run_folder)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for root, _, files in os.walk(self.run_folder):
                for filename in files:
                    abs_path = os.path.join(root, filename)
                    arcname = os.path.relpath(abs_path, base_dir)
                    zf.write(abs_path, arcname)
        shutil.rmtree(self.run_folder)
        self.run_folder = None

class SpaceDataHandling(DataHandling):
    """Space data handling."""
    def __init__(self, config_elem: Config):
        """Initialize the instance."""
        super().__init__(config_elem)

    def new_run(self, run: int, shapes, spins, metadata, ticks_per_second: int | None = None):
        """Create a new run."""
        super().new_run(run, shapes, spins, metadata, ticks_per_second)
        self.agent_name_order = []
        self.agent_lookup = {}
        self.mean_field_files = {}
        if shapes is not None:
            for key, entities in shapes.items():
                for idx, entity in enumerate(entities):
                    agent_id = self._agent_identifier(key, idx, entity)
                    self.agent_lookup[(key, idx)] = agent_id
                    if agent_id not in self.agent_name_order:
                        self.agent_name_order.append(agent_id)
                    if self.base_dump_enabled:
                        file_path = os.path.join(self.run_folder, f"{agent_id}.pkl")
                        if os.path.exists(file_path):
                            raise Exception(f"Error: file {file_path} already exists")
                        file_handle = open(file_path, "wb")
                        pickler = pickle.Pickler(file_handle, protocol=pickle.HIGHEST_PROTOCOL)
                        header = ["tick", "pos x", "pos y", "pos z", "linear_velocity_cmd", "angular_velocity_cmd"]
                        if self.hierarchy_enabled:
                            header.append("hierarchy_node")
                        pickler.dump({"type": "header", "value": header, "columns": header})
                        self.agents_files[(key, idx)] = {"handle": file_handle, "pickler": pickler, "columns": header}
                    if self.spin_dump_enabled:
                        spin_path = os.path.join(self.run_folder, f"{agent_id}_spins.pkl")
                        if os.path.exists(spin_path):
                            raise Exception(f"Error: file {spin_path} already exists")
                        spin_handle = open(spin_path, "wb")
                        spin_pickler = pickle.Pickler(spin_handle, protocol=pickle.HIGHEST_PROTOCOL)
                        spin_payload = self._resolve_spin_entry((spins or {}).get(key), idx)
                        spin_columns = ["tick"]
                        if spin_payload:
                            spin_columns.extend(list(spin_payload.keys()))
                        else:
                            spin_columns.extend(self.SPIN_FIELD_ORDER)
                        spin_pickler.dump({"type": "header", "value": spin_columns, "columns": spin_columns})
                        self.agent_spin_files[(key, idx)] = {"handle": spin_handle, "pickler": spin_pickler, "columns": spin_columns}
        self.agents_metadata = metadata or {}
        # Capture the bootstrap snapshot (tick 0) right away.
        if (
            self.base_dump_enabled
            or self.spin_dump_enabled
            or self.mean_field_logging_enabled
            or self.graph_messages_enabled
            or self.graph_detection_enabled
        ):
            self.save(shapes, spins, metadata, tick=0, ticks_per_second=self._ticks_per_second, force=True)

    def save(self, shapes, spins, metadata, tick: int, ticks_per_second: int | None = None, force: bool = False):
        """Save sampled data for the current tick."""
        if not (
            self.base_dump_enabled
            or self.spin_dump_enabled
            or self.mean_field_logging_enabled
            or self.graph_messages_enabled
            or self.graph_detection_enabled
        ):
            return
        self._update_tick_rate(ticks_per_second)
        if tick is None:
            return
        spin_data = spins or {}
        self.agents_metadata = metadata or self.agents_metadata
        if self._last_snapshot_tick == tick:
            if not force:
                return
            # Already stored this tick; nothing else to do.
            return
        capture = self._should_capture_tick(tick, force=force)
        if not capture:
            return
        if shapes is not None:
            for key, entities in shapes.items():
                spin_group = spin_data.get(key)
                for idx, entity in enumerate(entities):
                    spin_values = self._resolve_spin_entry(spin_group, idx)
                    if self.base_dump_enabled:
                        entry = self.agents_files.get((key, idx))
                        if entry:
                            com = entity.center_of_mass()
                            linear_velocity_cmd, angular_velocity_cmd = self._resolve_velocity_commands(entity)
                            row = {
                                "tick": tick,
                                "pos x": com.x,
                                "pos y": com.y,
                                "pos z": com.z,
                                "linear_velocity_cmd": linear_velocity_cmd,
                                "angular_velocity_cmd": angular_velocity_cmd,
                            }
                            if self.hierarchy_enabled:
                                row["hierarchy_node"] = self._resolve_hierarchy_node(entity)
                            entry["pickler"].dump({"type": "row", "value": row})
                    if self.spin_dump_enabled and self.agent_spin_files:
                        spin_entry = self.agent_spin_files.get((key, idx))
                        if spin_entry:
                            row = {"tick": tick}
                            if spin_values is not None:
                                row.update(spin_values)
                            spin_entry["pickler"].dump({"type": "row", "value": row})
                    if self.mean_field_logging_enabled:
                        self._write_mean_field_logs(key, idx, spin_values, tick, entity)
        elif self.spin_dump_enabled and self.agent_spin_files:
            for (key, idx), spin_entry in self.agent_spin_files.items():
                spin_values = self._resolve_spin_entry(spin_data.get(key), idx)
                row = {"tick": tick}
                if spin_values is not None:
                    row.update(spin_values)
                spin_entry["pickler"].dump({"type": "row", "value": row})
        self._write_graph_snapshot(shapes, tick)
        self._last_snapshot_tick = tick

    def close(self, shapes):
        """Close the component resources."""
        if self.agents_files:
            for entry in self.agents_files.values():
                entry["handle"].flush()
                entry["handle"].close()
            self.agents_files.clear()
        if self.agent_spin_files:
            for entry in self.agent_spin_files.values():
                entry["handle"].flush()
                entry["handle"].close()
            self.agent_spin_files.clear()
        if self.mean_field_files:
            for entry in self.mean_field_files.values():
                for fh in entry.values():
                    if fh and fh.get("handle"):
                        fh["handle"].flush()
                        fh["handle"].close()
            self.mean_field_files.clear()
        self._finalize_graph_archives()
        super().close(shapes)

    def _agent_identifier(self, key, idx, shape_obj):
        """Return a stable agent identifier for file names."""
        if shape_obj is not None:
            metadata = getattr(shape_obj, "metadata", None)
            if isinstance(metadata, dict):
                name = metadata.get("entity_name")
                if name:
                    return name
        return f"{key}_{idx}"

    def _resolve_spin_entry(self, spin_group, idx):
        """Return the spin payload for the given agent, normalized as a dict."""
        if not spin_group or idx >= len(spin_group):
            return None
        payload = spin_group[idx]
        if payload is None:
            return None
        if isinstance(payload, dict):
            return {str(k): payload.get(k) for k in payload.keys()}
        if isinstance(payload, (list, tuple)):
            normalized = {}
            for pos, key in enumerate(self.SPIN_FIELD_ORDER):
                if pos < len(payload):
                    normalized[key] = payload[pos]
            return normalized if normalized else None
        return {"states": payload}

    def _resolve_velocity_commands(self, shape_obj):
        """Return the latest motion commands stored in the shape metadata."""
        metadata = getattr(shape_obj, "metadata", None)
        if not isinstance(metadata, dict):
            return None, None
        linear = metadata.get("linear_velocity_cmd")
        angular = metadata.get("angular_velocity_cmd")
        try:
            linear = float(linear) if linear is not None else None
        except (TypeError, ValueError):
            linear = None
        try:
            angular = float(angular) if angular is not None else None
        except (TypeError, ValueError):
            angular = None
        return linear, angular

    def _write_mean_field_logs(self, key, idx, spin_values: dict | None, tick: int, shape_obj):
        """Persist mean-field neural/sensory/position data to CSV files."""
        if not self.mean_field_logging_enabled or spin_values is None:
            return
        if str(spin_values.get("model")) != "mean_field":
            return
        files = self._ensure_mean_field_files(key, idx, shape_obj)
        if not files:
            return
        raw_state = spin_values.get("mean_field_state")
        if raw_state is None:
            raw_state = spin_values.get("states")
        state_values = self._flatten_array(raw_state)
        raw_perception = spin_values.get("mean_field_perception_raw")
        if raw_perception is None:
            raw_perception = spin_values.get("mean_field_perception")
        if raw_perception is None:
            raw_perception = spin_values.get("external_field")
        perception_values = self._flatten_array(raw_perception)
        raw_sensory_map = spin_values.get("mean_field_sensory_map")
        if raw_sensory_map is None:
            raw_sensory_map = spin_values.get("mean_field_perception")
        if raw_sensory_map is None:
            raw_sensory_map = spin_values.get("external_field")
        sensory_map_values = self._flatten_array(raw_sensory_map)
        norm_z = float(np.linalg.norm(state_values)) if state_values else 0.0
        target_metadata = spin_values.get("mean_field_target_metadata")
        if target_metadata is None:
            target_metadata = spin_values.get("mean_field_entities", {}).get("targets", [])
        target_metadata_str = json.dumps(target_metadata)
        modulated_target_qualities = self._flatten_array(
            spin_values.get("mean_field_modulated_target_qualities")
        )
        modulated_target_qualities_str = json.dumps(modulated_target_qualities)
        channel = spin_values.get("channel") or ""

        lambda1 = spin_values.get("mean_field_lambda1")  # Re(λ₁) from BifurcationDetector; None before first tick
        omega = spin_values.get("mean_field_omega")     # Ω from BifurcationDetector (SFA model); None for standard model

        neural_entry = files.get("neural")
        if neural_entry:
            if not neural_entry["header_written"]:
                header = ["tick"]
                header.extend(f"neuron_{i}" for i in range(len(state_values)))
                header.append("norm_z")
                header.append("lambda1")
                header.append("omega")
                neural_entry["writer"].writerow(header)
                neural_entry["header_written"] = True
            row = [tick]
            row.extend(state_values)
            row.append(norm_z)
            row.append("" if lambda1 is None else lambda1)
            row.append("" if omega is None else omega)
            neural_entry["writer"].writerow(row)

        perception_entry = files.get("perception")
        if perception_entry:
            if not perception_entry["header_written"]:
                header = ["tick"]
                header.extend(f"perception_{i}" for i in range(len(perception_values)))
                header.append("channel")
                perception_entry["writer"].writerow(header)
                perception_entry["header_written"] = True
            row = [tick]
            row.extend(perception_values)
            row.append(channel)
            perception_entry["writer"].writerow(row)

        sensory_entry = files.get("sensory")
        if sensory_entry:
            if not sensory_entry["header_written"]:
                header = ["tick"]
                header.extend(f"sensory_map_{i}" for i in range(len(sensory_map_values)))
                header.append("channel")
                sensory_entry["writer"].writerow(header)
                sensory_entry["header_written"] = True
            row = [tick]
            row.extend(sensory_map_values)
            row.append(channel)
            sensory_entry["writer"].writerow(row)

        targets_entry = files.get("targets")
        if targets_entry:
            if not targets_entry["header_written"]:
                header = ["tick", "target_metadata", "modulated_target_qualities"]
                targets_entry["writer"].writerow(header)
                targets_entry["header_written"] = True
            row = [tick, target_metadata_str, modulated_target_qualities_str]
            targets_entry["writer"].writerow(row)

        position_entry = files.get("position")
        if position_entry and shape_obj is not None:
            if not position_entry["header_written"]:
                header = ["tick", "pos_x", "pos_y", "pos_z"]
                position_entry["writer"].writerow(header)
                position_entry["header_written"] = True
            com = shape_obj.center_of_mass()
            row = [tick, com.x, com.y, com.z]
            position_entry["writer"].writerow(row)

        self._write_ddm_row(files.get("ddm"), spin_values, tick)
        self._write_ddm_transitions(files.get("ddm_transitions"), spin_values)
        self._write_percept_row(files.get("percept"), spin_values, tick)

    @staticmethod
    def _write_percept_row(entry, spin_values: dict, tick: int) -> None:
        """Persist the sampled sensory percept q_hat and the protocol that produced it.

        One row per tick per target, in the SAME shape for every decision model, so the
        shared-stream claim can be checked by diffing two agents' `_percept.csv` files:
        under `sensory_stream.mode: shared` the q_hat columns must agree exactly, tick
        for tick. Under `legacy` the file records the mode and leaves q_hat empty —
        there is no shared percept to record, each model samples its own downstream.
        """
        if not entry:
            return
        mode = spin_values.get("sensory_stream_mode")
        if mode is None:
            return
        columns = ["tick", "target", "q_hat", "mode", "seed", "frozen_sd",
                   "white_rate", "dt"]
        if not entry["header_written"]:
            entry["writer"].writerow(columns)
            entry["header_written"] = True
        common = [
            mode,
            spin_values.get("sensory_stream_seed"),
            spin_values.get("sensory_stream_frozen_sd"),
            spin_values.get("sensory_stream_white_rate"),
            spin_values.get("sensory_stream_dt"),
        ]
        qhat = spin_values.get("sensory_stream_qhat") or {}
        rows = sorted(qhat.items()) if qhat else [("", None)]
        for target, value in rows:
            # repr(), not str(): the whole point is a bit-exact diff between two files.
            entry["writer"].writerow(
                [tick, target, "" if value is None else repr(float(value))]
                + ["" if v is None else v for v in common]
            )

    @staticmethod
    def _write_ddm_row(ddm_entry, spin_values: dict, tick: int) -> None:
        """Persist the pure DDM decision variable and its boundary for this tick.

        The ring-shaped CSVs carry nothing for the pure DDM — its `mean_field_state` is a
        placeholder of zeros — so without this the decision variable is lost and figures
        such as x(t) against +/-z(t) cannot be reconstructed after the fact. Written only
        when the payload actually comes from the pure DDM.

        Note `p_first` is the belief in the FIRST configured target (`target_ids[0]`),
        the one that +z favours; it is the value the model reports as `pure_ddm_p1`.
        """
        if not ddm_entry or spin_values.get("decision_model") != "embodied_pure_ddm":
            return
        columns = [
            ("tick", tick),
            ("x", spin_values.get("pure_ddm_x")),
            ("z", spin_values.get("pure_ddm_z")),
            ("t_evidence", spin_values.get("pure_ddm_t_evidence")),
            ("A_hat", spin_values.get("pure_ddm_A_hat")),
            ("A_true", spin_values.get("pure_ddm_A_true")),
            ("c", spin_values.get("pure_ddm_c")),
            ("p_first", spin_values.get("pure_ddm_p1")),
            ("committed", spin_values.get("pure_ddm_committed")),
            ("committed_id", spin_values.get("pure_ddm_committed_id")),
            ("rt", spin_values.get("pure_ddm_rt")),
            ("R_geom", spin_values.get("pure_ddm_R_geom")),
            ("bisector_guard_fired", spin_values.get("pure_ddm_bisector_guard_fired")),
            ("z_floor_analytic", spin_values.get("pure_ddm_z_floor_analytic")),
            ("rho", spin_values.get("pure_ddm_rho")),
            ("delta", spin_values.get("pure_ddm_delta")),
            # c_tau carries all the angular dependence: a CONSTANT c_tau column
            # across a run is the direct signature of the collapse being lost.
            ("c_tau", spin_values.get("pure_ddm_c_tau")),
            ("a_star", spin_values.get("pure_ddm_a_star")),
            ("z_star", spin_values.get("pure_ddm_z_star")),
            ("geometric_error_mode", spin_values.get("pure_ddm_geometric_error_mode")),
            ("cost_ratio_used", spin_values.get("pure_ddm_cost_ratio_used")),
            # --- post-commitment flexibility ---
            ("x_over_z", spin_values.get("pure_ddm_x_over_z")),
            ("n_commits", spin_values.get("pure_ddm_n_commits")),
            ("n_releases", spin_values.get("pure_ddm_n_releases")),
            ("n_reversals", spin_values.get("pure_ddm_n_reversals")),
            ("t_first_commit", spin_values.get("pure_ddm_t_first_commit")),
            ("final_target", spin_values.get("pure_ddm_final_target")),
            ("t_swap", spin_values.get("pure_ddm_t_swap")),
            ("x_at_swap", spin_values.get("pure_ddm_x_at_swap")),
            ("dwell_before_swap", spin_values.get("pure_ddm_dwell_before_swap")),
            ("release_latency", spin_values.get("pure_ddm_release_latency")),
            ("recommit_latency", spin_values.get("pure_ddm_recommit_latency")),
            ("total_path_length", spin_values.get("pure_ddm_total_path_length")),
            ("arrived_before_reversal", spin_values.get("pure_ddm_arrived_before_reversal")),
            # --- bellman policy ---
            ("z_bellman", spin_values.get("pure_ddm_z_bellman")),
            ("z_myopic", spin_values.get("pure_ddm_z_myopic")),
            ("z_gap", spin_values.get("pure_ddm_z_gap")),
            ("past_horizon", spin_values.get("pure_ddm_past_horizon")),
        ]
        if not ddm_entry["header_written"]:
            ddm_entry["writer"].writerow([name for name, _ in columns])
            ddm_entry["header_written"] = True
        ddm_entry["writer"].writerow(
            ["" if value is None else value for _, value in columns]
        )

    @staticmethod
    def _write_ddm_transitions(entry, spin_values: dict) -> None:
        """Append commit/release events not yet written.

        The model exposes the CUMULATIVE transition list, because snapshots are sampled
        rather than taken every tick and an event-per-snapshot scheme would silently
        drop transitions between samples. Tracking a write cursor keeps the file both
        complete and free of duplicates.
        """
        if not entry or spin_values.get("decision_model") != "embodied_pure_ddm":
            return
        transitions = spin_values.get("pure_ddm_transitions") or []
        already = entry.get("rows_written", 0)
        if len(transitions) <= already:
            return
        columns = [
            "t", "tick", "type", "committed_target", "x", "z", "a_star", "delta",
            "d1", "d2", "agent_x", "agent_y", "time_since_last_transition",
        ]
        if not entry["header_written"]:
            entry["writer"].writerow(columns)
            entry["header_written"] = True
        for row in transitions[already:]:
            entry["writer"].writerow(
                ["" if row.get(name) is None else row.get(name) for name in columns]
            )
        entry["rows_written"] = len(transitions)

    def _ensure_mean_field_files(self, key, idx, shape_obj):
        """Return or create the CSV writers used for mean-field logging."""
        if not self.run_folder:
            return None
        entry = self.mean_field_files.get((key, idx))
        if entry:
            return entry
        agent_id = self.agent_lookup.get((key, idx))
        if not agent_id:
            agent_id = self._agent_identifier(key, idx, shape_obj)
            self.agent_lookup[(key, idx)] = agent_id
        neural_path = os.path.join(self.run_folder, f"{agent_id}_neural.csv")
        perception_path = os.path.join(self.run_folder, f"{agent_id}_perception.csv")
        sensory_path = os.path.join(self.run_folder, f"{agent_id}_sensory.csv")
        targets_path = os.path.join(self.run_folder, f"{agent_id}_targets.csv")
        position_path = os.path.join(self.run_folder, f"{agent_id}_position.csv")
        ddm_path = os.path.join(self.run_folder, f"{agent_id}_ddm.csv")
        ddm_tr_path = os.path.join(self.run_folder, f"{agent_id}_ddm_transitions.csv")
        percept_path = os.path.join(self.run_folder, f"{agent_id}_percept.csv")
        neural_handle = open(neural_path, "w", newline="")
        perception_handle = open(perception_path, "w", newline="")
        sensory_handle = open(sensory_path, "w", newline="")
        targets_handle = open(targets_path, "w", newline="")
        position_handle = open(position_path, "w", newline="")
        ddm_handle = open(ddm_path, "w", newline="")
        ddm_tr_handle = open(ddm_tr_path, "w", newline="")
        percept_handle = open(percept_path, "w", newline="")
        entry = {
            "neural": {"handle": neural_handle, "writer": csv.writer(neural_handle), "header_written": False},
            "perception": {"handle": perception_handle, "writer": csv.writer(perception_handle), "header_written": False},
            "sensory": {"handle": sensory_handle, "writer": csv.writer(sensory_handle), "header_written": False},
            "targets": {"handle": targets_handle, "writer": csv.writer(targets_handle), "header_written": False},
            "position": {"handle": position_handle, "writer": csv.writer(position_handle), "header_written": False},
            # Decision-variable trace for the pure DDM. The ring-shaped CSVs above carry
            # nothing for that model (its mean_field_state is a placeholder), so without
            # this the decision variable x(t) and its boundary z(t) are never persisted.
            "ddm": {"handle": ddm_handle, "writer": csv.writer(ddm_handle), "header_written": False},
            # Event log for post-commitment flexibility: one row per commit/release.
            # `rows_written` is the append cursor into the model's cumulative list.
            "ddm_transitions": {
                "handle": ddm_tr_handle, "writer": csv.writer(ddm_tr_handle),
                "header_written": False, "rows_written": 0,
            },
            # Per-tick sensory percept q_hat, in one shape for every decision model, so
            # a shared-stream run can be verified by diffing two agents' files directly
            # (FEATURE_SHARED_SENSORY_STREAM.md Section 9).
            "percept": {
                "handle": percept_handle, "writer": csv.writer(percept_handle),
                "header_written": False,
            },
        }
        self.mean_field_files[(key, idx)] = entry
        return entry

    @staticmethod
    def _flatten_array(array: Any):
        """Return the flattened values for the provided array-like input."""
        if array is None:
            return []
        values = np.asarray(array, dtype=float).reshape(-1)
        return values.tolist()

    def _resolve_hierarchy_node(self, entity):
        """Return the hierarchy node identifier for the provided entity, if any."""
        metadata = getattr(entity, "metadata", None)
        if not isinstance(metadata, dict):
            return None
        node = metadata.get("hierarchy_node")
        if node is None:
            return None
        return str(node)

    def _write_graph_snapshot(self, shapes, tick: int):
        """Persist per-step adjacency graphs (messages/detection)."""
        if not shapes:
            return
        if not (self.graph_messages_enabled or self.graph_detection_enabled):
            return
        if tick is None:
            return
        metadata = self.agents_metadata or {}
        agents = []
        for key, entities in shapes.items():
            group_meta = metadata.get(key, [])
            for idx, entity in enumerate(entities):
                agent_id = self.agent_lookup.get((key, idx))
                if not agent_id:
                    agent_id = self._agent_identifier(key, idx, entity)
                    self.agent_lookup[(key, idx)] = agent_id
                    if agent_id not in self.agent_name_order:
                        self.agent_name_order.append(agent_id)
                center = entity.center_of_mass()
                entry_meta = group_meta[idx] if idx < len(group_meta) else {}
                agents.append((agent_id, center, entry_meta))
        for mode in ("messages", "detection"):
            if mode == "messages" and not self.graph_messages_enabled:
                continue
            if mode == "detection" and not self.graph_detection_enabled:
                continue
            dir_path = self._graph_step_dirs.get(mode)
            if not dir_path:
                continue
            edges = self._compute_graph_edges(mode, agents)
            filename = os.path.join(dir_path, f"step_{tick:09d}.pkl")
            with open(filename, "wb") as fh:
                rows = [{"source": name, "target": name} for name in self.agent_name_order]
                rows.extend({"source": src, "target": dst} for src, dst in edges)
                payload = {
                    "mode": mode,
                    "tick": tick,
                    "columns": ["source", "target"],
                    "rows": rows,
                    "description": "Two-column edge list (self loops + directed edges)."
                }
                pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def _compute_graph_edges(self, mode: str, agents):
        """Compute adjacency edges for the requested mode."""
        edges = set()
        for i, (name_a, pos_a, meta_a) in enumerate(agents):
            for j, (name_b, pos_b, meta_b) in enumerate(agents):
                if i == j:
                    continue
                distance = math.dist((pos_a.x, pos_a.y), (pos_b.x, pos_b.y))
                if mode == "messages":
                    if self._can_message(meta_a, meta_b, distance):
                        edges.add((name_a, name_b))
                else:
                    if self._can_detect(meta_a, distance):
                        edges.add((name_a, name_b))
        return sorted(edges)

    @staticmethod
    def _can_message(meta_src, meta_dst, distance):
        """Return True if the source metadata allows messaging another agent at the given distance."""
        if not meta_src or not meta_src.get("msg_enable"):
            return False
        if not meta_dst:
            return False
        try:
            rng = float(meta_src.get("msg_comm_range", 0.0))
        except (TypeError, ValueError):
            rng = 0.0
        if rng <= 0:
            return False
        try:
            tx_rate = float(meta_src.get("msg_tx_rate", 0.0))
        except (TypeError, ValueError):
            tx_rate = 0.0
        if tx_rate <= 0.0:
            return False
        try:
            rx_rate = float(meta_dst.get("msg_rx_rate", 0.0))
        except (TypeError, ValueError):
            rx_rate = 0.0
        if rx_rate <= 0.0:
            return False
        return math.isinf(rng) or distance <= rng

    @staticmethod
    def _can_detect(meta, distance):
        """Return True if detection metadata allows sensing at the given distance."""
        if not meta:
            return False
        try:
            rng = float(meta.get("detection_range", 0.1))
        except (TypeError, ValueError):
            rng = 0.1
        if rng <= 0:
            return False
        try:
            freq = float(meta.get("detection_frequency", math.inf))
        except (TypeError, ValueError):
            freq = 0.0
        if freq <= 0.0:
            return False
        return math.isinf(rng) or distance <= rng
