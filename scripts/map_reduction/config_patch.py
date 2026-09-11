# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Effective configs for the sensory-map reduction experiment (spec Sections 6.1-6.3).

ONE patcher, used by the runner, the smoke run, the sanity check and the SLURM plan
alike. The base template (`factors.BASE_CONFIG`, the current campaign's RA arm) is
never modified in place; everything the spec fixes in 6.2 is SET here and then
asserted, so drift in the template raises instead of silently changing the
experiment.

Two config families:

    cell_config / trial_config      the 6.1 three-target geometry, one cell of the
                                    6.3 design (reduction x Delta_close x q_C)
    sanity_config / sanity_trial_config
                                    the 6.6 two-target standard condition
                                    (Delta = 60, dQ = 1 %, u = 6.0), sum vs max

`sensory_map` is the swept parameter: it is written into `mean_field_model` the same
way `u` and `v` are, and lands in the run's saved config.json and in every per-tick
record (`map_reduction` column of `_sensory_noise.csv`).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
from pathlib import Path

try:                                   # package import (scripts/ on sys.path)
    from map_reduction import factors
except ImportError:                    # direct import from inside the package dir
    import factors                     # type: ignore

_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def git_sha(root: Path | None = None) -> str:
    """HEAD sha, '-dirty' when the tree has modifications; never raises."""
    root = root or _ROOT
    try:
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=15)
        if sha.returncode != 0:
            return "unknown"
        out = sha.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=30)
        if dirty.returncode == 0 and dirty.stdout.strip():
            out += "-dirty"
        return out
    except Exception:                  # noqa: BLE001 - provenance must never be fatal
        return "unknown"


def config_hash(cfg: dict) -> str:
    """Hash of the CELL-level config: per-trial fields (seeds, paths, metadata)
    stripped, so every trial of a cell shares one hash."""
    stripped = copy.deepcopy(cfg)
    env = stripped.get("environment", {})
    for arena in env.get("arenas", {}).values():
        if isinstance(arena, dict):
            arena.pop("random_seed", None)
    stream = env.get("sensory_stream")
    if isinstance(stream, dict):
        stream.pop("seed", None)
    results = env.get("results")
    if isinstance(results, dict):
        results.pop("base_path", None)
        results.pop("sweep_metadata", None)
    blob = json.dumps(stripped, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------
def load_template(path: str | Path | None = None) -> dict:
    """Load the RA template and assert the premises the experiment rests on."""
    path = Path(path) if path else (_ROOT / factors.BASE_CONFIG)
    if not path.is_file():
        raise SystemExit(f"Config template not found: {path}")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    env = data["environment"]
    mv = env["agents"]["movable_0"]
    mf = mv["mean_field_model"]
    if mv.get("moving_behavior") != "mean_field":
        raise SystemExit(f"{path}: movable_0 is not a mean_field agent")
    checks = {
        "num_neurons": factors.NUM_NEURONS,
        "beta": factors.BETA,
        "kappa": factors.KAPPA,
        "integration_time": factors.INTEGRATION_TIME,
        "integration_dt": factors.INTEGRATION_DT,
        "use_thresholding": factors.USE_THRESHOLDING,
        "scale_velocity": factors.SCALE_VELOCITY,
    }
    for key, expected in checks.items():
        got = mf.get(key)
        if got != expected:
            raise SystemExit(
                f"{path}: mean_field_model.{key} = {got!r}, expected {expected!r} "
                "(factors.py records the value the experiment was designed against)."
            )
    if float(mf.get("g_adapt", 0.0)) != 0.0:
        raise SystemExit(f"{path}: g_adapt must be 0 (no SFA in this experiment)")
    stream = env.get("sensory_stream") or {}
    for key, expected in factors.SENSORY_STREAM.items():
        if stream.get(key) != expected:
            raise SystemExit(
                f"{path}: sensory_stream.{key} = {stream.get(key)!r}, expected "
                f"{expected!r} — the shared stream must be the current campaign's."
            )
    if float(env["termination"]["radius"]) != factors.ARRIVAL_RADIUS:
        raise SystemExit(f"{path}: termination.radius != {factors.ARRIVAL_RADIUS}")
    if "mean_field" not in (env.get("results") or {}).get("agent_specs", []):
        raise SystemExit(f"{path}: results.agent_specs must include 'mean_field' "
                         "(the per-tick ring logs are what the metrics read)")
    _assert_no_prohibited_keys(data)
    return data


def _assert_no_prohibited_keys(node, path: str = "") -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in factors.PROHIBITED_KEYS:
                raise SystemExit(f"Prohibited DDM parameter '{key}' at {path or '<root>'}")
            _assert_no_prohibited_keys(value, f"{path}.{key}" if path else key)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            _assert_no_prohibited_keys(value, f"{path}[{i}]")


# ---------------------------------------------------------------------------
# Shared runtime patch (6.2)
# ---------------------------------------------------------------------------
def _apply_runtime(data: dict, reduction: str, sigma: float | None,
                   p: float | None = None) -> dict:
    env = data["environment"]
    env.pop("gui", None)
    env["logging"] = {"console": False, "console_level": "ERROR", "file_level": "ERROR"}
    env["num_runs"] = 1
    env["time_limit"] = int(factors.T_MAX_TICKS)
    env["ticks_per_second"] = int(factors.TICKS_PER_SECOND)
    env["target_position_swaps"] = []
    env.pop("post_bifurcation_swap", None)
    env["sensory_stream"] = dict(factors.SENSORY_STREAM, seed=None)

    mv = env["agents"]["movable_0"]
    mv["position"] = [[0.0, 0.0, 0.0]]
    mv["orientation"] = [[0, 0, 0]]
    mv["linear_velocity"] = float(factors.LINEAR_VELOCITY)
    mv["angular_velocity"] = float(factors.ANGULAR_VELOCITY)
    mv["ticks_per_second"] = int(factors.TICKS_PER_SECOND)
    mf = mv["mean_field_model"]
    mf["u"] = float(factors.U)
    mf["v"] = float(factors.V)
    if sigma is not None:
        mf["sigma"] = float(sigma)
    mf["sigma_s"] = float(factors.SIGMA_S)
    mf["g_adapt"] = 0.0
    mf["steps_per_tick"] = 1
    mf["reference"] = "egocentric"
    block = {"reduction": str(reduction)}
    if str(reduction) == "pnorm":
        block["p"] = float(p if p is not None else 8.0)
    mf["sensory_map"] = block
    bif = mf.setdefault("bifurcation", {})
    bif["mode"] = "behavioral"
    bif["alignment_tolerance_deg"] = float(factors.ALIGNMENT_TOL_DEG)
    bif["alignment_consecutive_ticks"] = int(factors.ALIGNMENT_CONSECUTIVE_TICKS)

    results = env.setdefault("results", {})
    results["base_path"] = ""
    results.pop("sweep_metadata", None)
    return data


def _object_prototype(env: dict) -> dict:
    proto = copy.deepcopy(env["objects"]["static_0"])
    for key in ("position", "strength", "uncertainty", "color"):
        proto.pop(key, None)
    return proto


def _assert_patched(cfg: dict, expect_sigma: float | None) -> None:
    env = cfg["environment"]
    mv = env["agents"]["movable_0"]
    mf = mv["mean_field_model"]
    turn_radius = float(mv["linear_velocity"]) / math.radians(float(mv["angular_velocity"]))
    if turn_radius > float(env["termination"]["radius"]):
        raise SystemExit(
            f"Minimum turn radius {turn_radius:.4f} m exceeds the termination radius "
            f"{env['termination']['radius']} m: the agent would orbit its target.")
    if float(mf["sigma_s"]) != 0.0:
        raise SystemExit("shared sensory mode requires mean_field_model.sigma_s = 0")
    if env["sensory_stream"]["mode"] != "shared":
        raise SystemExit("sensory_stream.mode must be 'shared'")
    if mf["u"] != factors.U or mf["v"] != factors.V:
        raise SystemExit("u / v drifted from the 6.2 values")
    if expect_sigma is not None and float(mf["sigma"]) != float(expect_sigma):
        raise SystemExit(f"sigma = {mf['sigma']} != {expect_sigma}")
    if mf["sensory_map"]["reduction"] not in ("sum", "max", "pnorm"):
        raise SystemExit(f"bad reduction {mf['sensory_map']}")
    if int(env["time_limit"]) != factors.T_MAX_TICKS:
        raise SystemExit("time_limit drifted")
    _assert_reachable(env)
    _assert_no_prohibited_keys(cfg)


def _arena_side(env: dict) -> float:
    arena = next(iter(env["arenas"].values()))
    if arena.get("_id") != "square":
        raise SystemExit(f"arena must be square, got {arena.get('_id')!r}")
    return float(arena.get("side", 1.0))


def _assert_reachable(env: dict) -> None:
    """Every target must be reachable: the agent is clamped to the box
    |x|, |y| <= side/2 - diameter/2 after each tick, so the box must come within the
    arrival radius of every target, or arrival is impossible by construction."""
    half = _arena_side(env) / 2.0 - float(env["agents"]["movable_0"].get("diameter", 0.033)) / 2.0
    radius = float(env["termination"]["radius"])
    for name, obj in env["objects"].items():
        x, y = float(obj["position"][0][0]), float(obj["position"][0][1])
        dx = max(abs(x) - half, 0.0)
        dy = max(abs(y) - half, 0.0)
        gap = math.hypot(dx, dy)
        if gap >= radius:
            raise SystemExit(
                f"{name} at ({x}, {y}) is {gap:.3f} m outside the reachable box "
                f"(+-{half:.4f}); arrival radius {radius} cannot be met")


# ---------------------------------------------------------------------------
# 6.1 / 6.3: one cell of the three-target design
# ---------------------------------------------------------------------------
def cell_config(cell: dict, template: dict | None = None,
                sigma: float | None = factors.SIGMA) -> dict:
    """Effective config for one cell, per-trial fields unset.

    `cell` is one manifest record: {cell_id, reduction, delta_close_deg, q_c_rel}.
    """
    data = copy.deepcopy(template if template is not None else load_template())
    data = _apply_runtime(data, cell["reduction"], sigma, cell.get("p"))
    env = data["environment"]
    for arena in env["arenas"].values():
        arena.pop("radius", None)          # ignored by the square arena; `side` counts
        arena["side"] = float(factors.ARENA_SIDE)

    bearings = factors.bearings_deg(cell["delta_close_deg"])
    strengths = factors.strengths(cell["q_c_rel"])
    proto = _object_prototype(env)
    objects = {}
    ids = []
    for label in factors.LABELS:
        key = factors.TARGET_IDS[label].split(".")[0]
        obj = copy.deepcopy(proto)
        obj["number"] = [1]
        obj["_id"] = "idle"
        obj["position"] = [factors.position_for_bearing(bearings[label])]
        obj["strength"] = [float(strengths[label])]
        obj["uncertainty"] = [0]
        obj["color"] = factors.TARGET_COLORS[label]
        objects[key] = obj
        ids.append(factors.TARGET_IDS[label])
    env["objects"] = objects
    env["termination"] = {"type": "proximity", "target_ids": list(ids),
                          "radius": float(factors.ARRIVAL_RADIUS), "agent_ids": "any"}
    mf = env["agents"]["movable_0"]["mean_field_model"]
    mf["num_targets"] = len(ids)
    mf["target_ids"] = list(ids)
    mf["num_guards"] = 0
    mf["guard_ids"] = []
    _assert_patched(data, sigma)
    return data


def trial_config(cell: dict, trial_idx: int, seed: int, out_dir: str,
                 cell_cfg: dict | None = None) -> dict:
    """Full per-trial config: the cell config plus the paired seed and output path."""
    data = copy.deepcopy(cell_cfg if cell_cfg is not None else cell_config(cell))
    env = data["environment"]
    # One number, both roles: the arena seed (every model-side RNG, incl. the ring's
    # sigma noise) and the shared percept stream's seed. Identical across cells.
    for arena in env.get("arenas", {}).values():
        if isinstance(arena, dict):
            arena["random_seed"] = int(seed)
    env["sensory_stream"]["seed"] = int(seed)
    results = env["results"]
    results["base_path"] = str(out_dir)
    results["sweep_metadata"] = {
        "sweep": factors.SWEEP,
        "spec": factors.SPEC,
        "cell_id": int(cell["cell_id"]),
        "reduction": str(cell["reduction"]),
        "delta_close_deg": float(cell["delta_close_deg"]),
        "q_c_rel": float(cell["q_c_rel"]),
        "quality_scale": float(factors.QUALITY_SCALE),
        "trial_idx": int(trial_idx),
        "seed": int(seed),
        "base_seed": int(factors.BASE_SEED),
        "sensory_map": dict(env["agents"]["movable_0"]["mean_field_model"]["sensory_map"]),
    }
    return data


# ---------------------------------------------------------------------------
# 6.6: the two-target standard condition, sum vs max
# ---------------------------------------------------------------------------
def sanity_config(reduction: str, template: dict | None = None,
                  sigma: float | None = None) -> dict:
    """The template's own two-target geometry (Delta = 60, 5.0 vs 4.95), with only
    the 6.2 runtime applied; `sigma=None` keeps the template's value."""
    data = copy.deepcopy(template if template is not None else load_template())
    data = _apply_runtime(data, reduction, sigma)
    env = data["environment"]
    got = [float(env["objects"][k]["strength"][0]) for k in ("static_0", "static_1")]
    if tuple(got) != tuple(factors.SANITY_QUALITIES):
        raise SystemExit(f"template strengths {got} != {factors.SANITY_QUALITIES}")
    p0 = env["objects"]["static_0"]["position"][0]
    p1 = env["objects"]["static_1"]["position"][0]
    sep = math.degrees(abs(math.atan2(-p1[1], p1[0]) - math.atan2(-p0[1], p0[0])))
    if abs(sep - factors.SANITY_DELTA_DEG) > 0.5:
        raise SystemExit(f"template separation {sep:.2f} deg != {factors.SANITY_DELTA_DEG}")
    _assert_patched(data, sigma)
    return data


def sanity_trial_config(variant: str, reduction: str, trial_idx: int, seed: int,
                        out_dir: str, cfg: dict | None = None) -> dict:
    data = copy.deepcopy(cfg if cfg is not None else sanity_config(
        reduction, sigma=factors.SANITY_VARIANTS[variant]))
    env = data["environment"]
    for arena in env.get("arenas", {}).values():
        if isinstance(arena, dict):
            arena["random_seed"] = int(seed)
    env["sensory_stream"]["seed"] = int(seed)
    results = env["results"]
    results["base_path"] = str(out_dir)
    results["sweep_metadata"] = {
        "sweep": factors.SWEEP + "_sanity",
        "spec": factors.SPEC,
        "variant": str(variant),
        "reduction": str(reduction),
        "sigma": float(env["agents"]["movable_0"]["mean_field_model"]["sigma"]),
        "trial_idx": int(trial_idx),
        "seed": int(seed),
        "base_seed": int(factors.BASE_SEED),
    }
    return data


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def env_summary(cfg: dict) -> dict:
    env = cfg["environment"]
    mv = env["agents"]["movable_0"]
    mf = mv["mean_field_model"]
    return {
        "num_neurons": mf["num_neurons"], "u": mf["u"], "v": mf["v"],
        "beta": mf["beta"], "kappa": mf["kappa"], "sigma": mf["sigma"],
        "sigma_s": mf["sigma_s"], "integration_time": mf["integration_time"],
        "integration_dt": mf["integration_dt"],
        "use_thresholding": mf["use_thresholding"],
        "scale_velocity": mf["scale_velocity"],
        "linear_velocity": mv["linear_velocity"],
        "angular_velocity": mv["angular_velocity"],
        "time_limit": env["time_limit"],
        "ticks_per_second": env["ticks_per_second"],
        "arrival_radius": env["termination"]["radius"],
        "arena": {k: v for k, v in next(iter(env["arenas"].values())).items()
                  if k != "random_seed"},
        "sensory_stream": {k: v for k, v in env["sensory_stream"].items() if k != "seed"},
        "bifurcation": mf.get("bifurcation", {}),
    }


def build_manifest(n_trials: int, template: dict | None = None) -> dict:
    template = template if template is not None else load_template()
    cells = []
    for cell in factors.iter_cells():
        bearings = factors.bearings_deg(cell["delta_close_deg"])
        cells.append({
            **cell,
            "n_trials": int(n_trials),
            "base_seed": int(factors.BASE_SEED),
            "bearings_deg": bearings,
            "positions": {k: factors.position_for_bearing(v) for k, v in bearings.items()},
            "strengths": factors.strengths(cell["q_c_rel"]),
            "config_hash": config_hash(cell_config(cell, template=template)),
            "excluded": False,
            "excluded_reason": "",
        })
    probe = cell_config(cells[0], template=template)
    return {
        "sweep": factors.SWEEP,
        "spec": factors.SPEC,
        "git_sha": git_sha(),
        "base_config": factors.BASE_CONFIG,
        "reductions": list(factors.REDUCTIONS),
        "delta_close_deg": list(factors.DELTA_CLOSE_DEG),
        "q_c_rel": list(factors.Q_C_REL),
        "quality_scale": factors.QUALITY_SCALE,
        "target_range": factors.TARGET_RANGE,
        "merge_threshold_deg": factors.merge_threshold_deg(),
        "n_cells": factors.N_CELLS,
        "n_trials": int(n_trials),
        "base_seed": int(factors.BASE_SEED),
        "t_max_ticks": factors.T_MAX_TICKS,
        "alignment_tol_deg": factors.ALIGNMENT_TOL_DEG,
        "target_ids": dict(factors.TARGET_IDS),
        "locked": env_summary(probe),
        "cells": cells,
    }


def write_config(cfg: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    return path


__all__ = [
    "git_sha", "config_hash", "load_template", "cell_config", "trial_config",
    "sanity_config", "sanity_trial_config", "env_summary", "build_manifest",
    "write_config",
]
