# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Per-cell / per-trial effective configs for the (u_hat, v) factorial.

Section 3 of `uhat-v-factorial-experiment.md`. ONE patcher, used by the runner,
the smoke test, the step-halving check and the SLURM dry run alike, so no code
path can quietly see a different configuration from another.

The patch is a straight port of the one inside
`submit-ra-frontier-sweep-bwunicluster.sh`, with three deliberate changes,
all recorded in RECON.md:

  * `time_limit` becomes `factors.T_MAX_TICKS` (100) rather than the template's
    1000, because the spec fixes T_max at 100 control ticks;
  * `sensory_stream.seed` is set explicitly to the trial seed rather than left
    null (null means "inherit the arena seed", and the arena seed IS the trial
    seed here, so this is behaviourally identical and merely auditable);
  * `u` comes from `u_hat * u_star(v)` in the manifest, never from a table typed
    into a shell script;
  * `angular_velocity` is SET to 120 deg/s. The template on disk says 10, but
    every archived config of the sweep that produced the Section 12 anchor says
    120, and at 10 the plant cannot turn inside the termination radius and orbits
    (RECON.md item 7). This is a repo drift, not a design choice, so the value is
    pinned here rather than inherited.

Everything else the frontier sweep locked is asserted rather than assumed:
drift in the sensory block, in `scale_velocity`, in the target geometry or in
the ring parameters raises instead of silently producing a different experiment.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import factors

_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def git_sha(root: Path | None = None) -> str:
    """Return the repo HEAD sha, suffixed '-dirty' when the tree is modified.

    Never raises: a sweep must not die because it was launched from an export.
    """
    root = root or _ROOT
    try:
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=15,
        )
        if sha.returncode != 0:
            return "unknown"
        out = sha.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=30,
        )
        if dirty.returncode == 0 and dirty.stdout.strip():
            out += "-dirty"
        return out
    except Exception:            # noqa: BLE001 - provenance must never be fatal
        return "unknown"


def config_hash(cfg: dict) -> str:
    """Hash of the CELL-level configuration.

    Per-trial fields (seeds, output path, replicate metadata) are stripped first,
    so every trial of a cell shares one hash and two cells that differ only in
    (u, v) get different hashes. sha1, 12 hex chars, matching the length the
    Bellman table cache uses for the same job.
    """
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
    """Load the RA-arm template and assert the premises this sweep rests on."""
    path = Path(path) if path else (_ROOT / factors.BASE_CONFIG)
    if not path.is_file():
        raise SystemExit(f"Config template not found: {path}")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    env = data["environment"]
    mv = env["agents"]["movable_0"]
    mf = mv["mean_field_model"]

    # The premises. Any drift here silently changes the experiment.
    if mf.get("scale_velocity", True):
        raise SystemExit(
            f"{path}: scale_velocity must be false — this sweep holds forward "
            "speed constant so the bump amplitude cannot feed back into speed."
        )
    checks = {
        "num_neurons": factors.NUM_NEURONS,
        "beta": factors.BETA,
        "kappa": factors.KAPPA,
        "integration_time": factors.INTEGRATION_TIME,
        "integration_dt": factors.INTEGRATION_DT,
        "g_threshold": factors.G_THRESHOLD,
        "sigma": factors.SIGMA,
        "use_thresholding": factors.USE_THRESHOLDING,
    }
    for key, expected in checks.items():
        got = mf.get(key)
        if got != expected:
            raise SystemExit(
                f"{path}: mean_field_model.{key} = {got!r}, expected {expected!r} "
                "(factors.py records the value this sweep was designed against; "
                "see RECON.md before changing either)."
            )
    for name, expected in (("static_0", factors.QUALITY_CORRECT),
                           ("static_1", factors.QUALITY_DISTRACTOR)):
        got = env["objects"][name]["strength"]
        if [float(x) for x in got] != [expected]:
            raise SystemExit(
                f"{path}: objects.{name}.strength = {got!r}, expected [{expected}]"
            )
    if env["termination"]["radius"] != factors.ARRIVAL_RADIUS:
        raise SystemExit(
            f"{path}: termination.radius = {env['termination']['radius']}, "
            f"expected {factors.ARRIVAL_RADIUS}"
        )
    _assert_no_prohibited_keys(data)
    return data


def _assert_no_prohibited_keys(node, path: str = "") -> None:
    """Section 3: DDM accumulator parameters may not appear in an RA config."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in factors.PROHIBITED_KEYS:
                raise SystemExit(
                    f"Prohibited DDM parameter '{key}' found at {path or '<root>'} — "
                    "eta_rate and lambda_t must never appear in a ring-attractor config."
                )
            _assert_no_prohibited_keys(value, f"{path}.{key}" if path else key)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            _assert_no_prohibited_keys(value, f"{path}[{i}]")


# ---------------------------------------------------------------------------
# The patch
# ---------------------------------------------------------------------------
def cell_config(cell: dict, template: dict | None = None,
                dt_override: float | None = None) -> dict:
    """Return the effective config for one cell, with per-trial fields unset.

    `cell` is one record of `manifest.json`: {cell_id, v, u_hat, u_star, u, ...}.
    """
    data = copy.deepcopy(template if template is not None else load_template())
    env = data["environment"]

    # Quiet: one line per tick per trial is pure noise in a SLURM .out file, and
    # the analysis reads the run archive, never the log.
    env["logging"] = {"console": False, "console_level": "ERROR",
                      "file_level": "ERROR"}
    env["num_runs"] = 1
    env["time_limit"] = int(factors.T_MAX_TICKS)
    env["ticks_per_second"] = int(factors.TICKS_PER_SECOND)

    # Campaign-wide matched sensory front end. `seed` is filled in per trial.
    stream = {
        "mode": factors.SENSORY_STREAM_MODE,
        "frozen_sd": float(factors.SENSORY_STREAM_FROZEN_SD),
        "white_rate": float(factors.SENSORY_STREAM_WHITE_RATE),
        "seed": None,
    }
    if (stream["mode"], stream["frozen_sd"], stream["white_rate"]) != (
            "shared", 0.0, 0.035):
        raise SystemExit(f"sensory_stream drifted from the campaign constant: {stream}")
    env["sensory_stream"] = stream

    mv = env["agents"]["movable_0"]
    mv["linear_velocity"] = float(factors.LINEAR_VELOCITY)
    # Minimum turn radius is linear/angular; at the template's 10 deg/s that is
    # 0.286 m against a 0.05 m termination radius and the agent orbits forever.
    # 120 deg/s is what the archived frontier-sweep configs carry.
    mv["angular_velocity"] = float(factors.ANGULAR_VELOCITY)
    mf = mv["mean_field_model"]
    mf["u"] = float(cell["u"])
    mf["v"] = float(cell["v"])
    # Shared mode moves the frozen sensory bias upstream; percept_stream.py
    # refuses sigma_s != 0. The RA's internal `sigma` is neural noise, is not
    # shared, and stays at the template value.
    mf["sigma_s"] = float(factors.SIGMA_S)
    if dt_override is not None:
        mf["integration_dt"] = float(dt_override)

    results = env.setdefault("results", {})
    results["base_path"] = ""
    results.pop("sweep_metadata", None)
    _assert_patched(data)
    return data


def _assert_patched(cfg: dict) -> None:
    """Post-patch assertions: what the runs will actually see."""
    env = cfg["environment"]
    mv = env["agents"]["movable_0"]
    mf = mv["mean_field_model"]
    turn_radius = float(mv["linear_velocity"]) / math.radians(
        float(mv["angular_velocity"]))
    if turn_radius > float(env["termination"]["radius"]):
        raise SystemExit(
            f"Minimum turn radius {turn_radius:.4f} m exceeds the termination "
            f"radius {env['termination']['radius']} m: the agent cannot turn "
            "inside the target and will orbit it. Raise angular_velocity."
        )
    if float(mf["sigma_s"]) != 0.0:
        raise SystemExit("shared sensory mode requires mean_field_model.sigma_s = 0")
    if env["sensory_stream"]["mode"] != "shared":
        raise SystemExit("sensory_stream.mode must be 'shared' for this sweep")
    _assert_no_prohibited_keys(cfg)


def trial_config(cell: dict, trial_idx: int, seed: int, out_dir: str,
                 cell_cfg: dict | None = None,
                 dt_override: float | None = None) -> dict:
    """Return the effective config for one trial of one cell."""
    data = copy.deepcopy(cell_cfg if cell_cfg is not None
                         else cell_config(cell, dt_override=dt_override))
    env = data["environment"]

    # The arena seed is what the simulator turns into (a) every agent-side RNG,
    # including the ring's internal sigma noise, and (b) the trial seed the
    # shared percept stream is reconstructed from. One number, both roles, so
    # the same seed in two cells means the same noise realisation in both.
    for arena in env.get("arenas", {}).values():
        if isinstance(arena, dict):
            arena["random_seed"] = int(seed)
    env["sensory_stream"]["seed"] = int(seed)

    results = env["results"]
    results["base_path"] = str(out_dir)
    results["sweep_metadata"] = {
        "sweep": "uhat_v_sweep",
        "cell_id": int(cell["cell_id"]),
        "v": float(cell["v"]),
        "u_hat": float(cell["u_hat"]),
        "u_star": float(cell["u_star"]),
        "u": float(cell["u"]),
        "trial_idx": int(trial_idx),
        "seed": int(seed),
        "base_seed": int(factors.BASE_SEED),
        "integration_dt": float(env["agents"]["movable_0"]
                                ["mean_field_model"]["integration_dt"]),
    }
    return data


def write_trial_config(cfg: dict, path: str | Path) -> Path:
    """Serialise one trial config; returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    return path


def seed_for(trial_idx: int) -> int:
    """Section 2's paired seed list: identical in every cell."""
    return int(factors.BASE_SEED) + int(trial_idx)


def env_summary(cfg: dict) -> dict:
    """The handful of values worth echoing in a startup line."""
    env = cfg["environment"]
    mf = env["agents"]["movable_0"]["mean_field_model"]
    return {
        "u": mf["u"], "v": mf["v"], "sigma": mf["sigma"],
        "angular_velocity": env["agents"]["movable_0"]["angular_velocity"],
        "integration_dt": mf["integration_dt"],
        "use_thresholding": mf["use_thresholding"],
        "time_limit": env["time_limit"],
        "linear_velocity": env["agents"]["movable_0"]["linear_velocity"],
        "sensory_stream": {k: v for k, v in env["sensory_stream"].items()
                           if k != "seed"},
    }


__all__ = [
    "git_sha", "config_hash", "load_template", "cell_config", "trial_config",
    "write_trial_config", "seed_for", "env_summary",
]
