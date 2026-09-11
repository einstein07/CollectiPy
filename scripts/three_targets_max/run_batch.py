#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2026 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Three equal targets, `max` sensory map: one array task = one (v, u) cell x run range.

    python3 scripts/three_targets_max/run_batch.py \\
        --v <v> --u <u> --first-run <a> --last-run <b> --base-root <dir> \\
        [--failures-dir <dir>] [--task-tag <s>] [--configs-only] [--subprocess] [--force]
    python3 scripts/three_targets_max/run_batch.py --print-grid      # the default u / v grid

The effective config is the current campaign's RA template
(`config/qd_sweep_ra_template.json`: sigma 1.5, shared sensory stream with
white_rate 0.07071068, 1 tick/s, time_limit 1000, linear velocity 0.05, angular
velocity 120, arrival radius 0.05, unthresholded readout, constant speed) with

    - THREE targets at the positions of `config/mean_field_3_targets_no_viz.json`:
      static_0 (0.383, -0.321), static_1 (0.5, 0.0), static_2 (0.383, 0.321),
      i.e. bearings +40, 0, -40 degrees at range 0.5 from the start pose;
    - every strength 5.0 (no quality difference);
    - mean_field_model.sensory_map = {"reduction": "max"}  (the ONLY arm run here;
      the `sum` data exists already);
    - the arena of the 3-target config (square, side 2): the pasted template's unit
      square cannot place static_1 — its 0.05 m cylinder at x = 0.5 straddles the
      wall and the simulator refuses ("Impossible to place object"); ARENA_SIDE /
      --arena-side override it;
    - u and v per cell; everything else inherited untouched.

Per replicate (the traceability contract of the qd sweep):

    <base-root>/v_<v>/u_<u>/replicate_<id>/
        config.json                 the exact effective config, written BEFORE the run
        run_meta.json               u, v, run_id, both seeds, scheme, git sha, config hash
        config_folder_0/run_1.zip   the simulator's native logs
        .done                       written only on verified success

A replicate with `.done` is skipped (idempotent resubmits). One process imports the
simulator once and runs the whole batch; `--subprocess` is the one-`main.py`-per-
replicate verification path. Failures append to `<failures-dir>/task_<tag>.log` and
the task exits non-zero.

Seeds follow the frontier-v1 scheme (`scripts/ra_ddm_frontier/seeding.py`): trial
identity = (dth_deg, diff_bp, run_id) with dth_deg = 40 (the pair separation here)
and diff_bp = 0 (equal qualities); the shared percept stream gets the `env` seed and
the arena (hence the ring's private sigma noise) the `model` seed. Set
SEED_DTH_DEG / SEED_DIFF_BP to key the seeds like an existing `sum` data set.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in (str(_ROOT / "scripts" / "ra_ddm_frontier"), str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import seeding  # noqa: E402  (scripts/ra_ddm_frontier/seeding.py: frontier-v1, verbatim)

# ---------------------------------------------------------------------------
# The design — single source of truth for the submit script as well
# ---------------------------------------------------------------------------
SWEEP = "three_targets_max_map"
TEMPLATE = _ROOT / "config" / "qd_sweep_ra_template.json"
POSITIONS_SOURCE = _ROOT / "config" / "mean_field_3_targets_no_viz.json"
REDUCTION = "max"
STRENGTH = 5.0
TARGET_IDS = ["static_0.s#0", "static_1.s#0", "static_2.s#0"]
#: The current campaign's RA surface (scripts/qd_sweep_fixed_noise/qd.py):
#: absolute u only, 0 + 2..35 in steps of 1; kernel shape 0.1..1.0.
U_GRID = [6.16] #+ [float(u) for u in range(2, 36)]
V_GRID = [0.5]#[round(0.1 * i, 1) for i in range(1, 11)]
N_RUNS = 100
#: Arena side for the three-target geometry (config/mean_field_3_targets_no_viz.json).
ARENA_SIDE = float(os.environ.get("ARENA_SIDE") or 2.0)
SEED_DTH_DEG = int(os.environ.get("SEED_DTH_DEG", "40"))
SEED_DIFF_BP = int(os.environ.get("SEED_DIFF_BP", "0"))

#: Values of the pasted template that the runs rest on; any drift halts.
TEMPLATE_EXPECT = {
    "num_neurons": 30, "beta": 1.0, "kappa": 20, "integration_time": 50,
    "integration_dt": 0.1, "sigma": 1.5, "sigma_s": 0.0, "use_thresholding": False,
    "scale_velocity": False, "g_adapt": 0.0,
}
STREAM_EXPECT = {"mode": "shared", "frozen_sd": 0.0, "white_rate": 0.07071068}


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def git_sha() -> str:
    try:
        sha = subprocess.run(["git", "-C", str(_ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=15)
        if sha.returncode != 0:
            return "unknown"
        out = sha.stdout.strip()
        dirty = subprocess.run(["git", "-C", str(_ROOT), "status", "--porcelain",
                                "--untracked-files=no"],
                               capture_output=True, text=True, timeout=30)
        if dirty.returncode == 0 and dirty.stdout.strip():
            out += "-dirty"
        return out
    except Exception:                       # noqa: BLE001 — provenance never fatal
        return "unknown"


def config_hash(cfg: dict) -> str:
    """Cell-level hash: per-replicate fields stripped."""
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
# Template -> cell config
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_template() -> dict:
    if not TEMPLATE.is_file():
        raise SystemExit(f"template not found: {TEMPLATE}")
    data = _load_json(TEMPLATE)
    env = data["environment"]
    mf = env["agents"]["movable_0"]["mean_field_model"]
    for key, want in TEMPLATE_EXPECT.items():
        if mf.get(key) != want:
            raise SystemExit(f"{TEMPLATE}: mean_field_model.{key} = {mf.get(key)!r}, "
                             f"expected {want!r} — the template drifted from the pasted config")
    stream = env.get("sensory_stream") or {}
    for key, want in STREAM_EXPECT.items():
        if stream.get(key) != want:
            raise SystemExit(f"{TEMPLATE}: sensory_stream.{key} = {stream.get(key)!r}, "
                             f"expected {want!r}")
    if int(env["time_limit"]) != 1000 or float(env["termination"]["radius"]) != 0.05:
        raise SystemExit(f"{TEMPLATE}: time_limit / termination.radius drifted")
    ag = env["agents"]["movable_0"]
    if float(ag["linear_velocity"]) != 0.05 or float(ag["angular_velocity"]) != 120:
        raise SystemExit(f"{TEMPLATE}: plant velocities drifted")
    return data


def target_positions() -> dict[str, list[float]]:
    """The three positions of config/mean_field_3_targets_no_viz.json, read from it."""
    data = _load_json(POSITIONS_SOURCE)
    objects = data["environment"]["objects"]
    out = {}
    for name in ("static_0", "static_1", "static_2"):
        pos = objects[name]["position"][0]
        out[name] = [float(pos[0]), float(pos[1]), float(pos[2]) if len(pos) > 2 else 0.0]
    return out


def build_cell_config(u: float, v: float, template: dict | None = None,
                      arena_side: float | None = ARENA_SIDE) -> dict:
    cfg = copy.deepcopy(template if template is not None else load_template())
    env = cfg["environment"]
    env.pop("gui", None)
    env["num_runs"] = 1
    env["sensory_stream"] = dict(STREAM_EXPECT, seed=None)

    # Three targets at the opened 3-target config's positions, all strength 5.0.
    proto = copy.deepcopy(env["objects"]["static_0"])
    for key in ("position", "strength", "uncertainty", "color"):
        proto.pop(key, None)
    objects = {}
    for name, pos in target_positions().items():
        obj = copy.deepcopy(proto)
        obj["number"] = [1]
        obj["_id"] = "idle"
        obj["position"] = [pos]
        obj["strength"] = [float(STRENGTH)]
        obj["uncertainty"] = [0]
        obj["color"] = "green"
        objects[name] = obj
    env["objects"] = objects
    env["termination"] = {"type": "proximity", "target_ids": list(TARGET_IDS),
                          "radius": float(env["termination"]["radius"]), "agent_ids": "any"}
    if arena_side is not None:
        for arena in env["arenas"].values():
            arena.pop("radius", None)
            arena["side"] = float(arena_side)

    mf = env["agents"]["movable_0"]["mean_field_model"]
    mf["u"] = float(u)
    mf["v"] = float(v)
    mf["num_targets"] = len(TARGET_IDS)
    mf["target_ids"] = list(TARGET_IDS)
    mf["num_guards"] = 0
    mf["guard_ids"] = []
    mf["sigma_s"] = 0.0
    mf["sensory_map"] = {"reduction": REDUCTION}

    results = env.setdefault("results", {})
    results["base_path"] = ""
    results.pop("sweep_metadata", None)
    _assert_cell(cfg)
    return cfg


def _assert_cell(cfg: dict) -> None:
    env = cfg["environment"]
    mf = env["agents"]["movable_0"]["mean_field_model"]
    if mf["sensory_map"] != {"reduction": REDUCTION}:
        raise SystemExit(f"sensory_map must be {{'reduction': '{REDUCTION}'}}")
    strengths = [float(o["strength"][0]) for o in env["objects"].values()]
    if len(strengths) != 3 or any(s != STRENGTH for s in strengths):
        raise SystemExit(f"strengths {strengths} != three times {STRENGTH}")
    if mf["target_ids"] != TARGET_IDS or env["termination"]["target_ids"] != TARGET_IDS:
        raise SystemExit("target ids drifted")
    if mf["u"] is None or mf["v"] is None:
        raise SystemExit("u / v unset")
    if float(mf["sigma_s"]) != 0.0 or env["sensory_stream"]["mode"] != "shared":
        raise SystemExit("shared stream requires sigma_s = 0")
    # Reachability: the agent is clamped to |x|, |y| <= side/2 - diameter/2; every
    # target must lie within the arrival radius of that box.
    arena = next(iter(env["arenas"].values()))
    side = float(arena.get("side", 1.0))
    half = side / 2.0 - float(env["agents"]["movable_0"].get("diameter", 0.033)) / 2.0
    radius = float(env["termination"]["radius"])
    for name, obj in env["objects"].items():
        x, y = float(obj["position"][0][0]), float(obj["position"][0][1])
        gap = ((max(abs(x) - half, 0.0)) ** 2 + (max(abs(y) - half, 0.0)) ** 2) ** 0.5
        if gap >= radius:
            raise SystemExit(f"{name} at ({x}, {y}) is unreachable inside the arena "
                             f"(side {side}, clamp +-{half:.4f}, arrival radius {radius})")


def apply_seeds(cfg: dict, run_id: int) -> dict:
    env = cfg["environment"]
    s = seeding.env_seed(SEED_DTH_DEG, SEED_DIFF_BP, int(run_id), "sensory")
    m = seeding.model_seed("ra", SEED_DTH_DEG, SEED_DIFF_BP, int(run_id))
    env["sensory_stream"]["seed"] = int(s)
    for arena in env["arenas"].values():
        if isinstance(arena, dict):
            arena["random_seed"] = int(m)
    return {"env_seed_sensory": int(s), "model_seed": int(m),
            "seed_scheme": seeding.SCHEME, "seed_dth_deg": SEED_DTH_DEG,
            "seed_diff_bp": SEED_DIFF_BP}


def replicate_dir(base_root: Path, v: float, u: float, run_id: int) -> Path:
    return Path(base_root) / f"v_{v:g}" / f"u_{u:g}" / f"replicate_{int(run_id)}"


def build_replicate(u: float, v: float, run_id: int, rep_dir: Path, cell_cfg: dict,
                    provenance: dict) -> Path:
    cfg = copy.deepcopy(cell_cfg)
    seeds = apply_seeds(cfg, run_id)
    cfg["environment"]["results"]["base_path"] = str(rep_dir)
    cfg["environment"]["results"]["sweep_metadata"] = {
        "sweep": SWEEP, "u": float(u), "v": float(v), "run_id": int(run_id),
        "reduction": REDUCTION, **seeds,
    }
    rep_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = rep_dir / "config.json"
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    meta = {"sweep": SWEEP, "template": str(TEMPLATE.relative_to(_ROOT)),
            "positions_source": str(POSITIONS_SOURCE.relative_to(_ROOT)),
            "reduction": REDUCTION, "strength": STRENGTH, "u": float(u), "v": float(v),
            "run_id": int(run_id), **seeds, **provenance}
    with open(rep_dir / "run_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return cfg_path


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
class InProcessRunner:
    def __init__(self):
        from config import Config                    # noqa: F401
        from environment import EnvironmentFactory   # noqa: F401
        self._configured_logging = False

    def run(self, config_path: Path) -> None:
        from config import Config
        from environment import EnvironmentFactory
        from logging_utils import configure_logging
        my_config = Config(config_path=str(config_path))
        if not self._configured_logging:
            configure_logging(my_config.environment.get("logging"),
                              config_path=config_path.resolve(), project_root=_ROOT)
            self._configured_logging = True
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            env = EnvironmentFactory.create_environment(my_config)
            env.start()


class SubprocessRunner:
    def __init__(self, timeout: float = 1800.0):
        self.timeout = timeout

    def run(self, config_path: Path) -> None:
        res = subprocess.run([sys.executable, str(_ROOT / "src" / "main.py"),
                              "-c", str(config_path)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                             timeout=self.timeout)
        if res.returncode != 0:
            raise RuntimeError(f"main.py exited {res.returncode}: {res.stderr[-2000:]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--print-grid", action="store_true",
                    help="print the default u and v grids (one line each) and exit")
    ap.add_argument("--v", type=float)
    ap.add_argument("--u", type=float)
    ap.add_argument("--first-run", type=int, default=1)
    ap.add_argument("--last-run", type=int, default=N_RUNS)
    ap.add_argument("--base-root", type=Path)
    ap.add_argument("--failures-dir", type=Path, default=None)
    ap.add_argument("--task-tag", default=None)
    ap.add_argument("--arena-side", type=float, default=ARENA_SIDE,
                    help=f"square arena side (default {ARENA_SIDE:g}, the 3-target config's; "
                         "the template's unit square cannot place static_1)")
    ap.add_argument("--configs-only", action="store_true")
    ap.add_argument("--subprocess", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if args.print_grid:
        print(" ".join(f"{u:g}" for u in U_GRID))
        print(" ".join(f"{v:g}" for v in V_GRID))
        return 0
    if args.v is None or args.u is None or args.base_root is None:
        raise SystemExit("--v, --u and --base-root are required (or --print-grid)")

    cfg0 = build_cell_config(args.u, args.v, arena_side=args.arena_side)
    provenance = {"git_sha": git_sha(), "config_hash": config_hash(cfg0)}
    ident = f"v{args.v:g}_u{args.u:g}"
    print(f"[{ident}] {REDUCTION} map, 3 targets @ {STRENGTH}, runs "
          f"{args.first_run}..{args.last_run} cfg={provenance['config_hash']} "
          f"sha={provenance['git_sha']}")
    sys.stdout.flush()

    runner = None
    if not args.configs_only:
        runner = SubprocessRunner() if args.subprocess else InProcessRunner()
    failures_dir = args.failures_dir or (Path(args.base_root) / "failures")
    tag = args.task_tag or f"{ident}_{args.first_run}"
    fail_log = failures_dir / f"task_{tag}.log"

    n_run = n_skip = n_fail = 0
    t0 = time.time()
    for run_id in range(args.first_run, args.last_run + 1):
        rep_dir = replicate_dir(args.base_root, args.v, args.u, run_id)
        done = rep_dir / ".done"
        if done.exists() and not args.force:
            n_skip += 1
            continue
        cfg_path = build_replicate(args.u, args.v, run_id, rep_dir, cfg0, provenance)
        if args.configs_only:
            print(cfg_path)
            continue
        try:
            runner.run(cfg_path)
            if not list(rep_dir.glob("config_folder_*/run_*.zip")):
                raise RuntimeError("run completed but produced no run archive")
            done.touch()
            n_run += 1
        except Exception as exc:                 # noqa: BLE001 — data, not abort
            n_fail += 1
            failures_dir.mkdir(parents=True, exist_ok=True)
            with open(fail_log, "a", encoding="utf-8") as fh:
                fh.write(f"{cfg_path}\t{exc!r}\n")
            print(f"  [{ident}] run {run_id} FAILED: {exc!r}")
            sys.stdout.flush()

    if not args.configs_only:
        dt = time.time() - t0
        print(f"[{ident}] done: ran {n_run}, skipped {n_skip}, failed {n_fail} in "
              f"{dt:.1f}s ({dt / max(n_run, 1):.2f} s/run)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
