# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Tests for the configurable reduction of the ring-attractor sensory map
(max-sensory-map-spec.md Section 5).

    5.1  regression: `sum` (block absent AND block present) is np.array_equal to the
         PRE-CHANGE map on every config in config/ that exercises the ring map
    5.2  single target: sum, max and pnorm coincide
    5.3  two equal targets at kappa = 20: the merge threshold and the max cusp
    5.4  amplitude bound of max; the sum overshoot inside Delta*
    5.5  pnorm limits (p = 1 -> sum, p = 64 -> max)
    5.6  wrap-around at 0 / 2 pi
    5.7  unequal qualities: both peaks survive, the crossover leans to the weaker side

plus the Section 4 plumbing: config validation, the default, run metadata, and
Section 4.5 (the same trial seed gives the same noise stream under sum and max).

The 5.1 fixture is generated from the PRE-CHANGE code, which is the only way the
comparison means anything. Regenerate it ONLY from a checkout that predates the
reduction parameter (commit cb4a50d or earlier):

    git worktree add /tmp/collectipy-pre <pre-change-commit>
    COLLECTIPY_SRC=/tmp/collectipy-pre/src .venv/bin/python -c \\
        "import tests.test_sensory_map_reduction as t; t.generate_reference()"

Run with:
    env -u PYTHONPATH .venv/bin/python -m pytest tests/test_sensory_map_reduction.py -q
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from random import Random

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
SRC_DIR = os.environ.get("COLLECTIPY_SRC") or str(_ROOT / "src")
SRC_DIR = str(Path(SRC_DIR).resolve())
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import models  # noqa: F401,E402  (registers plugins)
from geometry_utils.vector3D import Vector3D  # noqa: E402
from models.mean_field_systems import MeanFieldSystem  # noqa: E402
from plugin_registry import get_movement_model  # noqa: E402

CONFIG_DIR = _ROOT / "config"
FIXTURES_DIR = _HERE / "fixtures"
FIXTURE_PATH = FIXTURES_DIR / "sensory_map_sum_pre_change.npz"

KAPPA = 20.0
FINE_N = 3600          # 0.1 degree grid: analytic checks are not grid-limited
SNAPSHOT_TICKS = 3     # per config: exercises the per-tick sigma_s draw as well
SNAPSHOT_SEED = 20260911

#: Configs whose FULL config->model path is snapshotted (closed loop, 3 ticks). The
#: system-level snapshot covers every ring config; these additionally lock the
#: MeanFieldMovementModel parsing path with the block absent.
MODEL_PATH_CONFIGS = (
    "mean_field_2_targets",          # the standard 2-target RA config (legacy, sigma_s)
    "mean_field_3_targets_no_viz",
    "mean_field_4_targets_no_viz",
    "qd_sweep_ra_template",          # shared sensory stream, the current campaign
)


# ---------------------------------------------------------------------------
# Harness: configs -> (bearing, quality) per target, exactly as GPS detection does
# ---------------------------------------------------------------------------
def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _wrap_arr(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def _resolve_targets(env: dict, agent_cfg: dict, mf: dict) -> list[tuple[str, float, float, tuple]]:
    """(target_id, bearing_rad, quality, (x, y, z)) for every resolvable target id.

    Bearing follows `GPSDetectionModel._compute_relative_angle`: atan2(-dy, dx) minus the
    agent heading, wrapped. The agent starts at its configured pose.
    """
    objects = env.get("objects") or {}
    start = (agent_cfg.get("position") or [[0.0, 0.0, 0.0]])[0]
    orient = (agent_cfg.get("orientation") or [[0.0, 0.0, 0.0]])[0]
    heading_deg = float(orient[-1]) if orient else 0.0
    out = []
    for tid in mf.get("target_ids") or []:
        key = str(tid).split(".")[0]
        obj = objects.get(key)
        if not obj:
            continue
        pos = (obj.get("position") or [[0.0, 0.0, 0.0]])[0]
        strength = float((obj.get("strength") or [1.0])[0])
        dx = float(pos[0]) - float(start[0])
        dy = float(pos[1]) - float(start[1])
        angle_deg = math.degrees(math.atan2(-dy, dx)) - heading_deg
        angle_deg = ((angle_deg + 180.0) % 360.0) - 180.0
        z = float(pos[2]) if len(pos) > 2 else 0.0
        out.append((str(tid), math.radians(angle_deg), strength,
                    (float(pos[0]), float(pos[1]), z)))
    return out


def ring_config_cases() -> list[dict]:
    """Every config under config/ whose agent runs the ring attractor on >= 1 target."""
    cases = []
    for path in sorted(CONFIG_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        env = data.get("environment") or {}
        for aname, agent_cfg in (env.get("agents") or {}).items():
            if not isinstance(agent_cfg, dict):
                continue
            if agent_cfg.get("moving_behavior") != "mean_field":
                continue
            mf = agent_cfg.get("mean_field_model") or {}
            targets = _resolve_targets(env, agent_cfg, mf)
            if not targets:
                continue
            cases.append({"name": f"{path.stem}:{aname}", "stem": path.stem,
                          "env": env, "agent_cfg": agent_cfg, "mf": mf,
                          "targets": targets})
    assert cases, "no ring-attractor configs found under config/"
    return cases


def system_maps(case: dict, ticks: int = SNAPSHOT_TICKS, **extra) -> np.ndarray:
    """(ticks, n) sensory maps straight from MeanFieldSystem.compute_sensory_map.

    Only the parameters the map depends on are taken from the config; `extra` is
    forwarded to the constructor (used post-change to pass the reduction explicitly).
    """
    mf = case["mf"]
    ids = [t[0] for t in case["targets"]]
    angles = np.array([t[1] for t in case["targets"]], dtype=float)
    quals = np.array([t[2] for t in case["targets"]], dtype=float)
    system = MeanFieldSystem(
        num_neurons=int(mf.get("num_neurons", 100)),
        kappa=float(mf.get("kappa", 20.0)),
        sigma_s=float(mf.get("sigma_s", 0.0)),
        num_targets=len(ids),
        rng=np.random.default_rng(SNAPSHOT_SEED),
        noise_rng=np.random.default_rng(SNAPSHOT_SEED + 1),
        **extra,
    )
    maps = []
    for tick in range(ticks):
        b = system.compute_sensory_map(
            num_targets=len(ids), num_guards=0, target_ids=ids,
            target_angles=angles, target_qualities=quals, tick=tick,
        )
        maps.append(np.array(b, dtype=float, copy=True))
    return np.stack(maps)


class _Shape:
    def __init__(self, name, pos):
        self.metadata = {"entity_name": name}
        self._pos = pos

    def center_of_mass(self):
        return self._pos


class _Agent:
    """Stub agent with an arena-style seeded RNG, whose pose responds to commands."""

    def __init__(self, cfg, seed=0, ticks_per_second=1, velocity=0.05, trial_seed=None):
        self.config_elem = cfg
        self.ticks_per_second = ticks_per_second
        self.position = Vector3D(0.0, 0.0, 0.0)
        self.orientation = Vector3D(0.0, 0.0, 0.0)
        self.max_absolute_velocity = velocity / ticks_per_second
        self.max_angular_velocity = 120.0 / ticks_per_second
        self.linear_velocity_cmd = 0.0
        self.angular_velocity_cmd = 0.0
        self.detection = "GPS"
        self.detection_range = 5.0
        self.detection_config = {}
        self._task = None
        self.random_generator = Random(seed)
        self.trial_seed = trial_seed

    def get_name(self):
        return "movable_0"

    def get_task(self):
        return self._task

    def set_task(self, t):
        self._task = t

    def ticks(self):
        return self.ticks_per_second

    def get_detection_range(self):
        return self.detection_range

    def get_random_generator(self):
        return self.random_generator

    def should_sample_detection(self, tick=None):
        return True

    def integrate(self):
        self.orientation = Vector3D(0.0, 0.0, self.orientation.z + self.angular_velocity_cmd)
        th = math.radians(self.orientation.z)
        v = self.linear_velocity_cmd
        self.position = Vector3D(self.position.x + v * math.cos(th),
                                 self.position.y - v * math.sin(th), 0.0)


def _objects_from_case(case: dict) -> dict:
    objs = {}
    for tid, _angle, strength, (x, y, z) in case["targets"]:
        pos = Vector3D(x, y, z)
        objs[tid] = ([_Shape(tid, pos)], [pos], [strength], [0.0])
    return objs


def build_model(case: dict, seed: int = SNAPSHOT_SEED, block: dict | None = None,
                mf_overrides: dict | None = None):
    """(agent, model) for one config case through the real config-parsing path.

    `block`, when given, is inserted as `mean_field_model.sensory_map`.
    """
    mf = json.loads(json.dumps(case["mf"]))
    mf.pop("sensory_map", None)
    # Sweep templates leave u / v null for the patcher to fill; use the campaign's
    # standard values so the config path can be exercised on them too.
    for key, default in (("u", 6.0), ("v", 0.5)):
        if mf.get(key) is None:
            mf[key] = default
    if block is not None:
        mf["sensory_map"] = dict(block)
    if mf_overrides:
        mf.update(mf_overrides)
    cfg = {"mean_field_model": mf, "detection": "GPS"}
    stream = case["env"].get("sensory_stream")
    if stream:
        cfg["sensory_stream"] = dict(stream)
    agent = _Agent(cfg, seed=seed, trial_seed=seed)
    return agent, get_movement_model("mean_field", agent)


def model_maps(case: dict, ticks: int = SNAPSHOT_TICKS, seed: int = SNAPSHOT_SEED,
               block: dict | None = None) -> dict[str, np.ndarray]:
    """Closed-loop run through MeanFieldMovementModel: per-tick map and ring state."""
    agent, model = build_model(case, seed=seed, block=block)
    objs = _objects_from_case(case)
    maps, states = [], []
    for tick in range(ticks):
        model.step(agent, tick, None, objs, {})
        agent.integrate()
        maps.append(model.mean_field_system.get_sensory_map())
        states.append(model.mean_field_system.get_state())
    return {"map": np.stack(maps), "state": np.stack(states)}


def generate_reference(path: Path = FIXTURE_PATH) -> Path:
    """Write the 5.1 fixture. Call this from a PRE-CHANGE checkout only."""
    payload = {}
    for case in ring_config_cases():
        payload[f"system::{case['name']}"] = system_maps(case)
        if case["stem"] in MODEL_PATH_CONFIGS:
            out = model_maps(case)
            payload[f"model_map::{case['name']}"] = out["map"]
            payload[f"model_state::{case['name']}"] = out["state"]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    return path


# ---------------------------------------------------------------------------
# Reference implementation of the spec's map (Section 3), used by 5.2 - 5.7
# ---------------------------------------------------------------------------
def _theta(n: int = FINE_N) -> np.ndarray:
    return np.linspace(-np.pi, np.pi, n, endpoint=False)


def _map(phi, q, reduction, p=8.0, n=FINE_N, kappa=KAPPA):
    from models.mean_field_systems import sensory_map
    return sensory_map(_theta(n), np.asarray(phi, dtype=float), np.asarray(q, dtype=float),
                       kappa, reduction=reduction, p=p)


def _at(values: np.ndarray, angle: float, n: int = FINE_N) -> float:
    """Map value at the grid node nearest `angle`."""
    theta = _theta(n)
    idx = int(np.argmin(np.abs(_wrap_arr(theta - angle))))
    return float(values[idx])


def _argmax_set(values: np.ndarray, tol: float = 1e-9) -> set[int]:
    return set(np.flatnonzero(values >= values.max() - tol).tolist())


def _deg(x: float) -> float:
    return math.radians(x)


# ---------------------------------------------------------------------------
# 5.1 Regression, default mode
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not FIXTURE_PATH.exists(),
                    reason="pre-change reference missing; see module docstring")
def test_5_1_sum_is_bit_identical_to_pre_change_snapshot():
    """Block absent and block present with `sum` both reproduce the pre-change map,
    on every config exercising the ring map and through the full config path."""
    reference = np.load(FIXTURE_PATH)
    cases = ring_config_cases()
    seen = 0
    for case in cases:
        key = f"system::{case['name']}"
        assert key in reference.files, f"{key} missing from the fixture; regenerate"
        ref = reference[key]
        # (a) MeanFieldSystem default construction (nothing passed)
        got = system_maps(case)
        assert np.array_equal(got, ref), f"{case['name']}: default system path drifted"
        # (b) explicit reduction 'sum'
        got = system_maps(case, sensory_map_reduction="sum")
        assert np.array_equal(got, ref), f"{case['name']}: explicit sum drifted"
        seen += 1
        if case["stem"] in MODEL_PATH_CONFIGS:
            ref_map = reference[f"model_map::{case['name']}"]
            ref_state = reference[f"model_state::{case['name']}"]
            for block in (None, {"reduction": "sum"}, {"reduction": "sum", "p": 8.0}):
                out = model_maps(case, block=block)
                assert np.array_equal(out["map"], ref_map), (
                    f"{case['name']}: config path (block={block}) map drifted")
                assert np.array_equal(out["state"], ref_state), (
                    f"{case['name']}: config path (block={block}) ring state drifted")
    assert seen >= 1
    assert any(c["stem"] == "mean_field_2_targets" for c in cases), \
        "the standard 2-target RA config must be part of the regression set"


def test_5_1_max_actually_changes_the_map_on_close_targets():
    """Guard against a vacuous 5.1: on a config with targets inside Delta* the max
    map must differ from the sum map (else the regression test proves nothing)."""
    case = next(c for c in ring_config_cases() if c["stem"] == "mean_field_2_targets")
    # 60 degrees apart in the standard config: tails only, but not identical.
    a = system_maps(case, sensory_map_reduction="sum")
    b = system_maps(case, sensory_map_reduction="max")
    assert not np.array_equal(a, b)
    assert np.all(b <= a + 1e-15)


# ---------------------------------------------------------------------------
# 5.2 Single target
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("p", [1.0, 2.0, 8.0, 64.0])
def test_5_2_single_target_all_reductions_coincide(p):
    s = _map([0.7], [1.3], "sum")
    m = _map([0.7], [1.3], "max")
    pn = _map([0.7], [1.3], "pnorm", p=p)
    assert np.array_equal(s, m)
    np.testing.assert_allclose(pn, s, rtol=1e-10, atol=0.0)


# ---------------------------------------------------------------------------
# 5.3 Two equal targets, kappa = 20
# ---------------------------------------------------------------------------
def test_5_3_sum_merges_at_20_degrees():
    phi = [_deg(-10.0), _deg(10.0)]       # midpoint at 0
    s = _map(phi, [1.0, 1.0], "sum")
    assert _argmax_set(s) == {int(np.argmin(np.abs(_theta())))}, \
        "sum at 20 degrees must have its unique maximum at the pair midpoint"
    assert _at(s, 0.0) == pytest.approx(2.0 * math.exp(KAPPA * (math.cos(_deg(10)) - 1.0)), rel=1e-6)
    assert _at(s, 0.0) == pytest.approx(1.476, abs=2e-3)
    assert _at(s, _deg(10.0)) == pytest.approx(1.0 + math.exp(KAPPA * (math.cos(_deg(20)) - 1.0)), rel=1e-6)
    assert _at(s, _deg(10.0)) == pytest.approx(1.299, abs=2e-3)


def test_5_3_max_keeps_both_peaks_at_20_degrees():
    phi = [_deg(-10.0), _deg(10.0)]
    m = _map(phi, [1.0, 1.0], "max")
    theta = _theta()
    peaks = _argmax_set(m)
    expected = {int(np.argmin(np.abs(theta - phi[0]))), int(np.argmin(np.abs(theta - phi[1])))}
    assert peaks == expected, "max must peak at both target bearings"
    assert m.max() == pytest.approx(1.0, abs=1e-12)
    assert _at(m, 0.0) == pytest.approx(math.exp(KAPPA * (math.cos(_deg(10)) - 1.0)), rel=1e-6)
    assert _at(m, 0.0) == pytest.approx(0.738, abs=2e-3)


def test_5_3_sum_is_bimodal_at_30_degrees():
    phi = [_deg(-15.0), _deg(15.0)]
    s = _map(phi, [1.0, 1.0], "sum")
    at_target = _at(s, _deg(15.0))
    at_mid = _at(s, 0.0)
    assert at_target == pytest.approx(1.069, abs=2e-3)
    assert at_mid == pytest.approx(1.012, abs=2e-3)
    assert at_target > at_mid


def test_5_3_merge_threshold_between_25_and_30_degrees():
    """kappa sin^2(D/2) < cos(D/2) <=> unimodal; at kappa = 20 the boundary is ~25.5."""
    theta = _theta()
    mid = int(np.argmin(np.abs(theta)))          # the node at 0, the pair midpoint

    def unimodal(delta_deg: float) -> bool:
        """The midpoint is a local maximum, i.e. the two bumps have merged."""
        s = _map([_deg(-delta_deg / 2), _deg(delta_deg / 2)], [1.0, 1.0], "sum")
        return bool(s[mid] >= s[mid + 1] and s[mid] >= s[mid - 1])
    assert unimodal(25.0)
    assert not unimodal(30.0)
    # Exact boundary of kappa sin^2(D/2) = cos(D/2): cos(D/2) = (sqrt(1+4k^2) - 1) / 2k.
    delta_star = math.degrees(2.0 * math.acos(
        (math.sqrt(1.0 + 4.0 * KAPPA ** 2) - 1.0) / (2.0 * KAPPA)))
    assert delta_star == pytest.approx(25.5, abs=0.05)
    assert unimodal(delta_star - 1.0) and not unimodal(delta_star + 1.0)


def test_5_3_sum_and_max_agree_at_60_degrees():
    phi = [_deg(-30.0), _deg(30.0)]
    s = _map(phi, [1.0, 1.0], "sum")
    m = _map(phi, [1.0, 1.0], "max")
    assert np.max(np.abs(s - m)) < 0.15
    assert _argmax_set(s, tol=1e-9) == _argmax_set(m, tol=1e-9)
    # The largest discrepancy is in the tails between the bumps, never at a peak.
    assert abs(_at(s, _deg(30.0)) - _at(m, _deg(30.0))) < 0.08


# ---------------------------------------------------------------------------
# 5.4 Amplitude bound
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("m", [2, 3, 5])
def test_5_4_max_is_bounded_by_the_best_quality(m):
    rng = np.random.default_rng(m)
    for _ in range(20):
        q = rng.uniform(0.0, 1.0, size=m)
        phi = rng.uniform(-np.pi, np.pi, size=m)
        mx = _map(phi, q, "max", n=720)
        assert np.all(mx <= q.max() + 1e-12)
        # The peak sits at the best target's bearing up to the 0.5 degree grid: a node
        # is at most a quarter degree away from any bearing.
        grid_loss = math.exp(KAPPA * (math.cos(math.radians(0.25)) - 1.0))
        assert mx.max() >= q.max() * grid_loss - 1e-12


@pytest.mark.parametrize("m", [2, 3, 5])
def test_5_4_sum_overshoots_the_best_quality_inside_delta_star(m):
    # Exact root of kappa sin^2(D/2) = cos(D/2): cos(D/2) = (sqrt(1 + 4 kappa^2) - 1) / 2 kappa.
    delta_star = 2.0 * math.acos((math.sqrt(1.0 + 4.0 * KAPPA ** 2) - 1.0) / (2.0 * KAPPA))
    assert math.degrees(delta_star) == pytest.approx(25.5, abs=0.05)
    rng = np.random.default_rng(100 + m)
    for _ in range(20):
        q = rng.uniform(0.5, 1.0, size=m)
        q[0] = 1.0                                  # target 0 is the best target
        phi = rng.uniform(-np.pi, np.pi, size=m)
        # Force target 1 inside Delta* of the best target.
        delta = rng.uniform(-delta_star, delta_star)
        phi[1] = _wrap(phi[0] + delta)
        s = _map(phi, q, "sum", n=720)
        # At the best target's own bearing the sum is q_0 + q_1 vM(delta) + (>= 0):
        # the pair's tail lifts the map ABOVE the best quality by a real margin.
        tail = q[1] * math.exp(KAPPA * (math.cos(delta) - 1.0))
        assert tail > 0.05                          # vM(Delta*) = 0.14 at kappa = 20, q_1 >= 0.5
        # 0.9: the 0.5 degree grid may miss the exact bearing by a quarter degree.
        assert s.max() >= q.max() + 0.9 * tail


# ---------------------------------------------------------------------------
# 5.5 pnorm limits
# ---------------------------------------------------------------------------
def test_5_5_pnorm_p1_is_sum_and_p64_is_max():
    rng = np.random.default_rng(55)
    for m in (2, 3):
        q = rng.uniform(0.5, 1.0, size=m)
        phi = rng.uniform(-np.pi, np.pi, size=m)
        s = _map(phi, q, "sum", n=720)
        mx = _map(phi, q, "max", n=720)
        np.testing.assert_allclose(_map(phi, q, "pnorm", p=1.0, n=720), s, rtol=0, atol=1e-12)
        p64 = _map(phi, q, "pnorm", p=64.0, n=720)
        assert np.all(p64 >= mx - 1e-12)
        assert np.all(p64 <= mx * 1.02 + 1e-12)


# ---------------------------------------------------------------------------
# 5.6 Wrap-around
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("reduction", ["sum", "max", "pnorm"])
def test_5_6_wrap_around(reduction):
    straddling = _map([_deg(350.0), _deg(10.0)], [1.0, 0.97], reduction)
    rotated = _map([_deg(170.0), _deg(190.0)], [1.0, 0.97], reduction)
    # Rotating the pair by 180 degrees rotates the map by n/2 grid nodes.
    np.testing.assert_allclose(np.roll(rotated, FINE_N // 2), straddling, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# 5.7 Unequal qualities
# ---------------------------------------------------------------------------
def test_5_7_unequal_qualities_keep_both_peaks_and_lean_the_crossover():
    phi = [0.0, _deg(20.0)]
    q = [1.0, 0.98]
    m = _map(phi, q, "max")
    theta = _theta()
    i0 = int(np.argmin(np.abs(theta - phi[0])))
    i1 = int(np.argmin(np.abs(theta - phi[1])))
    assert m[i0] == pytest.approx(1.0, abs=1e-12)
    assert m[i1] == pytest.approx(0.98, abs=1e-12)
    # Both are local maxima.
    assert m[i0] > m[i0 - 1] and m[i0] > m[i0 + 1]
    assert m[i1] > m[i1 - 1] and m[i1] > m[i1 + 1]
    # Crossover: first node past target 0 where the weaker target's contribution wins.
    vm0 = np.exp(KAPPA * (np.cos(theta - phi[0]) - 1.0))
    vm1 = 0.98 * np.exp(KAPPA * (np.cos(theta - phi[1]) - 1.0))
    between = (theta > phi[0]) & (theta < phi[1])
    cross_idx = int(np.flatnonzero(between & (vm1 > vm0))[0])
    assert theta[cross_idx] > _deg(10.0), "crossover must lie on the weaker target's side"
    assert theta[cross_idx] < _deg(12.0)


# ---------------------------------------------------------------------------
# Section 4: configuration, defaults, metadata, pipeline consistency
# ---------------------------------------------------------------------------
def _standard_case():
    return next(c for c in ring_config_cases() if c["stem"] == "mean_field_2_targets")


def test_4_4_default_is_sum_and_block_is_parsed():
    case = _standard_case()
    _, model = build_model(case)
    assert model.sensory_map_reduction == "sum"
    assert model.mean_field_system.sensory_map_reduction == "sum"
    _, model = build_model(case, block={"reduction": "max"})
    assert model.sensory_map_reduction == "max"
    assert model.mean_field_system.sensory_map_reduction == "max"
    _, model = build_model(case, block={"reduction": "pnorm", "p": 4})
    assert model.mean_field_system.sensory_map_reduction == "pnorm"
    assert model.mean_field_system.sensory_map_p == 4.0


def test_4_4_invalid_blocks_raise():
    from models.mean_field_systems import normalize_sensory_map_config
    with pytest.raises(ValueError, match="reduction"):
        normalize_sensory_map_config({"reduction": "mean"})
    with pytest.raises(ValueError, match="p"):
        normalize_sensory_map_config({"reduction": "pnorm", "p": 0.5})
    with pytest.raises(ValueError):
        normalize_sensory_map_config("max")
    assert normalize_sensory_map_config(None) == ("sum", 8.0)
    assert normalize_sensory_map_config({}) == ("sum", 8.0)
    # p is only READ under pnorm: a nonsense p under sum is ignored, not rejected.
    assert normalize_sensory_map_config({"reduction": "sum", "p": 0.0}) == ("sum", 8.0)
    assert normalize_sensory_map_config({"reduction": "MAX"}) == ("max", 8.0)
    with pytest.raises(ValueError):
        MeanFieldSystem(num_neurons=8, sensory_map_reduction="median")


def test_4_4_reduction_is_stamped_into_run_metadata():
    case = _standard_case()
    agent, model = build_model(case, block={"reduction": "max"})
    objs = _objects_from_case(case)
    model.step(agent, 0, None, objs, {})
    data = model.get_spin_system_data()
    assert data["mean_field_sensory_map_reduction"] == "max"
    assert data["mean_field_sensory_map_p"] == 8.0
    snapshot = model.get_mean_field_data()
    assert snapshot["sensory_map_reduction"] == "max"


@pytest.mark.parametrize("reduction", ["max", "pnorm"])
def test_pipeline_map_matches_reference_function(reduction):
    """compute_sensory_map (noise off) equals the Section 3 reference divided by sqrt(n)."""
    from models.mean_field_systems import sensory_map
    n = 30
    phi = np.array([_deg(-10.0), _deg(10.0), _deg(-120.0)])
    q = np.array([5.0, 5.0, 5.1])
    system = MeanFieldSystem(num_neurons=n, kappa=KAPPA, num_targets=3,
                             sensory_map_reduction=reduction, sensory_map_p=6.0,
                             rng=np.random.default_rng(0), noise_rng=np.random.default_rng(1))
    b = system.compute_sensory_map(num_targets=3, num_guards=0, target_ids=["a", "b", "c"],
                                   target_angles=phi, target_qualities=q, tick=0)
    ref = sensory_map(system.theta, phi, q, KAPPA, reduction=reduction, p=6.0) / math.sqrt(n)
    np.testing.assert_allclose(b, ref, rtol=1e-12, atol=0.0)


def test_pipeline_sum_matches_reference_function_to_rounding():
    from models.mean_field_systems import sensory_map
    n = 30
    phi = np.array([_deg(-10.0), _deg(10.0)])
    q = np.array([5.0, 4.95])
    system = MeanFieldSystem(num_neurons=n, kappa=KAPPA, num_targets=2,
                             rng=np.random.default_rng(0), noise_rng=np.random.default_rng(1))
    b = system.compute_sensory_map(num_targets=2, num_guards=0, target_ids=["a", "b"],
                                   target_angles=phi, target_qualities=q, tick=0)
    ref = sensory_map(system.theta, phi, q, KAPPA, reduction="sum") / math.sqrt(n)
    np.testing.assert_allclose(b, ref, rtol=1e-12, atol=0.0)


def test_guards_are_still_added_on_top_of_the_reduced_map():
    """The reduction replaces the target sum only; the guard term is untouched."""
    n = 30
    phi = np.array([_deg(-10.0), _deg(10.0)])
    q = np.array([1.0, 1.0])
    kw = dict(num_neurons=n, kappa=KAPPA, num_targets=2, num_guards=1,
              rng=np.random.default_rng(0), noise_rng=np.random.default_rng(1))
    guard = dict(guard_angles=[math.pi], guard_qualities=[-0.5], guard_decay_rate=0.0,
                 guard_distances=[0.3])
    plain = MeanFieldSystem(sensory_map_reduction="max", **kw).compute_sensory_map(
        2, 0, ["a", "b"], phi, q)
    with_guard = MeanFieldSystem(sensory_map_reduction="max", **kw).compute_sensory_map(
        2, 1, ["a", "b"], phi, q, **guard)
    theta = np.linspace(-np.pi, np.pi, n, endpoint=False)
    expected_guard = -0.5 * np.exp(KAPPA * (np.cos(theta - math.pi) - 1.0)) / math.sqrt(n)
    np.testing.assert_allclose(with_guard - plain, expected_guard, rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------------------
# 4.5 Seeding: the same trial index gives the same noise stream under sum and max
# ---------------------------------------------------------------------------
def _stream_trace(case, block, ticks=12, seed=7, mf_overrides=None):
    agent, model = build_model(case, seed=seed, block=block, mf_overrides=mf_overrides)
    objs = _objects_from_case(case)
    noisy, qhat = [], []
    for tick in range(ticks):
        model.step(agent, tick, None, objs, {})
        agent.integrate()
        noisy.append(model.mean_field_system.get_noisy_target_qualities())
        qhat.append(dict(model.percept_stream_record().get("sensory_stream_qhat") or {}))
    mf = model.mean_field_system
    return (np.stack(noisy), qhat, json.dumps(mf.rng.bit_generator.state, sort_keys=True),
            json.dumps(mf.noise_rng.bit_generator.state, sort_keys=True))


def test_4_5_legacy_sigma_s_stream_is_paired_across_reductions():
    """Legacy mode, sigma_s > 0: the per-tick quality draws and both generator states
    are identical under sum and max for the same seed, although the trajectories
    (hence the bearings) differ."""
    case = _standard_case()                        # sigma_s = 0.5, legacy stream
    close = {"target_ids": case["mf"]["target_ids"]}
    a = _stream_trace(case, {"reduction": "sum"}, mf_overrides=close)
    b = _stream_trace(case, {"reduction": "max"}, mf_overrides=close)
    assert np.array_equal(a[0], b[0]), "sigma_s draws differ between sum and max"
    assert a[2] == b[2], "sigma_s generator state diverged"
    assert a[3] == b[3], "internal sigma generator state diverged"


def test_4_5_shared_stream_is_paired_across_reductions():
    case = next(c for c in ring_config_cases() if c["stem"] == "qd_sweep_ra_template")
    over = {"u": 6.0, "v": 0.5}
    a = _stream_trace(case, {"reduction": "sum"}, mf_overrides=over)
    b = _stream_trace(case, {"reduction": "max"}, mf_overrides=over)
    assert a[1] == b[1], "shared percept differs between sum and max"
    assert a[3] == b[3], "internal sigma generator state diverged"


def test_4_5_reductions_do_differ_on_close_targets_in_the_closed_loop():
    """Sanity: with the pair moved inside Delta*, sum and max produce different ring
    states from tick 1 — the pairing test above is not passing vacuously."""
    case = _standard_case()
    # Rebuild the case with a 20-degree pair at range 0.5.
    close = dict(case)
    r = 0.5
    close["targets"] = [
        ("static_0.s#0", _deg(-10.0), 5.0, (r * math.cos(_deg(10.0)), r * math.sin(_deg(10.0)), 0.0)),
        ("static_1.s#0", _deg(10.0), 5.0, (r * math.cos(_deg(10.0)), -r * math.sin(_deg(10.0)), 0.0)),
    ]
    over = {"u": 6.0, "v": 0.5, "sigma": 0.0, "sigma_s": 0.0}
    agent_s, model_s = build_model(close, block={"reduction": "sum"}, mf_overrides=over)
    agent_m, model_m = build_model(close, block={"reduction": "max"}, mf_overrides=over)
    objs = _objects_from_case(close)
    model_s.step(agent_s, 0, None, objs, {})
    model_m.step(agent_m, 0, None, objs, {})
    bs = model_s.mean_field_system.get_sensory_map()
    bm = model_m.mean_field_system.get_sensory_map()
    assert not np.array_equal(bs, bm)
    assert bm.max() < bs.max()
