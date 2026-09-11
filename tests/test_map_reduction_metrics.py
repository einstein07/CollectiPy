# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Unit tests for the pure functions behind the sensory-map reduction experiment
(scripts/map_reduction): the 6.4 outcome classification, the ring-shape test, the
sub-tick arrival refinement, and the design geometry.

Run with:
    env -u PYTHONPATH .venv/bin/python -m pytest tests/test_map_reduction_metrics.py -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT / "scripts"), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from map_reduction import config_patch, factors, metrics  # noqa: E402


# ---------------------------------------------------------------------------
# classify (6.4)
# ---------------------------------------------------------------------------
def test_classify_prefers_the_midpoint_when_closer_than_either_pair_member():
    bearings = {"A": 0.0, "B": 20.0, "C": -120.0}
    label, det = metrics.classify(9.0, bearings, tol_deg=5.0)
    assert label == "mean_of_pair"
    assert det["midpoint_deg"] == pytest.approx(10.0)


def test_classify_nearest_target_otherwise_and_none_outside_tolerance():
    bearings = {"A": 0.0, "B": 20.0, "C": -120.0}
    assert metrics.classify(1.5, bearings, tol_deg=5.0)[0] == "A"
    assert metrics.classify(18.0, bearings, tol_deg=5.0)[0] == "B"
    assert metrics.classify(-118.0, bearings, tol_deg=5.0)[0] == "C"
    assert metrics.classify(60.0, bearings, tol_deg=5.0)[0] is None
    # Without a tolerance every angle gets a class.
    assert metrics.classify(60.0, bearings, tol_deg=None)[0] == "B"


def test_classify_is_wrap_safe():
    bearings = {"A": 170.0, "B": -170.0, "C": 60.0}      # pair straddles +-180
    label, det = metrics.classify(179.0, bearings, tol_deg=5.0)
    assert label == "mean_of_pair"
    assert abs(metrics.wrap_deg(det["midpoint_deg"] - 180.0)) < 1e-9


def test_classify_at_the_neuron_spacing_needs_the_midpoint_rule():
    # Delta = 12: the midpoint is 6 degrees from each target; a CoM 2 degrees from A
    # is within 5 degrees of BOTH A and the midpoint and must go to A.
    bearings = {"A": 0.0, "B": 12.0, "C": -120.0}
    assert metrics.classify(2.0, bearings, tol_deg=5.0)[0] == "A"
    assert metrics.classify(5.0, bearings, tol_deg=5.0)[0] == "mean_of_pair"


# ---------------------------------------------------------------------------
# ring_peaks: one merged bump vs two bumps
# ---------------------------------------------------------------------------
def _bump(theta, centre_deg, width_deg=20.0):
    d = (theta - math.radians(centre_deg) + np.pi) % (2 * np.pi) - np.pi
    return np.exp(-0.5 * (d / math.radians(width_deg)) ** 2)


def test_ring_peaks_counts_one_merged_bump_and_two_separate_ones():
    theta = np.linspace(-np.pi, np.pi, 30, endpoint=False)
    merged = _bump(theta, 10.0)
    assert len(metrics.ring_peaks(merged)) == 1
    two = _bump(theta, 0.0) + _bump(theta, 60.0)
    assert len(metrics.ring_peaks(two)) == 2
    # A weak third bump below half the maximum does not count.
    three = two + 0.3 * _bump(theta, -120.0)
    assert len(metrics.ring_peaks(three)) == 2
    assert metrics.ring_peaks(np.zeros(30)) == []


# ---------------------------------------------------------------------------
# first_arrival: sub-tick refinement
# ---------------------------------------------------------------------------
def test_first_arrival_interpolates_the_crossing():
    targets = {"A": (0.5, 0.0)}
    positions = [(0, 0.0, 0.0), (1, 0.40, 0.0), (2, 0.48, 0.0)]
    tick, fine, hit = metrics.first_arrival(positions, targets, radius=0.05)
    assert (tick, hit) == (2, "A")
    # The circle |x - 0.5| = 0.05 is crossed at x = 0.45, i.e. 5/8 of the way.
    assert fine == pytest.approx(1.0 + 0.05 / 0.08)


def test_first_arrival_none_when_never_inside():
    assert metrics.first_arrival([(0, 0.0, 0.0), (1, 0.1, 0.0)], {"A": (0.5, 0.0)}, 0.05) == (None, None, None)


# ---------------------------------------------------------------------------
# geometry and config
# ---------------------------------------------------------------------------
def test_positions_follow_the_gps_bearing_convention():
    for b in (0.0, 20.0, -120.0, 60.0):
        x, y, _ = factors.position_for_bearing(b)
        assert math.degrees(math.atan2(-y, x)) == pytest.approx(b, abs=1e-4)
        assert math.hypot(x, y) == pytest.approx(factors.TARGET_RANGE, abs=1e-5)


def test_merge_threshold_matches_the_spec():
    assert factors.merge_threshold_deg(20.0) == pytest.approx(25.5, abs=0.05)


def test_cell_ids_are_a_bijection_in_canonical_order():
    seen = []
    for cell in factors.iter_cells():
        assert factors.cell_id(cell["reduction"], cell["delta_close_deg"], cell["q_c_rel"]) == cell["cell_id"]
        seen.append(cell["cell_id"])
    assert seen == list(range(factors.N_CELLS))
    assert factors.N_CELLS == 42


def test_cell_config_applies_the_locked_settings_and_the_reduction():
    template = config_patch.load_template()
    cell = {"cell_id": 0, "reduction": "max", "delta_close_deg": 20, "q_c_rel": 1.02}
    cfg = config_patch.cell_config(cell, template=template)
    env = cfg["environment"]
    mf = env["agents"]["movable_0"]["mean_field_model"]
    assert mf["sensory_map"] == {"reduction": "max"}
    assert (mf["u"], mf["v"], mf["sigma"], mf["sigma_s"]) == (6.0, 0.5, 0.1, 0.0)
    assert mf["num_targets"] == 3 and mf["target_ids"] == list(factors.TARGET_IDS.values())
    assert env["termination"]["target_ids"] == list(factors.TARGET_IDS.values())
    assert env["objects"]["static_2"]["strength"] == [pytest.approx(5.1)]
    assert env["objects"]["static_1"]["position"][0][1] == pytest.approx(-0.5 * math.sin(math.radians(20)), abs=1e-5)
    assert env["time_limit"] == factors.T_MAX_TICKS
    assert env["sensory_stream"]["mode"] == "shared" and env["sensory_stream"]["seed"] is None
    # Per-trial fields: paired seed in BOTH roles, and the reduction in the metadata.
    trial = config_patch.trial_config(cell, 3, factors.seed_for(3), "/tmp/x", cell_cfg=cfg)
    assert trial["environment"]["sensory_stream"]["seed"] == factors.BASE_SEED + 3
    assert all(a["random_seed"] == factors.BASE_SEED + 3 for a in trial["environment"]["arenas"].values())
    assert trial["environment"]["results"]["sweep_metadata"]["sensory_map"] == {"reduction": "max"}
    # The cell hash ignores per-trial fields.
    assert config_patch.config_hash(trial) == config_patch.config_hash(cfg)


def test_sanity_config_keeps_the_template_geometry():
    template = config_patch.load_template()
    cfg = config_patch.sanity_config("sum", template=template, sigma=None)
    mf = cfg["environment"]["agents"]["movable_0"]["mean_field_model"]
    assert mf["sigma"] == template["environment"]["agents"]["movable_0"]["mean_field_model"]["sigma"]
    assert mf["sensory_map"] == {"reduction": "sum"}
    assert [o["strength"][0] for o in cfg["environment"]["objects"].values()] == [5.0, 4.95]
