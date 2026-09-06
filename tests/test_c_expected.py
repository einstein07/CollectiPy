# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2026 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""`c_expected`: the noise scale the pure-DDM policy assumes, decoupled from the noise
the evidence carries (the noise-side counterpart of `A_expected`).

  test_default_is_the_physical_scale     c_expected None -> c_assumed == c
  test_validation                        non-positive / non-finite values are rejected
  test_log_odds_use_the_assumed_scale    implied_log_odds = 2 A |x| / c_assumed^2
  test_noise_is_untouched                same seed -> identical q_hat with and without c_expected
  test_policy_equals_a_sensor_with_that_noise
                                         a policy that ASSUMES c~ on a sensor with true c sets the
                                         same threshold as a correctly specified policy on a sensor
                                         whose true scale IS c~ (bayes_risk, closed form)
  test_record_carries_both_scales        the per-tick record reports c and c_assumed

Run with:
    cd CollectiPy && env -u PYTHONPATH .venv/bin/python -m pytest tests/test_c_expected.py -q
"""

import math
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "src")
for _p in (_SRC, os.path.join(_HERE, "..")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.ddm_systems import DriftDiffusionSystem            # noqa: E402
from tests.test_shared_sensory_stream import build, run_percepts, _targets   # noqa: E402

ETA = 0.5                       # per-target noise rate the evidence really has
C_TRUE = math.sqrt(2.0) * ETA
FACTOR = 1.6                    # the policy believes the sensor is 1.6x noisier


def _system(**kw):
    return DriftDiffusionSystem(eta_rate=(ETA, ETA), threshold_policy="manual", z_manual=1.0,
                                A_expected=0.25, rng=np.random.default_rng(0), **kw)


def test_default_is_the_physical_scale():
    d = _system()
    assert d.c_expected is None
    assert d.c == pytest.approx(C_TRUE)
    assert d.c_assumed == pytest.approx(d.c)


def test_validation():
    d = _system(c_expected=FACTOR * C_TRUE)
    assert d.c == pytest.approx(C_TRUE)                 # the physical scale is untouched
    assert d.c_assumed == pytest.approx(FACTOR * C_TRUE)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            _system(c_expected=bad)


def test_log_odds_use_the_assumed_scale():
    d = _system(c_expected=FACTOR * C_TRUE)
    d.resolve_ensemble_A(0.25) if hasattr(d, "resolve_ensemble_A") else None
    d.x = 0.7
    A = abs(float(d.A_hat))
    assert d.implied_log_odds == pytest.approx(2.0 * A * 0.7 / (FACTOR * C_TRUE) ** 2)


def _overrides(**extra):
    base = dict(eta_rate=[ETA, ETA], threshold_policy="bayes_risk", boundary_mode="static",
                cost_ratio=20.0, A_expected=0.25, n_sub=1)
    base.update(extra)
    return base


def test_noise_is_untouched():
    plain = run_percepts("embodied_pure_ddm", seed=3, ticks=8, **_overrides())
    wrong = run_percepts("embodied_pure_ddm", seed=3, ticks=8, **_overrides(c_expected=FACTOR * C_TRUE))
    assert len(plain) == len(wrong) == 8
    for a, b in zip(plain, wrong):
        assert a.keys() == b.keys()
        for k in a:
            assert a[k] == pytest.approx(b[k])          # the evidence stream does not know about the belief


def _threshold_after_one_step(**overrides):
    agent, model = build("embodied_pure_ddm", seed=1, **overrides)
    model.step(agent, 0, None, _targets(), {})
    return float(model.ddm.z0), float(model.ddm.c), float(model.ddm.c_assumed)


def test_policy_equals_a_sensor_with_that_noise():
    # misspecified policy on the true sensor ...
    z_wrong, c_phys, c_ass = _threshold_after_one_step(**_overrides(c_expected=FACTOR * C_TRUE))
    # ... versus a correctly specified policy on a sensor whose noise really is FACTOR x larger
    z_ref, c_ref, c_ref_ass = _threshold_after_one_step(**_overrides(eta_rate=[FACTOR * ETA, FACTOR * ETA]))
    # ... versus the correctly specified policy on the true sensor
    z_true, _, _ = _threshold_after_one_step(**_overrides())
    assert c_phys == pytest.approx(C_TRUE) and c_ass == pytest.approx(FACTOR * C_TRUE)
    assert c_ref == pytest.approx(FACTOR * C_TRUE) and c_ref_ass == pytest.approx(c_ref)
    assert z_wrong == pytest.approx(z_ref, rel=1e-9)
    assert z_wrong != pytest.approx(z_true, rel=1e-3)   # and it IS a different policy from the truth


def test_record_carries_both_scales():
    agent, model = build("embodied_pure_ddm", seed=1, **_overrides(c_expected=FACTOR * C_TRUE))
    model.step(agent, 0, None, _targets(), {})
    rec = model.get_spin_system_data()
    assert rec["pure_ddm_c"] == pytest.approx(C_TRUE)
    assert rec["pure_ddm_c_assumed"] == pytest.approx(FACTOR * C_TRUE)
