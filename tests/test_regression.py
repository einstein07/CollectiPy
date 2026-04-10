"""Regression snapshot tests (TEST-03).

Outputs of SpinSystem and movement utility functions must match reference .npy files
committed in tests/fixtures/. Run with --regen to regenerate snapshots after a
deliberate behaviour change.
"""
import math
from pathlib import Path
from random import Random

import numpy as np
import pytest

from models.spinsystem import SpinSystem
from models.utils import levy

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _make_spinsystem():
    """Create a SpinSystem with a fixed seed for reproducible output."""
    rng = Random(42)
    return SpinSystem(
        random_generator=rng,
        num_groups=4,
        num_spins_per_group=5,
        T=1.0,
        J=1.0,
        nu=0.5,
        p_spin_up=0.5,
        time_delay=1,
        dynamics="metropolis",
    )


def test_spinsystem_spins_snapshot(request):
    """SpinSystem spins after 100 steps must match the committed reference snapshot.

    This test locks the exact binary spin state produced by the Metropolis
    dynamics so that refactors cannot silently alter the Ising model update rule.
    """
    ss = _make_spinsystem()
    ss.run_spins(steps=100)
    result = ss.spins.copy()
    fixture_path = FIXTURES_DIR / "spinsystem_spins.npy"
    if request.config.getoption("--regen") or not fixture_path.exists():
        np.save(fixture_path, result)
    reference = np.load(fixture_path)
    np.testing.assert_array_equal(result, reference)


def test_spinsystem_avg_direction_snapshot(request):
    """SpinSystem average_direction_of_activity after 100 steps must match snapshot.

    The result may be None (encoded as NaN in the fixture) when all spins are
    identical. The test handles both finite and NaN reference values.
    """
    ss = _make_spinsystem()
    ss.run_spins(steps=100)
    result = ss.average_direction_of_activity()
    fixture_path = FIXTURES_DIR / "spinsystem_avg_direction.npy"
    # Store as a 1-element array; NaN represents None
    result_arr = np.array([result if result is not None else float("nan")])
    if request.config.getoption("--regen") or not fixture_path.exists():
        np.save(fixture_path, result_arr)
    reference = np.load(fixture_path)
    if math.isnan(reference[0]):
        assert result is None or (result is not None and math.isnan(result))
    else:
        np.testing.assert_allclose(result_arr, reference, rtol=1e-6)


def test_random_walk_levy_snapshot(request):
    """levy() samples with a fixed seed must match the committed reference snapshot.

    Locks the Lévy distribution sampling implementation so that changes to
    models.utils.levy() are caught immediately by this test.
    """
    rng = Random(123)
    samples = np.array([levy(rng, c=1.0, alpha=1.5) for _ in range(200)])
    fixture_path = FIXTURES_DIR / "random_walk_state.npy"
    if request.config.getoption("--regen") or not fixture_path.exists():
        np.save(fixture_path, samples)
    reference = np.load(fixture_path)
    np.testing.assert_allclose(samples, reference, rtol=1e-6)
