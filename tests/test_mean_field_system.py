"""Unit tests for MeanFieldSystem (TEST-01).

Tests:
  test_fixed_point_zero_input       — zero input keeps state bounded near zero
  test_bump_forms_from_uniform_ic   — single strong target creates non-uniform bump
  test_single_target_response       — bump angle converges within pi/8 of target
  test_divergence_raises_runtime_error — oversized dt triggers RuntimeError (D-01..D-03)
"""
import numpy as np
import pytest
from models.mean_field_systems import MeanFieldSystem, compute_center_of_mass


def test_fixed_point_zero_input():
    """Zero sensory input keeps the ring-attractor state bounded near zero.

    With no targets, no noise, and zero initial conditions, the system is at the
    trivial fixed point z=0. The state should remain bounded (norm << 1) after
    many integration ticks.
    """
    mf = MeanFieldSystem(
        num_neurons=50,
        sigma=0.0,
        dt=0.1,
        integration_time=50.0,
        u=6.0,
        beta=1.0,
        rng=np.random.default_rng(42),
    )
    mf.run(steps=20)
    assert np.linalg.norm(mf.neural_ring) < 1.0, (
        f"Expected norm < 1.0 at zero-input fixed point, got {np.linalg.norm(mf.neural_ring):.4f}"
    )


def test_bump_forms_from_uniform_ic():
    """A single strong target breaks the uniform initial condition into a localised bump.

    Starting from a near-uniform (non-zero) initial state, a strong sensory signal
    at 0 radians should drive the ring attractor into a non-uniform state with a
    clear peak near the target direction.
    """
    mf = MeanFieldSystem(
        num_neurons=50,
        sigma=0.0,
        dt=0.1,
        integration_time=50.0,
        u=6.0,
        beta=1.0,
        num_targets=1,
        kappa=20.0,
        rng=np.random.default_rng(0),
    )
    mf.neural_ring = np.ones(50) * 0.01
    mf.run(steps=30, target_angles=[0.0], target_qualities=[1.0])
    spread = np.max(mf.neural_ring) - np.min(mf.neural_ring)
    assert spread > 0.1, (
        f"Expected peak-to-trough spread > 0.1 (localised bump), got {spread:.4f}"
    )


def test_single_target_response():
    """The bump angle converges within pi/8 of the target direction.

    After sufficient integration with a target at pi/4 radians, the centre-of-mass
    of ring activity should point within 22.5 degrees of the target.
    """
    target_angle = np.pi / 4
    mf = MeanFieldSystem(
        num_neurons=100,
        sigma=0.0,
        dt=0.1,
        integration_time=50.0,
        u=6.0,
        beta=1.0,
        num_targets=1,
        kappa=20.0,
        rng=np.random.default_rng(7),
    )
    mf.run(steps=50, target_angles=[target_angle], target_qualities=[1.0])
    angle = compute_center_of_mass(mf.neural_ring, mf.theta)
    # Wrap angular difference into [-pi, pi]
    diff = abs((angle - target_angle + np.pi) % (2 * np.pi) - np.pi)
    assert diff < np.pi / 8, (
        f"Bump angle {angle:.4f} rad is more than pi/8 from target {target_angle:.4f} rad "
        f"(diff={diff:.4f})"
    )


def test_divergence_raises_runtime_error():
    """RuntimeError is raised with full context when ODE integration diverges (D-01..D-03).

    Using an unstable step size (dt=10.0, beta=0.0) with a large initial state, the
    Euler method diverges exponentially. The divergence guard in step() must detect
    the resulting NaN/Inf values and raise RuntimeError before the caller sees corrupt
    state.
    """
    mf = MeanFieldSystem(
        num_neurons=50,
        sigma=0.0,
        dt=10.0,
        beta=0.0,
        u=6.0,
        integration_time=10.0,
        rng=np.random.default_rng(0),
    )
    mf.neural_ring = np.ones(50) * 10.0
    with pytest.raises(RuntimeError, match="MeanFieldSystem diverged at tick"):
        for _ in range(1000):
            mf.step()
