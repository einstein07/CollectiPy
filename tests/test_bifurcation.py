"""Unit tests for BifurcationDetector (02-01).

Tests:
  test_spike_detection               — local-maximum above threshold fires one event
  test_no_spike_below_threshold      — local-maximum below threshold does not fire
  test_no_spike_monotonic            — monotonically decreasing sequence never fires
  test_spike_min_separation          — second spike within separation window suppressed
  test_retriggerable                 — second spike outside window fires
  test_jacobian_computation_standard — Jacobian is finite ndarray for g_adapt=0
  test_jacobian_computation_sfa      — Jacobian is finite ndarray for g_adapt>0, differs from standard
  test_target_assignment             — nearest target resolves correctly
  test_event_dict_schema             — event dict has required keys with correct types
  test_no_bifurcation_empty_list     — flat sequence produces empty events list
"""
import math
import numpy as np
import pytest

from models.bifurcation import BifurcationDetector
from models.mean_field_systems import MeanFieldSystem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_detector(
    agent_name: str = "test_agent",
    lambda_threshold: float = -0.1,
    spike_min_separation: int = 10,
) -> BifurcationDetector:
    return BifurcationDetector(
        agent_name=agent_name,
        lambda_threshold=lambda_threshold,
        spike_min_separation=spike_min_separation,
    )


def feed_sequence(detector: BifurcationDetector, lambda_values: list[float], start_tick: int = 0):
    """Feed synthetic lambda1 values directly into the detector buffer.

    Uses the internal _check_spike() method to bypass compute_lambda1(), allowing
    unit testing of the spike-detection logic with synthetic sequences.
    """
    events = []
    target_angles = [0.0, math.pi]
    target_ids = ["A", "B"]
    bump_angle = 0.0
    for i, v in enumerate(lambda_values):
        event = detector._check_spike(
            tick=start_tick + i,
            lambda1=v,
            bump_angle=bump_angle,
            target_angles=target_angles,
            target_ids=target_ids,
        )
        if event is not None:
            events.append(event)
    return events


def make_standard_mf(num_neurons: int = 20) -> MeanFieldSystem:
    return MeanFieldSystem(
        num_neurons=num_neurons,
        u=6.0,
        beta=1.0,
        sigma=0.0,
        dt=0.1,
        integration_time=5.0,
        num_targets=1,
        rng=np.random.default_rng(42),
    )


def make_sfa_mf(num_neurons: int = 20) -> MeanFieldSystem:
    return MeanFieldSystem(
        num_neurons=num_neurons,
        u=6.0,
        beta=1.0,
        sigma=0.0,
        dt=0.1,
        integration_time=5.0,
        num_targets=1,
        g_adapt=0.5,
        tau_adapt=10.0,
        rng=np.random.default_rng(42),
    )


# ---------------------------------------------------------------------------
# Spike detection tests (use synthetic lambda sequences via _check_spike)
# ---------------------------------------------------------------------------

def test_spike_detection():
    """Local max at index 3 above threshold fires exactly one event."""
    # lambda sequence: [-2.0, -1.5, -0.8, -0.05, -0.3, -0.8]
    # local max at index 3: value -0.05, prev=-0.8, next=-0.3 -> spike
    detector = make_detector()
    seq = [-2.0, -1.5, -0.8, -0.05, -0.3, -0.8]
    events = feed_sequence(detector, seq)
    assert len(events) == 1, f"Expected 1 event, got {len(events)}: {events}"
    # The peak is at tick 3 (index 3)
    assert events[0]["tick"] == 3


def test_no_spike_below_threshold():
    """Local max at -0.5 is below threshold -0.1. No event fires."""
    detector = make_detector(lambda_threshold=-0.1)
    # local max at index 2 (-0.5), which is below -0.1
    seq = [-2.0, -1.5, -0.5, -1.0, -1.5]
    events = feed_sequence(detector, seq)
    assert len(events) == 0, f"Expected 0 events, got {len(events)}: {events}"


def test_no_spike_monotonic():
    """Monotonically decreasing sequence never has a local max. No event fires."""
    detector = make_detector()
    seq = [-0.05, -0.1, -0.2, -0.3]
    events = feed_sequence(detector, seq)
    assert len(events) == 0, f"Expected 0 events, got {len(events)}: {events}"


def test_spike_min_separation():
    """Second spike within spike_min_separation=10 ticks is suppressed."""
    detector = make_detector(spike_min_separation=10)
    # First spike at tick 2 (local max), second spike at tick 5 (too close)
    # Build: prev < curr > next at ticks 2 and 5
    # tick 0: -2.0, tick 1: -1.0, tick 2: -0.05, tick 3: -1.0,
    # tick 4: -1.0, tick 5: -0.05, tick 6: -1.0
    seq = [-2.0, -1.0, -0.05, -1.0, -1.0, -0.05, -1.0]
    events = feed_sequence(detector, seq)
    assert len(events) == 1, f"Expected 1 event (second suppressed), got {len(events)}: {events}"
    assert events[0]["tick"] == 2


def test_retriggerable():
    """Two spikes separated by more than spike_min_separation both fire."""
    detector = make_detector(spike_min_separation=5)
    # First spike at tick 2, second spike at tick 9
    # separation = 9 - 2 = 7 > 5 -> both should fire
    seq = [-2.0, -1.0, -0.05, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.05, -1.0]
    events = feed_sequence(detector, seq)
    assert len(events) == 2, f"Expected 2 events (both fire), got {len(events)}: {events}"
    assert events[0]["tick"] == 2
    assert events[1]["tick"] == 9


# ---------------------------------------------------------------------------
# Jacobian computation tests (require MeanFieldSystem instances)
# ---------------------------------------------------------------------------

def test_jacobian_computation_standard():
    """Jacobian for standard model (g_adapt=0) is a finite (n x n) ndarray."""
    mf = make_standard_mf(num_neurons=20)
    # Run a few steps with a target to get non-trivial state
    mf.step(target_angles=[0.0], target_qualities=[1.0])
    mf.step(target_angles=[0.0], target_qualities=[1.0])

    detector = make_detector()
    J = detector.compute_jacobian(mf)

    assert isinstance(J, np.ndarray), f"Expected ndarray, got {type(J)}"
    assert J.shape == (20, 20), f"Expected shape (20, 20), got {J.shape}"
    assert np.all(np.isfinite(J)), "Jacobian contains non-finite values"


def test_jacobian_computation_sfa():
    """Jacobian for SFA model (g_adapt>0) is finite and differs from standard model."""
    num_neurons = 20
    mf_std = make_standard_mf(num_neurons=num_neurons)
    mf_sfa = make_sfa_mf(num_neurons=num_neurons)

    # Run both with the same target to get non-trivial, comparable states
    for _ in range(3):
        mf_std.step(target_angles=[0.0], target_qualities=[1.0])
        mf_sfa.step(target_angles=[0.0], target_qualities=[1.0])

    detector = make_detector()
    J_std = detector.compute_jacobian(mf_std)
    J_sfa = detector.compute_jacobian(mf_sfa)

    assert isinstance(J_sfa, np.ndarray), f"Expected ndarray, got {type(J_sfa)}"
    assert J_sfa.shape == (num_neurons, num_neurons), f"Expected shape ({num_neurons}, {num_neurons})"
    assert np.all(np.isfinite(J_sfa)), "SFA Jacobian contains non-finite values"
    # SFA adapt_ring is non-zero after a few steps (g_adapt > 0), so J_sfa should differ
    assert not np.allclose(J_std, J_sfa), (
        "SFA Jacobian should differ from standard Jacobian when adapt_ring is non-zero"
    )


# ---------------------------------------------------------------------------
# Target assignment tests
# ---------------------------------------------------------------------------

def test_target_assignment():
    """_nearest_target returns correct target ID based on angular proximity."""
    # Target A at 0.0, target B at pi
    target_angles = [0.0, math.pi]
    target_ids = ["0", "1"]

    # bump near 0 -> target "0"
    result = BifurcationDetector._nearest_target(0.1, target_angles, target_ids)
    assert result == "0", f"Expected target '0', got '{result}'"

    # bump near pi -> target "1"
    result = BifurcationDetector._nearest_target(3.0, target_angles, target_ids)
    assert result == "1", f"Expected target '1', got '{result}'"


# ---------------------------------------------------------------------------
# Event schema test
# ---------------------------------------------------------------------------

def test_event_dict_schema():
    """Detected event dict has required keys with correct types."""
    detector = make_detector(agent_name="agent_42")
    # Feed a spike: prev < curr > next AND curr > threshold
    seq = [-1.0, -0.05, -1.0]
    events = feed_sequence(detector, seq, start_tick=10)
    assert len(events) == 1, f"Expected 1 event, got {len(events)}"
    event = events[0]

    assert set(event.keys()) == {"agent", "tick", "lambda1", "target"}, (
        f"Event keys mismatch: {set(event.keys())}"
    )
    assert isinstance(event["agent"], str), f"'agent' should be str, got {type(event['agent'])}"
    assert isinstance(event["tick"], int), f"'tick' should be int, got {type(event['tick'])}"
    assert isinstance(event["lambda1"], float), f"'lambda1' should be float, got {type(event['lambda1'])}"
    assert isinstance(event["target"], str), f"'target' should be str, got {type(event['target'])}"


# ---------------------------------------------------------------------------
# No-bifurcation (flat sequence)
# ---------------------------------------------------------------------------

def test_no_bifurcation_empty_list():
    """Flat lambda sequence (all -1.0) produces an empty events list."""
    detector = make_detector()
    seq = [-1.0] * 20
    events = feed_sequence(detector, seq)
    assert events == [], f"Expected empty events list, got {events}"
    assert detector.events == [], f"Expected empty detector.events, got {detector.events}"
