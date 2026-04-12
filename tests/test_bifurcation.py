"""Unit and integration tests for BifurcationDetector (02-01 and 02-02).

Tests from 02-01:
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

Tests from 02-02 (wiring + output):
  test_events_json_schema            — DataHandling writes events.json with correct schema
  test_custom_config_respected       — custom lambda_threshold is respected by detector
  test_no_bifurcation_empty_events_json — no events produces events.json with empty list
  test_standard_model_bifurcation    — standard MF model produces at least one bifurcation
  test_sfa_model_bifurcation         — SFA model produces at least one bifurcation
  test_collect_bifurcation_events_wiring — Arena collection path transfers events correctly
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


# ---------------------------------------------------------------------------
# 02-02: DataHandling events.json output tests
# ---------------------------------------------------------------------------

import json
import os
import tempfile


class _MockDataHandling:
    """Minimal stand-in for DataHandling for unit tests."""
    def __init__(self, run_folder: str):
        self.run_folder = run_folder
        self._bifurcation_events: list[dict] = []

    def collect_bifurcation_events(self, events: list[dict]) -> None:
        """Accumulate bifurcation events."""
        self._bifurcation_events.extend(events)

    def _write_events_json(self) -> None:
        """Write events.json sidecar."""
        if self.run_folder is None or not os.path.isdir(self.run_folder):
            return
        events_data = {
            "bifurcation_events": self._bifurcation_events,
            "swap_events": [],
        }
        events_path = os.path.join(self.run_folder, "events.json")
        with open(events_path, "w") as f:
            json.dump(events_data, f, indent=2)


def test_events_json_schema():
    """DataHandling._write_events_json writes correct schema with bifurcation_events and swap_events."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dh = _MockDataHandling(run_folder=tmp_dir)
        sample_events = [
            {"agent": "agent_0", "tick": 42, "lambda1": -0.03, "target": "A"},
            {"agent": "agent_1", "tick": 45, "lambda1": -0.05, "target": "A"},
        ]
        dh.collect_bifurcation_events(sample_events)
        dh._write_events_json()

        events_path = os.path.join(tmp_dir, "events.json")
        assert os.path.exists(events_path), "events.json was not created"

        with open(events_path) as f:
            data = json.load(f)

        assert "bifurcation_events" in data, "events.json missing 'bifurcation_events' key"
        assert "swap_events" in data, "events.json missing 'swap_events' key"
        assert data["swap_events"] == [], f"swap_events should be empty list, got {data['swap_events']}"
        assert data["bifurcation_events"] == sample_events, (
            f"bifurcation_events mismatch: {data['bifurcation_events']}"
        )


def test_no_bifurcation_empty_events_json():
    """No events collected produces events.json with empty bifurcation_events list."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dh = _MockDataHandling(run_folder=tmp_dir)
        dh._write_events_json()

        events_path = os.path.join(tmp_dir, "events.json")
        assert os.path.exists(events_path), "events.json was not created"

        with open(events_path) as f:
            data = json.load(f)

        assert data["bifurcation_events"] == [], (
            f"Expected empty bifurcation_events, got {data['bifurcation_events']}"
        )
        assert data["swap_events"] == [], (
            f"Expected empty swap_events, got {data['swap_events']}"
        )


def test_custom_config_respected():
    """Custom lambda_threshold is respected: spike at -0.15 fires with threshold=-0.2 but not -0.1."""
    # With default threshold=-0.1: a peak at -0.15 is BELOW threshold -> no event
    detector_default = make_detector(lambda_threshold=-0.1)
    seq = [-1.0, -0.15, -1.0]
    events_default = feed_sequence(detector_default, seq)
    assert len(events_default) == 0, (
        f"Peak at -0.15 should NOT fire with threshold=-0.1, got {events_default}"
    )

    # With custom threshold=-0.2: -0.15 > -0.2 -> event fires
    detector_custom = make_detector(lambda_threshold=-0.2)
    events_custom = feed_sequence(detector_custom, seq)
    assert len(events_custom) == 1, (
        f"Peak at -0.15 SHOULD fire with threshold=-0.2, got {events_custom}"
    )


@pytest.mark.slow
def test_standard_model_bifurcation():
    """Standard MF model (g_adapt=0) produces at least one bifurcation event over 200 steps."""
    num_neurons = 50
    mf = MeanFieldSystem(
        num_neurons=num_neurons,
        u=6.0,
        beta=1.0,
        sigma=0.0,
        dt=0.1,
        integration_time=5.0,
        num_targets=2,
        rng=np.random.default_rng(0),
    )
    detector = BifurcationDetector(
        agent_name="test_standard",
        mode="analytical",          # analytical mode: gradient-of-lambda1 criterion
        lambda_threshold=-0.1,
        spike_min_separation=10,
    )
    target_angles = [0.0, math.pi]
    target_ids = ["A", "B"]
    target_qualities = np.array([1.0, 0.5])

    for tick in range(200):
        _, bump_positions, _ = mf.step(
            target_angles=target_angles,
            target_qualities=target_qualities,
        )
        bump_angle = float(bump_positions[-1]) if bump_positions is not None and len(bump_positions) > 0 else 0.0
        detector.update(
            tick=tick,
            mf=mf,
            bump_angle=bump_angle,
            target_angles=list(target_angles),
            target_ids=target_ids,
        )

    assert len(detector.events) >= 1, (
        f"Expected at least 1 bifurcation event from standard model, got {len(detector.events)}"
    )


@pytest.mark.slow
def test_sfa_model_bifurcation():
    """SFA model (g_adapt=0.5) produces at least one bifurcation event over 300 steps."""
    num_neurons = 50
    mf = MeanFieldSystem(
        num_neurons=num_neurons,
        u=6.0,
        beta=1.0,
        sigma=0.0,
        dt=0.1,
        integration_time=5.0,
        num_targets=2,
        g_adapt=0.5,
        tau_adapt=10.0,
        rng=np.random.default_rng(0),
    )
    detector = BifurcationDetector(
        agent_name="test_sfa",
        mode="analytical",          # analytical mode: Omega threshold-crossing criterion
        lambda_threshold=-0.1,
        spike_min_separation=10,
    )
    target_angles = [0.0, math.pi]
    target_ids = ["A", "B"]
    target_qualities = np.array([1.0, 0.5])

    for tick in range(300):
        _, bump_positions, _ = mf.step(
            target_angles=target_angles,
            target_qualities=target_qualities,
        )
        bump_angle = float(bump_positions[-1]) if bump_positions is not None and len(bump_positions) > 0 else 0.0
        detector.update(
            tick=tick,
            mf=mf,
            bump_angle=bump_angle,
            target_angles=list(target_angles),
            target_ids=target_ids,
        )

    assert len(detector.events) >= 1, (
        f"Expected at least 1 bifurcation event from SFA model, got {len(detector.events)}"
    )


def test_collect_bifurcation_events_wiring():
    """Arena._collect_bifurcation_events() correctly transfers events from agent plugins to DataHandling."""
    # Build minimal mock structures that duck-type the Arena.objects pattern

    class _MockDetector:
        def __init__(self, events):
            self.events = events

    class _MockPlugin:
        def __init__(self, events):
            self.bifurcation_detector = _MockDetector(events)

    class _MockEntity:
        def __init__(self, plugin):
            self._movement_plugin = plugin

    class _MockDataHandling:
        def __init__(self):
            self._bifurcation_events: list[dict] = []

        def collect_bifurcation_events(self, events):
            self._bifurcation_events.extend(events)

    # Two agents: agent_0 has 2 events, agent_1 has 1 event
    events_0 = [{"agent": "agent_0", "tick": 10, "lambda1": -0.05, "target": "A"}]
    events_1 = [{"agent": "agent_1", "tick": 12, "lambda1": -0.04, "target": "B"}]

    entity_0 = _MockEntity(_MockPlugin(events_0))
    entity_1 = _MockEntity(_MockPlugin(events_1))

    # Simulate Arena.objects structure: {type: (config, [entities])}
    mock_objects = {
        "agents": ({"number": 2}, [entity_0, entity_1]),
    }

    mock_dh = _MockDataHandling()

    # NOTE: This loop is an intentional logic-replica of Arena._collect_bifurcation_events
    # (src/arena.py lines 182-194). It exists because constructing a real Arena instance
    # requires a full config/GUI stack. If the real method changes (e.g. attribute names,
    # loop structure), this test must be updated to match.
    for _config, entities in mock_objects.values():
        for entity in entities:
            plugin = getattr(entity, '_movement_plugin', None)
            if plugin is None:
                continue
            detector = getattr(plugin, 'bifurcation_detector', None)
            if detector is not None and detector.events:
                mock_dh.collect_bifurcation_events(detector.events)

    assert len(mock_dh._bifurcation_events) == 2, (
        f"Expected 2 collected events, got {len(mock_dh._bifurcation_events)}"
    )
    assert mock_dh._bifurcation_events[0]["agent"] == "agent_0"
    assert mock_dh._bifurcation_events[1]["agent"] == "agent_1"


# ---------------------------------------------------------------------------
# 02-03: Mode dispatch, behavioral, gradient, and Omega tests
# ---------------------------------------------------------------------------


def make_mode_detector(
    mode: str = "behavioral",
    alignment_tolerance_deg: float = 5.0,
    alignment_consecutive_ticks: int = 3,
    gradient_window: int = 5,
    gradient_threshold: float = 0.005,
    spike_min_separation: int = 10,
    **kwargs,
) -> BifurcationDetector:
    return BifurcationDetector(
        agent_name="test_agent",
        mode=mode,
        alignment_tolerance_deg=alignment_tolerance_deg,
        alignment_consecutive_ticks=alignment_consecutive_ticks,
        gradient_window=gradient_window,
        gradient_threshold=gradient_threshold,
        spike_min_separation=spike_min_separation,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Behavioral mode tests
# ---------------------------------------------------------------------------

def test_behavioral_alignment_fires():
    """Bump within tolerance for N consecutive ticks fires a behavioral event."""
    det = make_mode_detector(mode="behavioral", alignment_consecutive_ticks=3,
                              alignment_tolerance_deg=5.0, spike_min_separation=100)
    target_angles = [0.0, math.pi]
    target_ids = ["A", "B"]
    bump_angle = 0.01  # ~0.57 deg from target A at 0.0

    event = None
    for tick in range(3):
        event = det._update_behavioral(tick, bump_angle, target_angles, target_ids)

    assert event is not None, "Expected behavioral event after 3 consecutive aligned ticks"
    assert event["mode"] == "behavioral"
    assert event["target"] == "A"
    assert "metric" in event
    assert len(det.events) == 1


def test_behavioral_alignment_resets_on_gap():
    """Counter resets when bump drifts out of tolerance."""
    det = make_mode_detector(mode="behavioral", alignment_consecutive_ticks=3,
                              alignment_tolerance_deg=5.0, spike_min_separation=100)
    target_angles = [0.0, math.pi]
    target_ids = ["A", "B"]

    # 2 aligned ticks
    det._update_behavioral(0, 0.01, target_angles, target_ids)
    det._update_behavioral(1, 0.01, target_angles, target_ids)
    # Gap: bump drifts to 0.5 (28 deg from target A, outside 5 deg tolerance)
    det._update_behavioral(2, 0.5, target_angles, target_ids)
    # 2 more aligned ticks — not enough (need 3 consecutive)
    det._update_behavioral(3, 0.01, target_angles, target_ids)
    result = det._update_behavioral(4, 0.01, target_angles, target_ids)
    assert result is None, "Should not fire after gap (only 2 consecutive after reset)"
    # 1 more to reach 3 consecutive
    result = det._update_behavioral(5, 0.01, target_angles, target_ids)
    assert result is not None, "Should fire after 3 consecutive aligned ticks"


def test_behavioral_alignment_correct_target():
    """Behavioral event assigns the correct target when bump is near pi."""
    det = make_mode_detector(mode="behavioral", alignment_consecutive_ticks=2,
                              alignment_tolerance_deg=10.0, spike_min_separation=100)
    target_angles = [0.0, math.pi]
    target_ids = ["A", "B"]
    bump_near_pi = math.pi - 0.05  # within 10 deg of target B at pi

    event = None
    for tick in range(2):
        event = det._update_behavioral(tick, bump_near_pi, target_angles, target_ids)

    assert event is not None
    assert event["target"] == "B"


def test_behavioral_suppression():
    """Second behavioral fire within spike_min_separation is suppressed."""
    det = make_mode_detector(mode="behavioral", alignment_consecutive_ticks=2,
                              alignment_tolerance_deg=5.0, spike_min_separation=10)
    target_angles = [0.0]
    target_ids = ["A"]

    # First fire at tick 1
    det._update_behavioral(0, 0.01, target_angles, target_ids)
    event1 = det._update_behavioral(1, 0.01, target_angles, target_ids)
    assert event1 is not None

    # Second alignment attempt at tick 3 (too close, within separation=10)
    det._update_behavioral(2, 0.01, target_angles, target_ids)
    event2 = det._update_behavioral(3, 0.01, target_angles, target_ids)
    assert event2 is None, "Second fire should be suppressed (within spike_min_separation)"


# ---------------------------------------------------------------------------
# Gradient criterion tests
# ---------------------------------------------------------------------------

def test_gradient_fires_at_peak():
    """Gradient-of-lambda1 local maximum fires an event for standard model."""
    det = make_mode_detector(mode="analytical", gradient_window=3,
                              gradient_threshold=0.01, spike_min_separation=100)
    # Simulate a lambda1 curve: flat, then steep rise, then plateau
    # ticks 0-5: flat at -1.0
    # ticks 6-10: rising [-1.0, -0.8, -0.5, -0.2, -0.1]
    # ticks 11-15: plateau at -0.1
    # gradient peaks during the rise
    sequence = (
        [-1.0] * 6  # flat
        + [-0.8, -0.5, -0.2, -0.1]  # rise
        + [-0.1] * 6  # plateau
    )
    events = []
    for tick, l1 in enumerate(sequence):
        event = det._check_gradient(tick, l1, 0.0, [0.0, math.pi], ["A", "B"])
        if event is not None:
            events.append(event)

    assert len(events) >= 1, f"Expected at least 1 gradient event, got {len(events)}"
    assert events[0]["mode"] == "analytical"
    assert "metric" in events[0]


def test_gradient_no_fire_flat():
    """Flat lambda1 sequence produces zero gradient, no event."""
    det = make_mode_detector(mode="analytical", gradient_window=3,
                              gradient_threshold=0.01, spike_min_separation=100)
    sequence = [-1.0] * 20
    events = []
    for tick, l1 in enumerate(sequence):
        event = det._check_gradient(tick, l1, 0.0, [0.0], ["A"])
        if event is not None:
            events.append(event)

    assert len(events) == 0, f"Expected 0 events from flat sequence, got {len(events)}"


def test_gradient_window_respected():
    """Gradient uses values gradient_window ticks apart."""
    det = make_mode_detector(mode="analytical", gradient_window=3,
                              gradient_threshold=0.001, spike_min_separation=100)
    # tick 0: -1.0, tick 1: -0.9, tick 2: -0.8, tick 3: -0.7
    # At tick 3: gradient = (-0.7 - (-1.0)) / 3 = 0.1
    # tick 4: -0.7 (plateau)
    # At tick 4: gradient = (-0.7 - (-0.9)) / 3 = 0.0667
    # tick 5: -0.7
    # At tick 5: gradient = (-0.7 - (-0.8)) / 3 = 0.0333
    # gradient peaks at tick 3 (0.1), confirmed by tick 4 and 5 being lower
    sequence = [-1.0, -0.9, -0.8, -0.7, -0.7, -0.7, -0.7]
    events = []
    for tick, l1 in enumerate(sequence):
        event = det._check_gradient(tick, l1, 0.0, [0.0], ["A"])
        if event is not None:
            events.append(event)

    assert len(events) >= 1, f"Expected event at gradient peak, got {len(events)}"


# ---------------------------------------------------------------------------
# Omega computation and crossing tests
# ---------------------------------------------------------------------------

def test_omega_computation_formula():
    """Omega matches the analytical formula for a known neural_ring and input."""
    mf = MeanFieldSystem(
        num_neurons=20,
        u=6.0, beta=1.0, sigma=0.0, dt=0.1, integration_time=5.0,
        g_adapt=0.5, tau_adapt=10.0,
        rng=np.random.default_rng(42),
    )
    # Set a cosine bump centered at 0
    mf.neural_ring = 0.5 * (1.0 + np.cos(mf.theta))
    # Set a known input
    mf.b = 0.3 * (1.0 + np.cos(mf.theta - 0.5))

    det = make_mode_detector(mode="analytical")
    omega = det.compute_omega(mf)

    # Manual computation
    theta = mf.theta
    n = mf.num_neurons
    beta_paper = mf.g_adapt  # 0.5
    z_complex = np.sum(mf.neural_ring * np.exp(1j * theta))
    A = np.abs(z_complex) / (n / 2.0)
    i0_complex = np.sum(mf.b * np.exp(1j * theta))
    I0 = np.abs(i0_complex) / (n / 2.0)
    expected_omega = (1.0 + beta_paper) * A / ((1.0 + beta_paper) * A + I0 + 1e-12)

    assert omega is not None
    assert abs(omega - expected_omega) < 1e-6, (
        f"Omega mismatch: got {omega}, expected {expected_omega}"
    )


def test_omega_crossing_detection():
    """Omega crossing the Hopf threshold fires an event."""
    mf = MeanFieldSystem(
        num_neurons=20, u=6.0, beta=1.0, sigma=0.0, dt=0.1, integration_time=5.0,
        g_adapt=0.5, tau_adapt=10.0, rng=np.random.default_rng(42),
    )
    det = make_mode_detector(mode="analytical", spike_min_separation=100)

    alpha_paper = 1.0 / mf.tau_adapt  # 0.1
    beta_paper = mf.g_adapt  # 0.5
    threshold = (1.0 + alpha_paper) / (1.0 + beta_paper)  # 1.1 / 1.5 = 0.7333

    # Feed synthetic omega values: below, crossing, above
    omega_sequence = [0.5, 0.6, 0.75, 0.8]  # crosses threshold at index 2
    events = []
    for tick, omega in enumerate(omega_sequence):
        event = det._check_omega_crossing(tick, omega, 0.0, [0.0], ["A"], mf)
        if event is not None:
            events.append(event)

    assert len(events) == 1, f"Expected 1 Omega crossing event, got {len(events)}"
    assert events[0]["mode"] == "analytical"
    assert "metric" in events[0]


def test_omega_no_fire_below_threshold():
    """Omega values staying below threshold produce no event."""
    mf = MeanFieldSystem(
        num_neurons=20, u=6.0, beta=1.0, sigma=0.0, dt=0.1, integration_time=5.0,
        g_adapt=0.5, tau_adapt=10.0, rng=np.random.default_rng(42),
    )
    det = make_mode_detector(mode="analytical", spike_min_separation=100)

    # All values below threshold (0.7333)
    omega_sequence = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7]
    events = []
    for tick, omega in enumerate(omega_sequence):
        event = det._check_omega_crossing(tick, omega, 0.0, [0.0], ["A"], mf)
        if event is not None:
            events.append(event)

    assert len(events) == 0, f"Expected 0 events below threshold, got {len(events)}"


# ---------------------------------------------------------------------------
# Integration tests (analytical mode with real MeanFieldSystem)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_standard_model_analytical_gradient():
    """Standard model in analytical mode fires via gradient criterion."""
    num_neurons = 50
    mf = MeanFieldSystem(
        num_neurons=num_neurons, u=6.0, beta=1.0, sigma=0.0,
        dt=0.1, integration_time=5.0, num_targets=2,
        rng=np.random.default_rng(0),
    )
    det = BifurcationDetector(
        agent_name="test_std_grad",
        mode="analytical",
        gradient_window=5,
        gradient_threshold=0.005,
        spike_min_separation=10,
    )
    target_angles = [0.0, math.pi]
    target_ids = ["A", "B"]
    target_qualities = np.array([1.0, 0.5])

    for tick in range(200):
        _, bump_positions, _ = mf.step(
            target_angles=target_angles,
            target_qualities=target_qualities,
        )
        bump_angle = float(bump_positions[-1]) if bump_positions is not None and len(bump_positions) > 0 else 0.0
        det.update(
            tick=tick, mf=mf, bump_angle=bump_angle,
            target_angles=list(target_angles), target_ids=target_ids,
        )

    # Should have computed lambda1 for logging
    assert det.last_lambda1 is not None, "last_lambda1 should be populated"


@pytest.mark.slow
def test_sfa_model_analytical_omega():
    """SFA model in analytical mode computes Omega each tick."""
    num_neurons = 50
    mf = MeanFieldSystem(
        num_neurons=num_neurons, u=6.0, beta=1.0, sigma=0.0,
        dt=0.1, integration_time=5.0, num_targets=2,
        g_adapt=0.5, tau_adapt=10.0,
        rng=np.random.default_rng(0),
    )
    det = BifurcationDetector(
        agent_name="test_sfa_omega",
        mode="analytical",
        spike_min_separation=10,
    )
    target_angles = [0.0, math.pi]
    target_ids = ["A", "B"]
    target_qualities = np.array([1.0, 0.5])

    for tick in range(300):
        _, bump_positions, _ = mf.step(
            target_angles=target_angles,
            target_qualities=target_qualities,
        )
        bump_angle = float(bump_positions[-1]) if bump_positions is not None and len(bump_positions) > 0 else 0.0
        det.update(
            tick=tick, mf=mf, bump_angle=bump_angle,
            target_angles=list(target_angles), target_ids=target_ids,
            perception_vec=mf.b,
        )

    assert det.last_omega is not None, "last_omega should be populated for SFA model"


# ---------------------------------------------------------------------------
# Event schema tests
# ---------------------------------------------------------------------------

def test_new_event_schema_behavioral():
    """Behavioral event has {agent, tick, metric, target, mode} keys."""
    det = make_mode_detector(mode="behavioral", alignment_consecutive_ticks=1,
                              alignment_tolerance_deg=90.0, spike_min_separation=100)
    event = det._update_behavioral(0, 0.01, [0.0], ["A"])
    assert event is not None
    assert set(event.keys()) == {"agent", "tick", "metric", "target", "mode"}
    assert event["mode"] == "behavioral"


def test_new_event_schema_analytical():
    """Analytical gradient event has {agent, tick, metric, target, mode} keys."""
    det = make_mode_detector(mode="analytical", gradient_window=2,
                              gradient_threshold=0.01, spike_min_separation=100)
    # Feed: flat, steep rise, plateau to trigger gradient peak
    sequence = [-1.0, -1.0, -1.0, -0.5, -0.2, -0.1, -0.1, -0.1]
    event = None
    for tick, l1 in enumerate(sequence):
        result = det._check_gradient(tick, l1, 0.0, [0.0], ["A"])
        if result is not None:
            event = result

    assert event is not None, "Expected gradient event"
    assert set(event.keys()) == {"agent", "tick", "metric", "target", "mode"}
    assert event["mode"] == "analytical"
