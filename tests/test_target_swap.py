"""Unit and integration tests for Phase 3 post-bifurcation target swap logic.

Covers:
  test_normalize_config_valid          — valid config parses correctly (D-03)
  test_normalize_config_absent         — None/empty config returns None (SWAP-03)
  test_normalize_config_invalid_pairs  — malformed pairs raises ValueError
  test_normalize_config_delay_default  — missing delay_ticks defaults to 0
  test_normalize_config_neg_delay      — negative delay_ticks raises ValueError
  test_normalize_config_same_ids       — duplicate IDs raises ValueError
  test_find_first_bifurcation_in_snapshots       — finds event from spins data
  test_find_first_bifurcation_none_when_empty    — returns None when no events
  test_find_first_bifurcation_picks_earliest     — returns earliest tick event
  test_check_post_bif_swap_schedules_on_first_bif — first bif schedules swap (SWAP-01)
  test_check_post_bif_swap_ignores_subsequent    — second bif does not re-trigger (D-01, D-04)
  test_execute_post_bif_swap_positions — XY positions are exchanged (SWAP-01)
  test_execute_post_bif_swap_strength  — strength values are exchanged (D-02, D-07)
  test_execute_post_bif_swap_event_schema — swap event matches D-05 schema (SWAP-02)
  test_collect_swap_events_and_write   — DataHandling stores and writes swap_events (SWAP-02)
  test_delay_ticks_zero                — delay_ticks=0 fires swap at bifurcation tick
  test_no_bif_no_swap                  — no bifurcation means swap_events remains empty
  test_multi_pair_swap                 — multiple pairs all get swapped
  test_sweep_independence              — two runs with different delay_ticks are independent (SWAP-04)
"""

import json
import os
import sys
import tempfile
import types

import pytest

# ---------------------------------------------------------------------------
# Import path: tests run from project root with src on sys.path
# (mirrors the pattern in test_bifurcation.py)
# ---------------------------------------------------------------------------

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from geometry_utils.vector3D import Vector3D
from arena import Arena
from dataHandling import DataHandling


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class _MockEntity:
    """Duck-types a target entity with position, strength, and name.

    Matches the interface used by Arena._swap_object_xy_positions and
    Arena._execute_post_bif_swap.
    """

    def __init__(self, name: str, x: float, y: float, strength: float, z: float = 0.0):
        self._name = name
        self._pos = Vector3D(x, y, z)
        self.strength = strength

    def get_position(self) -> Vector3D:
        return self._pos

    def set_position(self, pos: Vector3D) -> None:
        self._pos = pos

    def get_name(self) -> str:
        return self._name

    def entity(self) -> str:
        return "target"


class _MockDataHandling:
    """Minimal DataHandling stand-in for swap unit tests."""

    def __init__(self):
        self._swap_events: list[dict] = []
        self._bifurcation_events: list[dict] = []

    def collect_swap_events(self, events: list[dict]) -> None:
        self._swap_events.extend(events)

    def collect_bifurcation_events(self, events: list[dict]) -> None:
        self._bifurcation_events.extend(events)


class _ArenaStub:
    """Minimal Arena-like object with Phase 3 state attributes.

    Binds the real Arena instance methods so that methods calling
    ``self._find_first_bifurcation_in_snapshots`` and ``self._index_objects_by_name``
    work correctly — without requiring a full Arena config/GUI stack.
    """

    def __init__(
        self,
        pairs: list | None = None,
        delay_ticks: int = 0,
        objects: dict | None = None,
        data_handling=None,
    ):
        if pairs is not None:
            raw_cfg = {"pairs": pairs, "delay_ticks": delay_ticks}
            self._post_bif_swap_cfg = Arena._normalize_post_bif_swap_config(raw_cfg)
        else:
            self._post_bif_swap_cfg = None

        self._post_bif_swap_triggered = False
        self._post_bif_swap_event = None
        # objects: {key: (config_dict, [entities])} — mirrors Arena.objects
        self.objects = objects or {}
        self.data_handling = data_handling

    # Bind the real Arena methods so ``self`` resolution works correctly
    _find_first_bifurcation_in_snapshots = Arena._find_first_bifurcation_in_snapshots
    _check_post_bif_swap = Arena._check_post_bif_swap
    _execute_post_bif_swap = Arena._execute_post_bif_swap
    _index_objects_by_name = Arena._index_objects_by_name
    _swap_object_xy_positions = staticmethod(Arena._swap_object_xy_positions)


def _make_arena_stub(
    pairs: list | None = None,
    delay_ticks: int = 0,
    objects: dict | None = None,
    data_handling=None,
) -> _ArenaStub:
    """Return a minimal Arena-like stub for Phase 3 method testing."""
    return _ArenaStub(pairs=pairs, delay_ticks=delay_ticks, objects=objects, data_handling=data_handling)


def _make_bif_snapshot(agent: str, tick: int) -> dict:
    """Return an agent snapshot dict containing a single bifurcation event."""
    return {
        "status": [tick, 10],
        "agents_spins": {
            "group_0": [
                {
                    "new_bifurcation_events": [
                        {
                            "agent": agent,
                            "tick": tick,
                            "metric": 0.05,
                            "target": "A",
                            "mode": "behavioral",
                        }
                    ]
                }
            ]
        },
    }


# ---------------------------------------------------------------------------
# _normalize_post_bif_swap_config tests
# ---------------------------------------------------------------------------


def test_normalize_config_valid():
    """Valid config with pairs and delay_ticks parses correctly (D-03)."""
    cfg = {"pairs": [["target_A", "target_B"]], "delay_ticks": 10}
    result = Arena._normalize_post_bif_swap_config(cfg)
    assert result is not None
    assert result["delay_ticks"] == 10
    assert len(result["pairs"]) == 1
    left, right = result["pairs"][0]
    assert left == "target_A"
    assert right == "target_B"


def test_normalize_config_absent():
    """None or empty config returns None — backward compatible, no swap (SWAP-03)."""
    assert Arena._normalize_post_bif_swap_config(None) is None
    assert Arena._normalize_post_bif_swap_config({}) is None
    assert Arena._normalize_post_bif_swap_config(False) is None


def test_normalize_config_invalid_pairs():
    """Malformed pairs (wrong length) raises ValueError."""
    # Single-element pair
    with pytest.raises(ValueError):
        Arena._normalize_post_bif_swap_config({"pairs": [["only_one"]]})
    # Three-element pair
    with pytest.raises(ValueError):
        Arena._normalize_post_bif_swap_config({"pairs": [["A", "B", "C"]]})
    # pairs key is missing entirely
    with pytest.raises(ValueError):
        Arena._normalize_post_bif_swap_config({"delay_ticks": 5})


def test_normalize_config_delay_default():
    """Missing delay_ticks defaults to 0."""
    cfg = {"pairs": [["A", "B"]]}
    result = Arena._normalize_post_bif_swap_config(cfg)
    assert result is not None
    assert result["delay_ticks"] == 0


def test_normalize_config_neg_delay():
    """Negative delay_ticks raises ValueError."""
    with pytest.raises(ValueError):
        Arena._normalize_post_bif_swap_config({"pairs": [["A", "B"]], "delay_ticks": -1})


def test_normalize_config_same_ids():
    """Pair with identical IDs raises ValueError."""
    with pytest.raises(ValueError):
        Arena._normalize_post_bif_swap_config({"pairs": [["A", "A"]]})


# ---------------------------------------------------------------------------
# _find_first_bifurcation_in_snapshots tests
# ---------------------------------------------------------------------------


def test_find_first_bifurcation_in_snapshots():
    """_find_first_bifurcation_in_snapshots returns event from agent snapshot spins data."""
    snap = _make_bif_snapshot("agent_0", tick=50)
    stub = _make_arena_stub()
    result = Arena._find_first_bifurcation_in_snapshots(stub, [snap])
    assert result is not None
    assert result["agent"] == "agent_0"
    assert result["tick"] == 50


def test_find_first_bifurcation_none_when_empty():
    """Returns None when no bifurcation events exist in snapshots."""
    stub = _make_arena_stub()
    # Empty snapshot
    assert Arena._find_first_bifurcation_in_snapshots(stub, []) is None
    # Snapshot with no new_bifurcation_events key
    snap_no_events = {
        "status": [0, 10],
        "agents_spins": {
            "group_0": [{"new_bifurcation_events": []}]
        },
    }
    assert Arena._find_first_bifurcation_in_snapshots(stub, [snap_no_events]) is None


def test_find_first_bifurcation_picks_earliest():
    """With multiple events across agents, returns the event with the lowest tick."""
    snap_a = _make_bif_snapshot("agent_0", tick=30)
    snap_b = _make_bif_snapshot("agent_1", tick=20)  # earlier
    stub = _make_arena_stub()
    result = Arena._find_first_bifurcation_in_snapshots(stub, [snap_a, snap_b])
    assert result is not None
    assert result["agent"] == "agent_1"
    assert result["tick"] == 20


# ---------------------------------------------------------------------------
# _check_post_bif_swap tests
# ---------------------------------------------------------------------------


def test_check_post_bif_swap_schedules_on_first_bif():
    """First bifurcation event schedules swap at bif_tick + delay_ticks (SWAP-01)."""
    stub = _make_arena_stub(pairs=[["A", "B"]], delay_ticks=10)
    snap = _make_bif_snapshot("agent_0", tick=50)

    Arena._check_post_bif_swap(stub, tick=50, agent_snapshots=[snap])

    assert stub._post_bif_swap_triggered is True
    assert stub._post_bif_swap_event is not None
    assert stub._post_bif_swap_event["tick"] == 60  # 50 + 10
    assert stub._post_bif_swap_event["bifurcation_tick"] == 50
    assert stub._post_bif_swap_event["triggered_by_agent"] == "agent_0"


def test_check_post_bif_swap_ignores_subsequent():
    """Subsequent bifurcation events do not re-trigger the swap (D-01, D-04)."""
    stub = _make_arena_stub(pairs=[["A", "B"]], delay_ticks=10)

    # First bifurcation at tick 50
    snap1 = _make_bif_snapshot("agent_0", tick=50)
    Arena._check_post_bif_swap(stub, tick=50, agent_snapshots=[snap1])
    first_event = stub._post_bif_swap_event

    # Mark triggered, simulate second bifurcation at tick 55 BEFORE swap fires
    # Manually re-set _post_bif_swap_event to simulate pre-fire state
    stub._post_bif_swap_event = first_event  # ensure it's the same scheduled event
    snap2 = _make_bif_snapshot("agent_1", tick=55)
    Arena._check_post_bif_swap(stub, tick=55, agent_snapshots=[snap2])

    # The scheduled event should be unchanged (same swap_tick=60)
    assert stub._post_bif_swap_event is not None
    assert stub._post_bif_swap_event["tick"] == 60
    assert stub._post_bif_swap_event["triggered_by_agent"] == "agent_0"


def test_check_no_swap_when_cfg_absent():
    """If _post_bif_swap_cfg is None, _check_post_bif_swap is a no-op (SWAP-03)."""
    stub = _make_arena_stub(pairs=None)  # no config
    snap = _make_bif_snapshot("agent_0", tick=50)

    Arena._check_post_bif_swap(stub, tick=50, agent_snapshots=[snap])

    assert stub._post_bif_swap_triggered is False
    assert stub._post_bif_swap_event is None


# ---------------------------------------------------------------------------
# _execute_post_bif_swap tests
# ---------------------------------------------------------------------------


def test_execute_post_bif_swap_positions():
    """XY positions are exchanged between target entities (SWAP-01)."""
    entity_a = _MockEntity("target_A", x=10.0, y=5.0, strength=2.0)
    entity_b = _MockEntity("target_B", x=-10.0, y=5.0, strength=0.5)
    objects = {"targets": ({"number": 2}, [entity_a, entity_b])}
    dh = _MockDataHandling()
    stub = _make_arena_stub(pairs=[["target_A", "target_B"]], objects=objects, data_handling=dh)
    stub._post_bif_swap_event = {
        "tick": 60,
        "pairs": [("target_A", "target_B")],
        "triggered_by_agent": "agent_0",
        "bifurcation_tick": 50,
    }

    Arena._execute_post_bif_swap(stub, tick=60)

    # After swap: A gets B's position, B gets A's position (XY only)
    assert entity_a.get_position().x == pytest.approx(-10.0)
    assert entity_a.get_position().y == pytest.approx(5.0)
    assert entity_b.get_position().x == pytest.approx(10.0)
    assert entity_b.get_position().y == pytest.approx(5.0)


def test_execute_post_bif_swap_strength():
    """Strength values are exchanged between target entities (D-02, D-07)."""
    entity_a = _MockEntity("target_A", x=10.0, y=5.0, strength=2.0)
    entity_b = _MockEntity("target_B", x=-10.0, y=5.0, strength=0.5)
    objects = {"targets": ({"number": 2}, [entity_a, entity_b])}
    dh = _MockDataHandling()
    stub = _make_arena_stub(pairs=[["target_A", "target_B"]], objects=objects, data_handling=dh)
    stub._post_bif_swap_event = {
        "tick": 60,
        "pairs": [("target_A", "target_B")],
        "triggered_by_agent": "agent_0",
        "bifurcation_tick": 50,
    }

    Arena._execute_post_bif_swap(stub, tick=60)

    # Strengths exchanged: A now has 0.5, B now has 2.0
    assert entity_a.strength == pytest.approx(0.5)
    assert entity_b.strength == pytest.approx(2.0)


def test_execute_post_bif_swap_event_schema():
    """Swap event logged to DataHandling matches D-05 schema (SWAP-02)."""
    entity_a = _MockEntity("target_A", x=10.0, y=5.0, strength=2.0)
    entity_b = _MockEntity("target_B", x=-10.0, y=5.0, strength=0.5)
    objects = {"targets": ({"number": 2}, [entity_a, entity_b])}
    dh = _MockDataHandling()
    stub = _make_arena_stub(pairs=[["target_A", "target_B"]], objects=objects, data_handling=dh)
    scheduled_event = {
        "tick": 60,
        "pairs": [("target_A", "target_B")],
        "triggered_by_agent": "agent_0",
        "bifurcation_tick": 50,
    }
    stub._post_bif_swap_event = scheduled_event

    Arena._execute_post_bif_swap(stub, tick=60)

    assert len(dh._swap_events) == 1, f"Expected 1 swap event, got {len(dh._swap_events)}"
    ev = dh._swap_events[0]
    # D-05: required fields
    assert "tick" in ev, "swap event missing 'tick'"
    assert "pairs" in ev, "swap event missing 'pairs'"
    assert "triggered_by_agent" in ev, "swap event missing 'triggered_by_agent'"
    assert "bifurcation_tick" in ev, "swap event missing 'bifurcation_tick'"
    assert ev["tick"] == 60
    assert ev["triggered_by_agent"] == "agent_0"
    assert ev["bifurcation_tick"] == 50


# ---------------------------------------------------------------------------
# DataHandling collect_swap_events and _write_events_json tests
# ---------------------------------------------------------------------------


def test_collect_swap_events_and_write():
    """DataHandling.collect_swap_events stores events; _write_events_json outputs them (SWAP-02).

    Uses the real DataHandling class via a minimal mock Config.
    """
    swap_event = {
        "tick": 60,
        "pairs": [["target_A", "target_B"]],
        "triggered_by_agent": "agent_0",
        "bifurcation_tick": 50,
    }

    with tempfile.TemporaryDirectory() as tmp_base:
        # Minimal mock Config that DataHandling.__init__ needs
        cfg = types.SimpleNamespace(
            arena={"_id": "abstract"},
            results={"base_path": tmp_base},
        )
        dh = DataHandling(cfg)
        # Create run folder manually (new_run normally does this but requires shapes)
        run_folder = os.path.join(dh.config_folder, "run_1")
        os.makedirs(run_folder, exist_ok=True)
        dh.run_folder = run_folder
        dh._bifurcation_events = []
        dh._swap_events = []

        dh.collect_swap_events([swap_event])

        assert len(dh._swap_events) == 1, f"Expected 1 event stored, got {len(dh._swap_events)}"
        assert dh._swap_events[0]["tick"] == 60

        dh._write_events_json()

        events_path = os.path.join(run_folder, "events.json")
        assert os.path.exists(events_path), "events.json not created"

        with open(events_path) as f:
            data = json.load(f)

        assert "swap_events" in data, "events.json missing 'swap_events'"
        assert "bifurcation_events" in data, "events.json missing 'bifurcation_events'"
        assert len(data["swap_events"]) == 1
        assert data["swap_events"][0]["tick"] == 60
        assert data["swap_events"][0]["triggered_by_agent"] == "agent_0"
        assert data["swap_events"][0]["bifurcation_tick"] == 50


# ---------------------------------------------------------------------------
# Behavioral edge-case tests
# ---------------------------------------------------------------------------


def test_delay_ticks_zero():
    """delay_ticks=0 means swap fires at the exact bifurcation tick (D-03).

    When delay_ticks=0, swap_tick == bif_tick.  _check_post_bif_swap schedules
    the event AND immediately executes it in the same tick, so after the call
    _post_bif_swap_event is None (cleared by _execute_post_bif_swap) but the
    positions/strength have been exchanged and the swap event is in DataHandling.
    """
    entity_a = _MockEntity("target_A", x=10.0, y=5.0, strength=2.0)
    entity_b = _MockEntity("target_B", x=-10.0, y=5.0, strength=0.5)
    objects = {"targets": ({"number": 2}, [entity_a, entity_b])}
    dh = _MockDataHandling()
    stub = _make_arena_stub(pairs=[["target_A", "target_B"]], delay_ticks=0, objects=objects, data_handling=dh)

    snap = _make_bif_snapshot("agent_0", tick=50)
    Arena._check_post_bif_swap(stub, tick=50, agent_snapshots=[snap])

    # swap_tick == bif_tick + 0 == 50; swap executes immediately in the same call
    assert stub._post_bif_swap_triggered is True
    # _post_bif_swap_event is cleared to None after execution
    assert stub._post_bif_swap_event is None
    # The swap was actually applied: positions exchanged
    assert entity_a.get_position().x == pytest.approx(-10.0)
    assert entity_b.get_position().x == pytest.approx(10.0)
    # The event was logged to DataHandling with tick==50
    assert len(dh._swap_events) == 1
    assert dh._swap_events[0]["tick"] == 50


def test_no_bif_no_swap():
    """If no bifurcation fires, _post_bif_swap_triggered stays False and no events logged."""
    dh = _MockDataHandling()
    stub = _make_arena_stub(pairs=[["target_A", "target_B"]], delay_ticks=5, data_handling=dh)

    # Simulate several ticks with no bifurcation events
    for tick in range(1, 20):
        Arena._check_post_bif_swap(stub, tick=tick, agent_snapshots=[])

    assert stub._post_bif_swap_triggered is False
    assert stub._post_bif_swap_event is None
    assert len(dh._swap_events) == 0, f"Expected 0 swap events, got {len(dh._swap_events)}"


def test_multi_pair_swap():
    """Multiple pairs in config are all swapped in one execution (D-03)."""
    entity_a = _MockEntity("target_A", x=10.0, y=5.0, strength=2.0)
    entity_b = _MockEntity("target_B", x=-10.0, y=5.0, strength=0.5)
    entity_c = _MockEntity("target_C", x=0.0, y=10.0, strength=1.0)
    entity_d = _MockEntity("target_D", x=0.0, y=-10.0, strength=0.3)
    objects = {"targets": ({}, [entity_a, entity_b, entity_c, entity_d])}
    dh = _MockDataHandling()
    stub = _make_arena_stub(
        pairs=[["target_A", "target_B"], ["target_C", "target_D"]],
        delay_ticks=0,
        objects=objects,
        data_handling=dh,
    )
    stub._post_bif_swap_event = {
        "tick": 50,
        "pairs": [("target_A", "target_B"), ("target_C", "target_D")],
        "triggered_by_agent": "agent_0",
        "bifurcation_tick": 50,
    }

    Arena._execute_post_bif_swap(stub, tick=50)

    # Pair A <-> B positions
    assert entity_a.get_position().x == pytest.approx(-10.0)
    assert entity_b.get_position().x == pytest.approx(10.0)
    # Pair A <-> B strength
    assert entity_a.strength == pytest.approx(0.5)
    assert entity_b.strength == pytest.approx(2.0)
    # Pair C <-> D positions
    assert entity_c.get_position().y == pytest.approx(-10.0)
    assert entity_d.get_position().y == pytest.approx(10.0)
    # Pair C <-> D strength
    assert entity_c.strength == pytest.approx(0.3)
    assert entity_d.strength == pytest.approx(1.0)


def test_sweep_independence():
    """Two sequential runs with different delay_ticks produce independent correct swap ticks (SWAP-04).

    Simulates what happens in a parameter sweep: each run constructs its own Arena-like
    state with a different delay, bifurcates at the same tick, and produces the correct
    independent swap_tick.  Uses delay_ticks > 0 for both runs so the swap event is
    still pending after scheduling (allowing tick verification before execution).
    """
    bif_tick = 50
    collected: list[dict] = []

    for delay in (5, 15):
        entity_a = _MockEntity("target_A", x=10.0, y=5.0, strength=2.0)
        entity_b = _MockEntity("target_B", x=-10.0, y=5.0, strength=0.5)
        objects = {"targets": ({}, [entity_a, entity_b])}
        dh = _MockDataHandling()
        stub = _make_arena_stub(
            pairs=[["target_A", "target_B"]],
            delay_ticks=delay,
            objects=objects,
            data_handling=dh,
        )

        snap = _make_bif_snapshot("agent_0", tick=bif_tick)
        # Schedule the swap (swap_tick = bif_tick + delay > bif_tick, so not yet executed)
        Arena._check_post_bif_swap(stub, tick=bif_tick, agent_snapshots=[snap])
        assert stub._post_bif_swap_triggered is True
        assert stub._post_bif_swap_event is not None, (
            f"delay={delay}: swap event should be scheduled but not yet fired"
        )
        scheduled_tick = stub._post_bif_swap_event["tick"]
        assert scheduled_tick == bif_tick + delay, (
            f"delay={delay}: expected swap_tick={bif_tick + delay}, got {scheduled_tick}"
        )

        # Execute the swap at the scheduled tick
        Arena._execute_post_bif_swap(stub, tick=scheduled_tick)
        assert len(dh._swap_events) == 1
        collected.append(dh._swap_events[0])

    # Run with delay=5 fired at tick 55; run with delay=15 fired at tick 65
    assert collected[0]["tick"] == 55
    assert collected[1]["tick"] == 65
    # Each run is independent: same bif_tick, different swap_ticks
    assert collected[0]["bifurcation_tick"] == 50
    assert collected[1]["bifurcation_tick"] == 50
    # Runs do not share state: swap_ticks differ by exactly delta_delay
    assert collected[1]["tick"] - collected[0]["tick"] == 10
