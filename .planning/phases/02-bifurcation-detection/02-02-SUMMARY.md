---
phase: 02-bifurcation-detection
plan: "02"
subsystem: bifurcation-detection
tags: [bifurcation, wiring, events-json, arena, data-handling, integration-tests]
dependency_graph:
  requires: [BifurcationDetector, bifurcation-spike-detection]
  provides: [bifurcation-wiring, events.json-output, arena-event-collection]
  affects:
    - src/models/movement/mean_field_model.py
    - src/dataHandling.py
    - src/arena.py
    - tests/test_bifurcation.py
    - pyproject.toml
tech_stack:
  added: []
  patterns:
    - "BifurcationDetector per-agent instantiation from mean_field_model.bifurcation config namespace (D-07)"
    - "Duck-typed getattr(_movement_plugin, bifurcation_detector) for zero-coupling agent iteration"
    - "events.json sidecar written once at run-end by DataHandling before archiving (D-06)"
    - "T-02-05 mitigation: guard on run_folder is None/missing dir before writing events.json"
key_files:
  created: []
  modified:
    - src/models/movement/mean_field_model.py
    - src/dataHandling.py
    - src/arena.py
    - tests/test_bifurcation.py
    - pyproject.toml
decisions:
  - "Detector call placed after self._last_bump_angle = angle_rad in step() so it uses current tick bump angle (D-01)"
  - "Duck-typed getattr for bifurcation_detector access in Arena._collect_bifurcation_events() — only MeanFieldMovementModel agents contribute events, no isinstance check needed"
  - "reset() uses hasattr guard so BifurcationDetector reset is safe even if called before __init__ finishes"
  - "slow mark registered in pyproject.toml to suppress PytestUnknownMarkWarning for integration tests"
metrics:
  duration: "~3 minutes"
  completed: "2026-04-10"
  tasks_completed: 2
  files_created: 0
  files_modified: 5
  tests_added: 6
---

# Phase 2 Plan 2: BifurcationDetector Wiring and Output Pipeline Summary

**One-liner:** BifurcationDetector wired into MeanFieldMovementModel per-agent from config, events collected by Arena and written to events.json sidecar via DataHandling at run end.

## What Was Built

### Task 1: MeanFieldMovementModel Integration (`src/models/movement/mean_field_model.py`)

- **Import** added: `from models.bifurcation import BifurcationDetector`
- **`__init__`**: reads `mean_field_model.bifurcation` config namespace (D-07) to instantiate a `BifurcationDetector` per agent with `lambda_threshold` and `spike_min_separation` defaults
- **`step()`**: calls `self.bifurcation_detector.update()` after `self._last_bump_angle = angle_rad`, passing `tick`, `mf`, `angle_rad`, and target angles extracted from `self._mf_entities["targets"]`
- **`reset()`**: clears `bifurcation_detector.events`, `_buffer`, and `_last_fire_tick` between runs via `hasattr` guard

### Task 2: DataHandling events.json Output (`src/dataHandling.py`)

- **`_bifurcation_events: list[dict]`** instance variable added to `DataHandling.__init__`
- **`collect_bifurcation_events(events)`**: accumulates events from agents into `_bifurcation_events`
- **`_write_events_json()`**: writes `{"bifurcation_events": [...], "swap_events": []}` to `events.json` in the run folder; guarded by `run_folder is None or not os.path.isdir()` (T-02-05 mitigation)
- **`DataHandling.close()`**: calls `_write_events_json()` before `_archive_run_folder()`
- **`DataHandling.new_run()`**: resets `_bifurcation_events = []` to clear between runs

### Task 2: Arena Event Collection Wiring (`src/arena.py`)

- **`Arena._collect_bifurcation_events()`**: iterates `self.objects`, duck-types each entity's `_movement_plugin` for a `bifurcation_detector` attribute, transfers non-empty event lists to `data_handling.collect_bifurcation_events()`
- **`Arena.close()`**: calls `_collect_bifurcation_events()` before the entity close loop and before `data_handling.close()`
- **`SpaceArena.reset()`**: calls `_collect_bifurcation_events()` before `data_handling.close()` (between-run event capture)

### Task 2: Integration and Config Tests (`tests/test_bifurcation.py`)

6 new tests added (total now 16 in the file):

| Test | Coverage |
|------|---------|
| `test_events_json_schema` | `_write_events_json` creates correct keys, swap_events is empty |
| `test_no_bifurcation_empty_events_json` | No events → `bifurcation_events: []` |
| `test_custom_config_respected` | Peak at -0.15: rejected by threshold=-0.1, accepted by threshold=-0.2 |
| `test_standard_model_bifurcation` | Standard MF model (200 steps) produces >= 1 event |
| `test_sfa_model_bifurcation` | SFA model (300 steps, g_adapt=0.5) produces >= 1 event |
| `test_collect_bifurcation_events_wiring` | Arena collection path transfers events from mock agents correctly |

All 23 tests (16 bifurcation + 4 mean-field-system + 3 regression) pass.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed `get_bump_positions()` call — method does not exist on MeanFieldSystem**
- **Found during:** Task 2 (writing integration tests)
- **Issue:** Plan's test pseudocode called `mf.get_bump_positions()` which is not a method on `MeanFieldSystem`; `step()` returns `(neural_ring, bump_positions, final_norm)` directly
- **Fix:** Integration tests use `_, bump_positions, _ = mf.step(...)` to unpack the return value
- **Files modified:** `tests/test_bifurcation.py`
- **Commit:** 6b598be

**2. [Rule 2 - Correctness] Registered `slow` pytest mark in pyproject.toml**
- **Found during:** Task 2 test run
- **Issue:** `@pytest.mark.slow` produced `PytestUnknownMarkWarning` without registration
- **Fix:** Added `markers = ["slow: marks tests as slow"]` to `[tool.pytest.ini_options]` in `pyproject.toml`
- **Files modified:** `pyproject.toml`
- **Commit:** 6b598be

## Known Stubs

None. All wiring is complete. `swap_events` is an intentional placeholder (empty list) reserved for Phase 3 per D-06 — this is documented design, not a stub.

## Threat Flags

No new threat surface introduced. T-02-05 mitigation (guard on missing `run_folder`) is in place in `_write_events_json()`.

## Self-Check: PASSED

| Check | Result |
|-------|--------|
| `src/models/movement/mean_field_model.py` contains `BifurcationDetector` import | FOUND |
| `src/models/movement/mean_field_model.py` contains `bifurcation_detector.update(` | FOUND |
| `src/dataHandling.py` contains `write_events_json` and `collect_bifurcation_events` | FOUND |
| `src/arena.py` contains `_collect_bifurcation_events` | FOUND |
| Commit e364935 (Task 1) exists | FOUND |
| Commit 6b598be (Task 2) exists | FOUND |
| `pytest tests/test_bifurcation.py -x` exits 0 | 16/16 passed |
| `pytest tests/ -x` exits 0 | 23/23 passed |
