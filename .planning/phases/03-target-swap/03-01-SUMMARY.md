---
phase: 03-target-swap
plan: 01
subsystem: arena-swap
tags: [bifurcation, swap, events, mean-field, arena]
dependency_graph:
  requires: []
  provides: [post-bifurcation-swap, bifurcation-event-ipc, swap-events-json]
  affects: [arena.py, dataHandling.py, mean_field_model.py, entityManager.py]
tech_stack:
  added: [BifurcationDetector]
  patterns: [path-a-ipc-via-spin-data, once-per-run-flag, collect-then-write-events]
key_files:
  created:
    - src/models/bifurcation.py
    - tests/test_bifurcation.py
    - tests/test_mean_field_system.py
    - tests/test_regression.py
    - tests/conftest.py
  modified:
    - src/models/movement/mean_field_model.py
    - src/arena.py
    - src/dataHandling.py
decisions:
  - "Use Path A IPC: bifurcation events flow through per-tick spin data (get_spin_system_data new_bifurcation_events key), not EntityManager-level queue"
  - "BifurcationDetector ported from main repo (Phase 2 prerequisite); worktree was pre-Phase2"
  - "Events drain from detector.events each tick so they are not double-counted across ticks"
  - "post_bifurcation_swap config is optional; absent key means no swap (backward compatible)"
  - "Swap fires exactly once per run via _post_bif_swap_triggered flag reset at run start"
  - "Both XY position and strength (intensity) are swapped on target entity objects"
metrics:
  duration: ~35 minutes
  completed: 2026-04-13
  tasks_completed: 2
  files_modified: 4
  files_created: 6
---

# Phase 3 Plan 01: Post-Bifurcation Target Swap — Summary

**One-liner:** Bifurcation-triggered once-per-run swap of target XY positions and strength (intensity), logged to events.json, using Path A IPC via per-tick spin data.

## What Was Built

Post-bifurcation target swap lifecycle from config to logged event:

1. **BifurcationDetector** (`src/models/bifurcation.py`) — Ported from Phase 2 (main repo). Detects bifurcation events using behavioral (bump alignment) or analytical (gradient/Omega) criteria. Events accumulate in `detector.events`.

2. **MeanFieldMovementModel wiring** (`src/models/movement/mean_field_model.py`):
   - `BifurcationDetector` instantiated in `__init__` from `mean_field_model.bifurcation` config namespace
   - `bifurcation_detector.reset()` called in `reset()`
   - `bifurcation_detector.update()` called each tick after bump angle is computed
   - `get_spin_system_data()` now drains `detector.events` into `data["new_bifurcation_events"]` each tick — this is the Path A IPC mechanism

3. **Arena swap lifecycle** (`src/arena.py`):
   - `_normalize_post_bif_swap_config()`: validates `environment.post_bifurcation_swap` config; returns None for absent key (backward compat), raises ValueError for malformed input
   - `_find_first_bifurcation_in_snapshots()`: scans `agents_spins[grp][idx]["new_bifurcation_events"]` across all manager snapshots, returns earliest-tick event
   - `_check_post_bif_swap()`: called each tick; on first bifurcation found, sets `_post_bif_swap_triggered=True` and schedules swap at `bif_tick + delay_ticks`; when due, calls `_execute_post_bif_swap()`
   - `_execute_post_bif_swap()`: swaps XY positions (via `_swap_object_xy_positions`) and `entity.strength` values on target pairs; calls `data_handling.collect_swap_events()`
   - `_collect_bifurcation_events()`: fallback drain from in-process detectors (for same-process test scenarios)
   - `close()` and `reset()` flush bifurcation events to DataHandling before writing events.json
   - `_post_bif_swap_triggered` and `_post_bif_swap_event` reset to False/None at run start

4. **DataHandling event logging** (`src/dataHandling.py`):
   - `_bifurcation_events` and `_swap_events` lists, reset in `new_run()`
   - `collect_bifurcation_events()`: accumulates bifurcation events
   - `collect_swap_events()`: accumulates swap events
   - `_write_events_json()`: writes `{"bifurcation_events": [...], "swap_events": [...]}` to `events.json` in the run folder
   - `close()` now calls `_write_events_json()` before archiving

5. **Tests** (`tests/`): Copied Phase 2 test suite (33 tests pass, 4 skipped slow tests).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] BifurcationDetector prerequisite absent from worktree**
- **Found during:** Task 1 start
- **Issue:** Worktree was branched from `a672615` (pre-Phase2), so `src/models/bifurcation.py`, bifurcation wiring in mean_field_model, and DataHandling/Arena bifurcation infrastructure were all absent. Plan assumed Phase 2 was already merged.
- **Fix:** Ported `bifurcation.py` from main repo. Added full BifurcationDetector wiring to mean_field_model.py. Added `collect_bifurcation_events`, `_write_events_json` to DataHandling. Added `_collect_bifurcation_events`, `_latest_queue_bif_events` to Arena.
- **Files modified:** `src/models/bifurcation.py` (new), `src/models/movement/mean_field_model.py`, `src/dataHandling.py`, `src/arena.py`
- **Commit:** `4e59663`

**2. [Rule 2 - Missing functionality] Test suite absent from worktree**
- **Found during:** Task 2 verification
- **Issue:** Plan verification step says `python -m pytest ../tests/` but no `tests/` directory existed in worktree.
- **Fix:** Copied tests directory from main repo.
- **Files modified:** `tests/` (all files new)
- **Commit:** `0076a64`

**3. [Rule 2 - Missing functionality] DataHandling.close() did not write events.json**
- **Found during:** Task 2 implementation
- **Issue:** `DataHandling.close()` was a stub that only archived the run folder. `_write_events_json()` needed to be called before archiving to ensure events.json is in the zip.
- **Fix:** Added `self._write_events_json()` call at start of `DataHandling.close()`.
- **Commit:** `0076a64`

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| Task 1 | `4e59663` | feat(03-01): propagate bifurcation events through agent snapshots (Path A D-06) |
| Task 2 | `0076a64` | feat(03-01): Arena post-bifurcation swap config, scheduling, and execution |

## Verification Results

All plan acceptance criteria pass:
- `Arena._normalize_post_bif_swap_config(None)` returns None
- `Arena._normalize_post_bif_swap_config({"pairs": [["A","B"]], "delay_ticks": 10})` returns `{"pairs": [("A","B")], "delay_ticks": 10}`
- All 4 new Arena methods exist: `_normalize_post_bif_swap_config`, `_check_post_bif_swap`, `_find_first_bifurcation_in_snapshots`, `_execute_post_bif_swap`
- `DataHandling.collect_swap_events` exists
- `_swap_events` initialized in `new_run()`, collected in `collect_swap_events`, written in `_write_events_json`
- `left_obj.strength, right_obj.strength = right_obj.strength, left_obj.strength` present in `_execute_post_bif_swap`
- `_check_post_bif_swap(t, latest_agent_data)` called in tick loop
- `new_bifurcation_events` read from agent snapshots in `_find_first_bifurcation_in_snapshots`
- Test suite: 33 passed, 4 skipped (slow), 0 failures

## Known Stubs

None. All data flows are wired: bifurcation detection → spin data → Arena → swap execution → DataHandling → events.json.

## Threat Flags

None beyond the plan's threat model (T-03-01 through T-03-04 all addressed):
- T-03-01 (config tampering): `_normalize_post_bif_swap_config` validates all inputs, raises ValueError on malformed
- T-03-03 (DoS via crash): None config returns None silently; existing configs unchanged

## Self-Check: PASSED

- FOUND: src/models/bifurcation.py
- FOUND: src/models/movement/mean_field_model.py
- FOUND: src/arena.py
- FOUND: src/dataHandling.py
- FOUND: commit 4e59663 (Task 1)
- FOUND: commit 0076a64 (Task 2)
