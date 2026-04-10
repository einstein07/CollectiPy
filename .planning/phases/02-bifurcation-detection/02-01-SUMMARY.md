---
phase: 02-bifurcation-detection
plan: "01"
subsystem: bifurcation-detection
tags: [bifurcation, eigenvalue, jacobian, ring-attractor, tdd, unit-tests]
dependency_graph:
  requires: []
  provides: [BifurcationDetector, bifurcation-spike-detection]
  affects: [src/models/bifurcation.py, tests/test_bifurcation.py]
tech_stack:
  added: [pytest==9.0.3]
  patterns:
    - "3-sample deque local-maximum spike detection"
    - "z-subspace Jacobian eigenvalue (np.linalg.eigvals) for bifurcation detection"
    - "TDD: failing tests committed before implementation"
    - "try/except around eigvals for degenerate-matrix resilience (T-02-02)"
key_files:
  created:
    - src/models/bifurcation.py
    - tests/test_bifurcation.py
  modified: []
decisions:
  - "Split update() into compute_lambda1() + _check_spike() for testability — allows unit tests to feed synthetic lambda sequences without a MeanFieldSystem"
  - "T-02-02 mitigation: eigvals wrapped in try/except; NaN Jacobian logs warning and returns None (skips tick rather than crashing)"
  - "Unified Jacobian formula (diag((u-s)*sech2) @ M) handles both standard (s=0) and SFA (s=adapt_ring) in one code path"
metrics:
  duration: "~8 minutes"
  completed: "2026-04-10"
  tasks_completed: 1
  files_created: 2
  tests_added: 10
---

# Phase 2 Plan 1: BifurcationDetector Class and Unit Tests Summary

**One-liner:** BifurcationDetector using z-subspace Jacobian eigenvalue lambda_1 spike detection with 3-sample deque buffer, re-triggerable with suppression, and per-agent event dicts.

## What Was Built

### `src/models/bifurcation.py`

A standalone `BifurcationDetector` class implementing the core bifurcation detection algorithm from Phase 2 design decisions (D-01 through D-05):

- **`compute_jacobian(mf)`** — computes the z-subspace Jacobian per D-02:
  `J = -I + diag((u - s) * sech^2((u - s)*M@z + b - beta)) @ M`
  Single code path handles standard model (s=0) and SFA model (s=adapt_ring).

- **`compute_lambda1(mf)`** — calls `np.linalg.eigvals(J)` and returns the maximum real part Re(lambda_1). Implements T-02-02 mitigation: wrapped in try/except; returns None (skips tick) on degenerate matrix or NaN.

- **`_check_spike(tick, lambda1, bump_angle, target_angles, target_ids)`** — the spike-detection logic separated for testability. Uses a `deque(maxlen=3)` buffer to detect local maxima above `lambda_threshold`, with `spike_min_separation` suppression (D-03, D-04). Assigns nearest target via circular distance (D-05).

- **`update(tick, mf, bump_angle, target_angles, target_ids)`** — integrates compute_lambda1 + _check_spike into the single-call API used by the simulation.

- **`_nearest_target(bump_angle, target_angles, target_ids)`** — circular-distance argmin matching `_delta_angle` convention from `mean_field_systems.py`.

### `tests/test_bifurcation.py`

10 unit tests covering all specified behaviors:

| Test | Coverage |
|------|---------|
| `test_spike_detection` | Local max above threshold fires exactly one event |
| `test_no_spike_below_threshold` | Local max below threshold suppressed |
| `test_no_spike_monotonic` | Monotonic sequence: no events |
| `test_spike_min_separation` | Second spike within window suppressed |
| `test_retriggerable` | Two spikes > separation: both fire |
| `test_jacobian_computation_standard` | Standard model Jacobian is finite (n,n) ndarray |
| `test_jacobian_computation_sfa` | SFA Jacobian is finite and differs from standard |
| `test_target_assignment` | Nearest target resolves correctly via circular distance |
| `test_event_dict_schema` | Event dict has keys {agent, tick, lambda1, target} with correct types |
| `test_no_bifurcation_empty_list` | Flat sequence: events list empty |

All 10 tests pass: `pytest tests/test_bifurcation.py -x -v` exits 0.

## Deviations from Plan

### Auto-added (Rule 2 — Threat Model Mitigation)

**1. [Rule 2 - Security/Robustness] T-02-02: eigvals wrapped in try/except with NaN Jacobian check**
- **Found during:** Task 1 implementation (threat model mandates `mitigate` disposition)
- **Issue:** `np.linalg.eigvals` can fail on degenerate matrices; NaN in Jacobian would propagate silently
- **Fix:** `compute_lambda1()` checks `np.isfinite(J)` and wraps `eigvals` in `try/except (LinAlgError, ValueError)`. On failure: logs warning, returns None. `update()` checks for None and skips the tick.
- **Files modified:** `src/models/bifurcation.py`
- **Commit:** 8717aff

### Design Choice (not in plan)

**`_check_spike()` as a separate method:** The plan noted in the action that `update()` should be split into compute + detect for testability. Implemented as `_check_spike()` (internal but accessible to tests) — matches the plan's intent exactly.

## Known Stubs

None. The `BifurcationDetector` is fully implemented and tested. Integration into `MeanFieldMovementModel` and `events.json` writing are in Plan 02-02.

## Threat Flags

No new threat surface introduced beyond what the `<threat_model>` covers. T-02-02 mitigation is in place.

## Self-Check: PASSED

| Check | Result |
|-------|--------|
| `src/models/bifurcation.py` exists | FOUND |
| `tests/test_bifurcation.py` exists | FOUND |
| Commit a5d12c9 (tests, RED) exists | FOUND |
| Commit 8717aff (implementation, GREEN) exists | FOUND |
| `pytest tests/test_bifurcation.py` exits 0 | 10/10 passed |
