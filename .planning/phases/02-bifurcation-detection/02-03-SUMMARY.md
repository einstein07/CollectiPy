---
phase: 02-bifurcation-detection
plan: "03"
subsystem: bifurcation-detection
tags: [bifurcation, detection-modes, behavioral, gradient, omega, sfa, ring-attractor, tdd]
dependency_graph:
  requires: [BifurcationDetector, bifurcation-wiring, events.json-output]
  provides: [behavioral-mode, gradient-criterion, omega-criterion, mode-dispatch]
  affects:
    - src/models/bifurcation.py
    - src/models/movement/mean_field_model.py
    - tests/test_bifurcation.py
tech_stack:
  added: []
  patterns:
    - "Mode dispatch: behavioral vs analytical via mode= config param"
    - "Behavioral alignment counter: alignment_consecutive_ticks criterion (D-02)"
    - "Gradient-of-lambda1 local-max detection for standard model (D-03)"
    - "Omega threshold crossing (Ermentrout et al. 2014 Eq 4.23/4.26) for SFA model (D-04)"
    - "Always-on metric logging: lambda1/Omega computed every tick regardless of mode (D-05)"
    - "TDD: failing tests committed before implementation"
key_files:
  created: []
  modified:
    - src/models/bifurcation.py
    - src/models/movement/mean_field_model.py
    - tests/test_bifurcation.py
decisions:
  - "update() always computes the analytical metric (lambda1 or Omega) before dispatching to mode-specific detection, honoring D-05 always-log requirement"
  - "_check_spike() preserved unchanged with old lambda1 event key for backward compat; new modes use metric+mode schema"
  - "test_gradient_window_respected corrected: original plan sequence was monotonically decreasing (no gradient local max); fixed to flat -> slow rise -> steep rise -> plateau"
metrics:
  duration: "~6 minutes"
  completed: "2026-04-12"
  tasks_completed: 3
  files_created: 0
  files_modified: 3
  tests_added: 14
---

# Phase 2 Plan 3: Detection Modes, Gradient Criterion, and Omega SFA Criterion Summary

**One-liner:** BifurcationDetector extended with behavioral alignment-counter mode and analytical mode (gradient-of-lambda1 for standard model, Omega threshold-crossing for SFA model per Ermentrout et al. 2014).

## What Was Built

### Task 1: BifurcationDetector Extension (`src/models/bifurcation.py`)

Extended the existing `BifurcationDetector` class with full mode dispatch and three new detection criteria:

- **`__init__` extended**: added `mode`, `alignment_tolerance_deg`, `alignment_consecutive_ticks`, `gradient_window`, `gradient_threshold` parameters. All have defaults; existing instantiation without new params still works (defaults to `mode="behavioral"`).

- **`update()` redesigned**: always computes the analytical metric first (lambda1 for standard model, Omega for SFA) regardless of mode — satisfying D-05 always-log requirement. Then dispatches to mode-specific detection.

- **`_update_behavioral()`** (new): alignment counter that tracks consecutive ticks where `|bump_angle - target_angle| <= alignment_tolerance_rad`. Resets counter on target switch or gap. Fires with `{agent, tick, metric=bump_angle, target, mode="behavioral"}` schema.

- **`_check_gradient()`** (new): discrete gradient `dL = (lambda1[t] - lambda1[t-gradient_window]) / gradient_window`. Detects local max of gradient above `gradient_threshold` using 3-sample buffer — fires at the steepest part of the lambda1 rise rather than the plateau.

- **`compute_omega()`** (new): computes `Omega = (1+beta)*A / ((1+beta)*A + I0 + 1e-12)` where A = first Fourier mode of `neural_ring`, I0 = first Fourier mode of perception/`mf.b`. Implements T-02-08 guard (`+1e-12`) and T-02-09 guard (None if no perception).

- **`_check_omega_crossing()`** (new): detects upward crossing of Hopf threshold `(1+alpha)/(1+beta)` using 3-sample buffer. Fires when `omega[t-2] < threshold <= omega[t-1]` confirmed by `omega[t]`.

- **`_update_analytical()`** (new): dispatches to gradient or Omega based on `mf.g_adapt > 0`.

- **`reset()` extended**: clears `_alignment_counter`, `_alignment_target`, `_lambda1_history`, `_gradient_history`, `_omega_buffer`, `last_omega`.

- **`_check_spike()` preserved unchanged**: backward compat for existing tests calling it directly via `feed_sequence()`. Uses old `lambda1` key schema (not `metric`+`mode`).

### Task 2: MeanFieldMovementModel Config Wiring (`src/models/movement/mean_field_model.py`)

- **`__init__`**: `BifurcationDetector` construction now passes all new config params: `mode`, `alignment_tolerance_deg`, `alignment_consecutive_ticks`, `gradient_window`, `gradient_threshold` from `mean_field_model.bifurcation` JSON namespace (D-09).

- **`step()`**: `bifurcation_detector.update()` now passes `perception_vec=self.perception` to enable Omega I0 computation (D-04).

- **`get_spin_system_data()`**: added `mean_field_omega` key — returns `bifurcation_detector.last_omega` for SFA agents (`g_adapt > 0`), else `None` (D-05).

### Task 3: New Tests (`tests/test_bifurcation.py`)

14 new tests added (total now 30 in file, 26 fast + 4 slow):

| Test | Coverage |
|------|---------|
| `test_behavioral_alignment_fires` | N consecutive aligned ticks fires behavioral event |
| `test_behavioral_alignment_resets_on_gap` | Counter resets when bump drifts out of tolerance |
| `test_behavioral_alignment_correct_target` | Correct target assigned when bump near pi |
| `test_behavioral_suppression` | Second fire within suppression window blocked |
| `test_gradient_fires_at_peak` | Gradient local max fires analytical event |
| `test_gradient_no_fire_flat` | Flat lambda1 produces no gradient event |
| `test_gradient_window_respected` | Gradient uses values window-ticks apart |
| `test_omega_computation_formula` | Omega matches analytical formula within 1e-6 |
| `test_omega_crossing_detection` | Omega upward crossing fires analytical event |
| `test_omega_no_fire_below_threshold` | Omega below threshold produces no event |
| `test_standard_model_analytical_gradient` | Standard model populates last_lambda1 (slow) |
| `test_sfa_model_analytical_omega` | SFA model populates last_omega (slow) |
| `test_new_event_schema_behavioral` | Behavioral event has {agent, tick, metric, target, mode} |
| `test_new_event_schema_analytical` | Analytical event has {agent, tick, metric, target, mode} |

Additionally updated two existing slow tests:
- `test_standard_model_bifurcation`: added `mode="analytical"` to preserve lambda1-based detection intent
- `test_sfa_model_bifurcation`: added `mode="analytical"` to preserve lambda1-based detection intent (oscillating SFA bump would fail 5-consecutive-tick behavioral criterion)

All 26 fast tests pass: `pytest tests/test_bifurcation.py -v` exits 0.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed `test_gradient_window_respected` test sequence**
- **Found during:** Task 3 GREEN phase (test failed with 0 events)
- **Issue:** The plan's test sequence `[-1.0, -0.9, -0.8, -0.7, -0.7, -0.7, -0.7]` produces a monotonically decreasing gradient (0.3, 0.283, 0.253, 0.02) — no local maximum exists, so `_check_gradient()` never fires. The 3-sample local-max check requires `g_prev < g_peak > g_confirm`.
- **Fix:** Replaced with `[-1.0, -1.0, -1.0, -0.95, -0.5, -0.1, -0.09, -0.09]` which produces gradient sequence (0.017, 0.167, 0.300, 0.287) — a genuine local max at gradient=0.300 confirmed by the fall to 0.287.
- **Files modified:** `tests/test_bifurcation.py`
- **Commit:** 5172756

## Known Stubs

None. All detection modes are fully implemented and tested. The `mean_field_omega` key is always computed for SFA agents and logged to pkl regardless of detection mode.

## Threat Flags

No new threat surface introduced beyond the plan's `<threat_model>`. All mitigations implemented:
- T-02-08: `+1e-12` denominator guard in `compute_omega()` — present
- T-02-09: `None`/empty check in `compute_omega()` when `perception_vec` unavailable — present
- T-02-11: SFA path only entered when `g_adapt > 0`; `MeanFieldSystem` validates `tau_adapt > 0` — present

## Self-Check: PASSED

| Check | Result |
|-------|--------|
| `src/models/bifurcation.py` contains `_update_behavioral` | FOUND (1 match) |
| `src/models/bifurcation.py` contains `_update_analytical` | FOUND (1 match) |
| `src/models/bifurcation.py` contains `compute_omega` | FOUND (1 match) |
| `src/models/bifurcation.py` contains `_check_gradient` | FOUND (1 match) |
| `src/models/bifurcation.py` contains `_check_omega_crossing` | FOUND (1 match) |
| `src/models/movement/mean_field_model.py` contains `mean_field_omega` | FOUND (1 match) |
| `src/models/movement/mean_field_model.py` mode config wiring | FOUND |
| Commit c28222e (Task 1) exists | FOUND |
| Commit ba912f5 (Task 2) exists | FOUND |
| Commit 264c28e (Task 3 RED) exists | FOUND |
| Commit 5172756 (Task 3 GREEN) exists | FOUND |
| `pytest tests/test_bifurcation.py -v` 26 passed, 4 skipped | PASSED |
