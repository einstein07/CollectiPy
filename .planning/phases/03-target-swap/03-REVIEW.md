---
phase: 03-target-swap
reviewed: 2026-04-13T00:00:00Z
depth: standard
files_reviewed: 9
files_reviewed_list:
  - src/arena.py
  - src/dataHandling.py
  - src/models/bifurcation.py
  - src/models/movement/mean_field_model.py
  - tests/conftest.py
  - tests/test_bifurcation.py
  - tests/test_mean_field_system.py
  - tests/test_regression.py
  - tests/test_target_swap.py
findings:
  critical: 0
  warning: 5
  info: 5
  total: 10
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-04-13
**Depth:** standard
**Files Reviewed:** 9
**Status:** issues_found

## Summary

Phase 3 adds post-bifurcation target swap logic. The arena monitors each tick for the first bifurcation event from any agent, then schedules a swap of two target positions AND their strength values at `bifurcation_tick + delay_ticks`. The event is logged to `events.json` under `swap_events`.

The implementation is structurally sound. The core swap path — scheduling, executing, and logging — is correct. `_post_bif_swap_triggered` is reset at run start in both the `run()` loop body (line 989) and `SolidArena.reset()` (line 1161). `SolidArena.reset()` flushes `_latest_queue_bif_events` before calling `data_handling.close()` as required.

The findings below are real bugs and code quality issues, not style preferences. There are no critical security or data-loss bugs, but there are two logic errors that will cause wrong runtime behavior: a precedence bug in the suppression check in `BifurcationDetector` and a `_post_bif_swap_triggered` flag that is redundantly reset in `run()` but then overwritten by `reset()` — the double-reset sequence is fine, but the flag state at the end of `reset()` matters.

---

## Warnings

### WR-01: Operator-precedence bug silences behavioral bifurcation suppression

**File:** `src/models/bifurcation.py:394`
**Issue:** The suppression check in `_update_behavioral` uses `and`/`or` without parentheses, which causes incorrect short-circuit evaluation:

```python
if (self._last_fire_tick is not None
        and tick - self._last_fire_tick < self.spike_min_separation or self._last_fire_tick is not None and not self.retrigger):
    return None
```

Python evaluates this as:
```
(A and B) or (A and C)
```
which means the condition returns `True` (suppressing the event) whenever `_last_fire_tick is not None and not self.retrigger`, even if the separation window has already elapsed. This blocks re-triggering even after `retrigger` is set to `True` and the separation window has passed. The identical bug exists in `_update_behavioral_agent_angle` at line 474.

The intended logic is: suppress if (within separation window) OR (not re-triggerable). That requires explicit grouping:

```python
within_window = (
    self._last_fire_tick is not None
    and tick - self._last_fire_tick < self.spike_min_separation
)
not_retriggerable = self._last_fire_tick is not None and not self.retrigger
if within_window or not_retriggerable:
    return None
```

**Fix:**
```python
# _update_behavioral, line ~393
if (
    (self._last_fire_tick is not None
     and tick - self._last_fire_tick < self.spike_min_separation)
    or (self._last_fire_tick is not None and not self.retrigger)
):
    return None
```
Apply the same fix to `_update_behavioral_agent_angle` at line 474.

---

### WR-02: `_update_behavioral` is dead code — call site is commented out

**File:** `src/models/bifurcation.py:733-734`
**Issue:** In the `update()` dispatch, the call to `_update_behavioral` is commented out and replaced with `_update_behavioral_agent_angle`:

```python
#return self._update_behavioral(tick, bump_angle, target_angles, target_ids)
return self._update_behavioral_agent_angle(tick, agent_angle, target_angles, target_ids)
```

`_update_behavioral` is now unreachable from the normal execution path. Its tests in `test_bifurcation.py` do not directly call the `update()` dispatch, so the bug is masked. If `mode="behavioral"` is configured, the system silently uses agent heading angle instead of bump angle, which is a behavioral change from what "behavioral" mode implies.

**Fix:** Either restore the call to `_update_behavioral` (removing the agent-angle variant), or rename `_update_behavioral_agent_angle` to `_update_behavioral` and remove the commented line. Document the design intent explicitly.

---

### WR-03: `_apply_target_position_swap_event` does NOT swap `strength` — inconsistent with post-bif swap

**File:** `src/arena.py:561-582`
**Issue:** The time-scheduled swap path (`_apply_target_position_swap_event`) only swaps XY positions:

```python
self._swap_object_xy_positions(left_obj, right_obj)
```

The post-bifurcation swap path (`_execute_post_bif_swap`, line 509) additionally swaps `entity.strength`:

```python
self._swap_object_xy_positions(left_obj, right_obj)
left_obj.strength, right_obj.strength = right_obj.strength, left_obj.strength
```

If both swap types are used in the same experiment, the time-scheduled swaps will leave `strength` inconsistent with position. This is not documented as an intentional difference in the config schema or the method docstrings. If a researcher sets up `target_position_swaps` expecting the same semantics as `post_bifurcation_swap`, the results will be silently wrong.

**Fix:** Either (a) add strength swap to `_apply_target_position_swap_event` for consistency, or (b) add a docstring to `_apply_target_position_swap_event` explicitly stating it does NOT swap `strength`, and add a note to the config schema documentation.

---

### WR-04: `_post_bif_swap_triggered` reset in `run()` is partially redundant with `reset()` but different code paths diverge

**File:** `src/arena.py:988-990`
**Issue:** In `SolidArena.run()`, two consecutive lines reset the post-bif swap state at run start:

```python
self._prepare_target_position_swaps_for_run()
# Reset post-bifurcation swap state for this run (D-04: once per run)
self._post_bif_swap_triggered = False
self._post_bif_swap_event = None
```

`SolidArena.reset()` (line 1161) also resets these. For run > 1, the flow is: `reset()` is called (line 1125) then `run()` starts the next iteration and resets again. That double-reset is harmless. However, the `AbstractArena` subclass does not override `run()`, so any arena type using `AbstractArena` would rely solely on `reset()` — which is fine. The concern is that the `run()` body reset is the _only_ guard for `run == 1` since `reset()` is not called before the first run starts. Without the `run()` body reset, if `Arena.__init__` initializes `_post_bif_swap_triggered = False` (line 98), that is sufficient. The double-reset in `run()` is therefore a defensive redundancy.

This is not a bug as-is, but the duplicated reset across `__init__`, `run()`, and `reset()` makes the invariant hard to reason about. A future refactor that moves the reset out of one of these locations could silently introduce a bug.

**Fix:** Extract the state reset into a dedicated `_reset_post_bif_swap_state()` helper and call it from `run()` at the top of each run iteration and from `reset()`. This centralises the invariant.

---

### WR-05: `BifurcationDetector.reset()` does not reset `retrigger` flag

**File:** `src/models/bifurcation.py:757-775`
**Issue:** The `reset()` method clears all tracking state except `self.retrigger`:

```python
def reset(self) -> None:
    self.events.clear()
    self._buffer.clear()
    self._last_fire_tick = None
    self.last_lambda1 = None
    self._alignment_counter = 0
    self._alignment_target = None
    self._lambda1_history.clear()
    self._gradient_history.clear()
    self._omega_buffer.clear()
    self.last_omega = None
```

`self.retrigger` is set to `True` during target switching (lines 381, 461) and to `False` when alignment is lost (lines 388, 468). If `reset()` is called between runs, `retrigger` retains its value from the previous run. Combined with WR-01, this can cause incorrect suppression or premature firing on the first tick of a new run if the detector is reused across runs.

**Fix:**
```python
def reset(self) -> None:
    ...
    self.retrigger = False  # add this line
```

---

## Info

### IN-01: Commented-out call in `update()` dispatch should be removed or converted to a docstring note

**File:** `src/models/bifurcation.py:733`
**Issue:** The line `#return self._update_behavioral(tick, bump_angle, target_angles, target_ids)` is a commented-out code path. It constitutes dead code in source control and will confuse future readers about which method is actually called in behavioral mode.

**Fix:** Remove the commented line. If the bump-angle variant should be an option in the future, add a comment documenting the design decision instead.

---

### IN-02: `_normalize_post_bif_swap_config` accepts only list/tuple pairs — dict pairs supported by the time-scheduled path are silently rejected

**File:** `src/arena.py:390-405`
**Issue:** `_normalize_post_bif_swap_config` only accepts 2-element `list`/`tuple` pairs:

```python
if not isinstance(pair, (list, tuple)) or len(pair) != 2:
    raise ValueError(...)
```

The time-scheduled sibling `_normalize_target_position_swaps` (lines 319-361) accepts both list/tuple pairs AND dict pairs (`{"a": "target_A", "b": "target_B"}`). This inconsistency in the config API means a researcher using the same pair format across both config keys will get an unexpected `ValueError` for the post-bifurcation config.

**Fix:** Extend `_normalize_post_bif_swap_config` to also accept dict pairs using the same key aliases (`a`, `b`, `target_a`, `target_b`, etc.) as `_normalize_target_position_swaps`.

---

### IN-03: `test_collect_swap_events_and_write` manually constructs `run_folder` instead of using `new_run()`

**File:** `tests/test_target_swap.py:408-413`
**Issue:** The test bypasses `DataHandling.new_run()` and directly sets `dh.run_folder` and `dh._bifurcation_events`:

```python
run_folder = os.path.join(dh.config_folder, "run_1")
os.makedirs(run_folder, exist_ok=True)
dh.run_folder = run_folder
dh._bifurcation_events = []
dh._swap_events = []
```

This relies on private attributes and bypasses state initialization that `new_run()` performs. If `new_run()` changes its internal structure, this test will silently pass while real behavior changes.

**Fix:** Use `DataHandling.new_run()` with minimal stub `shapes`/`spins`/`metadata` arguments, or add a short comment explaining why the manual path is intentional here.

---

### IN-04: `_MockDataHandling` in `test_bifurcation.py` duplicates the one in `test_target_swap.py`

**File:** `tests/test_bifurcation.py:262-282`, `tests/test_target_swap.py:76-87`
**Issue:** Both test files define a `_MockDataHandling` class with different interfaces. The one in `test_bifurcation.py` has `_write_events_json()` hardcoded and does not include `collect_swap_events()`. The one in `test_target_swap.py` includes `collect_swap_events()` but no `_write_events_json()`.

This duplication means changes to `DataHandling`'s contract require updates in two places, and the two mocks can diverge silently.

**Fix:** Move shared mock infrastructure to `tests/conftest.py` as fixtures, or to a dedicated `tests/helpers.py` module.

---

### IN-05: `import logging` vs `logging.info` used inconsistently in arena.py swap methods

**File:** `src/arena.py:466`, `502`, `510`
**Issue:** In `_check_post_bif_swap` and `_execute_post_bif_swap`, the module-level `import logging` is used directly (`logging.info(...)`, `logging.warning(...)`). The rest of the codebase in arena.py uses the same pattern, so this is consistent within the file. However, `dataHandling.py` uses a named logger `logger = logging.getLogger("sim.data_handling")`, while arena.py uses the root `logging` module directly for all swap-related log messages.

This means swap log messages are not filterable by logger name (no `sim.arena` namespace) and will always appear in the root logger output, potentially drowning out other log channels.

**Fix:** Add `logger = logging.getLogger("sim.arena")` at module level in `arena.py` and use it in all swap-related log calls.

---

_Reviewed: 2026-04-13_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
