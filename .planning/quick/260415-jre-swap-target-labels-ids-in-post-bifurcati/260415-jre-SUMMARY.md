---
phase: quick-260415-jre
plan: "01"
subsystem: arena/entity
tags: [swap, entity, label, post-bifurcation, tdd]
dependency_graph:
  requires: []
  provides: [Entity.set_name, arena-name-swap]
  affects: [src/arena.py, src/entity.py, tests/test_target_swap.py]
tech_stack:
  added: []
  patterns: [set_name-metadata-sync]
key_files:
  created: []
  modified:
    - src/entity.py
    - src/arena.py
    - tests/test_target_swap.py
    - .planning/phases/03-target-swap/03-CONTEXT.md
decisions:
  - "Entity.set_name() added to base class so all entity subclasses inherit label mutation without override"
  - "Name swap in _execute_post_bif_swap uses left_obj.set_name(right_id) / right_obj.set_name(left_id) — right_id and left_id are the original pair IDs captured before any mutation"
metrics:
  duration: "~2 minutes"
  completed: "2026-04-15T12:18:19Z"
  tasks_completed: 3
  files_changed: 4
---

# Quick Task 260415-jre: Swap Target Labels/IDs in Post-Bifurcation Swap Summary

**One-liner:** Added `Entity.set_name()` with shape.metadata sync and wired it into `_execute_post_bif_swap` so entity labels exchange along with positions and strength, backed by a new regression test and updated D-02 in CONTEXT.md.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add set_name() to Entity | 7581d0f | src/entity.py |
| 2 | Swap entity names in _execute_post_bif_swap | 25c6d69 | src/arena.py, tests/test_target_swap.py |
| 3 | Add name-swap test and update CONTEXT.md D-02 | edc3bdb | tests/test_target_swap.py, .planning/phases/03-target-swap/03-CONTEXT.md |

## What Was Built

### Entity.set_name() (src/entity.py)

Added immediately after `get_name()` on the `Entity` base class:

- Sets `self._entity_uid = uid`
- Syncs `shape.metadata["entity_name"]` if the entity has a `shape` with `metadata`
- Uses `getattr(self, "shape", None)` guard — entities without a shape attribute are unaffected (no AttributeError)

### _execute_post_bif_swap name swap (src/arena.py)

After the existing strength swap, two calls are added:

```python
left_obj.set_name(right_id)
right_obj.set_name(left_id)
```

The log message was updated to say "positions + strength + names". The `left_id`/`right_id` variables hold the original pair IDs from the event dict, so their values are stable even after `set_name` runs on `left_obj`.

### Test coverage (tests/test_target_swap.py)

- `set_name()` added to `_MockEntity` (mirrors the real Entity interface)
- `test_execute_post_bif_swap_names` added after `test_execute_post_bif_swap_strength`
- Module docstring updated to list the new test
- Total: 21 tests, all passing

### D-02 update (.planning/phases/03-target-swap/03-CONTEXT.md)

Heading and body updated to state that labels ARE now swapped. Before/after example extended with `name=` fields. Implementation note added explaining `set_name()` and `shape.metadata['entity_name']` sync.

## Verification

```
21 passed in 0.22s
```

All 20 pre-existing tests continue to pass. New test `test_execute_post_bif_swap_names` passes.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] _MockEntity missing set_name() broke Task 2 verification**

- **Found during:** Task 2 (GREEN run)
- **Issue:** `_execute_post_bif_swap` now calls `set_name()` on entities; `_MockEntity` in tests did not have the method, causing `AttributeError` on 8 existing tests
- **Fix:** Added `set_name(uid)` to `_MockEntity` as part of Task 2's commit (before Task 3 which planned to add it)
- **Files modified:** tests/test_target_swap.py
- **Commit:** 25c6d69

This is the only deviation. The fix is identical to what Task 3 planned — it was simply applied one task earlier to unblock verification.

## Decisions Made

1. **Entity.set_name() on base class** — All entity subclasses (StaticAgent, MovableAgent, StaticObject, MovableObject) inherit label mutation without needing per-class overrides.
2. **Swap uses pre-mutation IDs** — `left_id` and `right_id` are read from `event.get("pairs", [])` before any `set_name` call, so the swap is correct regardless of order.

## Known Stubs

None.

## Threat Flags

None — no new network endpoints, auth paths, file access patterns, or schema changes introduced.

## Self-Check: PASSED

- src/entity.py: FOUND
- src/arena.py: FOUND
- tests/test_target_swap.py: FOUND
- .planning/phases/03-target-swap/03-CONTEXT.md: FOUND
- Commit 7581d0f: FOUND
- Commit 25c6d69: FOUND
- Commit edc3bdb: FOUND
