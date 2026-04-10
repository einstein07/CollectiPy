---
phase: 01-foundation
plan: 02
subsystem: infra
tags: [filesystem, concurrency, os.makedirs, atomic-operations]

# Dependency graph
requires: []
provides:
  - Atomic config_folder_N directory creation in DataHandling.__init__() using os.makedirs(exist_ok=False) retry loop
affects: [all phases that run parallel sweep experiments writing to ./data/]

# Tech tracking
tech-stack:
  added: []
  patterns: [atomic-claim-by-creation — claim a resource by trying to create it atomically rather than scan-then-create]

key-files:
  created: []
  modified:
    - src/dataHandling.py

key-decisions:
  - "Use os.makedirs(exist_ok=False) in a retry loop as the POSIX-atomic way to claim a unique directory name without a TOCTOU window"
  - "Start folder_id at 0 and increment on FileExistsError — handles both concurrent collisions and gaps from manual deletion"

patterns-established:
  - "Atomic-claim pattern: try os.makedirs(exist_ok=False); on FileExistsError increment counter and retry"

requirements-completed: [PERF-02]

# Metrics
duration: 5min
completed: 2026-04-09
---

# Phase 01 Plan 02: Atomic Config Folder Creation Summary

**Replaced scan-then-mkdir TOCTOU race in DataHandling.__init__() with os.makedirs(exist_ok=False) retry loop, eliminating silent output corruption in parallel sweep runs**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-04-09T00:00:00Z
- **Completed:** 2026-04-09T00:05:00Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Eliminated the TOCTOU race window between `os.listdir` scan and `os.mkdir` in `DataHandling.__init__()`
- Fixed the broken `len(existing)` heuristic that reused indices when folders were manually deleted
- Atomic `os.makedirs(exist_ok=False)` guarantees that two concurrent processes targeting the same base_path will always claim different `config_folder_N` names

## Task Commits

Each task was committed atomically:

1. **Task 1: Replace scan-then-mkdir with atomic os.makedirs retry loop** - `0fcf13b` (fix)

## Files Created/Modified

- `src/dataHandling.py` - Lines 58-65: replaced listdir scan + os.mkdir block with os.makedirs(exist_ok=False) retry loop

## Decisions Made

- `os.makedirs(exist_ok=False)` chosen over `os.mkdir` because it handles the case where the base_path itself may have sub-components, and raises `FileExistsError` atomically on POSIX — no separate existence check needed
- Starting `folder_id` at 0 on every call is intentional: the retry loop skips occupied indices, so sequential single-process runs still produce 0, 1, 2, ... in order

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- PERF-02 resolved; parallel sweep runs will no longer silently corrupt output by racing to the same `config_folder_N`
- No follow-up actions needed for this fix

---
*Phase: 01-foundation*
*Completed: 2026-04-09*
