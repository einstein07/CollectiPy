---
plan: 01-04
phase: 01-foundation
status: complete
completed: 2026-04-10
commits:
  - a78cc4b
  - d36a8a2
---

# Plan 01-04: Regression Snapshot Tests

## What Was Built

Regression snapshot tests (TEST-03) for SpinSystem and the `levy()` movement utility:

- **test_spinsystem_spins_snapshot** — SpinSystem(seed=42) run for 100 Metropolis steps; spin array matches `tests/fixtures/spinsystem_spins.npy`
- **test_spinsystem_avg_direction_snapshot** — `average_direction_of_activity()` result matches `tests/fixtures/spinsystem_avg_direction.npy` (NaN-safe for None return)
- **test_random_walk_levy_snapshot** — `levy(rng=Random(123), c=1.0, alpha=1.5)` sampled 200 times matches `tests/fixtures/random_walk_state.npy`

Also created:
- `tests/conftest.py` — `--regen` CLI flag to regenerate snapshots when model behaviour intentionally changes
- `tests/fixtures/.gitkeep` — ensures the fixtures directory is tracked before .npy files exist
- `tests/fixtures/*.npy` — three committed reference snapshots

## Key Files Created

- `tests/test_regression.py`
- `tests/conftest.py`
- `tests/fixtures/.gitkeep`
- `tests/fixtures/spinsystem_spins.npy`
- `tests/fixtures/spinsystem_avg_direction.npy`
- `tests/fixtures/random_walk_state.npy`

## Deviations From Plan

- Used `models.utils.levy()` directly (as specified in the plan interfaces section) instead of attempting to construct `RandomWalkMovement` with a full agent stack, which would require the entire simulation infrastructure
- Root conftest.py (from plan 01-03) provides the `src/` path setup; `tests/conftest.py` only adds the `--regen` option — no conflict

## Verification

```
pytest tests/test_regression.py --regen -v   → 3 passed (snapshots generated)
pytest tests/test_regression.py -v           → 3 passed (stable against committed fixtures)
```

Both runs completed in 0.19s (no Numba — SpinSystem and levy() are pure Python/NumPy).

## Self-Check

All 3 snapshot tests passing. Fixtures committed. --regen flag works.
