---
plan: 01-03
phase: 01-foundation
status: complete
completed: 2026-04-10
commits:
  - c97544b
  - 7d6d249
---

# Plan 01-03: MeanFieldSystem Unit Tests

## What Was Built

Pytest unit test suite for `MeanFieldSystem` (TEST-01) with four test functions:

1. **test_fixed_point_zero_input** — Verifies zero sensory input keeps ring state bounded near zero
2. **test_bump_forms_from_uniform_ic** — Verifies single strong target creates non-uniform bump (spread > 0.1)
3. **test_single_target_response** — Verifies bump angle converges within π/8 of target direction
4. **test_divergence_raises_runtime_error** — Verifies `step()` raises `RuntimeError("MeanFieldSystem diverged at tick ...")` on NaN/Inf

Also created:
- `conftest.py` (project root) — adds `src/` to `sys.path` for pytest 6.x compatibility (pytest 7+ `pythonpath` option not available)
- `tests/__init__.py` — makes tests a package

## Key Files Created

- `tests/test_mean_field_system.py`
- `tests/__init__.py`
- `conftest.py`

## Deviations From Plan

- Root `conftest.py` was added (not in plan) to fix pytest 6.2.5 import resolution — `pythonpath` in `pyproject.toml` is a pytest 7.0+ feature
- `cache=True` added to `@njit(parallel=True)` in `mean_field_systems.py` — first-run JIT compilation took 8+ minutes without disk cache; this makes subsequent runs fast
- Tests run against system Python 3.10 with numpy 1.26.4 (installed via pip; system numpy 1.21 was too old for numba)

## Verification

```
pytest tests/test_mean_field_system.py -v
→ 4 passed (first run ~8 min due to Numba JIT; subsequent runs fast with cache)
```

## Self-Check

All 4 required tests present and passing. Divergence test covers D-01..D-03 from CONTEXT.md.
