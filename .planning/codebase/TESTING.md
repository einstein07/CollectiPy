# Testing

**Analysis Date:** 2026-04-08

## Framework

**None configured.** There is no test framework, no test runner configuration, and no test files in the project source.

- No `pytest.ini`, `setup.cfg [tool:pytest]`, `pyproject.toml [tool.pytest]`, or `tox.ini` found
- No `tests/` directory
- No `test_*.py` or `*_test.py` files in `src/` or `plugins/`
- `pytest` is not listed in `requirements.txt`

## Test Structure

**Not applicable** — no tests exist in the project.

## Coverage

**Zero automated test coverage.**

No coverage tooling configured (no `.coveragerc`, no `coverage.ini`, no `pytest-cov`).

### Untested areas (entire codebase):

- `src/models/mean_field_systems.py` — MeanField ring attractor ODE integration, Numba-JIT paths
- `src/models/spinsystem.py` — Ising spin Hamiltonian computation
- `src/entity.py` — Entity UID generation (`make_agent_seed`, `splitmix32`, `_build_entity_uid`)
- `src/entityManager.py` — Tick-loop synchronisation, agent placement
- `src/config.py` — JSON config parsing and experiment expansion
- `src/dataHandling.py` — Output serialisation (pkl, csv, zip)
- `src/messagebus.py` — Agent messaging infrastructure
- `src/geometry_utils/` — Vector3D math, SpatialGrid spatial queries
- `src/models/movement/` — All movement model implementations
- `src/models/detection/` — Visual and GPS detection models
- `plugins/` — Plugin implementations

## Manual Testing Approach

Testing appears to be done by running full simulation experiments and visually inspecting GUI output or post-processing data files in `data/`. The `attention-2-beta/` and `data/` directories contain archived simulation runs used for validation.

## Recommendations

- Add `pytest` and write unit tests for pure-function utilities: `normalize_angle`, `_delta_angle`, `compute_center_of_mass`, `splitmix32`
- Add numerical tests for ODE integration stability (MeanFieldSystem, SpinSystem)
- Add integration tests for Config parsing with representative JSON fixtures
- Add tests for entity UID uniqueness guarantees

---

*Testing analysis: 2026-04-08*
