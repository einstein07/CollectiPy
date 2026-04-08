# CollectiPy

## What This Is

CollectiPy is a Python simulation framework for studying collective decision-making in multi-agent swarms using ring attractor neural dynamics. Agents implement biologically-inspired attractor models (Ising spin system, mean-field ring attractor) and interact in a configurable arena. It is used by a PhD researcher to run decision-making experiments — binary choices and N-option competition — and generate results for publications.

## Core Value

Agents running ring attractor dynamics should produce measurable, reproducible collective decisions that can be systematically explored via parameter sweeps.

## Requirements

### Validated

- ✓ Discrete-time multi-agent simulation with configurable tick loop — existing
- ✓ Plugin-based behavioural model system (MovementModel, LogicModel, DetectionModel) — existing
- ✓ Ising spin system (SpinSystem) for binary decision dynamics — existing
- ✓ Mean-field ring attractor model (MeanFieldSystem) with Numba JIT acceleration — existing
- ✓ Unicycle kinematic motion model — existing
- ✓ Visual and GPS detection models — existing
- ✓ Multi-process execution (one process per agent group) with IPC synchronisation — existing
- ✓ PySide6/Qt6 live visualisation with neural activation plots — existing
- ✓ JSON-based experiment configuration system — existing
- ✓ Output serialisation to pkl/csv/zip — existing
- ✓ Spatial grid for neighbour queries — existing
- ✓ Arena hierarchy with confinement zones — existing
- ✓ Spike frequency adaptation in mean-field model — existing

### Active

- [ ] Programmatic config generation for parameter sweeps (eliminate manual JSON editing)
- [ ] Results aggregation across sweep runs (collect, compare, summarise outputs)
- [ ] Automated decision metrics extraction (decision time, accuracy, choice distribution)
- [ ] Better visualisation: phase diagrams, time series, attractor landscapes
- [ ] Jupyter notebook integration for post-hoc analysis (load results, plot, explore)
- [ ] New attractor models as research demands (binary and N-option competition variants)
- [ ] Automated tests for model correctness and numerical stability
- [ ] Performance improvements (fix prange parallelism, reduce per-tick allocations)

### Out of Scope

- Distributed/cloud execution across HPC clusters — not needed for current experiments
- Web-based UI or remote visualisation — desktop Qt GUI is sufficient
- Real-time robot deployment — simulation-only tool
- Changing output format from pkl/csv — current format is acceptable, tooling around it will improve

## Context

**Codebase state:** Framework is functional and actively used for experiments. Known technical debt includes: no automated tests, prange parallelism not actually parallel (missing `parallel=True`), per-tick array allocation GC pressure in MeanFieldSystem, O(N²) Hamiltonian in SpinSystem, race-prone output folder numbering in parallel runs, and Italian-language comments in environment.py.

**Research domain:** Collective decision-making in swarms. Focus on binary choices (two-option decisions) and multi-option attractor competition (N alternatives). Agents use ring attractor dynamics to model heading/orientation consensus and option selection.

**Experiment workflow:** Write JSON config → run simulation → collect pkl/csv output → analyse in Python/notebooks. Parameter sweeps currently require manual JSON duplication. Results are manually collected across runs.

**Stack:** Python 3.10, NumPy 2.2, SciPy 1.15, Numba 0.62, PySide6 6.10, Matplotlib 3.10. No test framework currently present.

## Constraints

- **Tech stack**: Python 3.10 + NumPy/SciPy/Numba — all new code must remain compatible with this stack
- **Output format**: pkl/csv output kept as-is — analysis tooling wraps it, not replaces it
- **Research pace**: Framework improvements must not break existing experiment configs
- **Hardware**: Runs on a single Linux workstation — no distributed compute constraints to design for

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Keep pkl/csv output format | Format is already in use; better tooling (notebooks, metrics extraction) is higher ROI than migration | — Pending |
| Numba JIT for mean-field ODE integration | Performance-critical inner loop; significant speedup vs pure Python | ✓ Good |
| Plugin-based behavioural models | Allows new models to be added without touching core simulation loop | ✓ Good |
| Multi-process per agent group | Enables parallelism across groups; IPC overhead is acceptable for current swarm sizes | — Pending |

---

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-08 after initialization*
