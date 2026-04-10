# Roadmap: CollectiPy

## Overview

CollectiPy is a functional swarm simulation framework that needs correctness guarantees, new
decision-detection capabilities, and systematic experiment tooling. This roadmap starts by
fixing infrastructure bugs and establishing a test baseline, then adds bifurcation detection
and target-swap experiment logic to the mean-field model, and finishes with an analysis and
sweep pipeline that makes parameter exploration reproducible and publishable.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Foundation** - Fix infrastructure bugs and establish the test suite
- [ ] **Phase 2: Bifurcation Detection** - Add sliding-window decision detection to the mean-field model
- [ ] **Phase 3: Target Swap** - Implement post-bifurcation environment swap and re-decision logic
- [ ] **Phase 4: Analysis Pipeline** - Build notebook-ready result loading, metrics, and visualisation
- [ ] **Phase 5: Sweep Tooling** - Automate config generation and multi-run result aggregation

## Phase Details

### Phase 1: Foundation
**Goal**: The simulation runs correctly and its model behaviour is verifiable by an automated test suite
**Depends on**: Nothing (first phase)
**Requirements**: PERF-01, PERF-02, TEST-01, TEST-02, TEST-03
**Success Criteria** (what must be TRUE):
  1. Running `pytest` from the project root completes without errors and reports passing tests for MeanFieldSystem fixed points, bump formation, and single-target response
  2. A long mean-field run with prange-eligible loops demonstrably uses multiple CPU threads (verified via Numba parallel diagnostics or wall-clock comparison)
  3. Concurrent sweep runs writing output simultaneously do not collide on the same folder name
  4. Numerical divergence (NaN/Inf in the ODE state vector) during a run causes a clear reported error rather than silent corruption
  5. SpinSystem and movement model outputs match reference snapshots before and after any refactor
**Plans**: 4 plans
Plans:
- [x] 01-01-PLAN.md — PERF-01 prange fix + TEST-02 NaN/Inf guard + pytest infrastructure
- [x] 01-02-PLAN.md — PERF-02 atomic folder creation race fix
- [x] 01-03-PLAN.md — TEST-01 MeanFieldSystem unit tests
- [x] 01-04-PLAN.md — TEST-03 regression snapshots for SpinSystem and movement models

### Phase 2: Bifurcation Detection
**Goal**: The mean-field model can detect and log the moment agents commit to a target direction
**Depends on**: Phase 1
**Requirements**: BIF-01, BIF-02, BIF-03, BIF-04
**Success Criteria** (what must be TRUE):
  1. Running a two-target mean-field experiment produces a bifurcation event in the output with the tick at which the activity bump stabilised
  2. The detection fires correctly for both the standard model (single bump transition) and the SFA model (oscillating bump that eventually settles)
  3. A config with non-default window size, angular tolerance, and variance threshold is accepted and respected at runtime
  4. An experiment with no stable decision (bump never settles within tolerance) completes without error and logs no bifurcation event
**Plans**: TBD

### Phase 3: Target Swap
**Goal**: Experiments can test re-decision by swapping target positions after the initial commitment
**Depends on**: Phase 2
**Requirements**: SWAP-01, SWAP-02, SWAP-03, SWAP-04
**Success Criteria** (what must be TRUE):
  1. After bifurcation, the two specified targets in the environment exchange positions and quality values at the configured tick offset N
  2. The run output log contains both the bifurcation tick and the swap tick (bifurcation tick + N)
  3. Running a sweep over multiple N values produces one output folder per N, each with the correct swap tick recorded
  4. A config specifying the post-bifurcation delay N is accepted via JSON without code changes
**Plans**: TBD

### Phase 4: Analysis Pipeline
**Goal**: Experiment results can be loaded and visualised in a Jupyter notebook with minimal boilerplate
**Depends on**: Phase 1
**Requirements**: ANAL-01, ANAL-02, ANAL-03, ANAL-04, ANAL-05
**Success Criteria** (what must be TRUE):
  1. A single utility call (3 lines or fewer) loads a pkl output directory into a structured DataFrame or xarray object in a notebook
  2. Decision accuracy (fraction of runs choosing the higher-quality target) is computable from the loaded data and matches manually-verified ground truth on a reference run set
  3. A spatial trajectory plot of all agents coloured by group or decision state renders without error from the loaded data
  4. A neural heatmap (neuron x tick) showing ring attractor activity and bump evolution renders, with bifurcation and swap ticks overlaid as vertical markers
**Plans**: TBD
**UI hint**: yes

### Phase 5: Sweep Tooling
**Goal**: Parameter sweeps can be defined programmatically and their results aggregated into a single summary without manual file management
**Depends on**: Phase 4
**Requirements**: SWEEP-01, SWEEP-02
**Success Criteria** (what must be TRUE):
  1. Calling the sweep helper with a base config and a parameter grid (e.g., three coupling strengths x two noise levels) produces a correctly-named directory containing one JSON variant per grid point
  2. After running all sweep variants, a single aggregation call produces a CSV/DataFrame with one row per run and columns for each swept parameter plus decision time, accuracy, and bifurcation tick
  3. An existing base config runs unchanged without modification when the sweep helper is not invoked
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation | 0/4 | Not started | - |
| 2. Bifurcation Detection | 0/TBD | Not started | - |
| 3. Target Swap | 0/TBD | Not started | - |
| 4. Analysis Pipeline | 0/TBD | Not started | - |
| 5. Sweep Tooling | 0/TBD | Not started | - |
