# Requirements: CollectiPy

**Defined:** 2026-04-09
**Core Value:** Agents running ring attractor dynamics should produce measurable, reproducible collective decisions that can be systematically explored via parameter sweeps.

## v1 Requirements

### Bifurcation Detection

- [ ] **BIF-01**: The mean-field model emits a bifurcation event when the activity bump stabilises in a target direction, detected via a sliding-window stability criterion (low positional variance over W consecutive ticks AND peak within angular tolerance of a known target direction)
- [ ] **BIF-02**: Bifurcation detection handles both the standard model (bump transitions once from average direction to one target) and the SFA model (bump oscillates before settling), using the same stability criterion
- [ ] **BIF-03**: The tick at which bifurcation is detected is logged in the run output
- [ ] **BIF-04**: Detection parameters (window size W, angular tolerance, variance threshold) are configurable per experiment via JSON config

### Post-Bifurcation Target Swap

- [ ] **SWAP-01**: After bifurcation is detected, the simulation waits a configurable number of ticks (N), then swaps the positions and quality values of two specified targets in the environment
- [ ] **SWAP-02**: The swap tick (bifurcation tick + N) is logged in the run output alongside the bifurcation tick
- [ ] **SWAP-03**: The number of post-bifurcation ticks before swap (N) is configurable per experiment via JSON config
- [ ] **SWAP-04**: Multiple N values can be swept across runs to map the re-decision boundary (at which delay agents can no longer re-decide)

### Analysis Pipeline

- [ ] **ANAL-01**: A notebook-ready utility loads pkl experiment output into a structured Python object (DataFrame or xarray) in ≤3 lines of code
- [ ] **ANAL-02**: Decision accuracy metric: fraction of runs per parameter set where agents chose the higher-quality target, computed from final bump position relative to target positions
- [ ] **ANAL-03**: Agent trajectory plots: spatial paths of all agents over time, coloured by group or decision state
- [ ] **ANAL-04**: Neural heatmap over time: 2D heatmap of ring attractor activity (neuron × tick) showing bump evolution and bifurcation point
- [ ] **ANAL-05**: Bifurcation and swap events are overlaid as vertical markers on time-series and heatmap plots

### Parameter Sweep Tooling

- [ ] **SWEEP-01**: A sweep helper generates a set of JSON config variants from a base config and a parameter grid (dict of param paths → list of values), saving all variants to a named directory
- [ ] **SWEEP-02**: Sweep results are aggregated into a single summary CSV/DataFrame mapping parameter values to per-run decision metrics (decision time, accuracy, bifurcation tick)

### Model Correctness & Tests

- [ ] **TEST-01**: Pytest suite with unit tests for MeanFieldSystem: known fixed points, bump formation from uniform initial conditions, correct response to a single strong target
- [ ] **TEST-02**: Numerical stability test: detect and report divergence (NaN/Inf in state vector) during ODE integration
- [ ] **TEST-03**: Regression tests for SpinSystem and movement models: outputs match reference snapshots across refactors

### Performance Fixes

- [ ] **PERF-01**: Fix Numba `prange` parallelism bug in mean-field ODE integration — add `parallel=True` to the `@njit` decorator so prange loops actually execute in parallel
- [ ] **PERF-02**: Fix race-prone output folder numbering in `dataHandling.py` for concurrent parallel sweep runs (use atomic directory creation or UUID-based naming)

## v2 Requirements

### Extended Bifurcation Criteria

- **BIF-V2-01**: Pluggable bifurcation criterion interface — allow user-defined detection functions to be registered without modifying core model code
- **BIF-V2-02**: Fourier-mode based bifurcation detection as an alternative criterion (magnitude of first Fourier mode of ring activity crosses threshold)

### Advanced Analysis

- **ANAL-V2-01**: Phase diagram generation: decision accuracy as a function of two swept parameters (heatmap over 2D parameter space)
- **ANAL-V2-02**: Attractor landscape visualisation: energy/potential landscape of the ring attractor at selected ticks
- **ANAL-V2-03**: Re-decision boundary curve: plot of swap delay N vs re-decision rate across a parameter sweep

### Infrastructure

- **INFRA-V2-01**: Reduce per-tick array allocation in MeanFieldSystem (pre-allocate trajectory buffers for GC pressure reduction in long runs)
- **INFRA-V2-02**: O(N²) Hamiltonian incremental update for SpinSystem (maintain running delta rather than full recompute each tick)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Change output format from pkl/csv | Current format in active use; tooling wraps it — migration would break existing notebooks |
| HPC / distributed cluster execution | Single workstation is sufficient for current experiment scale |
| Web-based or remote visualisation | Desktop Qt GUI is adequate |
| Real-time robot deployment | Simulation-only research tool |
| N-target swap (>2 targets) | Current experiments are 2-target; generalise in v2 if needed |

## Traceability

Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| PERF-01 | Phase 1 | Pending |
| PERF-02 | Phase 1 | Pending |
| TEST-01 | Phase 1 | Pending |
| TEST-02 | Phase 1 | Pending |
| TEST-03 | Phase 1 | Pending |
| BIF-01 | Phase 2 | Pending |
| BIF-02 | Phase 2 | Pending |
| BIF-03 | Phase 2 | Pending |
| BIF-04 | Phase 2 | Pending |
| SWAP-01 | Phase 3 | Pending |
| SWAP-02 | Phase 3 | Pending |
| SWAP-03 | Phase 3 | Pending |
| SWAP-04 | Phase 3 | Pending |
| ANAL-01 | Phase 4 | Pending |
| ANAL-02 | Phase 4 | Pending |
| ANAL-03 | Phase 4 | Pending |
| ANAL-04 | Phase 4 | Pending |
| ANAL-05 | Phase 4 | Pending |
| SWEEP-01 | Phase 5 | Pending |
| SWEEP-02 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 20 total
- Mapped to phases: 20/20 (100%) ✓
- Unmapped: 0

---
*Requirements defined: 2026-04-09*
*Last updated: 2026-04-09 after roadmap creation*
