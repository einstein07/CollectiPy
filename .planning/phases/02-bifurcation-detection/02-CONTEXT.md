# Phase 2: Bifurcation Detection - Context

**Gathered:** 2026-04-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Add bifurcation detection to the mean-field model: detect when an agent's activity bump commits
to a target direction, log the event, and make detection parameters configurable via JSON.

No movement behaviour, no target swaps, no visualisation changes in this phase. The detector
produces structured event data (events.json sidecar) consumed by Phase 3 (swap) and Phase 4
(analysis pipeline).

</domain>

<decisions>
## Implementation Decisions

### Detection Algorithm (BIF-01, BIF-02)

- **D-01:** Use the **Jacobian eigenvalue λ₁ approach** from the paper *"From Vision to Decision:
  Neuromorphic Control for Autonomous Navigation and Tracking"* (arXiv:2602.05683). Compute
  Re(λ₁) — the largest real part of the eigenvalues of the Jacobian of the neural dynamics —
  at each tick. A bifurcation event is detected when λ₁ exhibits a local-maximum spike above a
  configurable threshold.

- **D-02:** The Jacobian is the **z-subspace Jacobian** with the current adaptation variable `s`
  plugged in:
  ```
  J(z) = -I + (u - s) · diag(sech²((u - s)·M @ z + b − β)) · M
  ```
  - Standard model: `s = 0` always → formula reduces to `J(z) = -I + u·diag(sech²(u·M@z + b − β))·M`
  - SFA model: `s = self.adapt_ring` (current adaptation state) → same formula, same code path
  - This means **BIF-02 is satisfied by a single implementation** — no separate criterion for SFA

- **D-03:** **Spike detection rule** (local maximum + threshold, 3-sample micro-buffer):
  ```python
  buffer = deque(maxlen=3)  # stores last 3 Re(λ₁) values
  buffer.append(λ1_current)
  if len(buffer) == 3:
      prev, curr, nxt = buffer
      if prev < curr > nxt and curr > lambda_threshold:
          bifurcation_detected(tick=current_tick - 1)  # curr is from tick-1
  ```
  The `lambda_threshold` parameter is configurable (see D-07). Pre-bifurcation equilibrium has
  λ₁ << 0 (typically -0.5 to -2.0). At the decision point λ₁ → 0. Typical threshold: -0.1.

### Fire Mode (BIF-01, multi-target extension)

- **D-04:** The detector is **re-triggerable** — it can fire multiple times per run. This supports
  experiments with more than two targets, where agents may commit sequentially to multiple
  directions. After each detected bifurcation, the detector suppresses re-detection for
  `spike_min_separation` ticks (configurable, default 10) to prevent double-firing on the
  trailing edge of the same spike.

### Per-Agent Event Storage (BIF-03)

- **D-05:** Bifurcation is **detected and stored per agent independently**. Each agent's
  `BifurcationDetector` appends to its own event list:
  ```python
  {"agent": "agent_0", "tick": 42, "lambda1": -0.03, "target": "A"}
  ```
  The `target` field is the ID of the nearest known target direction to the bump angle at the
  detection tick (argmin |Δangle(bump_angle, target_angle)|).

  Run-level aggregation (e.g., first agent to commit, majority tick) is done in the Phase 4
  analysis pipeline — not baked into the simulation.

### Output Format (BIF-03)

- **D-06:** Bifurcation events are written to an **`events.json` sidecar** in the run output
  folder alongside existing pkl files:
  ```json
  {
    "bifurcation_events": [
      {"agent": "agent_0", "tick": 42, "lambda1": -0.03, "target": "A"},
      {"agent": "agent_0", "tick": 91, "lambda1": -0.02, "target": "B"},
      {"agent": "agent_1", "tick": 45, "lambda1": -0.05, "target": "A"}
    ],
    "swap_events": []
  }
  ```
  The `swap_events` key is reserved for Phase 3. Phase 2 always writes an empty list for it.
  This shared sidecar format means Phase 3 only extends the same file, not creates a new one.

### Config Namespace (BIF-04)

- **D-07:** Detection parameters live **under `mean_field_model.bifurcation`** in the agent JSON
  config (per-agent, consistent with where other mean-field params live):
  ```json
  "mean_field_model": {
    "num_neurons": 100,
    "bifurcation": {
      "lambda_threshold": -0.1,
      "spike_min_separation": 10
    }
  }
  ```
  Both parameters are optional with the above defaults. An agent with no `bifurcation` key
  still detects bifurcations using defaults — no config required for basic use.

### Claude's Discretion

- **Detector class location:** A new `src/models/bifurcation.py` module with a `BifurcationDetector`
  class. Consumed by `MeanFieldMovementModel` — the movement model creates a detector instance
  per agent and calls it after each `mean_field_system.step()`. This keeps `MeanFieldSystem`
  (ODE integration) free of detection logic.

- **Jacobian computation:** Compute in NumPy (not Numba). The Jacobian is n×n (default n=100),
  eigendecomposition via `np.linalg.eigvals` is O(n³) ≈ 10⁶ ops — negligible vs ODE integration.
  For SFA, `adapt_ring` is the average of the adaptation variable across the agent's neurons
  (or the full vector if the Jacobian is extended to 2n×2n — Claude decides based on accuracy
  needed vs. cost).

- **events.json write timing:** Written at the end of each run (not per-tick) by `DataHandling`
  after the run loop completes. Matches existing pkl write pattern.

- **No-bifurcation runs:** If no spike is detected, `events.json` is written with an empty
  `bifurcation_events` list. This satisfies Phase 2 success criterion 4.

- **Tests (TEST coverage for Phase 2):** Unit tests for `BifurcationDetector` with synthetic λ₁
  sequences; integration test on a known 2-target configuration (standard model) that produces a
  detectable spike; SFA integration test with known-oscillating parameters that eventually settle.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Bifurcation Detection Theory
- arXiv:2602.05683 — "From Vision to Decision: Neuromorphic Control for Autonomous Navigation
  and Tracking". Section 4.4 (Methods) defines the Jacobian eigenvalue approach and spike
  detection criterion. The CollectiPy dynamics differ from the paper's — derive the Jacobian
  from CollectiPy's own ODE (D-02), not from the paper's equations verbatim.

### Model Implementation
- `src/models/mean_field_systems.py` — `MeanFieldSystem` class. Key fields: `self.u`, `self.b`,
  `self.M`, `self.beta`, `self.neural_ring` (z), `self.adapt_ring` (s, SFA only),
  `self.theta` (neuron angles). The Jacobian is computed from these at the current state.
- `src/models/movement/mean_field_model.py` — `MeanFieldMovementModel`. This is where
  `BifurcationDetector` will be instantiated and called. `step(agent, tick, ...)` receives the
  arena tick — this is the tick stored in bifurcation events.

### Output Infrastructure
- `src/dataHandling.py` — `DataHandling` class. `new_run(run_id, ...)` and run-end logic.
  `events.json` should be written here alongside existing pkl files using the run folder path.
- `src/arena.py` — Arena main loop. `_prepare_target_position_swaps_for_run()` shows the pattern
  for per-run event setup. The `run()` method drives the tick loop that calls agent steps.

### Phase 1 Context (decisions that carry forward)
- `.planning/phases/01-foundation/01-CONTEXT.md` — D-01 through D-03 cover NaN/Inf detection
  and logging patterns. The `sim.mean_field` logger is the correct logger to use for any
  diagnostic output from `BifurcationDetector`.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `compute_center_of_mass(z, theta_i)` in `mean_field_systems.py` — already available; used to
  get the current bump angle for "nearest target" assignment in bifurcation events.
- `_delta_angle(a1, a2)` in `mean_field_systems.py` — angle difference helper; use for nearest
  target assignment.
- `MeanFieldMovementModel._last_bump_angle` — already stored per tick; can be reused to avoid
  recomputing bump angle in the detector.
- `logging.getLogger("sim.mean_field")` — existing logger; `BifurcationDetector` should use
  this logger (or `sim.mean_field.bifurcation`).

### Established Patterns
- `mean_field_model` config key: agent JSON already has `mean_field_model: {...}`. The new
  `bifurcation` subkey nests cleanly without breaking existing configs.
- `DataHandling._write_mean_field_logs()`: per-tick pkl writes. The `events.json` write pattern
  should follow the same `run_folder` path but write once at run end (not per tick).
- `Arena._apply_due_target_position_swaps()`: existing per-tick event application; shows how
  the arena loop dispatches timed events. Phase 3 will extend this pattern for swaps triggered
  by bifurcation tick + N.

### Integration Points
- `MeanFieldMovementModel.step()` — add detector call after the `mean_field_system.step()` loop
  (lines ~192-201). Pass `tick`, current `bump_angle`, and current `target_angles` to the
  detector.
- `DataHandling` — add `write_events_json(run_folder, events)` method; call at run end.
- No changes needed to `MeanFieldSystem` itself — all detection logic lives in the new
  `BifurcationDetector` class.

</code_context>

<specifics>
## Specific Ideas

- **Reference paper equation (for deriving CollectiPy's Jacobian):** The paper uses a
  simplex-constrained Jacobian. CollectiPy does NOT use the simplex constraint — use the
  unconstrained z-subspace Jacobian from D-02 directly.
- **SFA Jacobian recommendation:** Start with the z-subspace Jacobian with current `s` plugged
  in (D-02). The full (z, s) Jacobian can be explored if the z-subspace misses SFA bifurcations
  in testing — but this is a v2 refinement if needed.
- **events.json is the Phase 3 contract:** Phase 3 (target swap) will write `swap_events` into
  the same file. The schema defined in D-06 (`bifurcation_events`, `swap_events` keys) should be
  treated as a cross-phase contract.

</specifics>

<deferred>
## Deferred Ideas

- **Full (z, s) Jacobian for SFA** — Using the 2n×2n block Jacobian that couples z and s
  subspaces may be more accurate for the SFA model. Deferred to v2 (BIF-V2-01 area).
- **Fourier-mode detection (BIF-V2-02)** — Alternative criterion: magnitude of first Fourier
  mode of ring activity crosses threshold. Explicitly a v2 requirement.
- **Pluggable criterion interface (BIF-V2-01)** — Full registry of user-defined criterion
  functions. Phase 2 implements eigenvalue-only; the class structure should be designed to allow
  this without a rewrite (clean `detect(tick, z, s, b, theta, target_angles) → event | None`
  interface).

</deferred>

---

*Phase: 02-bifurcation-detection*
*Context gathered: 2026-04-10*
