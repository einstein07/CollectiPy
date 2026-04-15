# Phase 2: Bifurcation Detection - Context (Revised)

**Gathered:** 2026-04-10
**Revised:** 2026-04-12 — post-experimental review; two detection modes added; SFA Ω criterion added; gradient-based standard-model criterion added.

**Revision trigger:** Experimental results showed (1) the λ₁ local-max + threshold criterion fires repeatedly once λ₁ plateaus above the threshold in the standard model; (2) the λ₁ signal is too noisy in the SFA model for threshold-based detection to be reliable. Phase 2 implementation is being updated to address both.

---

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

---

### Detection Modes (NEW — replaces original single-criterion D-03)

**D-01: Two named detection modes, selectable via JSON config.**

Every agent running a mean-field model picks one mode. The mode name is passed under
`mean_field_model.bifurcation.mode`. Both modes write to `events.json` and are re-triggerable.

| Mode | Purpose | Criterion |
|------|---------|-----------|
| `"behavioral"` | Runtime agent-behavior trigger (Phase 3 swap) | Bump angle ≤ 5° from a target for ≥ 5 consecutive ticks |
| `"analytical"` | Research-grade detection with model-specific metric | Standard: gradient of λ₁ local max; SFA: Ω threshold crossing |

Default if `mode` key is absent: `"behavioral"`.

---

### Mode 1 — Behavioral Detection (BIF-BEHAV)

**D-02:** Fire a bifurcation event when the ring attractor bump angle is within
`alignment_tolerance_deg` of any known target angle **for `alignment_consecutive_ticks`
consecutive simulation ticks**.

- Bump angle = `_last_bump_angle` from `MeanFieldMovementModel` (egocentric, radians)
- Target angles come from the per-tick entity list, same as existing code path
- Alignment check: `|Δangle(bump_angle, target_angle)| ≤ alignment_tolerance_deg * π/180`
- Counter resets if the bump drifts out of the tolerance window
- Re-triggerable: after firing for target A, a subsequent alignment with target B can fire again
  (using `spike_min_separation` ticks as suppression between consecutive events)
- Model-agnostic: works identically for standard and SFA ring attractors

Config defaults:
```json
"bifurcation": {
  "mode": "behavioral",
  "alignment_tolerance_deg": 5.0,
  "alignment_consecutive_ticks": 5,
  "spike_min_separation": 10
}
```

---

### Mode 2 — Analytical Detection (BIF-ANAL)

#### Standard Ring Attractor (g_adapt == 0)

**D-03:** Compute Re(λ₁) — the dominant Jacobian eigenvalue — at every tick (unchanged from
original Phase 2). Replace the local-max + threshold spike criterion with a
**gradient-of-λ₁ local-maximum** criterion:

1. Maintain a rolling window of the last `gradient_window` λ₁ values.
2. Compute the discrete first derivative: `dλ₁/dt ≈ (λ₁[t] - λ₁[t - gradient_window]) / gradient_window`
3. Detect when `dλ₁/dt` itself is at a **local maximum** above `gradient_threshold`.
   This fires at the steepest part of the λ₁ rise — the actual decision onset —
   rather than at the plateau where the old threshold criterion kept re-firing.

`gradient_window` and `gradient_threshold` are configurable. Defaults chosen by planner
based on the experimental λ₁ curves (typical rise spans ~20 ticks in the standard model).

**D-02 (Jacobian formula — unchanged):**
```
J(z) = -I + (u - s) · diag(sech²((u - s)·M @ z + b − β)) · M
Standard model: s = 0
```

#### SFA Ring Attractor (g_adapt > 0)

**D-04:** Use the **Ω criterion** from Ermentrout, Folias & Kilpatrick (2014),
*"Spatiotemporal Pattern Formation in Neural Fields with Linear Adaptation"*,
Chapter 4, Eq. 4.23 & 4.26.

The characteristic equation for linear stability of the stationary bump under odd perturbations is:

```
λ² + [1 + α - (1+β)Ω]λ + α(1+β)(1-Ω) = 0
where  Ω = (1+β)A / ((1+β)A + I₀)
```

A **Hopf bifurcation** occurs when `Ω = (1+α)/(1+β)` (Eq. 4.26).
Detection fires when Ω **crosses** this threshold (transition from below to above, or above to below,
depending on parameter regime — planner determines crossing direction from experimental data).

**Mapping from paper notation to CollectiPy:**

| Paper symbol | CollectiPy field |
|-------------|-----------------|
| β (adaptation strength) | `mf.g_adapt` |
| α (adaptation rate, = 1/τ) | `1.0 / mf.tau_adapt` |
| A (bump amplitude) | First Fourier mode magnitude of `neural_ring`: `\|∑ z_i exp(iθ_i)\| / (N/2)` |
| I₀ (input amplitude) | First Fourier mode magnitude of `mean_field_perception` (the perception vector) |

Bifurcation threshold:
```python
alpha_paper = 1.0 / mf.tau_adapt
beta_paper  = mf.g_adapt
threshold   = (1.0 + alpha_paper) / (1.0 + beta_paper)

# Fourier mode amplitudes
A  = np.abs(np.sum(mf.neural_ring * np.exp(1j * mf.theta))) / (mf.num_neurons / 2)
I0 = np.abs(np.sum(perception_vec * np.exp(1j * mf.theta))) / (mf.num_neurons / 2)

Omega = (1.0 + beta_paper) * A / ((1.0 + beta_paper) * A + I0 + 1e-12)  # guard /0

# Event fires when Omega crosses threshold (3-sample buffer, same logic as λ₁ spike)
```

---

### Per-Tick Metric Logging (BIF-LOG)

**D-05:** Log the analytical metric to the spins pkl file every tick, regardless of detection mode.
This ensures the full time-series is always available for post-processing.

| Model type | New pkl key | Value |
|-----------|------------|-------|
| Standard (g_adapt == 0) | `mean_field_lambda1` | Re(λ₁), float or None if Jacobian failed |
| SFA (g_adapt > 0) | `mean_field_omega` | Ω scalar ∈ [0, 1], or None if perception unavailable |

Both are written by `MeanFieldMovementModel.get_spin_system_data()` into the dict returned to
`DataHandling`, which writes it to `{agent_id}_spins.pkl` per tick.

`mean_field_lambda1` is already implemented. `mean_field_omega` is new.

Note: In `analytical` mode the detector also uses λ₁ / Ω to fire events. In `behavioral` mode
the pkl logging still happens (always on) but the detector ignores the metric value.

---

### Fire Mode (unchanged)

**D-06:** Re-triggerable — multiple bifurcation events per run supported. Suppression window
`spike_min_separation` ticks (default 10) between consecutive fires.

---

### Per-Agent Event Storage (unchanged)

**D-07:** Per-agent event list; each event dict:
```python
{"agent": "agent_0", "tick": 42, "metric": -0.03, "target": "A", "mode": "analytical"}
```
`metric` = λ₁ value (analytical/standard), Ω value (analytical/SFA), or bump alignment angle
(behavioral). `mode` field added for downstream distinguishability.

---

### Output Format (unchanged)

**D-08:** `events.json` sidecar in the run folder:
```json
{
  "bifurcation_events": [
    {"agent": "agent_0", "tick": 42, "metric": -0.03, "target": "A", "mode": "analytical"},
    {"agent": "agent_1", "tick": 45, "metric": 0.71,  "target": "A", "mode": "analytical"}
  ],
  "swap_events": []
}
```

---

### Config Namespace (extended)

**D-09:** All bifurcation parameters live under `mean_field_model.bifurcation`:

```json
"mean_field_model": {
  "num_neurons": 100,
  "bifurcation": {
    "mode": "behavioral",

    "alignment_tolerance_deg": 5.0,
    "alignment_consecutive_ticks": 5,

    "gradient_window": 5,
    "gradient_threshold": 0.01,
    "spike_min_separation": 10,

    "lambda_threshold": -0.1
  }
}
```

`lambda_threshold` retained for backward compatibility with existing tests and run configs.
Planner decides which params are active per mode (unused params silently ignored).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Bifurcation Theory — Standard Model
- arXiv:2602.05683 — "From Vision to Decision: Neuromorphic Control for Autonomous Navigation
  and Tracking". Section 4.4 (Methods) defines the Jacobian eigenvalue λ₁ approach and the
  original spike detection criterion. The gradient-based criterion (D-03) replaces the threshold
  criterion from this paper for the standard model.

### Bifurcation Theory — SFA Model
- `Literature/Spatiotemporal Pattern Formation in Neural Fields with Linear Adaptation.pdf`
  (Ermentrout, Folias & Kilpatrick, in *Neural Fields*, Springer 2014, DOI 10.1007/978-3-642-54593-1_4)
  — Eq. 4.23 (characteristic equation), Eq. 4.26 (Hopf bifurcation condition: α_H = (1+β)Ω − 1).
  This defines the Ω scalar and the bifurcation threshold (1+α)/(1+β). Section 4.3.2 gives the
  full derivation of Ω from the stationary bump solution.

### Model Implementation
- `src/models/mean_field_systems.py` — `MeanFieldSystem`. Key fields: `neural_ring` (z),
  `adapt_ring` (s, SFA), `theta` (neuron angles), `g_adapt`, `tau_adapt`, `u`, `b`, `M`, `beta`.
- `src/models/movement/mean_field_model.py` — `MeanFieldMovementModel`. Current integration
  point for `BifurcationDetector`. `get_spin_system_data()` is where `mean_field_lambda1` is
  added; `mean_field_omega` goes in the same dict.
- `src/models/bifurcation.py` — `BifurcationDetector` (Phase 2 deliverable). Must be extended
  to support both modes and Ω computation. `last_lambda1` attribute already exposed on the class.

### Output Infrastructure
- `src/dataHandling.py` — `DataHandling`. `events.json` write pattern established.
- `src/arena.py` — `Arena._collect_bifurcation_events()` established.

### Phase 1 Context
- `.planning/phases/01-foundation/01-CONTEXT.md` — NaN/Inf logging patterns; `sim.mean_field`
  logger namespace.

### Experimental Results (read for context)
- Run folder: `mean_field_2targets_runs_bifurcation_detection_test_20260411-110058`
  Standard model: λ₁ rises smoothly, plateaus above threshold → gradient criterion needed.
  SFA model: λ₁ is noisy and positive throughout → λ₁ approach unreliable; Ω criterion needed.

</canonical_refs>

<code_context>
## Existing Code Insights

### Already Implemented (Phase 2 v1)
- `BifurcationDetector` class: `compute_jacobian()`, `compute_lambda1()`, `_check_spike()`,
  `update()`, `reset()`, `last_lambda1` attribute, `events` list.
- `MeanFieldMovementModel.step()`: calls `self.bifurcation_detector.update(...)` after each tick.
- `get_spin_system_data()`: already returns `"mean_field_lambda1"` in the dict.
- `DataHandling.collect_bifurcation_events()` and `_write_events_json()`.
- `Arena._collect_bifurcation_events()`.

### What Needs to Change
1. **`BifurcationDetector`**: Add `mode` dispatch. For `behavioral` mode, add alignment-counter
   logic (no Jacobian needed). For `analytical` mode, keep Jacobian path for standard model;
   add Ω computation for SFA.
2. **`BifurcationDetector._check_spike()`**: Replace/extend with gradient-of-λ₁ criterion for
   analytical/standard mode. Add Ω threshold-crossing detector for analytical/SFA mode.
3. **`MeanFieldMovementModel.get_spin_system_data()`**: Add `mean_field_omega` key (SFA only).
   `mean_field_lambda1` remains for standard model.
4. **Tests**: Update unit tests for new criterion logic; add Ω computation test.

### Fourier Mode Computation (for Ω)
`mf.theta` is the array of neuron preferred angles (radians). First Fourier mode magnitude:
```python
A = np.abs(np.dot(mf.neural_ring, np.exp(1j * mf.theta))) / (mf.num_neurons / 2)
```
This is equivalent to `2/N * |∑_i z_i exp(i θ_i)|`. Already used internally in
`compute_center_of_mass()` for bump angle — can reuse the complex sum directly.

### Behavioral Mode Alignment Check
Bump angle is `self._last_bump_angle` (set in `step()` after `mean_field_system.step()`).
Target angles come from `self._mf_entities["targets"]`, same source used by current detector.
```python
def _aligned_with_any_target(self, bump_angle, target_angles, tol_rad):
    return any(
        abs(((bump_angle - t + np.pi) % (2*np.pi)) - np.pi) <= tol_rad
        for t in target_angles
    )
```

</code_context>

<specifics>
## Specific Notes

- **Backward compatibility**: Existing run configs with no `mode` key default to `"behavioral"`.
  Configs with `lambda_threshold` but no `mode` key also default to `"behavioral"` — the
  threshold param is silently ignored (not an error).
- **events.json `metric` field**: In behavioral mode, `metric` = the bump angle at the moment
  of detection (float, radians). In analytical/standard, `metric` = λ₁ peak value. In
  analytical/SFA, `metric` = Ω value at crossing tick.
- **`mean_field_omega` always logged for SFA**: Even in behavioral mode, Ω is computed and
  written to pkl every tick. This gives researchers the full Ω trace for post-hoc analysis
  regardless of which mode triggered the event.
- **Ω guard**: Add `+ 1e-12` to the denominator to prevent division by zero when both bump and
  perception are near zero at simulation start.
- **Paper's α/β notation**: In CollectiPy, `g_adapt` = paper's β (adaptation strength) and
  `1/tau_adapt` = paper's α (adaptation rate). The bifurcation threshold is
  `(1 + 1/tau_adapt) / (1 + g_adapt)`. Planner must verify sign conventions against
  `mean_field_systems.py` SFA integration code.

</specifics>

<deferred>
## Deferred Ideas

- **Full (z, s) Jacobian for SFA** — 2n×2n block Jacobian coupling z and s subspaces.
  May give a cleaner λ₁ signal for SFA. Deferred to v2 (BIF-V2-01).
- **Fourier-mode detection (BIF-V2-02)** — Alternative: magnitude of first Fourier mode of ring
  activity crosses a threshold. Related to A in the Ω computation but used directly.
- **Pluggable criterion interface** — Full registry of user-defined criterion functions. Phase 2
  handles the two named modes; a plugin system is a v2 refactor.
- **Drift instability detection** — For SFA with I₀=0, the bifurcation is a pitchfork (drift
  to traveling bump) rather than a Hopf. Eq. 4.24 gives this case (α = β). Separate detection
  criterion if needed for input-free SFA experiments.

</deferred>

---

*Phase: 02-bifurcation-detection*
*Context originally gathered: 2026-04-10*
*Revised: 2026-04-12*
