# Phase 2: Bifurcation Detection - Research

**Researched:** 2026-04-10
**Domain:** Numerical bifurcation detection in ring-attractor neural dynamics (Python/NumPy)
**Confidence:** HIGH — all architectural decisions are locked in CONTEXT.md; codebase thoroughly inspected

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**D-01 — Algorithm:** Jacobian eigenvalue λ₁ approach from arXiv:2602.05683. Compute Re(λ₁) — the largest real part of the eigenvalues of the z-subspace Jacobian — at each tick. Bifurcation is detected when λ₁ shows a local-maximum spike above a configurable threshold.

**D-02 — Jacobian formula:**
```
J(z) = -I + (u - s) · diag(sech²((u - s)·M @ z + b − β)) · M
```
Standard model: s = 0 (formula reduces to u-only). SFA model: s = self.adapt_ring. Same code path for both.

**D-03 — Spike detection (3-sample micro-buffer):**
```python
buffer = deque(maxlen=3)
buffer.append(λ1_current)
if len(buffer) == 3:
    prev, curr, nxt = buffer
    if prev < curr > nxt and curr > lambda_threshold:
        bifurcation_detected(tick=current_tick - 1)
```
Default `lambda_threshold`: -0.1.

**D-04 — Re-triggerable detector** with `spike_min_separation` suppression (default 10 ticks) after each firing.

**D-05 — Per-agent event storage:**
```python
{"agent": "agent_0", "tick": 42, "lambda1": -0.03, "target": "A"}
```
Target field: nearest known target to bump angle at detection tick.

**D-06 — Output: events.json sidecar** in run output folder:
```json
{
  "bifurcation_events": [...],
  "swap_events": []
}
```
`swap_events` reserved empty for Phase 3.

**D-07 — Config namespace:** `mean_field_model.bifurcation.{lambda_threshold, spike_min_separation}`. Both optional; defaults used when key absent.

### Claude's Discretion

- **BifurcationDetector class location:** New module `src/models/bifurcation.py`. Class created per agent by `MeanFieldMovementModel`; called after `mean_field_system.step()`.
- **Jacobian computation:** Pure NumPy (not Numba). `np.linalg.eigvals` for 100×100 matrix is negligible vs ODE cost.
- **SFA Jacobian:** z-subspace with current `adapt_ring` plugged in as scalar `s` (mean of adapt_ring, or full vector — decide based on accuracy vs cost tradeoff). Full 2n×2n is deferred.
- **events.json write timing:** Written once per run at run end by `DataHandling`.
- **No-bifurcation runs:** events.json written with empty `bifurcation_events` list.
- **Tests:** Unit tests for `BifurcationDetector` with synthetic λ₁ sequences; integration test on 2-target standard model; SFA integration test.

### Deferred Ideas (OUT OF SCOPE)

- Full (z, s) 2n×2n block Jacobian for SFA (BIF-V2-01 area)
- Fourier-mode detection criterion (BIF-V2-02)
- Pluggable criterion interface (BIF-V2-01)
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| BIF-01 | Mean-field model emits a bifurcation event when activity bump stabilises, detected via Jacobian eigenvalue λ₁ spike | D-01 through D-03 specify algorithm; Jacobian formula verified against MeanFieldSystem fields |
| BIF-02 | Bifurcation detection handles both standard model and SFA model with same stability criterion | D-02 shows s=0 (standard) and s=adapt_ring (SFA) use identical code path; verified adapt_ring field exists |
| BIF-03 | Tick at which bifurcation is detected is logged in run output | D-05/D-06 define per-agent event dict and events.json sidecar; DataHandling integration point verified |
| BIF-04 | Detection parameters configurable per experiment via JSON config | D-07 defines mean_field_model.bifurcation.{lambda_threshold, spike_min_separation}; config access pattern verified in MeanFieldMovementModel.__init__ |
</phase_requirements>

---

## Summary

Phase 2 adds a `BifurcationDetector` class (`src/models/bifurcation.py`) that computes the leading real eigenvalue of the z-subspace Jacobian at each tick and detects when it spikes above a threshold — indicating the moment the ring-attractor activity bump commits to a target direction. The detector is instantiated per agent inside `MeanFieldMovementModel`, called after each `mean_field_system.step()`, and accumulates events into a per-run list. At run end, `DataHandling` writes the events to an `events.json` sidecar alongside existing pkl files.

The implementation is fully constrained by CONTEXT.md decisions. The codebase investigation confirms that all required fields (`self.u`, `self.b`, `self.M`, `self.beta`, `self.neural_ring`, `self.adapt_ring`, `self.theta`) are present on `MeanFieldSystem` [VERIFIED: codebase inspection]. The integration point in `MeanFieldMovementModel.step()` is well-defined — the `for _ in range(self.steps_per_tick)` loop (lines 155–163) returns `neural_field, bump_positions, final_norm` and `self._last_bump_angle` is already stored immediately after [VERIFIED: codebase inspection]. `DataHandling.run_folder` is the path written to; there is no existing `events.json` mechanism — this is new [VERIFIED: codebase inspection].

The Phase 1 test infrastructure (pytest, `pyproject.toml`, `tests/conftest.py`) exists and is runnable. `pytest` itself is not yet installed in the venv [VERIFIED: `python -m pytest --version` failed], so Phase 2 Wave 0 must install it. Test patterns from Phase 1 (`test_mean_field_system.py`, inline config dicts, `pythonpath = ["src"]` in `pyproject.toml`) are directly reusable [VERIFIED: codebase inspection].

**Primary recommendation:** Implement `BifurcationDetector` exactly as specified in D-01 through D-07. The Jacobian and eigenvalue computation are 5–10 lines of NumPy; the complexity is in correct integration with `MeanFieldMovementModel.step()` and `DataHandling.close()`.

---

## Standard Stack

### Core (all already in project venv)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| NumPy | 2.2.6 | Jacobian matrix ops, eigvals, sech² | All model code uses NumPy; verified in venv |
| Python stdlib: `collections.deque` | 3.10 | 3-sample micro-buffer (D-03) | Zero-dep, exactly the right tool |
| Python stdlib: `json` | 3.10 | events.json read/write | Already used in dataHandling.py |
| Python stdlib: `logging` | 3.10 | `sim.mean_field.bifurcation` logger | Project logging convention |

[VERIFIED: venv inspection — numpy 2.2.6, numba 0.62.1, scipy 1.15.3 all present]

### Test Dependencies (need install)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | latest stable (~8.x) | Test runner | Phase 2 unit + integration tests |

[VERIFIED: `python -m pytest --version` fails — pytest not yet in venv. Phase 1 tests exist but were presumably run via a separate install or not yet executed.]

**Installation (one-time, Wave 0):**
```bash
.venv/bin/pip install pytest
```

**Version verification (already installed):**
```bash
.venv/bin/python -c "import numpy; print(numpy.__version__)"  # 2.2.6
```

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `np.linalg.eigvals` | `scipy.linalg.eigvals` | SciPy version is more robust for defective matrices but identical result for smooth Jacobians; project already has SciPy, but NumPy is simpler and sufficient [ASSUMED] |
| `collections.deque` | Explicit ring buffer | deque is the canonical Python idiom; no reason to hand-roll |

---

## Architecture Patterns

### Recommended Project Structure

```
src/
├── models/
│   ├── bifurcation.py          # NEW — BifurcationDetector class
│   ├── mean_field_systems.py   # UNCHANGED — MeanFieldSystem
│   └── movement/
│       └── mean_field_model.py # MODIFIED — add detector instantiation + call
src/
├── dataHandling.py             # MODIFIED — add write_events_json() method
tests/
├── test_bifurcation.py         # NEW — unit + integration tests for Phase 2
├── conftest.py                 # EXISTING — no changes needed
└── fixtures/                   # EXISTING — add any new .npy fixtures if needed
```

### Pattern 1: BifurcationDetector Class

**What:** Self-contained detector that holds the 3-sample buffer, suppression counter, and accumulated event list. Exposes a single `step(tick, z, s, b, theta, target_angles)` method that returns an event dict or None.

**When to use:** Instantiated once per agent per run inside `MeanFieldMovementModel.__init__` (or reset in `MeanFieldMovementModel.reset()`).

```python
# Source: CONTEXT.md D-01 through D-04, verified against MeanFieldSystem fields
from collections import deque
import numpy as np
import logging

logger = logging.getLogger("sim.mean_field.bifurcation")

class BifurcationDetector:
    def __init__(self, lambda_threshold: float = -0.1, spike_min_separation: int = 10):
        self.lambda_threshold = lambda_threshold
        self.spike_min_separation = spike_min_separation
        self._buffer: deque = deque(maxlen=3)
        self._suppression_ticks: int = 0
        self.events: list[dict] = []

    def step(self, tick: int, z, s, u, b, M, beta, theta, target_angles, agent_name: str) -> dict | None:
        """Compute λ₁, check spike, return event dict or None."""
        lambda1 = self._compute_lambda1(z, s, u, b, M, beta)
        self._buffer.append(lambda1)
        if self._suppression_ticks > 0:
            self._suppression_ticks -= 1
            return None
        if len(self._buffer) == 3:
            prev, curr, nxt = self._buffer
            if prev < curr > nxt and curr > self.lambda_threshold:
                bump_angle = _compute_bump_angle(z, theta)
                target_id = _nearest_target(bump_angle, target_angles)
                event = {"agent": agent_name, "tick": tick - 1, "lambda1": float(curr), "target": target_id}
                self.events.append(event)
                self._suppression_ticks = self.spike_min_separation
                logger.info("Bifurcation detected: agent=%s tick=%d lambda1=%.4f target=%s",
                            agent_name, tick - 1, curr, target_id)
                return event
        return None

    def reset(self):
        """Reset buffer and events for a new run."""
        self._buffer.clear()
        self._suppression_ticks = 0
        self.events = []
```

**Jacobian implementation:**
```python
# Source: CONTEXT.md D-02
def _compute_lambda1(self, z, s, u, b, M, beta) -> float:
    """Compute Re(λ₁) of the z-subspace Jacobian."""
    # s is scalar (mean of adapt_ring) or 0.0 for standard model
    effective_u = u - s
    drive = effective_u * (M @ z) + b - beta
    sech2 = 1.0 / np.cosh(drive) ** 2          # diag of sech²(...)
    J = -np.eye(len(z)) + effective_u * (sech2[:, None] * M)
    eigenvalues = np.linalg.eigvals(J)
    return float(np.max(eigenvalues.real))
```

**SFA scalar `s` decision (Claude's Discretion):** Use `s = float(np.mean(adapt_ring))` for the z-subspace approximation. This is the cheapest accurate approximation — mean adaptation strength shifts the effective coupling uniformly across neurons, which is the dominant effect captured by the z-only Jacobian. A per-neuron `s` vector would require treating s as a full diag term; mean is the correct scalar approximation.

[VERIFIED: MeanFieldSystem.adapt_ring is `np.zeros(num_neurons)` for standard model — mean is 0.0, formula reduces to D-02's standard-model case exactly]

### Pattern 2: Integration into MeanFieldMovementModel.step()

**What:** Add detector call after the `mean_field_system.step()` loop, using already-computed `self._last_bump_angle` and target angle metadata.

**When to use:** After line 182 (`self._last_bump_angle = angle_rad`) in `mean_field_model.py`.

```python
# Source: CONTEXT.md code_context section; verified against mean_field_model.py lines 155-182
# Existing: self._last_bump_angle = angle_rad  (line 182)
# ADD after that line:
if self.bifurcation_detector is not None:
    target_angles_for_detector = [e.get("angle", 0.0) for e in self._mf_entities.get("targets", [])]
    target_ids_for_detector = [str(e.get("id", i)) for i, e in enumerate(self._mf_entities.get("targets", []))]
    z = self.mean_field_system.neural_ring
    s = float(np.mean(self.mean_field_system.adapt_ring))
    self.bifurcation_detector.step(
        tick=tick,
        z=z, s=s,
        u=self.mean_field_system.u,
        b=self.mean_field_system.b,
        M=self.mean_field_system.M,
        beta=self.mean_field_system.beta,
        theta=self.mean_field_system.theta,
        target_angles=np.array(target_angles_for_detector) if target_angles_for_detector else np.array([0.0]),
        target_ids=target_ids_for_detector,
        agent_name=self.agent.get_name(),
    )
```

**Detector instantiation** in `MeanFieldMovementModel.__init__`:
```python
# After self.params is set:
bif_cfg = self.params.get("bifurcation") or {}
self.bifurcation_detector = BifurcationDetector(
    lambda_threshold=float(bif_cfg.get("lambda_threshold", -0.1)),
    spike_min_separation=int(bif_cfg.get("spike_min_separation", 10)),
)
```

Reset in `MeanFieldMovementModel.reset()` (already has reset pattern):
```python
self.bifurcation_detector.reset()
```

### Pattern 3: events.json Write via DataHandling

**What:** New method `write_events_json(run_folder, events_payload)` on `DataHandling`, called at run-end. Mirrors the pkl-write pattern of opening a path under `self.run_folder`.

**When to use:** Called from `arena.py` after the run loop completes (before `data_handling.close()`), passing the aggregated events from all agents.

```python
# Source: dataHandling.py patterns; CONTEXT.md D-06
def write_events_json(self, bifurcation_events: list[dict]) -> None:
    """Write events.json sidecar for the current run."""
    if not self.run_folder:
        return
    path = os.path.join(self.run_folder, "events.json")
    payload = {
        "bifurcation_events": bifurcation_events,
        "swap_events": [],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
```

**Critical note:** `DataHandling._archive_run_folder()` is called from `close()` and **compresses + deletes** `run_folder` into `run_folder.zip`. `events.json` must be written **before** `close()` is called. The write must happen at the end of each run, after the tick loop exits, before `data_handling.close(shapes)` triggers archiving. In `arena.py`, the run boundary is the `while True:` block ending at line 615 (`if t < ticks_limit and not reset: break`) — events.json should be written there, parallel to the final `data_handling.save(... force=True)` call.

[VERIFIED: arena.py lines 606–615 show final save + run boundary; dataHandling.py lines 208–223 show `_archive_run_folder` compresses and deletes run_folder]

**Arena integration:** The arena loop does not currently have a hook for per-run post-processing beyond `data_handling.save`. The cleanest approach is to have `DataHandling` itself call `write_events_json` — but it needs the event list. Two options:

1. **Arena calls `write_events_json` explicitly** (passing event list collected from agent managers) — keeps DataHandling unaware of bifurcation logic
2. **DataHandling stores the event list** and writes on `close()` — DataHandling becomes bifurcation-aware

Recommend **Option 1** for Phase 2 (simpler, no DataHandling coupling). The arena already has access to agent spins/metadata; adding a `bifurcation_events` key to the agents' metadata dict (passed via `agents_metadata`) lets the arena extract it and call `data_handling.write_events_json(events)` at the run boundary.

[ASSUMED — Option 1 vs Option 2 is Claude's discretion; both are viable]

### Anti-Patterns to Avoid

- **Computing Jacobian inside MeanFieldSystem:** The detection logic belongs in `BifurcationDetector`, not `MeanFieldSystem`. `MeanFieldSystem` provides state; the detector reads it.
- **Writing events.json per-tick:** Write once at run end. Per-tick JSON writes would be slow and inconsistent with existing pkl pattern.
- **Using `scipy.linalg.eigvals`:** NumPy's `eigvals` is sufficient for well-conditioned 100×100 matrices. Avoid pulling in SciPy for this unless numerical issues appear in testing.
- **Assuming `_mf_entities` always has target entries:** The `_mf_entities` dict can be `{"targets": [], "guards": []}` (empty). The detector must handle zero targets gracefully (return `target = "unknown"` or similar).
- **Using tick-1 off-by-one incorrectly:** The 3-sample buffer fires when `curr` (the *previous* tick's value) is a local max. The event tick is `current_tick - 1` per D-03. Be precise about this.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Leading eigenvalue of real matrix | Power iteration, manual eigendecomp | `np.linalg.eigvals(J)` + `np.max(real)` | NumPy wraps LAPACK dgeev; correct, tested, 10 lines [VERIFIED: numpy.linalg available] |
| Circular angle difference | Manual modulo arithmetic | `_delta_angle(a1, a2)` in `mean_field_systems.py` | Already implemented and correct; reuse it [VERIFIED: line 21–23 of mean_field_systems.py] |
| Bump angle from ring state | Custom weighted-mean | `compute_center_of_mass(z, theta_i)` in `mean_field_systems.py` | Already implemented; reuse it [VERIFIED: line 26–29 of mean_field_systems.py] |
| JSON output | Custom serialisation | `json.dump` | Standard library; no external dep needed |
| Ring buffer | List + slice | `collections.deque(maxlen=3)` | Idiomatic Python; O(1) append/discard |

**Key insight:** The Jacobian eigenvalue is O(n³) ≈ 10⁶ FLOPs for n=100. This runs in ~0.5 ms on a modern CPU — completely dominated by the ODE integration which runs 500 Euler steps per tick. Do not attempt to optimise or JIT the Jacobian.

[ASSUMED: 0.5 ms estimate; verified that n=100 and Euler integration is the hot path]

---

## Common Pitfalls

### Pitfall 1: events.json Archived Before Being Written

**What goes wrong:** `DataHandling._archive_run_folder()` compresses and deletes `run_folder`. If `events.json` is written after `close()` (or inside `close()` after archiving), it either goes into the already-compressed zip incorrectly or raises FileNotFoundError.

**Why it happens:** The archive step (`shutil.rmtree(self.run_folder)`) removes the folder. Any file path based on `self.run_folder` is invalid afterwards.

**How to avoid:** Write `events.json` before calling `data_handling.close(shapes)`. In arena.py, the sequence should be: (1) final `data_handling.save()`, (2) `data_handling.write_events_json(events)`, (3) `data_handling.close(shapes)`.

**Warning signs:** FileNotFoundError on the json.dump call, or events.json not appearing in the .zip archive.

[VERIFIED: dataHandling.py lines 204–223 confirm close() calls _archive_run_folder() which does shutil.rmtree]

### Pitfall 2: SFA `adapt_ring` Shape Mismatch

**What goes wrong:** `adapt_ring` is an ndarray of shape `(num_neurons,)`. If passed directly as scalar `s` in the Jacobian formula, the matrix broadcast `effective_u * (sech2[:, None] * M)` produces a 3D array instead of 2D.

**Why it happens:** `u - s` where `s` is a vector produces a vector; `vector * M` is a (num_neurons, num_neurons) outer product or broadcast error.

**How to avoid:** Use `s = float(np.mean(adapt_ring))` explicitly. This converts the adaptation state to a scalar before the Jacobian computation.

**Warning signs:** `ValueError: operands could not be broadcast together` during SFA Jacobian computation.

[VERIFIED: adapt_ring is `np.zeros(self.num_neurons)` in MeanFieldSystem.__init__ line 127]

### Pitfall 3: Tick Off-by-One in Spike Detection

**What goes wrong:** The 3-sample buffer detects `curr` as the local max. `curr` was appended one tick ago (tick T-1). The reported tick in the event should be T-1, not T.

**Why it happens:** Standard sliding-window logic — the "middle" sample is always one step behind when processed.

**How to avoid:** Follow D-03 exactly: `event["tick"] = current_tick - 1`. Test this explicitly with a synthetic λ₁ sequence where the spike tick is known.

**Warning signs:** Event tick is off by ±1 from the true stabilisation tick.

### Pitfall 4: Suppression Counter Not Decremented on Non-Spike Ticks

**What goes wrong:** If `spike_min_separation` counter only decrements when a potential spike is evaluated, early-return paths (zero-length buffer, detection not armed) can skip the decrement, causing permanent suppression.

**Why it happens:** Defensive returns at the top of the `step()` method skip the decrement branch.

**How to avoid:** Decrement the suppression counter on every call where `len(buffer) == 3`, not just when a spike is evaluated.

**Warning signs:** Detector fires once and then never fires again for the rest of the run.

### Pitfall 5: Empty Target Angle List

**What goes wrong:** If `_mf_entities["targets"]` is empty (e.g., agent perceives nothing), `target_angles` is an empty array. `np.argmin` on empty array raises ValueError.

**Why it happens:** Agents at the edge of perception range may have zero targets in their metadata.

**How to avoid:** Guard the nearest-target lookup: if `target_angles` is empty or None, set `target = "unknown"`.

**Warning signs:** ValueError during nearest-target assignment.

### Pitfall 6: events.json Absent When No Bifurcation Occurs

**What goes wrong:** Analysis pipeline (Phase 4) expects `events.json` to always exist, even with an empty `bifurcation_events` list. If it's only written when a bifurcation fires, no-bifurcation runs silently break downstream.

**Why it happens:** Conditional write ("only write if events list is non-empty").

**How to avoid:** Always write `events.json` at run end, even with empty lists. This is explicitly stated in CONTEXT.md (success criterion 4) and Claude's Discretion.

[VERIFIED: CONTEXT.md states "If no spike is detected, events.json is written with an empty bifurcation_events list"]

---

## Code Examples

Verified patterns from codebase inspection:

### Reading bifurcation config from agent JSON
```python
# Source: mean_field_model.py lines 38-48 — existing config access pattern
params = agent.config_elem.get("mean_field_model", {}) or {}
bif_cfg = params.get("bifurcation") or {}
lambda_threshold = float(bif_cfg.get("lambda_threshold", -0.1))
spike_min_separation = int(bif_cfg.get("spike_min_separation", 10))
```

### Reusing compute_center_of_mass for bump angle
```python
# Source: mean_field_systems.py lines 26-29 [VERIFIED]
from models.mean_field_systems import compute_center_of_mass, _delta_angle
bump_angle = compute_center_of_mass(z, theta)
```

### Nearest target assignment using _delta_angle
```python
# Source: mean_field_systems.py lines 21-23; CONTEXT.md D-05 [VERIFIED]
def _nearest_target(bump_angle: float, target_angles: np.ndarray, target_ids: list[str]) -> str:
    if len(target_angles) == 0:
        return "unknown"
    diffs = np.abs(_delta_angle(np.full_like(target_angles, bump_angle), target_angles))
    idx = int(np.argmin(diffs))
    return target_ids[idx] if idx < len(target_ids) else str(idx)
```

### Logger pattern (consistent with project conventions)
```python
# Source: mean_field_systems.py line 31; mean_field_model.py line 28 [VERIFIED]
import logging
logger = logging.getLogger("sim.mean_field.bifurcation")
```

### DataHandling run_folder access
```python
# Source: dataHandling.py lines 186, 446-448 [VERIFIED]
# run_folder is set in new_run() and is None after close()
if not self.run_folder:
    return
path = os.path.join(self.run_folder, "events.json")
```

### Existing test pattern (for test_bifurcation.py to follow)
```python
# Source: tests/test_mean_field_system.py lines 21-33 [VERIFIED]
# Pattern: inline construction, no JSON files, explicit RNG seed
mf = MeanFieldSystem(
    num_neurons=50,
    sigma=0.0,
    dt=0.1,
    integration_time=50.0,
    u=6.0,
    beta=1.0,
    rng=np.random.default_rng(42),
)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Sliding-window positional variance (original BIF-01 description in REQUIREMENTS.md) | Jacobian eigenvalue λ₁ spike (D-01, arXiv:2602.05683) | CONTEXT.md discussion 2026-04-10 | More theoretically grounded; works for SFA oscillating bump without special-casing |
| Separate SFA bifurcation criterion (original BIF-02 concern) | Same code path, s plugged in from adapt_ring | D-02 | Eliminates code branching |

**Deprecated/outdated:**
- REQUIREMENTS.md BIF-01 description (sliding-window, variance threshold, angular tolerance, window W): Superseded by D-01 through D-07. The requirements text reflects the original concept; the CONTEXT.md decisions are the authoritative specification for Phase 2. The config parameters lambda_threshold and spike_min_separation replace the window-W / variance / tolerance parameters from the original BIF-04 description.

[VERIFIED: REQUIREMENTS.md still shows the old description; CONTEXT.md decisions are locked replacements]

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `np.linalg.eigvals` is numerically sufficient for the 100×100 z-subspace Jacobian (no need for scipy.linalg.eigvals) | Standard Stack, Don't Hand-Roll | Low: scipy.linalg.eigvals available as easy fallback if edge-case divergence occurs |
| A2 | Scalar `s = mean(adapt_ring)` is an accurate enough approximation for the z-subspace Jacobian in SFA mode | Architecture Patterns Pattern 1 | Medium: if the SFA integration test fails to detect bifurcation, switch to per-neuron `s` vector treatment; full 2n×2n Jacobian is deferred per CONTEXT.md |
| A3 | Bifurcation events can be collected from agents and passed back to the arena level via the existing `agents_metadata` channel | Architecture Patterns Pattern 3 | Medium: if the metadata channel is not the right pipe, may need a dedicated sidecar mechanism; investigate agents_metadata flow in arena.py before implementing |
| A4 | `events.json` written before `data_handling.close()` is sufficient — no additional locking needed (single-process write) | Architecture Patterns Pattern 3 / Pitfall 1 | Low: simulation runs single-process per run group; no concurrent access to run_folder within one run |

**If this table has items:** Assumptions A2 and A3 are the ones most likely to require adjustment at implementation time.

---

## Open Questions

1. **How to route bifurcation events from agents back to the arena for events.json writing?**
   - What we know: `agents_metadata` is passed via queue from agent managers to the arena. `agents_spins` carries model data (including `mean_field_entities`).
   - What's unclear: Is `agents_metadata` the right channel, or should bifurcation events travel via a new key in the agent snapshot dict? The arena's `_combine_agent_snapshots` merges these dicts.
   - Recommendation: Add a `"bifurcation_events"` key to the agent snapshot payload alongside `agents_shapes`/`agents_spins`/`agents_metadata`. The arena aggregates it the same way. Investigate `environment.py` (the agent manager process) to confirm what keys it sends.

2. **Does `MeanFieldMovementModel.reset()` need to reset the BifurcationDetector between runs?**
   - What we know: `reset()` is called at the start of each run (via arena.reset → agent.reset → movement_model.reset). The detector's event list must be cleared between runs.
   - What's unclear: Whether the arena calls `reset()` before or after `DataHandling.new_run()`.
   - Recommendation: Always call `bifurcation_detector.reset()` in `MeanFieldMovementModel.reset()`. Events are collected per-run; stale events from the previous run must not carry over.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| NumPy | Jacobian computation, eigvals | Yes | 2.2.6 | — |
| Python stdlib (json, collections, logging) | BifurcationDetector, events.json | Yes | 3.10.12 | — |
| pytest | Test suite | No | — | Install: `.venv/bin/pip install pytest` |
| SciPy | Optional eigvals fallback | Yes | 1.15.3 | NumPy eigvals is primary |

**Missing dependencies with no fallback:** None — all required runtime dependencies present.

**Missing dependencies with fallback:** pytest — required for test suite but trivially installable. This is Wave 0 work.

[VERIFIED: venv inspection — numpy 2.2.6, scipy 1.15.3, numba 0.62.1 confirmed. `python -m pytest --version` failed confirming pytest absent.]

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (to be installed) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` — `testpaths = ["tests"]`, `pythonpath = ["src"]` |
| Quick run command | `.venv/bin/python -m pytest tests/test_bifurcation.py -x` |
| Full suite command | `.venv/bin/python -m pytest tests/ -x` |

[VERIFIED: pyproject.toml confirmed with testpaths and pythonpath settings]

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BIF-01 | BifurcationDetector fires on a synthetic λ₁ spike sequence above threshold | unit | `.venv/bin/python -m pytest tests/test_bifurcation.py::test_spike_detection -x` | No — Wave 0 |
| BIF-01 | BifurcationDetector does NOT fire when λ₁ is below threshold | unit | `.venv/bin/python -m pytest tests/test_bifurcation.py::test_no_spike_below_threshold -x` | No — Wave 0 |
| BIF-01 | BifurcationDetector does NOT fire when sequence is monotonic (no local max) | unit | `.venv/bin/python -m pytest tests/test_bifurcation.py::test_no_spike_monotonic -x` | No — Wave 0 |
| BIF-01 | Standard model integration test: 2-target config produces a detectable bifurcation event | integration | `.venv/bin/python -m pytest tests/test_bifurcation.py::test_standard_model_bifurcation -x` | No — Wave 0 |
| BIF-02 | SFA model integration test: oscillating bump eventually settles and produces bifurcation event | integration | `.venv/bin/python -m pytest tests/test_bifurcation.py::test_sfa_model_bifurcation -x` | No — Wave 0 |
| BIF-03 | events.json written with correct schema (bifurcation_events, swap_events keys) | unit | `.venv/bin/python -m pytest tests/test_bifurcation.py::test_events_json_schema -x` | No — Wave 0 |
| BIF-04 | Non-default config values are respected at runtime | unit | `.venv/bin/python -m pytest tests/test_bifurcation.py::test_custom_config_respected -x` | No — Wave 0 |
| BIF-04 | No-bifurcation run: events.json written with empty list, no error raised | unit | `.venv/bin/python -m pytest tests/test_bifurcation.py::test_no_bifurcation_empty_list -x` | No — Wave 0 |
| BIF-01 | Suppression window prevents double-firing on trailing spike edge | unit | `.venv/bin/python -m pytest tests/test_bifurcation.py::test_spike_min_separation -x` | No — Wave 0 |
| BIF-01 | Re-triggerable: detector fires twice on two separated spikes | unit | `.venv/bin/python -m pytest tests/test_bifurcation.py::test_retriggerable -x` | No — Wave 0 |

### Sampling Rate
- **Per task commit:** `.venv/bin/python -m pytest tests/test_bifurcation.py -x`
- **Per wave merge:** `.venv/bin/python -m pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/test_bifurcation.py` — all BIF-01 through BIF-04 tests
- [ ] pytest install: `.venv/bin/pip install pytest` (if Phase 1 work not yet merged to this worktree)

---

## Security Domain

> Security enforcement: not applicable for this phase — no user inputs, no network, no auth. All inputs are float arrays from internal simulation state. JSON output is written to a local research data directory with no external exposure.

---

## Project Constraints (from CLAUDE.md)

The following directives from CLAUDE.md are mandatory for all Phase 2 code:

1. **Python 3.10 compatibility** — All new code must use only Python 3.10 syntax and stdlib. `dict[str, ...]` lowercase generics are fine. `match` statements are available.
2. **Output format unchanged** — `events.json` is additive (new sidecar). The existing pkl/csv format must not be modified.
3. **NumPy/SciPy/Numba stack** — No new third-party runtime dependencies may be added. Jacobian computation uses NumPy only (Numba is not required; see Don't Hand-Roll).
4. **No existing experiment config breakage** — The `bifurcation` config key under `mean_field_model` is optional with defaults; existing configs without it must continue to work identically.
5. **Logging convention** — `logging.getLogger("sim.mean_field.bifurcation")` is the correct logger for `BifurcationDetector`. Do not create module-level loggers with other names.
6. **Naming conventions** — `BifurcationDetector` (PascalCase class), `bifurcation.py` (snake_case module), `_compute_lambda1` / `_nearest_target` (private snake_case helpers).
7. **Docstrings** — One-line docstrings following the project pattern: `"""Detect bifurcation events from ring-attractor Jacobian eigenvalue spikes."""`
8. **No CI/CD** — pytest must be run manually; no automated pipeline exists.

---

## Sources

### Primary (HIGH confidence)
- Codebase inspection: `src/models/mean_field_systems.py` — MeanFieldSystem fields, compute_center_of_mass, _delta_angle helpers [VERIFIED]
- Codebase inspection: `src/models/movement/mean_field_model.py` — MeanFieldMovementModel.step() structure, config access pattern, _last_bump_angle storage [VERIFIED]
- Codebase inspection: `src/dataHandling.py` — DataHandling.run_folder, close()/_archive_run_folder() sequence, new_run() pattern [VERIFIED]
- Codebase inspection: `src/arena.py` — run loop structure, data_handling.save() call site, run boundary [VERIFIED]
- Codebase inspection: `tests/`, `pyproject.toml`, `tests/conftest.py`, `tests/test_mean_field_system.py` — Phase 1 test infrastructure [VERIFIED]
- `.planning/phases/02-bifurcation-detection/02-CONTEXT.md` — All locked decisions D-01 through D-07 [VERIFIED]
- CLAUDE.md — Project constraints, stack, conventions [VERIFIED]

### Secondary (MEDIUM confidence)
- `.planning/phases/01-foundation/01-CONTEXT.md` — NaN/Inf detection pattern (D-01..D-03), logger convention, test structure decisions [VERIFIED from file]
- `.planning/REQUIREMENTS.md` — BIF-01 through BIF-04 requirement text [VERIFIED]

### Tertiary (LOW confidence — ASSUMED)
- `np.linalg.eigvals` numerical sufficiency for smooth ring-attractor Jacobians: training knowledge, not benchmarked in this session

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all runtime deps verified in venv; only gap is pytest (install confirmed as simple)
- Architecture: HIGH — integration points verified by codebase inspection; assumptions flagged in Assumptions Log
- Pitfalls: HIGH — each pitfall derived from specific codebase facts (archive timing, adapt_ring shape, etc.)
- Test map: HIGH — test structure mirrors Phase 1 exactly; all gaps are Wave 0 new files

**Research date:** 2026-04-10
**Valid until:** 2026-07-10 (stable domain — ring attractor math, NumPy, Python stdlib)
