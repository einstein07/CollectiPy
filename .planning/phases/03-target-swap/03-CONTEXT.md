# Phase 3: Target Swap - Context

**Gathered:** 2026-04-12

---

<domain>
## Phase Boundary

After a bifurcation event is detected, the arena waits a configurable number of ticks (N), then
swaps the positions **and quality values** of two specified targets. The swap tick is logged in
`events.json`. This phase adds no new visualisation changes, no new detection criteria, and no
sweep tooling — those belong to Phase 2 and Phase 5 respectively.

The deliverable is: a single swap that fires once per run, at `bifurcation_tick + delay_ticks`,
which is logged in `events.json` alongside the bifurcation event that triggered it.

</domain>

<decisions>
## Implementation Decisions

---

### D-01: Trigger Source — First Agent to Bifurcate

The post-bifurcation swap is triggered by the **first bifurcation event from any agent** in the
run. Subsequent bifurcation events (re-decisions after the swap) are logged in `bifurcation_events`
as normal but do NOT schedule additional swaps.

This policy is intentional: single-agent experiments (one agent, one bifurcation) work identically
to multi-agent runs. The swap fires as soon as commitment is detected anywhere in the swarm.

Implementation note: Arena must track a `_post_bif_swap_triggered: bool` flag per run, reset to
`False` at the start of each run. Once triggered, it stays `True` for the remainder of that run.

---

### D-02: What Gets Swapped — Positions AND Quality (intensity)

Both the XY position **and** the `intensity` value are swapped between the two named targets.
Target labels (IDs) remain fixed — only coordinates and intensity exchange.

```
Before swap:
  target_A: pos=(10, 5),  intensity=2.0
  target_B: pos=(-10, 5), intensity=0.5

After swap:
  target_A: pos=(-10, 5), intensity=0.5
  target_B: pos=(10, 5),  intensity=2.0
```

The existing `_swap_object_xy_positions()` handles coordinates. A new helper
`_swap_object_intensity()` (or inline logic) handles the `intensity` attribute.

How intensity is stored on entity objects must be confirmed by the planner from the entity
data model. If `intensity` is not a direct attribute, the swap must go through whatever
accessor the mean-field model uses (the `entry.get("intensity", 1.0)` path in
`mean_field_model.py` line 391).

---

### D-03: Config Location — `environment.post_bifurcation_swap`

The post-bifurcation swap is configured at **environment scope**, alongside the existing
`target_position_swaps` key. This keeps all environment-level event scheduling in one place
and does not couple the swap spec to per-agent model config.

Config schema:
```json
"environment": {
  "post_bifurcation_swap": {
    "pairs": [["target_A", "target_B"]],
    "delay_ticks": 10
  }
}
```

- `pairs`: list of two-element lists of target IDs to swap. Matches the format of
  `target_position_swaps[].pairs` for consistency.
- `delay_ticks`: non-negative integer. `0` means swap fires at the same tick as bifurcation.
- If `post_bifurcation_swap` key is absent, Phase 3 logic is silently skipped — backward
  compatible with all existing configs.

---

### D-04: Retrigger Policy — Once Per Run

Only the first bifurcation event (earliest tick, any agent) triggers the swap. After the swap
is scheduled (or after it fires), any subsequent bifurcation events are recorded in
`bifurcation_events` as usual but do not create additional swap entries or re-execute the swap.

If no bifurcation event fires before the run ends, no swap occurs and `swap_events` in
`events.json` remains an empty list.

---

### D-05: Event Schema — Minimal

The swap event logged to `events.json` under `"swap_events"`:

```json
{
  "tick": 60,
  "pairs": [["target_A", "target_B"]],
  "triggered_by_agent": "agent_0",
  "bifurcation_tick": 50
}
```

Fields:
- `tick`: the tick at which the swap was applied
- `pairs`: the target pairs that were swapped (mirrors `post_bifurcation_swap.pairs`)
- `triggered_by_agent`: ID of the first agent whose bifurcation event triggered the swap
- `bifurcation_tick`: tick of the triggering bifurcation event

`DataHandling` already has `swap_events: []` reserved in `_write_events_json()`. Phase 3 adds
a `collect_swap_events(events: list[dict])` method mirroring `collect_bifurcation_events`.

---

### D-06: Architecture — Arena Monitors Bifurcation Events In-Tick

The Arena polls for bifurcation events **within the main tick loop**, not only at run-end
(where `_collect_bifurcation_events()` currently runs in `close()`).

The planner must establish how Arena can observe new bifurcation events at each tick. Two
viable paths (planner to choose the cleanest given the IPC architecture):

**Path A — Agents include a bifurcation flag in per-tick metadata**
`MeanFieldMovementModel` adds `"new_bifurcation_events": [...]` to the tick-level metadata
it returns. Arena reads this from `latest_agent_data` after receiving each tick's agent
snapshot, checks for first bifurcation, and schedules the dynamic swap.

**Path B — Arena polls entity detectors directly (same-process path)**
If `entity._movement_plugin.bifurcation_detector` is accessible in-process (true for headless
single-process test runs), Arena can scan all detectors mid-tick, same as `_collect_bifurcation_events()`.

Path A is preferred for multi-process correctness. Path B can be the fallback for tests.

The planner must verify which path is consistent with the production IPC flow (agent managers
communicate via queues; Arena receives agent data through `agents_queues`).

---

### D-07: Intensity Swap Mechanism

The `intensity` attribute on target entities controls the bump-drive strength in the mean-field
model. It is accessed as `entry.get("intensity", 1.0)` from the entity dict in
`mean_field_model.py`. Confirming whether it is mutable on the live entity object or must be
updated via entity manager is required by the planner before implementation.

If `intensity` is a direct Python attribute on the entity object, `_apply_target_position_swap_event`
can be extended to also swap it. If it flows through a separate data path, the swap must reach
that path.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 2 Output Infrastructure
- `src/dataHandling.py` — `DataHandling.collect_bifurcation_events()`, `_write_events_json()`,
  `_bifurcation_events` list. Phase 3 mirrors this for swap events.
- `src/arena.py` — `_apply_target_position_swap_event()`, `_apply_due_target_position_swaps()`,
  `_normalize_target_position_swaps()`, `_collect_bifurcation_events()`, `close()`.
  The existing static swap infrastructure is the base Phase 3 extends.

### Entity / Target Data Model
- `src/models/movement/mean_field_model.py` — How `intensity` is read from entity dicts
  (line ~391). Where `_mf_entities` is populated and what its schema is.
- `src/environment.py` — Entity structure, how target objects are represented.

### Bifurcation Detector
- `src/models/bifurcation.py` — `BifurcationDetector.events` list schema:
  `{agent, tick, metric, target, mode}`. The `tick` and `agent` fields are used by Phase 3.

### Phase 2 Context
- `.planning/phases/02-bifurcation-detection/02-CONTEXT.md` — D-07 (event schema with `agent`,
  `tick`, `metric`, `target`, `mode` fields) and D-08 (events.json format).

### Requirements
- `.planning/REQUIREMENTS.md` — SWAP-01..04

</canonical_refs>

<code_context>
## Existing Code Insights

### What Already Exists (reusable for Phase 3)

- **`Arena._normalize_target_position_swaps()`** — Parses and validates a list of swap specs
  from config. Phase 3's `post_bifurcation_swap` uses the same `pairs` format. Can reuse the
  pair-normalisation logic.
- **`Arena._apply_target_position_swap_event()`** — Applies position swap for a list of pairs.
  Extend this to also swap `intensity` (Decision D-02).
- **`Arena._swap_object_xy_positions()`** — Swaps XY on two entity objects.
- **`DataHandling.collect_bifurcation_events()` / `_write_events_json()`** — Pattern to follow
  for `collect_swap_events()` + writing `swap_events` to `events.json`.
- **`DataHandling._write_events_json()`** — Already has `"swap_events": []` placeholder.

### What Needs to Change

1. **`Arena.__init__()`**: Parse `environment.post_bifurcation_swap` config. Store `_post_bif_swap_cfg`
   and `_post_bif_swap_triggered` flag.
2. **`Arena.run()` tick loop**: After agent snapshots arrive each tick, check for first
   bifurcation event (via D-06 architecture). If found and swap not yet scheduled, compute
   `swap_tick = bif_tick + delay_ticks`, schedule dynamic entry.
3. **`Arena._apply_target_position_swap_event()`**: Extend to swap `intensity` attribute in
   addition to XY position.
4. **`Arena` run-reset logic**: Reset `_post_bif_swap_triggered = False` at the start of each
   run (before the tick loop).
5. **`DataHandling`**: Add `collect_swap_events(events: list[dict])` method. Update
   `_write_events_json()` to write `_swap_events`. Clear in `new_run()`.
6. **Tests**: Unit test for swap scheduling on bifurcation detection; integration test verifying
   `events.json` contains correct `swap_events` entry.

</code_context>

<specifics>
## Specific Notes

- **`delay_ticks: 0`** is valid — swap fires at the exact bifurcation tick.
- **Backward compatibility**: No `post_bifurcation_swap` key → Phase 3 logic silently skipped.
  All existing configs continue to work unchanged.
- **Multi-run experiments**: `_post_bif_swap_triggered` resets to `False` at the start of each
  run. Each run is independent.
- **Multiple pairs**: `pairs` is a list to allow swapping more than one target pair in a single
  swap event, but for current 2-target experiments it will always be a one-element list.
  The planner should support multi-pair without requiring it.
- **Sweep usage** (Phase 5 context): Each sweep run has its own `delay_ticks` value. Phase 3
  only handles one value per run. Sweep tooling (Phase 5) generates multiple config variants.
- **IPC verification required**: The planner must confirm whether `_collect_bifurcation_events()`
  works in the production multi-process run (i.e., whether entity plugins are accessible from
  the Arena process mid-run). If not, Path A (per-tick metadata from agent) is required.

</specifics>

<deferred>
## Deferred Ideas

- **Quality-only swap (no position change)** — Swap intensity but keep targets at their
  original positions. Different experimental protocol; not in Phase 3 scope.
- **Multiple swaps per run** — Re-trigger on each bifurcation for oscillation studies.
  D-04 explicitly defers this.
- **Swap with quality offset** — Instead of exchanging intensity, add a delta (e.g., swap +
  perturbation). Phase 3 is pure exchange only.
- **N-target generalisation** — Extend to 3+ target cycles. Currently out of scope per
  REQUIREMENTS.md Out of Scope table.

</deferred>

---

*Phase: 03-target-swap*
*Context gathered: 2026-04-12*
