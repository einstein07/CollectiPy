# Phase 2: Bifurcation Detection - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-10
**Phase:** 02-bifurcation-detection
**Areas discussed:** Detection mechanism, Spike detection rule, Fire mode, Per-agent vs run-level, Output format, Config namespace, Re-trigger gate

---

## Gray Area Selection

| Option | Description | Selected |
|--------|-------------|----------|
| Detector placement | Standalone class vs inline logic | (superseded by algorithm discussion) |
| Per-agent vs run-level | Per-agent or single run-level event | ✓ |
| Output format | pkl, sidecar JSON, or both | ✓ |
| Config namespace | Under mean_field_model vs environment | ✓ |

**User's response to gray area prompt:** Rather than selecting from the predefined list, the user introduced the Jacobian eigenvalue (λ₁) approach from arXiv:2602.05683 — "From Vision to Decision: Neuromorphic Control for Autonomous Navigation and Tracking". This redirected the discussion to focus on the detection algorithm first.

---

## Detection Mechanism

| Option | Description | Selected |
|--------|-------------|----------|
| Eigenvalue λ₁ spike | Compute Re(λ₁) of Jacobian, detect local maximum spike | ✓ |
| Sliding-window variance | Low positional variance over W ticks + angular proximity | |
| Both, pluggable at runtime | Two implementations selectable via JSON config | |

**User's choice:** Eigenvalue λ₁ spike
**Notes:** User cited the paper as proposing a "more principled, analytically sound" approach. Asked specifically whether it could work for both the standard model and the SFA model — confirmed yes (same Jacobian formula, SFA just has non-zero s).

---

## Spike Detection Rule

| Option | Description | Selected |
|--------|-------------|----------|
| Local maximum + threshold | 3-sample micro-buffer, fire at local max above threshold | ✓ |
| Threshold crossing | Fire immediately when λ₁ > threshold | |

**User's choice:** Local maximum + threshold
**Notes:** Accepted the recommended option without modification.

---

## Fire Mode

| Option | Description | Selected |
|--------|-------------|----------|
| Once per run | Lock after first spike | |
| Re-triggerable | Detector can fire multiple times per run | ✓ |

**User's choice:** Re-triggerable
**Notes:** User noted experiments can have more than two targets, leading to multiple sequential bifurcation events. "What striked my interest is that what they describe sounds a more principled approach, and analytically sound."

---

## Per-Agent vs Run-Level

| Option | Description | Selected |
|--------|-------------|----------|
| Per-agent, stored independently | Each agent logs its own events; aggregation in Phase 4 | ✓ |
| First-agent-to-commit | Single scalar per run | |

**User's choice:** Per-agent, stored independently
**Notes:** Accepted the recommended option.

---

## Output Format

| Option | Description | Selected |
|--------|-------------|----------|
| Sidecar JSON per run (events.json) | Separate file alongside pkl files | ✓ |
| Into existing pkl structure | Add key to existing pkl dict | |
| Both JSON + pkl | Write to both | |

**User's choice:** Sidecar JSON per run
**Notes:** Accepted the recommended option. Noted the schema anticipates Phase 3 swap_events key.

---

## Config Namespace

| Option | Description | Selected |
|--------|-------------|----------|
| Under mean_field_model (per-agent) | Nested inside agent spec | ✓ |
| Under environment (per-experiment) | Single shared config | |

**User's choice:** Under mean_field_model per-agent
**Notes:** Accepted the recommended option.

---

## Re-trigger Gate

| Option | Description | Selected |
|--------|-------------|----------|
| spike_min_separation ticks | Suppress re-detection for N ticks after firing | ✓ |
| Explicit reset() call | Swap logic calls reset() to re-arm | |

**User's choice:** spike_min_separation ticks
**Notes:** Accepted the recommended option.

---

*Discussion conducted: 2026-04-10*
