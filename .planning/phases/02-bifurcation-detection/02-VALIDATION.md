---
phase: 02
slug: bifurcation-detection
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-10
---

# Phase 02 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 6.2.5 (already installed system-wide) |
| **Config file** | `pyproject.toml` — `testpaths = ["tests"]`, `pythonpath = ["src"]` (pytest 6.x: root conftest.py adds src/ to sys.path) |
| **Quick run command** | `python3 -m pytest tests/test_bifurcation.py -x` |
| **Full suite command** | `python3 -m pytest tests/ -x` |
| **Estimated runtime** | ~5 seconds (no Numba JIT — BifurcationDetector is pure Python/NumPy) |

---

## Sampling Rate

- **After every task commit:** Run `python3 -m pytest tests/test_bifurcation.py -x`
- **After every plan wave:** Run `python3 -m pytest tests/ -x`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** ~5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 02-W0-01 | W0 | 0 | BIF-01..04 | — | N/A | unit stubs | `python3 -m pytest tests/test_bifurcation.py -x` | ❌ W0 | ⬜ pending |
| 02-01-01 | BIF core | 1 | BIF-01 | — | N/A | unit | `python3 -m pytest tests/test_bifurcation.py::test_spike_detection -x` | ❌ W0 | ⬜ pending |
| 02-01-02 | BIF core | 1 | BIF-01 | — | N/A | unit | `python3 -m pytest tests/test_bifurcation.py::test_no_spike_below_threshold -x` | ❌ W0 | ⬜ pending |
| 02-01-03 | BIF core | 1 | BIF-01 | — | N/A | unit | `python3 -m pytest tests/test_bifurcation.py::test_no_spike_monotonic -x` | ❌ W0 | ⬜ pending |
| 02-01-04 | BIF core | 1 | BIF-01 | — | N/A | unit | `python3 -m pytest tests/test_bifurcation.py::test_spike_min_separation -x` | ❌ W0 | ⬜ pending |
| 02-01-05 | BIF core | 1 | BIF-01 | — | N/A | unit | `python3 -m pytest tests/test_bifurcation.py::test_retriggerable -x` | ❌ W0 | ⬜ pending |
| 02-02-01 | BIF integration | 2 | BIF-01,BIF-02 | — | N/A | integration | `python3 -m pytest tests/test_bifurcation.py::test_standard_model_bifurcation -x` | ❌ W0 | ⬜ pending |
| 02-02-02 | BIF integration | 2 | BIF-02 | — | N/A | integration | `python3 -m pytest tests/test_bifurcation.py::test_sfa_model_bifurcation -x` | ❌ W0 | ⬜ pending |
| 02-03-01 | Output | 2 | BIF-03 | — | N/A | unit | `python3 -m pytest tests/test_bifurcation.py::test_events_json_schema -x` | ❌ W0 | ⬜ pending |
| 02-04-01 | Config | 2 | BIF-04 | — | N/A | unit | `python3 -m pytest tests/test_bifurcation.py::test_custom_config_respected -x` | ❌ W0 | ⬜ pending |
| 02-04-02 | Config | 2 | BIF-04 | — | N/A | unit | `python3 -m pytest tests/test_bifurcation.py::test_no_bifurcation_empty_list -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_bifurcation.py` — test stubs for all BIF-01 through BIF-04 tests listed above

*Existing infrastructure from Phase 1 covers test runner setup (pyproject.toml, conftest.py, fixtures/).*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| events.json written at correct time (before close/archive) | BIF-03 | Requires running a full arena experiment | Run a 2-target mean-field experiment; verify run folder contains events.json before folder is archived |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 5s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
