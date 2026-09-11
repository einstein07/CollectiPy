# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Per-trial metrics from one run archive (spec Section 6.4).

Everything here is arithmetic on the simulator's own logs; no dynamics, motor
command or decision is touched. Per trial:

    commitment       (a) the BifurcationDetector's own event (events.json): tick and
                         the target it assigned — it always names a real target;
                     (b) the midpoint-aware commitment: the first logged tick from
                         which the ring state is UNIMODAL (one bump: a single local
                         maximum above half the peak activity) and its centre of
                         mass is within ALIGNMENT_TOL_DEG of A, B, C or the A–B
                         midpoint, with the same class on STREAK consecutive ticks.
                         The unimodality gate matters: the circular mean of a
                         symmetric TWO-bump state is also the midpoint, so without
                         it every undecided ring would read as "mean of pair". The
                         streak absorbs the onset transient (the ring starts from
                         zero, and its first tick is not yet its response shape).
                         This is the quantity the detector cannot see: a bump
                         parked between two targets never aligns with either.
    outcome class    at (b): "mean_of_pair" if the wrapped distance from the bump
                         CoM to the A–B midpoint is smaller than to either A or B,
                         otherwise the nearest target by wrapped distance
    ring shape       the number of bumps on ticks 1, 2, at commitment and at the
                         end (`n_peaks_*`), and the gated class on tick 2
                         (`state_t2`: a label, or "bimodal") — the direct evidence
                         of whether the map handed the ring one merged bump or two
    bump CoM         via `compute_center_of_mass` on the logged ring state, at the
                         commitment tick and at the final tick
    arrival          first tick inside `termination.radius` of a target (sub-tick
                         refined by intersecting the logged segment with the circle),
                         which target, or timeout

The bump CoM is read from `<agent>_neural.csv` (the ring state after each tick), the
bearings from `<agent>_targets.csv` (the egocentric bearings the ring was driven
with on that tick), positions from `<agent>_position.csv`.
"""

from __future__ import annotations

import csv
import io
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from models.mean_field_systems import compute_center_of_mass  # noqa: E402


# ---------------------------------------------------------------------------
# Angles
# ---------------------------------------------------------------------------
def wrap_deg(x: float) -> float:
    return (float(x) + 180.0) % 360.0 - 180.0


def circular_midpoint_deg(a: float, b: float) -> float:
    return wrap_deg(a + 0.5 * wrap_deg(b - a))


#: Consecutive ticks with the same gated class required for a commitment.
STREAK = 2
#: A ring node is a bump if it is a local maximum at or above this fraction of the
#: global maximum (and above an absolute floor that rejects the empty ring).
PEAK_FRACTION = 0.5
PEAK_FLOOR = 0.05


def ring_peaks(z: np.ndarray, frac: float = PEAK_FRACTION, floor: float = PEAK_FLOOR) -> list[int]:
    """Indices of the local maxima of the ring state at or above `frac` * max(z)."""
    z = np.asarray(z, dtype=float)
    n = z.shape[0]
    if n == 0:
        return []
    thr = max(frac * float(z.max()), floor)
    return [i for i in range(n)
            if z[i] >= thr and z[i] >= z[i - 1] and z[i] >= z[(i + 1) % n]]


def classify(com_deg: float, bearings: dict[str, float], pair=("A", "B"),
             tol_deg: float | None = None) -> tuple[str | None, dict]:
    """Outcome class of a bump at `com_deg` given target bearings (degrees).

    Returns (label, details). `label` is "mean_of_pair", a target label, or None
    when `tol_deg` is given and nothing (targets or midpoint) is within it.
    """
    mid = circular_midpoint_deg(bearings[pair[0]], bearings[pair[1]])
    dist = {k: abs(wrap_deg(com_deg - v)) for k, v in bearings.items()}
    d_mid = abs(wrap_deg(com_deg - mid))
    details = {"midpoint_deg": mid, "d_mid": d_mid, **{f"d_{k}": v for k, v in dist.items()}}
    if tol_deg is not None and min(min(dist.values()), d_mid) > tol_deg:
        return None, details
    if d_mid < min(dist[pair[0]], dist[pair[1]]):
        return "mean_of_pair", details
    return min(dist, key=dist.get), details


# ---------------------------------------------------------------------------
# Archive readers
# ---------------------------------------------------------------------------
def _member(names, suffix):
    return next((n for n in names if n.endswith(suffix)), None)


def _rows(zf, member):
    with zf.open(member) as raw:
        return list(csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8")))


def read_neural(zf, names) -> dict[int, np.ndarray]:
    """tick -> ring state z (only ticks with a non-zero state)."""
    member = _member(names, "_neural.csv")
    if member is None:
        return {}
    out = {}
    with zf.open(member) as raw:
        reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8"))
        header = next(reader, None)
        if not header:
            return {}
        cols = [i for i, h in enumerate(header) if h.startswith("neuron_")]
        for row in reader:
            try:
                z = np.array([float(row[i]) for i in cols], dtype=float)
            except (ValueError, IndexError):
                continue
            if not np.all(np.isfinite(z)) or not np.any(z != 0.0):
                continue
            out[int(row[0])] = z
    return out


def read_bearings(zf, names, id_to_label: dict[str, str]) -> dict[int, dict[str, float]]:
    """tick -> {label: bearing_deg} from the logged target metadata."""
    member = _member(names, "_targets.csv")
    if member is None:
        return {}
    out = {}
    for row in _rows(zf, member):
        try:
            meta = json.loads(row.get("target_metadata") or "[]")
        except ValueError:
            continue
        bearings = {}
        for entry in meta:
            label = id_to_label.get(str(entry.get("id", "")))
            if label is not None and "angle" in entry:
                bearings[label] = math.degrees(float(entry["angle"]))
        if len(bearings) == len(id_to_label):
            out[int(row["tick"])] = bearings
    return out


def read_positions(zf, names) -> list[tuple[int, float, float]]:
    member = _member(names, "_position.csv")
    if member is None:
        return []
    return [(int(r["tick"]), float(r["pos_x"]), float(r["pos_y"])) for r in _rows(zf, member)]


def read_events(zf, names) -> list[dict]:
    member = _member(names, "events.json")
    if member is None:
        return []
    with zf.open(member) as raw:
        data = json.load(io.TextIOWrapper(raw, encoding="utf-8"))
    return list(data.get("bifurcation_events") or [])


# ---------------------------------------------------------------------------
# Arrival
# ---------------------------------------------------------------------------
def _segment_circle_fraction(p0, p1, centre, radius):
    """Smallest s in [0, 1] with |p0 + s (p1 - p0) - centre| = radius, else None."""
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    fx, fy = x0 - centre[0], y0 - centre[1]
    a = dx * dx + dy * dy
    if a <= 0.0:
        return None
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - radius * radius
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    root = math.sqrt(disc)
    for s in sorted(((-b - root) / (2.0 * a), (-b + root) / (2.0 * a))):
        if -1e-9 <= s <= 1.0 + 1e-9:
            return min(max(s, 0.0), 1.0)
    return None


def first_arrival(positions, targets_xy: dict[str, tuple[float, float]], radius: float):
    """(tick, tick_fine, label) of the first entry inside `radius`, else (None,)*3."""
    prev = None
    for tick, px, py in positions:
        hit, best = None, float("inf")
        for label, (tx, ty) in targets_xy.items():
            d = math.hypot(tx - px, ty - py)
            if d <= radius + 1e-9 and d < best:
                hit, best = label, d
        if hit is not None:
            fine = float(tick)
            if prev is not None:
                s = _segment_circle_fraction((prev[1], prev[2]), (px, py), targets_xy[hit], radius)
                if s is not None:
                    fine = float(prev[0]) + s * (tick - prev[0])
            return tick, fine, hit
        prev = (tick, px, py)
    return None, None, None


# ---------------------------------------------------------------------------
# The per-trial reduction
# ---------------------------------------------------------------------------
def summarise_archive(run_zip: Path, cfg: dict, id_to_label: dict[str, str],
                      tol_deg: float, pair=("A", "B")) -> dict:
    """Reduce one run archive to the 6.4 metrics (a flat dict, None = undefined)."""
    env = cfg["environment"]
    radius = float(env["termination"]["radius"])
    targets_xy = {}
    for name, obj in env["objects"].items():
        label = id_to_label.get(f"{name}.s#0")
        if label is not None:
            targets_xy[label] = (float(obj["position"][0][0]), float(obj["position"][0][1]))
    n = int(env["agents"]["movable_0"]["mean_field_model"]["num_neurons"])
    theta = np.linspace(-np.pi, np.pi, n, endpoint=False)

    with zipfile.ZipFile(run_zip) as zf:
        names = zf.namelist()
        states = read_neural(zf, names)
        bearings = read_bearings(zf, names, id_to_label)
        positions = read_positions(zf, names)
        events = read_events(zf, names)

    out = {
        "n_ticks_logged": len(positions),
        "arrived": False, "reached": "", "t_arrival_ticks": None, "t_arrival_fine": None,
        "timeout": False,
        "committed": False, "t_commit_ticks": None, "commit_class": "none",
        "com_commit_deg": None, "midpoint_commit_deg": None, "n_peaks_commit": None,
        "n_peaks_t1": None, "n_peaks_t2": None, "state_t2": "",
        "t_first_unimodal": None,
        "class_final": "", "com_final_deg": None, "t_final": None, "n_peaks_final": None,
        "t_bif_ticks": None, "bif_target": "",
        "final_x": None, "final_y": None, "final_dist": None,
        "max_abs_state": None, "saw_nonfinite": False,
    }
    for label in id_to_label.values():
        out[f"bearing_{label}_commit_deg"] = None

    if positions:
        _, fx, fy = positions[-1]
        out["final_x"], out["final_y"] = fx, fy
        out["final_dist"] = min(math.hypot(tx - fx, ty - fy) for tx, ty in targets_xy.values())
        tick, fine, hit = first_arrival(positions, targets_xy, radius)
        if hit is not None:
            out.update({"arrived": True, "reached": hit, "t_arrival_ticks": int(tick),
                        "t_arrival_fine": float(fine)})
        else:
            out["timeout"] = True

    # (b) midpoint-aware commitment, gated on a unimodal ring state
    if states:
        peak = 0.0
        history = []            # (tick, gated label or None, com, n_peaks, details)
        for tick in sorted(states):
            z = states[tick]
            peak = max(peak, float(np.max(np.abs(z))))
            if tick not in bearings:
                continue
            com = math.degrees(float(compute_center_of_mass(z, theta)))
            n_peaks = len(ring_peaks(z))
            label, det = classify(com, bearings[tick], pair=pair, tol_deg=tol_deg)
            gated = label if n_peaks == 1 else None
            history.append((tick, gated, com, n_peaks, det))
            if tick == 1:
                out["n_peaks_t1"] = n_peaks
            if tick == 2:
                out["n_peaks_t2"] = n_peaks
                out["state_t2"] = gated if gated is not None else ("bimodal" if n_peaks >= 2 else "none")
            if out["t_first_unimodal"] is None and tick >= 2 and n_peaks == 1:
                out["t_first_unimodal"] = int(tick)
        out["max_abs_state"] = peak
        # Commitment: STREAK consecutive logged ticks with the same gated class; the
        # commitment tick is the FIRST tick of that streak.
        for i in range(len(history) - STREAK + 1):
            window = history[i:i + STREAK]
            labels = {w[1] for w in window}
            ticks_ = [w[0] for w in window]
            if len(labels) == 1 and None not in labels and ticks_ == list(range(ticks_[0], ticks_[0] + STREAK)):
                tick, label, com, n_peaks, det = window[0]
                out.update({
                    "committed": True, "t_commit_ticks": int(tick), "commit_class": label,
                    "com_commit_deg": com, "midpoint_commit_deg": det["midpoint_deg"],
                    "n_peaks_commit": n_peaks,
                })
                for k, v in bearings[tick].items():
                    out[f"bearing_{k}_commit_deg"] = v
                break
        if history:
            tick, _gated, com, n_peaks, _det = history[-1]
            label, _ = classify(com, bearings[tick], pair=pair, tol_deg=None)
            out.update({"class_final": label or "", "com_final_deg": com, "t_final": int(tick),
                        "n_peaks_final": n_peaks})

    # (a) the detector's own record
    if events:
        first = min(events, key=lambda e: e.get("tick", 1 << 30))
        out["t_bif_ticks"] = int(first.get("tick"))
        out["bif_target"] = id_to_label.get(str(first.get("target") or ""), str(first.get("target") or ""))
    return out
