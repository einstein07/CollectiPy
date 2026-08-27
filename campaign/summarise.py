# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Per-replicate summary rows (CAMPAIGN_SPEC.md Sections 1.3, 7.4).

One run archive in, one CSV row out. The bulk campaign ships ONLY these rows to
shared storage — the raw archives stay on node-local scratch — so the row must carry
everything the analysis needs:

- the choice and both clocks (the DDM's own commitment, and the physical arrival);
- the boundary at onset and at the crossing;
- the REALISED log-odds at the crossing, `a = 2 A |x| / c^2`, logged for every trial
  so the Section 1.3 discretisation contamination is measured rather than assumed
  (`x` is read at the first tick on which the accumulator reports committed, i.e. at
  tick resolution; the overshoot it contains relative to the predicted `a*` IS the
  quantity of interest);
- censoring status: a run that never arrives inside the time limit is excluded from
  accuracy downstream, never silently scored.
"""

from __future__ import annotations

import csv
import io
import json
import math
import zipfile
from pathlib import Path
from typing import Optional

CORRECT_ID = "static_0.s#0"

#: Column order of the chunk CSVs. Append-only; analysis code indexes by name.
FIELDS = [
    "condition", "arm", "replicate", "sensory_seed", "internal_seed",
    "censored", "arrived", "arrival_tick", "arrival_time", "reached_id",
    "arrival_correct",
    "committed", "committed_id", "commit_correct", "rt",
    "z_onset", "tick_commit", "x_at_commit", "z_at_commit", "a_realised",
    "A_hat", "A_true", "c", "discretisation_limited",
]


def _f(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def summarise_run(run_zip: Path, cfg: dict, cond_meta: dict, replicate: int) -> dict:
    """Reduce one run archive to its summary row."""
    env = cfg["environment"]
    tick_rate = max(int(env.get("ticks_per_second", 1)), 1)
    radius = float(env["termination"]["radius"])
    targets = {
        f"{name}.s#0": (float(obj["position"][0][0]), float(obj["position"][0][1]))
        for name, obj in env["objects"].items()
    }
    meta = env["results"]["sweep_metadata"]

    with zipfile.ZipFile(run_zip) as zf:
        names = zf.namelist()
        pos_member = next(n for n in names if n.endswith("_position.csv"))
        prows = list(csv.DictReader(io.TextIOWrapper(zf.open(pos_member))))
        ddm_member = next((n for n in names if n.endswith("_ddm.csv")), None)
        drows = (list(csv.DictReader(io.TextIOWrapper(zf.open(ddm_member))))
                 if ddm_member else [])

    # --- arrival, scored exactly like the termination criterion ------------
    last = prows[-1]
    tick = int(last["tick"])
    px, py = float(last["pos_x"]), float(last["pos_y"])
    reached, best = "", float("inf")
    for tid, (tx, ty) in targets.items():
        dist = math.hypot(tx - px, ty - py)
        if dist < best:
            reached, best = tid, dist
    arrived = best <= radius + 1e-9

    row = {
        "condition": cond_meta["condition"],
        "arm": cond_meta["arm"],
        "replicate": int(replicate),
        "sensory_seed": meta["sensory_seed"],
        "internal_seed": meta["internal_seed"],
        "censored": int(not arrived),
        "arrived": int(arrived),
        "arrival_tick": tick,
        "arrival_time": tick / tick_rate,
        "reached_id": reached if arrived else "",
        "arrival_correct": int(arrived and reached == CORRECT_ID),
        "committed": 0, "committed_id": "", "commit_correct": "",
        "rt": "", "z_onset": "", "tick_commit": "", "x_at_commit": "",
        "z_at_commit": "", "a_realised": "", "A_hat": "", "A_true": "", "c": "",
        "discretisation_limited": int(cond_meta.get("discretisation_limited", False)),
    }

    # --- the decision variable's own record --------------------------------
    live = [r for r in drows if (_f(r.get("z")) or 0.0) > 0.0]
    if live:
        row["z_onset"] = _f(live[0].get("z"))
        # `committed` carries the committed target INDEX ('' until commitment), so
        # the crossing row is the first with a non-empty value.
        first_commit = next(
            (r for r in live if (r.get("committed") or "") != ""), None
        )
        lastd = live[-1]
        row["A_hat"] = _f(lastd.get("A_hat"))     # the assumed (known) magnitude
        row["A_true"] = _f(lastd.get("A_true"))   # the realised running drift
        row["c"] = _f(lastd.get("c"))
        if first_commit is not None:
            # a_realised uses A_hat: it is the log-odds AT THE CROSSING under the
            # drift magnitude the boundary was solved for (Section 1.3).
            A_hat, c = _f(lastd.get("A_hat")), _f(lastd.get("c"))
            x_c, z_c = _f(first_commit.get("x")), _f(first_commit.get("z"))
            row.update({
                "committed": 1,
                "committed_id": lastd.get("committed_id") or "",
                "commit_correct": int((lastd.get("committed_id") or "") == CORRECT_ID),
                "rt": _f(lastd.get("rt")),
                "tick_commit": int(first_commit["tick"]),
                "x_at_commit": x_c,
                "z_at_commit": z_c,
                "a_realised": (
                    2.0 * A_hat * abs(x_c) / (c * c)
                    if None not in (A_hat, c, x_c) and c > 0 else ""
                ),
            })
    return row


def write_chunk_csv(path: Path, rows: list[dict]) -> None:
    """Write one chunk's rows; the caller moves the file into place atomically."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_chunk_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))
