# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Populate the Bellman table cache before the array runs (CAMPAIGN_SPEC 7.3).

    python3 -m campaign.precompute_tables --results-root <dir> [--workers N]

One table per MAIN-matrix condition-point (the control arms never solve the
free-boundary problem). The cache is populated BY RUNNING THE MODEL — one throwaway
execution of each condition's replicate 0, output to scratch and discarded — because
the cache key is computed inside `_bellman_threshold` from the exact floats the solver
receives; reimplementing that derivation here would invite a key mismatch that
silently re-solves 1000 times per condition.

The solver's 1.5x horizon check runs HERE, once per condition, instead of once per
replicate in the array (where it is off): if z(t) moves when the horizon moves, the
horizon is doing modelling work it should not be doing, and that is worth one loud
failure before 1400 tasks are queued.

Writes `<cache>/precompute_report.json`: condition -> {z0, z_myopic0, gap, T_max,
wall_time_s, horizon_check}. The dry run reads it for the Section 10 z_bellman(0)
column, which is also the Section 5.2 evidence for choosing the static control's
z_manual.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for p in (str(_ROOT), str(_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from campaign import genconfig, matrix  # noqa: E402

HORIZON_CHECK_FACTOR = 1.5


def solve_one(args) -> dict:
    """Worker: run replicate 0 of one condition with the cache dir set."""
    cond_name, cache_dir = args
    from campaign.run_chunk import InProcessRunner

    cond = matrix.find_condition(cond_name)
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix=f"precompute_{cond.name}_") as scratch:
        cfg = genconfig.replicate_config(
            cond, 0, os.path.join(scratch, "out"),
            table_cache_dir=cache_dir,
            horizon_check_factor=HORIZON_CHECK_FACTOR,
        )
        cfg_path = Path(scratch) / "config.json"
        cfg_path.write_text(json.dumps(cfg))
        InProcessRunner().run(cfg_path)
    return {"condition": cond.name, "wall_time_s": time.time() - t0}


def cache_entry_for(cond, tables) -> dict | None:
    """Match a condition to its cache file by (c_e, A, N_t) — unique across the grid,
    and exact: the model deduces A as |q0 - q1| from the very floats the generator
    wrote into the config."""
    A = cond.derived["q0"] - cond.derived["q1"]
    for path, meta in tables:
        if (meta.get("c_e") == float(cond.c_e)
                and int(meta.get("N_t", -1)) == int(cond.derived["N_t"])
                and math.isclose(meta.get("A", -1.0), A, rel_tol=1e-12)):
            return {"path": str(path), **meta}
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results-root", required=True, type=Path)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--force", action="store_true",
                    help="re-solve even for conditions already in the report")
    args = ap.parse_args(argv)

    cache_dir = (args.cache_dir or (args.results_root / "table_cache")).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    report_path = cache_dir / "precompute_report.json"
    report = {}
    if report_path.is_file() and not args.force:
        report = json.loads(report_path.read_text())

    from models.bellman_table_cache import scan_tables  # noqa: E402

    conds = [c for c in matrix.build_conditions() if c.arm == "main"]
    todo = [c for c in conds if c.name not in report]
    print(f"precompute: {len(conds)} bellman tables, {len(todo)} to solve "
          f"({len(conds) - len(todo)} already reported), workers={args.workers}, "
          f"horizon check x{HORIZON_CHECK_FACTOR} ON")

    t0 = time.time()
    work = [(f"{c.arm}/{c.name}", str(cache_dir)) for c in todo]
    if args.workers > 1 and len(work) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, res in enumerate(pool.map(solve_one, work)):
                print(f"  [{i + 1}/{len(work)}] {res['condition']} "
                      f"({res['wall_time_s']:.1f}s)")
    else:
        for i, w in enumerate(work):
            res = solve_one(w)
            print(f"  [{i + 1}/{len(work)}] {res['condition']} "
                  f"({res['wall_time_s']:.1f}s)")

    # Build the report by matching every main condition to its cached table.
    tables = list(scan_tables(cache_dir))
    missing = []
    for cond in conds:
        entry = cache_entry_for(cond, tables)
        if entry is None:
            missing.append(cond.name)
            continue
        gap = entry["z_myopic_onset"] - entry["z0"]
        h_ok = entry.get("horizon_ok")
        report[cond.name] = {
            "z0": entry["z0"],
            "z_myopic0": entry["z_myopic_onset"],
            "gap_pct": 100.0 * gap / max(entry["z_myopic_onset"], 1e-12),
            "T_max": entry.get("T_max"),
            "N_t": entry.get("N_t"),
            "wall_time_s": entry["wall_time_s"],
            # None = not recorded; False is EXPECTED here, not an error: T_max is
            # the physical arrival deadline (r0/v), so z(t) genuinely depends on
            # it — the check's premise (horizon as numerical cutoff) does not hold
            # in the embodied setting. Recorded so the dependence is measured.
            "horizon_ok": (None if h_ok is None or math.isnan(h_ok) else bool(h_ok)),
        }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    n_files = len(tables)
    n_hfail = sum(1 for e in report.values() if e.get("horizon_ok") is False)
    print(f"\ncache: {n_files} tables on disk, {len(report)} conditions mapped, "
          f"total {time.time() - t0:.0f}s -> {report_path}")
    if n_hfail:
        print(f"note : {n_hfail} conditions move their boundary when the horizon is "
              "scaled x1.5 — EXPECTED under an embodied arrival deadline (T_max = "
              "r0/v is physics, not a numerical cutoff); recorded per condition in "
              "the report.")
    if missing:
        print(f"ERROR: no cache entry matched for: {', '.join(missing)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
