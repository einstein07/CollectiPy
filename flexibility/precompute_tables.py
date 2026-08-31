# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Populate the Bellman table cache before the array runs (Section 9.7).

    python3 -m flexibility.precompute_tables --cache-dir <dir> [--workers N]

One table per ddm_bellman condition -- the only arm that solves a free-boundary
problem at all; the two ring-attractor arms have no boundary.

The cache is populated BY RUNNING THE MODEL — one throwaway execution of each
condition's replicate 0, output to scratch and discarded — because the cache key is
computed inside `_bellman_threshold` from the exact floats the solver receives, and
reimplementing that derivation here would invite a key mismatch that silently
re-solves once per replicate.

Why this matters more at this velocity than it used to. N_t is ceil((r0/v)/1e-3),
which the drop to v = 0.01 took from 8 660 to 43 302; a measured cold solve at
N_x = 1601 is ~8.6 s. Letting 100 replicates each re-solve the same Crank-Nicolson
PDE would be ~2.4 hours of pure duplication per condition, against ~3 minutes to
solve all 23 once. |A| changes with delta, so there are 23 distinct tables; the swap
preserves |A| and flips only its sign, so ONE table per condition stays valid on both
sides of the world change.

The solver's 1.5x horizon check runs HERE, once per condition, rather than once per
replicate in the array (where it is off): if z(t) moves when the horizon moves, the
horizon is doing modelling work it should not be doing, and that is worth one loud
failure before 920 tasks are queued.

Writes `<cache>/precompute_report.json`: condition -> {delta, A, c_e, N_t, T_max,
z_myopic_predicted, wall_time_s}, so the solved grid can be inspected before the
array runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from flexibility import factors, genconfig, matrix  # noqa: E402

HORIZON_CHECK_FACTOR = 1.5


def bellman_conditions() -> list[matrix.Condition]:
    """The conditions that actually solve a table.

    delta = 0 is included deliberately: it is a real cell that runs, and it is the
    one where the drift-dependent boundary degenerates, so it is worth seeing the
    solver's answer before the array does.
    """
    return [c for c in matrix.build() if c.arm == "ddm_bellman"]


def solve_one(args) -> dict:
    """Worker: run replicate 0 of one condition with the cache dir set."""
    cond_name, cache_dir = args
    from flexibility.run_chunk import InProcessRunner

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
    return {
        "condition": cond.name,
        "delta": cond.delta,
        "A": cond.derived["A"],
        "c_e": cond.derived["c_e"],
        "N_t": cond.derived["N_t"],
        "T_max": cond.derived["T_max"],
        "z_myopic_predicted": cond.derived["pred_z"],
        "wall_time_s": time.time() - t0,
    }


def main(argv=None) -> int:
    """Solve every ddm_bellman table into the cache."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", help="a single condition name, for debugging")
    args = ap.parse_args(argv)

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    conds = bellman_conditions()
    if args.only:
        conds = [c for c in conds if c.name == args.only]
        if not conds:
            raise SystemExit(f"no ddm_bellman condition named {args.only!r}")

    print(f"solving {len(conds)} Bellman tables into {args.cache_dir} "
          f"({args.workers} workers, N_t = {conds[0].derived['N_t']})")

    t0 = time.time()
    payload = [(c.name, str(args.cache_dir)) for c in conds]
    results, failures = [], []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for item, res in zip(payload, pool.map(solve_one, payload)):
                results.append(res)
                print(f"  {res['condition']:<28} {res['wall_time_s']:6.1f}s")
    else:
        for item in payload:
            try:
                res = solve_one(item)
                results.append(res)
                print(f"  {res['condition']:<28} {res['wall_time_s']:6.1f}s")
            except Exception as exc:  # noqa: BLE001
                failures.append({"condition": item[0], "error": repr(exc)})
                print(f"  {item[0]:<28} FAILED: {exc!r}")

    report = {
        "cost_of_error": factors.COST_OF_ERROR,
        "campaign_seed": factors.CAMPAIGN_SEED,
        "horizon_check_factor": HORIZON_CHECK_FACTOR,
        "conditions": results,
        "failures": failures,
    }
    report_path = args.cache_dir / "precompute_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    n_tables = len(list(args.cache_dir.glob("*.npz")))
    print(f"\ndone in {time.time() - t0:.1f}s; report -> {report_path}")
    print(f"cache holds {n_tables} table file(s) for {len(results)} condition(s)")
    if results and n_tables > len(results):
        print("  NOTE: more tables than conditions. Each condition solves exactly "
              "one table, so the extras are stale — most likely from a run at a "
              "different criterion. A stale table is never USED (the key includes "
              "c_e), but the cache should be cleared to keep it legible.")
    if failures:
        print(f"{len(failures)} condition(s) FAILED — do not launch the array")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
