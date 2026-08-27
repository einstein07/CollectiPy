# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""The Section 10 dry-run report. Submits nothing.

    python3 -m campaign.dry_run [--reps N] [--chunk N] [--results-root DIR]
                                [--max-concurrent N] [--sec-per-rep S]

Prints the full condition list with derived parameters and predictions, the asserted
matrix counts, the task/cost estimates, every arrival-censoring and discretisation
flag, z_bellman(0) per grid point (from the precompute report when the cache has been
populated), and the static-control refusal while its boundary is undefined.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from campaign import factors, matrix


def load_z0_report(results_root: Path | None) -> dict:
    if results_root is None:
        return {}
    path = results_root / "table_cache" / "precompute_report.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reps", type=int, default=factors.REPS)
    ap.add_argument("--chunk", type=int, default=factors.CHUNK)
    ap.add_argument("--results-root", type=Path, default=None,
                    help="read z_bellman(0) from <root>/table_cache if present")
    ap.add_argument("--max-concurrent", type=int, default=100)
    ap.add_argument("--sec-per-rep", type=float, default=1.2,
                    help="measured wall seconds per replicate with a warm table "
                         "cache (in-process runner, locally measured ~0.9-1.4)")
    args = ap.parse_args(argv)

    conds = matrix.build_conditions()
    z0map = load_z0_report(args.results_root)

    print("=" * 100)
    print("CAMPAIGN DRY RUN — embodied DDM (nothing is submitted)")
    print("=" * 100)

    # ----- the condition list -------------------------------------------------
    hdr = (f"{'#':>3} {'condition':<22} {'arm':<12} {'A':>6} {'dth':>4} "
           f"{'r0':>6} {'T_max':>6} {'c_e':>7} {'z_qs*':>8} {'z_bell(0)':>9} "
           f"{'a*':>7} {'acc*':>6} {'DT*':>7} {'DT/T':>6} flags")
    print(hdr)
    print("-" * len(hdr))
    censor_risk, disc_cells = [], []
    for c in conds:
        d, p = c.derived, c.predicted
        flags = []
        if p["DT_over_T_max"] > 0.5:
            flags.append("CENSOR-RISK")
            censor_risk.append(c.name)
        if c.discretisation_limited:
            flags.append("disc-limited")
            disc_cells.append(c.name)
        z0 = z0map.get(c.name, {}).get("z0")
        print(f"{c.index:>3} {c.name:<22} {c.arm:<12} {d['A']:>6.3f} "
              f"{c.ang_sep:>4} {d['r0']:>6.3f} {d['T_max']:>6.2f} "
              f"{(f'{c.c_e:g}' if c.c_e is not None else '-'):>7} "
              f"{p['z']:>8.4f} {(f'{z0:.4f}' if z0 is not None else '-'):>9} "
              f"{p['a']:>7.3f} {p['accuracy']:>6.3f} {p['DT']:>7.3f} "
              f"{p['DT_over_T_max']:>6.2f} {','.join(flags)}")

    # ----- counts, asserted ---------------------------------------------------
    n_main = sum(c.arm == "main" for c in conds)
    n_qs = sum(c.arm == "quasi_static" for c in conds)
    n_static = sum(c.arm == "static" for c in conds)
    expect_main = (len(factors.QUALITY_DIFFS) * len(factors.ANGULAR_SEPS)
                   * len(factors.C_E_GRID))
    assert n_main == expect_main, (n_main, expect_main)
    assert n_qs == len(factors.C_E_GRID)
    print(f"\nmatrix: MAIN {n_main} "
          f"({len(factors.QUALITY_DIFFS)} dQ x {len(factors.ANGULAR_SEPS)} dtheta x "
          f"{len(factors.C_E_GRID)} c_e)  +  quasi_static {n_qs}  +  static {n_static}"
          f"  =  {len(conds)} condition-points   [counts asserted]")

    # ----- tasks and cost -----------------------------------------------------
    n_chunks = matrix.chunks_per_condition(args.reps, args.chunk)
    total_tasks = len(conds) * n_chunks
    total_reps = len(conds) * args.reps
    core_h = total_reps * args.sec_per_rep / 3600.0
    waves = math.ceil(total_tasks / args.max_concurrent)
    wall_min = waves * args.chunk * args.sec_per_rep / 60.0
    print(f"tasks : {len(conds)} conditions x {n_chunks} chunks of {args.chunk} "
          f"= {total_tasks} array tasks, {total_reps} replicates")
    print(f"cost  : ~{core_h:.0f} core-hours at {args.sec_per_rep:g} s/replicate "
          f"(warm table cache); ~{wall_min:.0f} min wall-clock at "
          f"%{args.max_concurrent} concurrency")
    print("        (assumes the precompute job ran: cold-cache Bellman solves add "
          "~1 s x replicates that miss)")

    # ----- flags --------------------------------------------------------------
    print(f"\narrival-censoring risk (predicted DT > 0.5 T_max): "
          f"{', '.join(censor_risk) if censor_risk else 'none'}")
    print(f"discretisation-limited (Section 1.3, c_e in "
          f"{sorted(factors.DISCRETISATION_LIMITED_C_E)}): {len(disc_cells)} cells")

    # ----- z_bellman(0) for the Section 5.2 decision --------------------------
    print("\nz_bellman(0) per grid point"
          + (" [from precompute report]:" if z0map else
             ":  NOT AVAILABLE — run `python3 -m campaign.precompute_tables "
             "--results-root <dir>` first, then re-run this dry run with "
             "--results-root <dir>."))
    if z0map:
        base_cell = [c for c in conds if c.arm == "main"
                     and c.q_diff == factors.CONTROL_QUALITY_DIFF
                     and c.ang_sep == factors.CONTROL_ANGULAR_SEP]
        print("  baseline cell (the Section 5.2 static-control proposal "
              "z_manual := z_bellman(0) per c_e):")
        for c in base_cell:
            e = z0map.get(c.name)
            if e:
                print(f"    c_e {c.c_e:>7g}: z_bellman(0) = {e['z0']:.6f}   "
                      f"(quasi-static z*(0) = {e['z_myopic0']:.6f}, "
                      f"gap {e['gap_pct']:.1f}%)")

    # ----- Section 5.2 gate ---------------------------------------------------
    print()
    if factors.STATIC_CONTROL_Z is None:
        print("STATIC CONTROL: REFUSED. `controls.static.z_manual` "
              "(campaign/factors.py: STATIC_CONTROL_Z) is unset — this is an "
              "unresolved METHODOLOGICAL parameter, and no value is invented on "
              "your behalf (CAMPAIGN_SPEC 5.2). The main matrix and the "
              "quasi-static control run without it. To unblock, choose the "
              "boundary from the z_bellman(0) evidence above and set "
              "STATIC_CONTROL_Z.")
    else:
        print(f"static control: ENABLED with {n_static} boundary values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
