# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Pre-launch checks and the grid report (Sections 4, 5, 8, 9).

    python3 -m flexibility.preflight [--strict]

Everything here is cheap and runs before a single task is queued. It answers the two
questions that decide whether the campaign is worth launching:

  1. Are the arms actually matched? (Section 3 -- same seed, same world, differing
     only in the decision rule.) Checked by generating configs and diffing them,
     not by reading the templates.
  2. Does the delta grid land where Section 4.3 says it does? Checked against the
     boundary the MODEL'S OWN SOLVER returns at each cell's criterion, so a design
     assumption that the policy does not honour shows up here rather than as a
     structural zero in the results.

`--strict` turns warnings into a non-zero exit, for use in the submit script.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from flexibility import factors, genconfig, matrix  # noqa: E402


class Report:
    """Collects failures and warnings so every problem is reported, not just the first."""

    def __init__(self):
        """Initialize the instance."""
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def fail(self, msg: str) -> None:
        """Record a launch-blocking problem."""
        self.failures.append(msg)

    def warn(self, msg: str) -> None:
        """Record something the operator must read but that need not block."""
        self.warnings.append(msg)


# ---------------------------------------------------------------------------
def check_locked_constants(rep: Report) -> None:
    """Assert the constants restated in `factors` against the live code."""
    from models.ddm_systems import DriftDiffusionSystem

    live = DriftDiffusionSystem.c_tau_linearised(
        math.radians(factors.ANGULAR_SEP_DEG), "midpoint"
    )
    if not math.isclose(live, factors.C_TAU, rel_tol=1e-9):
        rep.fail(
            f"factors.C_TAU = {factors.C_TAU!r} disagrees with the model's "
            f"c_tau_linearised({factors.ANGULAR_SEP_DEG} deg, 'midpoint') = {live!r}. "
            "Every criterion in the campaign is derived from it."
        )

    c = math.sqrt(2.0) * factors.WHITE_RATE
    if not math.isclose(c, 0.1, rel_tol=1e-6):
        rep.warn(
            f"c = sqrt(2)*white_rate = {c:.6f}, not the 0.100 Section 4.1 locks. "
            "Every landmark scales with it."
        )


def check_templates(rep: Report) -> None:
    """Every arm's template must exist and carry the block that arm patches."""
    for arm in factors.ARMS:
        try:
            genconfig.load_template(arm)
        except genconfig.ConfigGenerationError as exc:
            rep.fail(str(exc))


def check_arms_matched(rep: Report) -> None:
    """Section 3/9: arms differ ONLY in the model block, and share the trial seed."""
    grid = [d for d in factors.delta_grid()]
    probes = [grid[0], factors.reference_delta(), grid[-1]]
    for delta in probes:
        for i in range(len(factors.ARMS) - 1):
            try:
                genconfig.assert_arms_matched(
                    factors.ARMS[i], factors.ARMS[i + 1], delta, 1
                )
            except AssertionError as exc:
                rep.fail(str(exc))
            except genconfig.ConfigGenerationError as exc:
                rep.fail(f"config generation failed: {exc}")


def check_grid(rep: Report) -> None:
    """Does the grid resolve the transition Section 4.3 says it should?"""
    marks = {m["name"]: m for m in matrix.landmarks()}
    conds = [c for c in matrix.build() if c.arm == "ddm_bellman"]

    band = matrix.usable_band()
    if band["n_grid_points"] < 5:
        best = max(((t, matrix.usable_band(ticks_per_second=t)) for t in (2, 5, 10, 20)),
                   key=lambda tb: tb[1]["n_grid_points"])
        rep.warn(
            f"only {band['n_grid_points']} grid point(s) are USABLE (first choice "
            "reliable enough for a reversal to be defined, and reversal latency "
            "resolvable). The delta grid of Section 5 was placed around landmarks "
            "derived at a constant zeta = 1.1; at a fixed cost of error those "
            "landmarks move and three of them cease to exist, so the grid no longer "
            "straddles anything. Two independent levers, neither applied here "
            "because both are design decisions:\n"
            f"           (a) tick rate — ticks_per_second = {best[0]} would give "
            f"{best[1]['n_grid_points']} usable points (band ratio "
            f"{best[1]['ratio']:.1f} vs {band['ratio'] or float('nan'):.1f}); it "
            "raises only the UPPER edge, which is sampling resolution;\n"
            f"           (b) grid placement — the usable band starts at "
            f"{band['lo'] * 100:.2f}%, so the log leg's current span of 0.10%-4.00% "
            "spends 11 of its 18 points below it, at near-chance accuracy."
        )

    infeasible = [c for c in conds
                  if c.delta > 0.0 and not c.derived["pred_reversal_feasible"]
                  and not c.derived["is_anchor"]]
    if len(infeasible) > len(conds) / 2:
        rep.fail(
            f"{len(infeasible)} of {len(conds) - 1} non-zero grid points cannot "
            "reverse within the travel budget at all. A 0% reversal rate from those "
            "cells is the arena, not the model. Either the cost of error "
            "(factors.COST_OF_ERROR) or the velocity is wrong for this grid."
        )

    rep.warn(
        f"cost of error is FIXED at c_e = {factors.COST_OF_ERROR}, so zeta varies "
        "across the grid (realised range "
        f"{min(c.derived['pred_zeta'] for c in conds if c.delta > 0):.3f} to "
        f"{max(c.derived['pred_zeta'] for c in conds if c.delta > 0):.3f}) and the "
        "Section 4.3 landmarks and Section 5 grid -- both derived at a constant "
        "zeta = 1.1 -- do NOT describe this campaign. Read the realised columns "
        "above, not the design document's numbers."
    )

    # How many cells give a first choice reliable enough for a reversal to be
    # defined on most trials? At a fixed cost of error this, not feasibility, is
    # what limits the usable grid.
    near_chance = [c for c in conds
                   if c.delta > 0.0 and c.derived["pred_initial_accuracy"] < 0.75]
    if near_chance:
        rep.warn(
            f"{len(near_chance)} of {len(conds) - 1} non-zero cells commit at under "
            "75% accuracy (lowest: "
            f"{min(c.derived['pred_initial_accuracy'] for c in near_chance) * 100:.1f}%"
            f" at delta = {min(near_chance, key=lambda c: c.delta).delta * 100:.4f}%). "
            "In those cells about half the trials commit to the WORSE option, where "
            "the swap makes the choice correct and there is nothing to reverse. "
            "Section 7 requires them analysed separately, so the effective replicate "
            "count there is roughly half of REPS -- budget for it rather than "
            "discovering it."
        )

    # Reversal latency is the DV's resolution limit at this tick rate.
    unresolvable = [c for c in conds
                    if c.delta > 0.0 and not c.derived["is_anchor"]
                    and not c.derived["pred_reversal_latency_resolvable"]]
    if len(unresolvable) > (len(conds) - 1) / 2:
        rep.warn(
            f"{len(unresolvable)} non-anchor cells have a reversal latency under 3 "
            f"ticks at ticks_per_second = {factors.TICKS_PER_SECOND}; only the "
            "occurrence of a reversal is observable there, not its timing."
        )

    rev = marks["reversal"]["delta"]
    if rev is not None and rev >= max(c.delta for c in conds):
        rep.fail("the reversal boundary lies above the top of the grid: no cell can "
                 "reverse within the travel budget.")


def check_time_limit(rep: Report) -> None:
    """The time limit has to cover the worst INFORMATIVE cell, not the worst cell."""
    t_b = matrix.geometry()["T_b"]
    worst = None
    for c in matrix.build():
        if c.arm != "ddm_bellman" or c.delta <= 0.0:
            continue
        if not c.derived["pred_reversal_feasible"]:
            continue
        # commit + delay + reverse + traverse to the OTHER target
        need = c.derived["pred_total"] + t_b
        if worst is None or need > worst[1]:
            worst = (c, need)
    if worst and worst[1] > factors.TIME_LIMIT:
        rep.warn(
            f"time_limit = {factors.TIME_LIMIT} s is below the {worst[1]:.0f} s the "
            f"worst reversal-feasible cell ({worst[0].name}) needs; it will censor. "
            "Censoring is a reported quantity, but it should not eat a cell the "
            "design counts as informative."
        )


# ---------------------------------------------------------------------------
def print_grid_table() -> None:
    """The Section 4/5 report: what each cell will actually do."""
    marks = matrix.landmarks()
    g = matrix.geometry()

    print("=" * 100)
    print("FLEXIBILITY CAMPAIGN — PREFLIGHT")
    print("=" * 100)
    print(f"  arms              : {', '.join(factors.ARMS)}")
    print(f"  cost of error     : c_e = {factors.COST_OF_ERROR}")
    print(f"  c = sqrt(2)*eta   : {factors.NOISE_C:.6f}")
    print(f"  v                 : {factors.LINEAR_VELOCITY} m/s   "
          f"T_b = R/v = {g['T_b']:.1f} s")
    print(f"  Bellman horizon   : T_max = r0/v = {g['T_max']:.4f} s   "
          f"N_t = {g['N_t']} (dt = {factors.BELLMAN_DT})")
    print(f"  swap delay        : {factors.SWAP_DELAY_TICKS} tick "
          f"= {factors.SWAP_DELAY_TICKS / factors.TICKS_PER_SECOND:.1f} s")
    print(f"  time limit        : {factors.TIME_LIMIT} s")
    print(f"  runs / tasks      : {matrix.total_runs()} / {matrix.total_tasks()}")
    print()

    print("LANDMARKS — solved on the REALISED policy, not on a fixed zeta")
    print(f"  {'name':<22} {'delta':>9} {'A/c':>8} {'t_c':>9} {'T_rev':>9} "
          f"{'init acc':>9}   meaning")
    for m in marks:
        if m["delta"] is None:
            print(f"  {m['name']:<22} {'--':>9} {'--':>8} {'--':>9} {'--':>9} "
                  f"{'--':>9}   NOT REACHED on delta in (0, 1]: {m['meaning']}")
            continue
        print(f"  {m['name']:<22} {m['delta_pct']:>8.4f}% {m['a_over_c']:>8.4f} "
              f"{m['t_commit']:>8.2f}s {m['t_reversal']:>8.2f}s "
              f"{m['initial_accuracy'] * 100:>8.2f}%   {m['meaning']}")
    print()

    print("GRID — realised per cell, from the model's own boundary solver")
    print(f"  {'delta':>9} {'A/c':>8} {'c_e':>10} {'z':>8} {'zeta':>6} "
          f"{'init acc':>9} {'t_c':>8} {'T_rev':>8} {'revers':>7} {'lat.res':>8}"
          f"  regime")
    for c in matrix.build():
        if c.arm != "ddm_bellman":
            continue
        d = c.derived
        if d["A"] <= 0.0:
            print(f"  {c.delta * 100:>8.4f}% {'--':>8} {'--':>10} {'--':>8} "
                  f"{'--':>6} {'--':>9} {'--':>8} {'--':>8} {'--':>7} {'--':>8}"
                  f"  {d['regime']}")
            continue
        print(f"  {c.delta * 100:>8.4f}% {d['a_over_c']:>8.4f} {d['c_e']:>10.3f} "
              f"{d['pred_z']:>8.5f} {d['pred_zeta']:>6.3f} "
              f"{d['pred_initial_accuracy'] * 100:>8.2f}% {d['pred_t_commit']:>7.2f}s "
              f"{d['pred_t_reversal']:>7.2f}s "
              f"{'yes' if d['pred_reversal_feasible'] else 'NO':>7} "
              f"{'yes' if d['pred_reversal_latency_resolvable'] else 'NO':>8}"
              f"  {d['regime']}")

    counts: dict[str, int] = {}
    for c in matrix.build():
        if c.arm == "ddm_bellman":
            counts[c.derived["regime"]] = counts.get(c.derived["regime"], 0) + 1
    print()
    print("  regime map: " + ",  ".join(
        f"{k} = {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))
    print()

    band = matrix.usable_band()
    print("USABLE BAND — initial accuracy >= 75% AND reversal latency >= 3 ticks")
    if band["lo"] is None:
        print("  EMPTY at this tick rate: no delta satisfies both conditions.")
    else:
        print(f"  delta in [{band['lo'] * 100:.4f}%, {band['hi'] * 100:.4f}%]  "
              f"(ratio {band['ratio']:.2f}), containing "
              f"{band['n_grid_points']} of {len(factors.delta_grid()) - 1} "
              f"non-zero grid points")
    print("  The LOWER edge is set by the noise and the cost of error; it does not")
    print("  move with the tick rate. The UPPER edge is pure sampling resolution:")
    print(f"  {'ticks/s':>9} {'band':>24} {'ratio':>7} {'grid pts':>9}")
    for tps in (1, 2, 5, 10, 20):
        b = matrix.usable_band(ticks_per_second=tps)
        here = "  <- current" if tps == factors.TICKS_PER_SECOND else ""
        if b["lo"] is None:
            print(f"  {tps:>9} {'--- empty ---':>24} {'--':>7} {0:>9}{here}")
        else:
            span = f"[{b['lo'] * 100:.4f}%, {b['hi'] * 100:.4f}%]"
            print(f"  {tps:>9} {span:>24} {b['ratio']:>7.2f} "
                  f"{b['n_grid_points']:>9}{here}")
    print()


def main(argv=None) -> int:
    """Run every check and print the report."""
    ap = argparse.ArgumentParser(description="Flexibility campaign preflight.")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero on warnings as well as failures")
    args = ap.parse_args(argv)

    print_grid_table()

    rep = Report()
    for check in (check_locked_constants, check_templates, check_arms_matched,
                  check_grid, check_time_limit):
        check(rep)

    if rep.failures:
        print("FAILURES — the campaign must not launch:")
        for f in rep.failures:
            print(f"  [FAIL] {f}")
        print()
    if rep.warnings:
        print("WARNINGS — read before launching:")
        for w in rep.warnings:
            print(f"  [WARN] {w}")
        print()
    if not rep.failures and not rep.warnings:
        print("All checks passed.")

    if rep.failures:
        return 1
    return 2 if (args.strict and rep.warnings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
