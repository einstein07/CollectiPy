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
    # Probe the symmetric cell, the bottom and top of the log leg, and a mid point:
    # the seed and the locked block must match at every delta, and these four catch
    # anything that varies with it.
    probes = [grid[0], grid[1], grid[len(grid) // 2], grid[-1]]
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
    """Is every cell one whose reversal rate means something?"""
    conds = [c for c in matrix.build() if c.arm == "ddm_bellman"]
    live = [c for c in conds if not c.derived["is_symmetric"]]

    # THE precondition for the whole campaign. If the agent cannot physically get
    # back before it arrives, a 0% reversal rate is the arena, not the model -- and
    # it would look exactly like the rigidity result the RA arms are meant to show.
    infeasible = [c for c in live if not c.derived["pred_reversal_feasible"]]
    if infeasible:
        worst = max(infeasible, key=lambda c: c.derived["pred_t_reversal"])
        rep.fail(
            f"{len(infeasible)} of {len(live)} non-zero cells cannot reverse within "
            f"the travel budget (worst: {worst.name}, needs "
            f"{worst.derived['pred_t_reversal']:.1f} s against a "
            f"{worst.derived['T_b']:.0f} s budget). A 0% reversal rate there is a "
            "property of the arena, not of the model, and is indistinguishable from "
            "the rigidity result. Fix the velocity or the cost of error."
        )

    # Trials that commit to the WORSE option have nothing to reverse: the swap makes
    # their choice correct. Section 6 analyses them separately, so a low-accuracy cell
    # costs effective replicates rather than being wrong.
    low = [c for c in live if c.derived["pred_initial_accuracy"] < 0.75]
    if low:
        worst = min(low, key=lambda c: c.derived["pred_initial_accuracy"])
        rep.warn(
            f"{len(low)} of {len(live)} cells have a predicted initial accuracy below "
            f"75% (lowest {worst.derived['pred_initial_accuracy'] * 100:.1f}% at "
            f"delta = {worst.delta * 100:.3f}%). In those cells a sizeable minority of "
            "trials commit to the worse option, where the swap makes the choice "
            "correct and there is nothing to reverse; they are analysed separately, so "
            "budget for the reduced effective replicate count rather than discovering "
            "it afterwards."
        )

    # Reversal RATE is observable everywhere; only the LATENCY needs the resolution.
    unresolved = [c for c in live
                  if not c.derived["pred_reversal_latency_resolvable"]]
    if unresolved:
        cutoff = min((c for c in live
                      if c.derived["pred_reversal_latency_resolvable"]),
                     key=lambda c: -c.delta, default=None)
        edge = f"{cutoff.delta * 100:.3f}%" if cutoff else "nowhere on the grid"
        rep.warn(
            f"reversal latency is discretisation-limited above delta = {edge} "
            f"({len(unresolved)} of {len(live)} cells have T_rev < 3 ticks at "
            f"ticks_per_second = {factors.TICKS_PER_SECOND}). This is EXPECTED and "
            "accepted: reversal RATE is the headline measurement and is fully "
            "resolved at every cell. Report latency only below that edge."
        )

    # The Section 2 confound, surfaced with a number rather than left in prose.
    lo, hi = factors.mean_drive(0.0), factors.mean_drive(factors.DIFF_MAX)
    rep.warn(
        f"pinned strengths mean the MEAN drive falls {lo:.2f} -> {hi:.2f} across the "
        "sweep. The DDM is blind to this; the ring attractor is not, so at the top of "
        "the grid the RA arms are driven differently, not merely discriminating a "
        "larger difference. Chosen for continuity with the earlier RA sweeps; report "
        "delta and mean drive together."
    )


def check_time_limit(rep: Report) -> None:
    """The time limit has to cover the worst INFORMATIVE cell, not the worst cell."""
    t_b = matrix.geometry()["T_b"]
    worst = None
    for c in matrix.build():
        if c.arm != "ddm_bellman" or c.delta <= 0.0:
            continue
        if not c.derived["pred_reversal_feasible"]:
            continue
        # commit + swap delay + reverse + traverse to the OTHER target
        d = c.derived
        need = (d["pred_t_commit"]
                + factors.SWAP_DELAY_TICKS / factors.TICKS_PER_SECOND
                + d["pred_t_reversal"]
                + t_b)
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
    """Report what each cell will actually do, before 660 tasks are queued."""
    g = matrix.geometry()

    print("=" * 96)
    print("FLEXIBILITY CAMPAIGN — PREFLIGHT")
    print("=" * 96)
    print(f"  arms              : {', '.join(factors.ARMS)}")
    print(f"  RA kernel / gains : v = 0.5, u = "
          f"{', '.join(str(factors.ARM_U[a]) for a in factors.RA_ARMS)}")
    print(f"  cost of error     : c_e = {factors.COST_OF_ERROR} s")
    print(f"  c = sqrt(2)*eta   : {factors.NOISE_C:.6f}  "
          f"(white_rate = {factors.WHITE_RATE})")
    print(f"  velocity          : {factors.LINEAR_VELOCITY} m/s   "
          f"T_b = R/v = {g['T_b']:.1f} s")
    print(f"  Bellman horizon   : T_max = r0/v = {g['T_max']:.4f} s   "
          f"N_t = {g['N_t']} (solver dt <= {factors.BELLMAN_DT})")
    print(f"  tick rate         : {factors.TICKS_PER_SECOND} /s (arena and agent)   "
          f"swap delay = {factors.SWAP_DELAY_TICKS} tick")
    print(f"  time limit        : {factors.TIME_LIMIT} s")
    print(f"  strengths         : static_0 = {factors.QUALITY_BETTER} PINNED, "
          f"static_1 = {factors.QUALITY_BETTER}*(1 - delta)")
    print(f"  grid / replicates : {len(factors.delta_grid())} deltas x "
          f"{factors.REPS} reps x {len(factors.ARMS)} arms")
    print(f"  runs / tasks      : {matrix.total_runs()} / {matrix.total_tasks()}")
    print()

    print("PREDICTED DDM BEHAVIOUR — from the model's own boundary solver")
    print(f"  {'delta':>9} {'mean drive':>11} {'A':>8} {'A/c':>7} {'z':>8} "
          f"{'init acc':>9} {'t_c':>8} {'T_rev':>8} {'can rev':>8} {'lat res':>8}")
    for c in matrix.build():
        if c.arm != "ddm_bellman":
            continue
        d = c.derived
        if d["is_symmetric"]:
            print(f"  {c.delta * 100:>8.4f}% {d['mean_drive']:>11.3f} {'--':>8} "
                  f"{'--':>7} {'--':>8} {'50.00%':>9} {'--':>8} {'--':>8} "
                  f"{'n/a':>8} {'n/a':>8}   symmetric control")
            continue
        print(f"  {c.delta * 100:>8.4f}% {d['mean_drive']:>11.3f} {d['A']:>8.4f} "
              f"{d['a_over_c']:>7.3f} {d['pred_z']:>8.5f} "
              f"{d['pred_initial_accuracy'] * 100:>8.2f}% "
              f"{d['pred_t_commit']:>7.2f}s {d['pred_t_reversal']:>7.2f}s "
              f"{'yes' if d['pred_reversal_feasible'] else 'NO':>8} "
              f"{'yes' if d['pred_reversal_latency_resolvable'] else 'no':>8}")
    print()
    live = [c for c in matrix.build()
            if c.arm == "ddm_bellman" and not c.derived["is_symmetric"]]
    n_feas = sum(1 for c in live if c.derived["pred_reversal_feasible"])
    n_lat = sum(1 for c in live if c.derived["pred_reversal_latency_resolvable"])
    n_acc = sum(1 for c in live if c.derived["pred_initial_accuracy"] >= 0.75)
    print(f"  reversal physically possible : {n_feas}/{len(live)} cells")
    print(f"  first choice >= 75% accurate : {n_acc}/{len(live)} cells")
    print(f"  reversal latency resolvable  : {n_lat}/{len(live)} cells "
          f"(rate is resolved at all {len(live)})")
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
