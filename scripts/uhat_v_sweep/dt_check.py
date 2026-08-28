#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Numerical guardrail: the step-halving check (Section 11).

Runs the three stiffest cells -- (v=0.1, u_hat=1.50), (v=0.1, u_hat=1.25),
(v=0.2, u_hat=1.50) -- at `integration_dt = 0.05` and at 0.1, on the IDENTICAL
seed list, and asks two questions per cell:

  1. were there any numerical failures at either step size?
  2. is the accuracy difference consistent with zero?

Question 2 is answered by a PAIRED bootstrap: the two step sizes share seeds, so
the difference is resampled trial-by-trial over the shared seeds. The cell passes
when zero lies inside the 95 % interval of that difference.

A failing cell is marked `excluded` in `manifest.json`. The runner then skips it
and `aggregate.py` reports the design as unbalanced -- which the analysis must
say out loud, because the shares are then conditional on the cells that ran.

    python3 scripts/uhat_v_sweep/dt_check.py --results-root results/uhat_v_sweep
    python3 scripts/uhat_v_sweep/dt_check.py --dry-run     # never edits the manifest

Run this BEFORE the full submission. It costs 6 x 50 trials, a few minutes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in (str(_HERE), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import factors            # noqa: E402
import config_patch       # noqa: E402
import run_cell           # noqa: E402


def paired_bootstrap(a: pd.Series, b: pd.Series, n_boot: int = 20000,
                     seed: int = 12345) -> tuple[float, float, float]:
    """(mean difference b - a, lo, hi) from a paired bootstrap over shared seeds."""
    shared = a.index.intersection(b.index)
    av = a.loc[shared].to_numpy(dtype=float)
    bv = b.loc[shared].to_numpy(dtype=float)
    diff = bv - av
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    draws = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(draws, 2.5)), \
        float(np.percentile(draws, 97.5))


def run_arm(cell: dict, trials: range, root: Path, dt: float, label: str,
            template: dict, force: bool) -> pd.DataFrame:
    out_root = root / "dt_check" / label
    run_cell.run_task(cell, trials, out_root, chunk=0, fmt="parquet",
                      force=force, dt_override=dt, template=template, quiet=True)
    path = run_cell._existing_output(out_root, cell["cell_id"], 0)
    frame = (pd.read_parquet(path) if path.suffix == ".parquet"
             else pd.read_csv(path))
    for col in ("decided", "correct", "numerical_failure"):
        frame[col] = frame[col].fillna(False).astype(bool)
    return frame


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results-root", type=Path,
                    default=_ROOT / "results" / "uhat_v_sweep")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--trials", type=int, default=factors.STEP_HALVING_TRIALS)
    ap.add_argument("--dry-run", action="store_true",
                    help="report only; never edit the manifest")
    ap.add_argument("--force", action="store_true", help="re-run completed arms")
    args = ap.parse_args(argv)

    root = args.results_root.resolve()
    manifest_path = args.manifest or (root / "manifest.json")
    manifest = run_cell.load_manifest(manifest_path)
    template = config_patch.load_template()
    trials = range(args.trials)

    print("Step-halving check (Section 11)")
    print(f"  cells    : {factors.STEP_HALVING_CELLS}")
    print(f"  dt       : {factors.INTEGRATION_DT} (baseline) vs "
          f"{factors.STEP_HALVING_DT} (halved)")
    print(f"  trials   : {args.trials} per cell per step size, identical seeds")
    print()

    results, failed, marginals = [], [], []
    for v, u_hat in factors.STEP_HALVING_CELLS:
        cell = run_cell.find_cell(manifest, factors.cell_id(v, u_hat))
        base = run_arm(cell, trials, root, factors.INTEGRATION_DT, "dt_0.1",
                       template, args.force)
        half = run_arm(cell, trials, root, factors.STEP_HALVING_DT, "dt_0.05",
                       template, args.force)
        n_fail = int(base["numerical_failure"].sum() + half["numerical_failure"].sum())
        mean, lo, hi = paired_bootstrap(
            base.set_index("seed")["correct"], half.set_index("seed")["correct"])
        contains_zero = lo <= 0.0 <= hi
        ok = (n_fail == 0) and contains_zero
        # "The interval contains zero" is a weak statement when zero sits on an
        # edge of it: at 50 trials the interval is wide and the test is
        # underpowered, so a pass can be a pass by a hair. Flag that rather than
        # report it as clean.
        width = hi - lo
        margin = ((min(0.0 - lo, hi - 0.0) / width)
                  if (contains_zero and width > 0) else 0.0)
        marginal = ok and margin < 0.10
        row = {
            "cell_id": int(cell["cell_id"]), "v": v, "u_hat": u_hat,
            "u": float(cell["u"]),
            "acc_dt_0.1": float(base["correct"].mean()),
            "acc_dt_0.05": float(half["correct"].mean()),
            "delta_acc": mean, "boot_lo": lo, "boot_hi": hi,
            "contains_zero": contains_zero,
            "zero_margin_frac": margin,
            "marginal": marginal,
            "numerical_failures": n_fail,
            "max_abs_state": float(pd.concat(
                [base["max_abs_state"], half["max_abs_state"]]).max()),
            "decided_frac_dt_0.1": float(base["decided"].mean()),
            "decided_frac_dt_0.05": float(half["decided"].mean()),
            "pass": ok,
        }
        results.append(row)
        if not ok:
            failed.append(cell["cell_id"])
        elif marginal:
            marginals.append(cell["cell_id"])
        print(f"  cell {cell['cell_id']:>2}  v={v:<4} u_hat={u_hat:<5} u={cell['u']:8.4f}  "
              f"acc {row['acc_dt_0.1']:.4f} -> {row['acc_dt_0.05']:.4f}  "
              f"delta {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
              f"failures {n_fail}  max|z| {row['max_abs_state']:.4g}  "
              f"{'PASS' if ok else 'FAIL'}"
              f"{' (MARGINAL)' if marginal else ''}")

    report_path = root / "dt_check" / "step_halving_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump({"trials": args.trials,
                   "dt_baseline": factors.INTEGRATION_DT,
                   "dt_halved": factors.STEP_HALVING_DT,
                   "results": results}, fh, indent=2)
    print(f"\n  wrote {report_path}")

    if not failed:
        print(f"\n  All {len(results)} cells pass: zero numerical failures and "
              "accuracy differences consistent with zero at "
              f"dt = {factors.INTEGRATION_DT}.")
        if marginals:
            print(f"\n  BUT cells {marginals} pass MARGINALLY — zero sits within "
                  "10 % of an edge of the bootstrap interval. At "
                  f"{args.trials} trials the interval is wide and the test is "
                  "underpowered, so this is 'not refuted', not 'shown equal'. "
                  "These are the stiffest cells in the grid; before leaning on "
                  "them, re-run with more trials:")
            print(f"    dt_check.py --results-root {root} --trials 200 --force")
        return 0

    print(f"\n  {len(failed)} cell(s) FAILED: {failed}")
    if args.dry_run:
        print("  --dry-run: manifest NOT modified. Re-run without --dry-run to "
              "mark them excluded.")
        return 1

    for cell in manifest["cells"]:
        if int(cell["cell_id"]) in failed:
            cell["excluded"] = True
            cell["excluded_reason"] = (
                f"step-halving check failed at dt {factors.INTEGRATION_DT} vs "
                f"{factors.STEP_HALVING_DT} (Section 11)")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"  marked excluded in {manifest_path}. The design is now UNBALANCED — "
          "aggregate.py and analyze_collapse.py will both say so, and the "
          "deviance shares become conditional on the cells that ran.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
