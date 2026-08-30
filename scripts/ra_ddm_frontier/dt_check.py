#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2026 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""§2 (Set U-v2): step-halving check at the top-up's stiffest cells — BLOCKING
before the wave-2 submission.

    python3 scripts/ra_ddm_frontier/dt_check.py \
        [--manifest <manifest_topup.csv>] [--cells id,id,...] [--trials 200]

Default cells: the three highest-u rows of the top-up manifest (u ≈ 31.5 at
v = 0.2 is stiffness territory). Each cell runs `--trials` replicates at
`integration_dt` 0.1 (the campaign value) and 0.05, on IDENTICAL frontier-v1
seeds, and passes when (a) zero numerical failures and no non-finite ring
state in either arm, and (b) the paired-bootstrap 95 % CI of the accuracy
difference contains zero. The factorial's check taught that 50 trials/arm
cannot resolve a 10-point step effect — hence 200 by default.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import zipfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in (str(_HERE), str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import frontier                      # noqa: E402
from run_batch import InProcessRunner   # noqa: E402

DT_CAMPAIGN = 0.1
DT_HALVED = 0.05
MAX_ABS_STATE = 1.0e3
N_BOOT = 5000


def run_arm(runner, row: dict, dt: float, trials: int, scratch: Path):
    """One (cell, dt) arm: [(run_id, correct, failed)], max |z| over the arm."""
    template = frontier._load_json(frontier.RA_TEMPLATE)
    out, peak = [], 0.0
    for run_id in range(1, trials + 1):
        cfg = frontier.patch_ra(template, float(row["u"]), float(row["v"]))
        cfg["environment"]["agents"]["movable_0"]["mean_field_model"][
            "integration_dt"] = float(dt)
        frontier.apply_seeds(cfg, "ra", run_id)
        rep = scratch / f"dt{dt:g}" / row["cell_id"] / f"replicate_{run_id}"
        rep.mkdir(parents=True, exist_ok=True)
        cfg["environment"]["results"]["base_path"] = str(rep)
        cfg_path = rep / "config.json"
        with open(cfg_path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh)
        try:
            runner.run(cfg_path)
            zf = zipfile.ZipFile(next(rep.glob("config_folder_*/run_*.zip")))
            names = zf.namelist()
            pos = list(csv.DictReader(io.TextIOWrapper(zf.open(
                next(n for n in names if n.endswith("_position.csv"))))))
            _t, _f, hit = frontier.first_crossing(
                pos, frontier.target_positions(cfg),
                float(cfg["environment"]["termination"]["radius"]))
            bad = False
            neural = next((n for n in names if n.endswith("_neural.csv")), None)
            if neural:
                rdr = csv.reader(io.TextIOWrapper(zf.open(neural)))
                header = next(rdr, [])
                cols = [i for i, h in enumerate(header)
                        if h.startswith("neuron_")]
                for r in rdr:
                    for i in cols:
                        try:
                            val = abs(float(r[i]))
                        except (ValueError, IndexError):
                            bad = True
                            continue
                        if val != val or val > MAX_ABS_STATE:
                            bad = True
                        peak = max(peak, val if val == val else peak)
            out.append((run_id, hit == frontier.CORRECT_TARGET_ID, bad))
        except Exception:                    # noqa: BLE001 — a failure is data
            out.append((run_id, False, True))
        finally:
            import shutil
            shutil.rmtree(rep, ignore_errors=True)
    return out, peak


def paired_boot_ci(a01, a005, n_boot=N_BOOT, seed=0):
    import numpy as np
    d = np.array(a005, float) - np.array(a01, float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), \
        float(np.percentile(means, 97.5))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path,
                    default=_ROOT / "results" / "ra_ddm_frontier"
                    / "manifest_topup.csv")
    ap.add_argument("--cells", default=None,
                    help="comma-separated cell_ids (default: 3 highest u)")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    rows = frontier.read_manifest(args.manifest)
    if args.cells:
        wanted = args.cells.split(",")
        picked = [r for r in rows if r["cell_id"] in wanted]
        missing = set(wanted) - {r["cell_id"] for r in picked}
        if missing:
            raise SystemExit(f"not in {args.manifest}: {sorted(missing)}")
    else:
        picked = sorted(rows, key=lambda r: float(r["u"]))[-3:]

    if not frontier.RA_TEMPLATE.is_file():
        frontier.write_templates()
    runner = InProcessRunner()
    import tempfile
    report, ok_all = [], True
    print(f"step-halving check: dt {DT_CAMPAIGN} vs {DT_HALVED}, "
          f"{args.trials} trials/arm, identical frontier-v1 seeds")
    with tempfile.TemporaryDirectory(prefix="frontier_dtcheck_") as scratch:
        scratch = Path(scratch)
        for row in picked:
            t0 = time.time()
            arm01, peak01 = run_arm(runner, row, DT_CAMPAIGN, args.trials, scratch)
            arm005, peak005 = run_arm(runner, row, DT_HALVED, args.trials, scratch)
            acc01 = [c for _r, c, _b in arm01]
            acc005 = [c for _r, c, _b in arm005]
            fails = sum(b for _r, _c, b in arm01 + arm005)
            diff, lo, hi = paired_boot_ci(acc01, acc005)
            passed = fails == 0 and lo <= 0.0 <= hi
            ok_all &= passed
            report.append({
                "cell_id": row["cell_id"], "v": float(row["v"]),
                "u": float(row["u"]), "u_hat": float(row["u_hat"]),
                "trials_per_arm": args.trials,
                "acc_dt_0.1": sum(acc01) / len(acc01),
                "acc_dt_0.05": sum(acc005) / len(acc005),
                "diff_mean": diff, "diff_ci95": [lo, hi],
                "numerical_failures": int(fails),
                "max_abs_state": max(peak01, peak005),
                "passed": passed})
            r = report[-1]
            print(f"  {r['cell_id']:>16} (u={r['u']:g}): acc {r['acc_dt_0.1']:.3f} "
                  f"vs {r['acc_dt_0.05']:.3f}  Δ={diff:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
                  f"failures={fails}  max|z|={r['max_abs_state']:.3g}  "
                  f"{'PASS' if passed else 'FAIL'}  "
                  f"({time.time() - t0:.0f}s)")

    dest = args.out or (args.manifest.parent / "dt_check_report.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump({"check": "PASS" if ok_all else "FAIL",
                   "dt": [DT_CAMPAIGN, DT_HALVED], "cells": report}, fh, indent=2)
    print(f"wrote {dest} — overall {'PASS' if ok_all else 'FAIL'}")
    if not ok_all:
        print("A failing cell means the Euler step does not resolve the "
              "dynamics at that u; mark the cell excluded (edit the top-up "
              "manifest) or lower integration_dt for the whole wave — do not "
              "submit as-is.")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
