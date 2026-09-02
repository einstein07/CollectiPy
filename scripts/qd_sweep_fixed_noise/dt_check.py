#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2026 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""§3 numerical guardrail (BLOCKING before submission): u = 35 is the
stiffest configuration run to date. Step-halving (integration_dt 0.05 vs the
campaign's 0.1) on (u = 35, v ∈ {0.1, 0.5, 1.0}) at actual δ_Q = 100 bp,
identical frontier-v1 seeds.

    python3 scripts/qd_sweep_fixed_noise/dt_check.py \
        [--trials 50] [--cells v0.1,v0.5,v1] [--out <report.json>]

Passes per cell when (a) zero numerical failures and no non-finite / blown-up
ring state in either arm, and (b) the paired-bootstrap 95 % CI of the
accuracy difference contains zero. The spec sets 50 trials; note the frontier
campaign's experience that 50/arm cannot resolve a ~10-point step effect —
raise `--trials 200` for a sharper test if wall time allows. A failing v
excludes the affected high-u cells with a report, exactly as in prior
campaigns — never submit as-is.
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
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import qd                               # noqa: E402
from run_batch import InProcessRunner   # noqa: E402

DT_CAMPAIGN = 0.1
DT_HALVED = 0.05
CHECK_U = 35.0
CHECK_V = [0.1, 0.5, 1.0]
CHECK_ACTUAL = 100
MAX_ABS_STATE = 1.0e3
N_BOOT = 5000


def run_arm(runner, v: float, dt: float, trials: int, scratch: Path):
    template = qd.load_template("ra")
    out, peak = [], 0.0
    for run_id in range(1, trials + 1):
        cfg = qd.patch_ra(template, CHECK_U, v, CHECK_ACTUAL)
        cfg["environment"]["agents"]["movable_0"]["mean_field_model"][
            "integration_dt"] = float(dt)
        qd.apply_seeds(cfg, "ra", CHECK_ACTUAL, run_id)
        rep = scratch / f"dt{dt:g}" / f"v{v:g}" / f"replicate_{run_id}"
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
            _t, _f, hit = qd.first_crossing(
                pos, qd.target_positions(cfg),
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
            out.append((run_id, hit == qd.CORRECT_TARGET_ID, bad))
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
    ap.add_argument("--trials", type=int, default=50,
                    help="trials per arm (spec §3: 50; use 200 for power)")
    ap.add_argument("--cells", default=None,
                    help=f"comma-separated v values (default {CHECK_V})")
    ap.add_argument("--out", type=Path,
                    default=qd.DEFAULT_OUT / "dt_check_report.json")
    args = ap.parse_args(argv)

    vs = ([float(x.lstrip('v')) for x in args.cells.split(",")]
          if args.cells else CHECK_V)
    qd.write_templates()
    runner = InProcessRunner()
    import tempfile
    report, ok_all = [], True
    print(f"step-halving check (§3): u = {CHECK_U:g}, actual δ_Q = "
          f"{CHECK_ACTUAL} bp, dt {DT_CAMPAIGN} vs {DT_HALVED}, "
          f"{args.trials} trials/arm, identical frontier-v1 seeds")
    with tempfile.TemporaryDirectory(prefix="qd_dtcheck_") as scratch:
        scratch = Path(scratch)
        for v in vs:
            t0 = time.time()
            arm01, peak01 = run_arm(runner, v, DT_CAMPAIGN, args.trials,
                                    scratch)
            arm005, peak005 = run_arm(runner, v, DT_HALVED, args.trials,
                                      scratch)
            acc01 = [c for _r, c, _b in arm01]
            acc005 = [c for _r, c, _b in arm005]
            fails = sum(b for _r, _c, b in arm01 + arm005)
            diff, lo, hi = paired_boot_ci(acc01, acc005)
            passed = fails == 0 and lo <= 0.0 <= hi
            ok_all &= passed
            report.append({
                "v": v, "u": CHECK_U, "actual_bp": CHECK_ACTUAL,
                "trials_per_arm": args.trials,
                "acc_dt_0.1": sum(acc01) / len(acc01),
                "acc_dt_0.05": sum(acc005) / len(acc005),
                "diff_mean": diff, "diff_ci95": [lo, hi],
                "numerical_failures": int(fails),
                "max_abs_state": max(peak01, peak005),
                "passed": passed})
            r = report[-1]
            print(f"  v = {v:g}: acc {r['acc_dt_0.1']:.3f} vs "
                  f"{r['acc_dt_0.05']:.3f}  Δ={diff:+.3f} [{lo:+.3f}, "
                  f"{hi:+.3f}]  failures={fails}  "
                  f"max|z|={r['max_abs_state']:.3g}  "
                  f"{'PASS' if passed else 'FAIL'}  ({time.time() - t0:.0f}s)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"check": "PASS" if ok_all else "FAIL",
                   "dt": [DT_CAMPAIGN, DT_HALVED], "u": CHECK_U,
                   "actual_bp": CHECK_ACTUAL, "cells": report}, fh, indent=2)
    print(f"wrote {args.out} — overall {'PASS' if ok_all else 'FAIL'}")
    if not ok_all:
        print("A failing v means the Euler step does not resolve the ring "
              "dynamics at u = 35 there: EXCLUDE the affected high-u cells "
              "(report which) — do not submit as-is (§3).")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
