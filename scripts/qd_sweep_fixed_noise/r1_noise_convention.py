#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2026 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""R-1 (§2, BLOCKING): verify the `white_rate → c` mapping empirically.

    python3 scripts/qd_sweep_fixed_noise/r1_noise_convention.py \
        [--runs 200] [--tolerance 0.05] [--out <r1_report.json>]

Every Bellman solve and every analytic check in this campaign assumes the
evidence-channel noise scale is c = 0.1 at the pinned
WHITE_RATE = 0.07071068, for EVERY δ_Q. This script measures it from the
simulator's own logs — never from the config it was told to use:

  1. runs `--runs` probe replicates per actual δ_Q ∈ {50, 100, 200} bp (a
     static bound far outside reach, so the accumulator runs uninterrupted),
     through the real patcher + seed path;
  2. measures the accumulator's per-unit-time increment variance from
     `_ddm.csv` (pre-commit ticks only, travel and halt windows separately)
     and asserts  ĉ = √(Var(Δx)/dt) = 0.1  within --tolerance at every
     actual δ_Q and pooled;
  3. measures the drift  mean(Δx)/dt  and asserts it tracks A(δ) = S0·δ_Q —
     the direct proof that SNR now moves with δ_Q while the noise stands
     still (the halted campaign's coupling made this exactly constant);
  4. checks the percept channel (`_percept.csv`): per-target white_rate as
     logged, the evidence-difference variance, and the logged `c` and
     `A_hat` columns of `_ddm.csv` (belief = design values);
  5. §6 pairing audit: percept streams bitwise-identical across models
     ('ra' / 'ddm-bellman' / 'ddm-static') at equal (actual δ_Q, run_id),
     and different across actual δ_Q.

If the sim's convention differs: HALT AND REPORT — do not rescale silently.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import shutil
import sys
import time
import zipfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import qd                             # noqa: E402
from run_batch import InProcessRunner   # noqa: E402

PROBE_BOUND = 6.0        # static b far outside reach pre-window (max E[x] ≈ 3)
TRAVEL_TICKS = (1, 8)    # increment pairs within the locomotion phase
HALT_TICKS = (12, 30)    # within the halt (window ends before any crossing
#                          risk at actual = 200 bp: E[x(30)] = 3.0 vs b = 6)


def _read_zip(rep: Path) -> tuple[list[dict], list[dict]]:
    z = next(rep.glob("config_folder_*/run_*.zip"))
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
        ddm = next((n for n in names if n.endswith("_ddm.csv")), None)
        per = next((n for n in names if n.endswith("_percept.csv")), None)
        drows = (list(csv.DictReader(io.TextIOWrapper(zf.open(ddm))))
                 if ddm else [])
        prows = (list(csv.DictReader(io.TextIOWrapper(zf.open(per))))
                 if per else [])
    return drows, prows


def _run_probe(runner, cfg_base: dict, model: str, actual: int, run_id: int,
               scratch: Path, tag: str) -> Path:
    import copy
    cfg = copy.deepcopy(cfg_base)
    qd.apply_seeds(cfg, model, actual, run_id)
    rep = scratch / tag / f"replicate_{run_id}"
    rep.mkdir(parents=True, exist_ok=True)
    cfg["environment"]["results"]["base_path"] = str(rep)
    cfg_path = rep / "config.json"
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    runner.run(cfg_path)
    return rep


def _percept_series(prows: list[dict]) -> dict[tuple[int, str], float]:
    out = {}
    for r in prows:
        if r.get("target"):
            out[(int(r["tick"]), r["target"])] = float(r["q_hat"])
    return out


def measure_actual(runner, actual: int, runs: int, scratch: Path) -> dict:
    """Increment + percept statistics for one actual δ_Q."""
    import numpy as np
    cfg_base = qd.patch_ddm(qd.load_template("ddm"), "static-cost",
                            qd.DESIGN_100, actual, PROBE_BOUND, None)
    inc_travel, inc_halt = [], []
    dev_by_target = {"static_0.s#0": [], "static_1.s#0": []}
    diff_percept = []
    logged_c, logged_A_hat, logged_wr = set(), set(), set()
    t0 = time.time()
    for run_id in range(1, runs + 1):
        rep = _run_probe(runner, cfg_base, "ddm-static", actual, run_id,
                         scratch, f"r1_a{actual}")
        drows, prows = _read_zip(rep)
        x = {int(r["tick"]): float(r["x"]) for r in drows
             if r.get("x") not in (None, "")}
        committed = {int(r["tick"]): (r.get("committed") or "")
                     for r in drows}
        for r in drows[1:3]:
            if r.get("c"):
                logged_c.add(float(r["c"]))
            if r.get("A_hat"):
                logged_A_hat.add(float(r["A_hat"]))
        for lo, hi, bucket in (TRAVEL_TICKS + (inc_travel,),
                               HALT_TICKS + (inc_halt,)):
            for t in range(lo, hi):
                if (t in x and t + 1 in x
                        and not committed.get(t) and not committed.get(t + 1)):
                    bucket.append(x[t + 1] - x[t])
        strengths = {"static_0.s#0": qd.S0,
                     "static_1.s#0": qd.strength_worse(actual)}
        per = _percept_series(prows)
        for r in prows[:3]:
            if r.get("white_rate"):
                logged_wr.add(float(r["white_rate"]))
        ticks = sorted({t for t, _tid in per})
        for t in ticks:
            q0 = per.get((t, "static_0.s#0"))
            q1 = per.get((t, "static_1.s#0"))
            if q0 is None or q1 is None:
                continue
            dev_by_target["static_0.s#0"].append(q0 - strengths["static_0.s#0"])
            dev_by_target["static_1.s#0"].append(q1 - strengths["static_1.s#0"])
            diff_percept.append(q0 - q1)
        shutil.rmtree(rep, ignore_errors=True)

    dt = 1.0
    inc = np.array(inc_travel + inc_halt, float)
    n = inc.size

    def c_hat(a):
        a = np.asarray(a, float)
        return float(math.sqrt(a.var(ddof=1) / dt)) if a.size > 3 else float("nan")

    # SE of a variance estimate ≈ var·√(2/(n−1)); halves for the SD.
    c_pooled = c_hat(inc)
    c_se = c_pooled * math.sqrt(0.5 / max(n - 1, 1))
    drift = float(inc.mean() / dt)
    drift_se = float(inc.std(ddof=1) / math.sqrt(n) / dt)
    eta = {tid: float(math.sqrt(np.var(v, ddof=1) * dt))
           for tid, v in dev_by_target.items()}
    return {
        "actual_bp": actual, "runs": runs, "n_increments": int(n),
        "c_hat": c_pooled, "c_hat_se": c_se,
        "c_hat_travel": c_hat(inc_travel), "c_hat_halt": c_hat(inc_halt),
        "drift_hat": drift, "drift_se": drift_se,
        "A_expected_actual": qd.drift_A(actual),
        "c_percept_diff": float(math.sqrt(np.var(diff_percept, ddof=1) * dt)),
        "white_rate_hat_per_target": eta,
        "logged_c_column": sorted(logged_c),
        "logged_A_hat_column": sorted(logged_A_hat),
        "logged_white_rate": sorted(logged_wr),
        "wall_s": round(time.time() - t0, 1),
    }


def pairing_audit(runner, scratch: Path, n_runs: int = 2) -> dict:
    """§6: identical percept realizations across models at equal (actual,
    run_id); different realizations across actual δ_Q. Exact float equality —
    the stream is a pure function of (seed, target, tick)."""
    ra_cfg = qd.patch_ra(qd.load_template("ra"), 6.0, 0.5, 100)
    bell_cfg = qd.patch_ddm(qd.load_template("ddm"), "bellman", 100, 100,
                            20.0, None)
    stat_cfg = qd.patch_ddm(qd.load_template("ddm"), "static-cost", 100, 100,
                            PROBE_BOUND, None)
    report = {"cross_model_identical": True, "cross_actual_different": True,
              "details": []}
    for run_id in range(1, n_runs + 1):
        series = {}
        for name, cfg, model in (("ra", ra_cfg, "ra"),
                                 ("ddm-bellman", bell_cfg, "ddm-bellman"),
                                 ("ddm-static", stat_cfg, "ddm-static")):
            rep = _run_probe(runner, cfg, model, 100, run_id, scratch,
                             f"audit_{name}")
            _d, prows = _read_zip(rep)
            series[name] = _percept_series(prows)
            shutil.rmtree(rep, ignore_errors=True)
        common = set(series["ra"]) & set(series["ddm-bellman"]) \
            & set(series["ddm-static"])
        same = all(series["ra"][k] == series["ddm-bellman"][k]
                   == series["ddm-static"][k] for k in common)
        report["cross_model_identical"] &= same and len(common) > 10
        report["details"].append({"run_id": run_id, "common_keys": len(common),
                                  "identical": same})
        # different actual -> different env stream (§6)
        alt = qd.patch_ddm(qd.load_template("ddm"), "static-cost", 100, 50,
                           PROBE_BOUND, None)
        rep = _run_probe(runner, alt, "ddm-static", 50, run_id, scratch,
                         "audit_a50")
        _d, prows = _read_zip(rep)
        alt_series = _percept_series(prows)
        shutil.rmtree(rep, ignore_errors=True)
        overlap = set(alt_series) & set(series["ddm-static"])
        # compare NOISE realizations (mean removed), not raw q_hat: the mean
        # differs by construction (strengths), the draw must differ by seed
        s100 = {"static_0.s#0": qd.S0, "static_1.s#0": qd.strength_worse(100)}
        s50 = {"static_0.s#0": qd.S0, "static_1.s#0": qd.strength_worse(50)}
        noise_equal = sum(
            1 for (t, tid) in overlap
            if abs((alt_series[(t, tid)] - s50[tid])
                   - (series["ddm-static"][(t, tid)] - s100[tid])) < 1e-12)
        report["cross_actual_different"] &= noise_equal < len(overlap) * 0.01
        report["details"][-1]["noise_draws_equal_across_actual"] = \
            f"{noise_equal}/{len(overlap)}"
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--tolerance", type=float, default=0.05,
                    help="max |c_hat/0.1 - 1| per actual δ_Q (pooled gate at "
                         "half this)")
    ap.add_argument("--out", type=Path,
                    default=qd.DEFAULT_OUT / "r1_report.json")
    args = ap.parse_args(argv)

    import numpy as np
    import tempfile
    qd.write_templates()
    qd.assert_noise_invariant_across_actuals("ra")
    qd.assert_noise_invariant_across_actuals("ddm")
    runner = InProcessRunner()
    results, failures = [], []
    print(f"R-1: measuring the accumulator's per-unit-time increment "
          f"variance at WHITE_RATE = {qd.WHITE_RATE} "
          f"({args.runs} probe runs per actual δ_Q; asserting c = "
          f"{qd.NOISE_SCALE_C} within ±{args.tolerance:.0%})")
    with tempfile.TemporaryDirectory(prefix="qd_r1_") as scratch:
        scratch = Path(scratch)
        for actual in qd.DIFF_BP:
            r = measure_actual(runner, actual, args.runs, scratch)
            results.append(r)
            rel = abs(r["c_hat"] / qd.NOISE_SCALE_C - 1.0)
            ok_c = rel <= args.tolerance
            drift_z = (abs(r["drift_hat"] - r["A_expected_actual"])
                       / max(r["drift_se"], 1e-12))
            ok_drift = drift_z <= 4.0
            if not ok_c:
                failures.append(f"actual {actual} bp: c_hat = {r['c_hat']:.4f}"
                                f" (rel err {rel:.1%} > {args.tolerance:.0%})")
            if not ok_drift:
                failures.append(f"actual {actual} bp: drift {r['drift_hat']:.4f}"
                                f" != A = {r['A_expected_actual']} "
                                f"(z = {drift_z:.1f})")
            # The model's believed c is sqrt(2)·white_rate computed from the
            # 8-decimal pinned rate, i.e. 0.10000000266… — assert THAT value
            # exactly (drift catcher), and closeness to the design 0.1.
            c_exact = math.sqrt(2.0) * qd.WHITE_RATE
            if r["logged_c_column"] and any(
                    abs(c - c_exact) > 1e-12 or abs(c - qd.NOISE_SCALE_C) > 1e-6
                    for c in r["logged_c_column"]):
                failures.append(f"actual {actual} bp: _ddm.csv c column = "
                                f"{r['logged_c_column']} != sqrt(2)·"
                                f"white_rate = {c_exact!r}")
            if r["logged_white_rate"] != [qd.WHITE_RATE]:
                failures.append(f"actual {actual} bp: logged white_rate "
                                f"{r['logged_white_rate']} != {qd.WHITE_RATE}")
            print(f"  actual {actual:>3d} bp: c_hat = {r['c_hat']:.4f} "
                  f"± {r['c_hat_se']:.4f} (travel {r['c_hat_travel']:.4f} / "
                  f"halt {r['c_hat_halt']:.4f}; n = {r['n_increments']})  "
                  f"drift = {r['drift_hat']:+.4f} ± {r['drift_se']:.4f} "
                  f"(A = {r['A_expected_actual']})  "
                  f"[{'ok' if ok_c and ok_drift else 'FAIL'}] "
                  f"({r['wall_s']}s)")

        pooled = np.array([r["c_hat"] for r in results])
        weights = np.array([r["n_increments"] for r in results], float)
        c_all = float(np.sqrt(np.average(pooled ** 2, weights=weights)))
        if abs(c_all / qd.NOISE_SCALE_C - 1.0) > args.tolerance / 2.0:
            failures.append(f"pooled c_hat = {c_all:.4f} "
                            f"(rel err > {args.tolerance / 2:.1%})")
        drifts = [r["drift_hat"] for r in results]
        if not (drifts[0] < drifts[1] < drifts[2]):
            failures.append(f"drift does not increase with δ_Q: {drifts} — "
                            "SNR is not moving; the coupling may be back")
        print(f"  pooled c_hat = {c_all:.4f} (target {qd.NOISE_SCALE_C}); "
              f"drift ladder {drifts[0]:.4f} / {drifts[1]:.4f} / "
              f"{drifts[2]:.4f} (expected 0.025 / 0.05 / 0.1)")

        print("\n§6 pairing audit (percept streams across models and "
              "across actual δ_Q):")
        audit = pairing_audit(runner, scratch)
        if not audit["cross_model_identical"]:
            failures.append("pairing audit: percept streams NOT identical "
                            "across models at equal (actual, run_id)")
        if not audit["cross_actual_different"]:
            failures.append("pairing audit: noise realizations do NOT differ "
                            "across actual δ_Q — env_seed key broken")
        print(f"  cross-model identical: {audit['cross_model_identical']}   "
              f"cross-actual different: {audit['cross_actual_different']}")

    gate = "PASS" if not failures else "FAIL"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"gate": gate, "white_rate": qd.WHITE_RATE,
                   "expected_c": qd.NOISE_SCALE_C,
                   "tolerance": args.tolerance,
                   "pooled_c_hat": c_all,
                   "per_actual": results, "pairing_audit": audit,
                   "failures": failures}, fh, indent=2)
    print(f"\nwrote {args.out} — R-1 {gate}")
    if failures:
        print("R-1 FAILED — the sim's noise convention does not match the "
              "design; HALT AND REPORT (do not rescale):")
        for f in failures:
            print(f"  - {f}")
    return 0 if gate == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
