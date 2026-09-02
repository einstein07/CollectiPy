#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2026 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""§7: emit both manifests, derive the templates, freeze the controllers.

    python3 scripts/qd_sweep_fixed_noise/generate_manifest.py \
        [--out-dir <dir>] [--n-runs-ra N] [--n-runs-ddm N] \
        [--ra-diffs 50,100,200] [--halted-ddm <ddm_trials.parquet>] [--force]

Blocking, in order, before anything is written:
  1. the §4 static-b* cross-check — the halted campaign's swept static trials
     are re-scored with the identical cost/RR functionals and the argmin /
     argmax must land on the sweep-derived 0.004189 / 0.1579 exactly;
  2. the §2 patcher assertions on probe configs at every actual δ_Q, plus the
     structural noise-invariance check (the sensory block byte-identical
     across actual δ_Q — nothing derived from δ_Q touches the noise).

Outputs into --out-dir:
    ra_manifest.csv             Arm A, 680 cells × |RA_SURFACE_DIFFS|
    ra_manifest_actual100.csv   the actual = 100 bp slice (§10 phase 1)
    ra_manifest_rest.csv        the remaining slices (§10 phase 2)
    ddm_manifest.csv            Arm B, 45 frozen-controller points
    frozen_controllers.json     the resolved freeze: b*(design), Wald anchors,
                                z_halt + halt-exit budget per point
plus `config/qd_sweep_{ra,ddm}_template.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import qd   # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=Path, default=qd.DEFAULT_OUT)
    ap.add_argument("--n-runs-ra", type=int, default=qd.N_RUNS_RA)
    ap.add_argument("--n-runs-ddm", type=int, default=qd.N_RUNS_DDM)
    ap.add_argument("--ra-diffs", default=None,
                    help="comma-separated actual δ_Q (bp) for the RA surface "
                         f"(default {qd.RA_SURFACE_DIFFS}; §3's cost knob — "
                         "never thin U_GRID)")
    ap.add_argument("--halted-ddm", type=Path, default=qd.HALTED_DDM_TRIALS,
                    help="the halted campaign's 1 %% ddm_trials.parquet "
                         "(the §4 b* source and cross-check)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    ra_diffs = ([int(x) for x in args.ra_diffs.split(",")]
                if args.ra_diffs else qd.RA_SURFACE_DIFFS)
    bad = [b for b in ra_diffs if b not in qd.DIFF_BP]
    if bad:
        raise SystemExit(f"--ra-diffs {bad} not in DIFF_BP {qd.DIFF_BP}; "
                         "adding a quality difference means appending to "
                         "DIFF_BP in qd.py (§2), not a CLI flag")

    # ---- 1. freeze the static controllers (§4, blocking cross-check) ------
    bstar100 = qd.derive_bstar_100(args.halted_ddm)
    print(f"§4 static-b* cross-check PASS: re-derived from {args.halted_ddm}")
    print(f"  b*_cost(100 bp) = {bstar100['cost']:g}   "
          f"b*_RR(100 bp) = {bstar100['rr']:g}   (sweep-derived, reproduced)")

    # ---- 2. templates + the §2 assertions on probes ------------------------
    qd.write_templates()
    print(f"wrote {qd.RA_TEMPLATE}")
    print(f"wrote {qd.DDM_TEMPLATE}")
    qd.assert_noise_invariant_across_actuals("ra")
    qd.assert_noise_invariant_across_actuals("ddm")
    print("§2 noise-invariance check PASS: sensory block byte-identical "
          f"across actual δ_Q ∈ {qd.DIFF_BP} bp (white_rate {qd.WHITE_RATE})")

    # ---- 3. manifests ------------------------------------------------------
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    ra_rows = qd.build_ra_rows(args.n_runs_ra, ra_diffs)
    ddm_rows = qd.build_ddm_rows(bstar100, args.n_runs_ddm)
    ra100 = [r for r in ra_rows if r["actual_bp"] == "100"]
    ra_rest = [r for r in ra_rows if r["actual_bp"] != "100"]

    jobs = [("ra", ra_rows, qd.RA_FIELDS, out / qd.RA_MANIFEST_NAME),
            ("ra actual=100 slice", ra100, qd.RA_FIELDS,
             out / "ra_manifest_actual100.csv"),
            ("ra remaining slices", ra_rest, qd.RA_FIELDS,
             out / "ra_manifest_rest.csv"),
            ("ddm", ddm_rows, qd.DDM_FIELDS, out / qd.DDM_MANIFEST_NAME)]
    for name, rows, fields, dest in jobs:
        if dest.exists() and not args.force:
            if qd.read_manifest(dest) == rows:
                print(f"[{name}] {dest} already up to date ({len(rows)} rows)")
                continue
            raise SystemExit(f"{dest} exists with DIFFERENT content — "
                             "re-keying a campaign mid-flight silently "
                             "invalidates it; pass --force only if intended.")
        qd.write_manifest(rows, fields, dest)
        print(f"[{name}] wrote {dest} ({len(rows)} rows)")

    # ---- 4. the frozen-controller record + halt budget ---------------------
    budget = qd.ddm_budget(ddm_rows)
    freeze = {
        "design_table": qd.design_table(),
        "white_rate": qd.WHITE_RATE, "c": qd.NOISE_SCALE_C,
        "bstar_100_sweep_derived": {"cost": bstar100["cost"],
                                    "rr": bstar100["rr"]},
        "bstar_100_source": str(args.halted_ddm),
        "frozen_static_bounds": {
            str(d): {"cost": qd.frozen_static_bound("cost", d, bstar100),
                     "rr": qd.frozen_static_bound("rr", d, bstar100),
                     "transport": "b(design) = b*(100)·k(100)/k(design) — "
                                  "preserves the believed commitment "
                                  "log-odds a = k·b (Wald invariant)"}
            for d in qd.DIFF_BP},
        "wald_analytic_anchors": {str(d): qd.wald_analytic_bstar(d)
                                  for d in qd.DIFF_BP},
        "frozen_ce": qd.FROZEN_CE,
        "halt_budget": budget,
    }
    with open(out / "frozen_controllers.json", "w", encoding="utf-8") as fh:
        json.dump(freeze, fh, indent=2)
    print(f"wrote {out / 'frozen_controllers.json'}")

    # ---- 5. report ---------------------------------------------------------
    print("\n§2 design table (fixed noise, SNR moves with δ_Q):")
    for r in qd.design_table():
        print(f"  δ_Q = {r['diff_bp']:>3d} bp   A = {r['A']:<6g} "
              f"A/c = {r['A_over_c']:<5g} k = {r['k']:g}")
    print("\nfrozen static bounds b*(design)  [+ Wald analytic anchors, "
          "reported only]:")
    for d in qd.DIFF_BP:
        w = freeze["wald_analytic_anchors"][str(d)]
        f = freeze["frozen_static_bounds"][str(d)]
        print(f"  design {d:>3d} bp:  b*_cost = {f['cost']:<10g} "
              f"b*_RR = {f['rr']:<9g} (Wald anchors {w['b_cost_wald']:.4g} / "
              f"{w['b_rr_wald']:.4g})")
    print("\nArm B halt budget (D at the ACTUAL SNR sets censor risk):")
    for r in budget:
        flags = ("  ** CENSOR RISK (mean + 3·D_actual > "
                 f"{qd.DDM_TIME_LIMIT} s — expect a visible censored tail, "
                 "reported not hidden)" if r["censor_risk"] else "") + \
                ("  (discretisation-limited: z_halt < substep "
                 f"{qd.EVIDENCE_SUBSTEP})"
                 if r["discretisation_limited"] else "")
        print(f"  {r['point_id']:>18s} [{r['variant']:>11s}] "
              f"z_halt = {r['z_halt']:.4g}  D_design = "
              f"{r['halt_mean_exit_design_s']:6.2f} s  D_actual = "
              f"{r['halt_mean_exit_actual_s']:6.2f} s{flags}")

    n_ra = len(ra_rows) * args.n_runs_ra
    n_ddm = len(ddm_rows) * args.n_runs_ddm
    print(f"\nvolumes: Arm A {len(ra_rows)} cells × {args.n_runs_ra} "
          f"= {n_ra:,} runs; Arm B {len(ddm_rows)} points × "
          f"{args.n_runs_ddm} = {n_ddm:,} runs")
    print("cost knobs, in order (§3): --n-runs-ra 600 (Wilson CIs ≈ ±0.03); "
          "then --ra-diffs 100 with the other δ_Q on a reduced v set — "
          "NEVER thin U_GRID (standing policy)")
    print(f"\ngit sha: {qd.git_sha()}")
    print(json.dumps({"seed_scheme": qd.seeding.SCHEME,
                      "dth_deg": qd.DTH_DEG,
                      "actual_bp": qd.DIFF_BP, "design_bp": qd.DIFF_BP,
                      "ra_frame": {"ticks_per_second": 1,
                                   "time_limit": qd.RA_TIME_LIMIT},
                      "ddm_frame": {"ticks_per_second": 1,
                                    "time_limit_s": qd.DDM_TIME_LIMIT}}))
    print("\nBLOCKING before any submission (§10): "
          "\n  1. R-1 noise-convention gate:   python3 "
          "scripts/qd_sweep_fixed_noise/r1_noise_convention.py"
          "\n  2. smoke (§10.2):               see README.md"
          "\n  3. step-halving at u = 35 (§3): python3 "
          "scripts/qd_sweep_fixed_noise/dt_check.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
