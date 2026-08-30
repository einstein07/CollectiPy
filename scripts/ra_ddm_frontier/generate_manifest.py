#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2026 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""§5: emit the campaign manifests and derive both config templates.

    python3 scripts/ra_ddm_frontier/generate_manifest.py \
        --campaign ra|ddm|both --out-dir <dir> [--n-runs N] [--force]

- computes u*(v) from the simulator's own kernel builder and applies the 1 %
  anchor gate (|u*(0.5) − 6.157| / 6.157 ≤ 0.01) before writing anything;
- writes `manifest.csv` (RA: 100 cells over Set R + Set U) and/or
  `ddm_manifest.csv` (10 points, the previous frontier's c_e grid verbatim);
- derives `config/ra_ddm_frontier_{ra,ddm}_template.json` from the DDM
  frontier's base template (§4 lineage; see RECON.md D-07 for the diff against
  the factorial's effective config).

The spec's submit script also calls this with `--diff/--n-runs/--out`
(campaign-specific); those spellings are accepted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import frontier   # noqa: E402
import seeding    # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--campaign", choices=("ra", "ddm", "both"), default="both")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="directory receiving the manifest(s)")
    ap.add_argument("--out", type=Path, default=None,
                    help="explicit manifest path (single-campaign form)")
    ap.add_argument("--n-runs", type=int, default=frontier.N_RUNS)
    ap.add_argument("--diff", type=float, default=frontier.DIFF,
                    help="accepted for interface compatibility; must equal "
                         f"{frontier.DIFF} (other panels are out of scope, §13)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing manifest")
    ap.add_argument("--topup-from", type=Path, default=None, metavar="CELLS_CSV",
                    help="wave-2 mode (§2 Set U-v2 + §11 DDM ceiling points): "
                         "derive the top-up grids from wave-1 cells.csv and "
                         "write manifest_topup/manifest_full (and DDM twins) "
                         "into --out-dir")
    args = ap.parse_args(argv)

    if args.topup_from is not None:
        return topup_main(args)

    if abs(args.diff - frontier.DIFF) > 1e-12:
        raise SystemExit(f"--diff {args.diff} != {frontier.DIFF}: other panels "
                         "are out of scope (§13); edit frontier.py deliberately "
                         "if a new panel is really intended.")
    if args.out is not None and args.campaign == "both":
        raise SystemExit("--out names one file; use it with --campaign ra|ddm")

    anchor = frontier.anchor_check()
    print(f"anchor u*({anchor['v']}) = {anchor['u_star']:.6f} "
          f"(rel. err {anchor['relative_error']:.2e} <= {anchor['tolerance']}) PASS")

    frontier.write_templates()
    print(f"wrote {frontier.RA_TEMPLATE}")
    print(f"wrote {frontier.DDM_TEMPLATE}")

    out_dir = args.out_dir or (frontier._ROOT / "results" / "ra_ddm_frontier")
    jobs = []
    if args.campaign in ("ra", "both"):
        jobs.append(("ra", frontier.build_ra_rows(args.n_runs),
                     frontier.RA_FIELDS,
                     args.out or out_dir / frontier.RA_MANIFEST_NAME))
    if args.campaign in ("ddm", "both"):
        jobs.append(("ddm", frontier.build_ddm_rows(args.n_runs),
                     frontier.DDM_FIELDS,
                     args.out or out_dir / frontier.DDM_MANIFEST_NAME))

    for name, rows, fields, dest in jobs:
        if dest.exists() and not args.force:
            existing = frontier.read_manifest(dest)
            if existing == rows:
                print(f"[{name}] {dest} already up to date ({len(rows)} rows)")
                continue
            raise SystemExit(
                f"{dest} exists with DIFFERENT content. Re-keying a campaign "
                "mid-flight silently invalidates it; pass --force only if that "
                "is really intended.")
        frontier.write_manifest(rows, fields, dest)
        print(f"[{name}] wrote {dest} ({len(rows)} rows, n_runs={args.n_runs}, "
              f"scheme={seeding.SCHEME})")

    if args.campaign in ("ra", "both"):
        print("\nu*(v) on the design grid:")
        for v in frontier.V_GRID:
            us, lam = frontier.u_star(v)
            print(f"  v = {v:g}: u* = {us:9.6f}  (lambda_max {lam:.6f}; "
                  f"Set R u in [{0.45 * us:7.4f}, {1.50 * us:7.4f}]; "
                  f"Set U u_hat up to {15.0 / us:.3f})")
    print(f"\ngit sha: {frontier.git_sha()}")
    print(json.dumps({"seed_scheme": seeding.SCHEME,
                      "dth_deg": frontier.DTH_DEG,
                      "diff_bp": frontier.DIFF_BP,
                      "ra_frame": {"ticks_per_second": frontier.RA_TICKS_PER_SECOND,
                                   "time_limit": frontier.RA_TIME_LIMIT},
                      "ddm_frame": {"ticks_per_second": frontier.DDM_TICKS_PER_SECOND,
                                    "time_limit_s": frontier.DDM_TIME_LIMIT},
                      "bellman_N_t": frontier.BELLMAN_N_T}))
    return 0


def topup_main(args) -> int:
    """Wave 2: Set U-v2 (measured-cliff sampling + matched tails) and the DDM
    ceiling points. Emits, per §2/§5:
        manifest_topup.csv       new RA cells only (submission)
        manifest_full.csv        wave 1 + wave 2 (analysis + completeness)
        ddm_manifest_topup.csv   the extreme-c_e ceiling points (submission)
        ddm_manifest_full.csv    previous grid + ceiling points
    """
    out_dir = args.out_dir or (frontier._ROOT / "results" / "ra_ddm_frontier")
    frontier.anchor_check()
    frontier.write_templates()

    topup, report = frontier.build_topup_rows(args.topup_from, args.n_runs)
    wave1 = frontier.build_ra_rows(args.n_runs)
    print("Set U-v2 — measured cliff windows and chosen u per v "
          f"(rule: û_hi = last acc_all > {frontier.TOPUP_ACC_HI}, "
          f"û_lo = first < {frontier.TOPUP_ACC_LO}, Δû = {frontier.TOPUP_DU_HAT}; "
          f"anchors {frontier.TOPUP_ANCHORS}·u*, committed "
          f"{frontier.TOPUP_COMMITTED}·u*, tail {frontier.TOPUP_TAIL} in û; "
          f"skip within {frontier.TOPUP_SKIP_REL:.0%} of an existing Set-U level):")
    for v, r in report.items():
        print(f"  v = {v:g} (u* = {r['u_star']:.4f}): window û ∈ "
              f"[{r['uhat_hi']:g}, {r['uhat_lo']:g}] -> new u = {r['new_u']}")
    print(f"  -> {len(topup)} new cells x {args.n_runs} runs "
          f"= {len(topup) * args.n_runs} additional replicates")

    ddm_ceiling = frontier.build_ddm_ceiling_rows(args.n_runs)
    ddm_full = frontier.build_ddm_rows(args.n_runs) + ddm_ceiling
    print(f"DDM ceiling points (§11): c_e = "
          f"{[float(r['c_e']) for r in ddm_ceiling]}; analytic "
          f"infinite-patience asymptote Phi((A/c)·sqrt(r0/v)) = "
          f"{frontier.ddm_ideal_ceiling():.4f}")

    jobs = [("ra topup", topup, frontier.RA_FIELDS,
             out_dir / "manifest_topup.csv"),
            ("ra full", wave1 + topup, frontier.RA_FIELDS,
             out_dir / "manifest_full.csv"),
            ("ddm topup", ddm_ceiling, frontier.DDM_FIELDS,
             out_dir / "ddm_manifest_topup.csv"),
            ("ddm full", ddm_full, frontier.DDM_FIELDS,
             out_dir / "ddm_manifest_full.csv")]
    for name, rows, fields, dest in jobs:
        if dest.exists() and not args.force:
            if frontier.read_manifest(dest) == rows:
                print(f"[{name}] {dest} already up to date ({len(rows)} rows)")
                continue
            raise SystemExit(f"{dest} exists with DIFFERENT content — a "
                             "re-derived top-up would re-key wave 2; pass "
                             "--force only if that is intended.")
        frontier.write_manifest(rows, fields, dest)
        print(f"[{name}] wrote {dest} ({len(rows)} rows)")

    hi = sorted(topup, key=lambda r: float(r["u"]))[-3:]
    print("\nBLOCKING before submission (§2): step-halving check at the three "
          "highest new u values:")
    print(f"  python3 scripts/ra_ddm_frontier/dt_check.py --manifest "
          f"{out_dir / 'manifest_topup.csv'}   # cells "
          + ", ".join(r['cell_id'] for r in hi))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
