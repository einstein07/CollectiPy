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
    ap.add_argument("--wave3-from", type=Path, default=None, metavar="CELLS_CSV",
                    help="wave-3 mode (§2 Set U-v3): output-plane arc-length "
                         "re-spacing at v ∈ {0.2..0.5} + kernel extension to "
                         "v ∈ {0.6, 0.8}; derived from the pooled waves-1+2 "
                         "cells.csv; writes manifest_wave3.csv and updates "
                         "manifest_full.csv in --out-dir")
    ap.add_argument("--factorial", type=Path, default=None, metavar="CELLS_CSV",
                    help="wave-3 mode: the factorial (uhat_v_sweep) cells.csv "
                         "supplying the initial (t, acc) maps at v ∈ {0.6, 0.8}")
    args = ap.parse_args(argv)

    if args.topup_from is not None:
        return topup_main(args)
    if args.wave3_from is not None:
        return wave3_main(args)

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
        # §2b: the DDM campaign is the halt-at-midpoint rerun — Bellman grid
        # verbatim + §11 ceiling extremes + the static-b family, one manifest.
        # (The frozen wave-1/2 forced-choice manifest is NOT regenerated; that
        # tree is the §9 regression reference.)
        jobs.append(("ddm", frontier.build_ddm_halt_rows(args.n_runs),
                     frontier.DDM_FIELDS,
                     args.out or out_dir / frontier.DDM_HALT_MANIFEST_NAME))

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
    if args.campaign in ("ddm", "both"):
        print("\n§2b halt campaign (terminal = "
              f"{frontier.DDM_TERMINAL}, c_h = {frontier.DDM_HALT_COST_RATE}) "
              "— z_halt and censoring margin per point:")
        for r in frontier.ddm_halt_budget():
            flags = ("  ** CENSOR RISK: mean + 3·D(z_halt) exceeds "
                     f"time_limit {frontier.DDM_TIME_LIMIT} s — expect a "
                     "visible censored fraction, reported by aggregate.py"
                     if r["censor_risk"] else "") + \
                    ("  (discretisation-limited: z_halt < substep "
                     f"{frontier.EVIDENCE_SUBSTEP})"
                     if r["discretisation_limited"] else "")
            print(f"  {r['point_id']:>12s} [{r['variant']:>7s}] "
                  f"z_halt = {r['z_halt']:.4g}  D = {r['halt_mean_exit_s']:6.2f} s  "
                  f"mean total ≈ {r['mean_total_s']:5.1f} s{flags}")
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
            ("ddm topup", ddm_ceiling, frontier.DDM_FIELDS_V1,
             out_dir / "ddm_manifest_topup.csv"),
            ("ddm full", ddm_full, frontier.DDM_FIELDS_V1,
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


def wave3_main(args) -> int:
    """Wave 3 (§2 Set U-v3): arc-length re-spacing at v ∈ {0.2..0.5} plus the
    kernel extension to v ∈ {0.6, 0.8}. Emits, per §2/§5:
        manifest_wave3.csv   new RA cells only (submission)
        manifest_full.csv    waves 1+2+3 (analysis + completeness reference)
    The wave-1+2 portion of manifest_full.csv is read from --out-dir, never
    re-derived (re-deriving the U-v2 windows from pooled data would re-key
    wave 2)."""
    out_dir = args.out_dir or (frontier._ROOT / "results" / "ra_ddm_frontier")
    frontier.anchor_check()
    frontier.write_templates()

    prev_full = out_dir / "manifest_full.csv"
    if not prev_full.is_file():
        raise SystemExit(f"{prev_full} not found — wave 3 appends to the "
                         "waves-1+2 manifest; copy it into --out-dir first")
    prior = frontier.read_manifest(prev_full)
    prior_ids = {r["cell_id"] for r in prior}

    wave3, report = frontier.build_wave3_rows(
        args.wave3_from, args.factorial, args.n_runs)
    wave3 = [r for r in wave3 if r["cell_id"] not in prior_ids]
    print("Set U-v3 — output-plane arc-length placement "
          f"({frontier.W3_PER_BRANCH}/branch; gap-fill at chords > "
          f"{frontier.W3_GAPFILL_FACTOR:g}x budget; skip within "
          f"{frontier.W3_SKIP_REL:.0%} of an existing level; extension v also "
          f"gets anchors {frontier.W3_ANCHORS}·u*, tails {frontier.W3_TAIL} "
          "in û, and a u = 0 control):")
    for v, rep in report.items():
        print(f"  v = {v:g} (u* = {rep['u_star']:.4f}, map: "
              f"{rep['map_source']}, u_peak ≈ {rep.get('u_peak', float('nan')):g}, "
              f"budget {rep.get('arc_length_budget', float('nan')):.3f}): "
              f"new u = {rep['new_u']}")
    print(f"  -> {len(wave3)} new cells x {args.n_runs} runs "
          f"= {len(wave3) * args.n_runs} additional replicates")

    jobs = [("ra wave3", wave3, frontier.RA_FIELDS,
             out_dir / "manifest_wave3.csv"),
            ("ra full", prior + wave3, frontier.RA_FIELDS, prev_full)]
    for name, rows, fields, dest in jobs:
        if dest.exists() and not args.force and dest.name != "manifest_full.csv":
            if frontier.read_manifest(dest) == rows:
                print(f"[{name}] {dest} already up to date ({len(rows)} rows)")
                continue
            raise SystemExit(f"{dest} exists with DIFFERENT content — a "
                             "re-derived wave 3 would re-key it; pass --force "
                             "only if that is intended.")
        frontier.write_manifest(rows, fields, dest)
        print(f"[{name}] wrote {dest} ({len(rows)} rows)")

    old_max = {float(r["v"]): max(float(x["u"]) for x in prior
                                  if x["v"] == r["v"])
               for r in prior}
    stiff = [r for r in wave3
             if float(r["u"]) > old_max.get(float(r["v"]), 0.0)]
    if stiff:
        hi = sorted(stiff, key=lambda r: float(r["u"]))[-3:]
        print("\nBLOCKING before submission (§2): step-halving check at any "
              "new per-v u maximum:")
        print(f"  python3 scripts/ra_ddm_frontier/dt_check.py --manifest "
              f"{out_dir / 'manifest_wave3.csv'} --cells "
              + ",".join(r["cell_id"] for r in hi))
    else:
        print("\nno wave-3 cell exceeds its v's previously checked u maximum "
              "— no new step-halving required (§2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
