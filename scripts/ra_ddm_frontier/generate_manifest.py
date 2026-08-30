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
    args = ap.parse_args(argv)

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


if __name__ == "__main__":
    raise SystemExit(main())
