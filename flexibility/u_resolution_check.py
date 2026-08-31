# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Pin the REALISED critical point at delta = 0 (Section 2).

    python3 -m flexibility.u_resolution_check --results-root <dir> [--reps 200]

The `ra_uc` arm sits at u = 6.156868, which is the critical coupling of the
CONTINUUM mean-field theory at v = 0.5. Two things follow, and they are why this
check exists rather than being assumed away:

  * a result obtained exactly AT u_c depends on how u_c was determined -- continuum
    vs. finite-N, and on beta, sigma and kappa -- so the method and its precision
    belong in the methods section;
  * finite-N effects at num_neurons = 30 are LARGEST exactly here, so the effective
    critical point of the simulated system may sit slightly off the analytical one.

Three gains bracketing u_c, at delta = 0 where the symmetry is exact and critical
slowing down is most visible, at more replicates than the main grid uses. This is
deliberately a SEPARATE campaign: it answers a question about the model, not about
flexibility, and folding it into the main array would confound the two.

Output goes to <results-root>/u_resolution/u_<value>/replicate_<n>/, outside the
main campaign's arm/delta tree so it can never be swept up by that analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from flexibility import factors, genconfig, matrix, seeds  # noqa: E402


def main(argv=None) -> int:
    """Run the three-gain resolution check at delta = 0."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results-root", required=True, type=Path)
    ap.add_argument("--reps", type=int, default=factors.U_RESOLUTION_CHECK_REPS)
    ap.add_argument("--u-values", type=float, nargs="+",
                    default=list(factors.U_RESOLUTION_CHECK))
    ap.add_argument("--first-rep", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    from flexibility.run_chunk import InProcessRunner

    # delta = 0 with the RA template. The gain is overridden per value below, so the
    # arm identity (ra_uc vs ra_u8) is irrelevant here -- ra_uc is the carrier.
    cond = matrix.find_condition(f"ra_uc__{matrix.delta_token(0.0)}")
    runner = None if args.dry_run else InProcessRunner()
    root = args.results_root / "u_resolution"

    print(f"u-resolution check at delta = 0: u in {args.u_values}, "
          f"{args.reps} replicates each -> {root}")

    failures = []
    t0 = time.time()
    for u in args.u_values:
        for rep in range(args.first_rep, args.first_rep + args.reps):
            rep_dir = root / f"u_{u:g}" / f"replicate_{rep}"
            if any(rep_dir.glob("config_folder_*/run_*.zip")):
                continue
            rep_dir.mkdir(parents=True, exist_ok=True)
            cfg = genconfig.replicate_config(cond, rep, str(rep_dir))
            # Override the gain AFTER generation: the arm's own value is whatever
            # ra_uc carries, and this check is precisely about varying it.
            agent = next(iter(cfg["environment"]["agents"].values()))
            agent["mean_field_model"]["u"] = float(u)
            cfg_path = rep_dir / "config.json"
            cfg_path.write_text(json.dumps(cfg, indent=2))
            if args.dry_run:
                continue
            try:
                runner.run(cfg_path)
            except Exception as exc:  # noqa: BLE001
                failures.append({"u": u, "replicate": rep, "error": repr(exc)})
                print(f"  u={u} rep={rep} FAILED: {exc!r}")
        print(f"  u = {u:g} done ({time.time() - t0:.0f}s elapsed)")

    print(f"\nfinished in {time.time() - t0:.0f}s, {len(failures)} failure(s)")
    print("The realised critical point is read off the delta = 0 statistics: the "
          "symmetry-breaking TIME diverges and its variance peaks at the effective "
          "u_c, which need not be the analytical one at N = 30.")
    if failures:
        (root / "failures.json").write_text(json.dumps(failures, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
