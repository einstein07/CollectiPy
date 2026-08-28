#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Step 1: compute u*(v) and emit the sweep manifest.

Section 5 of `uhat-v-factorial-experiment.md`.

    u*(v) = 1 / ( lambda_max(W(v)) * sech^2(beta) )

`W(v)` is built by IMPORTING the simulator's own kernel builder
(`MeanFieldSystem.compute_interaction_kernel`) at the runtime N and with the
runtime normalisation. Reimplementing the formula here would let a silent
normalisation mismatch invalidate u_hat for the entire sweep, so it is not done.

A hard anchor gate at v = 0.5 (u* = 6.157) halts before writing anything if the
kernel or its normalisation has moved. The discrepancy is NEVER rescaled away.

Usage:
    python3 scripts/uhat_v_sweep/generate_manifest.py \
        --results-root results/uhat_v_sweep [--trials N] [--force]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in (str(_HERE), str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import factors                                    # noqa: E402
import config_patch                               # noqa: E402
from models.mean_field_systems import MeanFieldSystem   # noqa: E402


# ---------------------------------------------------------------------------
# u*(v)
# ---------------------------------------------------------------------------
def lambda_max(v: float, n: int = factors.NUM_NEURONS) -> float:
    """Dominant eigenvalue of the simulator's OWN connectivity operator W(v).

    `MeanFieldSystem.__init__` builds `self.M` through
    `compute_interaction_kernel()`; reading it back is what keeps the manifest's
    arithmetic and the simulation's dynamics on the same matrix.
    """
    system = MeanFieldSystem(num_neurons=n, v=float(v), beta=factors.BETA)
    eigenvalues = np.linalg.eigvals(system.M)
    return float(np.max(np.real(eigenvalues)))


def u_star(v: float, beta: float = factors.BETA,
           n: int = factors.NUM_NEURONS) -> tuple[float, float]:
    """Return (u_star, lambda_max) for kernel shape `v`."""
    lam = lambda_max(v, n=n)
    sech2 = 1.0 / np.cosh(beta) ** 2
    if lam <= 0.0:
        raise SystemExit(
            f"lambda_max(v={v}) = {lam:.6g} is not positive; u* is undefined. "
            "The kernel construction has changed — halt for human review."
        )
    return float(1.0 / (lam * sech2)), lam


def anchor_check() -> dict:
    """Section 5 hard gate. Halts with diagnostics rather than 'fixing' a drift."""
    us, lam = u_star(factors.ANCHOR_V)
    rel = abs(us - factors.ANCHOR_U_STAR) / factors.ANCHOR_U_STAR
    report = {
        "v": factors.ANCHOR_V,
        "lambda_max": lam,
        "u_star": us,
        "expected_u_star": factors.ANCHOR_U_STAR,
        "relative_error": rel,
        "tolerance": factors.ANCHOR_TOL,
        "num_neurons": factors.NUM_NEURONS,
        "beta": factors.BETA,
        "kernel_normalisation": "(1/N) * cos(pi * (|dtheta|/pi)**v), "
                                "MeanFieldSystem.compute_interaction_kernel",
        "passed": rel <= factors.ANCHOR_TOL,
    }
    if not report["passed"]:
        print("ANCHOR CHECK FAILED — halting for human review.", file=sys.stderr)
        print(json.dumps(report, indent=2), file=sys.stderr)
        print(
            "\nDo NOT rescale to close this gap. Either the kernel formula, its "
            "1/N normalisation, num_neurons or beta has changed; u_hat would be "
            "meaningless for the entire sweep until that is understood.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return report


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def u_star_table(step: float = 0.05, v_max: float = 1.00) -> list[dict]:
    """Fine u*(v) grid for later overlays (and the u_WTA follow-up)."""
    rows = []
    n_steps = int(round(v_max / step))
    for i in range(1, n_steps + 1):
        v = round(i * step, 10)
        us, lam = u_star(v)
        rows.append({"v": v, "lambda_max": lam, "u_star": us})
    return rows


def build_manifest(n_trials: int) -> dict:
    """One record per cell. u is DERIVED, never typed in."""
    u_star_cache = {v: u_star(v) for v in factors.V_GRID}
    cells = []
    for cid, v, u_hat in factors.iter_cells():
        us, lam = u_star_cache[v]
        cells.append({
            "cell_id": cid,
            "v": v,
            "u_hat": u_hat,
            "lambda_max": lam,
            "u_star": us,
            "u": u_hat * us,
            "n_trials": int(n_trials),
            "base_seed": int(factors.BASE_SEED),
            "excluded": False,
            "excluded_reason": "",
        })
    template = config_patch.load_template()
    probe = config_patch.cell_config(cells[0], template=template)
    return {
        "sweep": "uhat_v_sweep",
        "spec": "uhat-v-factorial-experiment.md",
        "git_sha": config_patch.git_sha(),
        "base_config": factors.BASE_CONFIG,
        "config_hash_probe": config_patch.config_hash(probe),
        "u_hat_grid": factors.U_HAT_GRID,
        "v_grid": factors.V_GRID,
        "n_cells": factors.N_CELLS,
        "n_trials": int(n_trials),
        "base_seed": int(factors.BASE_SEED),
        "t_max_ticks": factors.T_MAX_TICKS,
        "u_star_formula": "1 / (lambda_max(W(v)) * sech^2(beta))",
        "anchor": anchor_check(),
        "locked": config_patch.env_summary(probe),
        "cells": cells,
    }


def _write_json(obj: dict, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


def _write_u_star_csv(rows: list[dict], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["v", "lambda_max", "u_star"])
        writer.writeheader()
        writer.writerows(rows)


def report(manifest: dict) -> None:
    """The Section 5 manifest report."""
    cells = manifest["cells"]
    anchor = manifest["anchor"]
    max_cell = max(cells, key=lambda c: c["u"])
    print("(u_hat, v) factorial — sweep manifest")
    print(f"  git sha            : {manifest['git_sha']}")
    print(f"  base config        : {manifest['base_config']}")
    print(f"  config hash        : {manifest['config_hash_probe']}")
    print(f"  cells              : {manifest['n_cells']} "
          f"({len(factors.U_HAT_GRID)} u_hat x {len(factors.V_GRID)} v)")
    print(f"  trials per cell    : {manifest['n_trials']}  "
          f"(total {manifest['n_cells'] * manifest['n_trials']})")
    print(f"  seeds              : {manifest['base_seed']} .. "
          f"{manifest['base_seed'] + manifest['n_trials'] - 1}, identical in every cell")
    print(f"  T_max              : {manifest['t_max_ticks']} control ticks")
    print(f"  anchor u*({anchor['v']})    : {anchor['u_star']:.6f} "
          f"(lambda_max {anchor['lambda_max']:.6f}, rel. err "
          f"{anchor['relative_error']:.2e} <= {anchor['tolerance']}) PASS")
    print(f"  locked             : {json.dumps(manifest['locked'], sort_keys=True)}")
    print()
    print("  u*(v) on the design grid:")
    for v in factors.V_GRID:
        us = next(c["u_star"] for c in cells if c["v"] == v)
        span = [c["u"] for c in cells if c["v"] == v]
        print(f"    v = {v:<4} u* = {us:9.6f}   u in [{min(span):8.4f}, {max(span):8.4f}]")
    print()
    print(f"  max resolved u     : {max_cell['u']:.4f} "
          f"(v = {max_cell['v']}, u_hat = {max_cell['u_hat']})")
    if max_cell["u"] > factors.MAX_U_WARN:
        print()
        print("  " + "!" * 68)
        print(f"  !! max u = {max_cell['u']:.4f} EXCEEDS {factors.MAX_U_WARN}. The "
              "stiffest cells are")
        print("  !! outside the range the integrator was last validated on. Run the")
        print("  !! step-halving check (dt_check.py) BEFORE submitting.")
        print("  " + "!" * 68)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results-root", type=Path,
                    default=_ROOT / "results" / "uhat_v_sweep")
    ap.add_argument("--trials", type=int, default=factors.N_TRIALS,
                    help=f"trials per cell (default {factors.N_TRIALS})")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing manifest")
    args = ap.parse_args(argv)

    manifest_path = args.results_root / "manifest.json"
    if manifest_path.exists() and not args.force:
        raise SystemExit(
            f"{manifest_path} already exists. Re-generating it after runs have "
            "started would silently re-key the sweep; pass --force if that is "
            "really what you want."
        )

    manifest = build_manifest(args.trials)
    _write_json(manifest, manifest_path)
    _write_u_star_csv(u_star_table(), args.results_root / "u_star_table.csv")

    report(manifest)
    print()
    print(f"  wrote {manifest_path}")
    print(f"  wrote {args.results_root / 'u_star_table.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
