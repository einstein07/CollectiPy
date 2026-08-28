#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Step 5: concatenate the per-task files, validate, emit trials + cell summaries.

Section 9 of `uhat-v-factorial-experiment.md`.

    python3 scripts/uhat_v_sweep/aggregate.py --results-root results/uhat_v_sweep

Emits
    trials.parquet   every trial, one row each (CSV fallback if pyarrow is absent)
    cells.csv        one row per cell
    aggregate_report.json / stdout report

Accuracy is reported BOTH ways, everywhere:

    acc_decided   correct / decided.  Conditioning on having decided is a
                  selection effect; it is not the headline on its own.
    acc_all       correct / n, scoring an undecided trial as an error. This is
                  the 0-1 terminal loss the design specifies.

The gap between them is itself informative, so `decided_frac` is carried
alongside rather than folded into either.
"""

from __future__ import annotations

import argparse
import json
import math
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
import run_cell           # noqa: E402


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson interval — behaves at p near 0 and 1 where the normal one does not."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_raw(raw_dir: Path) -> pd.DataFrame:
    paths = sorted(list(raw_dir.glob("cell*_chunk*.parquet"))
                   + list(raw_dir.glob("cell*_chunk*.csv")))
    if not paths:
        raise SystemExit(f"No per-task files under {raw_dir}")
    frames = []
    for path in paths:
        frame = (pd.read_parquet(path) if path.suffix == ".parquet"
                 else pd.read_csv(path))
        frame["_source"] = path.name
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    for col in ("decided", "correct", "timeout", "numerical_failure"):
        data[col] = data[col].fillna(False).astype(bool)
    return data


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
def validate(data: pd.DataFrame, manifest: dict, smoke: bool = False) -> dict:
    """Structural checks. Problems are REPORTED, never silently repaired."""
    problems: list[str] = []
    warnings: list[str] = []

    expected = {int(c["cell_id"]): c for c in manifest["cells"]
                if not c.get("excluded")}
    if smoke:
        # Section 12: the smoke run is two cells at 12 trials. Expecting the full
        # design here would bury the pipeline checks under 78 absent-cell lines.
        smoke_ids = {factors.cell_id(v, u) for v, u in factors.SMOKE_CELLS}
        expected = {cid: dict(c, n_trials=factors.SMOKE_TRIALS)
                    for cid, c in expected.items() if cid in smoke_ids}
    excluded = {int(c["cell_id"]) for c in manifest["cells"] if c.get("excluded")}
    present = set(int(x) for x in data["cell_id"].unique())

    missing = sorted(set(expected) - present)
    if missing:
        shown = missing[:20]
        problems.append(f"{len(missing)} cells absent from raw/: {shown}"
                        + (" ..." if len(missing) > len(shown) else ""))
    unexpected = sorted(present - set(expected) - excluded)
    if unexpected:
        problems.append(f"cell_ids not in the manifest: {unexpected}")

    counts = data.groupby("cell_id").size()
    short = []
    for cid, cell in expected.items():
        got = int(counts.get(cid, 0))
        want = int(cell["n_trials"])
        if got != want:
            short.append(f"cell {cid} (v={cell['v']}, u_hat={cell['u_hat']}): "
                         f"{got} trials, expected {want}")
    if short:
        head = "\n    ".join(short[:10])
        tail = (f"\n    ... and {len(short) - 10} more"
                if len(short) > 10 else "")
        problems.append(f"{len(short)} cells with the wrong trial count:\n"
                        f"    {head}{tail}")

    dupes = data.duplicated(subset=["cell_id", "seed"], keep=False)
    if dupes.any():
        rows = data.loc[dupes, ["cell_id", "seed", "_source"]]
        problems.append(
            f"{int(dupes.sum())} duplicate (cell_id, seed) rows, e.g.\n"
            + rows.head(10).to_string(index=False)
        )

    # The paired design only holds if every cell ran the identical seed list.
    seed_sets = data.groupby("cell_id")["seed"].apply(lambda s: frozenset(s.tolist()))
    if len(set(seed_sets)) > 1:
        sizes = {cid: len(s) for cid, s in seed_sets.items()}
        warnings.append(
            "cells do not share one seed list, so paired contrasts across cells "
            f"are not valid on the full set (sizes: {sizes})"
        )

    shas = sorted(str(x) for x in data["git_sha"].unique())
    if len(shas) > 1:
        warnings.append(f"rows carry {len(shas)} different git shas: {shas}")
    if any(s.endswith("-dirty") for s in shas):
        warnings.append("at least one git sha is '-dirty': the tree was modified "
                        "relative to any commit, so the code is not recoverable "
                        "from the sha alone")
    n_hashes = data.groupby("cell_id")["config_hash"].nunique()
    if (n_hashes > 1).any():
        bad = n_hashes[n_hashes > 1].index.tolist()
        problems.append(f"cells with more than one config_hash: {bad}")

    fail_frac = data.groupby("cell_id")["numerical_failure"].mean()
    hot = fail_frac[fail_frac > 0.01]
    if len(hot):
        warnings.append(
            "cells with numerical_failure_frac > 1 % (Section 11 says flag these "
            "prominently): "
            + ", ".join(f"{int(c)}:{f:.3f}" for c, f in hot.items())
        )
    return {"problems": problems, "warnings": warnings,
            "excluded_cells": sorted(excluded)}


# ---------------------------------------------------------------------------
# Cell summaries
# ---------------------------------------------------------------------------
def cell_table(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cid,), group in data.groupby(["cell_id"], sort=True):
        n = len(group)
        decided = group[group["decided"]]
        n_dec = len(decided)
        k_dec = int(decided["correct"].sum())
        k_all = int(group["correct"].sum())
        lo_d, hi_d = wilson(k_dec, n_dec)
        lo_a, hi_a = wilson(k_all, n)
        dt = decided["t_commit_ticks"].astype(float)
        dtf = decided["t_commit_fine"].astype(float)
        rows.append({
            "cell_id": int(cid),
            "v": float(group["v"].iloc[0]),
            "u_hat": float(group["u_hat"].iloc[0]),
            "u_star": float(group["u_star"].iloc[0]),
            "u": float(group["u"].iloc[0]),
            "n": n,
            "n_decided": n_dec,
            "decided_frac": n_dec / n if n else float("nan"),
            "acc_decided": k_dec / n_dec if n_dec else float("nan"),
            "acc_decided_lo": lo_d, "acc_decided_hi": hi_d,
            "acc_all": k_all / n if n else float("nan"),
            "acc_all_lo": lo_a, "acc_all_hi": hi_a,
            "t_commit_median": float(dt.median()) if n_dec else float("nan"),
            "t_commit_q25": float(dt.quantile(0.25)) if n_dec else float("nan"),
            "t_commit_q75": float(dt.quantile(0.75)) if n_dec else float("nan"),
            "t_commit_iqr": float(dt.quantile(0.75) - dt.quantile(0.25))
                            if n_dec else float("nan"),
            "t_commit_fine_median": float(dtf.median()) if n_dec else float("nan"),
            "t_commit_fine_iqr": float(dtf.quantile(0.75) - dtf.quantile(0.25))
                                 if n_dec else float("nan"),
            "numerical_failure_count": int(group["numerical_failure"].sum()),
            "numerical_failure_frac": float(group["numerical_failure"].mean()),
            "config_hash": str(group["config_hash"].iloc[0]),
        })
    return pd.DataFrame(rows).sort_values(["v", "u_hat"]).reset_index(drop=True)


def _write_trials(data: pd.DataFrame, root: Path) -> Path:
    dest = root / "trials.parquet"
    try:
        data.to_parquet(dest, index=False)
        return dest
    except Exception as exc:                        # noqa: BLE001
        print(f"  parquet unavailable ({exc!r}); writing CSV instead")
        dest = root / "trials.csv"
        data.to_csv(dest, index=False)
        return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results-root", type=Path,
                    default=_ROOT / "results" / "uhat_v_sweep")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="Section 12: validate against the smoke design "
                         "(2 cells x 12 trials) instead of the full 80-cell one")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="write the outputs even when validation finds problems")
    args = ap.parse_args(argv)

    root = args.results_root.resolve()
    manifest = run_cell.load_manifest(args.manifest or (root / "manifest.json"))
    data = load_raw(root / "raw")
    checks = validate(data, manifest, smoke=args.smoke)

    print(f"(u_hat, v) factorial — aggregation of {root}")
    print(f"  raw files          : {data['_source'].nunique()}")
    print(f"  trials             : {len(data)}")
    print(f"  cells present      : {data['cell_id'].nunique()} of "
          f"{len(factors.SMOKE_CELLS) if args.smoke else manifest['n_cells']}"
          + ("   [SMOKE: statistics are meaningless, this checks the pipeline]"
             if args.smoke else ""))
    print(f"  decided            : {int(data['decided'].sum())} "
          f"({data['decided'].mean():.4f})")
    print(f"  acc_all            : {data['correct'].mean():.4f}")
    print(f"  numerical failures : {int(data['numerical_failure'].sum())}")
    if checks["excluded_cells"]:
        print(f"  EXCLUDED cells     : {checks['excluded_cells']} — the design is "
              "unbalanced and the analysis must say so")
    for warning in checks["warnings"]:
        print(f"  WARNING: {warning}")
    for problem in checks["problems"]:
        print(f"  PROBLEM: {problem}")

    if checks["problems"] and not args.allow_incomplete:
        print("\nRefusing to write outputs. Re-run the missing tasks, or pass "
              "--allow-incomplete to aggregate anyway (the analysis will then be "
              "on an unbalanced design).")
        return 1

    cells = cell_table(data)
    trials_path = _write_trials(data.drop(columns=["_source"]), root)
    cells_path = root / "cells.csv"
    cells.to_csv(cells_path, index=False)
    report = {
        "results_root": str(root),
        "n_trials": int(len(data)),
        "n_cells": int(data["cell_id"].nunique()),
        "git_shas": sorted(str(x) for x in data["git_sha"].unique()),
        **checks,
    }
    with open(root / "aggregate_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    anchor_v, anchor_u_hat = factors.EXPECTED_ANCHOR_CELL
    anchor = cells[(cells["v"] == anchor_v) & (cells["u_hat"] == anchor_u_hat)]
    if len(anchor):
        row = anchor.iloc[0]
        print()
        print(f"  Section 12 expectation check — cell (v={anchor_v}, "
              f"u_hat={anchor_u_hat}), prior matched-critical result "
              f"acc {factors.EXPECTED_ACC}, DT {factors.EXPECTED_DT_TICKS} ticks:")
        print(f"    acc_decided {row['acc_decided']:.4f} "
              f"[{row['acc_decided_lo']:.4f}, {row['acc_decided_hi']:.4f}]  "
              f"acc_all {row['acc_all']:.4f}  "
              f"median t_commit {row['t_commit_median']:.2f} ticks")
        if (not (row["acc_decided_lo"] <= factors.EXPECTED_ACC <= row["acc_decided_hi"])
                or abs(row["t_commit_median"] - factors.EXPECTED_DT_TICKS) > 1.0):
            print("    DISAGREES with the prior result. Investigate a config or "
                  "normalisation drift before trusting anything else in this sweep.")

    print()
    print(f"  wrote {trials_path}")
    print(f"  wrote {cells_path}")
    print(f"  wrote {root / 'aggregate_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
