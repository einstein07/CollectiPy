# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""raw/ -> trials table + per-cell summary (spec 6.4), and the 6.6 sanity summary.

Per cell (summarised over the n paired trials):

    p_mean_of_pair   P(outcome class at commitment == "mean_of_pair")
    p_arrive_C       P(arrived at C)
    p_arrive_AB      P(arrived at A or B)
    p_timeout        P(no arrival within T_max)
    p_no_commit      P(the midpoint-aware commitment never fired)
    p_bimodal_t2     P(the ring holds two or more bumps on tick 2): the map kept the
                     pair apart at the ring level
    p_merged_t2      P(one bump at the A-B midpoint on tick 2): the map handed the
                     ring a fictitious super-target
    median_t_commit  median commitment tick (committed trials)
    median_t_arrival median arrival time, sub-tick refined (arrived trials)

every proportion with a 95 % Wilson interval. Paired contrasts (max - sum) per
(Delta_close, q_C) use the common seed list: the difference of per-trial indicators
has standard error sd(d)/sqrt(n).

The 6.6 check: accuracy (arrival at static_0) and median arrival time, sum vs max,
paired by seed; the shift is compared with the paired-trial standard error.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from map_reduction import factors
except ImportError:                    # pragma: no cover - direct import
    import factors                     # type: ignore


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson interval — behaves at p near 0 and 1."""
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
def load_raw(raw_dir: Path, pattern: str = "cell*_chunk*") -> pd.DataFrame:
    paths = sorted(list(raw_dir.glob(pattern + ".parquet")) + list(raw_dir.glob(pattern + ".csv")))
    if not paths:
        raise SystemExit(f"No per-task files under {raw_dir}")
    frames = []
    for path in paths:
        frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        frame["_source"] = path.name
        frames.append(frame)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        data = pd.concat(frames, ignore_index=True)
    for col in ("arrived", "timeout", "committed", "numerical_failure", "correct"):
        if col in data.columns:
            data[col] = data[col].map(_to_bool).fillna(False).astype(bool)
    for col in ("reached", "commit_class", "class_final", "bif_target", "state_t2"):
        if col in data.columns:
            data[col] = data[col].fillna("").astype(str)
    return data


def _to_bool(v):
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "t", "yes")
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return bool(v)


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
def validate(data: pd.DataFrame, manifest: dict, n_expected: int | None = None) -> dict:
    """Structural checks; problems are reported, never silently repaired."""
    problems, warnings = [], []
    expected = {int(c["cell_id"]): c for c in manifest["cells"] if not c.get("excluded")}
    present = set(int(x) for x in data["cell_id"].unique())
    missing = sorted(set(expected) - present)
    if missing:
        problems.append(f"{len(missing)} cells absent from raw/: {missing[:20]}"
                        + (" ..." if len(missing) > 20 else ""))
    unexpected = sorted(present - set(expected))
    if unexpected:
        problems.append(f"cell_ids not in the manifest: {unexpected}")
    counts = data.groupby("cell_id").size()
    short = []
    for cid, cell in expected.items():
        want = int(n_expected if n_expected is not None else cell["n_trials"])
        got = int(counts.get(cid, 0))
        if got != want:
            short.append(f"cell {cid}: {got} trials, expected {want}")
    if short:
        problems.append(f"{len(short)} cells with the wrong trial count:\n    "
                        + "\n    ".join(short[:10]))
    dupes = data.duplicated(subset=["cell_id", "seed"], keep=False)
    if dupes.any():
        problems.append(f"{int(dupes.sum())} duplicate (cell_id, seed) rows")
    seed_sets = data.groupby("cell_id")["seed"].apply(lambda s: frozenset(s.tolist()))
    if len(set(seed_sets)) > 1:
        warnings.append("cells do not share one seed list; paired contrasts are not "
                        "valid on the full set")
    hashes = data.groupby("cell_id")["config_hash"].nunique()
    if (hashes > 1).any():
        problems.append(f"cells with more than one config_hash: "
                        f"{hashes[hashes > 1].index.tolist()}")
    n_fail = int(data["numerical_failure"].sum())
    if n_fail:
        warnings.append(f"{n_fail} numerical failures (kept as data, excluded from rates)")
    return {"problems": problems, "warnings": warnings, "n_rows": int(len(data)),
            "n_cells_present": len(present), "n_failures": n_fail}


# ---------------------------------------------------------------------------
# Per-cell summary
# ---------------------------------------------------------------------------
def _prop(mask: pd.Series, n: int, name: str) -> dict:
    k = int(mask.sum())
    lo, hi = wilson(k, n)
    return {name: (k / n if n else float("nan")), f"{name}_lo": lo, f"{name}_hi": hi,
            f"n_{name}": k}


def cell_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["cell_id", "reduction", "delta_close_deg", "q_c_rel"]
    for (cid, red, delta, qc), g in data.groupby(keys, sort=True):
        ok = g[~g["numerical_failure"]]
        n = int(len(ok))
        row = {"cell_id": int(cid), "reduction": red, "delta_close_deg": float(delta),
               "q_c_rel": float(qc), "n": n, "n_failed": int(len(g) - n)}
        row.update(_prop(ok["commit_class"] == "mean_of_pair", n, "p_mean_of_pair"))
        row.update(_prop(ok["reached"] == "C", n, "p_arrive_C"))
        row.update(_prop(ok["reached"].isin(["A", "B"]), n, "p_arrive_AB"))
        row.update(_prop(ok["timeout"], n, "p_timeout"))
        row.update(_prop(~ok["committed"], n, "p_no_commit"))
        row.update(_prop(ok["class_final"] == "mean_of_pair", n, "p_final_mean_of_pair"))
        row.update(_prop(ok["state_t2"] == "bimodal", n, "p_bimodal_t2"))
        row.update(_prop(ok["state_t2"] == "mean_of_pair", n, "p_merged_t2"))
        for label in factors.LABELS:
            row[f"p_commit_{label}"] = float((ok["commit_class"] == label).mean()) if n else float("nan")
            row[f"p_arrive_{label}"] = float((ok["reached"] == label).mean()) if n else float("nan")
        committed = ok[ok["committed"]]
        arrived = ok[ok["arrived"]]
        row["median_t_commit"] = float(committed["t_commit_ticks"].median()) if len(committed) else float("nan")
        row["median_t_arrival"] = float(arrived["t_arrival_fine"].median()) if len(arrived) else float("nan")
        row["mean_t_arrival"] = float(arrived["t_arrival_fine"].mean()) if len(arrived) else float("nan")
        bif = ok[ok["t_bif_ticks"].notna()]
        row["p_detector_fired"] = float(len(bif) / n) if n else float("nan")
        row["median_t_bif"] = float(bif["t_bif_ticks"].median()) if len(bif) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def paired_contrasts(data: pd.DataFrame) -> pd.DataFrame:
    """max - sum per (Delta_close, q_C), trials paired by seed."""
    rows = []
    ok = data[~data["numerical_failure"]]
    for (delta, qc), g in ok.groupby(["delta_close_deg", "q_c_rel"], sort=True):
        s = g[g["reduction"] == "sum"].set_index("seed")
        m = g[g["reduction"] == "max"].set_index("seed")
        common = s.index.intersection(m.index)
        if len(common) == 0:
            continue
        s, m = s.loc[common], m.loc[common]
        row = {"delta_close_deg": float(delta), "q_c_rel": float(qc), "n_pairs": int(len(common))}
        for name, ind in (("mean_of_pair", lambda d: (d["commit_class"] == "mean_of_pair")),
                          ("arrive_C", lambda d: (d["reached"] == "C")),
                          ("timeout", lambda d: d["timeout"])):
            d = ind(m).astype(float).values - ind(s).astype(float).values
            se = float(np.std(d, ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")
            row[f"d_{name}"] = float(d.mean())
            row[f"se_{name}"] = se
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6.6 sanity summary
# ---------------------------------------------------------------------------
def _bootstrap_median_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 4000, seed: int = 0) -> float:
    """Paired bootstrap standard error of median(b) - median(a)."""
    if len(a) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = np.median(b[idx], axis=1) - np.median(a[idx], axis=1)
    return float(np.std(diffs, ddof=1))


def sanity_summary(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-arm table, paired sum-vs-max table) for the 6.6 check."""
    arms, pairs = [], []
    ok = data[~data["numerical_failure"]]
    for variant, gv in ok.groupby("variant", sort=True):
        sigma = float(gv["sigma"].iloc[0])
        for red, g in gv.groupby("reduction", sort=True):
            n = int(len(g))
            arr = g[g["arrived"]]
            k = int(g["correct"].sum())
            lo, hi = wilson(k, n)
            arms.append({
                "variant": variant, "sigma": sigma, "reduction": red, "n": n,
                "acc_all": k / n if n else float("nan"), "acc_all_lo": lo, "acc_all_hi": hi,
                "acc_decided": float(arr["correct"].mean()) if len(arr) else float("nan"),
                "decided_frac": float(len(arr) / n) if n else float("nan"),
                "median_t_arrival": float(arr["t_arrival_fine"].median()) if len(arr) else float("nan"),
                "mean_t_arrival": float(arr["t_arrival_fine"].mean()) if len(arr) else float("nan"),
            })
        s = gv[gv["reduction"] == "sum"].set_index("seed")
        m = gv[gv["reduction"] == "max"].set_index("seed")
        common = s.index.intersection(m.index)
        if len(common) == 0:
            continue
        s, m = s.loc[common], m.loc[common]
        d_acc = m["correct"].astype(float).values - s["correct"].astype(float).values
        se_acc = float(np.std(d_acc, ddof=1) / math.sqrt(len(d_acc)))
        both = s["arrived"].values & m["arrived"].values
        ta, tb = s["t_arrival_fine"].values[both].astype(float), m["t_arrival_fine"].values[both].astype(float)
        d_med = float(np.median(tb) - np.median(ta)) if both.sum() else float("nan")
        se_med = _bootstrap_median_diff(ta, tb) if both.sum() else float("nan")
        pairs.append({
            "variant": variant, "sigma": sigma, "n_pairs": int(len(common)),
            "d_acc_all": float(d_acc.mean()), "se_acc_all": se_acc,
            "acc_within_1se": bool(abs(d_acc.mean()) <= se_acc) if se_acc == se_acc else None,
            "n_both_arrived": int(both.sum()),
            "d_median_t_arrival": d_med, "se_median_t_arrival": se_med,
            # A paired median shift below 1e-3 tick is below anything the log resolves
            # (it also makes the bootstrap SE collapse to 0), so it counts as "within".
            "median_within_1se": (bool(abs(d_med) <= max(se_med, 1e-3)) if se_med == se_med
                                  else None),
            "identical_choice_frac": float((s["reached"].values == m["reached"].values).mean()),
        })
    return pd.DataFrame(arms), pd.DataFrame(pairs)


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------
def aggregate(results_root: Path, manifest: dict, n_expected: int | None = None,
              allow_incomplete: bool = False) -> dict:
    data = load_raw(results_root / "raw")
    report = validate(data, manifest, n_expected=n_expected)
    for w in report["warnings"]:
        print(f"  warning: {w}")
    if report["problems"]:
        print("  PROBLEMS:")
        for p in report["problems"]:
            print(f"    - {p}")
        if not allow_incomplete:
            raise SystemExit("aggregate: structural problems (pass --allow-incomplete to write anyway)")
    cells = cell_summary(data)
    contrasts = paired_contrasts(data)
    results_root.mkdir(parents=True, exist_ok=True)
    try:
        data.drop(columns=["_source"]).to_parquet(results_root / "trials.parquet", index=False)
        trials_path = results_root / "trials.parquet"
    except Exception:                                # noqa: BLE001
        data.drop(columns=["_source"]).to_csv(results_root / "trials.csv", index=False)
        trials_path = results_root / "trials.csv"
    data.drop(columns=["_source"]).to_csv(results_root / "trials.csv", index=False)
    cells.to_csv(results_root / "cells.csv", index=False)
    contrasts.to_csv(results_root / "paired_contrasts.csv", index=False)
    report.update({"trials": str(trials_path), "cells": str(results_root / "cells.csv"),
                   "paired_contrasts": str(results_root / "paired_contrasts.csv")})
    with open(results_root / "aggregate_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"  wrote {trials_path}, {results_root / 'trials.csv'}, "
          f"{results_root / 'cells.csv'}, {results_root / 'paired_contrasts.csv'}")
    return report


def aggregate_sanity(results_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_raw(results_root / "sanity" / "raw", pattern="*_chunk*")
    arms, pairs = sanity_summary(data)
    out = results_root / "sanity"
    data.drop(columns=["_source"]).to_csv(out / "sanity_trials.csv", index=False)
    arms.to_csv(out / "sanity_arms.csv", index=False)
    pairs.to_csv(out / "sanity_paired.csv", index=False)
    print(f"  wrote {out / 'sanity_trials.csv'}, {out / 'sanity_arms.csv'}, "
          f"{out / 'sanity_paired.csv'}")
    return arms, pairs


def print_cells(cells: pd.DataFrame) -> None:
    cols = ["reduction", "delta_close_deg", "q_c_rel", "n", "p_mean_of_pair", "p_merged_t2",
            "p_bimodal_t2", "p_arrive_C", "p_arrive_AB", "p_timeout", "p_no_commit",
            "median_t_commit", "median_t_arrival"]
    with pd.option_context("display.width", 200, "display.max_rows", 100,
                           "display.float_format", "{:.3f}".format):
        print(cells[cols].to_string(index=False))
