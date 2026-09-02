#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2026 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Walk the replicate tree, emit tidy trials + per-cell summaries + gates.

    python3 scripts/qd_sweep_fixed_noise/aggregate.py --arm ra \
        --base-root <campaign root> [--manifest <csv>] [--workers N] \
        [--previous-ra <halted slices_1.0 cells.csv>]
    python3 scripts/qd_sweep_fixed_noise/aggregate.py --arm ddm \
        --base-root <campaign root> \
        [--previous-halted <halted halt_1.0 ddm_trials.parquet>]

Outputs in <base-root>:
    ra_trials.parquet / ddm_trials.parquet    one row per replicate
    ra_cells.csv / ddm_points.csv             per-cell summaries (Wilson 95 %
                                              CI on accuracy, bootstrap CI on
                                              median arrival; acc_all AND
                                              acc_decided everywhere, §9)
    missing_replicates_{ra,ddm}.csv           every (cell, run_id) absent —
                                              resubmitting fills exactly these
    u0_gate.json                (ra)   §8.3: 10-way replicate check per δ_Q +
                                       the across-δ_Q monotonicity check
    ra_continuity_gate.json     (ra)   §8.4 with --previous-ra
    diagonal_regression_gate.json (ddm) §8.2 with --previous-halted

All three gates are BLOCKING (non-zero exit) — halt and diagnose before any
analysis. Censoring (`decided_frac`, `halt_frac`) is expected data at low
actual SNR under frozen bold thresholds, and is reported, never failed on.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import qd   # noqa: E402

RA_TRIAL_FIELDS = [
    "cell_id", "v", "u", "actual_bp", "run_id",
    "env_seed_sensory", "model_seed",
    "decided", "choice", "correct",
    "t_commit_ticks", "t_commit_fine", "t_arrival_s", "timeout",
    "t_bif_ticks", "bif_target",
    "n_ticks_logged", "git_sha", "config_hash", "error",
]
DDM_TRIAL_FIELDS = [
    "point_id", "variant", "design_bp", "actual_bp", "bound_param", "run_id",
    "env_seed_sensory", "model_seed",
    "decided", "choice", "correct",
    "t_commit_ticks", "t_commit_fine", "t_arrival_s", "timeout",
    "committed", "committed_id", "commit_correct", "rt", "tick_commit",
    "halted", "halt_duration", "z_halt", "halt_guard_hits",
    "n_ticks_logged", "git_sha", "config_hash", "error",
]


# ---------------------------------------------------------------------------
# Statistics helpers (frontier conventions)
# ---------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, centre - half), min(1.0, centre + half)


def boot_median_ci(values, n_boot: int = 2000, seed: int = 0):
    import numpy as np
    arr = np.asarray([v for v in values if v is not None and v == v], float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    meds = np.median(
        arr[rng.integers(0, arr.size, size=(n_boot, arr.size))], axis=1)
    return float(np.median(arr)), float(np.percentile(meds, 2.5)), \
        float(np.percentile(meds, 97.5))


def ci_overlap(lo1, hi1, lo2, hi2) -> bool:
    return not (hi1 < lo2 or hi2 < lo1)


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# One replicate -> one row
# ---------------------------------------------------------------------------
def score_replicate(rep_dir: Path, arm: str) -> dict | None:
    meta_path = rep_dir / "run_meta.json"
    if not meta_path.is_file():
        return None
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    with open(rep_dir / "config.json", encoding="utf-8") as fh:
        cfg = json.load(fh)
    env = cfg["environment"]
    tick_rate = max(int(env.get("ticks_per_second", 1)), 1)
    radius = float(env["termination"]["radius"])
    targets = qd.target_positions(cfg)

    fields = RA_TRIAL_FIELDS if arm == "ra" else DDM_TRIAL_FIELDS
    row = {k: None for k in fields}
    for k in fields:
        if k in meta:
            row[k] = meta[k]
    row.update({"decided": False, "timeout": False, "choice": "",
                "correct": False, "error": ""})

    zips = sorted(rep_dir.glob("config_folder_*/run_*.zip"))
    if not zips:
        row["error"] = "no run archive"
        return row
    try:
        with zipfile.ZipFile(zips[0]) as zf:
            names = zf.namelist()
            pos = next(n for n in names if n.endswith("_position.csv"))
            prows = list(csv.DictReader(io.TextIOWrapper(zf.open(pos))))
            events, drows = {}, []
            if arm == "ra":
                ev = next((n for n in names if n.endswith("events.json")), None)
                if ev:
                    events = json.load(io.TextIOWrapper(zf.open(ev)))
            else:
                dm = next((n for n in names if n.endswith("_ddm.csv")), None)
                if dm:
                    drows = list(csv.DictReader(io.TextIOWrapper(zf.open(dm))))
    except Exception as exc:                        # noqa: BLE001
        row["error"] = f"unreadable archive: {exc!r}"
        return row

    row["n_ticks_logged"] = len(prows)
    tick, fine, hit = qd.first_crossing(prows, targets, radius)
    if hit is not None:
        row.update({"decided": True, "choice": hit,
                    "correct": hit == qd.CORRECT_TARGET_ID,
                    "t_commit_ticks": int(tick), "t_commit_fine": float(fine),
                    "t_arrival_s": float(fine) / tick_rate})
    else:
        row["timeout"] = True

    if arm == "ra":
        bif = (events.get("bifurcation_events") or [])
        if bif:
            first = min(bif, key=lambda e: e.get("tick", 1 << 30))
            row["t_bif_ticks"] = int(first.get("tick"))
            row["bif_target"] = str(first.get("target") or "")
    else:
        live = [r for r in drows if _f(r.get("z"))]
        first_commit = next((r for r in live
                             if (r.get("committed") or "") != ""), None)
        if first_commit is not None and live:
            lastd = live[-1]
            row.update({
                "committed": True,
                "committed_id": lastd.get("committed_id") or "",
                "commit_correct": (lastd.get("committed_id") or "")
                == qd.CORRECT_TARGET_ID,
                "rt": _f(lastd.get("rt")),
                "tick_commit": int(first_commit["tick"])})
        else:
            row["committed"] = False
        if live:
            lastd = live[-1]
            if "halt_event" in lastd:
                row["halted"] = str(lastd.get("halt_event")).strip() in (
                    "True", "true", "1")
                row["halt_duration"] = _f(lastd.get("halt_duration"))
                row["z_halt"] = _f(lastd.get("z_halt"))
                guard = _f(lastd.get("halt_guard_hits"))
                row["halt_guard_hits"] = int(guard) if guard is not None else 0
    return row


def _score_cell_dir(task) -> list[dict]:
    cell_dir, arm = task
    rows = []
    for rep in sorted(cell_dir.glob("replicate_*")):
        r = score_replicate(rep, arm)
        if r is not None:
            rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# Summaries — acc_all AND acc_decided everywhere (§9)
# ---------------------------------------------------------------------------
def summarise_cells(trials: list[dict], arm: str) -> list[dict]:
    key = "cell_id" if arm == "ra" else "point_id"
    ident_cols = (["v", "u", "actual_bp"] if arm == "ra"
                  else ["variant", "design_bp", "actual_bp", "bound_param"])
    by_cell: dict[str, list[dict]] = {}
    for t in trials:
        by_cell.setdefault(t[key], []).append(t)
    out = []
    for ident in sorted(by_cell):
        ts = by_cell[ident]
        n = len(ts)
        dec = [t for t in ts if t["decided"]]
        k_all = sum(1 for t in ts if t["correct"])
        k_dec = sum(1 for t in dec if t["correct"])
        acc_all, alo, ahi = wilson(k_all, n)
        acc_dec, dlo, dhi = wilson(k_dec, len(dec)) if dec else (
            float("nan"),) * 3
        med, mlo, mhi = boot_median_ci([t["t_arrival_s"] for t in dec])
        arr = [t["t_arrival_s"] for t in dec if t["t_arrival_s"] is not None]
        commit = sorted(t["t_commit_ticks"] for t in dec) if dec else []
        row = {key: ident,
               **{c: ts[0].get(c) for c in ident_cols},
               "n": n, "decided_frac": len(dec) / n if n else float("nan"),
               "acc_all": acc_all, "acc_all_lo": alo, "acc_all_hi": ahi,
               "acc_decided": acc_dec, "acc_decided_lo": dlo,
               "acc_decided_hi": dhi,
               "median_arrival_s": med, "median_arrival_lo": mlo,
               "median_arrival_hi": mhi,
               "mean_arrival_s": (sum(arr) / len(arr)) if arr else None,
               "median_commit_tick": (commit[len(commit) // 2]
                                      if commit else None),
               "n_errors": sum(1 for t in ts if t["error"])}
        if arm == "ddm":
            halted = [t for t in ts if t.get("halted")]
            durs = [t["halt_duration"] for t in halted
                    if t.get("halt_duration") is not None]
            row["halt_frac"] = len(halted) / n if n else float("nan")
            row["median_halt_duration_s"] = (
                sorted(durs)[len(durs) // 2] if durs else None)
            row["halt_guard_hits"] = sum(
                int(t.get("halt_guard_hits") or 0) for t in ts)
        out.append(row)
    return out


def write_table(rows: list[dict], fields: list[str], stem: Path) -> Path:
    try:
        import pandas as pd
        path = stem.with_suffix(".parquet")
        pd.DataFrame(rows, columns=fields).to_parquet(path, index=False)
        return path
    except Exception as exc:                        # noqa: BLE001
        print(f"  parquet unavailable ({exc!r}); falling back to CSV")
        path = stem.with_suffix(".csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: ("" if r.get(k) is None else r.get(k))
                            for k in fields})
        return path


def write_csv(rows: list[dict], dest: Path) -> None:
    if not rows:
        dest.write_text("", encoding="utf-8")
        return
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# §8.3 — the u = 0 gates: 10-way replicate check per δ_Q (v is inert at
# u = 0) + across-δ_Q monotonicity (the cheap proof that SNR moved).
# ---------------------------------------------------------------------------
def u0_gate(cells: list[dict], dest: Path) -> bool:
    u0 = [c for c in cells if float(c["u"]) == 0.0]
    by_actual: dict[int, list[dict]] = {}
    for c in u0:
        by_actual.setdefault(int(c["actual_bp"]), []).append(c)
    report, ok = {"within": [], "across": None}, True
    for bp in sorted(by_actual):
        group = by_actual[bp]
        pairs_ok = all(
            ci_overlap(a["acc_all_lo"], a["acc_all_hi"],
                       b["acc_all_lo"], b["acc_all_hi"])
            for i, a in enumerate(group) for b in group[i + 1:])
        ok &= pairs_ok
        report["within"].append({
            "actual_bp": bp, "n_cells": len(group),
            "all_pairs_ci_overlap": pairs_ok,
            "cells": [{"cell_id": c["cell_id"], "acc_all": c["acc_all"],
                       "ci": [c["acc_all_lo"], c["acc_all_hi"]]}
                      for c in group]})
        verdict = ("PASS" if pairs_ok
                   else "FAIL — v leaks into the environment or seeding")
        print(f"  u = 0 within actual {bp} bp ({len(group)} cells): {verdict}")
    pooled = []
    for bp in sorted(by_actual):
        group = by_actual[bp]
        n = sum(c["n"] for c in group)
        k = sum(round(c["acc_all"] * c["n"]) for c in group)
        p, lo, hi = wilson(int(k), int(n))
        pooled.append({"actual_bp": bp, "n": n, "acc": p, "lo": lo, "hi": hi})
    increasing = all(a["acc"] < b["acc"]
                     for a, b in zip(pooled, pooled[1:]))
    separated = all(a["hi"] < b["lo"] for a, b in zip(pooled, pooled[1:]))
    mono_ok = increasing and separated
    if len(pooled) >= 2:
        ok &= mono_ok
        report["across"] = {"pooled": pooled, "strictly_increasing": increasing,
                            "ci_separated": separated, "passed": mono_ok}
        verdict = ("PASS" if mono_ok
                   else "FAIL — SNR did not move with δ_Q (the coupling?)")
        print("  u = 0 across δ_Q: "
              + " < ".join(f"{p['acc']:.3f}@{p['actual_bp']}bp"
                           for p in pooled)
              + f"  {verdict}")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump({"gate": "PASS" if ok else "FAIL", **report}, fh, indent=2)
    return ok


# ---------------------------------------------------------------------------
# §8.4 — RA continuity: (v = 0.5, u ∈ {4, 6, 8}) at actual = 100 bp within
# CI-overlap of the halted campaign's corresponding absolute-sweep cells
# (same calibration at 1 %, same frontier-v1 env seeds).
# ---------------------------------------------------------------------------
CONTINUITY_U = [4.0, 6.0, 8.0]


def continuity_gate(cells: list[dict], previous_csv: Path, dest: Path) -> bool:
    prev = qd.read_manifest(previous_csv)
    report, ok = [], True
    for u in CONTINUITY_U:
        mine = next((c for c in cells
                     if int(c["actual_bp"]) == 100
                     and float(c["v"]) == 0.5 and float(c["u"]) == u), None)
        old = next((c for c in prev
                    if c.get("sweep") == "absolute"
                    and float(c["v"]) == 0.5 and float(c["u"]) == u), None)
        if mine is None or old is None:
            report.append({"u": u, "status": "MISSING "
                           + ("new cell" if mine is None else "reference")})
            ok = False
            continue
        acc_ok = ci_overlap(mine["acc_all_lo"], mine["acc_all_hi"],
                            float(old["acc_all_lo"]), float(old["acc_all_hi"]))
        med_ok = ci_overlap(mine["median_arrival_lo"],
                            mine["median_arrival_hi"],
                            float(old["median_arrival_lo"]),
                            float(old["median_arrival_hi"]))
        ok &= acc_ok and med_ok
        report.append({
            "u": u, "status": "PASS" if (acc_ok and med_ok) else "FAIL",
            "new_cell": mine["cell_id"], "ref_cell": old["cell_id"],
            "new_acc": mine["acc_all"],
            "new_acc_ci": [mine["acc_all_lo"], mine["acc_all_hi"]],
            "ref_acc": float(old["acc_all"]),
            "ref_acc_ci": [float(old["acc_all_lo"]), float(old["acc_all_hi"])],
            "new_median_s": mine["median_arrival_s"],
            "ref_median_s": float(old["median_arrival_s"]),
            "acc_ci_overlap": acc_ok, "median_ci_overlap": med_ok})
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump({"gate": "PASS" if ok else "FAIL",
                   "previous": str(previous_csv), "points": report}, fh,
                  indent=2)
    print(f"\nRA continuity gate (§8.4) vs {previous_csv}:")
    for r in report:
        if "new_acc" in r:
            print(f"  u = {r['u']:g}: {r['status']}  acc {r['new_acc']:.3f} "
                  f"vs {r['ref_acc']:.3f}  median {r['new_median_s']:.1f}s "
                  f"vs {r['ref_median_s']:.1f}s")
        else:
            print(f"  u = {r['u']:g}: {r['status']}")
    verdict = ("PASS" if ok
               else "FAIL — template or seeding drift; halt before analysis")
    print(f"  gate: {verdict}")
    return ok


# ---------------------------------------------------------------------------
# §8.2 — 1 %-diagonal regression: design = actual = 100 bp IS the halted
# campaign's condition (same calibration, same frontier-v1 env seeds) —
# Bellman FROZEN_CE and both static b* points must reproduce within CIs.
# ---------------------------------------------------------------------------
def diagonal_gate(points: list[dict], previous_parquet: Path,
                  dest: Path) -> bool:
    import pandas as pd
    prev = pd.read_parquet(previous_parquet)
    report, ok = [], True
    diag = [p for p in points
            if int(p["design_bp"]) == 100 and int(p["actual_bp"]) == 100]
    for row in diag:
        if row["variant"] == "bellman":
            sl = prev[(prev["variant"] == "bellman")
                      & (prev["bound"].astype(float)
                         == float(row["bound_param"]))]
        else:
            sl = prev[(prev["variant"] == "static")
                      & (prev["bound"].astype(float)
                         == float(row["bound_param"]))]
        if sl.empty:
            report.append({"point_id": row["point_id"],
                           "status": "NO REFERENCE DATA"})
            ok = False
            continue
        n_prev = len(sl)
        p_acc, p_lo, p_hi = wilson(int(sl["correct"].astype(bool).sum()),
                                   n_prev)
        p_arr = sl.loc[sl["decided"].astype(bool), "t_arrival_s"]
        p_med, p_mlo, p_mhi = boot_median_ci(p_arr.astype(float).tolist())
        acc_ok = ci_overlap(row["acc_all_lo"], row["acc_all_hi"], p_lo, p_hi)
        med_ok = ci_overlap(row["median_arrival_lo"],
                            row["median_arrival_hi"], p_mlo, p_mhi)
        ok &= acc_ok and med_ok
        report.append({
            "point_id": row["point_id"], "variant": row["variant"],
            "bound_param": row["bound_param"],
            "status": "PASS" if (acc_ok and med_ok) else "FAIL",
            "acc": row["acc_all"],
            "acc_ci": [row["acc_all_lo"], row["acc_all_hi"]],
            "ref_acc": p_acc, "ref_acc_ci": [p_lo, p_hi],
            "median_s": row["median_arrival_s"],
            "median_ci": [row["median_arrival_lo"],
                          row["median_arrival_hi"]],
            "ref_median_s": p_med, "ref_median_ci": [p_mlo, p_mhi],
            "halt_frac": row.get("halt_frac"),
            "n": row["n"], "n_ref": n_prev})
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump({"gate": "PASS" if ok else "FAIL",
                   "note": ("BLOCKING (§8.2): same condition, same seeds as "
                            "the halted campaign's 1 % families — "
                            "disagreement is template/noise drift."),
                   "previous": str(previous_parquet), "points": report}, fh,
                  indent=2)
    print(f"\n1 %-diagonal regression gate (§8.2) vs {previous_parquet}:")
    for r in report:
        if "acc" in r:
            print(f"  {r['point_id']:>18s} [{r['variant']:>11s}] "
                  f"{r['status']}  acc {r['acc']:.3f} vs {r['ref_acc']:.3f}  "
                  f"median {r['median_s']:.2f}s vs {r['ref_median_s']:.2f}s")
        else:
            print(f"  {r['point_id']:>18s} {r['status']}")
    print(f"  gate: {'PASS' if ok else 'FAIL — halt and diagnose (§8.2)'}")
    return ok


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", choices=("ra", "ddm"), required=True)
    ap.add_argument("--base-root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--previous-ra", type=Path, default=None,
                    help="ra: the halted campaign's 1 %% cells.csv "
                         "(§8.4 continuity gate)")
    ap.add_argument("--previous-halted", type=Path, default=None,
                    help="ddm: the halted campaign's 1 %% ddm_trials.parquet "
                         "(§8.2 diagonal regression gate)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)

    root = args.base_root
    manifest = args.manifest or (root / (qd.RA_MANIFEST_NAME
                                         if args.arm == "ra"
                                         else qd.DDM_MANIFEST_NAME))
    rows_manifest = qd.read_manifest(manifest) if manifest.is_file() else []
    if not rows_manifest:
        print(f"WARNING: no manifest at {manifest}; completeness not "
              "checkable")

    tree = root / args.arm
    if args.arm == "ra":
        cell_dirs = sorted(tree.glob("actual_*/v_*/u_*"))
    else:
        cell_dirs = sorted(tree.glob("actual_*/design_*/*"))
    tasks = [(d, args.arm) for d in cell_dirs if d.is_dir()]
    trials: list[dict] = []
    if args.workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for chunk in ex.map(_score_cell_dir, tasks):
                trials.extend(chunk)
    else:
        for t in tasks:
            trials.extend(_score_cell_dir(t))
    print(f"scored {len(trials)} replicates from {len(tasks)} cell "
          f"directories under {tree}")

    fields = RA_TRIAL_FIELDS if args.arm == "ra" else DDM_TRIAL_FIELDS
    stem = root / ("ra_trials" if args.arm == "ra" else "ddm_trials")
    tpath = write_table(trials, fields, stem)
    print(f"wrote {tpath}")

    cells = summarise_cells(trials, args.arm)
    cpath = root / ("ra_cells.csv" if args.arm == "ra" else "ddm_points.csv")
    write_csv(cells, cpath)
    print(f"wrote {cpath} ({len(cells)} cells)")

    # ---- completeness vs the manifest -------------------------------------
    missing = []
    if rows_manifest:
        key = "cell_id" if args.arm == "ra" else "point_id"
        have: dict[str, set[int]] = {}
        for t in trials:
            have.setdefault(t[key], set()).add(int(t["run_id"]))
        for row in rows_manifest:
            got = have.get(row[key], set())
            for rid in range(1, int(row["n_runs"]) + 1):
                if rid not in got:
                    missing.append({key: row[key], "run_id": rid})
        write_csv(missing, root / f"missing_replicates_{args.arm}.csv")
        print(f"completeness vs {manifest.name}: {len(missing)} missing "
              f"replicates" + ("" if missing else " — COMPLETE"))

    # ---- gates -------------------------------------------------------------
    ok = True
    if args.arm == "ra":
        print("\nu = 0 gates (§8.3):")
        ok &= u0_gate(cells, root / "u0_gate.json")
        if args.previous_ra is not None:
            ok &= continuity_gate(cells, args.previous_ra,
                                  root / "ra_continuity_gate.json")
    else:
        guard_hits = sum(int(c.get("halt_guard_hits") or 0) for c in cells)
        if guard_hits:
            print(f"NOTE: {guard_hits} halt-guard hits across the campaign "
                  "(frozen controllers under actual < design can out-wait "
                  "their design guard) — carried per point in ddm_points.csv")
        if args.previous_halted is not None:
            ok &= diagonal_gate(cells, args.previous_halted,
                                root / "diagonal_regression_gate.json")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
