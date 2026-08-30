#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2026 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""§9: walk the replicate tree, emit tidy trials + per-cell summaries.

    python3 scripts/ra_ddm_frontier/aggregate.py --campaign ra \
        --base-root <dir> [--manifest <csv>] [--workers N]
    python3 scripts/ra_ddm_frontier/aggregate.py --campaign ddm \
        --base-root <dir> [--previous <tidy_trials.parquet>]

Outputs in <base-root>:
    trials.parquet / ddm_trials.parquet   one row per replicate
    cells.csv / ddm_points.csv            per-cell summaries (Wilson 95 % CI on
                                          accuracy, bootstrap CI on median
                                          arrival seconds)
    missing_replicates.csv                every (cell, run_id) absent from the
                                          tree — resubmitting the campaign
                                          script fills exactly these
    regression_gate.json                  (ddm, with --previous) rerun vs the
                                          archived q01_a60 frontier, per c_e

Regression gate (§9): the rerun must agree with the previous DDM frontier
within CIs at every c_e — disagreement means environment or template drift;
halt and diagnose before any overlay.
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
_ROOT = _HERE.parents[1]
for _p in (str(_HERE), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import frontier   # noqa: E402

RA_TRIAL_FIELDS = [
    "cell_id", "sweep", "v", "u_hat", "u_star", "u", "run_id",
    "env_seed_sensory", "model_seed",
    "decided", "choice", "correct",
    "t_commit_ticks", "t_commit_fine", "t_arrival_s", "timeout",
    "t_bif_ticks", "bif_target",
    "n_ticks_logged", "git_sha", "config_hash", "error",
]
DDM_TRIAL_FIELDS = [
    "point_id", "variant", "bound", "c_e", "run_id",
    "env_seed_sensory", "model_seed",
    "decided", "choice", "correct",
    "t_commit_ticks", "t_commit_fine", "t_arrival_s", "timeout",
    "committed", "committed_id", "commit_correct", "rt", "tick_commit",
    # §2b halt-at-midpoint policy: halted = the trial reached the midpoint
    # undecided; halt_duration = extra deliberation bought by the halt (s).
    "halted", "halt_duration", "z_halt", "halt_guard_hits",
    "n_ticks_logged", "git_sha", "config_hash", "error",
]


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float, float]:
    """(p, lo, hi): Wilson 95 % interval."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, centre - half), min(1.0, centre + half)


def boot_median_ci(values, n_boot: int = 2000, seed: int = 0):
    """(median, lo, hi) — percentile bootstrap on the median."""
    import numpy as np
    arr = np.asarray([v for v in values if v == v and v is not None], float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    meds = np.median(
        arr[rng.integers(0, arr.size, size=(n_boot, arr.size))], axis=1)
    return float(np.median(arr)), float(np.percentile(meds, 2.5)), \
        float(np.percentile(meds, 97.5))


def ci_overlap(lo1, hi1, lo2, hi2) -> bool:
    return not (hi1 < lo2 or hi2 < lo1)


# ---------------------------------------------------------------------------
# One replicate -> one row
# ---------------------------------------------------------------------------
def score_replicate(rep_dir: Path, campaign: str) -> dict | None:
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
    targets = frontier.target_positions(cfg)

    fields = RA_TRIAL_FIELDS if campaign == "ra" else DDM_TRIAL_FIELDS
    row = {k: None for k in fields}
    for k in fields:
        if k in meta:
            row[k] = meta[k]
    if campaign == "ddm":
        # Halt-campaign metadata carries (variant, bound); frozen wave-1/2
        # metadata carries c_e only (always the Bellman family). Normalize so
        # every trial has all three columns.
        if row.get("variant") is None:
            row["variant"] = "bellman"
        if row.get("bound") is None and row.get("c_e") is not None:
            row["bound"] = row["c_e"]
        if row.get("c_e") is None and row["variant"] == "bellman":
            row["c_e"] = row.get("bound")
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
            if campaign == "ra":
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
    tick, fine, hit = frontier.first_crossing(prows, targets, radius)
    if hit is not None:
        row.update({"decided": True, "choice": hit,
                    "correct": hit == frontier.CORRECT_TARGET_ID,
                    "t_commit_ticks": int(tick), "t_commit_fine": float(fine),
                    "t_arrival_s": float(fine) / tick_rate})
    else:
        row["timeout"] = True

    if campaign == "ra":
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
                == frontier.CORRECT_TARGET_ID,
                "rt": _f(lastd.get("rt")),
                "tick_commit": int(first_commit["tick"])})
        else:
            row["committed"] = False
        if live:
            # §2b halt logging: halted (reached the midpoint undecided),
            # halt duration, z_halt, runaway-guard hits — read off the last
            # live tick (the fields latch once set). Absent columns (frozen
            # forced-choice logs) leave the Nones from the field init.
            lastd = live[-1]
            if "halt_event" in lastd:
                row["halted"] = str(lastd.get("halt_event")).strip() in (
                    "True", "true", "1")
                row["halt_duration"] = _f(lastd.get("halt_duration"))
                row["z_halt"] = _f(lastd.get("z_halt"))
                guard = _f(lastd.get("halt_guard_hits"))
                row["halt_guard_hits"] = int(guard) if guard is not None else 0
    return row


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _score_cell_dir(task) -> list[dict]:
    cell_dir, campaign = task
    rows = []
    for rep in sorted(cell_dir.glob("replicate_*")):
        r = score_replicate(rep, campaign)
        if r is not None:
            rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def summarise_cells(trials: list[dict], campaign: str) -> list[dict]:
    key = "cell_id" if campaign == "ra" else "point_id"
    ident_cols = (["sweep", "v", "u_hat", "u_star", "u"] if campaign == "ra"
                  else ["variant", "bound", "c_e"])
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
        commit = sorted(t["t_commit_ticks"] for t in dec) if dec else []
        row = {key: ident,
               **{c: ts[0].get(c) for c in ident_cols},
               "n": n, "decided_frac": len(dec) / n if n else float("nan"),
               "acc_all": acc_all, "acc_all_lo": alo, "acc_all_hi": ahi,
               "acc_decided": acc_dec, "acc_decided_lo": dlo,
               "acc_decided_hi": dhi,
               "median_arrival_s": med, "median_arrival_lo": mlo,
               "median_arrival_hi": mhi,
               "median_commit_tick": (commit[len(commit) // 2]
                                      if commit else None),
               "n_errors": sum(1 for t in ts if t["error"])}
        if campaign == "ddm":
            # §2b: the halt makes the DT distribution bimodal by construction,
            # so the halt fraction is reported per point; a trial still halted
            # at time_limit never arrives and shows up in 1 - decided_frac.
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
# Regression gate (§9, DDM only)
# ---------------------------------------------------------------------------
def regression_gate(points: list[dict], trials: list[dict], previous: Path,
                    dest: Path) -> bool:
    """CI-overlap per c_e vs the archived frontier. The archive's arrival_time
    is the INTEGER termination tick / tick_rate, so the rerun side uses
    t_commit_ticks (not the sub-tick-refined t_arrival_s) — like for like."""
    import pandas as pd
    prev = pd.read_parquet(previous)
    prev = prev[(prev["arm"] == "main")
                & (prev["condition"].str.match(r"q01_a60_ce"))]
    by_point: dict[str, list[dict]] = {}
    for t in trials:
        by_point.setdefault(t["point_id"], []).append(t)
    report, ok_all = [], True
    for row in points:
        if row.get("c_e") in (None, ""):        # static family: no archive twin
            continue
        ce = float(row["c_e"])
        sl = prev[prev["c_e"].astype(float) == ce]
        if sl.empty:
            report.append({"c_e": ce, "status": "NO PREVIOUS DATA"})
            ok_all = False
            continue
        arrived = sl[sl["arrived"].astype(bool)]
        n_prev = len(sl)
        k_prev = int(sl["arrival_correct"].astype(float).sum())
        p_acc, p_lo, p_hi = wilson(k_prev, n_prev)       # censored scored 0
        p_med, p_mlo, p_mhi = boot_median_ci(
            arrived["arrival_time"].astype(float).tolist())
        tick_times = [t["t_commit_ticks"] / frontier.DDM_TICKS_PER_SECOND
                      for t in by_point.get(row["point_id"], [])
                      if t["decided"] and t["t_commit_ticks"] is not None]
        r_med, r_mlo, r_mhi = boot_median_ci(tick_times)
        acc_ok = ci_overlap(row["acc_all_lo"], row["acc_all_hi"], p_lo, p_hi)
        med_ok = ci_overlap(r_mlo, r_mhi, p_mlo, p_mhi)
        ok_all &= acc_ok and med_ok
        report.append({
            "c_e": ce, "status": "PASS" if (acc_ok and med_ok) else "FAIL",
            "rerun_acc_all": row["acc_all"],
            "rerun_acc_ci": [row["acc_all_lo"], row["acc_all_hi"]],
            "prev_acc_all": p_acc, "prev_acc_ci": [p_lo, p_hi],
            "rerun_median_s": r_med,
            "rerun_median_ci": [r_mlo, r_mhi],
            "prev_median_s": p_med, "prev_median_ci": [p_mlo, p_mhi],
            "acc_ci_overlap": acc_ok, "median_ci_overlap": med_ok,
            "n_rerun": row["n"], "n_prev": n_prev})
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump({"gate": "PASS" if ok_all else "FAIL",
                   "note": ("INFORMATIONAL: the archived frontier ran at "
                            "white_rate 0.035; this rerun is calibrated at "
                            f"{frontier.WHITE_RATE} (evidence-channel c = 2 x dQ), so "
                            "systematic differences are expected (RECON "
                            "D-09/D-10)."),
                   "previous": str(previous), "points": report}, fh, indent=2)
    print(f"\nDDM comparison vs the archived frontier {previous}")
    print(f"  INFORMATIONAL — archive at white_rate 0.035, rerun at "
          f"{frontier.WHITE_RATE} (evidence-channel c = 2 x dQ): systematic differences "
          "are EXPECTED, this is no longer a drift gate (RECON D-10):")
    for r in report:
        if "rerun_acc_all" in r:
            print(f"  c_e={r['c_e']:>7g}  {r['status']}  "
                  f"acc {r['rerun_acc_all']:.3f} vs {r['prev_acc_all']:.3f}  "
                  f"median {r['rerun_median_s']:.2f}s vs {r['prev_median_s']:.2f}s")
        else:
            print(f"  c_e={r['c_e']:>7g}  {r['status']}")
    print(f"  CI overlap at {sum(1 for r in report if r.get('status') == 'PASS')}"
          f"/{len(report)} points (informational)")
    return True                       # never blocks: calibrations differ


def halt_regression_gate(points: list[dict], trials: list[dict],
                         previous: Path, dest: Path) -> bool:
    """§9, re-scoped for the halt-at-midpoint policy: the halt rerun vs the
    FROZEN forced-choice rerun (same calibration, same frontier-v1 env seeds).

    Where the halt never triggers (halt_frac ≈ 0) behavior must reproduce —
    CI overlap on acc_all and median arrival is BLOCKING; disagreement there
    means environment or template drift. Where the halt does trigger, the
    expected signature of the policy fix is slower + more accurate — reported,
    never failed on. Bellman family only (the static family has no
    forced-choice twin)."""
    import pandas as pd
    prev = pd.read_parquet(previous)
    HALT_FREE = 0.005
    report, ok_all = [], True
    for row in points:
        if row.get("variant") != "bellman" or row.get("c_e") in (None, ""):
            continue
        ce = float(row["c_e"])
        sl = prev[prev["c_e"].astype(float) == ce]
        if sl.empty:
            report.append({"c_e": ce, "status": "NO PREVIOUS DATA"})
            continue
        n_prev = len(sl)
        p_acc, p_lo, p_hi = wilson(int(sl["correct"].astype(bool).sum()),
                                   n_prev)
        p_arr = sl.loc[sl["decided"].astype(bool), "t_arrival_s"]
        p_med, p_mlo, p_mhi = boot_median_ci(p_arr.astype(float).tolist())
        halt_frac = float(row.get("halt_frac") or 0.0)
        gated = halt_frac <= HALT_FREE
        acc_ok = ci_overlap(row["acc_all_lo"], row["acc_all_hi"], p_lo, p_hi)
        med_ok = ci_overlap(row["median_arrival_lo"], row["median_arrival_hi"],
                            p_mlo, p_mhi)
        if gated:
            status = "PASS" if (acc_ok and med_ok) else "FAIL (zero-halt drift)"
            ok_all &= acc_ok and med_ok
        else:
            expected = (row["acc_all"] >= p_acc - 1e-9
                        and row["median_arrival_s"] >= p_med - 1e-9)
            status = ("EXPECTED DEPARTURE (slower, more accurate)" if expected
                      else "DEPARTURE IN THE WRONG DIRECTION — inspect")
        report.append({
            "c_e": ce, "halt_frac": halt_frac, "gated": gated,
            "status": status,
            "halt_acc": row["acc_all"],
            "halt_acc_ci": [row["acc_all_lo"], row["acc_all_hi"]],
            "prev_acc": p_acc, "prev_acc_ci": [p_lo, p_hi],
            "halt_median_s": row["median_arrival_s"],
            "halt_median_ci": [row["median_arrival_lo"],
                               row["median_arrival_hi"]],
            "prev_median_s": p_med, "prev_median_ci": [p_mlo, p_mhi],
            "n_halt": row["n"], "n_prev": n_prev})
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump({"gate": "PASS" if ok_all else "FAIL",
                   "halt_free_threshold": HALT_FREE,
                   "note": ("BLOCKING at zero-halt points: same calibration, "
                            "same frontier-v1 seeds — behavior must "
                            "reproduce. At halted points the departure "
                            "(slower, more accurate) is the policy fix's "
                            "expected signature, not drift (§9)."),
                   "previous": str(previous), "points": report}, fh, indent=2)
    print(f"\nhalt-policy regression gate vs the forced-choice rerun "
          f"{previous}:")
    for r in report:
        if "halt_acc" in r:
            print(f"  c_e={r['c_e']:>7g}  halt_frac={r['halt_frac']:.3f}  "
                  f"{r['status']}  acc {r['halt_acc']:.3f} vs "
                  f"{r['prev_acc']:.3f}  median {r['halt_median_s']:.2f}s vs "
                  f"{r['prev_median_s']:.2f}s")
        else:
            print(f"  c_e={r['c_e']:>7g}  {r['status']}")
    verdict = ("PASS" if ok_all
               else "FAIL — halt and diagnose before any overlay (§9)")
    print(f"  gate: {verdict}")
    return ok_all


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--campaign", choices=("ra", "ddm"), required=True)
    ap.add_argument("--base-root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--previous", type=Path, default=None,
                    help="ddm: the ARCHIVED (white_rate 0.035) frontier "
                         "tidy_trials.parquet — informational comparison only")
    ap.add_argument("--previous-rerun", type=Path, default=None,
                    help="ddm halt campaign: the frozen forced-choice rerun's "
                         "ddm_trials.parquet (same calibration, same seeds). "
                         "§9 BLOCKING gate at every Bellman c_e whose halt "
                         "fraction ≈ 0; a documented departure (slower, more "
                         "accurate) is expected where the halt triggers")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)

    root = args.base_root
    if args.manifest is not None:
        manifest = args.manifest
    else:
        # §9: completeness is judged against the FULL manifest (all waves)
        # when it exists; otherwise the campaign's own manifest — for the DDM
        # that is the halt manifest first, then the frozen wave-1/2 name.
        candidates = (["manifest_full.csv", frontier.RA_MANIFEST_NAME]
                      if args.campaign == "ra" else
                      [frontier.DDM_HALT_MANIFEST_NAME,
                       "ddm_manifest_full.csv", frontier.DDM_MANIFEST_NAME])
        manifest = next((root / c for c in candidates
                         if (root / c).is_file()), root / candidates[0])
    rows_manifest = frontier.read_manifest(manifest) if manifest.is_file() else []
    if not rows_manifest:
        print(f"WARNING: no manifest at {manifest}; completeness not checkable")

    tree = root / ("cells" if args.campaign == "ra" else "points")
    if args.campaign == "ra":
        cell_dirs = sorted(d for d in tree.glob("**/")
                           if d.name.startswith("u_"))
    else:
        # Both layouts: the frozen wave-1/2 flat tree (points/ce_*) and the
        # halt campaign's variant tree (points/<variant>/{ce_*,b_*}).
        cell_dirs = sorted(list(tree.glob("ce_*"))
                           + list(tree.glob("*/ce_*"))
                           + list(tree.glob("*/b_*")))
    tasks = [(d, args.campaign) for d in cell_dirs]
    trials: list[dict] = []
    if args.workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for chunk in ex.map(_score_cell_dir, tasks):
                trials.extend(chunk)
    else:
        for t in tasks:
            trials.extend(_score_cell_dir(t))
    print(f"scored {len(trials)} replicates from {len(cell_dirs)} "
          f"cell directories under {tree}")

    fields = RA_TRIAL_FIELDS if args.campaign == "ra" else DDM_TRIAL_FIELDS
    stem = root / ("trials" if args.campaign == "ra" else "ddm_trials")
    tpath = write_table(trials, fields, stem)
    print(f"wrote {tpath}")

    cells = summarise_cells(trials, args.campaign)
    cpath = root / ("cells.csv" if args.campaign == "ra" else "ddm_points.csv")
    write_csv(cells, cpath)
    print(f"wrote {cpath} ({len(cells)} cells)")

    # ---- completeness vs the manifest (§9) --------------------------------
    missing = []
    if rows_manifest:
        key = "cell_id" if args.campaign == "ra" else "point_id"
        have: dict[str, set[int]] = {}
        for t in trials:
            have.setdefault(t[key], set()).add(int(t["run_id"]))
        for row in rows_manifest:
            got = have.get(row[key], set())
            for rid in range(1, int(row["n_runs"]) + 1):
                if rid not in got:
                    missing.append({key: row[key], "run_id": rid})
        write_csv(missing, root / "missing_replicates.csv")
        print(f"completeness: {len(missing)} missing replicates "
              f"(listed in missing_replicates.csv)"
              + ("" if missing else " — COMPLETE"))

    # ---- campaign-specific gates ------------------------------------------
    ok = True
    if args.campaign == "ra":
        # §12: v is inert at u = 0, so ALL u = 0 cells (four in wave 1, six
        # once U-v3 adds the v ∈ {0.6, 0.8} controls) are pure replicates.
        u0 = [c for c in cells if c["sweep"] == "absolute" and
              float(c["u"]) == 0.0]
        if len(u0) >= 2:
            base = u0[0]
            mutual = all(
                ci_overlap(base["acc_all_lo"], base["acc_all_hi"],
                           c["acc_all_lo"], c["acc_all_hi"]) for c in u0[1:])
            verdict = ("PASS" if mutual else
                       "FAIL — v leaks into the environment or seeding (§12)")
            print(f"\nu = 0 replicate gate ({len(u0)} cells): {verdict}")
            for c in u0:
                print(f"  {c['cell_id']}: acc_all {c['acc_all']:.3f} "
                      f"[{c['acc_all_lo']:.3f}, {c['acc_all_hi']:.3f}] "
                      f"decided {c['decided_frac']:.3f}")
            ok &= mutual
        anchor = next((c for c in cells if c["cell_id"] == "R_v0.5_h1"), None)
        if anchor is not None:
            print(f"\nfactorial-anchor cell (v=0.5, û=1.00): acc_all "
                  f"{anchor['acc_all']:.3f} [{anchor['acc_all_lo']:.3f}, "
                  f"{anchor['acc_all_hi']:.3f}], median commit "
                  f"{anchor['median_commit_tick']} ticks. INFORMATIONAL: the "
                  "factorial's 0.765 / 11 ticks was measured at white_rate "
                  f"0.035; this campaign runs at {frontier.WHITE_RATE} "
                  "(evidence-channel c = 2 x dQ), so lower accuracy is expected "
                  "(RECON D-10).")
    else:
        if args.previous is not None:
            ok &= regression_gate(cells, trials, args.previous,
                                  root / "regression_gate.json")
        if args.previous_rerun is not None:
            ok &= halt_regression_gate(cells, trials, args.previous_rerun,
                                       root / "halt_regression_gate.json")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
