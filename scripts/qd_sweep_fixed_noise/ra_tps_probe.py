#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2026 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""SIDE EXPERIMENT (own tree): is the RING ATTRACTOR clock-robust?

    python3 scripts/qd_sweep_fixed_noise/ra_tps_probe.py e0
    python3 scripts/qd_sweep_fixed_noise/ra_tps_probe.py run \
        [--tps 1,10,50] [--runs 200] [--out DIR] [--tree <cluster root>]
    python3 scripts/qd_sweep_fixed_noise/ra_tps_probe.py analyze [--out DIR]

Companion to `tps_dynamics.py` (the DDM probe). Confined to v = 0.5 at
actual δ_Q = 1 %: u ∈ {0 (uncoupled control), 4 (the measured accuracy
peak), 5 (sub-critical), 6.156868 (= u*(0.5), critical), 8 (super-critical
WTA)}. On-grid cells inherit the campaign's cluster replicate configs
verbatim (seeds included); the u* cell is synthesized through the campaign
patcher with the same frontier-v1 seeds (env seeds are cell-independent).

The clock patch, per tick rate T:
  - ticks_per_second = T on arena AND agent (they alias), snapshots = T;
  - mean_field_model.integration_time = 50/T — the ring integrates 50
    ring-units per WORLD SECOND at every T (the invariant); integration_dt
    stays 0.1, so Euler steps per tick = 500/T;
  - everything else byte-identical (time_limit is in seconds).

What T changes by construction: the shared stream hands ONE percept per
tick (per-draw SD = white_rate·√T, per-second power invariant), the ring's
input hold shortens, and the agent STEERS once per tick (120°/s = one 120°
turn at 1 Hz vs 2.4° increments at 50 Hz) — an embodied loop is inherently
tick-granular, so exact invariance is impossible; the question is how big
the departure is and where. `e0` isolates the deterministic
actuation+integration part (noise off); `run`/`analyze` measure the full
stochastic effect. The T = 1 arm must REPRODUCE the cluster campaign
bit-for-bit (checked in analyze). Comparisons across T are distributional
(draws are tick-keyed).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import zipfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import qd                               # noqa: E402
from run_batch import InProcessRunner   # noqa: E402
from aggregate import wilson, boot_median_ci   # noqa: E402

DEFAULT_TREE = (qd._ROOT.parent / "seoul-data" / "beta-1"
                / "ra_ddm_qd_sweep_fixed_noise" / "qd_sweep_fixed_noise")
DEFAULT_OUT = qd._ROOT / "results" / "ra_dynamics_tps"
V = 0.5
ACTUAL = 100
U_ONGRID = ["0", "4", "5", "8"]          # cluster directory names (u_<x>)
U_STAR = 6.156868                        # synthesized cell — no cluster twin
RING_UNITS_PER_S = 50.0                  # integration_time at the 1 Hz frame


def _u_label(u) -> str:
    return f"u_{u:g}" if isinstance(u, float) else f"u_{u}"


def _patch_clock(cfg: dict, tps: int) -> dict:
    env = cfg["environment"]
    env["ticks_per_second"] = int(tps)
    env["agents"]["movable_0"]["ticks_per_second"] = int(tps)  # they alias
    env.setdefault("results", {})["snapshots_per_second"] = int(tps)
    mf = env["agents"]["movable_0"]["mean_field_model"]
    mf["integration_time"] = RING_UNITS_PER_S / float(tps)
    return cfg


def _source_config(tree: Path, u, run_id: int) -> dict:
    """On-grid: the cluster replicate config verbatim. u*: synthesized via
    the campaign patcher + frontier-v1 seeds (same env stream as every cell)."""
    if isinstance(u, str):
        p = (tree / "ra" / f"actual_{ACTUAL}" / f"v_{V:g}" / f"u_{u}"
             / f"replicate_{run_id}" / "config.json")
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    cfg = qd.patch_ra(qd.load_template("ra"), float(u), V, ACTUAL)
    qd.apply_seeds(cfg, "ra", ACTUAL, run_id)
    return cfg


def _run_one(runner, cfg: dict, rep: Path) -> None:
    rep.mkdir(parents=True, exist_ok=True)
    cfg["environment"]["results"]["base_path"] = str(rep)
    cfg_path = rep / "config.json"
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    runner.run(cfg_path)
    if not list(rep.glob("config_folder_*/run_*.zip")):
        raise RuntimeError("no run archive")
    (rep / ".done").touch()


def _score(rep: Path):
    z = next(rep.glob("config_folder_*/run_*.zip"), None)
    if z is None:
        return None
    with open(rep / "config.json", encoding="utf-8") as fh:
        cfg = json.load(fh)
    tps = int(cfg["environment"]["ticks_per_second"])
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
        pos = next(n for n in names if n.endswith("_position.csv"))
        prows = list(csv.DictReader(io.TextIOWrapper(zf.open(pos))))
        ev = next((n for n in names if n.endswith("events.json")), None)
        events = json.load(io.TextIOWrapper(zf.open(ev))) if ev else {}
    tick, fine, hit = qd.first_crossing(
        prows, qd.target_positions(cfg),
        float(cfg["environment"]["termination"]["radius"]))
    bif = events.get("bifurcation_events") or []
    t_bif = (min(e.get("tick", 1 << 30) for e in bif) / tps) if bif else None
    return {"tps": tps, "decided": hit is not None, "choice": hit or "",
            "correct": hit == qd.CORRECT_TARGET_ID,
            "t_arrival_s": None if fine is None else float(fine) / tps,
            "t_bif_s": t_bif, "n_ticks": len(prows)}


def e0(args) -> int:
    """Deterministic clock check: noise OFF, one coupled cell, all T.
    Isolates the actuation + integration clock effect. Same choice is
    REQUIRED; the arrival spread across T is the reported granularity floor."""
    runner = InProcessRunner()
    out = args.out / "e0"
    print("E0 — deterministic (white_rate = 0) at v = 0.5, u = 6.156868:")
    rows = []
    for tps in args.tps:
        cfg = _source_config(args.tree, U_STAR, 1)
        cfg["environment"]["sensory_stream"]["white_rate"] = 0.0  # probe only
        _patch_clock(cfg, tps)
        rep = out / f"tps_{tps}"
        if not (rep / ".done").exists():
            _run_one(runner, cfg, rep)
        r = _score(rep)
        rows.append(r)
        print(f"  T = {tps:>3}: choice = {r['choice'] or 'NONE':>14}  "
              f"arrival = {r['t_arrival_s']:.3f} s"
              if r["decided"] else f"  T = {tps:>3}: NO ARRIVAL")
    choices = {r["choice"] for r in rows}
    arr = [r["t_arrival_s"] for r in rows if r["t_arrival_s"] is not None]
    spread = (max(arr) - min(arr)) if len(arr) > 1 else float("nan")
    ok = len(choices) == 1 and len(arr) == len(rows)
    print(f"  choice consistent: {len(choices) == 1}; arrival spread across "
          f"T = {spread:.3f} s (the deterministic actuation-granularity "
          f"floor)  {'PASS' if ok else 'FAIL — frame bug, fix first'}")
    return 0 if ok else 1


def run(args) -> int:
    cells = U_ONGRID + [U_STAR]
    runner = InProcessRunner()
    todo = [(tps, u, rid) for tps in args.tps for u in cells
            for rid in range(1, args.runs + 1)]
    print(f"RA tps probe: {len(args.tps)} rates x {len(cells)} cells x "
          f"{args.runs} runs = {len(todo)} local runs -> {args.out}")
    n_run = n_skip = n_fail = 0
    t0 = time.time()
    for i, (tps, u, rid) in enumerate(todo):
        rep = args.out / f"tps_{tps}" / _u_label(u) / f"replicate_{rid}"
        if (rep / ".done").exists():
            n_skip += 1
            continue
        try:
            cfg = _patch_clock(_source_config(args.tree, u, rid), tps)
            _run_one(runner, cfg, rep)
            n_run += 1
        except Exception as exc:                    # noqa: BLE001 — data
            n_fail += 1
            print(f"  FAIL tps={tps} {_u_label(u)} r{rid}: {exc!r}")
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(todo)}] ran {n_run} skipped {n_skip} "
                  f"failed {n_fail} "
                  f"({(time.time()-t0)/max(n_run,1):.2f} s/run)")
            sys.stdout.flush()
    print(f"done: ran {n_run}, skipped {n_skip}, failed {n_fail} in "
          f"{time.time() - t0:.0f}s")
    return 1 if n_fail else 0


def analyze(args) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    rows = []
    for tps_dir in sorted(p for p in args.out.glob("tps_*") if p.is_dir()):
        for u_dir in sorted(d for d in tps_dir.iterdir() if d.is_dir()):
            for rep in u_dir.glob("replicate_*"):
                r = _score(rep)
                if r is not None:
                    rows.append({"u": u_dir.name.split("_", 1)[1],
                                 "run_id": int(rep.name.split("_")[1]), **r})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"nothing under {args.out} — run first")

    summ = []
    for (tps, u), g in df.groupby(["tps", "u"]):
        acc, lo, hi = wilson(int(g["correct"].sum()), len(g))
        med, mlo, mhi = boot_median_ci(
            g.loc[g["decided"], "t_arrival_s"].tolist())
        summ.append({"tps": tps, "u": float(u), "n": len(g),
                     "acc_all": acc, "acc_lo": lo, "acc_hi": hi,
                     "median_arrival_s": med, "arr_lo": mlo, "arr_hi": mhi,
                     "decided_frac": g["decided"].mean(),
                     "median_t_bif_s": g["t_bif_s"].median()})
    summ = pd.DataFrame(summ).sort_values(["u", "tps"])
    args.out.mkdir(parents=True, exist_ok=True)
    summ.to_csv(args.out / "ra_tps_summary.csv", index=False)
    print(f"wrote {args.out / 'ra_tps_summary.csv'}")

    # ---- gate: T = 1 must REPRODUCE the cluster campaign (on-grid cells) ---
    ref_pq = args.tree / "ra_trials.parquet"
    if 1 in set(df["tps"]) and ref_pq.is_file():
        ref = pd.read_parquet(ref_pq)
        ref = ref[(ref["actual_bp"].astype(int) == ACTUAL)
                  & (ref["v"].astype(float) == V)]
        ref = ref.assign(u=ref["u"].astype(str))
        mine = df[(df["tps"] == 1) & df["u"].isin(U_ONGRID)]
        merged = mine.merge(ref[["u", "run_id", "choice", "t_arrival_s"]],
                            on=["u", "run_id"], suffixes=("", "_ref"))
        same = ((merged["choice"].fillna("") ==
                 merged["choice_ref"].fillna(""))
                & ((merged["t_arrival_s"] - merged["t_arrival_s_ref"])
                   .abs().fillna(0.0) < 1e-9))
        print(f"T = 1 reproduction gate vs cluster: {int(same.sum())}/"
              f"{len(merged)} trials identical"
              + ("  PASS" if bool(same.all()) else "  FAIL — inspect"))

    # ---- figure: accuracy and arrival vs T, per u -------------------------
    tps_levels = sorted(df["tps"].unique())
    us = sorted(summ["u"].unique())
    cmap = plt.get_cmap("plasma")
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2),
                             constrained_layout=True)
    for i, u in enumerate(us):
        g = summ[summ["u"] == u].sort_values("tps")
        col = cmap(i / max(len(us) - 1, 1))
        lbl = f"u = {u:g}" + (" (u*)" if abs(u - U_STAR) < 1e-6 else "")
        axes[0].errorbar(g["tps"], g["acc_all"],
                         yerr=[g["acc_all"] - g["acc_lo"],
                               g["acc_hi"] - g["acc_all"]],
                         fmt="-o", ms=4, capsize=2, color=col, label=lbl)
        axes[1].errorbar(g["tps"], g["median_arrival_s"],
                         yerr=[g["median_arrival_s"] - g["arr_lo"],
                               g["arr_hi"] - g["median_arrival_s"]],
                         fmt="-o", ms=4, capsize=2, color=col)
        axes[2].plot(g["tps"], g["decided_frac"], "-o", ms=4, color=col)
    for ax, ylab in ((axes[0], "accuracy (all trials)"),
                     (axes[1], "median arrival (s)"),
                     (axes[2], "decided fraction")):
        ax.set_xscale("log")
        ax.set_xticks(tps_levels, [str(t) for t in tps_levels])
        ax.set_xticks([], minor=True)
        ax.set_xlabel("ticks per second")
        ax.set_ylabel(ylab)
    axes[0].legend(fontsize=8)
    fig.suptitle("RA clock-robustness probe — v = 0.5, actual δ_Q = 1 %, "
                 "same campaign configs/seeds, only the clock patched "
                 "(integration_time ∝ 1/T keeps ring-time per world-second "
                 "fixed; 1 percept draw/tick at rate-convention noise)")
    fig.savefig(args.out / "ra_tps_scaling.png", dpi=150)
    plt.close(fig)
    print(f"wrote {args.out / 'ra_tps_scaling.png'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=("e0", "run", "analyze"))
    ap.add_argument("--tree", type=Path, default=DEFAULT_TREE)
    ap.add_argument("--tps", default="1,10,50")
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    args.tps = [int(x) for x in str(args.tps).split(",")]
    return {"e0": e0, "run": run, "analyze": analyze}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
