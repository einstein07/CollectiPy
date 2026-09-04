#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2026 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""SIDE EXPERIMENT (own tree, never mixed into the campaign): what does the
tick rate do to the DDM's within-decision dynamics?

    python3 scripts/qd_sweep_fixed_noise/tps_dynamics.py run \
        [--tree <cluster ddm tree>] [--tps 1,10,50] [--runs 100] [--out DIR]
    python3 scripts/qd_sweep_fixed_noise/tps_dynamics.py analyze [--out DIR]

Takes the VERY SAME replicate configs as the campaign's 1 % clairvoyant DDM
runs (actual_100/design_100 — all 12 Bellman c_e + the two static b*), from
the synced cluster tree, and re-runs them locally patching ONLY the clock:
`ticks_per_second` (arena + agent — they alias) and
`results.snapshots_per_second`. Everything else — seeds, strengths,
white_rate, A_expected, n_sub = 1, time_limit (in SECONDS: the arena builds
ticks_limit = time_limit·tps + 1) — is byte-for-byte the cluster config.

What changes with tps, by construction: the shared stream draws ONE percept
per tick keyed by (seed, target, tick), so at tps = 50 the accumulator gets
50 independent draws per second, each with per-draw SD white_rate·√tps (the
rate convention — information per second is invariant), and the boundary is
checked 50× per second. Same seed, but tick indices map to different world
times, so the sample PATHS differ across tps — comparisons are
distributional, not per-trial (except tps = 1, which must REPRODUCE the
cluster data bit-for-bit and is checked as a gate).

`analyze` writes tps_summary.csv, a per-controller steps/seconds/accuracy
figure, and example x(t) vs ±z(t) trajectories per tick rate.
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

DEFAULT_TREE = (qd._ROOT.parent / "seoul-data" / "beta-1"
                / "ra_ddm_qd_sweep_fixed_noise" / "qd_sweep_fixed_noise")
DEFAULT_OUT = qd._ROOT / "results" / "ddm_dynamics_tps"
SOURCE_REL = Path("ddm") / "actual_100" / "design_100"
N_TRAJ = 3          # example trajectories kept per (tps, controller)


def _controllers(tree: Path) -> list[Path]:
    src = tree / SOURCE_REL
    ctrls = sorted(d for d in src.iterdir() if d.is_dir())
    if not ctrls:
        raise SystemExit(f"no controller directories under {src}")
    return ctrls


def _patch_clock(cfg: dict, tps: int) -> dict:
    env = cfg["environment"]
    env["ticks_per_second"] = int(tps)
    env["agents"]["movable_0"]["ticks_per_second"] = int(tps)  # they alias
    env.setdefault("results", {})["snapshots_per_second"] = int(tps)
    # Everything else verbatim: seeds, strengths, white_rate, A_expected,
    # n_sub, time_limit (seconds). Assert the campaign's §2 invariants hold.
    qd.assert_config(cfg, "ddm", 100, 100)
    return cfg


def run(args) -> int:
    ctrls = _controllers(args.tree)
    runner = InProcessRunner()
    todo = [(tps, c, rid) for tps in args.tps for c in ctrls
            for rid in range(1, args.runs + 1)]
    print(f"tps sweep: {len(args.tps)} rates x {len(ctrls)} controllers x "
          f"{args.runs} runs = {len(todo)} local runs -> {args.out}")
    n_run = n_skip = n_fail = 0
    t0 = time.time()
    for i, (tps, ctrl, rid) in enumerate(todo):
        rep = args.out / f"tps_{tps}" / ctrl.name / f"replicate_{rid}"
        if (rep / ".done").exists():
            n_skip += 1
            continue
        src_cfg = ctrl / f"replicate_{rid}" / "config.json"
        if not src_cfg.is_file():
            print(f"  missing source config {src_cfg} — stop at --runs "
                  f"{args.runs} <= cluster n")
            return 1
        with open(src_cfg, encoding="utf-8") as fh:
            cfg = json.load(fh)
        _patch_clock(cfg, tps)
        rep.mkdir(parents=True, exist_ok=True)
        cfg["environment"]["results"]["base_path"] = str(rep)
        cfg_path = rep / "config.json"
        with open(cfg_path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
        try:
            runner.run(cfg_path)
            if not list(rep.glob("config_folder_*/run_*.zip")):
                raise RuntimeError("no run archive")
            (rep / ".done").touch()
            n_run += 1
        except Exception as exc:                    # noqa: BLE001 — data
            n_fail += 1
            print(f"  FAIL tps={tps} {ctrl.name} r{rid}: {exc!r}")
        if (i + 1) % 200 == 0:
            rate = (time.time() - t0) / max(n_run, 1)
            print(f"  [{i+1}/{len(todo)}] ran {n_run} skipped {n_skip} "
                  f"failed {n_fail} ({rate:.2f} s/run)")
            sys.stdout.flush()
    print(f"done: ran {n_run}, skipped {n_skip}, failed {n_fail} in "
          f"{time.time() - t0:.0f}s")
    return 1 if n_fail else 0


def _read_run(rep: Path):
    z = next(rep.glob("config_folder_*/run_*.zip"), None)
    if z is None:
        return None
    with open(rep / "config.json", encoding="utf-8") as fh:
        cfg = json.load(fh)
    tps = int(cfg["environment"]["ticks_per_second"])
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
        dm = next((n for n in names if n.endswith("_ddm.csv")), None)
        pos = next((n for n in names if n.endswith("_position.csv")), None)
        drows = (list(csv.DictReader(io.TextIOWrapper(zf.open(dm))))
                 if dm else [])
        prows = (list(csv.DictReader(io.TextIOWrapper(zf.open(pos))))
                 if pos else [])
    live = [r for r in drows if r.get("z") not in (None, "")]
    lastd = live[-1] if live else {}
    committed = (lastd.get("committed_id") or "") != ""
    tick, fine, hit = qd.first_crossing(
        prows, qd.target_positions(cfg),
        float(cfg["environment"]["termination"]["radius"]))
    rt = None
    try:
        rt = float(lastd.get("rt"))
    except (TypeError, ValueError):
        pass
    return {
        "tps": tps, "committed": committed,
        "committed_id": lastd.get("committed_id") or "",
        "commit_correct": (lastd.get("committed_id") or "")
        == qd.CORRECT_TARGET_ID,
        "rt_s": rt, "steps": None if rt is None else rt * tps,
        "halted": str(lastd.get("halt_event")).strip() in ("True", "true", "1"),
        "arrived": hit is not None,
        "arrival_correct": hit == qd.CORRECT_TARGET_ID,
        "t_arrival_s": None if fine is None else float(fine) / tps,
        "traj": [(int(r["tick"]) / tps, float(r["x"]), float(r["z"]))
                 for r in live if r.get("x") not in (None, "")],
    }


def analyze(args) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    rows, trajs = [], {}
    for tps_dir in sorted(args.out.glob("tps_*")):
        for ctrl_dir in sorted(d for d in tps_dir.iterdir() if d.is_dir()):
            for rep in sorted(ctrl_dir.glob("replicate_*"),
                              key=lambda p: int(p.name.split("_")[1])):
                r = _read_run(rep)
                if r is None:
                    continue
                rid = int(rep.name.split("_")[1])
                key = (r["tps"], ctrl_dir.name)
                if rid <= N_TRAJ:
                    trajs.setdefault(key, []).append((rid, r["traj"]))
                r.pop("traj")
                rows.append({"controller": ctrl_dir.name, "run_id": rid, **r})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"nothing to analyze under {args.out} — run first")

    summ = []
    for (tps, ctrl), g in df.groupby(["tps", "controller"]):
        com = g[g["committed"]]
        summ.append({
            "tps": tps, "controller": ctrl, "n": len(g),
            "commit_frac": g["committed"].mean(),
            "acc_all": g["arrival_correct"].mean(),
            "acc_committed": com["commit_correct"].mean() if len(com) else None,
            "median_rt_s": com["rt_s"].median() if len(com) else None,
            "median_steps": com["steps"].median() if len(com) else None,
            "halt_frac": g["halted"].mean(),
            "median_arrival_s": g.loc[g["arrived"], "t_arrival_s"].median(),
        })
    summ = pd.DataFrame(summ).sort_values(["controller", "tps"])
    args.out.mkdir(parents=True, exist_ok=True)
    summ.to_csv(args.out / "tps_summary.csv", index=False)
    print(f"wrote {args.out / 'tps_summary.csv'} ({len(summ)} rows)")

    # ---- gate: tps = 1 must REPRODUCE the cluster data (same configs) ------
    ref_pq = args.tree / "ddm_trials.parquet"
    if 1 in set(df["tps"]) and ref_pq.is_file():
        ref = pd.read_parquet(ref_pq)
        ref = ref[(ref["actual_bp"].astype(int) == 100)
                  & (ref["design_bp"].astype(int) == 100)]
        ref["controller"] = (ref["variant"].astype(str) + "_"
                            + ref["bound_param"].astype(str))
        mine = df[df["tps"] == 1]
        merged = mine.merge(
            ref[["controller", "run_id", "committed_id", "rt"]],
            on=["controller", "run_id"], how="inner",
            suffixes=("", "_ref"))
        same = ((merged["committed_id"] == merged["committed_id_ref"])
                & ((merged["rt_s"] - merged["rt"].astype(float)).abs()
                   .fillna(0.0) < 1e-9))
        print(f"tps = 1 reproduction gate vs cluster: {int(same.sum())}/"
              f"{len(merged)} trials identical (choice + rt)"
              + ("  PASS" if same.all() else "  FAIL — inspect"))

    # ---- figure 1: steps and seconds vs tps, per controller ----------------
    tps_levels = sorted(df["tps"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0),
                             constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    ctrls = sorted(summ["controller"].unique())
    for i, ctrl in enumerate(ctrls):
        g = summ[summ["controller"] == ctrl].sort_values("tps")
        col = cmap(i / max(len(ctrls) - 1, 1))
        axes[0].plot(g["tps"], g["median_steps"], "-o", ms=4, color=col,
                     label=ctrl)
        axes[1].plot(g["tps"], g["median_rt_s"], "-o", ms=4, color=col)
        axes[2].plot(g["tps"], g["acc_all"], "-o", ms=4, color=col)
    for ax, ylab, logy in ((axes[0], "median decision steps (rt × tps)", True),
                           (axes[1], "median decision time rt (s)", True),
                           (axes[2], "accuracy (all trials)", False)):
        ax.set_xscale("log")
        ax.set_xticks(tps_levels, [str(t) for t in tps_levels])
        ax.set_xticks([], minor=True)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("ticks per second")
        ax.set_ylabel(ylab)
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle("Tick-rate side experiment — 1 % clairvoyant DDM family, "
                 "same cluster configs & seeds, only the clock patched "
                 "(1 draw/tick, per-draw SD = white_rate·√tps)")
    fig.savefig(args.out / "tps_scaling.png", dpi=150)
    plt.close(fig)
    print(f"wrote {args.out / 'tps_scaling.png'}")

    # ---- figure 2: example trajectories x(t) vs ±z(t) ----------------------
    show = [c for c in ("bellman_0.3", "bellman_20", "bellman_300")
            if any(c == k[1] for k in trajs)] or ctrls[:3]
    fig, axes = plt.subplots(len(show), len(tps_levels),
                             figsize=(5.2 * len(tps_levels) + 1,
                                      3.4 * len(show)),
                             squeeze=False, constrained_layout=True)
    for r_i, ctrl in enumerate(show):
        for c_i, tps in enumerate(tps_levels):
            ax = axes[r_i][c_i]
            for rid, tr in trajs.get((tps, ctrl), []):
                t = [p[0] for p in tr]
                ax.plot(t, [p[1] for p in tr], lw=1.0, alpha=0.9)
                if rid == 1:
                    ax.plot(t, [p[2] for p in tr], "k--", lw=0.8)
                    ax.plot(t, [-p[2] for p in tr], "k--", lw=0.8)
            ax.axhline(0, color="gray", lw=0.5)
            ax.set_title(f"{ctrl} @ {tps} ticks/s", fontsize=9)
            if r_i == len(show) - 1:
                ax.set_xlabel("evidence time (s)")
            if c_i == 0:
                ax.set_ylabel("x(t)  (dashed: ±z(t))")
    fig.suptitle("Within-decision dynamics vs tick rate — first "
                 f"{N_TRAJ} replicates per cell (same seeds; sample paths "
                 "differ across tps because draws are tick-keyed)")
    fig.savefig(args.out / "tps_trajectories.png", dpi=150)
    plt.close(fig)
    print(f"wrote {args.out / 'tps_trajectories.png'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=("run", "analyze"))
    ap.add_argument("--tree", type=Path, default=DEFAULT_TREE,
                    help="synced cluster campaign root (source of configs "
                         "and the tps=1 reproduction reference)")
    ap.add_argument("--tps", default="1,10,50",
                    help="comma-separated tick rates")
    ap.add_argument("--runs", type=int, default=100,
                    help="replicates per (tps, controller); uses the cluster "
                         "replicates' own configs/seeds 1..N")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    args.tps = [int(x) for x in str(args.tps).split(",")]
    return run(args) if args.mode == "run" else analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
