#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2026 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""§10–§11: cross-validated envelope, overlay figures, regret statistics.

    python3 scripts/ra_ddm_frontier/analyze_overlay.py \
        --ra-root <dir> --ddm-root <dir> [--out <dir>]

Consumes `trials.parquet` (RA) and `ddm_trials.parquet` (DDM rerun) from
aggregate.py. Produces, in --out:

    overlay_main.png/.pdf     accuracy (acc_all) vs median arrival time (s):
                              DDM curve labeled by c_e; Set-R slices (one per v,
                              connected in û order, fold included) in panel A;
                              Set-U slices (connected in u order) in panel B;
                              the cross-validated envelope as a shaded band;
                              chance line; open markers for decided_frac < 1 or
                              discretisation-limited points; n per side.
    envelope.csv              both CV directions' evaluation-half points
    regret.json               Δt at matched accuracy (a ∈ {0.90, 0.95, 0.99})
                              and Δacc at matched time, with bootstrap CIs
    mcnemar.csv               per Set-R cell: paired McNemar vs the DDM point
                              nearest in median arrival (the §3 payoff)

Envelope protocol (§10): split by run_id parity; Pareto-select cells on one
half; re-evaluate exactly those cells on the other; swap and repeat. This kills
the winner's-curse bias a naive 100-cell max would carry against the DDM's 1-D
sweep.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import frontier   # noqa: E402
import seeding    # noqa: E402
from aggregate import wilson, boot_median_ci   # noqa: E402

DISCRETISATION_LIMITED_CE = {0.03, 0.1}   # campaign/factors.py, the DDM figure convention
MATCHED_ACC = [0.90, 0.95, 0.99]
N_BOOT = 500


# ---------------------------------------------------------------------------
def load_trials(path_stem: Path):
    import pandas as pd
    for suffix in (".parquet", ".csv"):
        p = path_stem.with_suffix(suffix)
        if p.is_file():
            df = (pd.read_parquet(p) if suffix == ".parquet"
                  else pd.read_csv(p))
            for col in ("correct", "decided", "timeout"):
                if col in df:
                    df[col] = df[col].astype(bool)
            return df
    raise SystemExit(f"no trials table at {path_stem}.parquet/.csv — run "
                     "aggregate.py first")


def group_stats(df, key: str):
    """Per-group acc_all (+Wilson), median arrival (+bootstrap CI), decided_frac."""
    out = []
    for ident, g in df.groupby(key, sort=False):
        n = len(g)
        acc, lo, hi = wilson(int(g["correct"].sum()), n)
        arr = g.loc[g["decided"], "t_arrival_s"].astype(float)
        med, mlo, mhi = boot_median_ci(arr.tolist())
        out.append({key: ident, "n": n, "acc_all": acc, "acc_lo": lo,
                    "acc_hi": hi, "median_t": med, "median_t_lo": mlo,
                    "median_t_hi": mhi,
                    "decided_frac": float(g["decided"].mean())})
    return out


# ---------------------------------------------------------------------------
# §10 envelope
# ---------------------------------------------------------------------------
def _cell_points(df, mask):
    """cell_id -> (median_t, acc_all) on the masked half."""
    pts = {}
    for cid, g in df[mask].groupby("cell_id"):
        arr = g.loc[g["decided"], "t_arrival_s"].astype(float)
        if arr.empty:
            continue
        pts[cid] = (float(arr.median()), float(g["correct"].mean()))
    return pts


def _pareto(points: dict) -> list[str]:
    """Cells not dominated (another cell faster AND at least as accurate,
    or as fast and strictly more accurate)."""
    keep = []
    for cid, (t, a) in points.items():
        dominated = any(
            (t2 <= t and a2 >= a) and (t2 < t or a2 > a)
            for c2, (t2, a2) in points.items() if c2 != cid)
        if not dominated:
            keep.append(cid)
    return keep


def envelope(df):
    """Both CV directions: [{direction, cell_id, median_t, acc_all}]."""
    parity = df["run_id"].astype(int) % 2
    out = []
    for direction, (sel, ev) in {"select_odd_eval_even": (1, 0),
                                 "select_even_eval_odd": (0, 1)}.items():
        chosen = _pareto(_cell_points(df, parity == sel))
        eval_pts = _cell_points(df, parity == ev)
        for cid in chosen:
            if cid in eval_pts:
                t, a = eval_pts[cid]
                out.append({"direction": direction, "cell_id": cid,
                            "median_t": t, "acc_all": a})
    return out


def staircase(points):
    """Upper-left staircase through (t, acc) points: sort by t, keep cummax acc."""
    pts = sorted(points)
    stair, best = [], -1.0
    for t, a in pts:
        if a > best:
            stair.append((t, a))
            best = a
    return stair


# ---------------------------------------------------------------------------
# §11 regret
# ---------------------------------------------------------------------------
def monotone_curve(points):
    """(t[], acc[]) sorted by t with acc made non-decreasing (cummax)."""
    pts = sorted(points)
    t = np.array([p[0] for p in pts], float)
    a = np.maximum.accumulate(np.array([p[1] for p in pts], float))
    return t, a


def t_at_acc(t, a, target):
    """Time the monotone curve first reaches accuracy `target` (interp), or nan."""
    if len(t) == 0 or target > a[-1] or target < a[0]:
        return float("nan")
    i = int(np.searchsorted(a, target))
    if a[i] == target or i == 0:
        return float(t[i])
    return float(np.interp(target, [a[i - 1], a[i]], [t[i - 1], t[i]]))


def acc_at_t(t, a, query):
    """Accuracy of the monotone curve at time `query` (step, last point ≤ t)."""
    if len(t) == 0 or query < t[0]:
        return float("nan")
    return float(a[min(int(np.searchsorted(t, query, side="right")) - 1,
                       len(a) - 1)])


def regret(ra_df, ddm_df, rng=None):
    """Point estimates (and optionally one bootstrap draw) of §11's regrets."""
    def resample(df, key):
        if rng is None:
            return df
        idx = np.concatenate([
            g.index.values[rng.integers(0, len(g), len(g))]
            for _k, g in df.groupby(key)])
        return df.loc[idx]

    ra = resample(ra_df, "cell_id")
    ddm = resample(ddm_df, "point_id")
    env_pts = [(e["median_t"], e["acc_all"]) for e in envelope(ra)]
    ddm_pts = [(s["median_t"], s["acc_all"])
               for s in group_stats(ddm, "point_id") if s["median_t"] == s["median_t"]]
    t_r, a_r = monotone_curve(env_pts)
    t_d, a_d = monotone_curve(ddm_pts)
    dt = {f"{a:.2f}": t_at_acc(t_r, a_r, a) - t_at_acc(t_d, a_d, a)
          for a in MATCHED_ACC}
    lo = max(t_r[0], t_d[0]) if len(t_r) and len(t_d) else float("nan")
    hi = min(t_r[-1], t_d[-1]) if len(t_r) and len(t_d) else float("nan")
    dacc = {}
    if lo == lo and hi == hi and hi > lo:
        for q in np.linspace(lo, hi, 5):
            dacc[f"{q:.2f}"] = acc_at_t(t_r, a_r, q) - acc_at_t(t_d, a_d, q)
    return dt, dacc, (lo, hi)


def regret_with_ci(ra_df, ddm_df):
    dt0, dacc0, support = regret(ra_df, ddm_df)
    rng = np.random.default_rng(20260829)
    boots_dt = {k: [] for k in dt0}
    boots_da = {k: [] for k in dacc0}
    for _ in range(N_BOOT):
        dt_b, dacc_b, _s = regret(ra_df, ddm_df, rng=rng)
        for k in boots_dt:
            boots_dt[k].append(dt_b.get(k, float("nan")))
        for k in boots_da:
            boots_da[k].append(dacc_b.get(k, float("nan")))

    def ci(vals):
        arr = np.array(vals, float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return [float("nan"), float("nan")]
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    return {
        "matched_accuracy_dt_s": {
            k: {"estimate": dt0[k], "ci95": ci(boots_dt[k]),
                "n_boot_defined": int(np.sum(~np.isnan(np.array(boots_dt[k]))))}
            for k in dt0},
        "matched_time_dacc": {
            k: {"estimate": dacc0[k], "ci95": ci(boots_da[k])}
            for k in dacc0},
        "time_support_overlap_s": list(support),
        "n_boot": N_BOOT,
        "note": ("dt > 0: the RA envelope reaches that accuracy later than the "
                 "DDM. Accuracies outside a curve's range give NaN rather than "
                 "an extrapolation (§11)."),
    }


# ---------------------------------------------------------------------------
# §11 paired McNemar (the payoff of §3)
# ---------------------------------------------------------------------------
def mcnemar(ra_df, ddm_df):
    ddm_stats = {s["point_id"]: s for s in group_stats(ddm_df, "point_id")}
    rows = []
    ra_r = ra_df[ra_df["sweep"] == "relative"]
    for cid, g in ra_r.groupby("cell_id"):
        arr = g.loc[g["decided"], "t_arrival_s"].astype(float)
        if arr.empty:
            continue
        med = float(arr.median())
        pid = min(ddm_stats,
                  key=lambda p: abs(ddm_stats[p]["median_t"] - med)
                  if ddm_stats[p]["median_t"] == ddm_stats[p]["median_t"]
                  else float("inf"))
        d = ddm_df[ddm_df["point_id"] == pid][["run_id", "correct"]]
        merged = g[["run_id", "correct"]].merge(
            d, on="run_id", suffixes=("_ra", "_ddm"))
        n = len(merged)
        if n == 0:
            continue
        b = int((merged["correct_ra"] & ~merged["correct_ddm"]).sum())
        c = int((~merged["correct_ra"] & merged["correct_ddm"]).sum())
        diff = (b - c) / n
        se = math.sqrt(max(b + c - (b - c) ** 2 / n, 0.0)) / n
        rows.append({"cell_id": cid, "ddm_point": pid,
                     "ra_median_t": med,
                     "ddm_median_t": ddm_stats[pid]["median_t"],
                     "n_paired": n, "b_ra_only": b, "c_ddm_only": c,
                     "paired_dacc": diff,
                     "dacc_lo": diff - 1.959964 * se,
                     "dacc_hi": diff + 1.959964 * se,
                     "mcnemar_z": ((b - c) / math.sqrt(b + c))
                     if b + c else float("nan")})
    return rows


# ---------------------------------------------------------------------------
# §11 ceiling verification
# ---------------------------------------------------------------------------
def ceiling_check(ra_df, ddm_df, env_rows):
    """The RA-beats-the-DDM-family claim stands only if the RA peak clears the
    DDM's infinite-patience asymptote with CI separation. The RA peak is taken
    from the cross-validated envelope (eval-half trials only) to avoid the
    100-cell winner's curse."""
    asymptote = frontier.ddm_ideal_ceiling()
    ddm_stats = group_stats(ddm_df, "point_id")
    ddm_best = max(ddm_stats, key=lambda s: s["acc_all"])
    peaks = []
    for direction in ("select_odd_eval_even", "select_even_eval_odd"):
        rows = [e for e in env_rows if e["direction"] == direction]
        if not rows:
            continue
        best = max(rows, key=lambda e: e["acc_all"])
        parity = 0 if direction.endswith("eval_even") else 1
        g = ra_df[(ra_df["cell_id"] == best["cell_id"])
                  & (ra_df["run_id"].astype(int) % 2 == parity)]
        acc, lo, hi = wilson(int(g["correct"].sum()), len(g))
        peaks.append({"direction": direction, "cell_id": best["cell_id"],
                      "acc": acc, "acc_lo": lo, "acc_hi": hi, "n_eval": len(g)})
    clears_asym = all(p["acc_lo"] > asymptote for p in peaks)
    clears_ddm = all(p["acc_lo"] > ddm_best["acc_hi"] for p in peaks)
    has_ceiling_points = any(
        float(str(p).replace("D_ce", "")) > max(frontier.C_E_GRID)
        for p in ddm_df["point_id"].unique())
    return {
        "analytic_asymptote": asymptote,
        "asymptote_formula": "Phi((A/c)*sqrt(r0/v)); ideal full-horizon observer",
        "ddm_best_point": {k: ddm_best[k] for k in
                           ("point_id", "acc_all", "acc_lo", "acc_hi")},
        "empirical_ceiling_points_present": bool(has_ceiling_points),
        "ra_envelope_peaks_eval_half": peaks,
        "ra_clears_analytic_asymptote_with_CI": bool(clears_asym),
        "ra_clears_measured_ddm_with_CI": bool(clears_ddm),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _thinned(points, dt_min=0.4, dacc_min=0.02):
    """Indices to label: endpoints always; otherwise only when the point moved
    at least (dt_min, dacc_min) from the last labeled one (§11: in committed
    clusters thin the labels, not the points)."""
    if not points:
        return set()
    keep, last = {0, len(points) - 1}, points[0]
    for i, p in enumerate(points[1:-1], start=1):
        if abs(p[0] - last[0]) >= dt_min or abs(p[1] - last[1]) >= dacc_min:
            keep.add(i)
            last = p
    return keep


def draw_main_per_v(ra_df, ddm_df, out_stem: Path):
    """§11 main figure: one panel per v — the absolute-u sweep (Set U + U-v2
    pooled) against the DDM curve, points connected in u order."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ddm_stats = sorted(group_stats(ddm_df, "point_id"),
                       key=lambda s: s["median_t"])
    n_ra = int(ra_df.groupby("cell_id").size().median())
    n_ddm = int(ddm_df.groupby("point_id").size().median())
    asym = frontier.ddm_ideal_ceiling()
    cmap = plt.get_cmap("viridis")
    v_colors = {v: cmap(i / (len(frontier.V_GRID) - 1))
                for i, v in enumerate(frontier.V_GRID)}

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)
    for ax, v in zip(axes.ravel(), frontier.V_GRID):
        ax.axhline(asym, color="crimson", ls="--", lw=1,
                   label="DDM infinite-patience asymptote")
        ax.plot([s["median_t"] for s in ddm_stats],
                [s["acc_all"] for s in ddm_stats], "-", color="black", lw=1.5,
                zorder=3, label=f"DDM (Bellman, n={n_ddm}/point)")
        ddm_pts = [(s["median_t"], s["acc_all"]) for s in ddm_stats]
        lab = _thinned(ddm_pts)
        for i, s in enumerate(ddm_stats):
            ce = float(str(s["point_id"]).replace("D_ce", ""))
            open_m = ce in DISCRETISATION_LIMITED_CE or s["decided_frac"] < 1
            ax.plot(s["median_t"], s["acc_all"], "o", ms=5, zorder=4,
                    mfc="white" if open_m else "black", mec="black")
            if i in lab:
                ax.annotate(f"{ce:g}", ddm_pts[i], textcoords="offset points",
                            xytext=(5, -10), fontsize=7)

        import numpy as _np
        gv = ra_df[(ra_df["sweep"] == "absolute")
                   & _np.isclose(ra_df["v"].astype(float), v)]
        order = gv.groupby("cell_id")["u"].first().astype(float)
        stats = group_stats(gv, "cell_id")
        stats.sort(key=lambda s: order[s["cell_id"]])
        stats = [s for s in stats if s["median_t"] == s["median_t"]]
        pts = [(s["median_t"], s["acc_all"]) for s in stats]
        color = v_colors[v]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=color,
                lw=1.3, zorder=2, label=f"RA v = {v:g} (absolute u)")
        lab = _thinned(pts)
        for i, s in enumerate(stats):
            ax.plot(s["median_t"], s["acc_all"], "s", ms=4.5, zorder=3,
                    mfc="white" if s["decided_frac"] < 1 else color, mec=color)
            if i in lab:
                ax.annotate(f"{order[s['cell_id']]:g}", pts[i],
                            textcoords="offset points", xytext=(4, 4),
                            fontsize=6.5, color=color)
        ax.axhline(0.5, color="gray", ls=":", lw=1, label="chance")
        ax.set_title(f"v = {v:g}", fontsize=11)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="lower right")
    for ax in axes[-1]:
        ax.set_xlabel("median arrival time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy (acc_all)")
    fig.suptitle(
        "RA absolute-u sweep vs the DDM frontier, per kernel shape — "
        "δ_Q = 1 %, Δθ = 60°; c = 2ΔQ = 0.1; frontier-v1;\n"
        f"n = {n_ra}/RA cell, {n_ddm}/DDM point; labels: u / c_e (thinned in "
        "clusters); open markers: decided_frac < 1 or discretisation-limited",
        fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in (".png", ".pdf"):
        fig.savefig(out_stem.with_suffix(ext), dpi=200)
    plt.close(fig)


def draw_tuning_curves(ra_df, out_stem: Path):
    """§11 companion: accuracy vs u on log-x, per v — the tuning-curve view an
    audience reads u_dec(v) from. u = 0 is drawn as the noise-floor line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as _np

    cmap = plt.get_cmap("viridis")
    v_colors = {v: cmap(i / (len(frontier.V_GRID) - 1))
                for i, v in enumerate(frontier.V_GRID)}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for ax, v in zip(axes.ravel(), frontier.V_GRID):
        gv = ra_df[(ra_df["sweep"] == "absolute")
                   & _np.isclose(ra_df["v"].astype(float), v)]
        order = gv.groupby("cell_id")["u"].first().astype(float)
        stats = sorted(group_stats(gv, "cell_id"),
                       key=lambda s: order[s["cell_id"]])
        u_star = float(gv["u_star"].astype(float).iloc[0])
        color = v_colors[v]
        floor = next((s for s in stats if order[s["cell_id"]] == 0.0), None)
        if floor is not None:
            ax.axhline(floor["acc_all"], color="gray", ls="--", lw=1,
                       label=f"u = 0 floor ({floor['acc_all']:.3f})")
        pos = [s for s in stats if order[s["cell_id"]] > 0]
        us = [order[s["cell_id"]] for s in pos]
        acc = [s["acc_all"] for s in pos]
        yerr = [[s["acc_all"] - s["acc_lo"] for s in pos],
                [s["acc_hi"] - s["acc_all"] for s in pos]]
        ax.errorbar(us, acc, yerr=yerr, fmt="s-", color=color, ms=4, lw=1.2,
                    elinewidth=0.8, capsize=2, label=f"RA v = {v:g}")
        ax.axvline(u_star, color=color, ls=":", lw=1,
                   label=f"u* = {u_star:.2f}")
        ax.axhline(0.5, color="gray", ls=":", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(f"v = {v:g}", fontsize=11)
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=7.5, loc="lower left")
    for ax in axes[-1]:
        ax.set_xlabel("coupling u (log scale)")
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy (acc_all)")
    fig.suptitle("RA tuning curves — accuracy vs absolute coupling u, per "
                 "kernel shape (Wilson 95 % bars; dashed line: the u = 0 "
                 "no-coupling floor; dotted vertical: u*(v))", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in (".png", ".pdf"):
        fig.savefig(out_stem.with_suffix(ext), dpi=200)
    plt.close(fig)


def draw_overlay(ra_df, ddm_df, env_rows, out_stem: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ddm_stats = sorted(group_stats(ddm_df, "point_id"),
                       key=lambda s: s["median_t"])
    n_ra = int(ra_df.groupby("cell_id").size().median())
    n_ddm = int(ddm_df.groupby("point_id").size().median())

    stair_by_dir = {}
    for direction in ("select_odd_eval_even", "select_even_eval_odd"):
        pts = [(e["median_t"], e["acc_all"]) for e in env_rows
               if e["direction"] == direction]
        stair_by_dir[direction] = staircase(pts)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    cmap = plt.get_cmap("viridis")
    v_colors = {v: cmap(i / max(len(frontier.V_GRID) - 1, 1))
                for i, v in enumerate(frontier.V_GRID)}

    for ax, sweep, order_col, label_fmt in (
            (axes[0], "relative", "u_hat", "û={val:g}"),
            (axes[1], "absolute", "u", "u={val:g}")):
        # envelope band behind everything
        stairs = [s for s in stair_by_dir.values() if s]
        if all(stairs) and len(stairs) == 2:
            tmin = min(s[0][0] for s in stairs)
            tmax = max(s[-1][0] for s in stairs)
            grid = np.linspace(tmin, tmax, 200)
            curves = []
            for s in stairs:
                t, a = monotone_curve(s)
                curves.append([acc_at_t(t, a, q) for q in grid])
            lo = np.nanmin(curves, axis=0)
            hi = np.nanmax(curves, axis=0)
            ax.fill_between(grid, lo, hi, step="post", alpha=0.18,
                            color="tab:orange", zorder=1,
                            label="RA envelope (cross-validated)")

        # DDM frontier
        td = [s["median_t"] for s in ddm_stats]
        ad = [s["acc_all"] for s in ddm_stats]
        ax.plot(td, ad, "-", color="black", lw=1.5, zorder=3,
                label=f"DDM (bellman, n={n_ddm}/point)")
        for s in ddm_stats:
            ce = float(next(str(p).replace("D_ce", "") for p in [s["point_id"]]))
            open_marker = (ce in DISCRETISATION_LIMITED_CE
                           or s["decided_frac"] < 1.0)
            ax.plot(s["median_t"], s["acc_all"], "o", ms=6, zorder=4,
                    mfc="white" if open_marker else "black", mec="black")
            ax.annotate(f"{ce:g}", (s["median_t"], s["acc_all"]),
                        textcoords="offset points", xytext=(5, -9),
                        fontsize=7, color="black")

        # RA slices, connected in û (Set R) / u (Set U) order — fold included
        sub = ra_df[ra_df["sweep"] == sweep]
        for v in frontier.V_GRID:
            gv = sub[np.isclose(sub["v"].astype(float), v)]
            stats = group_stats(gv, "cell_id")
            order = {s["cell_id"]: float(
                gv[gv["cell_id"] == s["cell_id"]][order_col].iloc[0])
                for s in stats}
            stats.sort(key=lambda s: order[s["cell_id"]])
            stats = [s for s in stats if s["median_t"] == s["median_t"]]
            if not stats:
                continue
            ax.plot([s["median_t"] for s in stats],
                    [s["acc_all"] for s in stats], "-", color=v_colors[v],
                    lw=1.2, alpha=0.9, zorder=2, label=f"RA v={v:g}")
            for s in stats:
                open_marker = s["decided_frac"] < 1.0
                ax.plot(s["median_t"], s["acc_all"], "s", ms=4, zorder=3,
                        mfc="white" if open_marker else v_colors[v],
                        mec=v_colors[v])
                ax.annotate(label_fmt.format(val=order[s["cell_id"]]),
                            (s["median_t"], s["acc_all"]),
                            textcoords="offset points", xytext=(4, 3),
                            fontsize=5.5, color=v_colors[v], alpha=0.9)

        ax.axhline(0.5, color="gray", ls=":", lw=1, label="chance")
        ax.set_xlabel("median arrival time (s)")
        ax.set_title(f"Set {'R (û sweep)' if sweep == 'relative' else 'U (absolute u)'}")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("accuracy (acc_all: undecided scored as error)")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle(
        f"RA slices vs the DDM frontier — δ_Q = 1 %, Δθ = 60°; "
        f"white_rate {frontier.WHITE_RATE:g} (evidence-channel noise "
        f"c = √2·η = {frontier.NOISE_SCALE_C:g} = 2×ΔQ); "
        f"seed scheme {seeding.SCHEME}; n = {n_ra}/RA cell, {n_ddm}/DDM point; "
        f"both models at {frontier.RA_TICKS_PER_SECOND} tick/s (tick = 1 s) — "
        f"seed-paired percept realizations; "
        f"open markers: decided_frac < 1 "
        f"or discretisation-limited", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in (".png", ".pdf"):
        fig.savefig(out_stem.with_suffix(ext), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ra-root", type=Path, required=True)
    ap.add_argument("--ddm-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    out = args.out or (args.ra_root / "analysis")
    out.mkdir(parents=True, exist_ok=True)

    ra = load_trials(args.ra_root / "trials")
    ddm = load_trials(args.ddm_root / "ddm_trials")
    print(f"RA trials: {len(ra)} over {ra['cell_id'].nunique()} cells; "
          f"DDM trials: {len(ddm)} over {ddm['point_id'].nunique()} points")

    env_rows = envelope(ra)
    import pandas as pd
    pd.DataFrame(env_rows).to_csv(out / "envelope.csv", index=False)
    for direction in ("select_odd_eval_even", "select_even_eval_odd"):
        cells = sorted(e["cell_id"] for e in env_rows
                       if e["direction"] == direction)
        print(f"envelope [{direction}]: {len(cells)} cells: "
              f"{', '.join(cells)}")
    a_sets = [frozenset(e["cell_id"] for e in env_rows
                        if e["direction"] == d)
              for d in ("select_odd_eval_even", "select_even_eval_odd")]
    if a_sets[0] != a_sets[1]:
        print("NOTE: the two CV directions select different cell sets — "
              "selection noise is visible; report both (§10).")

    print("\ncomputing regret statistics "
          f"({N_BOOT} bootstrap resamples) ...")
    reg = regret_with_ci(ra, ddm)
    with open(out / "regret.json", "w", encoding="utf-8") as fh:
        json.dump(reg, fh, indent=2)
    for a, r in reg["matched_accuracy_dt_s"].items():
        print(f"  Δt(acc={a}) = {r['estimate']:+.2f} s  "
              f"CI95 [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]"
              if r["estimate"] == r["estimate"] else
              f"  Δt(acc={a}) undefined (outside a curve's range)")
    lo, hi = reg["time_support_overlap_s"]
    if not (lo == lo and hi == hi and hi > lo):
        print("  matched-time Δacc: time supports barely overlap — "
              "reported plainly, not extrapolated (§11)")

    mc = mcnemar(ra, ddm)
    pd.DataFrame(mc).to_csv(out / "mcnemar.csv", index=False)
    sig = [r for r in mc if r["dacc_lo"] > 0 or r["dacc_hi"] < 0]
    print(f"\npaired McNemar: {len(mc)} Set-R cells vs nearest-time DDM "
          f"points; {len(sig)} with CI excluding zero -> mcnemar.csv")

    # §11 ceiling verification — required before claiming the RA sub-critical
    # peak exceeds the DDM family.
    ceiling = ceiling_check(ra, ddm, env_rows)
    with open(out / "ceiling_check.json", "w", encoding="utf-8") as fh:
        json.dump(ceiling, fh, indent=2)
    print(f"\nDDM ceiling verification: analytic asymptote "
          f"{ceiling['analytic_asymptote']:.4f}; best measured DDM point "
          f"{ceiling['ddm_best_point']['acc_all']:.4f} "
          f"[{ceiling['ddm_best_point']['acc_lo']:.4f}, "
          f"{ceiling['ddm_best_point']['acc_hi']:.4f}]"
          + ("" if ceiling["empirical_ceiling_points_present"] else
             " (extreme-c_e ceiling points NOT yet in the data — run the "
             "ddm_manifest_topup campaign)"))
    for p in ceiling["ra_envelope_peaks_eval_half"]:
        print(f"  RA envelope peak [{p['direction']}]: {p['cell_id']} "
              f"acc {p['acc']:.4f} [{p['acc_lo']:.4f}, {p['acc_hi']:.4f}] "
              f"(n_eval = {p['n_eval']})")
    verdict = ("STANDS" if ceiling["ra_clears_analytic_asymptote_with_CI"]
               and ceiling["ra_clears_measured_ddm_with_CI"] else
               "DOES NOT STAND (no CI separation)")
    print(f"  claim 'RA sub-critical peak exceeds the DDM family': {verdict}")

    draw_main_per_v(ra, ddm, out / "overlay_main")
    draw_tuning_curves(ra, out / "tuning_curves")
    draw_overlay(ra, ddm, env_rows, out / "overlay_slices")
    print(f"\nwrote {out / 'overlay_main.png'} (per-v absolute-u panels), "
          f"{out / 'tuning_curves.png'}, {out / 'overlay_slices.png'} "
          "(Set-R + envelope supplement), envelope.csv, regret.json, "
          "ceiling_check.json, mcnemar.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
