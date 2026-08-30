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

#: Per-v colors — Okabe-Ito, no yellow (readable, colorblind-safe).
V_COLORS = {0.2: "#0072B2", 0.3: "#009E73", 0.4: "#D55E00", 0.5: "#CC79A7",
            0.6: "#E69F00", 0.8: "#56B4E9"}
STATIC_COLOR = "#7A6FBE"                  # the §2b static-bound family
MATCHED_ACC = [0.90, 0.95, 0.99]
N_BOOT = 500
#: b*_cost (§2b): the cost functional the collapsing policy optimizes, with
#: c = 2ΔQ = 0.1 — expected error cost + c · E[T_decision].
COST_TIME_RATE = frontier.NOISE_SCALE_C


def ddm_families(ddm_df):
    """(bellman_df, static_df). Frozen forced-choice tables have no `variant`
    column — everything there is the Bellman family."""
    if "variant" not in ddm_df.columns:
        return ddm_df, ddm_df.iloc[0:0]
    variant = ddm_df["variant"].fillna("bellman")
    return ddm_df[variant == "bellman"], ddm_df[variant == "static"]


def panel_vs(ra_df):
    """The v panels actually present in the data, in V_GRID_ALL order —
    figures stay correct before and after the U-v3 kernel extension."""
    have = {round(float(v), 3) for v in ra_df["v"].astype(float).unique()}
    return [v for v in frontier.V_GRID_ALL if round(v, 3) in have]


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
# §2b — the static family's optimum boundaries, DERIVED from the swept data
# ---------------------------------------------------------------------------
def static_bstar(static_df):
    """b*_cost = argmin over the swept b of [P(error) + c·E[T_decision]] with
    c = 2ΔQ = 0.1 (the collapsing policy's cost functional); b*_RR = argmax of
    the reward rate P(correct)/E[T_arrival]. Both with bootstrap CIs over
    trials, plus the Wald constant-drift analytic as a sanity anchor — the
    embodied task's time-varying geometry means they will not coincide
    exactly; the discrepancy is reported, not hidden."""
    if static_df.empty:
        return None
    per_b = {}
    for pid, g in static_df.groupby("point_id"):
        b = float(g["bound"].iloc[0])
        per_b[b] = g
    bs = sorted(per_b)

    def objectives(sample):
        cost, rr = {}, {}
        for b, g in sample.items():
            err = 1.0 - float(g["correct"].mean())
            rt = g.loc[g["decided"], "rt"].astype(float)
            arr = g.loc[g["decided"], "t_arrival_s"].astype(float)
            # Undecided (censored) trials count the full budget — scoring them
            # cheaper than the timeout would reward censoring.
            n_cens = int((~g["decided"]).sum())
            mean_rt = ((rt.sum() + n_cens * frontier.DDM_TIME_LIMIT)
                       / len(g)) if len(g) else float("nan")
            mean_arr = ((arr.sum() + n_cens * frontier.DDM_TIME_LIMIT)
                        / len(g)) if len(g) else float("nan")
            cost[b] = err + COST_TIME_RATE * mean_rt
            rr[b] = (1.0 - err) / mean_arr if mean_arr > 0 else float("nan")
        b_cost = min(cost, key=cost.get)
        b_rr = max(rr, key=rr.get)
        return b_cost, b_rr, cost, rr

    b_cost0, b_rr0, cost0, rr0 = objectives(per_b)
    rng = np.random.default_rng(20260830)
    boots_cost, boots_rr = [], []
    for _ in range(N_BOOT):
        sample = {b: g.iloc[rng.integers(0, len(g), len(g))]
                  for b, g in per_b.items()}
        bc, br, _c, _r = objectives(sample)
        boots_cost.append(bc)
        boots_rr.append(br)

    def ci(vals):
        return [float(np.percentile(vals, 2.5)),
                float(np.percentile(vals, 97.5))]

    # Wald constant-drift analytic: ER(b) = 1/(1+e^{kb}), DT(b) = (b/A)tanh(kb/2)
    k = 2.0 * frontier.QUALITY_DELTA / frontier.NOISE_SCALE_C ** 2
    A = frontier.QUALITY_DELTA
    t_travel = frontier.R0 / frontier.LINEAR_VELOCITY
    dense = np.geomspace(bs[0] / 2.0, bs[-1] * 2.0, 4000)
    er = 1.0 / (1.0 + np.exp(k * dense))
    dt = dense / A * np.tanh(0.5 * k * dense)
    wald_cost = er + COST_TIME_RATE * dt
    wald_rr = (1.0 - er) / (t_travel + dt)
    b_cost_wald = float(dense[int(np.argmin(wald_cost))])
    b_rr_wald = float(dense[int(np.argmax(wald_rr))])

    return {
        "b_grid": bs,
        "cost_functional": ("P(error) + c*E[T_decision], c = 2*dQ = "
                            f"{COST_TIME_RATE:g}; censored trials count the "
                            f"full {frontier.DDM_TIME_LIMIT} s budget"),
        "b_star_cost": {"estimate": b_cost0, "ci95": ci(boots_cost),
                        "cost_at_b": {f"{b:g}": cost0[b] for b in bs}},
        "b_star_rr": {"estimate": b_rr0, "ci95": ci(boots_rr),
                      "rr_at_b": {f"{b:g}": rr0[b] for b in bs}},
        "wald_analytic": {
            "b_star_cost": b_cost_wald, "b_star_rr": b_rr_wald,
            "note": ("constant-drift, constant-cost anchor; the embodied "
                     "task's time-varying geometry shifts the empirical "
                     "optimum — report the discrepancy, do not expect "
                     "coincidence (§2b)"),
            "discrepancy_cost": b_cost0 - b_cost_wald,
            "discrepancy_rr": b_rr0 - b_rr_wald},
        "n_boot": N_BOOT,
    }


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
    bellman, _static = ddm_families(ddm_df)
    ce_col = ("c_e" if "c_e" in bellman.columns else "bound"
              if "bound" in bellman.columns else None)
    has_ceiling_points = bool(ce_col) and any(
        float(ce) > max(frontier.C_E_GRID)
        for ce in bellman[ce_col].dropna().unique())
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


def _point_bounds(df):
    """point_id -> float bound (c_e or b); tolerant of the frozen schema."""
    col = "bound" if "bound" in df.columns else "c_e"
    return {pid: float(g[col].iloc[0]) for pid, g in df.groupby("point_id")
            if g[col].notna().any()}


def _draw_ddm_families(ax, ddm_df, bstar=None, label_points=True):
    """The §11 DDM layer: Bellman labeled by c_e, static labeled by b with
    b*_cost and b*_RR marked. Open markers: discretisation-limited (boundary
    below the 1 s-tick evidence substep) or decided_frac < 1."""
    n_ddm = int(ddm_df.groupby("point_id").size().median())
    for fam_df, color, marker, name in (
            (ddm_families(ddm_df)[0], "black", "o", "Bellman (c_e)"),
            (ddm_families(ddm_df)[1], STATIC_COLOR, "^", "static (b)")):
        if fam_df.empty:
            continue
        stats = sorted(group_stats(fam_df, "point_id"),
                       key=lambda s: s["median_t"])
        stats = [s for s in stats if s["median_t"] == s["median_t"]]
        bounds = _point_bounds(fam_df)
        ax.plot([s["median_t"] for s in stats],
                [s["acc_all"] for s in stats], "-", color=color, lw=1.5,
                zorder=3, label=f"DDM {name}, halt-at-midpoint, "
                                f"n={n_ddm}/point")
        pts = [(s["median_t"], s["acc_all"]) for s in stats]
        lab = _thinned(pts) if label_points else set()
        for i, s in enumerate(stats):
            b = bounds.get(s["point_id"], float("nan"))
            if name.startswith("Bellman"):
                open_m = (b in DISCRETISATION_LIMITED_CE
                          or s["decided_frac"] < 1)
            else:
                open_m = (b < frontier.EVIDENCE_SUBSTEP
                          or s["decided_frac"] < 1)
            ax.plot(s["median_t"], s["acc_all"], marker, ms=5, zorder=4,
                    mfc="white" if open_m else color, mec=color)
            if i in lab:
                ax.annotate(f"{b:g}", pts[i], textcoords="offset points",
                            xytext=(5, -10), fontsize=7, color=color)
        if name.startswith("static") and bstar is not None:
            # Distinct offsets: the two optima can land on the SAME grid point.
            for key, mk, txt, off in (("b_star_cost", "*", "b*_cost", (6, 8)),
                                      ("b_star_rr", "D", "b*_RR", (6, -16))):
                b_opt = bstar[key]["estimate"]
                s_opt = next((s for s in stats
                              if abs(bounds[s["point_id"]] - b_opt)
                              < 1e-9 * max(b_opt, 1)), None)
                if s_opt is not None:
                    ax.plot(s_opt["median_t"], s_opt["acc_all"], mk, ms=11,
                            mfc="none", mec=STATIC_COLOR, mew=1.6, zorder=5)
                    ax.annotate(txt, (s_opt["median_t"], s_opt["acc_all"]),
                                textcoords="offset points", xytext=off,
                                fontsize=7.5, color=STATIC_COLOR,
                                fontweight="bold")


def _panel_grid(plt, n_panels):
    ncols = 2 if n_panels <= 4 else 3
    nrows = max((n_panels + ncols - 1) // ncols, 1)
    return plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.7 * nrows),
                        sharex=True, sharey=True, squeeze=False)


def draw_main_per_v(ra_df, ddm_df, out_stem: Path, bstar=None):
    """§11 main figure: one panel per v (six once U-v3 lands) — the absolute-u
    sweep (all waves pooled) against BOTH DDM families, connected in u order."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_ra = int(ra_df.groupby("cell_id").size().median())
    n_ddm = int(ddm_df.groupby("point_id").size().median())
    asym = frontier.ddm_ideal_ceiling()
    v_colors = V_COLORS
    vs = panel_vs(ra_df)

    fig, axes = _panel_grid(plt, len(vs))
    for ax in axes.ravel()[len(vs):]:
        ax.set_visible(False)
    for ax, v in zip(axes.ravel(), vs):
        ax.axhline(asym, color="crimson", ls="--", lw=1,
                   label="DDM infinite-patience asymptote")
        _draw_ddm_families(ax, ddm_df, bstar=bstar)

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
        "DDM motion policy: HALT-AT-MIDPOINT (undecided at the midpoint ⇒ "
        "stop and keep integrating; Bellman bounds floor at z_halt, static "
        "bounds b with derived b*_cost / b*_RR marked);\n"
        f"n = {n_ra}/RA cell, {n_ddm}/DDM point; labels: u / c_e / b (thinned "
        "in clusters); open markers: decided_frac < 1 or "
        "discretisation-limited",
        fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
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

    v_colors = V_COLORS
    vs = panel_vs(ra_df)
    fig, axes = _panel_grid(plt, len(vs))
    for ax in axes.ravel()[len(vs):]:
        ax.set_visible(False)
    for ax, v in zip(axes.ravel(), vs):
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


def draw_overlay(ra_df, ddm_df, env_rows, out_stem: Path, bstar=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_ra = int(ra_df.groupby("cell_id").size().median())
    n_ddm = int(ddm_df.groupby("point_id").size().median())

    stair_by_dir = {}
    for direction in ("select_odd_eval_even", "select_even_eval_odd"):
        pts = [(e["median_t"], e["acc_all"]) for e in env_rows
               if e["direction"] == direction]
        stair_by_dir[direction] = staircase(pts)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    v_colors = V_COLORS

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

        # DDM frontier — both §2b families, halt-at-midpoint
        _draw_ddm_families(ax, ddm_df, bstar=bstar)

        # RA slices, connected in û (Set R) / u (Set U) order — fold included
        sub = ra_df[ra_df["sweep"] == sweep]
        for v in panel_vs(ra_df):
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
        f"seed-paired percept realizations; DDM motion policy: "
        f"halt-at-midpoint (§2b, both families); "
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

    # §2b: the static family's optimum boundaries, derived from the sweep.
    _bellman_df, static_df = ddm_families(ddm)
    bstar = static_bstar(static_df)
    if bstar is not None:
        with open(out / "static_bstar.json", "w", encoding="utf-8") as fh:
            json.dump(bstar, fh, indent=2)
        print(f"\nstatic family (§2b): b*_cost = "
              f"{bstar['b_star_cost']['estimate']:g} "
              f"CI95 {bstar['b_star_cost']['ci95']} (Wald analytic "
              f"{bstar['wald_analytic']['b_star_cost']:.4g}); "
              f"b*_RR = {bstar['b_star_rr']['estimate']:g} "
              f"CI95 {bstar['b_star_rr']['ci95']} (Wald "
              f"{bstar['wald_analytic']['b_star_rr']:.4g}) "
              "-> static_bstar.json")
    else:
        print("\nstatic family (§2b): no static-variant trials in the DDM "
              "table — run the halt campaign before the static overlay")

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

    draw_main_per_v(ra, ddm, out / "overlay_main", bstar=bstar)
    draw_tuning_curves(ra, out / "tuning_curves")
    draw_overlay(ra, ddm, env_rows, out / "overlay_slices", bstar=bstar)
    print(f"\nwrote {out / 'overlay_main.png'} (per-v absolute-u panels), "
          f"{out / 'tuning_curves.png'}, {out / 'overlay_slices.png'} "
          "(Set-R + envelope supplement), envelope.csv, regret.json, "
          "ceiling_check.json, mcnemar.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
