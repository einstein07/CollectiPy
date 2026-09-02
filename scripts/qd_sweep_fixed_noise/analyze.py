#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2026 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""§9: analysis and figures for both arms.

    python3 scripts/qd_sweep_fixed_noise/analyze.py --base-root <root> \
        [--out-dir <root>/analysis] [--sat-v 0.2,0.5,0.8]

Consumes ra_trials.parquet / ra_cells.csv and ddm_trials.parquet /
ddm_points.csv (whichever exist; robust to a single arm or smoke-sized data).
Everything is reported in ABSOLUTE u (standing convention — no û, no u*
anywhere), and acc_all AND acc_decided are carried everywhere.

RA (per actual δ_Q):
    ra_heatmap_<metric>.png     (u, v) surfaces: acc_all, acc_decided,
                                median arrival, decided_frac
    ra_tuning_curves.png        acc vs u, LINEAR x, one line per v
    ra_peak_track.png/.csv      empirical u_peak(v; δ_Q), bootstrap CIs

DDM (the §4 design × actual matrix; regret vs the same-actual diagonal):
    ddm_matrix_acc.png / ddm_matrix_arrival.png
    ddm_regret.png + regret.csv     paired per-run_id CIs (pairing holds
                                    within actual δ_Q)
    ddm_misspec_curves.png          performance vs actual SNR, design marked,
                                    the analytic 1/(1+e^{-k·b}) overlay on
                                    every static panel
    ddm_censoring.png               decided_frac + halt_frac (expected data
                                    at low actual SNR, not failure)
    static_analytic_check.json      §5.2: deviations quantified as implied
                                    k_eff and b_eff (substep overshoot)

Combined:
    sat_plane_a<bp>.png             per-δ_Q SAT planes: selected RA v-panels
                                    against the frozen and clairvoyant DDM
                                    families (captions carry design/actual
                                    SNR and the halt policy)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt         # noqa: E402
import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402

import qd   # noqa: E402

N_BOOT = 500
HALT_NOTE = "motion policy: halt-at-midpoint (terminal 'halt_sprt'), c_h = 1"
VARIANT_ORDER = ["bellman_3", "bellman_20", "bellman_300",
                 "static-cost", "static-rr"]


def controller_key(row) -> str:
    """One label per frozen controller FAMILY (its design-specific bound is a
    property, not an identity): bellman_<ce> or static-cost / static-rr."""
    if row["variant"] == "bellman":
        return f"bellman_{float(row['bound_param']):g}"
    return row["variant"]


def controller_label(key: str) -> str:
    if key.startswith("bellman_"):
        return f"Bellman c_e = {key.split('_', 1)[1]}"
    return {"static-cost": "static b*_cost",
            "static-rr": "static b*_RR"}.get(key, key)


# ---------------------------------------------------------------------------
# RA
# ---------------------------------------------------------------------------
def ra_heatmaps(cells: pd.DataFrame, out: Path) -> None:
    metrics = [("acc_all", "accuracy (all trials)", "viridis", (0.5, 1.0)),
               ("acc_decided", "accuracy (decided)", "viridis", (0.5, 1.0)),
               ("median_arrival_s", "median arrival (s)", "magma", None),
               ("decided_frac", "decided fraction", "cividis", (0.0, 1.0))]
    actuals = sorted(cells["actual_bp"].astype(int).unique())
    us = sorted(cells["u"].astype(float).unique())
    vs = sorted(cells["v"].astype(float).unique())
    for metric, title, cmap, lim in metrics:
        fig, axes = plt.subplots(1, len(actuals),
                                 figsize=(5.2 * len(actuals) + 1, 4.2),
                                 squeeze=False, constrained_layout=True)
        for ax, bp in zip(axes[0], actuals):
            sub = cells[cells["actual_bp"].astype(int) == bp]
            grid = np.full((len(vs), len(us)), np.nan)
            for _i, r in sub.iterrows():
                grid[vs.index(float(r["v"])), us.index(float(r["u"]))] = \
                    float(r[metric]) if pd.notna(r[metric]) else np.nan
            # cell edges honest to the non-uniform u axis (the 0 -> 2 gap)
            ue = [us[0] - 0.25] + [(a + b) / 2 for a, b in zip(us, us[1:])] \
                + [us[-1] + 0.25]
            ve = [vs[0] - 0.05] + [(a + b) / 2 for a, b in zip(vs, vs[1:])] \
                + [vs[-1] + 0.05]
            kw = {} if lim is None else {"vmin": lim[0], "vmax": lim[1]}
            pcm = ax.pcolormesh(ue, ve, grid, cmap=cmap, **kw)
            ax.set_xlabel("u (absolute)")
            ax.set_ylabel("v")
            a = qd.drift_A(bp)
            ax.set_title(f"actual δ_Q = {bp} bp  (A = {a:g}, A/c = "
                         f"{a / qd.NOISE_SCALE_C:g})")
            fig.colorbar(pcm, ax=ax, label=title)
        fig.suptitle(f"RA surface — {title}; noise fixed at white_rate = "
                     f"{qd.WHITE_RATE} (c = {qd.NOISE_SCALE_C})")
        fig.savefig(out / f"ra_heatmap_{metric}.png", dpi=160)
        plt.close(fig)
        print(f"wrote {out / f'ra_heatmap_{metric}.png'}")


def ra_tuning_curves(cells: pd.DataFrame, out: Path) -> None:
    actuals = sorted(cells["actual_bp"].astype(int).unique())
    vs = sorted(cells["v"].astype(float).unique())
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, len(actuals),
                             figsize=(5.6 * len(actuals), 4.4),
                             squeeze=False, sharey=True,
                             constrained_layout=True)
    for ax, bp in zip(axes[0], actuals):
        sub = cells[cells["actual_bp"].astype(int) == bp]
        for v in vs:
            sv = sub[sub["v"].astype(float) == v].copy()
            sv["u"] = sv["u"].astype(float)
            sv = sv.sort_values("u")
            color = cmap((v - min(vs)) / max(max(vs) - min(vs), 1e-9))
            ax.plot(sv["u"], sv["acc_all"], "-o", ms=2.5, lw=1.1,
                    color=color, label=f"v = {v:g}")
            ax.fill_between(sv["u"], sv["acc_all_lo"], sv["acc_all_hi"],
                            color=color, alpha=0.12, lw=0)
        ax.set_xlabel("u (absolute, linear)")
        ax.set_title(f"actual δ_Q = {bp} bp")
        ax.axhline(0.5, color="gray", lw=0.6, ls=":")
    axes[0][0].set_ylabel("acc_all")
    axes[0][-1].legend(fontsize=7, ncol=2, loc="lower left")
    fig.suptitle("RA tuning curves — accuracy vs absolute u (fixed noise)")
    fig.savefig(out / "ra_tuning_curves.png", dpi=160)
    plt.close(fig)
    print(f"wrote {out / 'ra_tuning_curves.png'}")


def ra_peak_track(trials: pd.DataFrame, cells: pd.DataFrame,
                  out: Path) -> pd.DataFrame:
    """u_peak(v; δ_Q) = the empirical argmax of acc_all over the u grid,
    bootstrap CI over trials (per-cell resampling). Absolute u only."""
    rng = np.random.default_rng(20260902)
    rows = []
    for (bp, v), sub in cells.groupby([cells["actual_bp"].astype(int),
                                       cells["v"].astype(float)]):
        sub = sub.copy()
        sub["u"] = sub["u"].astype(float)
        sub = sub.sort_values("u")
        u_arr = sub["u"].to_numpy()
        peak = float(u_arr[int(np.argmax(sub["acc_all"].to_numpy()))])
        tr = trials[(trials["actual_bp"].astype(int) == bp)
                    & (trials["v"].astype(float) == v)]
        counts = {float(u): (int(g["correct"].astype(bool).sum()), len(g))
                  for u, g in tr.groupby(tr["u"].astype(float))}
        # For Bernoulli trials the per-cell bootstrap of the accuracy IS a
        # binomial resample — one vectorized draw per cell instead of N_BOOT
        # index resamples (identical distribution, ~1000x faster at 2 M rows).
        acc_mat = np.full((N_BOOT, len(u_arr)), np.nan)
        for j, u in enumerate(u_arr):
            if u in counts and counts[u][1] > 0:
                k, n = counts[u]
                acc_mat[:, j] = rng.binomial(n, k / n, size=N_BOOT) / n
        boots = u_arr[np.nanargmax(acc_mat, axis=1)]
        rows.append({"actual_bp": bp, "v": v, "u_peak": peak,
                     "u_peak_lo": float(np.percentile(boots, 2.5)),
                     "u_peak_hi": float(np.percentile(boots, 97.5)),
                     "acc_at_peak": float(sub["acc_all"].max())})
    track = pd.DataFrame(rows)
    track.to_csv(out / "peak_track.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    for bp, sub in track.groupby("actual_bp"):
        sub = sub.sort_values("v")
        ax.errorbar(sub["v"], sub["u_peak"],
                    yerr=[sub["u_peak"] - sub["u_peak_lo"],
                          sub["u_peak_hi"] - sub["u_peak"]],
                    fmt="-o", ms=4, capsize=2, label=f"actual {bp} bp")
    ax.set_xlabel("v")
    ax.set_ylabel("u_peak (absolute)")
    ax.set_title("Empirical peak track u_peak(v; δ_Q), bootstrap 95 % CI")
    ax.legend()
    fig.savefig(out / "ra_peak_track.png", dpi=160)
    plt.close(fig)
    print(f"wrote {out / 'ra_peak_track.png'} and peak_track.csv")
    return track


# ---------------------------------------------------------------------------
# DDM
# ---------------------------------------------------------------------------
def _matrix(points: pd.DataFrame, key: str, value: str) -> np.ndarray:
    designs = qd.DIFF_BP
    m = np.full((len(designs), len(designs)), np.nan)
    for _i, r in points.iterrows():
        if controller_key(r) != key:
            continue
        i = designs.index(int(r["design_bp"]))
        j = designs.index(int(r["actual_bp"]))
        m[i, j] = float(r[value]) if pd.notna(r[value]) else np.nan
    return m


def ddm_matrices(points: pd.DataFrame, out: Path) -> None:
    # §9: accuracy and MEAN arrival per (design, actual).
    for value, fname, cmap, fmt in (
            ("acc_all", "ddm_matrix_acc.png", "viridis", "{:.3f}"),
            ("mean_arrival_s", "ddm_matrix_arrival.png", "magma", "{:.1f}")):
        keys = [k for k in VARIANT_ORDER
                if any(controller_key(r) == k for _i, r in points.iterrows())]
        fig, axes = plt.subplots(1, len(keys),
                                 figsize=(3.4 * len(keys) + 1, 3.9),
                                 squeeze=False, constrained_layout=True)
        for ax, key in zip(axes[0], keys):
            m = _matrix(points, key, value)
            pcm = ax.imshow(m, cmap=cmap, aspect="auto")
            for i in range(m.shape[0]):
                for j in range(m.shape[1]):
                    if m[i, j] == m[i, j]:
                        ax.text(j, i, fmt.format(m[i, j]), ha="center",
                                va="center", fontsize=8,
                                color="white" if pcm.norm(m[i, j]) < 0.6
                                else "black")
                    if i == j:
                        ax.add_patch(plt.Rectangle(
                            (j - 0.5, i - 0.5), 1, 1, fill=False,
                            edgecolor="red", lw=1.4))
            ax.set_xticks(range(len(qd.DIFF_BP)),
                          [str(b) for b in qd.DIFF_BP])
            ax.set_yticks(range(len(qd.DIFF_BP)),
                          [str(b) for b in qd.DIFF_BP])
            ax.set_xlabel("actual δ_Q (bp)")
            ax.set_ylabel("design δ_Q (bp)")
            ax.set_title(controller_label(key), fontsize=9)
        fig.suptitle(f"Frozen-DDM {value} — design × actual "
                     f"(red = clairvoyant diagonal); {HALT_NOTE}")
        fig.colorbar(pcm, ax=axes[0].tolist(), shrink=0.85)
        fig.savefig(out / fname, dpi=160)
        plt.close(fig)
        print(f"wrote {out / fname}")


def ddm_regret(trials: pd.DataFrame, points: pd.DataFrame,
               out: Path) -> pd.DataFrame:
    """Regret of every cell against the SAME-ACTUAL diagonal cell of the same
    controller, paired per run_id (seed pairing holds within actual δ_Q):
    Δacc = acc(cell) − acc(diagonal) and Δarrival likewise, with paired
    bootstrap CIs. Censored trials count as incorrect (acc_all convention)
    and are excluded from the arrival pairing."""
    rng = np.random.default_rng(20260902)
    trials = trials.copy()
    bound_g = trials["bound_param"].astype(float).map("{:g}".format)
    trials["ckey"] = np.where(trials["variant"] == "bellman",
                              "bellman_" + bound_g, trials["variant"])
    rows = []
    for (key, actual), sub in trials.groupby(["ckey",
                                              trials["actual_bp"].astype(int)]):
        diag = sub[sub["design_bp"].astype(int) == actual]
        if diag.empty:
            continue
        diag = diag.set_index("run_id")
        for design, cell in sub.groupby(sub["design_bp"].astype(int)):
            cell = cell.set_index("run_id")
            common = cell.index.intersection(diag.index)
            if len(common) == 0:
                continue
            dc = (cell.loc[common, "correct"].astype(float)
                  - diag.loc[common, "correct"].astype(float)).to_numpy()
            both = common[(cell.loc[common, "decided"].astype(bool))
                          & (diag.loc[common, "decided"].astype(bool))]
            dt_ = (cell.loc[both, "t_arrival_s"].astype(float)
                   - diag.loc[both, "t_arrival_s"].astype(float)).to_numpy()

            def boot_ci(d):
                if d.size == 0:
                    return (float("nan"),) * 3
                bm = d[rng.integers(0, d.size, (2000, d.size))].mean(axis=1)
                return (float(d.mean()), float(np.percentile(bm, 2.5)),
                        float(np.percentile(bm, 97.5)))

            da, dal, dah = boot_ci(dc)
            dtm, dtl, dth = boot_ci(dt_)
            rows.append({"controller": key, "design_bp": design,
                         "actual_bp": actual, "n_paired": int(len(common)),
                         "n_paired_arrival": int(len(both)),
                         "d_acc": da, "d_acc_lo": dal, "d_acc_hi": dah,
                         "d_arrival_s": dtm, "d_arrival_lo": dtl,
                         "d_arrival_hi": dth})
    reg = pd.DataFrame(rows)
    reg.to_csv(out / "regret.csv", index=False)
    if reg.empty:
        print("regret: no paired cells yet")
        return reg
    keys = [k for k in VARIANT_ORDER if k in set(reg["controller"])]
    fig, axes = plt.subplots(2, len(keys), figsize=(3.2 * len(keys) + 1, 6.4),
                             squeeze=False, sharex=True,
                             constrained_layout=True)
    for c, key in enumerate(keys):
        sub = reg[reg["controller"] == key]
        for design, g in sub.groupby("design_bp"):
            g = g.sort_values("actual_bp")
            axes[0][c].errorbar(g["actual_bp"], g["d_acc"],
                                yerr=[g["d_acc"] - g["d_acc_lo"],
                                      g["d_acc_hi"] - g["d_acc"]],
                                fmt="-o", ms=3.5, capsize=2,
                                label=f"design {design}")
            axes[1][c].errorbar(g["actual_bp"], g["d_arrival_s"],
                                yerr=[g["d_arrival_s"] - g["d_arrival_lo"],
                                      g["d_arrival_hi"] - g["d_arrival_s"]],
                                fmt="-o", ms=3.5, capsize=2)
        for ax in (axes[0][c], axes[1][c]):
            ax.axhline(0.0, color="gray", lw=0.7, ls=":")
            ax.set_xscale("log")
            ax.set_xticks(qd.DIFF_BP, [str(b) for b in qd.DIFF_BP])
            ax.set_xticks([], minor=True)
        axes[0][c].set_title(controller_label(key), fontsize=9)
        axes[1][c].set_xlabel("actual δ_Q (bp)")
    axes[0][0].set_ylabel("Δ acc_all vs diagonal")
    axes[1][0].set_ylabel("Δ arrival (s) vs diagonal")
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Regret against the same-actual clairvoyant diagonal "
                 f"(paired per run_id); {HALT_NOTE}")
    fig.savefig(out / "ddm_regret.png", dpi=160)
    plt.close(fig)
    print(f"wrote {out / 'ddm_regret.png'} and regret.csv")
    return reg


def ddm_misspec_curves(points: pd.DataFrame, out: Path) -> None:
    """Performance vs actual SNR, one panel per controller, one curve per
    design (design marked on its own curve); static panels carry the §5.2
    analytic overlay acc = 1/(1+e^{-k_actual·b}) for their frozen b."""
    keys = [k for k in VARIANT_ORDER
            if any(controller_key(r) == k for _i, r in points.iterrows())]
    fig, axes = plt.subplots(2, len(keys), figsize=(3.3 * len(keys) + 1, 6.6),
                             squeeze=False, sharex=True,
                             constrained_layout=True)
    for c, key in enumerate(keys):
        sub = points[[controller_key(r) == key
                      for _i, r in points.iterrows()]]
        for design, g in sub.groupby(sub["design_bp"].astype(int)):
            g = g.copy()
            g["actual_bp"] = g["actual_bp"].astype(int)
            g = g.sort_values("actual_bp")
            line, = axes[0][c].plot(g["actual_bp"], g["acc_all"], "-o",
                                    ms=3.5, label=f"design {design}")
            axes[0][c].fill_between(g["actual_bp"], g["acc_all_lo"],
                                    g["acc_all_hi"], alpha=0.15, lw=0,
                                    color=line.get_color())
            axes[1][c].plot(g["actual_bp"], g["median_arrival_s"], "-o",
                            ms=3.5, color=line.get_color())
            at = g[g["actual_bp"] == design]
            if not at.empty:
                axes[0][c].plot(design, float(at["acc_all"].iloc[0]), "*",
                                ms=13, color=line.get_color(),
                                mec="k", mew=0.5, zorder=5)
            if key.startswith("static"):
                b = float(g["bound_param"].iloc[0])
                ks = np.array([qd.wald_k(a) for a in g["actual_bp"]])
                axes[0][c].plot(g["actual_bp"],
                                1.0 / (1.0 + np.exp(-ks * b)), "--", lw=1.0,
                                color=line.get_color(), alpha=0.8)
        axes[0][c].set_title(controller_label(key)
                             + ("\n(dashed: 1/(1+e^{-k·b}) analytic)"
                                if key.startswith("static") else ""),
                             fontsize=9)
        for ax in (axes[0][c], axes[1][c]):
            ax.set_xscale("log")
            ax.set_xticks(qd.DIFF_BP, [str(b) for b in qd.DIFF_BP])
            ax.set_xticks([], minor=True)
        axes[1][c].set_xlabel("actual δ_Q (bp)")
        axes[0][c].axhline(0.5, color="gray", lw=0.6, ls=":")
    axes[0][0].set_ylabel("acc_all")
    axes[1][0].set_ylabel("median arrival (s)")
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Frozen-DDM misspecification curves — ★ = design point "
                 f"(clairvoyant); {HALT_NOTE}")
    fig.savefig(out / "ddm_misspec_curves.png", dpi=160)
    plt.close(fig)
    print(f"wrote {out / 'ddm_misspec_curves.png'}")


def ddm_censoring(points: pd.DataFrame, out: Path) -> None:
    keys = [k for k in VARIANT_ORDER
            if any(controller_key(r) == k for _i, r in points.iterrows())]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0),
                             constrained_layout=True)
    for ax, value, title in ((axes[0], "decided_frac", "decided fraction"),
                             (axes[1], "halt_frac", "halt fraction")):
        width = 0.8 / max(len(qd.DIFF_BP), 1)
        labels, pos = [], []
        i = 0
        for key in keys:
            sub = points[[controller_key(r) == key
                          for _i, r in points.iterrows()]]
            for design, g in sub.groupby(sub["design_bp"].astype(int)):
                g = g.copy()
                g["actual_bp"] = g["actual_bp"].astype(int)
                g = g.sort_values("actual_bp")
                for k_a, (_j, r) in enumerate(g.iterrows()):
                    v = r.get(value)
                    ax.bar(i + k_a * width, 0.0 if pd.isna(v) else float(v),
                           width=width * 0.9,
                           color=plt.get_cmap("viridis")(k_a / 2.9))
                pos.append(i + width)
                labels.append(f"{controller_label(key)}\nd{design}")
                i += 1.0
        ax.set_xticks(pos, labels, fontsize=6, rotation=45, ha="right")
        ax.set_title(title + " (bar triplets: actual 50/100/200 bp)")
        ax.set_ylim(0, 1.05)
    fig.suptitle("Censoring is expected data at low actual SNR under frozen "
                 f"bold thresholds (§9); {HALT_NOTE}")
    fig.savefig(out / "ddm_censoring.png", dpi=160)
    plt.close(fig)
    print(f"wrote {out / 'ddm_censoring.png'}")


def static_analytic_check(points: pd.DataFrame, out: Path) -> None:
    """§5.2: a frozen boundary b at actual SNR k has accuracy
    1/(1+e^{-k·b}) exactly. Quantify deviations two ways: the implied
    k_eff = ln(acc/(1-acc))/b (an R-1-style constant check) and the implied
    b_eff = ln(acc/(1-acc))/k (substep overshoot — expected to exceed b when
    b sits below the 1 s-tick evidence substep 0.025)."""
    rows = []
    for _i, r in points.iterrows():
        if not str(r["variant"]).startswith("static"):
            continue
        b = float(r["bound_param"])
        k = qd.wald_k(int(r["actual_bp"]))
        pred = 1.0 / (1.0 + math.exp(-k * b))
        acc = float(r["acc_decided"]) if pd.notna(r["acc_decided"]) else None
        if acc is None or not (0.0 < acc < 1.0):
            continue
        logit = math.log(acc / (1.0 - acc))
        rows.append({
            "point_id": r["point_id"], "variant": r["variant"],
            "design_bp": int(r["design_bp"]), "actual_bp": int(r["actual_bp"]),
            "b": b, "k_actual": k, "acc_pred_wald": pred,
            "acc_decided": acc,
            "acc_decided_ci": [float(r["acc_decided_lo"]),
                               float(r["acc_decided_hi"])],
            "within_ci": float(r["acc_decided_lo"]) <= pred
            <= float(r["acc_decided_hi"]),
            "k_eff_implied": logit / b if b > 0 else float("nan"),
            "b_eff_implied": logit / k,
            "b_below_substep": b < qd.EVIDENCE_SUBSTEP})
    with open(out / "static_analytic_check.json", "w", encoding="utf-8") as fh:
        json.dump({"note": ("§5.2 backbone: deviations flag the R-1 constant "
                            "(k_eff) or substep overshoot (b_eff > b at "
                            f"b < {qd.EVIDENCE_SUBSTEP})"),
                   "points": rows}, fh, indent=2)
    n_in = sum(1 for r in rows if r["within_ci"])
    print(f"wrote {out / 'static_analytic_check.json'} "
          f"({n_in}/{len(rows)} static points within CI of the analytic)")


# ---------------------------------------------------------------------------
# Combined SAT planes
# ---------------------------------------------------------------------------
def sat_planes(ra_cells: pd.DataFrame | None, points: pd.DataFrame | None,
               out: Path, sat_v: list[float]) -> None:
    actuals = set()
    if ra_cells is not None:
        actuals |= set(ra_cells["actual_bp"].astype(int).unique())
    if points is not None:
        actuals |= set(points["actual_bp"].astype(int).unique())
    for bp in sorted(actuals):
        fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
        if ra_cells is not None:
            sub = ra_cells[ra_cells["actual_bp"].astype(int) == bp]
            cmap = plt.get_cmap("viridis")
            for v in sat_v:
                sv = sub[sub["v"].astype(float) == v].copy()
                if sv.empty:
                    continue
                sv["u"] = sv["u"].astype(float)
                sv = sv.sort_values("u")
                color = cmap((v - 0.1) / 0.9)
                ax.plot(sv["median_arrival_s"], sv["acc_all"], "-", lw=0.9,
                        alpha=0.7, color=color)
                ax.scatter(sv["median_arrival_s"], sv["acc_all"], s=10,
                           color=color, label=f"RA v = {v:g} (u-swept)")
        if points is not None:
            subp = points[points["actual_bp"].astype(int) == bp]
            markers = {"bellman": "s", "static-cost": "D", "static-rr": "^"}
            for _i, r in subp.iterrows():
                design = int(r["design_bp"])
                clair = design == bp
                mk = markers.get(str(r["variant"]),
                                 markers.get("bellman"))
                ax.scatter(r["median_arrival_s"], r["acc_all"], s=70,
                           marker=mk,
                           facecolor="crimson" if clair else "none",
                           edgecolor="crimson" if clair else "steelblue",
                           lw=1.3, zorder=5)
                ax.annotate(f"d{design} "
                            + (f"ce{float(r['bound_param']):g}"
                               if r["variant"] == "bellman"
                               else str(r["variant"]).replace("static-", "b*")),
                            (r["median_arrival_s"], r["acc_all"]),
                            fontsize=5.5, xytext=(3, 3),
                            textcoords="offset points")
        a = qd.drift_A(bp)
        ax.set_xlabel("median arrival (s)")
        ax.set_ylabel("acc_all")
        ax.set_title(f"SAT plane at actual δ_Q = {bp} bp (A = {a:g}, "
                     f"A/c = {a / qd.NOISE_SCALE_C:g}, k = "
                     f"{qd.wald_k(bp):g})\nfilled red = clairvoyant "
                     f"(design = actual); open blue = frozen off-design; "
                     f"{HALT_NOTE}")
        ax.legend(fontsize=7, loc="lower right")
        fig.savefig(out / f"sat_plane_a{bp}.png", dpi=160)
        plt.close(fig)
        print(f"wrote {out / f'sat_plane_a{bp}.png'}")


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--sat-v", default="0.2,0.5,0.8",
                    help="RA kernels drawn on the SAT planes")
    args = ap.parse_args(argv)
    root = args.base_root
    out = args.out_dir or (root / "analysis")
    out.mkdir(parents=True, exist_ok=True)
    sat_v = [float(x) for x in args.sat_v.split(",")]

    ra_trials = ra_cells = ddm_trials = ddm_points = None
    if (root / "ra_trials.parquet").is_file():
        ra_trials = pd.read_parquet(root / "ra_trials.parquet")
        ra_cells = pd.read_csv(root / "ra_cells.csv")
    if (root / "ddm_trials.parquet").is_file():
        ddm_trials = pd.read_parquet(root / "ddm_trials.parquet")
        ddm_points = pd.read_csv(root / "ddm_points.csv")
    if ra_cells is None and ddm_points is None:
        raise SystemExit(f"nothing to analyze under {root} — run "
                         "aggregate.py first")

    if ra_cells is not None and not ra_cells.empty:
        ra_heatmaps(ra_cells, out)
        ra_tuning_curves(ra_cells, out)
        ra_peak_track(ra_trials, ra_cells, out)
    if ddm_points is not None and not ddm_points.empty:
        ddm_matrices(ddm_points, out)
        ddm_regret(ddm_trials, ddm_points, out)
        ddm_misspec_curves(ddm_points, out)
        ddm_censoring(ddm_points, out)
        static_analytic_check(ddm_points, out)
    sat_planes(ra_cells, ddm_points, out, sat_v)
    print(f"\nanalysis complete under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
