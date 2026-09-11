# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Figures of the sensory-map reduction experiment.

    plot_maps      (spec 7.3a) the sum and max maps for the 6.1 geometry at each
                   Delta_close, on the fine ring and at the 30 runtime nodes
    plot_outcomes  (spec 7.3d) P(mean_of_pair) and P(arrive at C) vs Delta_close,
                   both reductions, one panel per q_C, 95 % Wilson bands
    plot_extras    the ring shape on tick 2 (merged / bimodal), P(timeout),
                   P(no commitment) and P(arrive at A or B) the same way

Two series everywhere, colour fixed to the entity: sum = slot-1 blue, max = slot-2
orange (never re-assigned), a legend plus sparing direct labels, text in ink tokens,
hairline solid grid, thin marks. PNG and PDF, light surface.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

try:
    from map_reduction import factors
except ImportError:                    # pragma: no cover
    import factors                     # type: ignore

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

# Reference palette (dataviz skill, light mode): categorical slots 1 and 2, ink,
# chrome. Colour follows the entity: sum is always blue, max always orange.
SERIES = {"sum": "#2a78d6", "max": "#eb6834"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
LINE_W = 1.6          # ~2 px
MARKER = 6.0          # ~8 px


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.titlecolor": INK, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8,
    })
    return plt


def _save(fig, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in ("png", "pdf"):
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=200 if ext == "png" else None, bbox_inches="tight")
        out.append(path)
    return out


# ---------------------------------------------------------------------------
# (a) the maps
# ---------------------------------------------------------------------------
def plot_maps(out_dir: Path, q_c_rel: float = 1.02, n_fine: int = 3600) -> list[Path]:
    from models.mean_field_systems import sensory_map
    plt = _style()
    theta_fine = np.linspace(-np.pi, np.pi, n_fine, endpoint=False)
    theta_ring = np.linspace(-np.pi, np.pi, factors.NUM_NEURONS, endpoint=False)
    deg_fine, deg_ring = np.degrees(theta_fine), np.degrees(theta_ring)
    strengths = factors.strengths(q_c_rel)
    q = np.array([strengths[k] for k in factors.LABELS])
    deltas = factors.DELTA_CLOSE_DEG
    ncol = 4
    nrow = math.ceil(len(deltas) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.7 * nrow), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    ymax = 0.0
    for k, delta in enumerate(deltas):
        ax = axes.flat[k]
        bearings = factors.bearings_deg(delta)
        phi = np.radians([bearings[k2] for k2 in factors.LABELS])
        for red in factors.REDUCTIONS:
            fine = sensory_map(theta_fine, phi, q, factors.KAPPA, reduction=red)
            ring = sensory_map(theta_ring, phi, q, factors.KAPPA, reduction=red)
            ymax = max(ymax, float(fine.max()))
            ax.plot(deg_fine, fine, color=SERIES[red], lw=LINE_W, solid_capstyle="round",
                    label=red, zorder=3)
            ax.plot(deg_ring, ring, ls="none", marker="o", ms=MARKER * 0.75,
                    mfc=SERIES[red], mec=SURFACE, mew=1.0, zorder=4)
        # Target bearings as recessive markers, labelled once per panel (at the top,
        # clear of the tick labels; A and B share one label when they nearly touch).
        for label, b in bearings.items():
            ax.axvline(b, color=AXIS, lw=0.8, zorder=1)
        ax.text(bearings["C"], 0.97, "C", transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=7.5, color=INK_2)
        ax.text(0.5 * (bearings["A"] + bearings["B"]), 0.97, "A  B" if delta < 30 else "A     B",
                transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7.5,
                color=INK_2)
        # Selective direct labels: the A-B peak under sum (above it) and the dip at the
        # A-B midpoint under max (below it) — the two numbers the reduction changes.
        sel = (deg_fine >= min(bearings["A"], bearings["B"]) - 2) & \
              (deg_fine <= max(bearings["A"], bearings["B"]) + 2)
        fine_sum = sensory_map(theta_fine, phi, q, factors.KAPPA, reduction="sum")
        peak = float(fine_sum[sel].max())
        x_peak = float(deg_fine[sel][np.argmax(fine_sum[sel])])
        ax.annotate(f"sum peak {peak:.2f}", (x_peak, peak), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7.5,
                    color=INK_2, bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))
        fine_max = sensory_map(theta_fine, phi, q, factors.KAPPA, reduction="max")
        x_mid = 0.5 * (bearings["A"] + bearings["B"])
        i_mid = int(np.argmin(np.abs(deg_fine - x_mid)))
        dip = float(fine_max[i_mid])
        if dip >= 2.0:
            ax.annotate(f"max dip {dip:.2f}", (x_mid, dip), xytext=(0, -6),
                        textcoords="offset points", ha="center", va="top", fontsize=7.5,
                        color=INK_2, bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))
        else:
            # A shallow dip sits on the tick labels: lead the label to the empty
            # region right of B instead.
            ax.annotate(f"max dip {dip:.2f}", (x_mid, dip), xytext=(x_mid + 70, 3.6),
                        textcoords="data", ha="center", va="center", fontsize=7.5, color=INK_2,
                        arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8, shrinkB=2))
        merged = delta < factors.merge_threshold_deg()
        ax.set_title(f"Δ_close = {delta}°" + ("  (below Δ*)" if merged else ""), fontsize=9,
                     loc="left")
        ax.set_xlim(-180, 180)
        ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
    for ax in axes.flat[len(deltas):]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("ring angle θ (deg, egocentric)")
    for ax in axes[:, 0]:
        ax.set_ylabel("input  Σ / max of q·vM(θ−φ)")
    axes.flat[0].set_ylim(0, ymax * 1.18)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    legend_ax = axes.flat[len(deltas)] if len(deltas) < axes.size else axes.flat[0]
    legend_ax.legend(handles, [f"{l}: " + ("Σ_j q_j vM" if l == "sum" else "max_j q_j vM")
                               for l in labels], loc="upper left", frameon=False)
    fig.suptitle(
        f"Sensory map by reduction — κ = {factors.KAPPA}, n = {factors.NUM_NEURONS} ring nodes "
        f"(dots), q_A = q_B = {strengths['A']:.2f}, q_C = {strengths['C']:.2f}; "
        f"A–B merge threshold Δ* = {factors.merge_threshold_deg():.1f}°",
        fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    paths = _save(fig, out_dir / "maps_sum_vs_max")
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# (d) outcomes vs Delta_close
# ---------------------------------------------------------------------------
def _panel(ax, cells, metric: str, q_c: float, show_legend: bool, show_dstar: bool):
    sub = cells[cells["q_c_rel"] == q_c]
    end_labels = []
    for red in factors.REDUCTIONS:
        s = sub[sub["reduction"] == red].sort_values("delta_close_deg")
        if s.empty:
            continue
        x = s["delta_close_deg"].values
        y = s[metric].values
        lo, hi = s[f"{metric}_lo"].values, s[f"{metric}_hi"].values
        ax.fill_between(x, lo, hi, color=SERIES[red], alpha=0.12, lw=0, zorder=2)
        ax.plot(x, y, color=SERIES[red], lw=LINE_W, marker="o", ms=MARKER, mfc=SERIES[red],
                mec=SURFACE, mew=1.2, solid_capstyle="round", label=red, zorder=4)
        end_labels.append((x[-1], y[-1], red))
    # Direct end labels; when the two series end together, stack the labels
    # instead of overprinting them (sum above, max below).
    stacked = len(end_labels) == 2 and abs(end_labels[0][1] - end_labels[1][1]) < 0.06
    for x_end, y_end, red in end_labels:
        dy = (5 if red == "sum" else -5) if stacked else 0
        ax.annotate(red, (x_end, y_end), xytext=(6, dy), textcoords="offset points",
                    va="center", ha="left", fontsize=8, color=INK_2)
    if show_dstar:
        dstar = factors.merge_threshold_deg()
        ax.axvline(dstar, color=AXIS, lw=0.8, zorder=1)
        ax.text(dstar + 0.8, 0.97, f"Δ* = {dstar:.1f}°", fontsize=7.5, color=MUTED, va="top")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(min(factors.DELTA_CLOSE_DEG) - 3, max(factors.DELTA_CLOSE_DEG) + 9)
    ax.set_xticks(factors.DELTA_CLOSE_DEG)
    if show_legend:
        ax.legend(loc="upper right", frameon=False)


def plot_outcomes(cells, out_dir: Path, n_trials: int | None = None,
                  stem: str = "outcomes_vs_delta") -> list[Path]:
    plt = _style()
    qcs = sorted(cells["q_c_rel"].unique())
    metrics = [("p_mean_of_pair", "P(mean of pair at commitment)"),
               ("p_arrive_C", "P(arrive at C)")]
    fig, axes = plt.subplots(len(metrics), len(qcs), figsize=(3.6 * len(qcs), 2.9 * len(metrics)),
                             sharex=True, sharey=True, squeeze=False)
    for i, (metric, ylabel) in enumerate(metrics):
        for j, q_c in enumerate(qcs):
            ax = axes[i, j]
            _panel(ax, cells, metric, q_c, show_legend=(i == 0 and j == 0), show_dstar=(i == 0))
            if i == 0:
                ax.set_title(f"q_C = {q_c:.2f}  (q_A = q_B = 1.00)", fontsize=9, loc="left")
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == len(metrics) - 1:
                ax.set_xlabel("Δ_close (deg)")
    n_txt = f", n = {n_trials} paired trials per cell" if n_trials else ""
    fig.suptitle(f"Close-target failure mode by sensory-map reduction — "
                 f"u = {factors.U}, v = {factors.V}, κ = {factors.KAPPA}, σ = {factors.SIGMA}"
                 f"{n_txt}; bands: 95 % Wilson", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    paths = _save(fig, out_dir / stem)
    plt.close(fig)
    return paths


def plot_extras(cells, out_dir: Path, n_trials: int | None = None) -> list[Path]:
    plt = _style()
    qcs = sorted(cells["q_c_rel"].unique())
    metrics = [("p_merged_t2", "P(one bump at midpoint, tick 2)"),
               ("p_bimodal_t2", "P(two bumps, tick 2)"),
               ("p_timeout", "P(timeout)"), ("p_no_commit", "P(no commitment)"),
               ("p_arrive_AB", "P(arrive at A or B)")]
    fig, axes = plt.subplots(len(metrics), len(qcs), figsize=(3.6 * len(qcs), 2.7 * len(metrics)),
                             sharex=True, sharey=True, squeeze=False)
    for i, (metric, ylabel) in enumerate(metrics):
        for j, q_c in enumerate(qcs):
            ax = axes[i, j]
            _panel(ax, cells, metric, q_c, show_legend=(i == 0 and j == 0), show_dstar=(i == 0))
            if i == 0:
                ax.set_title(f"q_C = {q_c:.2f}", fontsize=9, loc="left")
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == len(metrics) - 1:
                ax.set_xlabel("Δ_close (deg)")
    n_txt = f", n = {n_trials} paired trials per cell" if n_trials else ""
    fig.suptitle(f"Secondary outcomes by reduction{n_txt}; bands: 95 % Wilson",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    paths = _save(fig, out_dir / "outcomes_extra_vs_delta")
    plt.close(fig)
    return paths
