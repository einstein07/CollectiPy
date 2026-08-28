#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Step 6: does performance collapse onto u_hat?

Section 10 of `uhat-v-factorial-experiment.md`. Runs standalone on
`trials.parquet`.

    python3 scripts/uhat_v_sweep/analyze_collapse.py \
        --results-root results/uhat_v_sweep [--paired]

This is an ESTIMATION problem, not a significance-testing one: at ~32k trials
everything is "significant", so the deviance ladder reports SHARES and the
plots report intervals. No p-value decides anything here.

Deviance ladder, trial-level logistic regression of `correct`:

    M0: ~ 1
    M1: ~ C(u_hat)
    M2: ~ C(v)
    M3: ~ C(u_hat) + C(v)
    M4: ~ C(u_hat) * C(v)        saturated in cell means

    interaction share = [D(M3) - D(M4)] / [D(M0) - D(M4)]
    v share beyond u_hat = [D(M1) - D(M3)] / [D(M0) - D(M4)]

Pre-registered reading (Section 10):
    collapse     v main effect + interaction together explain < 10 %
    additive     interaction alone < 10 %, but v's main effect is substantial
    interaction  otherwise; u_hat is not sufficient and the M3 residual heatmap
                 says where it fails

statsmodels is used when importable; otherwise an equivalent IRLS/least-squares
fit built on numpy runs instead, so the analysis never fails to produce the
ladder for want of a package on a login node. Which path ran is reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
import pandas as pd                                 # noqa: E402
from matplotlib.colors import Normalize             # noqa: E402

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in (str(_HERE), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import factors                                      # noqa: E402
from aggregate import cell_table, wilson            # noqa: E402

INK, MUTED, GRID = "#3d3d3a", "#8a8a85", "#e8e8e4"
V_CMAP = plt.get_cmap("viridis")
U_CMAP = plt.get_cmap("plasma")


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------
def _dummies(levels: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Treatment-coded dummies, first level as reference."""
    return np.column_stack([(values == lev).astype(float) for lev in levels[1:]])


def design(data: pd.DataFrame, terms: str) -> np.ndarray:
    """Design matrix for one rung of the ladder. Intercept always present."""
    u_levels = np.array(sorted(data["u_hat"].unique()))
    v_levels = np.array(sorted(data["v"].unique()))
    blocks = [np.ones((len(data), 1))]
    du = _dummies(u_levels, data["u_hat"].to_numpy())
    dv = _dummies(v_levels, data["v"].to_numpy())
    if "u" in terms:
        blocks.append(du)
    if "v" in terms:
        blocks.append(dv)
    if "x" in terms:
        blocks.append(np.column_stack(
            [du[:, i] * dv[:, j] for i in range(du.shape[1])
             for j in range(dv.shape[1])]))
    return np.column_stack(blocks)


def _irls_logistic(X: np.ndarray, y: np.ndarray, iters: int = 100):
    """Plain IRLS. Returns (fitted probabilities, converged)."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = np.clip(X @ beta, -30.0, 30.0)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1.0 - mu), 1e-9, None)
        z = eta + (y - mu) / w
        wx = X * w[:, None]
        try:
            step = np.linalg.solve(X.T @ wx, wx.T @ z)
        except np.linalg.LinAlgError:
            step, *_ = np.linalg.lstsq(X.T @ wx, wx.T @ z, rcond=None)
        if np.max(np.abs(step - beta)) < 1e-10:
            beta = step
            return 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30.0, 30.0))), True
        beta = step
    return 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30.0, 30.0))), False


def fit_logistic(X: np.ndarray, y: np.ndarray, backend: str):
    """Return (deviance, fitted probabilities, n_params)."""
    if backend == "statsmodels":
        import statsmodels.api as sm
        res = sm.GLM(y, X, family=sm.families.Binomial()).fit()
        mu = np.asarray(res.fittedvalues, dtype=float)
    else:
        mu, _ = _irls_logistic(X, y)
    mu = np.clip(mu, 1e-12, 1 - 1e-12)
    deviance = -2.0 * float(np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu)))
    return deviance, mu, int(np.linalg.matrix_rank(X))


def fit_ols(X: np.ndarray, y: np.ndarray):
    """Return (residual sum of squares, fitted values, rank)."""
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    return float(np.sum((y - fitted) ** 2)), fitted, int(rank)


LADDER = [("M0", ""), ("M1", "u"), ("M2", "v"), ("M3", "uv"), ("M4", "uvx")]


def deviance_ladder(data: pd.DataFrame, backend: str) -> dict:
    y = data["correct"].to_numpy(dtype=float)
    out = {}
    fitted = {}
    for name, terms in LADDER:
        X = design(data, terms)
        dev, mu, rank = fit_logistic(X, y, backend)
        out[name] = {"deviance": dev, "df": rank, "terms": terms or "1"}
        fitted[name] = mu
    total = out["M0"]["deviance"] - out["M4"]["deviance"]
    shares = {
        "between_cell_deviance": total,
        "u_hat_share": (out["M0"]["deviance"] - out["M1"]["deviance"]) / total
                       if total > 0 else float("nan"),
        "v_share_beyond_u_hat": (out["M1"]["deviance"] - out["M3"]["deviance"]) / total
                                if total > 0 else float("nan"),
        "interaction_share": (out["M3"]["deviance"] - out["M4"]["deviance"]) / total
                             if total > 0 else float("nan"),
    }
    return {"models": out, "shares": shares, "fitted_M3": fitted["M3"]}


def m3_residuals(data: pd.DataFrame, fitted_m3: np.ndarray) -> pd.DataFrame:
    """Observed minus additive-model cell proportion, in probability points."""
    frame = data[["v", "u_hat", "correct"]].copy()
    frame["fitted"] = fitted_m3
    grouped = frame.groupby(["v", "u_hat"]).agg(
        observed=("correct", "mean"), fitted=("fitted", "mean"),
        n=("correct", "size")).reset_index()
    grouped["residual_pp"] = 100.0 * (grouped["observed"] - grouped["fitted"])
    return grouped


def verdict(shares: dict) -> str:
    if not np.isfinite(shares["between_cell_deviance"]) or \
            shares["between_cell_deviance"] <= 0:
        return ("DEGENERATE: there is no between-cell deviance to apportion "
                "(every cell has the same accuracy, or the design has one cell). "
                "No verdict.")
    v_plus_x = shares["v_share_beyond_u_hat"] + shares["interaction_share"]
    if v_plus_x < 0.10:
        return ("COLLAPSE onto u_hat: v's main effect and the interaction together "
                f"explain {100 * v_plus_x:.1f} % of the between-cell deviance "
                "(< 10 %). u_hat is sufficient.")
    if shares["interaction_share"] < 0.10:
        return ("ADDITIVE: the interaction alone explains "
                f"{100 * shares['interaction_share']:.1f} % (< 10 %), but v carries "
                f"{100 * shares['v_share_beyond_u_hat']:.1f} % beyond u_hat. "
                "Performance is f(u_hat) + g(v); the curves are parallel, "
                "not coincident.")
    return ("INTERACTION regime: the interaction alone explains "
            f"{100 * shares['interaction_share']:.1f} % (>= 10 %). u_hat is NOT "
            "sufficient; read the M3 residual heatmap for where it fails.")


def time_ladder(data: pd.DataFrame) -> dict:
    """Same factor decomposition on log(t_commit), decided trials, OLS."""
    decided = data[data["decided"] & data["t_commit_fine"].notna()].copy()
    if decided.empty:
        return {"error": "no decided trials"}
    y = np.log(decided["t_commit_fine"].to_numpy(dtype=float))
    out = {}
    for name, terms in LADDER:
        rss, _, rank = fit_ols(design(decided, terms), y)
        out[name] = {"rss": rss, "df": rank, "terms": terms or "1"}
    total = out["M0"]["rss"] - out["M4"]["rss"]
    return {
        "models": out,
        "n_decided": int(len(decided)),
        "shares": {
            "between_cell_ss": total,
            "u_hat_share": (out["M0"]["rss"] - out["M1"]["rss"]) / total
                           if total > 0 else float("nan"),
            "v_share_beyond_u_hat": (out["M1"]["rss"] - out["M3"]["rss"]) / total
                                    if total > 0 else float("nan"),
            "interaction_share": (out["M3"]["rss"] - out["M4"]["rss"]) / total
                                 if total > 0 else float("nan"),
        },
        "caveat": (
            "log(t_commit) is fitted on DECIDED trials only. A trial that never "
            "arrived within T_max has no commitment time, and cells differ in how "
            "often that happens, so this decomposition is conditioned on a "
            "selection that is itself an outcome. Read it next to the "
            "decided_frac heatmap, never on its own."
        ),
    }


# ---------------------------------------------------------------------------
# Paired contrasts (Section 10, optional)
# ---------------------------------------------------------------------------
def paired_contrasts(data: pd.DataFrame) -> pd.DataFrame:
    """McNemar-style contrasts between adjacent u_hat levels at fixed v.

    Valid only because every cell ran the identical seed list, so a trial can be
    matched across cells by its seed.
    """
    rows = []
    u_levels = sorted(data["u_hat"].unique())
    for v in sorted(data["v"].unique()):
        block = data[data["v"] == v]
        for lo, hi in zip(u_levels[:-1], u_levels[1:]):
            a = block[block["u_hat"] == lo].set_index("seed")["correct"]
            b = block[block["u_hat"] == hi].set_index("seed")["correct"]
            shared = a.index.intersection(b.index)
            if len(shared) == 0:
                continue
            av, bv = a.loc[shared].to_numpy(bool), b.loc[shared].to_numpy(bool)
            n01 = int(np.sum(~av & bv))     # wrong at lo, right at hi
            n10 = int(np.sum(av & ~bv))
            disc = n01 + n10
            # Exact binomial two-sided p on the discordant pairs.
            if disc:
                from scipy.stats import binomtest
                p = float(binomtest(min(n01, n10), disc, 0.5).pvalue)
            else:
                p = float("nan")
            rows.append({
                "v": v, "u_hat_lo": lo, "u_hat_hi": hi, "n_paired": len(shared),
                "acc_lo": float(av.mean()), "acc_hi": float(bv.mean()),
                "delta_acc": float(bv.mean() - av.mean()),
                "n_lo_only": n10, "n_hi_only": n01, "n_discordant": disc,
                "mcnemar_p": p,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_color(INK)


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    {out_dir / name}.png / .pdf")


def _v_colour(v, v_levels):
    return V_CMAP(0.08 + 0.84 * (v_levels.index(v) / max(len(v_levels) - 1, 1)))


def plot_accuracy(cells: pd.DataFrame, out_dir: Path) -> None:
    """Plot 1: the collapse plot. Under H_collapse every line coincides."""
    v_levels = sorted(cells["v"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
    for ax, (col, lo, hi, title) in zip(axes, [
            ("acc_all", "acc_all_lo", "acc_all_hi",
             "acc_all — undecided scored as error (0-1 terminal loss)"),
            ("acc_decided", "acc_decided_lo", "acc_decided_hi",
             "acc_decided — conditioned on arriving (a selection effect)")]):
        for v in v_levels:
            sub = cells[cells["v"] == v].sort_values("u_hat")
            colour = _v_colour(v, v_levels)
            ax.errorbar(sub["u_hat"], sub[col],
                        yerr=[sub[col] - sub[lo], sub[hi] - sub[col]],
                        marker="o", ms=4, lw=1.4, capsize=2, color=colour,
                        label=f"v = {v}", zorder=3)
        ax.axvline(1.0, color=MUTED, lw=0.9, ls="--", zorder=1)
        ax.axhline(0.5, color=MUTED, lw=0.9, ls=":", zorder=1)
        ax.set_xlabel(r"$\hat{u} = u / u^*(v)$", color=INK)
        ax.set_title(title, color=INK, fontsize=10)
        _style(ax)
    axes[0].set_ylabel("accuracy", color=INK)
    axes[1].legend(fontsize=8, ncol=2, frameon=False,
                   loc="lower left", bbox_to_anchor=(1.01, 0.0))
    fig.suptitle("Accuracy against relative coupling, one line per kernel shape "
                 "(95 % Wilson intervals)", color=INK, fontsize=11)
    _save(fig, out_dir, "01_accuracy_vs_uhat")


def plot_time(cells: pd.DataFrame, out_dir: Path) -> None:
    """Plot 2: median commitment time against u_hat, log y."""
    v_levels = sorted(cells["v"].unique())
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for v in v_levels:
        sub = cells[cells["v"] == v].sort_values("u_hat")
        ax.plot(sub["u_hat"], sub["t_commit_fine_median"], marker="o", ms=4,
                lw=1.4, color=_v_colour(v, v_levels), label=f"v = {v}", zorder=3)
    ax.axvline(1.0, color=MUTED, lw=0.9, ls="--", zorder=1)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\hat{u} = u / u^*(v)$", color=INK)
    ax.set_ylabel("median commitment time (ticks, decided trials)", color=INK)
    ax.set_title("Commitment time against relative coupling", color=INK, fontsize=11)
    ax.legend(fontsize=8, ncol=2, frameon=False)
    _style(ax)
    _save(fig, out_dir, "02_time_vs_uhat")


def plot_plane(cells: pd.DataFrame, out_dir: Path) -> None:
    """Plot 3: the speed-accuracy plane, coloured by u_hat, labelled by v."""
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    norm = Normalize(vmin=cells["u_hat"].min(), vmax=cells["u_hat"].max())
    ax.scatter(cells["t_commit_fine_median"], cells["acc_all"],
               c=cells["u_hat"], cmap=U_CMAP, norm=norm, s=46,
               edgecolor="white", linewidth=0.5, zorder=3)
    for _, row in cells.iterrows():
        ax.annotate(f"{row['v']:g}",
                    (row["t_commit_fine_median"], row["acc_all"]),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=6.5, color=MUTED)
    ax.axhline(0.5, color=MUTED, lw=0.9, ls=":", zorder=1)
    ax.annotate("chance", (ax.get_xlim()[0], 0.5), xytext=(3, 4),
                textcoords="offset points", fontsize=8, color=MUTED)
    span = cells["t_commit_fine_median"].to_numpy(dtype=float)
    lo, hi = np.nanmin(span), np.nanmax(span)
    if np.isfinite(lo) and np.isfinite(hi) and hi / max(lo, 1e-9) > 4:
        ax.set_xscale("log")
    ax.set_xlabel("median commitment time (ticks)", color=INK)
    ax.set_ylabel("acc_all", color=INK)
    ax.set_title("Speed-accuracy plane: colour is $\\hat{u}$, label is $v$",
                 color=INK, fontsize=11)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=U_CMAP), ax=ax,
                 label=r"$\hat{u}$")
    _style(ax)
    _save(fig, out_dir, "03_speed_accuracy_plane")


def plot_heatmaps(cells: pd.DataFrame, residuals: pd.DataFrame,
                  out_dir: Path) -> None:
    """Plot 4: five (u_hat, v) heatmaps."""
    panels = [
        ("acc_all", cells, "acc_all", "viridis", None),
        ("acc_decided", cells, "acc_decided", "viridis", None),
        ("decided_frac", cells, "decided_frac", "cividis", None),
        ("median t_commit (ticks)", cells, "t_commit_fine_median", "magma", None),
        ("M3 residual (probability points)", residuals, "residual_pp", "RdBu_r",
         "symmetric"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(24.0, 4.6))
    for ax, (title, frame, col, cmap, scale) in zip(axes, panels):
        grid = frame.pivot(index="v", columns="u_hat", values=col).sort_index()
        kwargs = {}
        if scale == "symmetric":
            values = np.abs(grid.to_numpy(dtype=float))
            lim = float(np.nanmax(values)) if np.any(np.isfinite(values)) else 1.0
            lim = lim if lim > 0 else 1.0
            kwargs = {"vmin": -lim, "vmax": lim}
        image = ax.imshow(grid.to_numpy(), aspect="auto", origin="lower",
                          cmap=cmap, **kwargs)
        ax.set_xticks(range(len(grid.columns)))
        ax.set_xticklabels([f"{c:g}" for c in grid.columns], fontsize=8)
        ax.set_yticks(range(len(grid.index)))
        ax.set_yticklabels([f"{i:g}" for i in grid.index], fontsize=8)
        ax.set_xlabel(r"$\hat{u}$", color=INK)
        ax.set_ylabel("v", color=INK)
        ax.set_title(title, color=INK, fontsize=10)
        fig.colorbar(image, ax=ax, fraction=0.046)
        ax.tick_params(colors=MUTED)
    fig.suptitle("Cell-wise structure over the (u_hat, v) factorial",
                 color=INK, fontsize=12)
    # Each panel carries its own colourbar; without extra room the next panel's
    # y-label lands on top of it.
    fig.subplots_adjust(wspace=0.45)
    _save(fig, out_dir, "04_heatmaps")


def plot_u_star(results_root: Path, cells: pd.DataFrame, out_dir: Path) -> None:
    """Plot 5: u*(v), with the 6.157 anchor marked."""
    table_path = results_root / "u_star_table.csv"
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    if table_path.is_file():
        table = pd.read_csv(table_path)
        ax.plot(table["v"], table["u_star"], lw=1.6, color="#1c5cab",
                label=r"$u^*(v) = 1 / (\lambda_{max}(W(v))\,\mathrm{sech}^2\beta)$",
                zorder=2)
    design_points = cells.drop_duplicates("v")[["v", "u_star"]].sort_values("v")
    ax.scatter(design_points["v"], design_points["u_star"], s=28, zorder=3,
               color="#0d366b", label="design grid")
    ax.scatter([factors.ANCHOR_V], [factors.ANCHOR_U_STAR], marker="*", s=180,
               color="#c1440e", zorder=4,
               label=f"anchor: $u^*({factors.ANCHOR_V}) = {factors.ANCHOR_U_STAR}$")
    ax.set_yscale("log")
    ax.set_xlabel("kernel shape $v$", color=INK)
    ax.set_ylabel("critical coupling $u^*$", color=INK)
    ax.set_title("Critical coupling against kernel shape", color=INK, fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    _save(fig, out_dir, "05_u_star")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_ladder(title: str, models: dict, key: str, shares: dict) -> None:
    print(f"\n  {title}")
    print(f"    {'model':<5} {'terms':<22} {'df':>4} {key:>16}   drop from previous")
    previous = None
    for name, _ in LADDER:
        value = models[name][key]
        drop = "" if previous is None else f"{previous - value:16.3f}"
        print(f"    {name:<5} {models[name]['terms']:<22} "
              f"{models[name]['df']:>4} {value:16.3f}   {drop}")
        previous = value
    print(f"    u_hat share            : {100 * shares['u_hat_share']:6.2f} %")
    print(f"    v share beyond u_hat   : {100 * shares['v_share_beyond_u_hat']:6.2f} %")
    print(f"    interaction share      : {100 * shares['interaction_share']:6.2f} %")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results-root", type=Path,
                    default=_ROOT / "results" / "uhat_v_sweep")
    ap.add_argument("--trials", type=Path, default=None,
                    help="default: <results-root>/trials.parquet")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <results-root>/analysis")
    ap.add_argument("--paired", action="store_true",
                    help="also run paired McNemar-style contrasts between "
                         "adjacent u_hat levels at fixed v")
    ap.add_argument("--backend", choices=("auto", "statsmodels", "numpy"),
                    default="auto")
    args = ap.parse_args(argv)

    root = args.results_root.resolve()
    trials_path = args.trials
    if trials_path is None:
        trials_path = root / "trials.parquet"
        if not trials_path.is_file():
            trials_path = root / "trials.csv"
    if not trials_path.is_file():
        raise SystemExit(f"No trials file at {trials_path}. Run aggregate.py first.")
    out_dir = args.out_dir or (root / "analysis")

    data = (pd.read_parquet(trials_path) if trials_path.suffix == ".parquet"
            else pd.read_csv(trials_path))
    for col in ("decided", "correct", "timeout", "numerical_failure"):
        data[col] = data[col].fillna(False).astype(bool)

    backend = args.backend
    if backend == "auto":
        try:
            import statsmodels.api  # noqa: F401
            backend = "statsmodels"
        except ImportError:
            backend = "numpy"

    cells = cell_table(data)
    n_u, n_v = data["u_hat"].nunique(), data["v"].nunique()

    print(f"(u_hat, v) factorial — collapse analysis of {trials_path}")
    print(f"  trials             : {len(data)}")
    print(f"  cells              : {len(cells)}  ({n_u} u_hat x {n_v} v)")
    print(f"  decided            : {int(data['decided'].sum())} "
          f"({data['decided'].mean():.4f})")
    print(f"  numerical failures : {int(data['numerical_failure'].sum())}")
    print(f"  GLM backend        : {backend}")
    hot = cells[cells["numerical_failure_frac"] > 0.01]
    if len(hot):
        print("\n  " + "!" * 66)
        print("  !! Cells with numerical_failure_frac > 1 % — Section 11 says read")
        print("  !! these with the integrator in mind, not the dynamics:")
        for _, row in hot.iterrows():
            print(f"  !!   v = {row['v']:<4} u_hat = {row['u_hat']:<5} "
                  f"frac = {row['numerical_failure_frac']:.3f}")
        print("  " + "!" * 66)
    if n_u * n_v != len(cells):
        print(f"\n  NOTE: the design is UNBALANCED — {len(cells)} cells present of "
              f"{n_u * n_v} in the crossing. Excluded cells make the shares below "
              "conditional on the cells that ran.")

    if len(cells) < 2 or n_u < 2 or n_v < 2:
        print("\n  Fewer than two levels on a factor: the deviance ladder is not "
              "identified. Plots only.\n  (Expected on smoke output.)")
        ladder = None
    else:
        ladder = deviance_ladder(data, backend)
        print_ladder("Deviance ladder — logistic on `correct` (trial level)",
                     ladder["models"], "deviance", ladder["shares"])

    residuals = None
    if ladder is not None:
        residuals = m3_residuals(data, ladder["fitted_M3"])
        worst = residuals.loc[residuals["residual_pp"].abs().idxmax()]
        print(f"    max |M3 cell residual| : {abs(worst['residual_pp']):6.2f} "
              f"probability points  (v = {worst['v']}, u_hat = {worst['u_hat']}; "
              f"observed {worst['observed']:.4f} vs additive {worst['fitted']:.4f})")
        print()
        print("  VERDICT: " + verdict(ladder["shares"]))

        times = time_ladder(data)
        if "error" not in times:
            print_ladder(
                f"Sum-of-squares ladder — OLS on log(t_commit), "
                f"{times['n_decided']} decided trials",
                times["models"], "rss", times["shares"])
            print(f"    CAVEAT: {times['caveat']}")
        else:
            times = None
    else:
        times = None

    print("\n  Plots:")
    plot_accuracy(cells, out_dir)
    plot_time(cells, out_dir)
    plot_plane(cells, out_dir)
    if residuals is not None:
        plot_heatmaps(cells, residuals, out_dir)
    plot_u_star(root, cells, out_dir)

    paired = None
    if args.paired:
        paired = paired_contrasts(data)
        paired.to_csv(out_dir / "paired_contrasts.csv", index=False)
        print(f"\n  Paired contrasts (adjacent u_hat at fixed v), "
              f"{len(paired)} rows -> {out_dir / 'paired_contrasts.csv'}")
        if len(paired):
            print(paired.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    summary = {
        "trials_file": str(trials_path),
        "n_trials": int(len(data)),
        "n_cells": int(len(cells)),
        "backend": backend,
        "accuracy_ladder": None if ladder is None else {
            "models": ladder["models"], "shares": ladder["shares"],
            "verdict": verdict(ladder["shares"]),
            "max_abs_m3_residual_pp": (
                float(residuals["residual_pp"].abs().max())
                if residuals is not None else None),
        },
        "time_ladder": times,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "collapse_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)
    cells.to_csv(out_dir / "cells_analysis.csv", index=False)
    if residuals is not None:
        residuals.to_csv(out_dir / "m3_residuals.csv", index=False)
    print(f"\n  wrote {out_dir / 'collapse_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
