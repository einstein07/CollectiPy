#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Sensory-map reduction experiment (max-sensory-map-spec.md Sections 6 and 7.3).

    maps       (a) plot the sum and max maps for the 6.1 geometry at each Delta_close
    manifest   write <results-root>/manifest.json: cells, paired seeds, locked settings
    run        (b) run the 6.3 design: --index I (SLURM array), --cell-id K [--chunk J],
               or --all [--workers W] (local, parallel over cells)
    aggregate  (c) raw/ -> trials.parquet|csv, cells.csv (per-cell summary),
               paired_contrasts.csv
    plot       (d) P(mean_of_pair) and P(arrive at C) vs Delta_close, both modes
    sanity     6.6: the two-target standard condition, sum vs max, n paired trials,
               at the template's sigma AND the spec's sigma; runs + summarises
    report     copy the summary tables and figures into scripts/map_reduction/results/
               (the tracked results note's data) and print the markdown summary table
    all        maps + manifest + run --all + aggregate + plot   (--smoke: n = 20)

Everything lands under --results-root (default results/map_reduction; --smoke
appends /smoke). Reads and writes one file per task, idempotently, exactly like
scripts/uhat_v_sweep; the SLURM array is slurm/map_reduction.sbatch.

    PY=.venv/bin/python
    $PY scripts/compare_map_reduction.py all --smoke --workers 16      # local pipeline
    $PY scripts/compare_map_reduction.py manifest                      # full design
    $PY scripts/compare_map_reduction.py run --all --workers 16        # or the array
    $PY scripts/compare_map_reduction.py aggregate && $PY scripts/compare_map_reduction.py plot
    $PY scripts/compare_map_reduction.py sanity --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_HERE), str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from map_reduction import aggregate as agg          # noqa: E402
from map_reduction import config_patch, factors, plots, run_cell  # noqa: E402

DEFAULT_ROOT = _ROOT / "results" / "map_reduction"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _results_root(args) -> Path:
    root = Path(args.results_root).resolve()
    if getattr(args, "smoke", False) and root.name != "smoke":
        root = root / "smoke"
    return root


def _load_manifest(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Manifest not found: {path}\nRun `compare_map_reduction.py manifest` first.")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(obj, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


def _print_manifest(manifest: dict, path: Path) -> None:
    print("Sensory-map reduction experiment — manifest")
    print(f"  spec            : {manifest['spec']}")
    print(f"  git sha         : {manifest['git_sha']}")
    print(f"  base config     : {manifest['base_config']}")
    print(f"  cells           : {manifest['n_cells']}  ({len(manifest['reductions'])} reductions x "
          f"{len(manifest['delta_close_deg'])} Delta_close x {len(manifest['q_c_rel'])} q_C)")
    print(f"  trials per cell : {manifest['n_trials']}  (total {manifest['n_cells'] * manifest['n_trials']})")
    print(f"  seeds           : {manifest['base_seed']} .. {manifest['base_seed'] + manifest['n_trials'] - 1}, "
          "identical in every cell (arena seed = percept-stream seed)")
    print(f"  T_max           : {manifest['t_max_ticks']} ticks")
    print(f"  quality scale   : {manifest['quality_scale']} (config strength = scale x relative q)")
    print(f"  Delta*          : {manifest['merge_threshold_deg']:.2f} deg (kappa = {manifest['locked']['kappa']})")
    print(f"  locked          : {json.dumps(manifest['locked'], sort_keys=True)}")
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------
def cmd_maps(args) -> int:
    root = _results_root(args)
    paths = plots.plot_maps(root / "figures", q_c_rel=args.q_c)
    for p in paths:
        print(f"  wrote {p}")
    return 0


def cmd_manifest(args) -> int:
    root = _results_root(args)
    path = root / "manifest.json"
    if path.exists() and not args.force:
        raise SystemExit(f"{path} exists; re-keying a sweep mid-flight is not something to do "
                         "by accident. Pass --force to overwrite.")
    n = args.trials or (factors.N_TRIALS_SMOKE if args.smoke else factors.N_TRIALS_FULL)
    manifest = config_patch.build_manifest(n)
    _write_json(manifest, path)
    _print_manifest(manifest, path)
    return 0


def _task_args(args, root: Path, manifest: dict):
    per_chunk = args.trials_per_chunk or int(manifest["n_trials"])
    chunks = run_cell.n_chunks(manifest["n_trials"], per_chunk)
    return per_chunk, chunks


def _run_commands_locally(commands: list[tuple[str, list[str]]], workers: int,
                          log_dir: Path) -> int:
    """Run `commands` (label, argv) as separate processes, at most `workers` at a
    time — one process per task, exactly as the SLURM array runs them. The
    simulator forks its own agent processes, which a daemonic pool worker may
    not do; a plain subprocess may. Each task's stdout/stderr goes to a log file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = list(commands)
    running: dict = {}
    status = 0
    n_done = 0
    while pending or running:
        while pending and len(running) < workers:
            label, argv = pending.pop(0)
            log = open(log_dir / f"{label}.log", "w", encoding="utf-8")
            proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, cwd=str(_ROOT))
            running[proc.pid] = (label, proc, log)
        for pid, (label, proc, log) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close()
            del running[pid]
            n_done += 1
            status |= int(code != 0)
            tail = ""
            try:
                lines = [ln for ln in (log_dir / f"{label}.log").read_text(encoding="utf-8").splitlines()
                         if " done:" in ln or "already complete" in ln]
                tail = lines[-1].strip() if lines else ""
            except OSError:
                pass
            print(f"  [{n_done}/{len(commands)}] {label}: exit {code}  {tail}")
            sys.stdout.flush()
        time.sleep(0.2)
    return status


def _passthrough(args) -> list[str]:
    out = ["--format", args.format]
    if args.force:
        out.append("--force")
    if args.scratch:
        out += ["--scratch", str(args.scratch)]
    if args.subprocess:
        out.append("--subprocess")
    if getattr(args, "keep_raw", False):
        out.append("--keep-raw")
    return out


def cmd_run(args) -> int:
    root = _results_root(args)
    manifest_path = Path(args.manifest) if args.manifest else root / "manifest.json"
    manifest = _load_manifest(manifest_path)
    per_chunk, chunks = _task_args(args, root, manifest)
    n_tasks = int(manifest["n_cells"]) * chunks

    if args.all:
        workers = max(1, int(args.workers))
        print(f"Running {n_tasks} tasks ({manifest['n_cells']} cells x {chunks} chunks, "
              f"{manifest['n_trials']} trials/cell) on {workers} workers -> {root}")
        base = [sys.executable, str(Path(__file__).resolve()), "run",
                "--results-root", str(root), "--manifest", str(manifest_path)]
        if args.trials_per_chunk:
            base += ["--trials-per-chunk", str(args.trials_per_chunk)]
        base += _passthrough(args)
        commands = [(f"task_{i:04d}", base + ["--index", str(i)]) for i in range(n_tasks)]
        t0 = time.time()
        status = _run_commands_locally(commands, workers, root / "local_logs")
        print(f"all tasks done in {time.time() - t0:.0f}s (status {status})")
        return status

    if args.index is not None:
        cell_id, chunk = divmod(int(args.index), chunks)
    elif args.cell_id is not None:
        cell_id, chunk = int(args.cell_id), int(args.chunk)
    else:
        raise SystemExit("pass --index, --cell-id (with --chunk), or --all")
    cell = run_cell.find_cell(manifest, cell_id)
    if cell.get("excluded"):
        print(f"[cell {cell_id}:{chunk}] excluded in the manifest; nothing to do")
        return 0
    trials = run_cell.chunk_trials(chunk, int(cell["n_trials"]), per_chunk)
    code, _ = run_cell.run_task(cell, trials, root, chunk, fmt=args.format, force=args.force,
                                scratch=(Path(args.scratch) if args.scratch else None),
                                subprocess_mode=args.subprocess, keep_raw=args.keep_raw)
    return code


def cmd_aggregate(args) -> int:
    root = _results_root(args)
    manifest = _load_manifest(Path(args.manifest) if args.manifest else root / "manifest.json")
    report = agg.aggregate(root, manifest, allow_incomplete=args.allow_incomplete)
    import pandas as pd
    cells = pd.read_csv(root / "cells.csv")
    agg.print_cells(cells)
    print(f"  rows {report['n_rows']}, cells {report['n_cells_present']}, "
          f"numerical failures {report['n_failures']}")
    return 0


def cmd_plot(args) -> int:
    root = _results_root(args)
    import pandas as pd
    cells_path = root / "cells.csv"
    if not cells_path.is_file():
        raise SystemExit(f"{cells_path} missing; run `aggregate` first")
    cells = pd.read_csv(cells_path)
    n = int(cells["n"].max()) if len(cells) else None
    for p in plots.plot_outcomes(cells, root / "figures", n_trials=n):
        print(f"  wrote {p}")
    for p in plots.plot_extras(cells, root / "figures", n_trials=n):
        print(f"  wrote {p}")
    return 0


def cmd_sanity_arm(args) -> int:
    """One arm of the 6.6 check in this process (the `sanity` driver's worker)."""
    root = _results_root(args)
    code, _ = run_cell.run_sanity_task(args.variant, args.reduction, range(int(args.trials)),
                                       root, 0, fmt=args.format, force=args.force,
                                       scratch=(Path(args.scratch) if args.scratch else None),
                                       subprocess_mode=args.subprocess)
    return code


def cmd_sanity(args) -> int:
    root = _results_root(args)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in factors.SANITY_VARIANTS:
            raise SystemExit(f"unknown variant {v!r}; choose from {list(factors.SANITY_VARIANTS)}")
    base = [sys.executable, str(Path(__file__).resolve()), "sanity-arm",
            "--results-root", str(root), "--trials", str(int(args.trials))] + _passthrough(args)
    commands = [(f"sanity_{v}_{r}", base + ["--variant", v, "--reduction", r])
                for v in variants for r in factors.REDUCTIONS]
    print(f"6.6 sanity check: {len(commands)} arms x {args.trials} paired trials -> {root / 'sanity'}")
    t0 = time.time()
    status = _run_commands_locally(commands, max(1, int(args.workers)), root / "local_logs")
    print(f"  arms done in {time.time() - t0:.0f}s (status {status})")
    arms, pairs = agg.aggregate_sanity(root)
    import pandas as pd
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(arms.to_string(index=False))
        print(pairs.to_string(index=False))
    return 0


def _markdown_table(cells) -> str:
    """The 6.4 per-cell summary as a markdown table (one row per cell)."""
    cols = [("reduction", "reduction"), ("delta_close_deg", "Δ_close"), ("q_c_rel", "q_C"),
            ("n", "n"), ("p_mean_of_pair", "P(mean of pair)"), ("p_merged_t2", "P(merged, t2)"),
            ("p_bimodal_t2", "P(bimodal, t2)"), ("p_arrive_C", "P(arrive C)"),
            ("p_arrive_AB", "P(arrive A|B)"), ("p_timeout", "P(timeout)"),
            ("p_no_commit", "P(no commit)"), ("median_t_commit", "med. t_commit"),
            ("median_t_arrival", "med. t_arrival")]
    lines = ["| " + " | ".join(h for _, h in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in cells.sort_values(["reduction", "delta_close_deg", "q_c_rel"],
                                    ascending=[False, True, True]).iterrows():
        cells_ = []
        for key, _ in cols:
            v = row[key]
            if key in ("reduction",):
                cells_.append(str(v))
            elif key in ("delta_close_deg",):
                cells_.append(f"{v:.0f}°")
            elif key in ("q_c_rel",):
                cells_.append(f"{v:.2f}")
            elif key in ("n",):
                cells_.append(f"{int(v)}")
            elif key.startswith("median"):
                cells_.append("–" if v != v else f"{v:.1f}")
            else:
                cells_.append(f"{v:.3f}")
        lines.append("| " + " | ".join(cells_) + " |")
    return "\n".join(lines)


def cmd_report(args) -> int:
    import shutil
    import pandas as pd
    root = _results_root(args)
    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for rel in ("cells.csv", "paired_contrasts.csv", "manifest.json", "aggregate_report.json",
                "sanity/sanity_arms.csv", "sanity/sanity_paired.csv",
                "figures/maps_sum_vs_max.png", "figures/maps_sum_vs_max.pdf",
                "figures/outcomes_vs_delta.png", "figures/outcomes_vs_delta.pdf",
                "figures/outcomes_extra_vs_delta.png", "figures/outcomes_extra_vs_delta.pdf"):
        src = root / rel
        if src.is_file():
            shutil.copyfile(src, dest / src.name)
            copied.append(src.name)
    cells = pd.read_csv(root / "cells.csv")
    table = _markdown_table(cells)
    (dest / "summary_table.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\n  copied {', '.join(copied)} -> {dest}")
    return 0


def cmd_all(args) -> int:
    root = _results_root(args)
    if not (root / "manifest.json").exists() or args.force:
        args.force = True
        cmd_manifest(args)
    cmd_maps(args)
    args.all = True
    args.index = None
    args.cell_id = None
    code = cmd_run(args)
    args.allow_incomplete = True
    cmd_aggregate(args)
    cmd_plot(args)
    return code


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p, smoke=True):
        p.add_argument("--results-root", type=Path, default=DEFAULT_ROOT)
        if smoke:
            p.add_argument("--smoke", action="store_true",
                           help=f"n = {factors.N_TRIALS_SMOKE} trials/cell under <results-root>/smoke")

    p = sub.add_parser("maps", help="(a) sum vs max maps for the 6.1 geometry")
    common(p)
    p.add_argument("--q-c", type=float, default=1.02, help="relative q_C drawn (default 1.02)")
    p.set_defaults(func=cmd_maps)

    p = sub.add_parser("manifest", help="write the manifest")
    common(p)
    p.add_argument("--trials", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_manifest)

    def run_opts(p):
        p.add_argument("--manifest", type=Path, default=None)
        p.add_argument("--index", type=int, default=None, help="flat array index = cell_id * n_chunks + chunk")
        p.add_argument("--cell-id", type=int, default=None)
        p.add_argument("--chunk", type=int, default=0)
        p.add_argument("--all", action="store_true", help="run every task locally")
        p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
        p.add_argument("--trials-per-chunk", type=int, default=None)
        p.add_argument("--format", choices=("parquet", "csv"), default="parquet")
        p.add_argument("--force", action="store_true")
        p.add_argument("--scratch", type=str, default=None)
        p.add_argument("--subprocess", action="store_true")
        p.add_argument("--keep-raw", action="store_true")

    p = sub.add_parser("run", help="(b) run the design")
    common(p)
    run_opts(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("aggregate", help="(c) per-trial and per-cell tables")
    common(p)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--allow-incomplete", action="store_true")
    p.set_defaults(func=cmd_aggregate)

    p = sub.add_parser("plot", help="(d) outcome plots")
    common(p)
    p.set_defaults(func=cmd_plot)

    p = sub.add_parser("sanity", help="6.6 two-target check, sum vs max")
    common(p)
    p.add_argument("--trials", type=int, default=factors.SANITY_N_TRIALS)
    p.add_argument("--variants", type=str, default=",".join(factors.SANITY_VARIANTS))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--format", choices=("parquet", "csv"), default="parquet")
    p.add_argument("--force", action="store_true")
    p.add_argument("--scratch", type=str, default=None)
    p.add_argument("--subprocess", action="store_true")
    p.set_defaults(func=cmd_sanity)

    p = sub.add_parser("report", help="copy tables + figures to scripts/map_reduction/results/")
    common(p)
    p.add_argument("--dest", type=Path, default=_HERE / "map_reduction" / "results")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("sanity-arm", help=argparse.SUPPRESS)
    common(p)
    p.add_argument("--variant", required=True)
    p.add_argument("--reduction", required=True)
    p.add_argument("--trials", type=int, default=factors.SANITY_N_TRIALS)
    p.add_argument("--format", choices=("parquet", "csv"), default="parquet")
    p.add_argument("--force", action="store_true")
    p.add_argument("--scratch", type=str, default=None)
    p.add_argument("--subprocess", action="store_true")
    p.set_defaults(func=cmd_sanity_arm)

    p = sub.add_parser("all", help="maps + manifest + run --all + aggregate + plot")
    common(p)
    p.add_argument("--trials", type=int, default=None)
    p.add_argument("--q-c", type=float, default=1.02)
    run_opts(p)
    p.set_defaults(func=cmd_all)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
