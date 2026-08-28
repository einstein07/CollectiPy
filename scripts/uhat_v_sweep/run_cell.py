#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Step 3: run one (cell, chunk) of the (u_hat, v) factorial.

Sections 6 and 7 of `uhat-v-factorial-experiment.md`. This is the ONLY program
the SLURM array runs.

    python3 scripts/uhat_v_sweep/run_cell.py --cell-id K --chunk J \
        --trials-per-chunk M --results-root <R> [--smoke] [--force] [--dt-override X]

Behaviour:

- **One output file per array task.** `raw/cell<K>_chunk<J>.parquet` (or `.csv`).
  Nothing ever appends to a shared file: a previous sweep lost data to
  concurrent-write CSV corruption and this is a hard requirement.
- **Idempotent.** A complete output file makes the task exit 0 immediately;
  `--force` overrides.
- **Scratch staging.** Every trial's run archive is written to node-local scratch
  (`$TMPDIR`, or `--scratch`) and deleted with it. Exactly one file per task is
  published to shared storage, atomically (temp name in the destination
  directory, then rename).
- **In-process trials.** Interpreter startup plus imports (~0.5 s) is a third of a
  trial's runtime, so the simulator is imported once and each trial builds a fresh
  Config + Environment. `--subprocess` falls back to one `src/main.py` per trial;
  the paths are equivalent because every source of randomness is seeded from the
  per-trial config, never from process state.
- **A failed trial is DATA, not an abort.** A trial that raises (the usual cause
  is the ring diverging at high u, which `MeanFieldSystem.step` turns into a
  RuntimeError inside the agent process, which `Environment.start` re-raises) is
  written out with `numerical_failure = True` and the sweep continues.

Scoring (see RECON.md items 2-3, and the discrepancy note on `t_commit`):

    decided        the agent arrived inside `termination.radius` of a target
                   within T_max ticks
    choice         the target it arrived at
    correct        choice == static_0.s#0, the 5.00-quality target (0-1 loss)
    t_commit_ticks the arrival tick -- the "decision time" of every prior RA
                   analysis in this repo, and the quantity the 11-tick anchor in
                   Section 12 refers to
    t_commit_fine  the same crossing, resolved below the tick by interpolating
                   the logged trajectory against the termination radius
    t_bif_ticks    the BifurcationDetector's own commitment event, kept as a
                   secondary record; it is NOT used for scoring (RECON item 3)
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in (str(_HERE), str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import factors            # noqa: E402
import config_patch       # noqa: E402


#: Row schema (Section 6). Append-only; downstream code indexes by name.
FIELDS = [
    "cell_id", "v", "u_hat", "u_star", "u",
    "trial_idx", "seed",
    "decided", "choice", "correct",
    "t_commit_ticks", "t_commit_fine", "t_arrival_s",
    "timeout", "numerical_failure", "max_abs_state",
    "t_bif_ticks", "bif_target", "bif_agrees_with_choice",
    "n_ticks_logged", "final_x", "final_y", "final_dist",
    "integration_dt", "error",
    "git_sha", "config_hash",
]

_BOOL_FIELDS = {"decided", "correct", "timeout", "numerical_failure",
                "bif_agrees_with_choice"}


# ---------------------------------------------------------------------------
# Run-archive -> one row
# ---------------------------------------------------------------------------
def _target_positions(cfg: dict) -> dict[str, tuple[float, float]]:
    objects = cfg["environment"]["objects"]
    return {
        f"{name}.s#0": (float(obj["position"][0][0]), float(obj["position"][0][1]))
        for name, obj in objects.items()
    }


def _first_crossing(rows, targets, radius):
    """First tick at which the agent is inside `radius` of a target.

    Returns (tick, tick_fine, target_id) or (None, None, None).

    `tick_fine` resolves the crossing BELOW the control tick by intersecting the
    straight segment between the two logged positions with the termination
    circle. This is arithmetic on the logged trajectory only: no dynamics, no
    motor command and no decision is touched by it. It is the sub-tick
    refinement Section 6 asks for; the readout-side refinement it proposes does
    not apply here (RECON item 3).
    """
    prev = None
    for row in rows:
        tick = int(row["tick"])
        px, py = float(row["pos_x"]), float(row["pos_y"])
        hit = None
        best = float("inf")
        for tid, (tx, ty) in targets.items():
            dist = math.hypot(tx - px, ty - py)
            if dist <= radius + 1e-9 and dist < best:
                hit, best = tid, dist
        if hit is not None:
            fine = float(tick)
            if prev is not None:
                s = _segment_circle_fraction(prev, (px, py), targets[hit], radius)
                if s is not None:
                    fine = float(prev[2]) + s * (tick - prev[2])
            return tick, fine, hit
        prev = (px, py, tick)
    return None, None, None


def _segment_circle_fraction(p0, p1, centre, radius):
    """Smallest s in [0, 1] with |p0 + s (p1 - p0) - centre| = radius, else None."""
    (x0, y0, _t0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    fx, fy = x0 - centre[0], y0 - centre[1]
    a = dx * dx + dy * dy
    if a <= 0.0:
        return None
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - radius * radius
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    root = math.sqrt(disc)
    for s in sorted(((-b - root) / (2.0 * a), (-b + root) / (2.0 * a))):
        if -1e-9 <= s <= 1.0 + 1e-9:
            return min(max(s, 0.0), 1.0)
    return None


def _state_extremes(zf, names):
    """(max |z_i| over all logged ticks, saw_nan_or_inf) from the neural log."""
    member = next((n for n in names if n.endswith("_neural.csv")), None)
    if member is None:
        return float("nan"), False
    peak, bad = 0.0, False
    with zf.open(member) as raw:
        reader = csv.reader(io.TextIOWrapper(raw))
        header = next(reader, None)
        if not header:
            return float("nan"), False
        cols = [i for i, h in enumerate(header) if h.startswith("neuron_")]
        for row in reader:
            for i in cols:
                if i >= len(row):
                    continue
                try:
                    value = float(row[i])
                except ValueError:
                    bad = True
                    continue
                if math.isnan(value) or math.isinf(value):
                    bad = True
                    continue
                peak = max(peak, abs(value))
    return peak, bad


def summarise_run(run_zip: Path, cfg: dict, cell: dict, trial_idx: int,
                  seed: int, provenance: dict) -> dict:
    """Reduce one run archive to its row."""
    env = cfg["environment"]
    tick_rate = max(int(env.get("ticks_per_second", 1)), 1)
    radius = float(env["termination"]["radius"])
    targets = _target_positions(cfg)

    with zipfile.ZipFile(run_zip) as zf:
        names = zf.namelist()
        pos_member = next(n for n in names if n.endswith("_position.csv"))
        with zf.open(pos_member) as raw:
            prows = list(csv.DictReader(io.TextIOWrapper(raw)))
        events = {}
        ev_member = next((n for n in names if n.endswith("events.json")), None)
        if ev_member is not None:
            with zf.open(ev_member) as raw:
                events = json.load(io.TextIOWrapper(raw))
        max_abs_state, saw_bad = _state_extremes(zf, names)

    row = _blank_row(cell, trial_idx, seed, cfg, provenance)
    row["n_ticks_logged"] = len(prows)
    if not prows:
        row["numerical_failure"] = True
        row["error"] = "empty position log"
        return row

    last = prows[-1]
    fx, fy = float(last["pos_x"]), float(last["pos_y"])
    row["final_x"], row["final_y"] = fx, fy
    row["final_dist"] = min(math.hypot(tx - fx, ty - fy)
                            for tx, ty in targets.values())

    tick, fine, hit = _first_crossing(prows, targets, radius)
    if hit is not None:
        row.update({
            "decided": True,
            "timeout": False,
            "choice": hit,
            "correct": hit == factors.CORRECT_TARGET_ID,
            "t_commit_ticks": int(tick),
            "t_commit_fine": float(fine),
            "t_arrival_s": float(fine) / tick_rate,
        })
    else:
        row.update({"decided": False, "timeout": True})

    bif = (events.get("bifurcation_events") or [])
    if bif:
        first = min(bif, key=lambda e: e.get("tick", 1 << 30))
        row["t_bif_ticks"] = int(first.get("tick"))
        row["bif_target"] = str(first.get("target") or "")
        if row["choice"]:
            row["bif_agrees_with_choice"] = row["bif_target"] == row["choice"]

    row["max_abs_state"] = max_abs_state
    if saw_bad or (max_abs_state == max_abs_state
                   and max_abs_state > factors.MAX_ABS_STATE):
        row["numerical_failure"] = True
        row["error"] = ("non-finite ring state" if saw_bad
                        else f"max|z| = {max_abs_state:.4g} > {factors.MAX_ABS_STATE:g}")
    return row


def _blank_row(cell, trial_idx, seed, cfg, provenance) -> dict:
    mf = cfg["environment"]["agents"]["movable_0"]["mean_field_model"]
    return {
        "cell_id": int(cell["cell_id"]),
        "v": float(cell["v"]),
        "u_hat": float(cell["u_hat"]),
        "u_star": float(cell["u_star"]),
        "u": float(cell["u"]),
        "trial_idx": int(trial_idx),
        "seed": int(seed),
        "decided": False,
        "choice": "",
        "correct": False,
        "t_commit_ticks": None,
        "t_commit_fine": None,
        "t_arrival_s": None,
        "timeout": False,
        "numerical_failure": False,
        "max_abs_state": None,
        "t_bif_ticks": None,
        "bif_target": "",
        "bif_agrees_with_choice": None,
        "n_ticks_logged": 0,
        "final_x": None,
        "final_y": None,
        "final_dist": None,
        "integration_dt": float(mf["integration_dt"]),
        "error": "",
        "git_sha": provenance["git_sha"],
        "config_hash": provenance["config_hash"],
    }


def failed_row(cell, trial_idx, seed, cfg, provenance, error: str) -> dict:
    """A trial that never produced an archive: recorded, not dropped."""
    row = _blank_row(cell, trial_idx, seed, cfg, provenance)
    row.update({"numerical_failure": True, "timeout": False,
                "decided": False, "error": error[:500]})
    return row


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
class InProcessRunner:
    """Run trial configs in this interpreter, importing the simulator once."""

    def __init__(self):
        from config import Config                    # noqa: F401
        from environment import EnvironmentFactory   # noqa: F401
        self._configured_logging = False

    def run(self, config_path: Path) -> None:
        from config import Config
        from environment import EnvironmentFactory
        from logging_utils import configure_logging

        my_config = Config(config_path=str(config_path))
        if not self._configured_logging:
            configure_logging(
                my_config.environment.get("logging"),
                config_path=config_path.resolve(),
                project_root=_ROOT,
            )
            self._configured_logging = True
        # The arena prints per-tick progress on stdout; 400 trials of that is
        # pure noise in a SLURM .out file.
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            env = EnvironmentFactory.create_environment(my_config)
            env.start()


class SubprocessRunner:
    """One fresh src/main.py per trial (isolation / verification path)."""

    def __init__(self, timeout: float = 900.0):
        self.timeout = timeout

    def run(self, config_path: Path) -> None:
        import subprocess
        res = subprocess.run(
            [sys.executable, str(_ROOT / "src" / "main.py"), "-c", str(config_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            timeout=self.timeout,
        )
        if res.returncode != 0:
            raise RuntimeError(f"main.py exited {res.returncode}: {res.stderr[-2000:]}")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _coerce(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        clean = dict(row)
        for key in _BOOL_FIELDS:
            if clean.get(key) is not None:
                clean[key] = bool(clean[key])
        out.append({k: clean.get(k) for k in FIELDS})
    return out


def write_rows(stem: Path, rows: list[dict], fmt: str) -> Path:
    """Write one task's rows to `<stem>.<ext>`. Returns the path actually written."""
    rows = _coerce(rows)
    if fmt == "parquet":
        path = stem.with_suffix(".parquet")
        try:
            import pandas as pd
            frame = pd.DataFrame(rows, columns=FIELDS)
            for key in _BOOL_FIELDS:
                frame[key] = frame[key].astype("boolean")
            frame.to_parquet(path, index=False)
            return path
        except Exception as exc:                    # noqa: BLE001
            print(f"  parquet unavailable ({exc!r}); falling back to CSV")
    path = stem.with_suffix(".csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
    return path


def read_rows(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        import pandas as pd
        return pd.read_parquet(path).to_dict("records")
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def output_path(results_root: Path, cell_id: int, chunk: int, fmt: str) -> Path:
    suffix = "parquet" if fmt == "parquet" else "csv"
    return (results_root / "raw" /
            f"cell{int(cell_id):02d}_chunk{int(chunk):03d}.{suffix}")


def _existing_output(results_root: Path, cell_id: int, chunk: int) -> Path | None:
    for fmt in ("parquet", "csv"):
        path = output_path(results_root, cell_id, chunk, fmt)
        if path.is_file():
            return path
    return None


def _publish(src: Path, dest: Path) -> None:
    """Copy atomically: temp name in the destination directory, then rename."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".tmp")
    os.close(fd)
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# Task resolution
# ---------------------------------------------------------------------------
def load_manifest(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(
            f"Manifest not found: {path}\n"
            "Run scripts/uhat_v_sweep/generate_manifest.py first."
        )
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def find_cell(manifest: dict, cell_id: int) -> dict:
    for cell in manifest["cells"]:
        if int(cell["cell_id"]) == int(cell_id):
            return cell
    raise SystemExit(f"cell_id {cell_id} not in manifest "
                     f"(0..{len(manifest['cells']) - 1})")


def chunk_trials(chunk: int, n_trials: int, per_chunk: int) -> range:
    start = int(chunk) * int(per_chunk)
    return range(min(start, n_trials), min(start + int(per_chunk), n_trials))


def n_chunks(n_trials: int, per_chunk: int) -> int:
    return (int(n_trials) + int(per_chunk) - 1) // int(per_chunk)


# ---------------------------------------------------------------------------
# The task
# ---------------------------------------------------------------------------
def run_task(cell: dict, trials: range, results_root: Path, chunk: int,
             fmt: str = "parquet", force: bool = False,
             dt_override: float | None = None, scratch: Path | None = None,
             subprocess_mode: bool = False, keep_raw: bool = False,
             template: dict | None = None, quiet: bool = False) -> tuple[int, Path | None]:
    """Run one (cell, chunk). Returns (exit_code, published_path)."""
    if len(trials) == 0:
        print(f"[cell {cell['cell_id']}:{chunk}] empty trial range; nothing to do")
        return 0, None

    existing = _existing_output(results_root, cell["cell_id"], chunk)
    if existing and not force:
        try:
            if len(read_rows(existing)) == len(trials):
                print(f"[cell {cell['cell_id']}:{chunk}] already complete "
                      f"({len(trials)} rows at {existing}); exit 0. "
                      "Use --force to re-run.")
                return 0, existing
        except Exception:                            # noqa: BLE001
            pass

    cell_cfg = config_patch.cell_config(cell, template=template,
                                        dt_override=dt_override)
    provenance = {"git_sha": config_patch.git_sha(),
                  "config_hash": config_patch.config_hash(cell_cfg)}

    print(f"[cell {cell['cell_id']}:{chunk}] v={cell['v']} u_hat={cell['u_hat']} "
          f"u={cell['u']:.6f} (u*={cell['u_star']:.6f})  trials "
          f"{trials.start}..{trials.stop - 1}  cfg={provenance['config_hash']}  "
          f"sha={provenance['git_sha']}")
    if not quiet:
        print(f"  locked: {json.dumps(config_patch.env_summary(cell_cfg), sort_keys=True)}")
    sys.stdout.flush()

    scratch_base = scratch or Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
    scratch_base.mkdir(parents=True, exist_ok=True)
    runner = SubprocessRunner() if subprocess_mode else InProcessRunner()

    rows: list[dict] = []
    t0 = time.time()
    with tempfile.TemporaryDirectory(
            prefix=f"uhatv_c{cell['cell_id']}_k{chunk}_", dir=scratch_base) as tmp:
        tmp = Path(tmp)
        for trial_idx in trials:
            seed = config_patch.seed_for(trial_idx)
            trial_dir = tmp / f"trial_{trial_idx}"
            cfg = config_patch.trial_config(cell, trial_idx, seed, str(trial_dir),
                                            cell_cfg=cell_cfg)
            cfg_path = config_patch.write_trial_config(
                cfg, tmp / f"config_trial_{trial_idx}.json")
            try:
                runner.run(cfg_path)
                run_zip = next(trial_dir.glob("config_folder_*/run_*.zip"))
                rows.append(summarise_run(run_zip, cfg, cell, trial_idx, seed,
                                          provenance))
            except Exception as exc:                 # noqa: BLE001 - data, not abort
                rows.append(failed_row(cell, trial_idx, seed, cfg, provenance,
                                       repr(exc)))
                print(f"  [cell {cell['cell_id']}:{chunk}] trial {trial_idx} "
                      f"(seed {seed}) FAILED: {exc!r}")
            if keep_raw and trial_dir.is_dir():
                shutil.copytree(trial_dir,
                                results_root / "keep_raw" /
                                f"cell{cell['cell_id']:02d}_chunk{chunk:03d}" /
                                f"trial_{trial_idx}",
                                dirs_exist_ok=True)
            if trial_dir.is_dir():
                shutil.rmtree(trial_dir, ignore_errors=True)

        staged = write_rows(tmp / "rows", rows, fmt)
        dest = output_path(results_root, cell["cell_id"], chunk,
                           staged.suffix.lstrip("."))
        _publish(staged, dest)

    elapsed = time.time() - t0
    n_fail = sum(1 for r in rows if r["numerical_failure"])
    n_dec = sum(1 for r in rows if r["decided"])
    n_cor = sum(1 for r in rows if r["correct"])
    print(f"[cell {cell['cell_id']}:{chunk}] done: {len(rows)} trials in "
          f"{elapsed:.1f}s ({elapsed / max(len(rows), 1):.2f} s/trial) — "
          f"decided {n_dec}/{len(rows)}, correct {n_cor}, "
          f"numerical failures {n_fail} -> {dest}")
    return 0, dest


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cell-id", type=int, help="cell index into manifest.json")
    ap.add_argument("--chunk", type=int, default=0, help="chunk index within the cell")
    ap.add_argument("--trials-per-chunk", type=int, default=None,
                    help="trials per array task (default: the whole cell)")
    ap.add_argument("--index", type=int,
                    help="flat SLURM array index; equals cell_id * n_chunks + chunk")
    ap.add_argument("--results-root", type=Path,
                    default=_ROOT / "results" / "uhat_v_sweep")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="default: <results-root>/manifest.json")
    ap.add_argument("--smoke", action="store_true",
                    help=f"Section 12: cells {factors.SMOKE_CELLS} at "
                         f"{factors.SMOKE_TRIALS} trials each, locally")
    ap.add_argument("--force", action="store_true",
                    help="re-run even if the output file is complete")
    ap.add_argument("--dt-override", type=float, default=None,
                    help="integration_dt for the Section 11 step-halving check ONLY")
    ap.add_argument("--format", choices=("parquet", "csv"), default="parquet")
    ap.add_argument("--scratch", type=Path, default=None,
                    help="staging directory (default: $TMPDIR)")
    ap.add_argument("--subprocess", action="store_true",
                    help="one src/main.py process per trial (debug/verification)")
    ap.add_argument("--keep-raw", action="store_true",
                    help="also copy the run archives to <results-root>/keep_raw")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    results_root = args.results_root.resolve()
    manifest = load_manifest(args.manifest or (results_root / "manifest.json"))

    if args.smoke:
        results_root = (results_root if args.results_root.name.endswith("smoke")
                        else results_root / "smoke")
        template = config_patch.load_template()
        status = 0
        for v, u_hat in factors.SMOKE_CELLS:
            cell = find_cell(manifest, factors.cell_id(v, u_hat))
            code, _ = run_task(cell, range(factors.SMOKE_TRIALS), results_root, 0,
                               fmt=args.format, force=True,
                               dt_override=args.dt_override, scratch=args.scratch,
                               subprocess_mode=args.subprocess,
                               keep_raw=args.keep_raw, template=template)
            status |= code
        print(f"\nSmoke output in {results_root / 'raw'}. Statistics are "
              f"meaningless at n={factors.SMOKE_TRIALS}: this validates the "
              "pipeline, nothing else.")
        return status

    per_chunk = args.trials_per_chunk or int(manifest["n_trials"])
    if args.index is not None:
        chunks = n_chunks(manifest["n_trials"], per_chunk)
        cell_id, chunk = divmod(int(args.index), chunks)
    elif args.cell_id is not None:
        cell_id, chunk = int(args.cell_id), int(args.chunk)
    else:
        raise SystemExit("pass --cell-id (with --chunk), --index, or --smoke")

    cell = find_cell(manifest, cell_id)
    if cell.get("excluded"):
        print(f"[cell {cell_id}:{chunk}] marked excluded in the manifest "
              f"({cell.get('excluded_reason')}); nothing to do")
        return 0
    trials = chunk_trials(chunk, int(cell["n_trials"]), per_chunk)
    code, _ = run_task(cell, trials, results_root, chunk, fmt=args.format,
                       force=args.force, dt_override=args.dt_override,
                       scratch=args.scratch, subprocess_mode=args.subprocess,
                       keep_raw=args.keep_raw)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
