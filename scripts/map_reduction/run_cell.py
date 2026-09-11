# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Run one (cell, chunk) of the 6.3 design, or one arm of the 6.6 sanity check.

Same conventions as scripts/uhat_v_sweep/run_cell.py, the array scaffolding this
experiment reuses:

- **One output file per task.** `raw/cell<K>_chunk<J>.parquet` (CSV fallback);
  nothing ever appends to a shared file.
- **Idempotent.** A complete output file makes the task exit 0 immediately;
  `force=True` overrides.
- **Scratch staging.** Every trial's run archive is written to node-local scratch
  and deleted with it; exactly one file per task is published atomically.
- **In-process trials.** The simulator is imported once per process; each trial
  builds a fresh Config + Environment from its own config, so every source of
  randomness is seeded from that config, never from process state.
- **A failed trial is DATA, not an abort** (`numerical_failure = True`).
"""

from __future__ import annotations

import contextlib
import csv
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in (str(_HERE.parent), str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from map_reduction import config_patch, factors, metrics  # noqa: E402


#: Row schema of the design (append-only; downstream code indexes by name).
FIELDS = [
    "cell_id", "reduction", "delta_close_deg", "q_c_rel", "trial_idx", "seed",
    "arrived", "reached", "t_arrival_ticks", "t_arrival_fine", "t_arrival_s", "timeout",
    "committed", "t_commit_ticks", "commit_class", "com_commit_deg", "midpoint_commit_deg",
    "n_peaks_commit", "bearing_A_commit_deg", "bearing_B_commit_deg", "bearing_C_commit_deg",
    "n_peaks_t1", "n_peaks_t2", "state_t2", "t_first_unimodal",
    "class_final", "com_final_deg", "t_final", "n_peaks_final",
    "t_bif_ticks", "bif_target",
    "n_ticks_logged", "final_x", "final_y", "final_dist", "max_abs_state",
    "numerical_failure", "error", "git_sha", "config_hash",
]

#: Row schema of the 6.6 sanity check.
SANITY_FIELDS = [
    "variant", "sigma", "reduction", "trial_idx", "seed",
    "arrived", "reached", "correct", "t_arrival_ticks", "t_arrival_fine", "t_arrival_s",
    "timeout", "t_bif_ticks", "bif_target", "n_ticks_logged", "final_dist",
    "max_abs_state", "numerical_failure", "error", "git_sha", "config_hash",
]

_BOOL_FIELDS = {"arrived", "timeout", "committed", "numerical_failure", "correct"}

SANITY_ID_TO_LABEL = {"static_0.s#0": "A", "static_1.s#0": "B"}


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
class InProcessRunner:
    """Run trial configs in this interpreter, importing the simulator once."""

    def __init__(self, log_root: Path | None = None):
        from config import Config                    # noqa: F401
        from environment import EnvironmentFactory   # noqa: F401
        self._configured_logging = False
        # Log artifacts go under <log_root>/logs; pointing that at scratch keeps
        # thousands of per-trial log files out of the repository's logs/ tree.
        self._log_root = Path(log_root) if log_root else _ROOT

    def run(self, config_path: Path) -> None:
        from config import Config
        from environment import EnvironmentFactory
        from logging_utils import configure_logging

        my_config = Config(config_path=str(config_path))
        if not self._configured_logging:
            configure_logging(my_config.environment.get("logging"),
                              config_path=config_path.resolve(),
                              project_root=self._log_root)
            self._configured_logging = True
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


def make_runner(subprocess_mode: bool, log_root: Path | None):
    return SubprocessRunner() if subprocess_mode else InProcessRunner(log_root)


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------
def _blank_row(cell: dict, trial_idx: int, seed: int, provenance: dict) -> dict:
    row = {k: None for k in FIELDS}
    row.update({
        "cell_id": int(cell["cell_id"]), "reduction": str(cell["reduction"]),
        "delta_close_deg": float(cell["delta_close_deg"]), "q_c_rel": float(cell["q_c_rel"]),
        "trial_idx": int(trial_idx), "seed": int(seed),
        "arrived": False, "reached": "", "timeout": False, "committed": False,
        "commit_class": "none", "class_final": "", "bif_target": "",
        "n_ticks_logged": 0, "numerical_failure": False, "error": "",
        "git_sha": provenance["git_sha"], "config_hash": provenance["config_hash"],
    })
    return row


def summarise_run(run_zip: Path, cfg: dict, cell: dict, trial_idx: int, seed: int,
                  provenance: dict) -> dict:
    row = _blank_row(cell, trial_idx, seed, provenance)
    tick_rate = max(int(cfg["environment"].get("ticks_per_second", 1)), 1)
    m = metrics.summarise_archive(run_zip, cfg, factors.ID_TO_LABEL,
                                  tol_deg=factors.ALIGNMENT_TOL_DEG, pair=("A", "B"))
    for key, value in m.items():
        if key in row:
            row[key] = value
    if m["t_arrival_fine"] is not None:
        row["t_arrival_s"] = float(m["t_arrival_fine"]) / tick_rate
    if row["n_ticks_logged"] == 0:
        row.update({"numerical_failure": True, "error": "empty position log"})
    elif m["saw_nonfinite"]:
        row.update({"numerical_failure": True, "error": "non-finite ring state"})
    return row


def failed_row(cell: dict, trial_idx: int, seed: int, provenance: dict, error: str) -> dict:
    row = _blank_row(cell, trial_idx, seed, provenance)
    row.update({"numerical_failure": True, "error": error[:500]})
    return row


def _blank_sanity_row(variant, sigma, reduction, trial_idx, seed, provenance) -> dict:
    row = {k: None for k in SANITY_FIELDS}
    row.update({
        "variant": str(variant), "sigma": float(sigma), "reduction": str(reduction),
        "trial_idx": int(trial_idx), "seed": int(seed),
        "arrived": False, "reached": "", "correct": False, "timeout": False,
        "bif_target": "", "n_ticks_logged": 0, "numerical_failure": False, "error": "",
        "git_sha": provenance["git_sha"], "config_hash": provenance["config_hash"],
    })
    return row


def summarise_sanity_run(run_zip, cfg, variant, sigma, reduction, trial_idx, seed,
                         provenance) -> dict:
    row = _blank_sanity_row(variant, sigma, reduction, trial_idx, seed, provenance)
    tick_rate = max(int(cfg["environment"].get("ticks_per_second", 1)), 1)
    m = metrics.summarise_archive(run_zip, cfg, SANITY_ID_TO_LABEL,
                                  tol_deg=factors.ALIGNMENT_TOL_DEG, pair=("A", "B"))
    for key, value in m.items():
        if key in row:
            row[key] = value
    row["correct"] = bool(m["arrived"] and m["reached"] == "A")
    if m["t_arrival_fine"] is not None:
        row["t_arrival_s"] = float(m["t_arrival_fine"]) / tick_rate
    if row["n_ticks_logged"] == 0:
        row.update({"numerical_failure": True, "error": "empty position log"})
    elif m["saw_nonfinite"]:
        row.update({"numerical_failure": True, "error": "non-finite ring state"})
    return row


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _coerce(rows: list[dict], fields: list[str]) -> list[dict]:
    out = []
    for row in rows:
        clean = dict(row)
        for key in _BOOL_FIELDS:
            if key in clean and clean.get(key) is not None:
                clean[key] = bool(clean[key])
        out.append({k: clean.get(k) for k in fields})
    return out


def write_rows(stem: Path, rows: list[dict], fmt: str, fields: list[str] = FIELDS) -> Path:
    """Write one task's rows to `<stem>.<ext>`; returns the path written."""
    rows = _coerce(rows, fields)
    if fmt == "parquet":
        path = stem.with_suffix(".parquet")
        try:
            import pandas as pd
            frame = pd.DataFrame(rows, columns=fields)
            for key in _BOOL_FIELDS & set(fields):
                frame[key] = frame[key].astype("boolean")
            frame.to_parquet(path, index=False)
            return path
        except Exception as exc:                    # noqa: BLE001
            print(f"  parquet unavailable ({exc!r}); falling back to CSV")
    path = stem.with_suffix(".csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
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
    return results_root / "raw" / f"cell{int(cell_id):02d}_chunk{int(chunk):03d}.{suffix}"


def sanity_output_path(results_root: Path, variant: str, reduction: str, chunk: int,
                       fmt: str) -> Path:
    suffix = "parquet" if fmt == "parquet" else "csv"
    return (results_root / "sanity" / "raw" /
            f"{variant}_{reduction}_chunk{int(chunk):03d}.{suffix}")


def _existing(stem_candidates) -> Path | None:
    for path in stem_candidates:
        if path.is_file():
            return path
    return None


def _drop_siblings(dest: Path, candidates) -> None:
    """A task owns ONE output file: remove the same task's file in the other format
    (left behind by an earlier run with a different --format) so nothing is
    counted twice by the aggregator."""
    for path in candidates:
        if path != dest and path.exists():
            path.unlink()


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
# Task geometry
# ---------------------------------------------------------------------------
def chunk_trials(chunk: int, n_trials: int, per_chunk: int) -> range:
    start = int(chunk) * int(per_chunk)
    return range(min(start, n_trials), min(start + int(per_chunk), n_trials))


def n_chunks(n_trials: int, per_chunk: int) -> int:
    return (int(n_trials) + int(per_chunk) - 1) // int(per_chunk)


def find_cell(manifest: dict, cell_id: int) -> dict:
    for cell in manifest["cells"]:
        if int(cell["cell_id"]) == int(cell_id):
            return cell
    raise SystemExit(f"cell_id {cell_id} not in manifest (0..{len(manifest['cells']) - 1})")


# ---------------------------------------------------------------------------
# The tasks
# ---------------------------------------------------------------------------
def run_task(cell: dict, trials: range, results_root: Path, chunk: int,
             fmt: str = "parquet", force: bool = False, scratch: Path | None = None,
             subprocess_mode: bool = False, keep_raw: bool = False,
             template: dict | None = None, quiet: bool = False) -> tuple[int, Path | None]:
    """Run one (cell, chunk) of the design. Returns (exit_code, published_path)."""
    tag = f"[cell {cell['cell_id']}:{chunk}]"
    if len(trials) == 0:
        print(f"{tag} empty trial range; nothing to do")
        return 0, None
    existing = _existing([output_path(results_root, cell["cell_id"], chunk, f)
                          for f in ("parquet", "csv")])
    if existing and not force:
        try:
            if len(read_rows(existing)) == len(trials):
                print(f"{tag} already complete ({len(trials)} rows at {existing}); "
                      "exit 0. Use --force to re-run.")
                return 0, existing
        except Exception:                            # noqa: BLE001
            pass

    cell_cfg = config_patch.cell_config(cell, template=template)
    provenance = {"git_sha": config_patch.git_sha(),
                  "config_hash": config_patch.config_hash(cell_cfg)}
    print(f"{tag} {cell['reduction']} delta={cell['delta_close_deg']} "
          f"q_C={cell['q_c_rel']:.2f} trials {trials.start}..{trials.stop - 1} "
          f"cfg={provenance['config_hash']} sha={provenance['git_sha']}")
    if not quiet:
        import json
        print(f"  locked: {json.dumps(config_patch.env_summary(cell_cfg), sort_keys=True)}")
    sys.stdout.flush()

    scratch_base = scratch or Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
    scratch_base.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t0 = time.time()
    with tempfile.TemporaryDirectory(
            prefix=f"mapred_c{cell['cell_id']}_k{chunk}_", dir=scratch_base) as tmp:
        tmp = Path(tmp)
        runner = make_runner(subprocess_mode, tmp)
        for trial_idx in trials:
            seed = factors.seed_for(trial_idx)
            trial_dir = tmp / f"trial_{trial_idx}"
            cfg = config_patch.trial_config(cell, trial_idx, seed, str(trial_dir),
                                            cell_cfg=cell_cfg)
            cfg_path = config_patch.write_config(cfg, tmp / f"config_trial_{trial_idx}.json")
            try:
                runner.run(cfg_path)
                run_zip = next(trial_dir.glob("config_folder_*/run_*.zip"))
                rows.append(summarise_run(run_zip, cfg, cell, trial_idx, seed, provenance))
            except Exception as exc:                 # noqa: BLE001 - data, not abort
                rows.append(failed_row(cell, trial_idx, seed, provenance, repr(exc)))
                print(f"  {tag} trial {trial_idx} (seed {seed}) FAILED: {exc!r}")
            if keep_raw and trial_dir.is_dir():
                shutil.copytree(trial_dir, results_root / "keep_raw" /
                                f"cell{cell['cell_id']:02d}_chunk{chunk:03d}" /
                                f"trial_{trial_idx}", dirs_exist_ok=True)
            if trial_dir.is_dir():
                shutil.rmtree(trial_dir, ignore_errors=True)
        staged = write_rows(tmp / "rows", rows, fmt, FIELDS)
        dest = output_path(results_root, cell["cell_id"], chunk, staged.suffix.lstrip("."))
        _publish(staged, dest)
        _drop_siblings(dest, [output_path(results_root, cell["cell_id"], chunk, f)
                              for f in ("parquet", "csv")])

    elapsed = time.time() - t0
    n_fail = sum(1 for r in rows if r["numerical_failure"])
    n_arr = sum(1 for r in rows if r["arrived"])
    n_mid = sum(1 for r in rows if r["commit_class"] == "mean_of_pair")
    n_c = sum(1 for r in rows if r["reached"] == "C")
    print(f"{tag} done: {len(rows)} trials in {elapsed:.1f}s "
          f"({elapsed / max(len(rows), 1):.2f} s/trial) — arrived {n_arr}, at C {n_c}, "
          f"mean_of_pair {n_mid}, failures {n_fail} -> {dest}")
    return 0, dest


def run_sanity_task(variant: str, reduction: str, trials: range, results_root: Path,
                    chunk: int = 0, fmt: str = "parquet", force: bool = False,
                    scratch: Path | None = None, subprocess_mode: bool = False,
                    template: dict | None = None) -> tuple[int, Path | None]:
    """One arm of the 6.6 check: `variant` x `reduction`, the paired seed list."""
    tag = f"[sanity {variant}/{reduction}:{chunk}]"
    if len(trials) == 0:
        print(f"{tag} empty trial range; nothing to do")
        return 0, None
    existing = _existing([sanity_output_path(results_root, variant, reduction, chunk, f)
                          for f in ("parquet", "csv")])
    if existing and not force:
        try:
            if len(read_rows(existing)) == len(trials):
                print(f"{tag} already complete ({len(trials)} rows at {existing}); exit 0.")
                return 0, existing
        except Exception:                            # noqa: BLE001
            pass

    sigma = factors.SANITY_VARIANTS[variant]
    cfg0 = config_patch.sanity_config(reduction, template=template, sigma=sigma)
    sigma_eff = float(cfg0["environment"]["agents"]["movable_0"]["mean_field_model"]["sigma"])
    provenance = {"git_sha": config_patch.git_sha(),
                  "config_hash": config_patch.config_hash(cfg0)}
    print(f"{tag} sigma={sigma_eff:g} trials {trials.start}..{trials.stop - 1} "
          f"cfg={provenance['config_hash']} sha={provenance['git_sha']}")
    sys.stdout.flush()

    scratch_base = scratch or Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
    scratch_base.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t0 = time.time()
    with tempfile.TemporaryDirectory(
            prefix=f"mapred_sanity_{variant}_{reduction}_", dir=scratch_base) as tmp:
        tmp = Path(tmp)
        runner = make_runner(subprocess_mode, tmp)
        for trial_idx in trials:
            seed = factors.seed_for(trial_idx)
            trial_dir = tmp / f"trial_{trial_idx}"
            cfg = config_patch.sanity_trial_config(variant, reduction, trial_idx, seed,
                                                   str(trial_dir), cfg=cfg0)
            cfg_path = config_patch.write_config(cfg, tmp / f"config_trial_{trial_idx}.json")
            try:
                runner.run(cfg_path)
                run_zip = next(trial_dir.glob("config_folder_*/run_*.zip"))
                rows.append(summarise_sanity_run(run_zip, cfg, variant, sigma_eff, reduction,
                                                 trial_idx, seed, provenance))
            except Exception as exc:                 # noqa: BLE001
                row = _blank_sanity_row(variant, sigma_eff, reduction, trial_idx, seed,
                                        provenance)
                row.update({"numerical_failure": True, "error": repr(exc)[:500]})
                rows.append(row)
                print(f"  {tag} trial {trial_idx} (seed {seed}) FAILED: {exc!r}")
            if trial_dir.is_dir():
                shutil.rmtree(trial_dir, ignore_errors=True)
        staged = write_rows(tmp / "rows", rows, fmt, SANITY_FIELDS)
        dest = sanity_output_path(results_root, variant, reduction, chunk,
                                  staged.suffix.lstrip("."))
        _publish(staged, dest)
        _drop_siblings(dest, [sanity_output_path(results_root, variant, reduction, chunk, f)
                              for f in ("parquet", "csv")])

    elapsed = time.time() - t0
    n_cor = sum(1 for r in rows if r["correct"])
    n_arr = sum(1 for r in rows if r["arrived"])
    print(f"{tag} done: {len(rows)} trials in {elapsed:.1f}s — arrived {n_arr}, "
          f"correct {n_cor} -> {dest}")
    return 0, dest
