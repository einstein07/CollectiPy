# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Execute one array task: one condition-point x one chunk of replicates.

CAMPAIGN_SPEC.md Sections 7 and 8. This is the ONLY program the SLURM array runs.

    python3 -m campaign.run_chunk --index <task_id> --results-root <dir> [...]
    python3 -m campaign.run_chunk --only q01_a60_ce20:2 --results-root <dir> [...]

Behaviour, in order:

- **Idempotent** (Section 7.5): if this chunk's CSV already exists and is complete,
  exit 0 immediately. `--force` overrides.
- **Scratch staging** (Section 7.4): every replicate's raw archive is written to
  node-local scratch (`$TMPDIR`, or --scratch); exactly ONE file per chunk — the
  summary CSV — is moved to shared storage, atomically (temp name in the destination
  directory, then rename). Raw archives are deleted with the scratch dir unless
  `--keep-raw`, which copies them across in the standard sweep layout for the
  Section 9.3 diagnostic subset.
- **In-process replicates**: interpreter startup (~1.2 s of imports) would otherwise
  cost more than a cache-hit replicate, so the simulator is imported once and each
  replicate builds a fresh Config + Environment. `--subprocess` falls back to one
  `main.py` process per replicate; the two paths are bit-identical (pre-flight 9.1
  runs both) because every source of randomness is seeded from the per-replicate
  config, never from process state.
- A replicate that RAISES is recorded and the chunk CSV is left under a `.partial`
  name: the completeness check then fails, the task exits non-zero, and a resubmit
  retries the whole chunk. Nothing partial ever occupies the final path.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for p in (str(_ROOT), str(_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from campaign import factors, genconfig, matrix, summarise  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--index", type=int, help="array task index into the task table")
    sel.add_argument("--only", help="condition[:chunk] selector, e.g. q01_a60_ce20:2")
    ap.add_argument("--results-root", required=True, type=Path)
    ap.add_argument("--reps", type=int, default=factors.REPS)
    ap.add_argument("--chunk", type=int, default=factors.CHUNK)
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="Bellman table cache (default: <results-root>/table_cache)")
    ap.add_argument("--scratch", type=Path, default=None,
                    help="staging dir (default: $TMPDIR)")
    ap.add_argument("--force", action="store_true",
                    help="re-run even if the chunk CSV is complete")
    ap.add_argument("--keep-raw", action="store_true",
                    help="copy the raw run archives to shared storage too")
    ap.add_argument("--subprocess", action="store_true",
                    help="one main.py process per replicate (debug/verification)")
    ap.add_argument("--horizon-check", type=float, default=None,
                    help="enable the solver's T_max_check_factor (precompute only)")
    return ap.parse_args(argv)


def resolve_task(args):
    """Return (condition, chunk_idx) from --index or --only."""
    if args.index is not None:
        table = matrix.task_table(args.reps, args.chunk)
        if not (0 <= args.index < len(table)):
            raise SystemExit(
                f"--index {args.index} out of range [0, {len(table)}) for "
                f"reps={args.reps} chunk={args.chunk}"
            )
        return table[args.index]
    name, _, k = args.only.partition(":")
    return matrix.find_condition(name), int(k or 0)


def chunk_is_complete(csv_path: Path, expected_rows: int) -> bool:
    if not csv_path.is_file():
        return False
    try:
        return len(summarise.read_chunk_csv(csv_path)) == expected_rows
    except Exception:
        return False


def _atomic_publish(src: Path, dest: Path) -> None:
    """Copy `src` to `dest` atomically: temp name in dest's directory, then rename."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".tmp")
    os.close(fd)
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _write_json_atomic(obj: dict, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class InProcessRunner:
    """Run replicate configs inside this interpreter, importing the simulator once."""

    def __init__(self):
        from config import Config                    # noqa: F401  (import check)
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
        # The arena prints per-tick progress to stdout; 100 replicates of that is
        # pure noise in a SLURM .out file.
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            env = EnvironmentFactory.create_environment(my_config)
            env.start()


class SubprocessRunner:
    """One fresh main.py process per replicate (verification / isolation path)."""

    def run(self, config_path: Path) -> None:
        import subprocess
        res = subprocess.run(
            [sys.executable, str(_ROOT / "src" / "main.py"), "-c", str(config_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            timeout=20 * 60,
        )
        if res.returncode != 0:
            raise RuntimeError(
                f"main.py exited {res.returncode}: {res.stderr[-2000:]}"
            )


def main(argv=None) -> int:
    args = parse_args(argv)
    cond, chunk_idx = resolve_task(args)
    rep_range = matrix.chunk_range(chunk_idx, args.reps, args.chunk)
    if len(rep_range) == 0:
        print(f"[{cond.name}:{chunk_idx}] empty replicate range; nothing to do")
        return 0

    results_root = args.results_root.resolve()
    cond_dir = results_root / cond.rel_dir
    csv_dest = cond_dir / "chunks" / f"chunk_{chunk_idx:04d}.csv"
    cache_dir = (args.cache_dir or (results_root / "table_cache")).resolve()

    print(genconfig.startup_line(cond, rep_range))
    sys.stdout.flush()

    if not args.force and chunk_is_complete(csv_dest, len(rep_range)):
        print(f"[{cond.name}:{chunk_idx}] already complete "
              f"({len(rep_range)} rows at {csv_dest}); exit 0. Use --force to re-run.")
        return 0

    # Condition-level config + manifest (Section 8). Identical content from every
    # task of the condition, so concurrent atomic writes are benign.
    cond_cfg = genconfig.condition_config(cond, table_cache_dir=str(cache_dir))
    _write_json_atomic(cond_cfg, cond_dir / "config.json")
    _write_json_atomic(
        genconfig.manifest(cond, args.reps, args.chunk, cond_cfg),
        cond_dir / "manifest.json",
    )

    scratch_base = args.scratch or Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
    runner = SubprocessRunner() if args.subprocess else InProcessRunner()

    rows, failures = [], []
    t0 = time.time()
    with tempfile.TemporaryDirectory(
        prefix=f"campaign_{cond.name}_{chunk_idx}_", dir=scratch_base
    ) as scratch:
        scratch = Path(scratch)
        for rep in rep_range:
            rep_dir = scratch / f"replicate_{rep}"
            cfg = genconfig.replicate_config(
                cond, rep, str(rep_dir), cond_cfg=cond_cfg,
                horizon_check_factor=args.horizon_check,
            )
            cfg_path = scratch / f"config_rep_{rep}.json"
            with open(cfg_path, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh)
            try:
                runner.run(cfg_path)
                run_zip = next(rep_dir.glob("config_folder_*/run_*.zip"))
                rows.append(summarise.summarise_run(
                    run_zip, cfg,
                    {"condition": cond.name, "arm": cond.arm,
                     "discretisation_limited": cond.discretisation_limited},
                    rep,
                ))
            except Exception as exc:  # noqa: BLE001 — one bad replicate, not the task
                failures.append({"replicate": rep, "error": repr(exc)})
                print(f"[{cond.name}:{chunk_idx}] replicate {rep} FAILED: {exc!r}")
            if args.keep_raw and rep_dir.is_dir():
                raw_dest = (cond_dir / "raw" / f"chunk_{chunk_idx:04d}"
                            / f"replicate_{rep}")
                shutil.copytree(rep_dir, raw_dest, dirs_exist_ok=True)

        chunk_csv = scratch / "chunk.csv"
        summarise.write_chunk_csv(chunk_csv, rows)
        if failures:
            _atomic_publish(chunk_csv, csv_dest.with_suffix(".csv.partial"))
            _write_json_atomic(
                {"failures": failures},
                csv_dest.with_suffix(".csv.errors.json"),
            )
            print(f"[{cond.name}:{chunk_idx}] {len(failures)} of {len(rep_range)} "
                  f"replicates failed; wrote PARTIAL csv, exiting non-zero")
            return 1
        _atomic_publish(chunk_csv, csv_dest)

    dt = time.time() - t0
    print(f"[{cond.name}:{chunk_idx}] done: {len(rows)} replicates in {dt:.1f}s "
          f"({dt / max(len(rows), 1):.2f} s/replicate) -> {csv_dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
