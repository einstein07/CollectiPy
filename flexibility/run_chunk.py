# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Execute one array task: one condition-point x one chunk of replicates.

    python3 -m flexibility.run_chunk --index <task_id> --results-root <dir> [...]
    python3 -m flexibility.run_chunk --only ra_u8__d1.9296pct:3 --results-root <dir>

Raw output is left in the sweep layout of Section 9.8,

    <results-root>/{arm}/diff_{pct}/replicate_{n}/

i.e. the arm level replaces the historical `u_{value}` level, and the existing
analysis tooling reads it unchanged. No summariser runs here: the flexibility
measures of Section 7 (commitment before arrival, reversal latency from the swap
event, number of identity sign changes, bump geometry) are extracted downstream from
the snapshot tables and events.json, where the raw traces are still available.

Replicates run IN-PROCESS by default: interpreter startup costs more than a short
replicate, and every source of randomness is seeded from the per-replicate config
rather than from process state, so the in-process and subprocess paths agree.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from flexibility import factors, genconfig, matrix  # noqa: E402


class InProcessRunner:
    """Run replicate configs inside this interpreter, importing the simulator once."""

    def __init__(self):
        """Initialize the instance."""
        from config import Config                    # noqa: F401  (import check)
        from environment import EnvironmentFactory   # noqa: F401
        self._configured_logging = False

    def run(self, config_path: Path) -> None:
        """Run one replicate config to completion."""
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
        # The arena prints per-tick progress to stdout; a chunk of that is pure noise
        # in a SLURM .out file.
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            env = EnvironmentFactory.create_environment(my_config)
            env.start()


class SubprocessRunner:
    """One `main.py` process per replicate — the historical path, for cross-checking."""

    def run(self, config_path: Path) -> None:
        """Run one replicate config in a fresh interpreter."""
        import subprocess
        subprocess.run(
            [sys.executable, str(_ROOT / "src" / "main.py"), "-c", str(config_path)],
            check=True, stdout=subprocess.DEVNULL,
        )


def _resolve_task(args) -> tuple[matrix.Condition, int, range]:
    """Return (condition, chunk_index, replicate range) for this task."""
    table = matrix.task_table()
    if args.index is not None:
        if not 0 <= args.index < len(table):
            raise SystemExit(
                f"--index {args.index} is outside the array 0..{len(table) - 1}"
            )
        _, cond, first = table[args.index]
        chunk_idx = (first - 1) // factors.CHUNK
    else:
        name, _, chunk_str = args.only.partition(":")
        cond = matrix.find_condition(name)
        chunk_idx = int(chunk_str) if chunk_str else 0
        first = chunk_idx * factors.CHUNK + 1
    last = min(first + factors.CHUNK - 1, cond.reps)
    return cond, chunk_idx, range(first, last + 1)


def _is_complete(rep_dir: Path) -> bool:
    """True when a replicate directory already holds a finished run archive."""
    return any(rep_dir.glob("config_folder_*/run_*.zip"))


def main(argv=None) -> int:
    """Run one array task."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--index", type=int,
                     help="array task index into the task table")
    sel.add_argument("--only", help="condition[:chunk], e.g. ra_u8__d1.9296pct:3")
    ap.add_argument("--results-root", required=True, type=Path)
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="Bellman table cache (default: <results-root>/table_cache)")
    ap.add_argument("--force", action="store_true",
                    help="re-run replicates whose output already exists")
    ap.add_argument("--subprocess", action="store_true",
                    help="one main.py process per replicate")
    ap.add_argument("--dry-run", action="store_true",
                    help="write the configs and stop, without simulating")
    args = ap.parse_args(argv)

    cond, chunk_idx, reps = _resolve_task(args)
    cache_dir = args.cache_dir or (args.results_root / "table_cache")

    print(f"[{cond.name}:{chunk_idx}] arm={cond.arm} delta={cond.delta * 100:.4f}% "
          f"replicates={reps.start}-{reps.stop - 1}")

    runner = None if args.dry_run else (
        SubprocessRunner() if args.subprocess else InProcessRunner()
    )

    failures, done, skipped = [], 0, 0
    t0 = time.time()
    for rep in reps:
        rep_dir = Path(genconfig.output_dir(str(args.results_root), cond, rep))
        if not args.force and _is_complete(rep_dir):
            skipped += 1
            continue
        rep_dir.mkdir(parents=True, exist_ok=True)
        cfg = genconfig.replicate_config(
            cond, rep, str(rep_dir), table_cache_dir=str(cache_dir)
        )
        cfg_path = rep_dir / "config.json"
        with open(cfg_path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
        if args.dry_run:
            done += 1
            continue
        try:
            runner.run(cfg_path)
            done += 1
        except Exception as exc:  # noqa: BLE001 — one bad replicate, not the task
            failures.append({"replicate": rep, "error": repr(exc)})
            print(f"[{cond.name}:{chunk_idx}] replicate {rep} FAILED: {exc!r}")

    dt = time.time() - t0
    print(f"[{cond.name}:{chunk_idx}] {done} run, {skipped} already present, "
          f"{len(failures)} failed, in {dt:.1f}s")

    if failures:
        err_path = args.results_root / cond.arm / (
            f"diff_{cond.delta * 100:.4f}pct"
        ) / f"errors_chunk_{chunk_idx:04d}.json"
        err_path.parent.mkdir(parents=True, exist_ok=True)
        with open(err_path, "w", encoding="utf-8") as fh:
            json.dump({"failures": failures}, fh, indent=2)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
