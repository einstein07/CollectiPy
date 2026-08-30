#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2026 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""§6: one array task = one manifest row × one run_id range, either campaign.

    python3 scripts/ra_ddm_frontier/run_batch.py \
        --campaign ra|ddm --manifest <csv> --row <k> \
        --first-run <a> --last-run <b> --base-root <dir> \
        [--table-cache-dir <dir>] [--failures-dir <dir>] [--task-tag <s>] \
        [--configs-only] [--subprocess] [--force]

Per replicate (the §7 traceability contract, all of it):

    <base-root>/cells/<sweep>/v_<v>/u_<u>/replicate_<run_id>/   (RA)
    <base-root>/points/ce_<c_e>/replicate_<run_id>/             (DDM)
        config.json      the exact effective config, written BEFORE the run
        run_meta.json    identity + both seeds + scheme + git sha
        config_folder_0/run_1.zip   the simulator's native logs
        .done            written only on verified success

A replicate with `.done` is skipped (idempotent: resubmitting a partial array
is a plain re-run). One process imports the simulator once and runs the whole
batch (RECON D-03); `--subprocess` keeps the one-`main.py`-per-replicate
verification path. Failures append to `<failures-dir>/task_<tag>.log` — one
file per task, never a shared append — and the task exits non-zero.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in (str(_HERE), str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import frontier   # noqa: E402


class InProcessRunner:
    """Run configs in this interpreter, importing the simulator once."""

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
            configure_logging(my_config.environment.get("logging"),
                              config_path=config_path.resolve(),
                              project_root=_ROOT)
            self._configured_logging = True
        with open(os.devnull, "w") as devnull, \
                contextlib.redirect_stdout(devnull):
            env = EnvironmentFactory.create_environment(my_config)
            env.start()


class SubprocessRunner:
    """One fresh src/main.py per replicate (verification path)."""

    def __init__(self, timeout: float = 1800.0):
        self.timeout = timeout

    def run(self, config_path: Path) -> None:
        import subprocess
        res = subprocess.run(
            [sys.executable, str(_ROOT / "src" / "main.py"),
             "-c", str(config_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            timeout=self.timeout)
        if res.returncode != 0:
            raise RuntimeError(
                f"main.py exited {res.returncode}: {res.stderr[-2000:]}")


def build_replicate(campaign: str, row: dict, run_id: int, rep_dir: Path,
                    cell_cfg: dict, provenance: dict) -> Path:
    """Write config.json + run_meta.json for one replicate; returns config path."""
    import copy
    cfg = copy.deepcopy(cell_cfg)
    seeds = frontier.apply_seeds(cfg, campaign, run_id)
    cfg["environment"].setdefault("results", {})["base_path"] = str(rep_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = rep_dir / "config.json"
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    meta = {"campaign": campaign, "run_id": int(run_id),
            "dth_deg": frontier.DTH_DEG, "diff_bp": frontier.DIFF_BP,
            **{k: row[k] for k in row}, **seeds, **provenance}
    with open(rep_dir / "run_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return cfg_path


def cell_config(campaign: str, row: dict, table_cache_dir: str | None) -> dict:
    if campaign == "ra":
        template = frontier._load_json(frontier.RA_TEMPLATE)
        return frontier.patch_ra(template, float(row["u"]), float(row["v"]))
    template = frontier._load_json(frontier.DDM_TEMPLATE)
    return frontier.patch_ddm(template, float(row["c_e"]), table_cache_dir)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--campaign", choices=("ra", "ddm"), required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--row", type=int, required=True,
                    help="0-based data-row index into the manifest")
    ap.add_argument("--first-run", type=int, required=True)
    ap.add_argument("--last-run", type=int, required=True)
    ap.add_argument("--base-root", type=Path, required=True)
    ap.add_argument("--table-cache-dir", type=Path, default=None,
                    help="DDM only: bellman table cache directory")
    ap.add_argument("--failures-dir", type=Path, default=None)
    ap.add_argument("--task-tag", default=None,
                    help="failure-file tag (default: <row>_<first-run>)")
    ap.add_argument("--configs-only", action="store_true",
                    help="write configs + metadata, run nothing")
    ap.add_argument("--subprocess", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-run replicates even when .done exists")
    args = ap.parse_args(argv)

    rows = frontier.read_manifest(args.manifest)
    if not (0 <= args.row < len(rows)):
        raise SystemExit(f"--row {args.row} outside manifest (0..{len(rows)-1})")
    row = rows[args.row]
    ident = row.get("cell_id") or row.get("point_id")
    if row.get("seed_scheme") and row["seed_scheme"] != frontier.seeding.SCHEME:
        raise SystemExit(f"manifest scheme {row['seed_scheme']!r} != code scheme "
                         f"{frontier.seeding.SCHEME!r}: refusing to mix universes")

    cache = (str(args.table_cache_dir) if args.table_cache_dir else
             (str(Path(args.base_root) / "table_cache")
              if args.campaign == "ddm" else None))
    if cache:
        Path(cache).mkdir(parents=True, exist_ok=True)
    cfg0 = cell_config(args.campaign, row, cache)
    provenance = {"git_sha": frontier.git_sha(),
                  "config_hash": frontier.config_hash(cfg0)}

    print(f"[{args.campaign}:{ident}] runs {args.first_run}..{args.last_run} "
          f"cfg={provenance['config_hash']} sha={provenance['git_sha']}")
    sys.stdout.flush()

    runner = None
    if not args.configs_only:
        runner = SubprocessRunner() if args.subprocess else InProcessRunner()

    failures_dir = args.failures_dir or (Path(args.base_root) / "failures")
    tag = args.task_tag or f"{args.campaign}_{args.row}_{args.first_run}"
    fail_log = failures_dir / f"task_{tag}.log"

    n_run = n_skip = n_fail = 0
    t0 = time.time()
    for run_id in range(args.first_run, args.last_run + 1):
        rep_dir = frontier.replicate_dir(args.base_root, row, run_id)
        done = rep_dir / ".done"
        if done.exists() and not args.force:
            n_skip += 1
            continue
        cfg_path = build_replicate(args.campaign, row, run_id, rep_dir,
                                   cfg0, provenance)
        if args.configs_only:
            print(cfg_path)
            continue
        try:
            runner.run(cfg_path)
            archives = list(rep_dir.glob("config_folder_*/run_*.zip"))
            if not archives:
                raise RuntimeError("run completed but produced no run archive")
            done.touch()
            n_run += 1
        except Exception as exc:                     # noqa: BLE001 — data, not abort
            n_fail += 1
            failures_dir.mkdir(parents=True, exist_ok=True)
            with open(fail_log, "a", encoding="utf-8") as fh:
                fh.write(f"{cfg_path}\t{exc!r}\n")
            print(f"  [{ident}] run {run_id} FAILED: {exc!r}")
            sys.stdout.flush()

    if not args.configs_only:
        total = max(n_run, 1)
        print(f"[{args.campaign}:{ident}] done: ran {n_run}, skipped {n_skip}, "
              f"failed {n_fail} in {time.time() - t0:.1f}s "
              f"({(time.time() - t0) / total:.2f} s/run)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
