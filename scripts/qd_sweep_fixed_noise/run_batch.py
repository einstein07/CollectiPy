#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2026 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""One array task = one manifest row × one run_id range, either arm.

    python3 scripts/qd_sweep_fixed_noise/run_batch.py \
        --arm ra|ddm --manifest <csv> --row <k> \
        --first-run <a> --last-run <b> --base-root <dir> \
        [--table-cache-dir <dir>] [--failures-dir <dir>] [--task-tag <s>] \
        [--configs-only] [--subprocess] [--force]

Per replicate (the §7 traceability contract, inherited from the frontier
campaign):

    <base-root>/ra/actual_<bp>/v_<v>/u_<u>/replicate_<id>/            (Arm A)
    <base-root>/ddm/actual_<bp>/design_<bp>/<variant>_<param>/
        replicate_<id>/                                               (Arm B)
        config.json      the exact effective config, written BEFORE the run
                         (every §2 assertion has already passed on it)
        run_meta.json    identity + actual/design δ_Q + resolved A, c, k +
                         both seeds + scheme + git sha
        config_folder_0/run_1.zip   the simulator's native logs
        .done            written only on verified success

A replicate with `.done` is skipped (idempotent resubmits). One process
imports the simulator once and runs the whole batch; `--subprocess` keeps the
one-`main.py`-per-replicate verification path. Failures append to
`<failures-dir>/task_<tag>.log` and the task exits non-zero.
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

import qd   # noqa: E402


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


def cell_config(arm: str, row: dict, table_cache_dir: str | None) -> dict:
    if arm == "ra":
        return qd.patch_ra(qd.load_template("ra"),
                           float(row["u"]), float(row["v"]),
                           int(row["actual_bp"]))
    return qd.patch_ddm(qd.load_template("ddm"), row["variant"],
                        int(row["design_bp"]), int(row["actual_bp"]),
                        float(row["bound_param"]), table_cache_dir)


def build_replicate(arm: str, row: dict, run_id: int, rep_dir: Path,
                    cell_cfg: dict, provenance: dict) -> Path:
    """Write config.json + run_meta.json for one replicate."""
    import copy
    cfg = copy.deepcopy(cell_cfg)
    actual = int(row["actual_bp"])
    seeds = qd.apply_seeds(cfg, qd.model_key(row), actual, run_id)
    cfg["environment"].setdefault("results", {})["base_path"] = str(rep_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = rep_dir / "config.json"
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    # §7: run_meta additionally records actual_bp, design_bp (DDM only) and
    # the resolved A, c, k values used.
    meta = {"campaign": "qd_sweep_fixed_noise", "arm": arm,
            **{k: row[k] for k in row},
            "run_id": int(run_id), "dth_deg": qd.DTH_DEG,
            "actual_bp": actual,
            "A_actual": qd.drift_A(actual), "k_actual": qd.wald_k(actual),
            "white_rate": qd.WHITE_RATE, "c": qd.NOISE_SCALE_C,
            **seeds, **provenance}
    if arm == "ddm":
        design = int(row["design_bp"])
        meta.update({"design_bp": design, "A_design": qd.drift_A(design),
                     "k_design": qd.wald_k(design)})
    with open(rep_dir / "run_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return cfg_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", choices=("ra", "ddm"), required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--row", type=int, required=True,
                    help="0-based data-row index into the manifest")
    ap.add_argument("--first-run", type=int, required=True)
    ap.add_argument("--last-run", type=int, required=True)
    ap.add_argument("--base-root", type=Path, required=True)
    ap.add_argument("--table-cache-dir", type=Path, default=None,
                    help="ddm only: bellman table cache directory")
    ap.add_argument("--failures-dir", type=Path, default=None)
    ap.add_argument("--task-tag", default=None)
    ap.add_argument("--configs-only", action="store_true")
    ap.add_argument("--subprocess", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    rows = qd.read_manifest(args.manifest)
    if not (0 <= args.row < len(rows)):
        raise SystemExit(f"--row {args.row} outside manifest (0..{len(rows)-1})")
    row = rows[args.row]
    ident = row.get("cell_id") or row.get("point_id")
    if ("cell_id" in row) != (args.arm == "ra"):
        raise SystemExit(f"--arm {args.arm} does not match manifest schema "
                         f"({sorted(row)})")
    if row.get("seed_scheme") and row["seed_scheme"] != qd.seeding.SCHEME:
        raise SystemExit(f"manifest scheme {row['seed_scheme']!r} != code "
                         f"scheme {qd.seeding.SCHEME!r}: refusing to mix "
                         "universes")

    cache = (str(args.table_cache_dir) if args.table_cache_dir else
             (str(Path(args.base_root) / "table_cache")
              if args.arm == "ddm" else None))
    if cache:
        Path(cache).mkdir(parents=True, exist_ok=True)
    cfg0 = cell_config(args.arm, row, cache)   # every §2 assertion runs here
    provenance = {"git_sha": qd.git_sha(),
                  "config_hash": qd.config_hash(cfg0)}

    print(f"[{args.arm}:{ident}] runs {args.first_run}..{args.last_run} "
          f"cfg={provenance['config_hash']} sha={provenance['git_sha']}")
    sys.stdout.flush()

    runner = None
    if not args.configs_only:
        runner = SubprocessRunner() if args.subprocess else InProcessRunner()

    failures_dir = args.failures_dir or (Path(args.base_root) / "failures")
    tag = args.task_tag or f"{args.arm}_{args.row}_{args.first_run}"
    fail_log = failures_dir / f"task_{tag}.log"

    n_run = n_skip = n_fail = 0
    t0 = time.time()
    for run_id in range(args.first_run, args.last_run + 1):
        rep_dir = qd.replicate_dir(args.base_root, row, run_id)
        done = rep_dir / ".done"
        if done.exists() and not args.force:
            n_skip += 1
            continue
        cfg_path = build_replicate(args.arm, row, run_id, rep_dir,
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
        except Exception as exc:                 # noqa: BLE001 — data, not abort
            n_fail += 1
            failures_dir.mkdir(parents=True, exist_ok=True)
            with open(fail_log, "a", encoding="utf-8") as fh:
                fh.write(f"{cfg_path}\t{exc!r}\n")
            print(f"  [{ident}] run {run_id} FAILED: {exc!r}")
            sys.stdout.flush()

    if not args.configs_only:
        total = max(n_run, 1)
        print(f"[{args.arm}:{ident}] done: ran {n_run}, skipped {n_skip}, "
              f"failed {n_fail} in {time.time() - t0:.1f}s "
              f"({(time.time() - t0) / total:.2f} s/run)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
