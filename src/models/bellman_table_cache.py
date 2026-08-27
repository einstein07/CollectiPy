# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Disk cache for solved Bellman boundary tables (CAMPAIGN_SPEC Section 7.3).

`z(t)` depends only on `(A, c, c_e, geometry, solver grid)` and never on the seed, so
it is identical across every replicate of a campaign condition-point. Solving it once
and loading it thereafter is the single biggest cost lever in the campaign: in the
pilot the solve consumed 64 of 85 core-hours.

Design constraints, in order:

- **The cache must never change the answer.** A hit returns the exact arrays a solve
  would have produced, because they ARE a solve's arrays, written by whichever process
  got there first. Everything that determines the table is in the key; anything not in
  the key (seeds, replicate indices, logging cadence) provably does not enter the solve.
- **Keys are computed from the solver's own inputs at the call site**, inside
  `_bellman_threshold`, after the model has resolved `A`, `c` and the onset geometry.
  Reproducing that float pipeline anywhere else would be fragile, which is why the
  precompute job populates the cache BY RUNNING THE MODEL (one throwaway replicate per
  condition) rather than by reimplementing the key derivation.
- **Writes are atomic** (temp file in the same directory, then `os.replace`), so
  concurrent array tasks racing on a cold cache produce one valid file and no lock.

The key is the sha1 of the canonical rendering of
`(A, c, c_e, r0, L, v, T_max, N_x, N_t, X_max_factor, scheme)` — the spec's list plus
`X_max_factor` and `scheme`, which also parameterise the solve; omitting them could
alias two different tables under one key.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("sim.bellman_table_cache")

#: Bump when the on-disk layout changes; part of every key, so stale caches miss.
_FORMAT = 1


def table_key(
    *,
    A: float,
    c: float,
    c_e: float,
    r0: float,
    L: float,
    v: float,
    T_max: float,
    N_x: int,
    N_t: int,
    X_max_factor: float,
    scheme: str,
) -> str:
    """Return the cache key for one solve. Floats render via repr: bit-identical
    inputs give the same key, and any real difference in inputs changes it."""
    parts = [
        f"fmt={_FORMAT}",
        f"A={float(A)!r}", f"c={float(c)!r}", f"c_e={float(c_e)!r}",
        f"r0={float(r0)!r}", f"L={float(L)!r}", f"v={float(v)!r}",
        f"T_max={float(T_max)!r}",
        f"N_x={int(N_x)}", f"N_t={int(N_t)}",
        f"X_max_factor={float(X_max_factor)!r}",
        f"scheme={str(scheme)}",
    ]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()


def _path(cache_dir, key: str) -> Path:
    return Path(cache_dir) / f"bellman_{key}.npz"


def load_table(cache_dir, key: str):
    """Return `(t_grid, z_arr, meta)` for a cached solve, or None on a miss.

    A file that exists but cannot be read (torn copy, wrong format) is treated as a
    miss: the caller re-solves and rewrites it, which is always safe.
    """
    path = _path(cache_dir, key)
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            t_grid = np.array(data["t_grid"], dtype=float)
            z_arr = np.array(data["z_arr"], dtype=float)
            meta = {
                "z_myopic_onset": float(data["z_myopic_onset"]),
                "wall_time_s": float(data["wall_time_s"]),
                "inputs": {k: v for k, v in zip(
                    [str(s) for s in data["input_names"]],
                    [float(x) for x in data["input_values"]],
                )},
                "scheme": str(data["scheme"]),
            }
    except Exception as exc:  # torn/foreign file: miss, never fatal
        logger.warning("bellman table cache: unreadable %s (%s); re-solving", path, exc)
        return None
    return t_grid, z_arr, meta


def save_table(
    cache_dir,
    key: str,
    t_grid: np.ndarray,
    z_arr: np.ndarray,
    *,
    inputs: dict,
    z_myopic_onset: float,
    wall_time_s: float,
    scheme: str,
    horizon_ok: Optional[bool] = None,
) -> Optional[Path]:
    """Atomically persist one solved table. Failure to write is logged, never raised:
    the solve already succeeded, and the run must not die on a full cache disk."""
    cache_dir = Path(cache_dir)
    path = _path(cache_dir, key)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        names = sorted(inputs)
        fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                np.savez(
                    fh,
                    t_grid=np.asarray(t_grid, dtype=float),
                    z_arr=np.asarray(z_arr, dtype=float),
                    z_myopic_onset=float(z_myopic_onset),
                    wall_time_s=float(wall_time_s),
                    horizon_ok=float("nan") if horizon_ok is None else float(horizon_ok),
                    input_names=np.array(names),
                    input_values=np.array([float(inputs[k]) for k in names]),
                    scheme=np.array(str(scheme)),
                )
            os.replace(tmp, path)          # atomic on POSIX: last writer wins, whole
        finally:                           # files only, no partial state visible
            if os.path.exists(tmp):
                os.unlink(tmp)
    except Exception as exc:
        logger.warning("bellman table cache: could not write %s (%s)", path, exc)
        return None
    return path


def scan_tables(cache_dir):
    """Yield `(path, meta_with_z0)` for every readable table in the cache.

    For reporting (the dry run's z_bellman(0) column); never on the run path.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        return
    for path in sorted(cache_dir.glob("bellman_*.npz")):
        try:
            with np.load(path, allow_pickle=False) as data:
                meta = {k: float(v) for k, v in zip(
                    [str(s) for s in data["input_names"]],
                    [float(x) for x in data["input_values"]],
                )}
                meta["z0"] = float(np.array(data["z_arr"])[0])
                meta["z_myopic_onset"] = float(data["z_myopic_onset"])
                meta["wall_time_s"] = float(data["wall_time_s"])
                if "horizon_ok" in data:
                    meta["horizon_ok"] = float(data["horizon_ok"])
        except Exception:
            continue
        yield path, meta
