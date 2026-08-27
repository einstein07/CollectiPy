# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Effective config generation (CAMPAIGN_SPEC.md Sections 5, 6, 8).

One effective config per (condition, replicate), produced by overriding the base
template in memory. The base file is NEVER modified. Only factor values are touched:
target strengths and positions, the criterion, the policy selector for the control
arms, the solver's N_t, the two seeds, and the output paths. Every locked parameter
rides through from the base config untouched — which is what makes the base file the
single statement of Section 1.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Optional

from campaign import factors, seeds
from campaign.matrix import Condition

_ROOT = Path(__file__).resolve().parent.parent


def load_base() -> dict:
    """Read the base template (read-only; callers get their own deep copy)."""
    with open(_ROOT / factors.BASE_CONFIG, encoding="utf-8") as fh:
        return json.load(fh)


def _strip_comments(obj):
    """Drop the `_comment*` documentation keys from generated configs: the manifest
    documents the condition, and 140 copies of the template prose are pure noise."""
    if isinstance(obj, dict):
        return {k: _strip_comments(v) for k, v in obj.items()
                if not str(k).startswith("_comment")}
    if isinstance(obj, list):
        return [_strip_comments(v) for v in obj]
    return obj


def condition_config(
    cond: Condition,
    base: Optional[dict] = None,
    table_cache_dir: Optional[str] = None,
) -> dict:
    """The replicate-INDEPENDENT effective config for one condition-point.

    Seeds are left null here; `replicate_config()` fills them. This is the config
    written into the condition directory (Section 8): together with the manifest's
    seed scheme it identifies every replicate's exact config.
    """
    data = copy.deepcopy(base if base is not None else load_base())
    env = data["environment"]
    env.pop("gui", None)
    env["num_runs"] = 1
    env["time_limit"] = int(factors.TIME_LIMIT)

    d = cond.derived
    env["objects"]["static_0"]["position"] = [list(d["pos_static_0"])]
    env["objects"]["static_1"]["position"] = [list(d["pos_static_1"])]
    env["objects"]["static_0"]["strength"] = [d["q0"]]
    env["objects"]["static_1"]["strength"] = [d["q1"]]

    ag = env["agents"]["movable_0"]
    ag["position"] = [[0.0, 0.0, 0.0]]
    blk = ag["embodied_pure_ddm"]

    blk["threshold_policy"] = cond.threshold_policy
    if cond.arm == "main":
        blk["cost_ratio"] = float(cond.c_e)
    elif cond.arm == "quasi_static":
        # Section 5.1: the static optimum re-evaluated EACH TICK at the live
        # geometry. The error mode and cost_ratio are the same keys the Bellman
        # arm uses [READ FROM CODE: _update_threshold -> _geometric_threshold].
        blk["cost_ratio"] = float(cond.c_e)
        blk["threshold_update_ticks"] = 1
        blk["boundary_mode"] = "static"
    elif cond.arm == "static":
        blk["z_manual"] = float(cond.z_manual)
        # The manual policy reads no cost; cost_ratio is retained purely as
        # provenance of where the z value came from (Section 5.2 proposal).
        if cond.from_c_e is not None:
            blk["cost_ratio"] = float(cond.from_c_e)

    bell = blk["bellman"]
    bell["T_max"] = None                       # r0/v from the live onset geometry
    bell["T_max_check_factor"] = None          # precompute runs the check once
    bell["N_t"] = int(d["N_t"])                # dt = BELLMAN_DT on every geometry
    bell["table_cache_dir"] = (
        str(table_cache_dir) if table_cache_dir else None
    )

    return _strip_comments(data)


def replicate_config(
    cond: Condition,
    replicate: int,
    base_path: str,
    cond_cfg: Optional[dict] = None,
    table_cache_dir: Optional[str] = None,
    horizon_check_factor: Optional[float] = None,
) -> dict:
    """The full per-replicate config: the condition config plus the two seeds
    (Section 6) and this replicate's output path."""
    data = copy.deepcopy(
        cond_cfg if cond_cfg is not None
        else condition_config(cond, table_cache_dir=table_cache_dir)
    )
    env = data["environment"]

    s_seed = seeds.sensory_seed(
        factors.CAMPAIGN_SEED, cond.q_tok, cond.a_tok, replicate
    )
    i_seed = seeds.internal_seed(
        factors.CAMPAIGN_SEED, cond.q_tok, cond.a_tok, cond.crit_tok, replicate
    )
    env["sensory_stream"]["seed"] = s_seed
    for arena in env["arenas"].values():
        arena["random_seed"] = i_seed

    if horizon_check_factor is not None:
        env["agents"]["movable_0"]["embodied_pure_ddm"]["bellman"][
            "T_max_check_factor"] = float(horizon_check_factor)

    results = env.setdefault("results", {})
    results["base_path"] = str(base_path)
    results["sweep_metadata"] = {
        "campaign": "embodied_ddm_campaign",
        "condition": cond.name,
        "arm": cond.arm,
        "replicate": int(replicate),
        "sensory_seed": s_seed,
        "internal_seed": i_seed,
    }
    return data


def config_hash(cfg: dict) -> str:
    """Order-independent hash of an effective config."""
    return hashlib.sha1(
        json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def git_commit() -> str:
    """The current commit, or 'unknown' outside a git checkout."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, capture_output=True,
            text=True, timeout=10, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def manifest(cond: Condition, reps: int, chunk: int, cfg: dict) -> dict:
    """The Section 8 manifest for one condition directory."""
    d, p = cond.derived, cond.predicted
    return {
        "campaign": "embodied_ddm_campaign",
        "condition": cond.name,
        "arm": cond.arm,
        "threshold_policy": cond.threshold_policy,
        "dQ": cond.q_diff,
        "A": d["A"],
        "quality_static_0": d["q0"],
        "quality_static_1": d["q1"],
        "dtheta_deg": cond.ang_sep,
        "r0": d["r0"],
        "L": d["L"],
        "T_max": d["T_max"],
        "c_tau0": d["c_tau0"],
        "v": d["v"],
        "c_e": cond.c_e,
        "z_manual": cond.z_manual,
        "z_manual_from_c_e": cond.from_c_e,
        "sensory_stream": {
            "mode": "shared",
            "frozen_sd": factors.FROZEN_SD,
            "white_rate": factors.WHITE_RATE,
            "noise_scale_c": d["c"],
        },
        "reps": int(reps),
        "chunk": int(chunk),
        "seed_scheme": {
            "campaign_seed": factors.CAMPAIGN_SEED,
            "sensory": "H(campaign_seed, 'sensory', q_tok, a_tok, replicate)",
            "internal": "H(campaign_seed, 'internal', q_tok, a_tok, crit_tok, replicate)",
            "H": "blake2b(digest_size=8) over '|'-joined tokens, mod 2^31",
            "q_tok": cond.q_tok, "a_tok": cond.a_tok, "crit_tok": cond.crit_tok,
        },
        "n_sub": factors.N_SUB,
        "ticks_per_second": factors.TICKS_PER_SECOND,
        "bellman_N_t": d["N_t"],
        "bellman_dt": factors.BELLMAN_DT,
        "predicted": {
            "z": p["z"], "a": p["a"], "accuracy": p["accuracy"],
            "DT": p["DT"], "DT_over_T_max": p["DT_over_T_max"],
        },
        "discretisation_limited": cond.discretisation_limited,
        "evidence_step": p["evidence_step"],
        "git_commit": git_commit(),
        "config_hash": config_hash(cfg),
    }


def startup_line(cond: Condition, rep_range: range) -> str:
    """The Section 8 per-task startup log."""
    d, p = cond.derived, cond.predicted
    crit = (f"c_e: {cond.c_e:g}" if cond.arm != "static"
            else f"z_manual: {cond.z_manual:g}")
    return (
        f"dQ: {cond.q_diff:.0%} (A={d['A']:.3f}) | "
        f"dtheta: {cond.ang_sep}deg (r0={d['r0']:.3f}, L={d['L']:.3f}, "
        f"T_max={d['T_max']:.2f}s, c_tau0={d['c_tau0']:.3f})\n"
        f"policy: {cond.threshold_policy} | {crit} | v: {d['v']:g} | "
        f"n_sub: {factors.N_SUB}\n"
        f"stream: shared (frozen_sd={factors.FROZEN_SD:g}, "
        f"white_rate={factors.WHITE_RATE:g})\n"
        f"replicates: [{rep_range.start}, {rep_range.stop}) | "
        f"predicted: acc {p['accuracy']:.3f}, DT {p['DT']:.2f}s "
        f"({p['DT_over_T_max']:.2f} T_max)"
        + ("  [discretisation_limited]" if cond.discretisation_limited else "")
    )
