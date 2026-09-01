# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Effective config generation (Sections 3, 8, 9).

One effective config per (condition, replicate), produced by overriding an arm's base
template in memory. The base files are NEVER modified.

The design's central claim is that the arms differ in the decision rule and in
NOTHING ELSE, so this module is written to make a violation loud rather than silent:

  * `_patch_shared` writes every locked parameter of Section 4.1 into every arm from
    `flexibility.factors`, regardless of what the template happened to hold. Two arms
    at the same (delta, replicate) then differ only inside the model block, and
    `assert_arms_matched` checks exactly that by diffing two generated configs.
  * `_patch_model` REQUIRES the model block it is asked to patch to exist. The
    historical generator only knew how to reach `mean_field_model` and would have
    silently no-opped on a DDM template, leaving the arm at whatever the template
    said -- the same failure class as the seeding bug it also had.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Optional

from flexibility import factors, matrix, seeds
from flexibility.matrix import Condition

_ROOT = Path(__file__).resolve().parent.parent


class ConfigGenerationError(RuntimeError):
    """Raised when a template cannot carry the condition it was asked to carry."""


def load_template(arm: str) -> dict:
    """Read an arm's base template (read-only; callers get their own deep copy)."""
    rel = factors.TEMPLATES[arm]
    path = _ROOT / rel
    if not path.is_file():
        raise ConfigGenerationError(f"template for arm {arm!r} not found: {path}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _strip_comments(obj):
    """Drop `_comment*` documentation keys from generated configs.

    The manifest documents the condition; 9 200 copies of the template prose are
    noise, and they make a config diff between two arms unreadable.
    """
    if isinstance(obj, dict):
        return {k: _strip_comments(v) for k, v in obj.items()
                if not str(k).startswith("_comment")}
    if isinstance(obj, list):
        return [_strip_comments(v) for v in obj]
    return obj


def _require(container: dict, key: str, where: str) -> dict:
    """Fetch a required mapping, naming what is missing if it is not there."""
    value = container.get(key)
    if not isinstance(value, dict):
        raise ConfigGenerationError(
            f"{where}: expected a '{key}' mapping, got {type(value).__name__}. "
            "The template does not match the arm it was selected for."
        )
    return value


# ---------------------------------------------------------------------------
# Section 3: what is held identical across arms
# ---------------------------------------------------------------------------
def _patch_shared(env: dict, cond: Condition) -> None:
    """Write every locked parameter into the config, whatever the template held.

    This is deliberately unconditional. Reading a locked value out of a template is
    how the two arms drifted apart in the first place (one config at white_rate
    0.035, the other at 0.07071068, both carrying the same comment claiming c =
    0.0495); sourcing all of them from `factors` makes that impossible.
    """
    d = cond.derived

    env["num_runs"] = 1
    env["time_limit"] = int(factors.TIME_LIMIT)
    env["ticks_per_second"] = factors.TICKS_PER_SECOND
    env.pop("gui", None)
    # A leftover position-swap list would fire alongside the strength swap.
    env.pop("target_position_swaps", None)

    env["sensory_stream"] = {
        "mode": "shared",
        "frozen_sd": float(factors.FROZEN_SD),
        "white_rate": float(factors.WHITE_RATE),
        "seed": None,
    }

    # The world change. 'attributes' is mandatory: the arena defaults it to
    # ("position",), which for this experiment is a no-op -- both models sign their
    # decision variable by target_ids order, so exchanging coordinates leaves q0 - q1
    # untouched and produces no drift reversal.
    env["post_bifurcation_swap"] = {
        "pairs": [["static_0.s#0", "static_1.s#0"]],
        "delay_ticks": int(factors.SWAP_DELAY_TICKS),
        "attributes": ["strength", "color"],
    }

    env["termination"] = {
        "type": "proximity",
        "target_ids": ["static_0.s#0", "static_1.s#0"],
        "radius": float(factors.TERMINATION_RADIUS),
        "agent_ids": "any",
    }

    # The arena is written WHOLESALE rather than patched, so no template key can
    # survive to differ between arms -- not even a cosmetic one. SquareArena reads
    # 'side' and SILENTLY IGNORES 'radius', so a template carrying only 'radius' runs
    # the default side; dropping the whole block removes that trap too. The seed is
    # filled in per replicate by `replicate_config`.
    arenas = _require(env, "arenas", "environment")
    for name in list(arenas):
        arenas[name] = {
            "random_seed": 0,
            "side": factors.ARENA_SIDE,
            "_id": "square",
            "color": "white",
        }

    objects = _require(env, "objects", "environment")
    for oid, (pos, q, color) in {
        "static_0": (d["pos_static_0"], d["q0"], "green"),
        "static_1": (d["pos_static_1"], d["q1"], "red"),
    }.items():
        obj = _require(objects, oid, "environment.objects")
        obj["position"] = [list(pos)]
        obj["strength"] = [float(q)]
        obj["color"] = color

    agents = _require(env, "agents", "environment")
    for agent in agents.values():
        if not isinstance(agent, dict):
            continue
        agent["position"] = [[0.0, 0.0, 0.0]]
        agent["linear_velocity"] = float(factors.LINEAR_VELOCITY)
        agent["angular_velocity"] = factors.ANGULAR_VELOCITY
        # Equality with the arena rate is a FATAL precondition under the shared
        # stream; set from the same constant so it cannot drift.
        agent["ticks_per_second"] = factors.TICKS_PER_SECOND

    results = _require(env, "results", "environment")
    results["snapshots_per_second"] = factors.SNAPSHOTS_PER_SECOND


# ---------------------------------------------------------------------------
# Section 9.6: the arm-specific patch, which must never silently no-op
# ---------------------------------------------------------------------------
def _patch_model(env: dict, cond: Condition) -> None:
    """Apply the one thing that actually differs between arms: the decision rule."""
    agents = _require(env, "agents", "environment")
    block_key = "mean_field_model" if cond.is_ra else "embodied_pure_ddm"

    patched = 0
    for agent_name, agent in agents.items():
        if not isinstance(agent, dict):
            continue
        block = agent.get(block_key)
        if not isinstance(block, dict):
            raise ConfigGenerationError(
                f"arm {cond.arm!r} needs agent {agent_name!r} to carry a "
                f"'{block_key}' block, and it does not. Template "
                f"{factors.TEMPLATES[cond.arm]!r} is wrong for this arm; patching "
                "would otherwise no-op silently and the arm would run as whatever "
                "the template said."
            )
        if cond.is_ra:
            _patch_ra(block, cond)
        else:
            _patch_ddm(block, cond)
        patched += 1

    if patched != 1:
        raise ConfigGenerationError(
            f"expected exactly one agent to patch, patched {patched}. The design is "
            "a SINGLE agent committing to one of two targets."
        )


def _patch_ra(block: dict, cond: Condition) -> None:
    """Ring-attractor arm: the gain, and the shared-stream precondition."""
    block["u"] = float(factors.ARM_U[cond.arm])
    # The kernel is FIXED at v = 0.5: u = 6.2 is near-critical for THIS kernel, so
    # changing v would silently invalidate the near-critical arm's identity.
    block["v"] = 0.5
    # Fatal under sensory_stream.mode 'shared': the model-owned sensory noise has
    # moved upstream, and a non-zero sigma_s would add a second, UNSHARED noise
    # source on top -- silently unpairing the arms while every seed still matched.
    block["sigma_s"] = 0.0


def _patch_ddm(block: dict, cond: Condition) -> None:
    """DDM arm: the boundary policy, and the solver horizon that follows from v."""
    d = cond.derived
    if cond.arm != "ddm_bellman":
        raise ConfigGenerationError(f"unknown DDM arm {cond.arm!r}")
    # Required to be 0 under the shared stream, for the same reason as sigma_s.
    block["eta_rate"] = [0.0, 0.0]
    # c_e, the cost of an error in seconds. Fixed across the campaign, so this is
    # the same number in every cell; it is written from the condition rather than
    # from the template so a template edit cannot silently change the criterion.
    block["cost_ratio"] = float(d["c_e"])
    block["n_sub"] = int(factors.N_SUB)

    block["threshold_policy"] = "bellman"
    bellman = _require(block, "bellman", "embodied_pure_ddm")
    bellman["terminal"] = "halt_sprt"
    # T_max stays null: the model derives r0/v from the onset geometry. N_t does
    # NOT follow it automatically, so a stale value coarsens the solver grid
    # rather than shortening the horizon -- 43.30 s over the historical 8660
    # steps is dt = 5 ms, five times the intended resolution.
    bellman["T_max"] = None
    bellman["N_t"] = int(d["N_t"])


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def condition_config(
    cond: Condition,
    base: Optional[dict] = None,
    table_cache_dir: Optional[str] = None,
) -> dict:
    """The replicate-INDEPENDENT effective config for one condition-point.

    The seed is left null here; `replicate_config()` fills it.
    """
    data = copy.deepcopy(base if base is not None else load_template(cond.arm))
    env = _require(data, "environment", "config root")
    _patch_shared(env, cond)
    _patch_model(env, cond)

    if table_cache_dir and cond.arm == "ddm_bellman":
        agent = next(iter(env["agents"].values()))
        agent["embodied_pure_ddm"]["bellman"]["table_cache_dir"] = str(table_cache_dir)

    return data


def replicate_config(
    cond: Condition,
    replicate: int,
    out_dir: str,
    base: Optional[dict] = None,
    table_cache_dir: Optional[str] = None,
    horizon_check_factor: Optional[float] = None,
) -> dict:
    """The effective config for one run, seeded and pointed at its output directory."""
    data = condition_config(cond, base=base, table_cache_dir=table_cache_dir)
    env = data["environment"]

    # BOTH seeds are written explicitly, never left null for the simulator to resolve.
    # The sensory seed excludes the arm, so every arm replays the same realisation at
    # this (delta, replicate); the arena seed includes it, so model-internal
    # randomness is deliberately independent. Writing both out makes the pairing
    # readable straight from the config rather than implied by a resolution rule.
    token = matrix.delta_token(cond.delta)
    env["sensory_stream"]["seed"] = int(seeds.sensory_seed(token, replicate))
    internal = int(seeds.internal_seed(cond.arm, token, replicate))
    for arena in env["arenas"].values():
        if isinstance(arena, dict):
            arena["random_seed"] = internal

    if horizon_check_factor is not None and cond.arm == "ddm_bellman":
        agent = next(iter(env["agents"].values()))
        bellman = agent["embodied_pure_ddm"].get("bellman")
        if isinstance(bellman, dict):
            bellman["T_max_check_factor"] = float(horizon_check_factor)

    env["results"]["base_path"] = str(out_dir)
    return _strip_comments(data)


def output_dir(root: str, cond: Condition, replicate: int) -> str:
    """`root/{arm}/diff_{pct}/replicate_{n}` — the arm replaces the old `u_{value}`."""
    return os.path.join(
        root, cond.arm, f"diff_{cond.delta * 100:.4f}pct", f"replicate_{replicate}"
    )


# ---------------------------------------------------------------------------
# The check the design asks for in Section 9
# ---------------------------------------------------------------------------
def assert_arms_matched(arm_a: str, arm_b: str, delta: float, replicate: int) -> None:
    """Verify two arms at the same (delta, replicate) are matched where it matters.

    Three separate requirements, checked rather than eyeballed from a diff:

      1. the SENSORY seed is IDENTICAL -- this is what pairs the arms;
      2. the arena (internal) seed DIFFERS -- model-internal randomness is deliberately
         independent, and two arms sharing it would imply a coupling that does not
         exist;
      3. everything outside the model block is otherwise identical, so the arms differ
         in the decision rule and in nothing else.

    The arena seed is elided before the structural comparison precisely because it is
    *expected* to differ; requirement 2 checks it explicitly instead.
    """
    def strip_models(cfg: dict) -> dict:
        cfg = copy.deepcopy(cfg)
        for agent in cfg["environment"]["agents"].values():
            if isinstance(agent, dict):
                agent.pop("mean_field_model", None)
                agent.pop("embodied_pure_ddm", None)
                agent.pop("moving_behavior", None)
        cfg["environment"]["results"]["base_path"] = "<elided>"
        for arena in cfg["environment"]["arenas"].values():
            if isinstance(arena, dict):
                arena["random_seed"] = "<elided: checked separately>"
        return cfg

    conds = {c.name: c for c in matrix.build()}
    token = matrix.delta_token(delta)
    try:
        ca = conds[f"{arm_a}__{token}"]
        cb = conds[f"{arm_b}__{token}"]
    except KeyError as exc:
        grid = ", ".join(f"{d * 100:.4f}%" for d in factors.delta_grid())
        raise KeyError(
            f"delta={delta} ({token}) is not a grid point. The grid is: {grid}. "
            "The Section 4.3 landmarks are boundaries between regimes, not cells; "
            "they do not have conditions of their own."
        ) from exc

    a = replicate_config(ca, replicate, "<out>")
    b = replicate_config(cb, replicate, "<out>")

    # 1. The sensory seed MUST match — this is the pairing.
    sens_a = a["environment"]["sensory_stream"]["seed"]
    sens_b = b["environment"]["sensory_stream"]["seed"]
    if sens_a is None or sens_b is None:
        raise AssertionError(
            f"{arm_a} or {arm_b} has a null sensory_stream.seed at delta={delta}, "
            f"replicate={replicate}. Both seeds are written explicitly so the pairing "
            "is readable from the config; a null here means the generator did not."
        )
    if sens_a != sens_b:
        raise AssertionError(
            f"{arm_a} and {arm_b} at delta={delta}, replicate={replicate} have "
            f"different SENSORY seeds ({sens_a} vs {sens_b}); the arms are UNPAIRED."
        )

    # 2. The arena (internal) seed must DIFFER — independence, not a defect.
    int_a = next(iter(a["environment"]["arenas"].values()))["random_seed"]
    int_b = next(iter(b["environment"]["arenas"].values()))["random_seed"]
    if int_a == int_b:
        raise AssertionError(
            f"{arm_a} and {arm_b} at delta={delta}, replicate={replicate} share an "
            f"arena seed ({int_a}). Model-internal randomness is meant to be "
            "independent across arms; the arm is missing from the internal seed key."
        )

    sa, sb = strip_models(a), strip_models(b)
    if sa != sb:
        diffs = [
            k for k in set(sa["environment"]) | set(sb["environment"])
            if sa["environment"].get(k) != sb["environment"].get(k)
        ]
        raise AssertionError(
            f"{arm_a} and {arm_b} differ OUTSIDE the model block at "
            f"environment keys {sorted(diffs)}; the comparison is not controlled."
        )
