# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2026 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Shared machinery for BOTH frontier campaigns (RA slices + DDM rerun).

Implements §2 (grids), §4 (template lineage), §5 (manifests) and the two model
patchers of §6 in one module, so neither campaign can quietly see a different
environment from the other. See RECON.md for every place this departs from the
spec document's sketch.

Single source of truth for the environment: `config/campaign_ddm_base.json`
(the DDM frontier's template). The RA template is that file with ONLY the model
block swapped (RECON §2); the mean_field block is copied from the factorial's
template so the ring parameters cannot drift from the validated values.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in (str(_HERE), str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import seeding  # noqa: E402

# ---------------------------------------------------------------------------
# Trial identity (§1) — one panel: 1 %, Δθ = 60°.
# ---------------------------------------------------------------------------
DTH_DEG = 60
DIFF = 0.01
DIFF_BP = round(DIFF * 10000)          # 100 basis points
QUALITY_BETTER = 5.0
QUALITY_WORSE = round(QUALITY_BETTER * (1.0 - DIFF), 8)   # 4.95
CORRECT_TARGET_ID = "static_0.s#0"

# ---------------------------------------------------------------------------
# Design grids (§2). 52 + 48 = 100 RA cells; 10 DDM points.
# ---------------------------------------------------------------------------
V_GRID = [0.2, 0.3, 0.4, 0.5]
UHAT_GRID = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
             0.90, 1.00, 1.10, 1.25, 1.50]
U_ABS_GRID = [0.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
C_E_GRID = [0.03, 0.1, 0.3, 1, 3, 8, 20, 50, 125, 300]   # verbatim from campaign/factors.py
N_RUNS = 1000                                            # run_id 1..1000, both campaigns

# ---------------------------------------------------------------------------
# Ring constants (asserted against the factorial's template at derivation).
# ---------------------------------------------------------------------------
NUM_NEURONS = 30
BETA = 1.0
ANCHOR_V = 0.5
ANCHOR_U_STAR = 6.157
ANCHOR_TOL = 0.01

# ---------------------------------------------------------------------------
# Task frames. Each model runs in ITS OWN established frame (RECON D-01, the
# researcher's decision): the RA in the archived ra_ddm_frontier_sweep frame,
# the DDM in the campaign_ddm_base frame. Physically shared quantities —
# velocity, angular velocity, geometry, strengths, sensory statistics,
# termination — are identical; comparison is in seconds.
# ---------------------------------------------------------------------------
TARGET_RANGE = 0.5
LINEAR_VELOCITY = 0.05
ANGULAR_VELOCITY = 120

#: Researcher's calibration (2026-08-29): the noise on the EVIDENCE CHANNEL —
#: the difference percept q_hat_0 - q_hat_1, i.e. the DDM's noise scale
#: c = sqrt(2) * white_rate — is exactly TWICE the absolute quality
#: difference: c = 2 dQ = 0.1, hence white_rate = sqrt(2) * dQ ~= 0.070711.
#: `white_rate` is a RATE — per-draw SD = white_rate/sqrt(dt) — so the DDM's
#: 10 Hz stream draws sqrt(10) larger per tick while the integrated noise per
#: second of world time is identical in both frames. Consequences:
#:   difference-channel SD per sqrt(s) (c): 0.1000  = 2.00 x dQ  (the definition)
#:   per-target noise SD per sqrt(s)      : 0.07071 = 1.41 x dQ
#: (The rejected alternative — per-TARGET SD = 2 dQ — would be white_rate 0.1.)
#: This deliberately overrides the DDM campaign's locked 0.035 and the beta-1
#: archive; the archived frontier and the factorial anchor are therefore
#: reference points at a DIFFERENT calibration, not CI-gates (RECON D-09/D-10).
QUALITY_DELTA = round(QUALITY_BETTER - QUALITY_WORSE, 8)      # 0.05
WHITE_RATE = round(math.sqrt(2.0) * QUALITY_DELTA, 8)         # 0.07071068
NOISE_SCALE_C = round(math.sqrt(2.0) * WHITE_RATE, 8)         # 0.1 = 2 x dQ

RA_TICKS_PER_SECOND = 1         # archived frontier RA frame: tick = 1 s
RA_TIME_LIMIT = 1000            # ticks ≡ seconds at 1 tick/s

#: Researcher's instruction (2026-08-30): the DDM runs on the SAME 1 s tick
#: clock as the RA (arena + agent), snapshots_per_second = 1 — overriding the
#: campaign base's 10. With both models on one tick clock the shared percepts
#: are bitwise-identical realizations at equal (seed, target, tick), i.e. the
#: spec's full §3 pairing, not just shared deviates. Evidence substep grows to
#: c*sqrt(1/16) = 0.025; the quasi-static boundaries keep the
#: discretisation-limited set at {0.03, 0.1} (z* = 0.0028, 0.0093 < 0.025;
#: c_e = 0.3 clears it marginally at z* = 0.028 — see RECON D-11).
DDM_TICKS_PER_SECOND = 1        # tick = 1 s, matching the RA
DDM_TIME_LIMIT = 60             # seconds (61 ticks; arrival measured ~9-12 s)
BELLMAN_DT = 1e-3

R0 = TARGET_RANGE * math.cos(math.radians(DTH_DEG) / 2.0)      # 0.4330127…
BELLMAN_N_T = math.ceil((R0 / LINEAR_VELOCITY) / BELLMAN_DT)   # 8661
POS_STATIC_0 = [TARGET_RANGE * math.cos(math.radians(DTH_DEG) / 2.0),
                -TARGET_RANGE * math.sin(math.radians(DTH_DEG) / 2.0), 0]
POS_STATIC_1 = [POS_STATIC_0[0], -POS_STATIC_0[1], 0]

# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
DDM_BASE_CONFIG = _ROOT / "config" / "campaign_ddm_base.json"
FACTORIAL_RA_CONFIG = _ROOT / "config" / "mean_field_2_targets_no_viz.json"
RA_TEMPLATE = _ROOT / "config" / "ra_ddm_frontier_ra_template.json"
DDM_TEMPLATE = _ROOT / "config" / "ra_ddm_frontier_ddm_template.json"

RA_MANIFEST_NAME = "manifest.csv"
DDM_MANIFEST_NAME = "ddm_manifest.csv"
RA_FIELDS = ["cell_id", "sweep", "v", "u_hat", "u_star", "u",
             "diff", "n_runs", "seed_scheme"]
DDM_FIELDS = ["point_id", "c_e", "diff", "n_runs", "seed_scheme"]


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def git_sha() -> str:
    """HEAD sha, '-dirty' when modified; never raises."""
    try:
        sha = subprocess.run(["git", "-C", str(_ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=15)
        if sha.returncode != 0:
            return "unknown"
        out = sha.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(_ROOT), "status", "--porcelain",
             "--untracked-files=no"],
            capture_output=True, text=True, timeout=30)
        if dirty.returncode == 0 and dirty.stdout.strip():
            out += "-dirty"
        return out
    except Exception:                       # noqa: BLE001 — provenance never fatal
        return "unknown"


def config_hash(cfg: dict) -> str:
    """Cell-level hash: per-replicate fields (seeds, paths, metadata) stripped,
    so all replicates of a cell share one hash."""
    stripped = copy.deepcopy(cfg)
    env = stripped.get("environment", {})
    for arena in env.get("arenas", {}).values():
        if isinstance(arena, dict):
            arena.pop("random_seed", None)
    stream = env.get("sensory_stream")
    if isinstance(stream, dict):
        stream.pop("seed", None)
    results = env.get("results")
    if isinstance(results, dict):
        results.pop("base_path", None)
        results.pop("sweep_metadata", None)
    for ag in env.get("agents", {}).values():
        bell = (ag.get("embodied_pure_ddm") or {}).get("bellman")
        if isinstance(bell, dict):
            bell.pop("table_cache_dir", None)   # machine-local, not physics
    blob = json.dumps(stripped, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


# ---------------------------------------------------------------------------
# u*(v) — from the simulator's own kernel builder (factorial rule, §2 Set R).
# ---------------------------------------------------------------------------
def u_star(v: float) -> tuple[float, float]:
    """Return (u_star, lambda_max) for kernel shape v at the runtime N and β."""
    import numpy as np
    from models.mean_field_systems import MeanFieldSystem
    system = MeanFieldSystem(num_neurons=NUM_NEURONS, v=float(v), beta=BETA)
    lam = float(np.max(np.real(np.linalg.eigvals(system.M))))
    if lam <= 0.0:
        raise SystemExit(f"lambda_max(v={v}) = {lam:.6g} is not positive; "
                         "u* is undefined — halt for human review.")
    sech2 = 1.0 / math.cosh(BETA) ** 2
    return 1.0 / (lam * sech2), lam


def anchor_check() -> dict:
    """§2's 1 % anchor gate at v = 0.5. Halts rather than rescaling."""
    us, lam = u_star(ANCHOR_V)
    rel = abs(us - ANCHOR_U_STAR) / ANCHOR_U_STAR
    report = {"v": ANCHOR_V, "u_star": us, "lambda_max": lam,
              "expected": ANCHOR_U_STAR, "relative_error": rel,
              "tolerance": ANCHOR_TOL, "passed": rel <= ANCHOR_TOL}
    if not report["passed"]:
        print("ANCHOR CHECK FAILED — halting for human review.", file=sys.stderr)
        print(json.dumps(report, indent=2), file=sys.stderr)
        raise SystemExit(2)
    return report


# ---------------------------------------------------------------------------
# Template derivation (§4). Both templates come from the DDM base.
# ---------------------------------------------------------------------------
def _strip_comments(obj):
    if isinstance(obj, dict):
        return {k: _strip_comments(v) for k, v in obj.items()
                if not str(k).startswith("_comment")}
    if isinstance(obj, list):
        return [_strip_comments(v) for v in obj]
    return obj


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def build_ddm_template() -> dict:
    """The q01_a60 condition config, seeds and cost_ratio left for patching."""
    data = _strip_comments(_load_json(DDM_BASE_CONFIG))
    env = data["environment"]
    env.pop("gui", None)
    env["logging"] = {"console": False, "console_level": "ERROR",
                      "file_level": "ERROR"}
    env["num_runs"] = 1
    env["time_limit"] = int(DDM_TIME_LIMIT)
    env["objects"]["static_0"]["position"] = [list(POS_STATIC_0)]
    env["objects"]["static_1"]["position"] = [list(POS_STATIC_1)]
    env["objects"]["static_0"]["strength"] = [QUALITY_BETTER]
    env["objects"]["static_1"]["strength"] = [QUALITY_WORSE]
    ag = env["agents"]["movable_0"]
    ag["position"] = [[0.0, 0.0, 0.0]]
    # Researcher's instruction (2026-08-30): DDM on the RA's 1 s tick clock,
    # arena AND agent (they alias otherwise), per-tick snapshots.
    env["ticks_per_second"] = int(DDM_TICKS_PER_SECOND)
    ag["ticks_per_second"] = int(DDM_TICKS_PER_SECOND)
    env.setdefault("results", {})["snapshots_per_second"] = int(
        DDM_TICKS_PER_SECOND)
    _check_stream(env)
    blk = env["agents"]["movable_0"]["embodied_pure_ddm"]
    blk["threshold_policy"] = "bellman"
    blk["cost_ratio"] = None                # per point
    bell = blk["bellman"]
    bell["T_max"] = None                    # r0/v from the live onset geometry
    bell["T_max_check_factor"] = None       # precompute runs the check once
    bell["N_t"] = int(BELLMAN_N_T)
    bell["table_cache_dir"] = None          # set at batch time
    _assert_frame(data, "ddm")
    return data


def build_ra_template() -> dict:
    """The archived ra_ddm_frontier_sweep environment, exactly (RECON D-01).

    Source: `config/mean_field_2_targets_no_viz.json` — the template that sweep
    patched — with precisely the patches its archived replicate configs carry:
    linear_velocity 0.05, angular_velocity 120, sigma_s 0, the shared sensory
    block (white_rate 0.035 — the archived runs' value, NOT the 0.1 that
    drifted into one cluster copy's env block; RECON D-09), quiet logging.
    Everything else — 1 tick/s, time_limit 1000, arena radius 1, target
    positions (0.433, ∓0.25), strengths 5.0/4.95, the whole mean_field block —
    is inherited untouched; only u and v are patched per cell.
    """
    data = _strip_comments(_load_json(FACTORIAL_RA_CONFIG))
    env = data["environment"]
    env.pop("gui", None)
    env["logging"] = {"console": False, "console_level": "ERROR",
                      "file_level": "ERROR"}
    env["num_runs"] = 1
    env["sensory_stream"] = {"mode": "shared", "frozen_sd": 0.0,
                             "white_rate": WHITE_RATE, "seed": None}
    ag = env["agents"]["movable_0"]
    ag["linear_velocity"] = LINEAR_VELOCITY        # template says 0.001
    ag["angular_velocity"] = ANGULAR_VELOCITY      # template says 10 (orbits)
    mf = ag["mean_field_model"]

    # The archived sweep's premises, asserted (any drift silently changes the
    # experiment; same checks as the factorial's config_patch.py).
    expected = {"num_neurons": NUM_NEURONS, "beta": BETA, "kappa": 20,
                "integration_time": 50, "integration_dt": 0.1,
                "g_threshold": 0.6, "sigma": 1.5, "use_thresholding": False,
                "scale_velocity": False, "g_adapt": 0.0}
    for key, want in expected.items():
        if mf.get(key) != want:
            raise SystemExit(
                f"{FACTORIAL_RA_CONFIG}: mean_field_model.{key} = "
                f"{mf.get(key)!r}, expected {want!r} — see the factorial's "
                "RECON.md before changing either.")
    for name, want in (("static_0", QUALITY_BETTER), ("static_1", QUALITY_WORSE)):
        got = [float(x) for x in env["objects"][name]["strength"]]
        if got != [want]:
            raise SystemExit(f"objects.{name}.strength = {got}, expected [{want}]")
    mf["sigma_s"] = 0.0                     # shared-stream rule (percept_stream.py)
    mf["u"] = None                          # per cell
    mf["v"] = None                          # per cell
    _assert_frame(data, "ra")
    return data


def _check_stream(env: dict) -> None:
    """Enforce the matched sensory front end on a template's stream block.

    mode/frozen_sd must already be the campaign's (shared / 0.0); white_rate
    is SET to the noise = 2 x dQ calibration — deliberately overriding the DDM
    base's locked 0.035 (RECON D-09)."""
    stream = env["sensory_stream"]
    if (stream.get("mode"), float(stream.get("frozen_sd", -1))) != ("shared", 0.0):
        raise SystemExit(f"sensory_stream mode/frozen_sd drifted "
                         f"(expected shared / 0.0): {stream}")
    stream["white_rate"] = WHITE_RATE
    stream["seed"] = None


def _assert_frame(cfg: dict, model: str) -> None:
    """Post-derivation assertions, per model frame (RECON D-01)."""
    env = cfg["environment"]
    ag = env["agents"]["movable_0"]
    tps = RA_TICKS_PER_SECOND if model == "ra" else DDM_TICKS_PER_SECOND
    tl = RA_TIME_LIMIT if model == "ra" else DDM_TIME_LIMIT
    if int(env["ticks_per_second"]) != tps or int(ag["ticks_per_second"]) != tps:
        raise SystemExit(f"[{model}] arena/agent ticks_per_second must both be "
                         f"{tps} (they alias otherwise)")
    if int(env["time_limit"]) != tl:
        raise SystemExit(f"[{model}] time_limit must be {tl}")
    _check_stream(env)
    if float(ag["linear_velocity"]) != LINEAR_VELOCITY:
        raise SystemExit(f"linear_velocity must be {LINEAR_VELOCITY}")
    if float(ag["angular_velocity"]) != ANGULAR_VELOCITY:
        raise SystemExit(f"angular_velocity must be {ANGULAR_VELOCITY}")
    turn_radius = float(ag["linear_velocity"]) / math.radians(
        float(ag["angular_velocity"]))
    if turn_radius > float(env["termination"]["radius"]):
        raise SystemExit(
            f"minimum turn radius {turn_radius:.4f} m exceeds the termination "
            f"radius {env['termination']['radius']} m — the agent would orbit")
    if model == "ra":
        def walk(node, path=""):
            if isinstance(node, dict):
                for k, v in node.items():
                    if k in ("eta_rate", "lambda_t"):
                        raise SystemExit(
                            f"DDM accumulator key '{k}' at {path or '<root>'} "
                            "in an RA config")
                    walk(v, f"{path}.{k}" if path else k)
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    walk(v, f"{path}[{i}]")
        walk(cfg)


def write_templates() -> None:
    """Serialise both derived templates into config/ (deterministic)."""
    for path, template in ((RA_TEMPLATE, build_ra_template()),
                           (DDM_TEMPLATE, build_ddm_template())):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(template, fh, indent=2)
            fh.write("\n")


# ---------------------------------------------------------------------------
# Patchers (§6) — the ONLY per-cell / per-point edits.
# ---------------------------------------------------------------------------
def patch_ra(template: dict, u: float, v: float) -> dict:
    cfg = copy.deepcopy(template)
    mf = cfg["environment"]["agents"]["movable_0"]["mean_field_model"]
    mf["u"] = float(u)
    mf["v"] = float(v)
    return cfg


def patch_ddm(template: dict, c_e: float, table_cache_dir: str | None) -> dict:
    cfg = copy.deepcopy(template)
    blk = cfg["environment"]["agents"]["movable_0"]["embodied_pure_ddm"]
    blk["cost_ratio"] = float(c_e)
    blk["bellman"]["table_cache_dir"] = (str(table_cache_dir)
                                         if table_cache_dir else None)
    return cfg


def apply_seeds(cfg: dict, model: str, run_id: int) -> dict:
    """Route the §3 seeds into the two receiving fields (RECON §5):

    - sensory_stream.seed  <- env_seed(...)   the shared exogenous stream
    - arena random_seed    <- model_seed(...) every model-PRIVATE generator
    """
    env = cfg["environment"]
    s = seeding.env_seed(DTH_DEG, DIFF_BP, run_id, "sensory")
    m = seeding.model_seed(model, DTH_DEG, DIFF_BP, run_id)
    env["sensory_stream"]["seed"] = int(s)
    for arena in env["arenas"].values():
        if isinstance(arena, dict):
            arena["random_seed"] = int(m)
    return {"env_seed_sensory": int(s), "model_seed": int(m),
            "seed_scheme": seeding.SCHEME}


# ---------------------------------------------------------------------------
# Manifests (§5)
# ---------------------------------------------------------------------------
def build_ra_rows(n_runs: int = N_RUNS) -> list[dict]:
    star = {v: u_star(v) for v in V_GRID}
    rows = []
    for v in V_GRID:
        us, _lam = star[v]
        for u_hat in UHAT_GRID:
            rows.append({
                "cell_id": f"R_v{v:g}_h{u_hat:g}", "sweep": "relative",
                "v": f"{v:g}", "u_hat": f"{u_hat:.6f}",
                "u_star": f"{us:.6f}", "u": f"{u_hat * us:.6f}",
                "diff": f"{DIFF:g}", "n_runs": str(int(n_runs)),
                "seed_scheme": seeding.SCHEME})
    for v in V_GRID:
        us, _lam = star[v]
        for u in U_ABS_GRID:
            rows.append({
                "cell_id": f"U_v{v:g}_u{u:g}", "sweep": "absolute",
                "v": f"{v:g}", "u_hat": f"{u / us:.6f}",
                "u_star": f"{us:.6f}", "u": f"{u:.6f}",
                "diff": f"{DIFF:g}", "n_runs": str(int(n_runs)),
                "seed_scheme": seeding.SCHEME})
    assert len(rows) == len(V_GRID) * (len(UHAT_GRID) + len(U_ABS_GRID)) == 100
    return rows


def build_ddm_rows(n_runs: int = N_RUNS) -> list[dict]:
    return [{"point_id": f"D_ce{ce:g}", "c_e": f"{ce:g}",
             "diff": f"{DIFF:g}", "n_runs": str(int(n_runs)),
             "seed_scheme": seeding.SCHEME}
            for ce in C_E_GRID]


def write_manifest(rows: list[dict], fields: list[str], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def replicate_dir(base_root: Path, row: dict, run_id: int) -> Path:
    """§6 layout. The CSV's own strings key the directories, so the manifest,
    the filesystem and run_meta.json can never disagree on formatting."""
    if "cell_id" in row:            # RA
        return (Path(base_root) / "cells" / row["sweep"] / f"v_{row['v']}" /
                f"u_{row['u']}" / f"replicate_{int(run_id)}")
    return (Path(base_root) / "points" / f"ce_{row['c_e']}" /
            f"replicate_{int(run_id)}")


# ---------------------------------------------------------------------------
# Arrival scoring (ported from scripts/uhat_v_sweep/run_cell.py — RECON §7)
# ---------------------------------------------------------------------------
def target_positions(cfg: dict) -> dict[str, tuple[float, float]]:
    return {f"{name}.s#0": (float(o["position"][0][0]), float(o["position"][0][1]))
            for name, o in cfg["environment"]["objects"].items()}


def segment_circle_fraction(p0, p1, centre, radius):
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


def first_crossing(rows, targets, radius):
    """(tick, tick_fine, target_id) of the first entry into a target's
    termination disc on the logged trajectory, or (None, None, None)."""
    prev = None
    for row in rows:
        tick = int(row["tick"])
        px, py = float(row["pos_x"]), float(row["pos_y"])
        hit, best = None, float("inf")
        for tid, (tx, ty) in targets.items():
            dist = math.hypot(tx - px, ty - py)
            if dist <= radius + 1e-9 and dist < best:
                hit, best = tid, dist
        if hit is not None:
            fine = float(tick)
            if prev is not None:
                s = segment_circle_fraction(prev, (px, py), targets[hit], radius)
                if s is not None:
                    fine = float(prev[2]) + s * (tick - prev[2])
            return tick, fine, hit
        prev = (px, py, tick)
    return None, None, None
