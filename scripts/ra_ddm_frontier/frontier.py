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
# Design grids (§2). 52 + 48 = 100 RA cells in wave 1; 10 DDM Bellman points.
# ---------------------------------------------------------------------------
V_GRID = [0.2, 0.3, 0.4, 0.5]
#: Wave 3 (§2 Set U-v3): kernel extension beyond the frontier-dominant range.
V_GRID_EXT = [0.6, 0.8]
V_GRID_ALL = V_GRID + V_GRID_EXT
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

#: §2b — the halt-at-midpoint motion policy, BOTH DDM families. An agent that
#: reaches the midpoint undecided halts (v = 0) and keeps integrating against
#: the flat threshold z_halt until a bound is crossed or time_limit hits.
#: Wired in src as bellman.terminal = 'halt_sprt' (motor hold + flat plateau);
#: halt_cost_rate = 1.0 is the physical value (halted = zero progress).
#: time_limit is NOT raised for the halt campaign (§13: timeout changes are
#: out of scope): trials still halted at 60 s are censored and reported —
#: `ddm_halt_budget()` sizes that risk per point before submission.
DDM_TERMINAL = "halt_sprt"
DDM_HALT_COST_RATE = 1.0
DDM_VARIANTS = ("bellman", "static")

#: §2b Family 2 — static bounds. b swept log-spaced over
#: [z*_quasistatic(min c_e) — the boundary of the fastest Bellman point, so
#:  the same speed by construction —, b at the accuracy ceiling: Wald
#:  acc(b) = 1/(1+exp(-k b)) = STATIC_ACC_CEILING with k = 2A/c²].
#: The optimum boundaries b*_cost / b*_RR are DERIVED from the swept data in
#: analyze_overlay.py, never assumed (§2b).
STATIC_N_LEVELS = 14
STATIC_ACC_CEILING = 0.995
#: The evidence substep at the 1 s tick (RECON D-11): boundaries below it are
#: discretisation-limited — the same convention as the Bellman c_e ∈ {0.03, 0.1}.
EVIDENCE_SUBSTEP = 0.025        # c * sqrt(dt / n_sub) = 0.1 * sqrt(1/16)

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
DDM_MANIFEST_NAME = "ddm_manifest.csv"           # wave 1/2 (forced_choice) — frozen
DDM_HALT_MANIFEST_NAME = "ddm_manifest_halt.csv"  # §2b rerun (halt_sprt, both families)
RA_FIELDS = ["cell_id", "sweep", "v", "u_hat", "u_star", "u",
             "diff", "n_runs", "seed_scheme"]
#: Wave-1/2 DDM schema, kept only so the frozen manifests on the cluster can
#: still be read (the forced-choice tree is the §9 regression reference).
DDM_FIELDS_V1 = ["point_id", "c_e", "diff", "n_runs", "seed_scheme"]
#: §5 halt-campaign schema: `variant` ∈ {bellman, static}; `bound` = c_e or b.
DDM_FIELDS = ["point_id", "variant", "bound", "diff", "n_runs", "seed_scheme"]


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
    # §2b: BOTH families run under the halt-at-midpoint motion policy. The
    # base config carries the researcher's setting; assert rather than set, so
    # a hand edit back to forced_choice is caught instead of silently inherited.
    if bell.get("terminal") != DDM_TERMINAL:
        raise SystemExit(
            f"{DDM_BASE_CONFIG}: bellman.terminal = {bell.get('terminal')!r}, "
            f"expected {DDM_TERMINAL!r} — the §2b halt-at-midpoint campaign "
            "requires it (RECON D-13)")
    if float(bell.get("halt_cost_rate", 0.0)) != DDM_HALT_COST_RATE:
        raise SystemExit(
            f"{DDM_BASE_CONFIG}: bellman.halt_cost_rate = "
            f"{bell.get('halt_cost_rate')!r}, expected {DDM_HALT_COST_RATE} "
            "(the physical value; RECON D-13)")
    bell["static_bound"] = None             # per point (static family only)
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


def patch_ddm(template: dict, variant: str, bound: float,
              table_cache_dir: str | None) -> dict:
    """§2b: one template, two parameterizations — never a parallel implementation.

    bellman: `bound` is c_e; the boundary is the Bellman solve (collapsing onto
             the z_halt plateau under terminal 'halt_sprt').
    static:  `bound` is the constant boundary height b, installed as a flat
             table through the SAME bellman machinery (`bellman.static_bound`);
             cost_ratio is not consumed for the boundary and is pinned to 0.0
             so any stray read is loud in the diagnostics; no PDE solve, so no
             table cache either.
    """
    cfg = copy.deepcopy(template)
    blk = cfg["environment"]["agents"]["movable_0"]["embodied_pure_ddm"]
    if variant == "bellman":
        blk["cost_ratio"] = float(bound)
        blk["bellman"]["static_bound"] = None
        blk["bellman"]["table_cache_dir"] = (str(table_cache_dir)
                                             if table_cache_dir else None)
    elif variant == "static":
        blk["cost_ratio"] = 0.0            # diagnostic only under static_bound
        blk["bellman"]["static_bound"] = float(bound)
        blk["bellman"]["table_cache_dir"] = None
    else:
        raise SystemExit(f"unknown DDM variant {variant!r} "
                         f"(expected one of {DDM_VARIANTS})")
    return cfg


def apply_seeds(cfg: dict, model: str, run_id: int) -> dict:
    """Route the §3 seeds into the two receiving fields (RECON §5):

    - sensory_stream.seed  <- env_seed(...)   the shared exogenous stream
    - arena random_seed    <- model_seed(...) every model-PRIVATE generator

    `model` is the §3 model string: 'ra', 'ddm-bellman' or 'ddm-static'
    (§6; the wave-1/2 forced-choice rerun used plain 'ddm' — the DDM draws no
    private noise, so the re-key changes nothing physical; RECON D-13).
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


# ---------------------------------------------------------------------------
# Wave 2 (spec v2): Set U-v2 top-up + DDM ceiling points
# ---------------------------------------------------------------------------
#: §11 ceiling verification: ~10x and ~100x the previous maximum c_e. The
#: claim "the RA sub-critical peak exceeds the DDM family" stands only if the
#: RA peak clears the DDM's infinite-patience asymptote with CI separation.
DDM_CEILING_CE = [3000.0, 30000.0]

#: §2 Set U-v2 rule parameters.
TOPUP_ACC_HI = 0.95      # û_hi = last Set-R level (in û order) with acc_all > this
TOPUP_ACC_LO = 0.85      # û_lo = first Set-R level after û_hi with acc_all < this
TOPUP_DU_HAT = 0.025     # cliff-window sampling step in û
TOPUP_ANCHORS = [0.3, 0.5]     # sub-branch anchors, in units of u*
TOPUP_COMMITTED = 1.2          # one committed point, in units of u*
TOPUP_TAIL = [1.75, 2.00, 2.40]  # deep super-critical tail, matched in û
TOPUP_SKIP_REL = 0.05    # skip a target within 5 % of an existing Set-U level


def _round25(u: float) -> float:
    return round(round(u / 0.25) * 0.25, 2)


def build_topup_rows(cells_csv: Path, n_runs: int = N_RUNS) -> tuple[list, dict]:
    """Set U-v2 rows (§2), derived from wave-1 `cells.csv` — inverse-mapping
    the measured cliff, not sampling uniformly. Returns (rows, report)."""
    cells = read_manifest(Path(cells_csv))
    rows, report = [], {}
    for v in V_GRID:
        rel = sorted((c for c in cells
                      if c["sweep"] == "relative" and float(c["v"]) == v),
                     key=lambda c: float(c["u_hat"]))
        if not rel:
            raise SystemExit(f"{cells_csv}: no Set-R rows for v = {v}")
        us = float(rel[0]["u_star"])
        existing = sorted(float(c["u"]) for c in cells
                          if c["sweep"] == "absolute" and float(c["v"]) == v)

        hi_levels = [float(c["u_hat"]) for c in rel
                     if float(c["acc_all"]) > TOPUP_ACC_HI]
        if not hi_levels:
            raise SystemExit(f"v={v}: no Set-R level with acc_all > "
                             f"{TOPUP_ACC_HI} — the cliff rule has no anchor")
        uhat_hi = max(hi_levels)
        uhat_lo = next((float(c["u_hat"]) for c in rel
                        if float(c["u_hat"]) > uhat_hi
                        and float(c["acc_all"]) < TOPUP_ACC_LO), None)
        if uhat_lo is None:
            raise SystemExit(f"v={v}: no Set-R level below {TOPUP_ACC_LO} "
                             f"beyond û = {uhat_hi} — no cliff to sample")

        # (target u, may_sit_near_existing): cliff samples deliberately
        # interleave the integer wave-1 grid, so they are dropped only on an
        # EXACT collision; anchors / committed / tail apply the 5 % skip.
        targets = []
        k = 0
        while uhat_hi + k * TOPUP_DU_HAT <= uhat_lo + 1e-9:      # cliff window
            targets.append(((uhat_hi + k * TOPUP_DU_HAT) * us, True))
            k += 1
        targets += [(a * us, False) for a in TOPUP_ANCHORS]      # sub-branch
        targets.append((TOPUP_COMMITTED * us, False))            # committed
        targets += [(t * us, False) for t in TOPUP_TAIL]         # deep tail

        chosen = []
        for t, is_cliff in targets:
            u = _round25(t)
            taken = existing + chosen
            if is_cliff:
                if any(abs(u - e) < 1e-9 for e in taken):
                    continue
            elif any(abs(u - e) / e <= TOPUP_SKIP_REL
                     for e in taken if e > 0):
                continue
            chosen.append(u)
        chosen = sorted(set(chosen))
        report[v] = {"u_star": us, "uhat_hi": uhat_hi, "uhat_lo": uhat_lo,
                     "existing": existing, "new_u": chosen}
        for u in chosen:
            rows.append({"cell_id": f"U2_v{v:g}_u{u:g}", "sweep": "absolute",
                         "v": f"{v:g}", "u_hat": f"{u / us:.6f}",
                         "u_star": f"{us:.6f}", "u": f"{u:.6f}",
                         "diff": f"{DIFF:g}", "n_runs": str(int(n_runs)),
                         "seed_scheme": seeding.SCHEME})
    return rows, report


def build_ddm_ceiling_rows(n_runs: int = N_RUNS) -> list[dict]:
    return [{"point_id": f"D_ce{ce:g}", "c_e": f"{ce:g}",
             "diff": f"{DIFF:g}", "n_runs": str(int(n_runs)),
             "seed_scheme": seeding.SCHEME}
            for ce in DDM_CEILING_CE]


# ---------------------------------------------------------------------------
# Wave 3 (spec v3): Set U-v3 — output-plane arc-length re-spacing at
# v ∈ V_GRID + full kernel-extension grids at v ∈ V_GRID_EXT.
# ---------------------------------------------------------------------------
W3_PER_BRANCH = 9        # levels per branch at equal arc-length increments
W3_ANCHORS = TOPUP_ANCHORS       # extension-v sub-branch anchors, units of u*
W3_TAIL = TOPUP_TAIL             # deep tails û {1.75, 2.0, 2.4}, matched in û
W3_SKIP_REL = TOPUP_SKIP_REL     # skip within 5 % of an existing level
W3_GAPFILL_FACTOR = 2.0          # fill a realized chord > 2x the budget
W3_MIN_MAP_CELLS = 6   # frontier cells at v needed before its own map is used


def _w3_branch_levels(us, ts, accs, t_range, acc_range, n_levels):
    """Levels at equal arc-length increments in normalized (t, acc) plane
    coordinates along the monotone (PCHIP) interpolants of one branch."""
    import numpy as np
    from scipy.interpolate import PchipInterpolator
    if len(us) < 2:
        return []
    t_i = PchipInterpolator(us, ts)
    a_i = PchipInterpolator(us, accs)
    grid = np.linspace(us[0], us[-1], 512)
    dt = np.diff(t_i(grid)) / t_range
    da = np.diff(a_i(grid)) / acc_range
    s = np.concatenate([[0.0], np.cumsum(np.hypot(dt, da))])
    if s[-1] <= 0.0:
        return []
    targets = np.linspace(0.0, s[-1], n_levels)
    return [float(np.interp(t, s, grid)) for t in targets], float(s[-1])


def _w3_v_levels(points, report):
    """The §2 U-v3 rule for one v: `points` = sorted measured (u, t, acc).

    Branch split at the measured accuracy peak; equal arc-length levels per
    branch; then a gap-fill pass — realized chords between adjacent measured
    points longer than W3_GAPFILL_FACTOR x the arc-length budget get equally
    spaced u midpoints (the cliff stays steep: some chord length there is
    irreducible physics, and the fill caps at 3 points per chord)."""
    import numpy as np
    us = np.array([p[0] for p in points], float)
    ts = np.array([p[1] for p in points], float)
    accs = np.array([p[2] for p in points], float)
    t_range = max(float(ts.max() - ts.min()), 1e-9)
    acc_range = max(float(accs.max() - accs.min()), 1e-9)
    i_peak = int(np.argmax(accs))
    report["u_peak"] = float(us[i_peak])
    levels, lengths = [], []
    for lo, hi in ((0, i_peak + 1), (i_peak, len(us))):
        got = _w3_branch_levels(us[lo:hi], ts[lo:hi], accs[lo:hi],
                                t_range, acc_range, W3_PER_BRANCH)
        if got:
            branch_levels, length = got
            levels += branch_levels
            lengths.append(length)
    budget = (sum(lengths) / (len(lengths) * W3_PER_BRANCH)
              if lengths else float("inf"))
    report["arc_length_budget"] = budget
    fills = []
    for j in range(len(us) - 1):
        chord = math.hypot((ts[j + 1] - ts[j]) / t_range,
                           (accs[j + 1] - accs[j]) / acc_range)
        if chord > W3_GAPFILL_FACTOR * budget:
            n_fill = min(int(math.ceil(chord / budget)) - 1, 3)
            fills += list(np.linspace(us[j], us[j + 1], n_fill + 2)[1:-1])
    report["gap_fills"] = [float(f"{u:.4g}") for u in fills]
    return levels + fills


def build_wave3_rows(cells_csv: Path, factorial_csv: Path | None = None,
                     n_runs: int = N_RUNS) -> tuple[list, dict]:
    """Set U-v3 rows (§2), derived from measured data. Returns (rows, report).

    v ∈ V_GRID: re-spacing top-up — the (t, acc) map comes from ALL existing
    frontier cells at that v (both sweeps pooled on the u axis, waves 1+2+…).
    v ∈ V_GRID_EXT: kernel extension — the map comes from the factorial's
    n = 400 (û, v) sweep at that v (coarse but sufficient for placement;
    t there is the commit tick ≡ seconds at 1 tick/s) until the frontier tree
    itself holds ≥ W3_MIN_MAP_CELLS cells at that v, after which re-running
    the generator performs the gap-fill pass on the measured map. Extension
    grids additionally get the {0.3, 0.5}·u* sub-branch anchors (the factorial
    never sampled below û = 0.5), the û ∈ {1.75, 2.0, 2.4} deep tails and a
    u = 0 control — the §12 u = 0 replicate gate becomes six-way."""
    cells = read_manifest(Path(cells_csv))
    factorial = (read_manifest(Path(factorial_csv))
                 if factorial_csv is not None else [])
    rows, report = [], {}
    for v in V_GRID_ALL:
        rep = {"v": v}
        us_v, _lam = u_star(v)
        rep["u_star"] = us_v
        mine = [c for c in cells if abs(float(c["v"]) - v) < 1e-9
                and c.get("median_arrival_s") not in (None, "", "nan")]
        existing = sorted({float(c["u"]) for c in mine})
        extension = v in V_GRID_EXT
        targets = []

        if not extension or len(mine) >= W3_MIN_MAP_CELLS:
            rep["map_source"] = "frontier"
            pts = {}
            for c in mine:
                pts.setdefault(float(c["u"]), []).append(
                    (float(c["median_arrival_s"]), float(c["acc_all"])))
            points = sorted((u, sum(t for t, _a in g) / len(g),
                             sum(a for _t, a in g) / len(g))
                            for u, g in pts.items())
            if len(points) < 4:
                raise SystemExit(f"v={v}: only {len(points)} measured cells "
                                 "in the frontier map — too few for U-v3")
            targets += [(u, False) for u in _w3_v_levels(points, rep)]
        else:
            rep["map_source"] = "factorial"
            fac = [c for c in factorial if abs(float(c["v"]) - v) < 1e-9]
            if len(fac) < 4:
                raise SystemExit(
                    f"v={v}: kernel extension needs the factorial map — pass "
                    "--factorial <uhat_v_sweep cells.csv> (or run wave 3 after "
                    "frontier data exists at this v)")
            fac_us = float(fac[0]["u_star"])
            rep["u_star_factorial"] = fac_us
            rep["u_star_rel_err"] = abs(us_v - fac_us) / fac_us
            if rep["u_star_rel_err"] > ANCHOR_TOL:
                raise SystemExit(
                    f"v={v}: u*(v) = {us_v:.4f} here vs {fac_us:.4f} in the "
                    "factorial — kernel builder drift; halt for human review")
            points = sorted((float(c["u"]),
                             float(c["t_commit_fine_median"]),
                             float(c["acc_all"])) for c in fac)
            targets += [(u, False) for u in _w3_v_levels(points, rep)]

        if extension:
            targets += [(a * us_v, False) for a in W3_ANCHORS]
            targets += [(t * us_v, False) for t in W3_TAIL]

        chosen = []
        for t, _flag in targets:
            u = _round25(t)
            if u < 0.25:
                continue
            taken = existing + chosen
            if any(abs(u - e) / e <= W3_SKIP_REL for e in taken if e > 0):
                continue
            chosen.append(u)
        if extension and 0.0 not in existing:
            chosen.append(0.0)              # the u = 0 control (§12 gate)
        chosen = sorted(set(chosen))
        rep["existing"] = existing
        rep["new_u"] = chosen
        report[v] = rep
        for u in chosen:
            rows.append({"cell_id": f"U3_v{v:g}_u{u:g}", "sweep": "absolute",
                         "v": f"{v:g}", "u_hat": f"{u / us_v:.6f}",
                         "u_star": f"{us_v:.6f}", "u": f"{u:.6f}",
                         "diff": f"{DIFF:g}", "n_runs": str(int(n_runs)),
                         "seed_scheme": seeding.SCHEME})
    return rows, report


# ---------------------------------------------------------------------------
# §2b — the DDM halt campaign: Bellman (collapsing + floor) and static bounds
# ---------------------------------------------------------------------------
def _onset_c_tau() -> float:
    """c_tau at evidence onset, from the SAME code path the model uses."""
    from models.ddm_systems import DriftDiffusionSystem
    return float(DriftDiffusionSystem.c_tau_linearised(
        math.radians(DTH_DEG), predecision_motion="midpoint"))


def _wald_k() -> float:
    """k = 2A/c² of the evidence channel (A = ΔQ, c = 2ΔQ) — here 10 exactly."""
    return 2.0 * QUALITY_DELTA / NOISE_SCALE_C ** 2


def static_b_grid() -> list[float]:
    """§2b: ~14 log-spaced boundary heights.

    Lower end: the quasi-static z* of the fastest Bellman point (min c_e at the
    onset geometry) — same boundary, same speed by construction. Upper end: the
    b whose Wald accuracy 1/(1+e^{-kb}) reaches STATIC_ACC_CEILING. Both ends
    are derived from the simulator's own solvers/constants, never typed."""
    import numpy as np
    from models.bellman_boundary import myopic_z
    b_lo = myopic_z(QUALITY_DELTA, NOISE_SCALE_C, min(C_E_GRID), _onset_c_tau())
    b_hi = math.log(STATIC_ACC_CEILING / (1.0 - STATIC_ACC_CEILING)) / _wald_k()
    if not (0.0 < b_lo < b_hi):
        raise SystemExit(f"static b grid derivation broke: [{b_lo}, {b_hi}]")
    grid = np.geomspace(b_lo, b_hi, STATIC_N_LEVELS)
    return [float(f"{b:.4g}") for b in grid]


def build_ddm_halt_rows(n_runs: int = N_RUNS) -> list[dict]:
    """The §2b campaign manifest: Bellman grid verbatim + the §11 ceiling
    extremes + the static-b sweep, all variant-tagged (schema DDM_FIELDS)."""
    rows = [{"point_id": f"D_ce{ce:g}", "variant": "bellman",
             "bound": f"{ce:g}", "diff": f"{DIFF:g}",
             "n_runs": str(int(n_runs)), "seed_scheme": seeding.SCHEME}
            for ce in C_E_GRID + DDM_CEILING_CE]
    rows += [{"point_id": f"S_b{b:g}", "variant": "static",
              "bound": f"{b:g}", "diff": f"{DIFF:g}",
              "n_runs": str(int(n_runs)), "seed_scheme": seeding.SCHEME}
             for b in static_b_grid()]
    return rows


def ddm_halt_budget() -> list[dict]:
    """Per halt-campaign point: z_halt, the mean extra deliberation D(z_halt),
    and the margin against time_limit. A point whose mean total time plus
    3·D(z_halt) exceeds the budget will show visible censoring — reported
    here BEFORE submission (time_limit itself is out of scope, §13)."""
    from models.bellman_boundary import halt_exit_potential, solve_z_halt
    k = _wald_k()
    A = QUALITY_DELTA
    t_arrive = R0 / LINEAR_VELOCITY
    out = []
    for row in build_ddm_halt_rows(1):
        b = float(row["bound"])
        z_halt = (solve_z_halt(k, A, b, DDM_HALT_COST_RATE)
                  if row["variant"] == "bellman" else b)
        d_mean = float(halt_exit_potential(z_halt, k, A))
        mean_total = t_arrive + d_mean
        out.append({
            "point_id": row["point_id"], "variant": row["variant"],
            "bound": b, "z_halt": z_halt, "halt_mean_exit_s": d_mean,
            "mean_total_s": mean_total,
            "time_limit_s": DDM_TIME_LIMIT,
            "censor_risk": mean_total + 3.0 * d_mean > DDM_TIME_LIMIT,
            "discretisation_limited": z_halt < EVIDENCE_SUBSTEP,
        })
    return out


def ddm_ideal_ceiling() -> float:
    """The fixed-bound infinite-patience asymptote (§11): an observer that
    integrates the full deliberation horizon T = r0/v and reports sign(x) has
    accuracy Phi((A/c) * sqrt(T)) — no boundary policy can beat it."""
    from statistics import NormalDist
    return NormalDist().cdf((QUALITY_DELTA / NOISE_SCALE_C)
                            * math.sqrt(R0 / LINEAR_VELOCITY))


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
    the filesystem and run_meta.json can never disagree on formatting.

    DDM: wave-1/2 rows (schema v1, `c_e` column) keep the flat `points/ce_*`
    layout of the frozen forced-choice tree; halt-campaign rows (`variant` +
    `bound`) get `points/<variant>/{ce_|b_}<bound>` in their own fresh tree."""
    if "cell_id" in row:            # RA
        return (Path(base_root) / "cells" / row["sweep"] / f"v_{row['v']}" /
                f"u_{row['u']}" / f"replicate_{int(run_id)}")
    if "variant" in row:            # DDM halt campaign (§2b)
        prefix = "ce_" if row["variant"] == "bellman" else "b_"
        return (Path(base_root) / "points" / row["variant"] /
                f"{prefix}{row['bound']}" / f"replicate_{int(run_id)}")
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
