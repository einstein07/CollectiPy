# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2026 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Shared machinery for the QD-sweep-at-fixed-noise campaign (both arms).

Implements `qd-sweep-fixed-noise-experiment.md`: §2 (the parameter block —
single source of truth), §3 (Arm A grids), §4 (the frozen-DDM controllers),
§6 (seed routing) and §7 (manifests, layout, patchers WITH the blocking
assertions). Infrastructure is inherited from the frontier campaign
(`scripts/ra_ddm_frontier/`) by import — templates, arrival scoring and the
seed scheme are the frontier's own code, so the two campaigns cannot drift
apart silently. See RECON.md for every decision and departure.

THE POINT OF THIS CAMPAIGN (and of the assertions below): the halted frontier
campaign swept δ_Q by editing `frontier.DIFF`, and because
`frontier.WHITE_RATE = √2·ΔQ` was DERIVED from it, the noise silently scaled
with the signal and SNR never moved (c = 2ΔQ at every δ_Q). Here WHITE_RATE
is a pinned literal, never computed from δ_Q, and every emitted config is
asserted to carry it exactly — the coupling is structurally unable to return.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_FRONTIER = _ROOT / "scripts" / "ra_ddm_frontier"
for _p in (str(_HERE), str(_FRONTIER), str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import frontier   # noqa: E402  — the frontier campaign's machinery, inherited
import seeding    # noqa: E402  — frontier-v1, verbatim (single seed universe)

# ---------------------------------------------------------------------------
# §2 — parameter block, single source of truth. Values are LITERALS where the
# spec pins them; derived quantities are derived from S0 and the actual/design
# δ_Q only — NEVER from any noise field, and no noise field from them.
# ---------------------------------------------------------------------------
DTH_DEG = 60
DIFF_BP = [50, 100, 200]            # δ_Q in basis points; THE tunable knob
#: PINNED literal (the §2 hard constraint). NOT √2·ΔQ of any condition — at
#: δ_Q = 1 % the two coincide numerically, which is exactly how the coupling
#: hid; the assertions below check the literal, not the coincidence.
WHITE_RATE = 0.07071068
#: The evidence-channel noise scale the DDM believes and experiences:
#: c = √2·white_rate. Fixed for every condition and both models.
NOISE_SCALE_C = 0.1
S0 = 5.0                            # static_0 strength; static_1 = S0·(1−diff)
U_GRID = [0.0] + [round(2.0 + 0.5 * i, 1) for i in range(67)]   # 68 levels
V_GRID = [round(0.1 * i, 1) for i in range(1, 11)]              # 10 kernels
N_RUNS_RA = 1000
N_RUNS_DDM = 1000
RA_SURFACE_DIFFS = list(DIFF_BP)    # subset this to trim RA volume
FROZEN_CE = [3, 20, 300]            # bold / balanced / patient Bellman
#: Timeouts inherited from the halted campaign (recon slot in §2): the RA in
#: its archived frame (1000 ticks ≡ s at 1 tick/s), the DDM at 60 s — both
#: verified against the halted campaign's replicate configs (RECON R-0).
RA_TIME_LIMIT = frontier.RA_TIME_LIMIT          # 1000
DDM_TIME_LIMIT = frontier.DDM_TIME_LIMIT        # 60
DESIGN_100 = 100                    # the halted campaign's own condition

#: §4 — the sweep-derived static optima of the halted campaign's 1 % b-sweep
#: (its static_bstar.json; grid levels of its own 14-level b grid). The
#: frozen static-cost/static-rr controllers at design = 100 bp ARE these two
#: points — that is what makes gate 2's "reproduce within CIs" well-posed —
#: and generate_manifest re-derives them from the halted campaign's swept
#: trials with the same functional before freezing anything (blocking).
EXPECTED_BSTAR_100 = {"cost": 0.004189, "rr": 0.1579}
COST_TIME_RATE = 0.1                # the §2b cost functional's time rate (= c)

DDM_VARIANTS = ("bellman", "static-cost", "static-rr")

if WHITE_RATE != 0.07071068:        # the §2 patcher assertion, at import
    raise SystemExit("WHITE_RATE drifted from the pinned 0.07071068")
if abs(math.sqrt(2.0) * WHITE_RATE - NOISE_SCALE_C) > 1e-7:
    raise SystemExit("NOISE_SCALE_C must equal sqrt(2)*WHITE_RATE = 0.1")
assert len(U_GRID) == 68 and U_GRID[1] == 2.0 and U_GRID[-1] == 35.0
assert len(V_GRID) == 10 and V_GRID[0] == 0.1 and V_GRID[-1] == 1.0


def drift_A(diff_bp: int) -> float:
    """Signal per condition: A(δ) = S0 · δ_Q (the strength gap)."""
    return round(S0 * int(diff_bp) / 10000.0, 8)


def strength_worse(diff_bp: int) -> float:
    return round(S0 * (1.0 - int(diff_bp) / 10000.0), 8)


def wald_k(diff_bp: int) -> float:
    """k = 2A/c² of the evidence channel at the given δ_Q (5 / 10 / 20)."""
    return round(2.0 * drift_A(diff_bp) / NOISE_SCALE_C ** 2, 9)


# Design-table sanity (§2): at fixed c the SNR now genuinely scales with δ_Q.
assert [drift_A(b) for b in DIFF_BP] == [0.025, 0.05, 0.1]
assert [wald_k(b) for b in DIFF_BP] == [5.0, 10.0, 20.0]

# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
RA_TEMPLATE = _ROOT / "config" / "qd_sweep_ra_template.json"
DDM_TEMPLATE = _ROOT / "config" / "qd_sweep_ddm_template.json"
DEFAULT_OUT = _ROOT / "results" / "qd_sweep_fixed_noise"
#: The halted campaign's 1 % DDM tree (gate 2's reference and the source the
#: static b* optima are re-derived from).
HALTED_DDM_TRIALS = (_ROOT.parent / "seoul-data" / "beta-1"
                     / "ra_ddm_frontier_ddm_halt" / "ra_ddm_frontier_ddm_halt_1.0"
                     / "ddm_trials.parquet")
#: The halted campaign's 1 % RA cells (gate 4's continuity reference).
HALTED_RA_CELLS = (_ROOT.parent / "seoul-data" / "beta-1"
                   / "ra_ddm_frontier_slices" / "ra_ddm_frontier_slices_1.0"
                   / "cells.csv")

RA_MANIFEST_NAME = "ra_manifest.csv"
DDM_MANIFEST_NAME = "ddm_manifest.csv"
RA_FIELDS = ["cell_id", "v", "u", "actual_bp", "n_runs", "seed_scheme"]
DDM_FIELDS = ["point_id", "variant", "design_bp", "actual_bp", "bound_param",
              "n_runs", "seed_scheme"]

# Re-exported frontier machinery (single implementation, no copies).
git_sha = frontier.git_sha
config_hash = frontier.config_hash
read_manifest = frontier.read_manifest
write_manifest = frontier.write_manifest
target_positions = frontier.target_positions
first_crossing = frontier.first_crossing
CORRECT_TARGET_ID = frontier.CORRECT_TARGET_ID
EVIDENCE_SUBSTEP = frontier.EVIDENCE_SUBSTEP


# ---------------------------------------------------------------------------
# Templates (§7): the frontier's own builders, plus this campaign's two
# deltas — an explicit `A_expected` slot (the frozen-controller belief knob)
# on the DDM side, and nothing at all on the RA side. Strengths stay at the
# builders' 1 % values in the TEMPLATE; the patcher writes both strengths per
# actual condition (asserted below).
# ---------------------------------------------------------------------------
def build_ra_template() -> dict:
    return frontier.build_ra_template()


def build_ddm_template() -> dict:
    data = frontier.build_ddm_template()
    blk = data["environment"]["agents"]["movable_0"]["embodied_pure_ddm"]
    # The misspecification knob (§4): drift_knowledge 'known_magnitude' +
    # A_source 'ensemble' + explicit A_expected makes the agent's believed |A|
    # a config constant — the Bellman solve, z_halt and the halt guard all use
    # it, while the percepts carry the ACTUAL strengths. Left null in the
    # template; the patcher writes A_design per point.
    if blk.get("drift_knowledge") != "known_magnitude":
        raise SystemExit("DDM template drift_knowledge must be known_magnitude")
    if blk.get("A_source") != "ensemble":
        raise SystemExit("DDM template A_source must be ensemble")
    blk["A_expected"] = None
    return data


def write_templates() -> None:
    for path, template in ((RA_TEMPLATE, build_ra_template()),
                           (DDM_TEMPLATE, build_ddm_template())):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(template, fh, indent=2)
            fh.write("\n")


def load_template(model: str) -> dict:
    path = RA_TEMPLATE if model == "ra" else DDM_TEMPLATE
    if not path.is_file():
        write_templates()
    return frontier._load_json(path)


# ---------------------------------------------------------------------------
# §4 — the frozen static controllers. b*(design = 100 bp) are the halted
# campaign's own sweep-derived optima, RE-DERIVED here from its trials with
# the identical cost/RR functionals (never typed into a config by hand);
# other designs get the Wald log-odds transport b·k100/k_design — the unique
# analytic map that (i) preserves the controller's believed commitment
# log-odds a = k_design·b (the Wald invariant that sets accuracy) and
# (ii) reproduces the sweep-derived values exactly at design = 100.
# ---------------------------------------------------------------------------
def derive_bstar_100(halted_parquet: Path = HALTED_DDM_TRIALS) -> dict:
    """Re-run the halted campaign's §2b optimum derivation on its own swept
    static trials. BLOCKING cross-check: the argmin/argmax must land on the
    expected grid levels 0.004189 / 0.1579 — disagreement means the wrong
    functional, the wrong file, or a solve bug (§4). Returns
    {"cost": b, "rr": b, "table": {...}}."""
    import pandas as pd
    df = pd.read_parquet(halted_parquet)
    static = df[df["variant"] == "static"]
    if static.empty:
        raise SystemExit(f"{halted_parquet}: no static-variant trials — "
                         "wrong file for the b* derivation")
    cost, rr = {}, {}
    for b, g in static.groupby(static["bound"].astype(float)):
        err = 1.0 - float(g["correct"].mean())
        dec = g["decided"].astype(bool)
        n_cens = int((~dec).sum())
        # Censored trials charged the full budget, exactly as the halted
        # campaign's analyze_overlay.static_bstar did.
        mean_rt = (g.loc[dec, "rt"].astype(float).sum()
                   + n_cens * DDM_TIME_LIMIT) / len(g)
        mean_arr = (g.loc[dec, "t_arrival_s"].astype(float).sum()
                    + n_cens * DDM_TIME_LIMIT) / len(g)
        cost[float(b)] = err + COST_TIME_RATE * mean_rt
        rr[float(b)] = (1.0 - err) / mean_arr if mean_arr > 0 else float("nan")
    b_cost = min(cost, key=cost.get)
    b_rr = max(rr, key=rr.get)
    for name, got, want in (("b*_cost", b_cost, EXPECTED_BSTAR_100["cost"]),
                            ("b*_rr", b_rr, EXPECTED_BSTAR_100["rr"])):
        if abs(got - want) > 1e-9:
            raise SystemExit(
                f"§4 cross-check FAILED: re-derived {name} = {got:g} from "
                f"{halted_parquet}, expected the sweep-derived {want:g} — "
                "solve bug or wrong reference; HALT (do not freeze).")
    return {"cost": b_cost, "rr": b_rr,
            "table": {"cost": {f"{b:g}": c for b, c in sorted(cost.items())},
                      "rr": {f"{b:g}": r for b, r in sorted(rr.items())}}}


def frozen_static_bound(kind: str, design_bp: int, bstar100: dict) -> float:
    """Wald log-odds transport: preserve a = k·b of the design belief."""
    b100 = float(bstar100[kind])
    b = b100 * wald_k(DESIGN_100) / wald_k(design_bp)
    return float(f"{b:.6g}")


def wald_analytic_bstar(design_bp: int) -> dict:
    """The constant-drift Wald anchors at k_design (§9 overlay + reporting):
    ER(b) = 1/(1+e^{kb}), DT(b) = (b/A)·tanh(kb/2). NOT the frozen values —
    the embodied task's empirical optimum is what gets frozen (§4)."""
    import numpy as np
    k, A = wald_k(design_bp), drift_A(design_bp)
    t_travel = frontier.R0 / frontier.LINEAR_VELOCITY
    dense = np.geomspace(1e-4, 2.0, 6000)
    er = 1.0 / (1.0 + np.exp(k * dense))
    dt = dense / A * np.tanh(0.5 * k * dense)
    return {"b_cost_wald": float(dense[int(np.argmin(er + COST_TIME_RATE * dt))]),
            "b_rr_wald": float(dense[int(np.argmax((1.0 - er) / (t_travel + dt)))])}


# ---------------------------------------------------------------------------
# Manifests (§7)
# ---------------------------------------------------------------------------
def build_ra_rows(n_runs: int = N_RUNS_RA,
                  diffs: list[int] | None = None) -> list[dict]:
    """Arm A: U_GRID × V_GRID per actual δ_Q. Absolute u only — no û, no
    u*(v), anywhere (standing convention)."""
    rows = []
    for bp in (diffs if diffs is not None else RA_SURFACE_DIFFS):
        for v in V_GRID:
            for u in U_GRID:
                rows.append({
                    "cell_id": f"a{int(bp)}_v{v:g}_u{u:g}",
                    "v": f"{v:g}", "u": f"{u:g}", "actual_bp": str(int(bp)),
                    "n_runs": str(int(n_runs)),
                    "seed_scheme": seeding.SCHEME})
    return rows


def build_ddm_rows(bstar100: dict, n_runs: int = N_RUNS_DDM) -> list[dict]:
    """Arm B: (3 Bellman c_e + 2 static) × 3 design × 3 actual = 45 points.
    `bound_param` is c_e for bellman rows and the RESOLVED frozen b for the
    static rows; controllers are never re-tuned between conditions."""
    rows = []
    for design in DIFF_BP:
        bounds = ([("bellman", f"{ce:g}") for ce in FROZEN_CE]
                  + [("static-cost",
                      f"{frozen_static_bound('cost', design, bstar100):g}"),
                     ("static-rr",
                      f"{frozen_static_bound('rr', design, bstar100):g}")])
        for actual in DIFF_BP:
            for variant, bound in bounds:
                tag = (f"ce{bound}" if variant == "bellman"
                       else variant.split("-")[1])
                rows.append({
                    "point_id": f"d{design}_a{actual}_{tag}",
                    "variant": variant, "design_bp": str(int(design)),
                    "actual_bp": str(int(actual)), "bound_param": bound,
                    "n_runs": str(int(n_runs)),
                    "seed_scheme": seeding.SCHEME})
    assert len(rows) == len(DIFF_BP) ** 2 * (len(FROZEN_CE) + 2) == 45
    return rows


def replicate_dir(base_root: Path, row: dict, run_id: int) -> Path:
    """§7 layout, keyed by the manifest's own strings:
        <root>/ra/actual_<bp>/v_<v>/u_<u>/replicate_<id>
        <root>/ddm/actual_<bp>/design_<bp>/<variant>_<param>/replicate_<id>
    """
    if "cell_id" in row:
        return (Path(base_root) / "ra" / f"actual_{row['actual_bp']}" /
                f"v_{row['v']}" / f"u_{row['u']}" / f"replicate_{int(run_id)}")
    return (Path(base_root) / "ddm" / f"actual_{row['actual_bp']}" /
            f"design_{row['design_bp']}" /
            f"{row['variant']}_{row['bound_param']}" /
            f"replicate_{int(run_id)}")


# ---------------------------------------------------------------------------
# Patchers (§7) — the ONLY per-cell / per-point edits, plus the §2 blocking
# assertions on every emitted config.
# ---------------------------------------------------------------------------
def _patch_strengths(cfg: dict, actual_bp: int) -> None:
    env = cfg["environment"]
    env["objects"]["static_0"]["strength"] = [S0]
    env["objects"]["static_1"]["strength"] = [strength_worse(actual_bp)]


def patch_ra(template: dict, u: float, v: float, actual_bp: int) -> dict:
    cfg = copy.deepcopy(template)
    mf = cfg["environment"]["agents"]["movable_0"]["mean_field_model"]
    mf["u"] = float(u)
    mf["v"] = float(v)
    _patch_strengths(cfg, actual_bp)
    assert_config(cfg, "ra", actual_bp)
    return cfg


def patch_ddm(template: dict, variant: str, design_bp: int, actual_bp: int,
              bound_param: float, table_cache_dir: str | None) -> dict:
    """One template, three frozen-controller parameterizations (§4):

    bellman:      cost_ratio = c_e; boundary = the Bellman solve at
                  (A_design, c = 0.1) via A_expected (cache keys include A,
                  so each design's tables are distinct automatically).
    static-cost / static-rr: bellman.static_bound = the frozen b — the flat
                  table through the same machinery; cost_ratio pinned 0.0
                  (never read by the boundary). A_expected still carries the
                  design belief (it parameterizes the halt guard and the log
                  record; the flat boundary itself never reads it).
    """
    cfg = copy.deepcopy(template)
    blk = cfg["environment"]["agents"]["movable_0"]["embodied_pure_ddm"]
    if variant == "bellman":
        blk["cost_ratio"] = float(bound_param)
        blk["bellman"]["static_bound"] = None
        blk["bellman"]["table_cache_dir"] = (str(table_cache_dir)
                                             if table_cache_dir else None)
    elif variant in ("static-cost", "static-rr"):
        blk["cost_ratio"] = 0.0
        blk["bellman"]["static_bound"] = float(bound_param)
        blk["bellman"]["table_cache_dir"] = None
    else:
        raise SystemExit(f"unknown DDM variant {variant!r} "
                         f"(expected one of {DDM_VARIANTS})")
    blk["A_expected"] = drift_A(design_bp)
    _patch_strengths(cfg, actual_bp)
    assert_config(cfg, "ddm", actual_bp, design_bp)
    return cfg


def assert_config(cfg: dict, model: str, actual_bp: int,
                  design_bp: int | None = None) -> None:
    """§2's patcher assertions, on EVERY emitted config (blocking):

    - `white_rate == 0.07071068` exactly — the pinned literal, not any
      δ_Q-derived quantity;
    - no field derived from δ_Q may touch the noise: the DDM's own noise
      generator is off (eta_rate [0,0] — the stream is the only noise source
      and its scale is the pinned rate), the RA's private percept noise is
      off (sigma_s 0);
    - both strengths follow (S0, actual δ_Q) exactly;
    - the DDM belief block carries the DESIGN drift, terminal 'halt_sprt',
      the physical halt cost, and the inherited timeout; the RA carries its
      inherited timeout.
    """
    env = cfg["environment"]
    wr = env["sensory_stream"]["white_rate"]
    if wr != WHITE_RATE:
        raise SystemExit(f"[{model}] sensory_stream.white_rate = {wr!r} != "
                         f"pinned {WHITE_RATE} — the δ_Q→noise coupling is "
                         "back; HALT")
    if env["sensory_stream"].get("mode") != "shared" \
            or float(env["sensory_stream"].get("frozen_sd", -1)) != 0.0:
        raise SystemExit(f"[{model}] sensory_stream mode/frozen_sd drifted")
    s0 = [float(x) for x in env["objects"]["static_0"]["strength"]]
    s1 = [float(x) for x in env["objects"]["static_1"]["strength"]]
    if s0 != [S0] or s1 != [strength_worse(actual_bp)]:
        raise SystemExit(f"[{model}] strengths {s0}/{s1} do not encode "
                         f"actual δ_Q = {actual_bp} bp "
                         f"(expected [{S0}]/[{strength_worse(actual_bp)}])")
    ag = env["agents"]["movable_0"]
    if model == "ra":
        if int(env["time_limit"]) != RA_TIME_LIMIT:
            raise SystemExit(f"RA time_limit != inherited {RA_TIME_LIMIT}")
        if float(ag["mean_field_model"]["sigma_s"]) != 0.0:
            raise SystemExit("RA sigma_s must be 0 (stream is the only "
                             "exogenous noise)")
        if ag["mean_field_model"].get("g_adapt") != 0.0:
            raise SystemExit("RA g_adapt must be 0 (SFA out of scope)")
    else:
        if int(env["time_limit"]) != DDM_TIME_LIMIT:
            raise SystemExit(f"DDM time_limit != inherited {DDM_TIME_LIMIT}")
        blk = ag["embodied_pure_ddm"]
        if [float(x) for x in blk["eta_rate"]] != [0.0, 0.0]:
            raise SystemExit("DDM eta_rate must be [0, 0] under the shared "
                             "stream — any other value is a second noise path")
        if design_bp is None or blk.get("A_expected") != drift_A(design_bp):
            raise SystemExit(f"DDM A_expected = {blk.get('A_expected')!r} "
                             f"does not encode design δ_Q = {design_bp} bp")
        bell = blk["bellman"]
        if bell.get("terminal") != "halt_sprt":
            raise SystemExit("DDM bellman.terminal must be 'halt_sprt' "
                             "(inherited motion policy)")
        if float(bell.get("halt_cost_rate", 0.0)) != 1.0:
            raise SystemExit("DDM bellman.halt_cost_rate must be 1.0")
        if int(bell.get("N_t", 0)) != frontier.BELLMAN_N_T:
            raise SystemExit(f"DDM bellman.N_t != {frontier.BELLMAN_N_T}")


def assert_noise_invariant_across_actuals(model: str = "ra") -> None:
    """The structural §2 guarantee, checked end-to-end: patch one config per
    actual δ_Q and demand the ENTIRE sensory_stream block (the seed excluded)
    is byte-identical across them — nothing derived from δ_Q reaches the
    noise. Run at manifest generation and in the smoke."""
    blocks = []
    for bp in DIFF_BP:
        if model == "ra":
            cfg = patch_ra(load_template("ra"), 6.0, 0.5, bp)
        else:
            cfg = patch_ddm(load_template("ddm"), "bellman", DESIGN_100, bp,
                            20.0, None)
        blk = dict(cfg["environment"]["sensory_stream"])
        blk.pop("seed", None)
        blocks.append(json.dumps(blk, sort_keys=True))
    if len(set(blocks)) != 1:
        raise SystemExit(f"[{model}] sensory_stream differs across actual "
                         f"δ_Q: {blocks} — the coupling is back; HALT")


# ---------------------------------------------------------------------------
# Seeds (§6): env streams keyed by (DTH_DEG, ACTUAL diff_bp, run_id) — the
# design δ_Q is a controller property and never enters any seed. Model tags
# 'ra' / 'ddm-bellman' / 'ddm-static' (both static variants are the same
# model). At actual = 100 bp this reproduces the halted campaign's seed
# universe exactly, which is what makes gates 2 and 4 seed-paired.
# ---------------------------------------------------------------------------
def model_key(row: dict) -> str:
    if "cell_id" in row:
        return "ra"
    return "ddm-bellman" if row["variant"] == "bellman" else "ddm-static"


def apply_seeds(cfg: dict, model: str, actual_bp: int, run_id: int) -> dict:
    env = cfg["environment"]
    s = seeding.env_seed(DTH_DEG, int(actual_bp), run_id, "sensory")
    m = seeding.model_seed(model, DTH_DEG, int(actual_bp), run_id)
    env["sensory_stream"]["seed"] = int(s)
    for arena in env["arenas"].values():
        if isinstance(arena, dict):
            arena["random_seed"] = int(m)
    return {"env_seed_sensory": int(s), "model_seed": int(m),
            "seed_scheme": seeding.SCHEME}


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def design_table() -> list[dict]:
    return [{"diff_bp": bp, "A": drift_A(bp),
             "A_over_c": round(drift_A(bp) / NOISE_SCALE_C, 6),
             "k": round(wald_k(bp), 6)} for bp in DIFF_BP]


def ddm_budget(rows: list[dict]) -> list[dict]:
    """Per Arm-B point: the controller's believed z_halt and mean halt exit
    D(z_halt) (design parameters — the guard uses them), plus the ACTUAL mean
    exit at each actual δ_Q, since censoring risk at 60 s is set by the
    actual dynamics: a bold belief met by a weak world deliberates far longer
    than designed. Reported before submission; time_limit is inherited."""
    from models.bellman_boundary import halt_exit_potential, solve_z_halt
    t_arrive = frontier.R0 / frontier.LINEAR_VELOCITY
    out = []
    for row in rows:
        design, actual = int(row["design_bp"]), int(row["actual_bp"])
        kd, ad = wald_k(design), drift_A(design)
        ka, aa = wald_k(actual), drift_A(actual)
        b = float(row["bound_param"])
        z_halt = (solve_z_halt(kd, ad, b, 1.0)
                  if row["variant"] == "bellman" else b)
        d_design = float(halt_exit_potential(z_halt, kd, ad))
        d_actual = float(halt_exit_potential(z_halt, ka, aa))
        out.append({
            "point_id": row["point_id"], "variant": row["variant"],
            "design_bp": design, "actual_bp": actual, "bound_param": b,
            "z_halt": z_halt,
            "halt_mean_exit_design_s": d_design,
            "halt_mean_exit_actual_s": d_actual,
            "mean_total_actual_s": t_arrive + d_actual,
            "time_limit_s": DDM_TIME_LIMIT,
            "censor_risk": t_arrive + d_actual + 3.0 * d_actual
            > DDM_TIME_LIMIT,
            "discretisation_limited": z_halt < EVIDENCE_SUBSTEP,
        })
    return out
