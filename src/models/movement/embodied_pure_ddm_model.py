# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
#  You may use, modify, and redistribute this file according to the terms of the
#  license. Attribution is required if this code is used in other works.
# ------------------------------------------------------------------------------

"""Embodied pure two-alternative drift-diffusion movement model.

Third decision substrate alongside `MeanFieldMovementModel` (ring attractor) and
`EmbodiedDDMMovementModel` (LCA). Shares the perception spine (`TargetModel`), the
evidence-construction stages (`AccumulatorSystem.compute_evidence`), and the actuation
code path with the other two, so any behavioural difference is attributable to the
decision substrate. See EMBODIED_DDM_PURE_PLAN.md.

K = 2 ONLY. The pure DDM is the SPRT for a binary hypothesis test; there is no scalar
log-likelihood ratio for K > 2. For K > 2 use `accumulator_mode: race` (the MSPRT) in
the `embodied_ddm` model — a race of K accumulators, not a DDM.

Boundary-policy coherence (plan Section 4.1, Appendix C): `bayes_risk` and `reward_rate`
give the optimal *constant* boundary. Under their own assumptions the constant boundary
is provably optimal and any collapse strictly increases Bayes risk, so combining either
with `boundary_mode: collapsing` is flagged, logged, and stamped into the per-trial
record as `boundary_policy_incoherent`.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from typing import Optional, Sequence

import numpy as np

from models.accumulator_systems import AccumulatorSystem
from models.bifurcation import BifurcationDetector
from models.ddm_systems import DriftDiffusionSystem
from models.egocentric_target_model import TargetModel
from models.readout import circular_readout
from models.utils import normalize_angle
from plugin_registry import register_movement_model

logger = logging.getLogger("sim.embodied_pure_ddm")
logger.setLevel(logging.DEBUG)


def _wrap_pi(angle: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


class EmbodiedPureDDMMovementModel(TargetModel):
    """Movement model driven by a pure 2AFC drift-diffusion process."""

    def __init__(self, agent):
        """Initialize the instance."""
        params = agent.config_elem.get("embodied_pure_ddm", {}) or {}
        self._init_target_model(agent, params)

        # --- structure: K == 2 is a hard requirement (Task 6.3) ---
        if len(self.target_ids) != 2:
            raise ValueError(
                f"{agent.get_name()}: embodied_pure_ddm requires exactly 2 target_ids "
                f"(got {len(self.target_ids)}). The pure DDM is the SPRT for a binary "
                "hypothesis test; there is no scalar LLR for K > 2. Use "
                "embodied_ddm with accumulator_mode 'race' (MSPRT) for K > 2."
            )

        # --- evidence ---
        self.eta_rate = params.get("eta_rate", [0.3, 0.3])
        self.evidence_mode = str(params.get("evidence_mode", "difference"))
        self.dist_mode = str(params.get("dist_mode", "none"))
        self.d_0 = float(params.get("d_0", 1.0))
        self.target_radius = float(params.get("target_radius", 0.05))
        self.loom_filter_ticks = float(params.get("loom_filter_ticks", 4.0))
        self.attention_mode = str(params.get("attention_mode", "none"))
        self.kappa_a = float(params.get("kappa_a", 4.0))
        self.saccade_rate_hz = float(params.get("saccade_rate_hz", 2.0))
        self.extended = bool(params.get("extended", False))

        # --- boundary ---
        self.boundary_mode = str(params.get("boundary_mode", "static"))
        self.threshold_policy = str(params.get("threshold_policy", "bayes_risk"))
        self.cost_ratio = float(params.get("cost_ratio", 10.0))
        self.T0 = float(params.get("T0", 0.0))
        self.threshold_update_ticks = max(0, int(params.get("threshold_update_ticks", 0)))
        self.geometric_error_mode = str(params.get("geometric_error_mode", "terminal_graded"))
        self.delta_min = float(params.get("delta_min", 1e-3))
        self.A_source = str(params.get("A_source", "ensemble"))
        # TASK_A_KNOWN_DRIFT.md: the implemented model assumes the agent knows
        # |A| but not its sign. 'estimated' opts into the Task B arm, where |A|
        # is inferred from the evidence and `a` is no longer a confidence.
        self.drift_knowledge = str(
            params.get("drift_knowledge", "known_magnitude")
        ).strip().lower()
        self.A_lognormal_s = float(params.get("A_lognormal_s", 0.0))
        self.A_lognormal_debias = bool(params.get("A_lognormal_debias", True))
        self.A_lognormal_redraw = str(params.get("A_lognormal_redraw", "onset"))
        self.A_hat_min = float(params.get("A_hat_min", 1e-9))

        # lambda_t: opportunity cost of time in quality units per second (the currency of
        # Charnov's marginal value theorem). Only read under terminal_graded.
        #   float  -> used directly
        #   null   -> analytic default E[q]*v/E[d], resolved at evidence onset
        #   "mvt"  -> the marginal-value fixed point; iteration k uses lambda_t_measured
        #             fed back from run k-1, iteration 0 falls back to the analytic default
        raw_lambda_t = params.get("lambda_t", None)
        self.lambda_t_mode = "explicit"
        self.lambda_t: Optional[float] = None
        if isinstance(raw_lambda_t, str):
            if raw_lambda_t.strip().lower() != "mvt":
                raise ValueError("lambda_t must be a positive number, null, or 'mvt'")
            self.lambda_t_mode = "mvt"
            measured = params.get("lambda_t_measured")
            if measured is not None:
                self.lambda_t = float(measured)
                if self.lambda_t <= 0.0:
                    raise ValueError("lambda_t_measured must be > 0")
        elif raw_lambda_t is None:
            self.lambda_t_mode = "analytic"
        else:
            self.lambda_t = float(raw_lambda_t)
            if self.lambda_t <= 0.0:
                raise ValueError("lambda_t must be > 0 (quality units per second)")

        # evidence_target: strictly, under an asymmetric layout the agent should
        # accumulate sampled NET VALUE differences (quality net of travel cost), since a
        # nearer but slightly worse target can be the correct choice. Irrelevant in the
        # symmetric case and a larger change than this fix.
        self.evidence_target = str(params.get("evidence_target", "quality")).strip().lower()
        if self.evidence_target == "net_value":
            raise NotImplementedError(
                "evidence_target 'net_value' (accumulating q_i - lambda_t*d_i/v rather "
                "than q_i) is specified but not implemented; use 'quality'."
            )
        if self.evidence_target != "quality":
            raise ValueError("evidence_target must be 'quality' or 'net_value'")

        # --- readout / motion ---
        self.readout_mode = str(params.get("readout_mode", "normalized"))
        # --- bellman policy (FEATURE_BELLMAN_POLICY) ---
        self.bellman_cfg = dict(params.get("bellman") or {})
        if self.threshold_policy == "bellman" and self.threshold_update_ticks not in (0, 1):
            # The whole trajectory is already solved, so per-tick re-evaluation has
            # nothing to do. Say so rather than letting a config carried over from
            # `geometric` look as though it is still doing something.
            logger.warning(
                "%s: threshold_update_ticks=%d is IGNORED under threshold_policy "
                "'bellman' -- z(t) is precomputed for the whole horizon.",
                agent.get_name(), self.threshold_update_ticks,
            )
        self._bellman_diag = None
        self._bellman_solved = False

        # --- terminal condition of the bellman solve (BELLMAN_KNOWN_A_TERMINAL_HALT) ---
        #   forced_choice - commit at arrival whatever you believe; z(T_max) = 0.
        #   halt_sprt     - halt at the midpoint at arrival and keep integrating at
        #                   delay rate halt_cost_rate against the flat threshold z_halt.
        self.bellman_terminal = str(
            self.bellman_cfg.get("terminal", "forced_choice")
        ).strip().lower()
        if self.bellman_terminal not in {"forced_choice", "halt_sprt"}:
            raise ValueError(
                "bellman.terminal must be 'forced_choice' or 'halt_sprt'; got "
                f"'{self.bellman_terminal}'"
            )
        self.halt_cost_rate = float(self.bellman_cfg.get("halt_cost_rate", 1.0))
        if self.bellman_terminal == "halt_sprt":
            if self.halt_cost_rate <= 0.0:
                raise ValueError("bellman.halt_cost_rate must be > 0")
            if self.drift_knowledge != "known_magnitude":
                # Section 3: the halted problem is stationary ONLY when |A| is known.
                # Under the unknown-A arm the reparameterised running cost grows in time
                # and the belief map is time-dependent, so no flat-boundary closed form
                # exists. Refuse rather than produce a plausible wrong plateau.
                raise ValueError(
                    "bellman.terminal 'halt_sprt' requires drift_knowledge "
                    "'known_magnitude': the halted problem is stationary only when |A| "
                    f"is known (got drift_knowledge '{self.drift_knowledge}')."
                )
            if self.halt_cost_rate != 1.0:
                logger.warning(
                    "%s: bellman.halt_cost_rate = %.3g != 1.0 is a sensitivity knob, "
                    "not the physical value (halted = zero progress = rate 1)%s",
                    agent.get_name(), self.halt_cost_rate,
                    "; c_h < 1 voids the halt-only-at-arrival domination lemma "
                    "(Section 1)." if self.halt_cost_rate < 1.0 else ".",
                )
        # Per-run halt state; re-initialised in reset() like the solver flags above.
        self._bellman_z_halt = None
        self._bellman_T_max = None
        self._halt_mean_exit = None
        self._halted_now = False
        self._halt_event = False
        self._x_at_arrival = None
        self._t_halt_commit = None
        self._halt_guard_hits = 0

        # --- static bounds as the degenerate case (RA-DDM frontier spec §2b) ---
        # bellman.static_bound = b > 0 replaces the PDE solve with the flat table
        # z(t) = b on [0, T_max] through the SAME machinery — the accumulator's
        # hold-past-the-horizon rule and (under terminal 'halt_sprt') the
        # halt-at-midpoint motor hold apply unchanged with z_halt = b. One code
        # path, two parameterizations; never a parallel implementation.
        raw_static = self.bellman_cfg.get("static_bound")
        self.bellman_static_bound = None if raw_static is None else float(raw_static)
        if self.bellman_static_bound is not None:
            if self.bellman_static_bound <= 0.0:
                raise ValueError("bellman.static_bound must be > 0")
            if self.threshold_policy != "bellman":
                raise ValueError(
                    "bellman.static_bound is only read under threshold_policy "
                    f"'bellman' (got '{self.threshold_policy}'); it would be "
                    "silently ignored otherwise."
                )

        # --- post-commitment flexibility (FEATURE_POST_COMMITMENT_FLEXIBILITY) ---
        self.flexibility = bool(params.get("flexibility", False))
        self.post_commit_accumulation = str(
            params.get("post_commit_accumulation", "unbounded")
        ).strip().lower()
        if self.post_commit_accumulation not in {"unbounded", "clamp"}:
            raise ValueError(
                "post_commit_accumulation must be 'unbounded' or 'clamp'; got "
                f"'{self.post_commit_accumulation}'"
            )
        self.quality_swap = self._parse_quality_swap(params.get("quality_swap"))

        self.predecision_motion = str(params.get("predecision_motion", "average")).strip().lower()
        if self.predecision_motion not in {"average", "midpoint", "forward", "stationary"}:
            raise ValueError(
                "predecision_motion must be 'average', 'midpoint', 'forward' or 'stationary'"
            )
        self.scaling_mode = str(params.get("scaling_mode", "concentration")).strip().lower()
        if self.scaling_mode not in {"concentration", "constant", "magnitude"}:
            raise ValueError("scaling_mode must be 'concentration', 'constant' or 'magnitude'")
        self.norm_scale = float(params.get("norm_scale", 1.0))
        self.bisector_eps = float(params.get("bisector_eps", 1e-6))
        if self.predecision_motion == "midpoint" and self.scaling_mode == "concentration":
            logger.info(
                "%s: predecision_motion 'midpoint' with scaling_mode 'concentration' — "
                "concentration is then purely geometric (= cos(Delta/2)), carries no "
                "decision information, and drives speed to zero as Delta -> pi, so the "
                "agent creeps to the midpoint and hovers. Use scaling_mode 'constant' if "
                "you want the mode to do exactly one thing (decouple heading from evidence).",
                agent.get_name(),
            )

        # Task 6.3: enforce the Section 4.1 compatibility table.
        self.boundary_policy_incoherent = (
            self.boundary_mode == "collapsing"
            and self.threshold_policy in {"bayes_risk", "reward_rate"}
        )
        if self.boundary_policy_incoherent:
            logger.warning(
                "%s: threshold_policy '%s' gives the optimal CONSTANT boundary; combining it "
                "with boundary_mode 'collapsing' is NOT justified by that criterion (under its "
                "own assumptions the constant boundary is provably optimal — see "
                "EMBODIED_DDM_PURE_PLAN.md Section 4.1 / Appendix C). Flagging the trial as "
                "boundary_policy_incoherent; do not report this combination as optimal.",
                agent.get_name(),
                self.threshold_policy,
            )

        # Runtime state.
        self.ddm: Optional[DriftDiffusionSystem] = None
        self.evidence: Optional[AccumulatorSystem] = None
        self._last_heading: Optional[float] = None
        self._last_state = None
        self._last_weights = np.array([0.5, 0.5])
        self._last_concentration = 0.0
        self._last_magnitude = 0.0
        self._evidence_started = False
        self._tick_count = 0
        self._commit_effective_at: Optional[float] = None
        self._slot_phi = np.zeros(2, dtype=float)
        self._slot_d = np.zeros(2, dtype=float)
        self._last_q = np.zeros(2, dtype=float)
        self._geom_log: dict = {}
        self._log_odds_at_commit: Optional[float] = None
        # Midpoint-mode readout diagnostics (Task M4).
        self._last_r_geom = 0.0
        self._bisector_guard_fired = False
        self._diagnostics_done = False
        # Turn kinematics: commitment steps the heading target by ~Delta/2, and the
        # actuator clips to max_angular_velocity, so the turn takes finite time. This is
        # the motor-level signature of the bifurcation.
        self._commit_tick: Optional[int] = None
        self._turn_duration_ticks: Optional[int] = None
        self._heading_at_commitment: Optional[float] = None
        self._heading_error_at_commitment: Optional[float] = None
        self.alignment_tolerance_deg = float(
            (params.get("bifurcation", {}) or {}).get("alignment_tolerance_deg", 5.0)
        )

        bif_cfg = params.get("bifurcation", {})
        self.bifurcation_detector = BifurcationDetector(
            agent_name=str(agent.get_name()),
            lambda_threshold=float(bif_cfg.get("lambda_threshold", -0.1)),
            spike_min_separation=int(bif_cfg.get("spike_min_separation", 10)),
            mode=str(bif_cfg.get("mode", "behavioral")),
            alignment_tolerance_deg=float(bif_cfg.get("alignment_tolerance_deg", 5.0)),
            alignment_consecutive_ticks=int(bif_cfg.get("alignment_consecutive_ticks", 5)),
            gradient_window=int(bif_cfg.get("gradient_window", 5)),
            gradient_threshold=float(bif_cfg.get("gradient_threshold", 0.005)),
        )
        # Sensory percept stream (FEATURE_SHARED_SENSORY_STREAM.md). Under `shared` the
        # white sensory noise moves upstream into the stream, so eta_rate must be 0 in
        # the config; `reset()` then hands the stream's own `white_rate` to the DDM as
        # the noise SCALE (c), while the realisation comes from the stream.
        self._init_percept_stream(
            dist_mode=self.dist_mode,
            attention_mode=self.attention_mode,
            sigma_s=0.0,
            eta_rate=self.eta_rate,
        )
        self.reset()
        logger.info(
            "%s embodied pure-DDM instantiated (policy=%s, boundary=%s, motion=%s, c=%.4f)",
            self.agent.get_name(),
            self.threshold_policy,
            self.boundary_mode,
            self.predecision_motion,
            self.ddm.c if self.ddm else float("nan"),
        )

    # ------------------------------------------------------------------
    def _resolve_evidence_noise(self) -> tuple[Sequence[float], bool]:
        """Return `(eta_rate, external_percept)` for the configured sensory protocol.

        `legacy`: the configured `eta_rate`, and this model draws its own `q_hat`.

        `shared`: the percept is drawn upstream and handed in already sampled, so the
        DDM must not draw again. `eta_rate` is then the stream's `white_rate` on both
        channels — not to generate noise, but because `c^2 = sum(eta_rate^2)` is the
        noise SCALE every boundary quantity is expressed in (`z* = a c^2 / 2A`, ER, DT).
        Leaving it at the configured 0 would drive `c` to zero and collapse every
        derived threshold, which would silently change the decision policy rather than
        just the noise source. The config's own `eta_rate` is required to be 0 under
        `shared` precisely so there is no ambiguity about where this number came from.

        Note the frozen channel `s_beta` deliberately does NOT enter `c`: it is constant
        within a trial, so it shifts the drift `A` of that trial rather than adding
        diffusion, exactly as the ring attractor's `sigma_s` does.
        """
        spec = getattr(self, "sensory_stream_spec", None)
        if spec is None or not spec.is_shared:
            return self.eta_rate, False
        eta = float(spec.white_rate)
        return (eta, eta), True

    def reset(self) -> None:
        """Reset the decision process and the evidence pipeline."""
        self.perception = None
        # Built before the DDM: under `shared` the stream supplies the noise scale the
        # boundary mathematics needs (see `_resolve_evidence_noise`).
        self._build_percept_stream()
        self._last_heading = None
        self._evidence_started = False
        self._tick_count = 0
        self._commit_effective_at = None
        self._log_odds_at_commit = None
        self._last_r_geom = 0.0
        self._bisector_guard_fired = False
        self._diagnostics_done = False
        self._commit_tick = None
        self._turn_duration_ticks = None
        self._heading_at_commitment = None
        self._heading_error_at_commitment = None
        # --- flexibility / quality-swap per-trial state ---
        self._transitions: list[dict] = []
        self._swap_applied_at: Optional[float] = None
        self._swap_armed_at: Optional[float] = None
        self._x_at_swap: Optional[float] = None
        self._dwell_before_swap: Optional[float] = None
        self._t_release_after_swap: Optional[float] = None
        self._t_recommit_after_swap: Optional[float] = None
        self._total_path_length = 0.0
        self._last_xy: Optional[tuple] = None
        self._feasibility_checked = False
        self._bellman_solved = False
        self._bellman_diag = None
        # --- halt_sprt per-run state (BELLMAN_KNOWN_A_TERMINAL_HALT Section 4.3) ---
        self._bellman_z_halt = None          # z_halt of the installed table, halt mode only
        self._bellman_T_max = None           # arrival horizon of the installed table
        self._halt_mean_exit = None          # D(z_halt), sizes the runaway guard cap
        self._halted_now = False             # v = 0 hold active this tick
        self._halt_event = False             # reached arrival uncommitted
        self._x_at_arrival = None            # x when the halt began
        self._t_halt_commit = None           # t_evidence of the post-arrival commit
        self._halt_guard_hits = 0            # forced commits by the runaway guard
        self._A_expected_resolved = False
        params = self.params

        # The evidence pipeline is literally the LCA's: same quality modulation, same
        # dist_mode, same attention_mode, same loom filtering. Configured with
        # normalize 'none' and gamma 1 so it returns the raw per-target quality q_i,
        # which this model then samples. Its accumulators are never integrated.
        self.evidence = AccumulatorSystem(
            max_targets=2,
            target_ids=self.target_ids,
            masked_policy="leak",
            dist_mode=self.dist_mode,
            d_0=self.d_0,
            target_radius=self.target_radius,
            loom_filter_ticks=self.loom_filter_ticks,
            attention_mode=self.attention_mode,
            kappa_a=self.kappa_a,
            saccade_rate_hz=self.saccade_rate_hz,
            normalize="none",
            sigma_n=0.0,
            gamma=1.0,
            sigma_s=0.0,
            target_quality_modulations=self.target_quality_modulations,
            sensory_time_mode=self.sensory_time_mode,
            sensory_dt=self.sensory_dt,
            accumulator_mode="race",
            sigma=0.0,
            y_floor=None,
            n_sub=1,
            rng=self._make_rng("evidence"),
        )

        eta_rate_effective, external_percept = self._resolve_evidence_noise()
        self.ddm = DriftDiffusionSystem(
            eta_rate=eta_rate_effective,
            evidence_mode=self.evidence_mode,
            external_percept=external_percept,
            extended=self.extended,
            s_A=float(params.get("s_A", 0.0)),
            s_x=float(params.get("s_x", 0.0)),
            x0=float(params.get("x0", 0.0)),
            n_sub=max(1, int(params.get("n_sub", 1))),
            boundary_mode=self.boundary_mode,
            threshold_policy=self.threshold_policy,
            z_manual=float(params.get("z_manual", 1.0)),
            cost_ratio=self.cost_ratio,
            D_iti=float(params.get("D_iti", 1.0)),
            D_penalty=float(params.get("D_penalty", 0.0)),
            T0=self.T0,
            threshold_update_ticks=self.threshold_update_ticks,
            geometric_error_mode=self.geometric_error_mode,
            lambda_t=self.lambda_t,
            delta_min=self.delta_min,
            A_source=self.A_source,
            drift_knowledge=self.drift_knowledge,
            A_lognormal_s=self.A_lognormal_s,
            A_lognormal_debias=self.A_lognormal_debias,
            A_lognormal_redraw=self.A_lognormal_redraw,
            A_hat_min=self.A_hat_min,
            A_rng=self._make_rng("A_hat"),   # separate stream from the evidence noise
            collapse_form=str(params.get("collapse_form", "weibull")),
            z_min=float(params.get("z_min", 0.05)),
            collapse_rate=float(params.get("collapse_rate", 0.5)),
            tau_c=float(params.get("tau_c", 2.0)),
            weibull_k=float(params.get("weibull_k", 3.0)),
            weibull_a_inf=float(params.get("weibull_a_inf", 0.2)),
            deadline_T=params.get("deadline_T"),
            urgency_mode=str(params.get("urgency_mode", "off")),
            urgency_tau=float(params.get("urgency_tau", 0.5)),
            urgency_slope=float(params.get("urgency_slope", 1.0)),
            readout_mode=self.readout_mode,
            posterior_gain=params.get("posterior_gain"),
            allow_changes_of_mind=bool(params.get("allow_changes_of_mind", False)),
            com_window=float(params.get("com_window", 0.3)),
            flexibility=self.flexibility,
            post_commit_accumulation=self.post_commit_accumulation,
            A_expected=params.get("A_expected"),
            # The scenario already declares the target qualities, so A_expected is
            # redundant under 'ensemble': defer it and deduce the gap from those declared
            # qualities at evidence onset (_resolve_A_expected). NOT the deleted
            # running-mean fallback -- the value comes from the scenario definition, so it
            # is a block constant, known before the first tick and free of trial history.
            A_expected_deferred=(
                params.get("A_expected") is None and self.A_source == "ensemble"
            ),
            rng=self._make_rng("ddm"),
        )
        self.ddm.flag_policy_incoherent(self.boundary_policy_incoherent)
        # collapse_form 'geometric' IS the per-tick re-evaluation of rho from the current
        # angular separation — with the threshold frozen at onset there is nothing to
        # collapse, so force the recompute rather than silently producing a flat boundary.
        if (
            self.boundary_mode == "collapsing"
            and self.ddm.collapse_form == "geometric"
            and self.threshold_update_ticks == 0
        ):
            self.threshold_update_ticks = 1
            self.ddm.threshold_update_ticks = 1
            logger.warning(
                "%s: collapse_form 'geometric' re-evaluates rho from the live geometry "
                "each tick, so threshold_update_ticks=0 would freeze it; forcing it to 1.",
                self.agent.get_name(),
            )
        if hasattr(self, "bifurcation_detector"):
            self.bifurcation_detector.reset()
        logger.debug("%s pure-DDM reset", self.agent.get_name())

    # ------------------------------------------------------------------
    def _zero_commands(self) -> None:
        """Stop the agent and clear transient readout state."""
        self.agent.linear_velocity_cmd = 0.0
        self.agent.angular_velocity_cmd = 0.0
        self._last_concentration = 0.0
        self._last_magnitude = 0.0

    def _speed_per_second(self) -> float:
        """Agent speed in metres per SECOND (max_absolute_velocity is per tick)."""
        return float(self.agent.max_absolute_velocity) * self._resolve_agent_tick_rate()

    def step(self, agent, tick: int, arena_shape, objects: dict, agents: dict) -> None:
        """Execute a simulation step."""
        start_time = time.perf_counter()
        if self.ddm is None:
            self.reset()
        try:
            self._decide_and_actuate(objects, agents, tick, arena_shape)
        finally:
            if logger.isEnabledFor(logging.DEBUG):
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                logger.debug(
                    "%s pure-DDM step duration = %.3f ms", self.agent.get_name(), elapsed_ms
                )

    # ------------------------------------------------------------------
    def _decide_and_actuate(self, objects, agents, tick, arena_shape) -> None:
        """Per-tick pipeline (Task 6.2)."""
        self._update_perception(objects, agents, tick, arena_shape)

        entities = (self._mf_entities or {}).get("targets") or []
        percept = self._build_target_percept()

        # Step 2 — gate. Fewer than two targets means no evidence this tick, so the
        # evidence clock must NOT advance: otherwise a collapsing boundary would decay
        # during time in which the agent had nothing to integrate (plan Section 4.3).
        if len(percept.ids) < 2:
            self._zero_commands()
            return
        if len(percept.ids) > 2:
            raise ValueError(
                f"{self.agent.get_name()}: embodied_pure_ddm saw {len(percept.ids)} targets; "
                "K = 2 only. Check target_ids."
            )

        # Step 4 — resolve slots by configured order so the sign of x is stable.
        order = []
        for tid in self.target_ids:
            if tid not in percept.ids:
                self._zero_commands()
                return
            order.append(percept.ids.index(tid))
        order = np.asarray(order, dtype=int)
        ids = [percept.ids[i] for i in order]
        phi = percept.phi[order]
        d = percept.d[order]
        s = percept.s[order]
        self._slot_phi = phi.copy()
        self._slot_d = d.copy()

        dt = 1.0 / self._resolve_agent_tick_rate()
        heading_prev = self._last_heading if self._last_heading is not None else 0.0

        # Step 5 — shared evidence stages (quality modulation, dist_mode, attention).
        indices = self.evidence.register_targets(ids)
        self.evidence.mask[:] = False
        self.evidence.mask[indices] = True
        self.evidence._advance_saccade(phi, dt)
        mu = self.evidence.compute_evidence(indices, phi, d, s, ids, heading_prev, dt)
        self.evidence._advance_sensory_time()
        q = mu[indices]
        # The world change enters HERE, through the same evidence path as everything
        # else: it alters the distribution the accumulator samples from and is never
        # special-cased inside the accumulator (Section 2).
        q = self._maybe_swap_qualities(q)
        self._last_q = q.copy()

        # Steps 6-7 — resolve the drift estimate, refresh the boundary, then integrate.
        # update_A_hat must run BEFORE _update_threshold: the threshold consumes
        # A_hat, and at evidence onset ddm.step() has not run yet.
        # The deduction must read the DECLARED quality of each target — the scenario
        # definition — never a sampled percept. Under sensory_stream 'shared',
        # `percept.s` is already one noisy realisation (the stream is applied upstream
        # in `_build_target_percept`), so handing it to `_resolve_A_expected` would make
        # the agent's "known" |A| a random variable with sd ~ eta*sqrt(2/dt) — silently
        # breaking drift_knowledge 'known_magnitude' and solving each replicate's
        # boundary for a different drift. `_percept_clean_s` carries the pre-stream
        # values; under 'legacy' the stream is a pass-through and they equal `s`.
        clean_s = getattr(self, "_percept_clean_s", None)
        if clean_s and all(t in clean_s for t in ids):
            s_declared = np.array([clean_s[t] for t in ids], dtype=float)
        else:
            s_declared = s
        self._resolve_A_expected(s_declared, ids)
        self.ddm.update_A_hat(q)
        self._update_threshold(q, phi, d)
        state = self.ddm.step(q, dt)
        # Terminal halt (BELLMAN_KNOWN_A_TERMINAL_HALT): past arrival the boundary is
        # the flat z_halt and the agent parks at the midpoint until |x| crosses it. May
        # refresh `state` if the runaway guard forced a commit.
        state = self._update_halt_state(state)
        self._last_state = state

        self._check_reversal_feasibility(d)

        if state.released_this_step:
            # The boundary is non-absorbing: dropping back below z returns the agent to
            # pre-decision motion, so the actuator latch must be cleared or the stale
            # commitment time would keep the heading pinned to the abandoned target.
            self._commit_effective_at = None

        if state.crossed_this_step and state.committed is not None:
            # Non-decision time is actuator lag here, not a fitted nuisance parameter:
            # latch internally now, reach the motors T0 seconds later.
            self._commit_effective_at = state.t_evidence + self.T0
            self._log_odds_at_commit = self.ddm.implied_log_odds

        if self.flexibility and self.ddm.pending_transitions:
            self._record_transition(state, phi, d, tick)

        # Path length, for the overshoot of Section 3.3.
        xy = (float(self.agent.position.x), float(self.agent.position.y))
        if self._last_xy is not None:
            self._total_path_length += math.hypot(xy[0] - self._last_xy[0],
                                                  xy[1] - self._last_xy[1])
        self._last_xy = xy

        # Steps 9-10 — readout and actuation.
        committed_effective = self._commit_is_effective(state)
        # An effective commitment releases the terminal-halt motor hold: the agent
        # leaves the midpoint for the chosen target.
        self._halted_now = self._halted_now and not committed_effective

        # Beliefs are ALWAYS evaluated and logged, in every motion mode. Under
        # `midpoint` only the *heading* ignores them — the comparison against `average`
        # depends on having P(t) recorded under both with everything else identical.
        weights = self.ddm.weights(force_belief=not committed_effective)
        self._last_weights = weights.copy()

        # Purely geometric concentration |sum of unit bearing vectors|/2 == cos(Delta/2).
        # Logged separately from `concentration` so the geometric and belief-derived
        # values are never conflated when comparing modes.
        r_geom = 0.5 * math.hypot(
            math.sin(phi[0]) + math.sin(phi[1]), math.cos(phi[0]) + math.cos(phi[1])
        )
        self._last_r_geom = float(r_geom)
        self._bisector_guard_fired = False

        heading_weights = self._heading_weights(weights, state, committed_effective)
        heading, magnitude, concentration = circular_readout(
            heading_weights, phi, threshold=0.0
        )

        # Delta -> pi makes the two unit vectors cancel, so atan2(0, 0) is undefined.
        # Hold the previous heading rather than snapping to an arbitrary angle. A guard
        # that fires repeatedly means the agent is parked between the targets, which is a
        # result (see the stall-at-the-midpoint behaviour), not an error.
        if (
            self.predecision_motion == "midpoint"
            and not committed_effective
            and r_geom < self.bisector_eps
        ):
            self._bisector_guard_fired = True
            heading = self._last_heading if self._last_heading is not None else 0.0
            if not getattr(self, "_warned_bisector_guard", False):
                self._warned_bisector_guard = True
                logger.warning(
                    "%s: bisector guard fired (R_geom=%.3g < %.3g, Delta -> pi): the two "
                    "bearings cancel, so the midpoint heading is undefined and the "
                    "previous heading is held. The agent is parked between the targets.",
                    self.agent.get_name(), r_geom, self.bisector_eps,
                )

        self._last_heading = heading
        self._last_concentration = float(concentration)
        self._last_magnitude = float(magnitude)

        self._actuate(heading, magnitude, concentration, committed_effective, phi, state)
        self._track_turn(state, phi, tick, committed_effective)

        # Step 11 — bifurcation detection, identical detector and mode as the other models.
        self.bifurcation_detector.update(
            tick=tick if tick is not None else 0,
            mf=_PureDDMBifShim(self.ddm),
            bump_angle=heading,
            target_angles=[float(a) for a in phi],
            target_ids=self.target_ids,
            perception_vec=self.perception,
            agent_angle=0.0,
            agent_x=float(self.agent.position.x),
            agent_y=float(self.agent.position.y),
            agent_orientation=float(self.agent.orientation.z),
        )
        self._tick_count += 1

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "%s pure-DDM x=%.4f z=%.4f t_ev=%.3f P0=%.3f committed=%s",
                self.agent.get_name(), state.x, state.z, state.t_evidence,
                weights[0], state.committed,
            )

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Post-commitment flexibility (FEATURE_POST_COMMITMENT_FLEXIBILITY)
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_quality_swap(raw) -> Optional[dict]:
        """Validate the `quality_swap` block; returns None when absent or disabled."""
        if not isinstance(raw, dict):
            return None
        cfg = {
            "enabled": bool(raw.get("enabled", False)),
            "trigger": str(raw.get("trigger", "on_commit")).strip().lower(),
            "delay": float(raw.get("delay", 1.0)),
            "at_time": raw.get("at_time"),
            "mode": str(raw.get("mode", "exchange")).strip().lower(),
            "new_qualities": raw.get("new_qualities"),
        }
        if not cfg["enabled"]:
            return cfg
        if cfg["trigger"] not in {"on_commit", "at_time"}:
            raise ValueError("quality_swap.trigger must be 'on_commit' or 'at_time'")
        if cfg["mode"] not in {"exchange", "set"}:
            raise ValueError("quality_swap.mode must be 'exchange' or 'set'")
        if cfg["trigger"] == "at_time" and cfg["at_time"] is None:
            raise ValueError("quality_swap.trigger 'at_time' requires at_time")
        if cfg["mode"] == "set":
            vals = np.asarray(cfg["new_qualities"], dtype=float).reshape(-1)
            if vals.size != 2:
                raise ValueError("quality_swap.new_qualities must hold exactly 2 values")
            cfg["new_qualities"] = vals
        if cfg["delay"] < 0.0:
            raise ValueError("quality_swap.delay must be >= 0")
        return cfg

    def _maybe_swap_qualities(self, q):
        """Apply the world change to `q` BEFORE it is sampled.

        The swap deliberately enters through the same evidence path as everything else:
        it changes the sampling distribution the accumulator draws from, and the
        accumulator is not told about it. Times are on the DDM's evidence clock, and
        `delay` applies to both triggers (Section 2).
        """
        cfg = self.quality_swap
        if cfg is None or not cfg["enabled"]:
            return q
        t = float(self.ddm.t_evidence)
        if self._swap_applied_at is None:
            if self._swap_armed_at is None:
                if cfg["trigger"] == "on_commit":
                    if self.ddm.t_first_commit is not None:
                        self._swap_armed_at = float(self.ddm.t_first_commit) + cfg["delay"]
                else:
                    self._swap_armed_at = float(cfg["at_time"]) + cfg["delay"]
                if self._swap_armed_at is not None:
                    logger.info(
                        "%s: quality_swap armed for t_evidence = %.3f s (trigger '%s')",
                        self.agent.get_name(), self._swap_armed_at, cfg["trigger"],
                    )
            if self._swap_armed_at is None or t < self._swap_armed_at:
                return q
            self._swap_applied_at = t
            self._x_at_swap = float(self.ddm.x)
            self._dwell_before_swap = (
                t - float(self.ddm.t_first_commit)
                if self.ddm.t_first_commit is not None else None
            )
            logger.info(
                "%s: quality_swap APPLIED at t_evidence = %.3f s (mode '%s'), "
                "x = %.4f, dwell = %s s",
                self.agent.get_name(), t, cfg["mode"], self._x_at_swap,
                "n/a" if self._dwell_before_swap is None
                else f"{self._dwell_before_swap:.3f}",
            )
        q = np.asarray(q, dtype=float).copy()
        if cfg["mode"] == "exchange":
            q[0], q[1] = q[1], q[0]
        else:
            q[:2] = cfg["new_qualities"]
        return q

    def _record_transition(self, state, phi, d, tick) -> None:
        """Drain the accumulator's transitions and stamp each with the current pose.

        The events themselves are timed at sub-step resolution by the accumulator; the
        geometry is per-tick, which is correct because the pose does not change within a
        tick. Draining rather than re-deriving is what keeps commit/release pairs that
        open and close inside one tick from being lost.
        """
        geom = self._geom_log or {}
        for ev in self.ddm.pending_transitions:
            prev_t = self._transitions[-1]["t"] if self._transitions else 0.0
            self._transitions.append({
                "t": ev["t"],
                "tick": int(tick) if tick is not None else self._tick_count,
                "type": ev["type"],
                "committed_target": self._target_id(ev["target"]),
                "x": ev["x"], "z": ev["z"],
                "a_star": geom.get("a_star"),
                "delta": geom.get("delta"),
                "d1": float(d[0]), "d2": float(d[1]),
                "agent_x": float(self.agent.position.x),
                "agent_y": float(self.agent.position.y),
                "time_since_last_transition": ev["t"] - prev_t,
            })
            if ev["type"] == "release":
                if self._swap_applied_at is not None and self._t_release_after_swap is None:
                    self._t_release_after_swap = ev["t"]
            elif (
                self._swap_applied_at is not None
                and self._t_release_after_swap is not None
                and self._t_recommit_after_swap is None
            ):
                self._t_recommit_after_swap = ev["t"]
        self.ddm.pending_transitions.clear()

    def _target_id(self, index) -> Optional[str]:
        """Map a slot index onto its configured target id."""
        if index is None:
            return None
        ids = getattr(self, "target_ids", None)
        try:
            return str(ids[int(index)])
        except (TypeError, IndexError, ValueError):
            return str(index)

    def _check_reversal_feasibility(self, d) -> None:
        """Section 3.6 — flexibility can be geometrically impossible.

        The agent must be able to reverse before it arrives:

            d / v   <   delay + 2 z / A     ->  IMPOSSIBLE (arrives first)

        NOTE: the feature document states this inequality with the sign reversed, but
        its own worked example (trip 10 s vs reversal 13.5 s, "the agent reaches the
        wrong target before it can change its mind") is the form implemented here.
        Without this check a run reports a 0% reversal rate that looks like a broken
        mechanism but is really the arena.
        """
        if self._feasibility_checked or not self.flexibility:
            return
        v = self._speed_per_second()
        A = abs(float(self.ddm.A_hat))
        z = float(self.ddm.z_current)
        if v <= 1e-12 or A <= 1e-12 or z <= 0.0:
            return
        self._feasibility_checked = True
        d_mean = float(np.mean(np.asarray(d, dtype=float)))
        t_arrive = d_mean / v
        delay = (
            float(self.quality_swap["delay"])
            if self.quality_swap and self.quality_swap["enabled"] else 0.0
        )
        t_reverse = delay + 2.0 * z / A
        logger.info(
            "%s flexibility feasibility: travel d/v = %.2f s  vs  reversal "
            "delay + 2z/A = %.2f + %.2f = %.2f s",
            self.agent.get_name(), t_arrive, delay, 2.0 * z / A, t_reverse,
        )
        if t_arrive < t_reverse:
            logger.warning(
                "%s: REVERSAL IS GEOMETRICALLY IMPOSSIBLE — the agent arrives in "
                "%.2f s but needs %.2f s to reverse (delay %.2f + 2z/A = 2*%.3g/%.3g). "
                "Any 0%% reversal rate from this run is the arena, not the model. "
                "Lower linear_velocity (<= %.3g), start further out, or swap sooner.",
                self.agent.get_name(), t_arrive, t_reverse, delay, z, A,
                d_mean / t_reverse,
            )

    def _commit_is_effective(self, state) -> bool:
        """True once a latched decision has cleared the non-decision (actuator) delay."""
        if state.committed is None or self._commit_effective_at is None:
            return False
        return state.t_evidence >= self._commit_effective_at

    def _resolve_A_expected(self, s, ids) -> None:
        """Deduce the ensemble |A| from the declared target qualities, and report it.

        A_expected is redundant information: the scenario declares each target's quality,
        and under `A_source: ensemble` the magnitude the agent is assumed to know is
        exactly their gap. Requiring it in the config as well means keeping two numbers in
        sync by hand, and nothing catches the drift when they diverge -- the run simply
        solves for a boundary that does not belong to the process it is simulating.

        This is NOT the running-mean fallback TASK_A_KNOWN_DRIFT.md Section 3 removed. The
        value comes from the scenario DEFINITION (`s` is the configured object strength,
        passed through the detection layer untouched), so it is a block-level constant,
        defined before the first tick and independent of trial history -- which is what
        keeps z*(Delta) invariant to the per-trial gap (test A2) and the collapse
        attributable to the geometry alone.

        Runs once per run, at evidence onset. Reports the number either way, because the
        experimenter's sanity check is `is the |A| the agent assumes the |A| the arena
        actually presents?`.
        """
        if self._A_expected_resolved:
            return
        self._A_expected_resolved = True

        declared = abs(float(s[0]) - float(s[1]))
        listing = ", ".join(
            f"{tid} = {float(val):.6g}" for tid, val in zip(ids, np.asarray(s, dtype=float))
        )
        name = self.agent.get_name()

        # The DDM clears the deferral wherever A_hat does not determine the boundary
        # (threshold_policy 'manual', drift_knowledge 'estimated'), so a symmetric-tie
        # scenario is never failed by a deduction nothing would have read.
        if self.ddm.A_expected_deferred:
            self.ddm.resolve_ensemble_A(declared, source="declared target qualities")
            logger.info(
                "%s: ensemble |A| = %.6g, DEDUCED from the declared target qualities "
                "(%s). Fixed for the block; set A_expected in the config to override it.",
                name, declared, listing,
            )
            if self.dist_mode != "none" or self.target_quality_modulations:
                logger.warning(
                    "%s: dist_mode '%s' / quality modulations make the REALISED drift "
                    "differ from the declared gap %.6g, which is what the deduced |A| "
                    "reports. The agent's assumed magnitude is the declared one by "
                    "construction; do not read the two as the same number.",
                    name, self.dist_mode, declared,
                )
        elif self.ddm.A_expected is not None:
            logger.info(
                "%s: ensemble |A| = %.6g (A_expected, set explicitly). The declared "
                "target qualities (%s) give a gap of %.6g.",
                name, float(self.ddm.A_expected), listing, declared,
            )
            if not math.isclose(float(self.ddm.A_expected), declared, rel_tol=1e-6,
                                abs_tol=1e-12):
                logger.warning(
                    "%s: A_expected = %.6g DISAGREES with the declared gap %.6g (%.0f%% "
                    "off). Under the known-|A| model the agent is assumed to KNOW the "
                    "magnitude, so with these two apart `a = 2Az/c^2` is no longer the "
                    "log-odds at commitment and the boundary does not belong to the "
                    "process being simulated. Remove A_expected to deduce it from the "
                    "qualities, or change the target strengths to match.",
                    name, float(self.ddm.A_expected), declared,
                    100.0 * abs(float(self.ddm.A_expected) - declared) / max(declared, 1e-12),
                )
        else:
            # 'oracle' (tracks the live gap by design) or a policy that never reads |A|.
            logger.info(
                "%s: |A| is not a declared constant here (A_source '%s', threshold_policy "
                "'%s'). The declared target qualities (%s) give a gap of %.6g.",
                name, self.A_source, self.threshold_policy, listing, declared,
            )

    def _update_threshold(self, q, phi, d) -> None:
        """Recompute the baseline boundary per `threshold_policy` (plan Sections 3.2-3.5)."""
        onset = not self._evidence_started
        if onset:
            self._evidence_started = True
        elif self.threshold_update_ticks <= 0:
            return  # computed once at evidence onset
        elif self._tick_count % self.threshold_update_ticks != 0:
            return

        # The A entering rho and z* is the agent's ESTIMATE (A_source), not the true
        # percept difference: the agent does not know the true drift.
        A = abs(float(self.ddm.A_hat))
        c = self.ddm.c
        policy = self.threshold_policy

        # Perfectly symmetric targets give A = 0, where every drift-dependent optimal
        # boundary correctly degenerates to z -> 0 ("evidence is worthless, do not pay
        # for it") and is then floored at z_min. That is the theory's answer, but it
        # makes a symmetric-tie experiment commit almost immediately, so say so loudly
        # once rather than letting it look like a tuning artefact.
        if policy in ("bayes_risk", "reward_rate", "geometric") and A <= 1e-9:
            if not getattr(self, "_warned_zero_drift", False):
                self._warned_zero_drift = True
                logger.warning(
                    "%s: drift A = q0 - q1 is ~0 (perfectly symmetric targets), so the "
                    "'%s' optimal boundary degenerates to z -> 0 and is floored at "
                    "z_min=%.3g; the agent will commit almost immediately by noise. For "
                    "symmetric-tie experiments use threshold_policy 'manual', or set "
                    "A_expected to the discriminability the agent should assume.",
                    self.agent.get_name(), policy, self.ddm.z_min,
                )

        if policy == "manual":
            z = self.ddm.z_manual
        elif policy == "bayes_risk":
            z = DriftDiffusionSystem.solve_threshold_bayes_risk(A, c, self.cost_ratio)
        elif policy == "reward_rate":
            z = DriftDiffusionSystem.solve_threshold_reward_rate(
                A, c, self.T0, self.ddm.D_iti, self.ddm.D_penalty
            )
        elif policy == "bellman":
            z = self._bellman_threshold(A, c, q, phi, d)
        else:  # geometric
            z = self._geometric_threshold(A, c, q, phi, d)

        if not np.isfinite(z):
            z = self.ddm.z_manual
        elif z < self.ddm.z_min and not getattr(self, "_warned_z_floored", False):
            # The derived optimum is below the floor, so the boundary being used is
            # z_min, not the policy's answer. Say so once: otherwise the agent looks
            # like it is deciding at random for no visible reason.
            self._warned_z_floored = True
            logger.warning(
                "%s: the '%s' optimal boundary z*=%.4g is below z_min=%.4g, so the "
                "boundary is CLAMPED at the floor and commitments will be fast and "
                "near-random. Drift A=%.4g is small; under terminal_graded rho ~ A^3, so "
                "near-tied targets collapse the threshold. Raise the quality difference, "
                "lower z_min, lower lambda_t, or use threshold_policy 'manual'.",
                self.agent.get_name(), policy, z, self.ddm.z_min, A,
            )
        self.ddm.set_static_threshold(z)

    def _resolve_lambda_t(self, q, d) -> float:
        """Resolve the opportunity cost of time (quality units per second).

        Analytic default (corrected plan Section 4): `lambda_t = E[q]*v/E[d]` — the rate
        the agent would achieve going straight to a typical target. Computed once, at
        evidence onset, from the scenario actually in front of the agent.
        """
        if self.lambda_t is not None:
            return self.lambda_t
        v = self._speed_per_second()
        q_mean = float(np.mean(np.abs(np.asarray(q, dtype=float))))
        d_mean = float(np.mean(np.asarray(d, dtype=float)))
        if d_mean <= 1e-12 or v <= 1e-12 or q_mean <= 0.0:
            self.lambda_t = 1.0
            logger.warning(
                "%s: could not compute the analytic lambda_t default (E[q]=%.4g, "
                "E[d]=%.4g, v=%.4g); falling back to lambda_t = 1.0",
                self.agent.get_name(), q_mean, d_mean, v,
            )
        else:
            self.lambda_t = q_mean * v / d_mean
        logger.info(
            "%s: lambda_t = %.6g quality/s (%s)%s",
            self.agent.get_name(), self.lambda_t, self.lambda_t_mode,
            "  [MVT iteration 0 — feed the measured rate back as lambda_t_measured]"
            if self.lambda_t_mode == "mvt" else "",
        )
        return self.lambda_t

    def _bellman_threshold(self, A: float, c: float, q, phi, d) -> float:
        """Solve the free-boundary problem once, then hand the table to the accumulator.

        `geometric` is the QUASI-STATIC approximation to this: it solves the STATIC
        optimum and re-evaluates it each tick at the current geometry. This solves the
        actual optimal-stopping problem, with the geometry entering as the time-varying
        running cost c_tau(t). The tick loop is unchanged either way -- only the way the
        z array is populated differs.
        """
        from models.bellman_boundary import bellman_boundary, myopic_z

        if self._bellman_solved:
            # Solved for the whole horizon at onset; nothing to recompute per tick.
            return float(self.ddm.boundary(self.ddm.t_evidence))

        cfg = self.bellman_cfg
        delta0 = abs(_wrap_pi(float(phi[0]) - float(phi[1])))
        delta0 = max(delta0, self.delta_min)
        d_mean = float(np.mean(d))
        v = self._speed_per_second()
        # Recover the arena geometry the c_tau derivation assumes: the agent closes on
        # the bisector foot at r0, with the targets +-L/2 either side of it.
        r0 = d_mean * math.cos(delta0 / 2.0)
        half_L = d_mean * math.sin(delta0 / 2.0)

        def c_tau_of_t(t: float) -> float:
            # SAME geometry code as the geometric policy: c_tau = 1 - cos(Delta/2).
            # Re-deriving it here would let the two policies drift apart and make the
            # comparison meaningless (Section 5.1).
            remaining = max(r0 - v * float(t), 1e-6)
            delta_t = 2.0 * math.atan2(half_L, remaining)
            return DriftDiffusionSystem.c_tau_linearised(
                delta_t, predecision_motion=self.predecision_motion
            )

        T_max = cfg.get("T_max")
        T_max = float(T_max) if T_max else (r0 / v if v > 1e-12 else 10.0)
        c_e = float(self.cost_ratio)

        # --- static bounds (§2b): the degenerate flat table, no PDE solve ---
        if self.bellman_static_bound is not None:
            b = self.bellman_static_bound
            self.ddm.set_bellman_table(np.array([0.0, T_max]), np.array([b, b]))
            self._bellman_solved = True
            # z(T_max) = b IS the halt plateau: under terminal 'halt_sprt' the
            # agent parks at the midpoint at arrival and integrates against b.
            self._record_halt_terminal(T_max, b)
            self._bellman_c_tau_of_t = c_tau_of_t
            self._bellman_A, self._bellman_c, self._bellman_c_e = A, c, c_e
            self._bellman_diag = {"z_myopic_onset": float("nan"),
                                  "wall_time_s": 0.0, "table_cache": "static"}
            self._geom_log = {
                "d_1": float(d[0]), "d_2": float(d[1]), "v": v, "delta": delta0,
                "c_tau": float(c_tau_of_t(0.0)),
                "c_tau_eff": float(c_tau_of_t(0.0)),
                "c_err_eff": c_e, "rho": float("nan"), "rho_branch": "static",
                "a_star": float("nan"), "z_star": float(b),
                "z_floor_analytic": float("nan"), "lambda_t_used": None,
                "cost_ratio_used": c_e,
                "geometric_error_mode": self.geometric_error_mode,
                "T_arr": None,
            }
            logger.info(
                "%s: static bound installed as a flat bellman table -- b = %.4g "
                "over T_max = %.2fs (terminal '%s')",
                self.agent.get_name(), b, T_max, self.bellman_terminal,
            )
            return float(b)

        # Optional disk cache (CAMPAIGN_SPEC Section 7.3). Default OFF: with
        # `bellman.table_cache_dir` unset (null) this block is inert and the solve
        # path below is byte-for-byte the historical behaviour. The key is computed
        # HERE, from the same floats the solver is about to receive, so a precompute
        # job that populates the cache by running this model is guaranteed to agree.
        cache_dir = cfg.get("table_cache_dir") or None
        cache_key = None
        if cache_dir:
            from models.bellman_table_cache import load_table, save_table, table_key
            cache_inputs = dict(
                A=A, c=c, c_e=c_e, r0=r0, L=2.0 * half_L, v=v, T_max=T_max,
                N_x=int(cfg.get("N_x", 801)), N_t=int(cfg.get("N_t", 10000)),
                X_max_factor=float(cfg.get("X_max_factor", 4.0)),
                scheme=str(cfg.get("scheme", "crank_nicolson")),
                terminal=self.bellman_terminal,
                halt_cost_rate=self.halt_cost_rate,
            )
            cache_key = table_key(**cache_inputs)
            hit = load_table(cache_dir, cache_key)
            if hit is not None:
                t_grid, z_arr, meta = hit
                self.ddm.set_bellman_table(t_grid, z_arr)
                self._bellman_solved = True
                self._record_halt_terminal(T_max, float(z_arr[-1]))
                # The subset of the solver's diag the rest of the model reads
                # (`z_myopic_onset` for the gap log, `wall_time_s` for the record).
                self._bellman_diag = {
                    "z_myopic_onset": meta["z_myopic_onset"],
                    "wall_time_s": meta["wall_time_s"],
                    "table_cache": "hit",
                }
                self._bellman_c_tau_of_t = c_tau_of_t
                self._bellman_A, self._bellman_c, self._bellman_c_e = A, c, c_e
                logger.info(
                    "%s: bellman boundary LOADED from table cache (key %s) -- "
                    "z(0)=%.4g, T_max=%.2fs, solved elsewhere in %.2fs",
                    self.agent.get_name(), cache_key[:12], z_arr[0], T_max,
                    meta["wall_time_s"],
                )
                self._geom_log = {
                    "d_1": float(d[0]), "d_2": float(d[1]), "v": v, "delta": delta0,
                    "c_tau": float(c_tau_of_t(0.0)), "c_tau_eff": float(c_tau_of_t(0.0)),
                    "c_err_eff": c_e, "rho": float("nan"), "rho_branch": "bellman",
                    "a_star": float("nan"), "z_star": float(z_arr[0]),
                    "z_floor_analytic": float("nan"), "lambda_t_used": None,
                    "cost_ratio_used": c_e,
                    "geometric_error_mode": self.geometric_error_mode,
                    "T_arr": None,
                }
                return float(z_arr[0])

        try:
            t_grid, z_arr, diag = bellman_boundary(
                A, c, c_e, c_tau_of_t, T_max,
                N_x=int(cfg.get("N_x", 801)),
                N_t=int(cfg.get("N_t", 10000)),
                X_max_factor=float(cfg.get("X_max_factor", 4.0)),
                z_myopic_onset=myopic_z(A, c, c_e, c_tau_of_t(0.0)),
                scheme=str(cfg.get("scheme", "crank_nicolson")),
                horizon_check_factor=cfg.get("T_max_check_factor", 1.5),
                terminal=self.bellman_terminal,
                halt_cost_rate=self.halt_cost_rate,
            )
        except ValueError as exc:
            logger.error(
                "%s: the bellman solve failed (%s). Falling back to the geometric "
                "quasi-static policy for this run.", self.agent.get_name(), exc,
            )
            self._bellman_solved = True
            return self._geometric_threshold(A, c, q, phi, d)

        self.ddm.set_bellman_table(t_grid, z_arr)
        self._record_halt_terminal(T_max, float(diag.get("z_halt", 0.0)))
        if cache_dir and cache_key:
            # Persisted only after a SUCCESSFUL solve; the fallback path above never
            # writes. Atomic write, so racing tasks on a cold cache are safe.
            from models.bellman_table_cache import save_table
            hc = diag.get("horizon_check")
            save_table(
                cache_dir, cache_key, t_grid, z_arr,
                # scheme and terminal are strings; the npz meta stores floats only, and
                # both are already part of the key.
                inputs={k: v_ for k, v_ in cache_inputs.items()
                        if k not in ("scheme", "terminal")},
                z_myopic_onset=float(diag["z_myopic_onset"]),
                wall_time_s=float(diag["wall_time_s"]),
                scheme=cache_inputs["scheme"],
                horizon_ok=(None if not isinstance(hc, dict) else hc.get("ok")),
            )
        self._bellman_solved = True
        self._bellman_diag = diag
        self._bellman_c_tau_of_t = c_tau_of_t
        self._bellman_A, self._bellman_c, self._bellman_c_e = A, c, c_e
        logger.info(
            "%s: bellman boundary solved -- z(0)=%.4g vs quasi-static z*(0)=%.4g "
            "(gap %.1f%%), T_max=%.2fs, horizon check %s",
            self.agent.get_name(), z_arr[0], diag["z_myopic_onset"],
            100.0 * (diag["z_myopic_onset"] - z_arr[0]) / max(diag["z_myopic_onset"], 1e-12),
            T_max,
            "ok" if diag.get("horizon_check", {}).get("ok", True) else "FAILED",
        )
        self._geom_log = {
            "d_1": float(d[0]), "d_2": float(d[1]), "v": v, "delta": delta0,
            "c_tau": float(c_tau_of_t(0.0)), "c_tau_eff": float(c_tau_of_t(0.0)),
            "c_err_eff": c_e, "rho": float("nan"), "rho_branch": "bellman",
            "a_star": float("nan"), "z_star": float(z_arr[0]),
            "z_floor_analytic": float("nan"), "lambda_t_used": None,
            "cost_ratio_used": c_e, "geometric_error_mode": self.geometric_error_mode,
            "T_arr": None,
        }
        return float(z_arr[0])

    def _bellman_myopic_now(self) -> Optional[float]:
        """`z_myopic(t)` at the current evidence time, for the Section 8 comparison.

        One Newton solve per tick, which is what makes the Section 10 figure free.
        """
        if not self._bellman_solved or self._bellman_diag is None:
            return None
        from models.bellman_boundary import myopic_z
        return myopic_z(self._bellman_A, self._bellman_c, self._bellman_c_e,
                        self._bellman_c_tau_of_t(self.ddm.t_evidence))

    # ------------------------------------------------------------------
    # Terminal halt (BELLMAN_KNOWN_A_TERMINAL_HALT Section 4.3)
    # ------------------------------------------------------------------
    def _record_halt_terminal(self, T_max: float, z_halt: float) -> None:
        """Stash the installed table's arrival data; no-op under forced choice.

        Called with `z_halt = z_arr[-1]` on a cache hit: under halt_sprt the persisted
        table's final row IS the explicitly recorded `z(T_max) = z_halt`, so the value
        round-trips without widening the cache format.
        """
        if self.bellman_terminal != "halt_sprt":
            return
        from models.bellman_boundary import halt_exit_potential
        self._bellman_T_max = float(T_max)
        self._bellman_z_halt = float(z_halt)
        A = abs(float(self.ddm.A_hat))
        k = 2.0 * A / float(self.ddm.c) ** 2
        self._halt_mean_exit = float(halt_exit_potential(z_halt, k, A))
        if z_halt < self.ddm.z_min:
            logger.warning(
                "%s: z_halt = %.4g sits below z_min = %.4g, so the runtime floor binds "
                "past arrival: the agent will halt against z_min instead and beat "
                "ER(z_halt) at extra cost. Lower z_min if the halt analysis matters.",
                self.agent.get_name(), z_halt, self.ddm.z_min,
            )

    def _update_halt_state(self, state):
        """Track the halt phase each tick; force a commit if the runaway cap is hit.

        The commitment rule stays `|x| >= z(t)` with `z` flat at `z_halt` past arrival
        (the table's explicit final row plus the accumulator's hold-past-the-horizon
        rule). This method OBSERVES that phase — and enforces the episode cap
        `T_max + 10 D(z_halt)`, which exists as a diagnostic, not a truncation: exit is
        a.s. finite with exponential tails, so expect zero hits.

        Returns the state, refreshed if the guard forced a commit so the caller's
        commit latch sees `crossed_this_step` on the same tick.
        """
        self._halted_now = False
        if (
            self.bellman_terminal != "halt_sprt"
            or not self._bellman_solved
            or self._bellman_T_max is None
            or not getattr(self.ddm, "past_horizon", False)
        ):
            return state
        if state.committed is None and not self._halt_event:
            # First uncommitted tick at/past arrival: the halt begins.
            self._halt_event = True
            self._x_at_arrival = float(state.x)
            logger.info(
                "%s: arrived undecided (x = %.4g); halting at the midpoint against "
                "z_halt = %.4g (expected extra deliberation ~%.2fs).",
                self.agent.get_name(), state.x, self._bellman_z_halt,
                self._halt_mean_exit or float("nan"),
            )
        if state.committed is None and self._halt_mean_exit is not None:
            cap = self._bellman_T_max + 10.0 * self._halt_mean_exit
            if state.t_evidence > cap:
                self.ddm.force_commit()
                self._halt_guard_hits += 1
                state = self.ddm.get_state()
                logger.warning(
                    "%s: halt runaway guard fired at t = %.2fs (cap %.2fs): forced a "
                    "commit to sign(x). One hit is a diagnostic; repeated hits mean the "
                    "halt threshold or the noise model is wrong.",
                    self.agent.get_name(), state.t_evidence, cap,
                )
        if self._halt_event and state.committed is not None and self._t_halt_commit is None:
            self._t_halt_commit = float(self.ddm.t_commit or state.t_evidence)
        # Provisional: the motor hold applies past arrival until the decision reaches
        # the motors. _decide_and_actuate finalises it once commit effectiveness for
        # THIS tick is known (the latch for a commit landing this tick runs after us).
        self._halted_now = True
        return state

    def _geometric_threshold(self, A: float, c: float, q, phi, d) -> float:
        """Threshold from the embodied cost geometry, and record its diagnostics.

        Under the default `terminal_graded` the commitment is final, so the error cost is
        the forfeited quality `A/lambda_t` — independent of where the agent is standing.
        Absolute range and speed therefore drop out of rho entirely, and the collapse is
        bounded by a finite floor rather than reaching a deadline.
        """
        delta = abs(_wrap_pi(float(phi[0]) - float(phi[1])))
        d_mean = float(np.mean(d))
        v = self._speed_per_second()
        lambda_t = (
            self._resolve_lambda_t(q, d)
            if self.geometric_error_mode == "terminal_graded" else None
        )

        rho = DriftDiffusionSystem.geometric_rho(
            A, c, delta, d=d_mean, v=v, lambda_t=lambda_t, cost_ratio=self.cost_ratio,
            mode=self.geometric_error_mode, predecision_motion=self.predecision_motion,
            delta_min=self.delta_min,
        )
        z = DriftDiffusionSystem._z_from_rho(rho, A, c)

        # c_tau is now defined for every error mode and is logged explicitly: a constant
        # c_tau column across a run is the direct signature of the collapse being lost.
        c_tau, c_e = DriftDiffusionSystem.geometric_costs(
            A=A, delta=max(delta, self.delta_min), d=d_mean, v=v,
            lambda_t=lambda_t, cost_ratio=self.cost_ratio,
            mode=self.geometric_error_mode,
            predecision_motion=self.predecision_motion,
        )

        z_floor = float("nan")
        t_arr = None
        if self.geometric_error_mode in ("terminal_graded", "terminal_categorical"):
            # The bounded collapse asymptotes here as Delta -> pi. It never reaches zero.
            z_floor = DriftDiffusionSystem.geometric_z_floor(
                A, c, lambda_t, predecision_motion=self.predecision_motion,
                cost_ratio=self.cost_ratio, mode=self.geometric_error_mode,
            )
        elif self.geometric_error_mode == "correctable":
            # Only a correctable error gives the geometry its own deadline (arrival).
            t_arr = d_mean / v if v > 1e-12 else None
            self.ddm.set_geometry(rho0=rho if np.isfinite(rho) else None, T_arr=t_arr)

        a_star = (
            float(DriftDiffusionSystem.solve_a_star(rho)) if np.isfinite(rho) else float("inf")
        )
        self._geom_log = {
            "d_1": float(d[0]), "d_2": float(d[1]), "v": v, "delta": delta,
            "c_tau": float(c_tau), "c_tau_eff": float(c_tau), "c_err_eff": float(c_e),
            "rho": float(rho),
            "rho_branch": "small" if rho < 1e-6 else "large",
            "a_star": a_star,
            "z_star": float(z),
            "z_floor_analytic": float(z_floor),
            "lambda_t_used": lambda_t,
            "cost_ratio_used": (
                float(self.cost_ratio)
                if self.geometric_error_mode == "terminal_categorical" else None
            ),
            "geometric_error_mode": self.geometric_error_mode,
            "T_arr": t_arr,
        }
        if not self._diagnostics_done:
            self._startup_diagnostics(A, c, z, z_floor, a_star)
        return z

    def _startup_diagnostics(self, A, c, z_star, z_floor, a_star) -> None:
        """Warn at evidence onset about parameter combinations that cannot work.

        Each ratio catches a distinct failure mode that would otherwise only be visible
        after a full run (or not at all).
        """
        self._diagnostics_done = True
        dt_sub = (1.0 / self._resolve_agent_tick_rate()) / max(self.ddm.n_sub, 1)
        noise_step = c * math.sqrt(dt_sub)
        drift_step = abs(A) * (1.0 / self._resolve_agent_tick_rate())

        collapse_depth = z_star / z_floor if z_floor and np.isfinite(z_floor) and z_floor > 0 else float("nan")
        noise_ratio = noise_step / z_star if z_star > 0 else float("inf")
        drift_ratio = drift_step / z_star if z_star > 0 else float("inf")

        logger.info(
            "%s geometric startup: z*=%.4g z_floor=%.4g depth=%.2gx | a*=%.3g | "
            "noise_step/z*=%.3g drift_step/z*=%.3g",
            self.agent.get_name(), z_star, z_floor, collapse_depth, a_star,
            noise_ratio, drift_ratio,
        )
        if np.isfinite(collapse_depth) and collapse_depth < 1.5:
            logger.warning(
                "%s: z*(onset)/z_floor = %.2g — the boundary has almost nothing to "
                "collapse through. Check that c_tau varies with Delta.",
                self.agent.get_name(), collapse_depth,
            )
        if noise_ratio > 0.2:
            logger.warning(
                "%s: one noise step is %.0f%% of z*(onset) — the boundary will be "
                "jumped rather than approached, giving near-instant commitment. "
                "Raise n_sub, lower eta_rate, or raise the threshold.",
                self.agent.get_name(), 100.0 * noise_ratio,
            )
        if drift_ratio > 0.2:
            logger.warning(
                "%s: one drift step is %.0f%% of z*(onset) — commitment is essentially "
                "deterministic rather than evidence-driven.",
                self.agent.get_name(), 100.0 * drift_ratio,
            )
        if a_star <= 1.0:
            logger.warning(
                "%s: a*(onset) = %.3g <= 1, so the reference error rate is %.0f%% — "
                "this run is close to a coin flip by construction.",
                self.agent.get_name(), a_star,
                100.0 * DriftDiffusionSystem.er_from_a(a_star),
            )

    def _track_turn(self, state, phi, tick, committed_effective) -> None:
        """Measure the post-commitment turn (Section 5).

        On commitment the heading target steps by ~Delta/2, and the actuator clips the
        error to max_angular_velocity, so the turn takes roughly
        `(Delta/2) / max_angular_velocity` and the trajectory shows a rounded corner
        rather than a kink. Under `midpoint` this is a clean step response, which makes
        it the sharpest available motor-level measurement of the bifurcation.
        """
        if not committed_effective or state is None or state.committed is None:
            return
        # In the egocentric frame the target bearing IS the heading error.
        error_deg = abs(normalize_angle(math.degrees(float(phi[state.committed]))))
        if self._commit_tick is None:
            self._commit_tick = int(tick) if tick is not None else self._tick_count
            self._heading_at_commitment = float(phi[state.committed])
            self._heading_error_at_commitment = error_deg
        if self._turn_duration_ticks is None and error_deg < self.alignment_tolerance_deg:
            now = int(tick) if tick is not None else self._tick_count
            self._turn_duration_ticks = max(0, now - self._commit_tick)

    def _heading_weights(self, beliefs, state, committed_effective) -> np.ndarray:
        """Select the readout weights that determine the HEADING (Task M1).

        All modes go through the one shared `circular_readout` code path; they differ
        only in the weights handed to it:

          committed  -> one-hot, so the heading is exactly the chosen bearing
          midpoint   -> [0.5, 0.5], the fixed angular bisector of the two BEARINGS,
                        independent of the decision variable
          otherwise  -> the belief (P_0, P_1)

        Note the bisector of the bearings is NOT the bearing of the midpoint between the
        two target positions; they coincide only at equal ranges, and it is the bisector
        that the c_tau derivation requires.
        """
        if committed_effective and state.committed is not None:
            one_hot = np.zeros(2, dtype=float)
            one_hot[state.committed] = 1.0
            return one_hot
        if self.predecision_motion == "midpoint":
            return np.array([0.5, 0.5], dtype=float)
        return beliefs

    def _actuate(
        self, heading, magnitude, concentration, committed_effective, phi=None, state=None
    ) -> None:
        """Actuation — identical code path to the ring attractor and the LCA."""
        max_v = float(self.agent.max_absolute_velocity)

        if committed_effective:
            scaling = 1.0
        elif self._halted_now:
            # Terminal halt: arrived at the midpoint undecided, integrating against the
            # flat z_halt. Physically stopped — no translation, no rotation — until the
            # commitment reaches the motors (Section 4.3).
            self._zero_commands()
            return
        elif self.predecision_motion == "stationary":
            # The non-embodied control: no translation AND no rotation, since rotating
            # would change the egocentric bearings and reintroduce the feedback loop.
            self._zero_commands()
            return
        elif self.predecision_motion == "forward":
            scaling = 1.0
        elif self.scaling_mode == "constant":
            # No gating at all: full speed until commitment. Recommended with `midpoint`
            # so that mode does exactly one thing (decouple heading from evidence).
            scaling = float(self.norm_scale)
        elif self.scaling_mode == "magnitude":
            scaling = float(magnitude)
        else:  # concentration
            # Under `midpoint` this equals cos(Delta/2): PURELY GEOMETRIC, carrying no
            # decision information, and driving speed to zero as Delta -> pi so the agent
            # creeps to the midpoint and hovers until noise resolves the tie.
            scaling = float(concentration)

        # Bound the command in every mode (LCA plan Task 0.4): weights here sum to 1 so
        # magnitude <= 1 already, but the clip keeps the guarantee explicit.
        scaling = float(np.clip(scaling, 0.0, 1.0))

        angle_rad = heading
        if self.reference == "allocentric":
            angle_rad = angle_rad - math.radians(self.agent.orientation.z)
        angle_deg = normalize_angle(math.degrees(angle_rad))
        angle_deg = max(
            min(angle_deg, self.agent.max_angular_velocity), -self.agent.max_angular_velocity
        )
        self.agent.angular_velocity_cmd = angle_deg
        self.agent.linear_velocity_cmd = max_v * scaling

    # ------------------------------------------------------------------
    # GUI / logging shim (Tasks 6.4, 6.5)
    # ------------------------------------------------------------------
    def get_spin_system_data(self):
        """Return a payload the GUI and the mean_field pkl logger accept unchanged."""
        if self.ddm is None:
            return None
        state = self._last_state
        entities_copy = copy.deepcopy(self._mf_entities) if self._mf_entities else {
            "targets": [], "guards": []
        }
        target_metadata = copy.deepcopy(entities_copy.get("targets", []))
        committed = None if state is None else state.committed
        committed_id = self.target_ids[committed] if committed is not None else None

        signals = []
        for idx, tid in enumerate(self.target_ids):
            signals.append(
                {
                    "id": tid,
                    "label": tid,
                    "base_quality": float(self._last_q[idx]) if self._last_q.size > idx else 0.0,
                    "modulated_quality": float(self._last_q[idx]) if self._last_q.size > idx else 0.0,
                    "angle": float(self._slot_phi[idx]) if self._slot_phi.size > idx else 0.0,
                    "distance": float(self._slot_d[idx]) if self._slot_d.size > idx else 0.0,
                    "q": float(self._last_q[idx]) if self._last_q.size > idx else 0.0,
                    "q_hat": float(self.ddm.last_q_hat[idx]),
                    "belief": float(self._last_weights[idx]),
                }
            )

        num_groups = self.num_neurons
        data = {
            # model == "mean_field" so the existing GUI panel and mean_field pkl logger
            # accept the payload; decision_model selects the pure-DDM inspector.
            "model": "mean_field",
            "decision_model": "embodied_pure_ddm",
            "states": np.zeros((num_groups, 1), dtype=float),
            "angles": (np.repeat(self.group_angles, 1), num_groups, 1),
            "external_field": np.zeros(num_groups, dtype=float),
            "avg_direction_of_activity": self._last_heading,
            "mean_field_state": np.zeros(num_groups, dtype=float),
            "mean_field_perception": None if self.perception is None else self.perception.copy(),
            "mean_field_perception_raw": None if self.perception is None else self.perception.copy(),
            "mean_field_sensory_map": np.zeros(num_groups, dtype=float),
            "mean_field_target_metadata": target_metadata,
            "mean_field_modulated_target_qualities": self._last_q.copy(),
            "mean_field_target_signals": signals,
            "mean_field_sensory_time": float(getattr(self.evidence, "sensory_time", 0.0)),
            "mean_field_entities": entities_copy,
            "mean_field_norm": float(self._last_concentration),
            "mean_field_beta": 0.0,
            "mean_field_lambda1": None,
            "mean_field_omega": None,
            "channel": self._active_perception_channel,
            # --- pure-DDM per-tick fields (Task 6.5) ---
            "pure_ddm_x": 0.0 if state is None else float(state.x),
            "pure_ddm_z": 0.0 if state is None else float(state.z),
            "pure_ddm_t_evidence": 0.0 if state is None else float(state.t_evidence),
            "pure_ddm_A_hat": float(self.ddm.A_hat),
            "pure_ddm_A_true": float(self.ddm.A_true),
            "pure_ddm_A_source": self.ddm.A_source,
            "pure_ddm_A_lognormal_s": float(self.A_lognormal_s),
            "pure_ddm_A_lognormal_debias": bool(self.A_lognormal_debias),
            "pure_ddm_A_hat_over_A_true": (
                float(self.ddm.A_hat / self.ddm.A_true) if self.ddm.A_true > 0 else None
            ),
            "pure_ddm_c": float(self.ddm.c),
            "pure_ddm_p1": float(self._last_weights[0]),
            "pure_ddm_q": self._last_q.copy(),
            "pure_ddm_q_hat": self.ddm.last_q_hat.copy(),
            "pure_ddm_labels": list(self.target_ids),
            "pure_ddm_committed": committed,
            "pure_ddm_committed_id": committed_id,
            "pure_ddm_commit_effective": self._commit_is_effective(state) if state else False,
            # --- per-trial fields ---
            "pure_ddm_rt": self.ddm.rt,
            "pure_ddm_z_star_onset": self.ddm.z_star_at_onset,
            "pure_ddm_threshold_policy": self.threshold_policy,
            "pure_ddm_boundary_mode": self.boundary_mode,
            "pure_ddm_collapse_form": self.ddm.collapse_form,
            "pure_ddm_predecision_motion": self.predecision_motion,
            # --- midpoint / motor readout diagnostics (Task M4) ---
            "pure_ddm_R_geom": float(self._last_r_geom),
            "pure_ddm_heading_mode": self.predecision_motion,
            "pure_ddm_scaling_mode": self.scaling_mode,
            "pure_ddm_bisector_guard_fired": bool(self._bisector_guard_fired),
            "pure_ddm_turn_duration_ticks": self._turn_duration_ticks,
            "pure_ddm_turn_duration_s": (
                None if self._turn_duration_ticks is None
                else self._turn_duration_ticks / self._resolve_agent_tick_rate()
            ),
            "pure_ddm_heading_at_commitment": self._heading_at_commitment,
            "pure_ddm_heading_error_at_commitment": self._heading_error_at_commitment,
            "pure_ddm_changed_mind": bool(self.ddm.changed_mind),
            "pure_ddm_log_odds_at_commit": self._log_odds_at_commit,
            # --- post-commitment flexibility (per-tick + per-trial) ---
            "pure_ddm_flexibility": bool(self.flexibility),
            "pure_ddm_post_commit_accumulation": self.post_commit_accumulation,
            "pure_ddm_x_over_z": (
                None if self._last_state is None else float(self._last_state.x_over_z)
            ),
            "pure_ddm_n_commits": int(self.ddm.n_commits),
            "pure_ddm_n_releases": int(self.ddm.n_releases),
            # n_reversals is the primary flexibility measure: it counts commitments to a
            # DIFFERENT target, so boundary chatter (Section 3.2) cannot inflate it.
            "pure_ddm_n_reversals": int(self.ddm.n_reversals),
            "pure_ddm_t_first_commit": self.ddm.t_first_commit,
            "pure_ddm_final_target": self._target_id(self.ddm.last_committed_target),
            "pure_ddm_t_swap": self._swap_applied_at,
            "pure_ddm_x_at_swap": self._x_at_swap,
            "pure_ddm_dwell_before_swap": self._dwell_before_swap,
            "pure_ddm_t_release": self._t_release_after_swap,
            "pure_ddm_t_recommit": self._t_recommit_after_swap,
            "pure_ddm_release_latency": (
                None if (self._t_release_after_swap is None or self._swap_applied_at is None)
                else self._t_release_after_swap - self._swap_applied_at
            ),
            "pure_ddm_recommit_latency": (
                None if (self._t_recommit_after_swap is None
                         or self._t_release_after_swap is None)
                else self._t_recommit_after_swap - self._t_release_after_swap
            ),
            "pure_ddm_total_path_length": float(self._total_path_length),
            "pure_ddm_arrived_before_reversal": (
                self._swap_applied_at is not None and self._t_recommit_after_swap is None
            ),
            "pure_ddm_transitions": list(self._transitions),
            # --- bellman policy (FEATURE_BELLMAN_POLICY Section 8) ---
            # z_myopic is computed alongside unconditionally under this policy: it costs
            # one Newton solve per tick and makes the Section 10 comparison figure free.
            "pure_ddm_z_bellman": (
                float(self._last_state.z) if (self._bellman_solved
                                              and self._last_state is not None) else None
            ),
            "pure_ddm_z_myopic": self._bellman_myopic_now(),
            "pure_ddm_z_gap": (
                None if (zm := self._bellman_myopic_now()) is None
                or self._last_state is None else float(zm - self._last_state.z)
            ),
            "pure_ddm_past_horizon": bool(getattr(self.ddm, "past_horizon", False)),
            "pure_ddm_bellman_solve_s": (
                None if self._bellman_diag is None
                else float(self._bellman_diag["wall_time_s"])
            ),
            # --- terminal halt (BELLMAN_KNOWN_A_TERMINAL_HALT Section 4.3) ---
            "pure_ddm_bellman_terminal": (
                self.bellman_terminal if self.threshold_policy == "bellman" else None
            ),
            "pure_ddm_z_halt": self._bellman_z_halt,
            "pure_ddm_bellman_T_max": self._bellman_T_max,
            "pure_ddm_halted": bool(self._halted_now),
            "pure_ddm_halt_event": bool(self._halt_event),
            "pure_ddm_x_at_arrival": self._x_at_arrival,
            # Extra deliberation bought by the halt: commit time past arrival once
            # committed, the running excess while still halted, None before arrival.
            "pure_ddm_halt_duration": (
                None if not self._halt_event or self._bellman_T_max is None
                else (self._t_halt_commit - self._bellman_T_max)
                if self._t_halt_commit is not None
                else float(self.ddm.t_evidence) - self._bellman_T_max
            ),
            "pure_ddm_halt_guard_hits": int(self._halt_guard_hits),
            "boundary_policy_incoherent": bool(self.boundary_policy_incoherent),
        }
        # Stamp which sensory protocol produced this record, so no result is ambiguous.
        data.update(self.percept_stream_record())
        if self.threshold_policy == "geometric" and self._geom_log:
            data.update({f"pure_ddm_{k}": v for k, v in self._geom_log.items()})
        # Drain new bifurcation events detected this tick (Path A IPC: events flow through
        # per-tick spin data from the agent process to the Arena, where the
        # post-bifurcation swap trigger scans for them).
        new_bif = list(self.bifurcation_detector.events) if hasattr(self, 'bifurcation_detector') else []
        if new_bif:
            self.bifurcation_detector.events.clear()
        data["new_bifurcation_events"] = new_bif
        return data


class _PureDDMBifShim:
    """Duck-typed `MeanFieldSystem` stand-in so `BifurcationDetector` runs unchanged.

    `behavioral` mode (the mode used for all three models, so switch counts and
    alignment events are measured by identical code) ignores `mf` entirely; the detector
    still calls `compute_lambda1(mf)` for logging, so this exposes a 1x1 Jacobian whose
    eigenvalue is 0 — the pure DDM is a perfect integrator, sitting exactly on the LCA's
    critical point `beta_inh == lambda_leak` where `lambda_1 = 0`.
    """

    def __init__(self, ddm: DriftDiffusionSystem):
        self._ddm = ddm
        self.M = np.ones((1, 1), dtype=float)
        self.neural_ring = np.zeros(1, dtype=float)
        self.b = np.zeros(1, dtype=float)
        self.u = 1.0
        self.beta = 0.0
        self.num_neurons = 1
        self.adapt_ring = np.zeros(1, dtype=float)
        self.g_adapt = 0.0
        self._step_count = 0


register_movement_model(
    "embodied_pure_ddm", lambda agent: EmbodiedPureDDMMovementModel(agent)
)
