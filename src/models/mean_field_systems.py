# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
#  You may use, modify, and redistribute this file according to the terms of the
#  license. Attribution is required if this code is used in other works.
# ------------------------------------------------------------------------------

"""Mean-field spiking ring attractor model."""

from __future__ import annotations
import logging
import math
from typing import Iterable, Mapping
from numba import njit, prange
import numpy as np

from models.readout import circular_readout

#====================== Helper Functions ======================
# === Helper: find angular distance between two angles ===
def _delta_angle(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    """Return smallest signed difference between angles (radians)."""
    return (a1 - a2 + np.pi) % (2 * np.pi) - np.pi

# === Helper: Compute center of mass of neural activation ===
def compute_center_of_mass(z, theta_i):
    sin_sum = np.sum(z * np.sin(theta_i))
    cos_sum = np.sum(z * np.cos(theta_i))
    return np.arctan2(sin_sum, cos_sum)
# The thresholded circular readout formerly named `compute_command` now lives in
# models.readout.circular_readout so that the embodied accumulator can call the exact
# same code path. `circular_readout(z, theta, threshold=g_threshold)` is byte-identical
# to the old `compute_command(z, theta, g_threshold=g_threshold)`.

logger = logging.getLogger("sim.mean_field")
logger.setLevel(logging.DEBUG)

# Sentinel for "no tick has been seen yet". A plain object(), so it can never compare
# equal to a real tick index (including a negative pre-run one) and the first
# compute_sensory_map call of a trial always draws fresh sensory noise.
_NO_TICK = object()

class MeanFieldSystem:
    """
    Phenomenological spiking ring attractor (mean-field version).

    Core dynamics:
        z_dot = -z + tanh((u0 - s) * M @ z + b - beta) - tanh(-beta) + noise
        tau * s_dot = -s + k * ||z||^4 => necessary for spiking behavior. right now not included
    """

    def __init__(
        self,
        num_neurons: int,
        u: float = 6.0,
        beta: float = 1.0,
        v: float = 0.5,
        kappa: float = 20.0,
        spatial_decay: float = 2.0,
        num_targets: int = 0,
        num_guards: int = 0,
        target_qualities: Iterable[float] | None = None,
        guard_qualities: Iterable[float] | None = None,
        target_quality_modulations: Mapping[str, Mapping[str, float]] | None = None,
        sigma: float = 0.01,
        sigma_s: float = 0.0,
        dt: float = 0.1,
        integration_time: float = 50.0,
        sensory_time_mode: str = "world_time",
        sensory_dt: float | None = None,
        rng: np.random.Generator | None = None,
        noise_rng: np.random.Generator | None = None,
        # SFA parameters
        g_adapt: float = 0.0, # set > 0 to enable SFA
        tau_adapt: float = 0.0, # adaptation time constant
        # Thresholding parameters
        g_threshold: float = 0.6,
        use_thresholding: bool = True,
        # Readout scaling: which order parameter drives forward speed.
        #   "concentration" (default) -> angular coherence in [0, 1], bounded.
        #   "magnitude"               -> raw readout magnitude (legacy use_thresholding=True).
        #   "norm"                    -> L2 norm of z (legacy use_thresholding=False).
        scaling_mode: str = "concentration",
    ):
        """
        Initialize the mean-field system.

        Args:
            num_neurons: Number of units in the ring.
            u: Baseline coupling strength.
            beta: Implicit self-excitation offset.
            v: Shape parameter for the interaction kernel.
            kappa: Concentration for sensory von Mises inputs.
            spatial_decay: Spatial decay rate for guard influence.
            sigma: Noise standard deviation (added to z dynamics, scaled by sqrt(n)).
            sigma_s: Sensory noise std on the PERCEIVED TARGET QUALITIES. One draw per
                target per arena tick, added to the (already modulated) quality before
                the von Mises scatter onto the ring: q_hat_i = q_i + sigma_s * xi_i(t).
                It therefore lives in quality units, like the DDM's own sigma_s and the
                shared percept stream's `white_rate`, and NOT in ring-field units.
                Note this is per-tick noise that averages out over a trial, not the
                frozen per-neuron bias on `b` it replaced, and not the same thing as
                AccumulatorSystem.sigma_s, which is still a frozen per-slot bias.
                Guards are unaffected: only target qualities are noised.
            dt: Integration time step.
            initial_state: Optional initial state vector z.
            external_input: Optional initial external input vector b.
            rng: Generator for the sensory quality noise (sigma_s), drawn once per
                arena tick. A sequential stream, so the realisation depends on how many
                ticks (and how many targets per tick) have been consumed.
            noise_rng: Generator for the internal neural noise (sigma), drawn every
                Euler sub-step. Deliberately a SEPARATE stream from `rng`: the sensory
                draws must not shift when the number of integration steps changes, and
                neither may depend on the other's consumption. Both should be seeded
                from the arena seed by the caller (MeanFieldMovementModel does this);
                the unseeded defaults exist only for bare unit-test construction and
                make a run irreproducible.
            g_adapt: Adaptation strength (set > 0 to enable spike-frequency adaptation).
            tau_adapt: Adaptation time constant (used when g_adapt > 0).
            g_threshold: Threshold parameter for thresholding.
            use_thresholding: Whether to use thresholding.
        """
        if num_neurons <= 0:
            raise ValueError("num_neurons must be positive")
        self.num_neurons = int(num_neurons)
        self.u = float(u)
        self.beta = float(beta)
        self.v = float(v)
        self.kappa = float(kappa)
        self.spatial_decay = spatial_decay
        self.sigma = float(sigma)
        self.sigma_s = float(sigma_s)
        self.dt = float(dt)
        self.integration_time = float(integration_time)
        self.sensory_time_mode = self._normalize_sensory_time_mode(sensory_time_mode)
        default_sensory_dt = (
            self.integration_time
            if self.sensory_time_mode == "integration_time"
            else 1.0
        )
        self.sensory_dt = float(default_sensory_dt if sensory_dt is None else sensory_dt)
        if self.sensory_dt < 0.0:
            raise ValueError("sensory_dt must be non-negative")
        if rng is None and self.sigma_s > 0.0:
            logger.warning(
                "MeanFieldSystem: sigma_s=%.4g but no `rng` was supplied, so the "
                "sensory quality noise comes from an UNSEEDED generator and this run "
                "cannot be reproduced from its config. Pass a generator derived from "
                "the arena seed (MeanFieldMovementModel does this).",
                self.sigma_s,
            )
        self.rng = rng or np.random.default_rng()
        self.noise_rng = noise_rng or np.random.default_rng()

        self.theta = np.linspace(-np.pi, np.pi, self.num_neurons, endpoint=False)

        self.num_targets = int(num_targets)
        self.num_guards = int(num_guards)
        self.target_qualities: np.ndarray | None = (
            None if target_qualities is None
            else np.asarray(target_qualities, dtype=float).reshape(-1).copy()
        )
        self.guard_qualities: np.ndarray | None = (
            None if guard_qualities is None
            else np.asarray(guard_qualities, dtype=float).reshape(-1).copy()
        )
        self.target_quality_modulations = self._normalize_target_quality_modulations(
            target_quality_modulations
        )
        self.sensory_time = 0.0

        self.M = self.compute_interaction_kernel()

        self.neural_ring = (
            np.zeros(self.num_neurons, dtype=float)
        )
        if self.neural_ring.shape[0] != self.num_neurons:
            raise ValueError("initial_state dimension must match num_neurons")

        self.b = (
            np.zeros(self.num_neurons, dtype=float)
        )

        if self.b.shape[0] != self.num_neurons:
            raise ValueError("external_input dimension must match num_neurons")

        # Per-tick sensory quality noise. Redrawn by compute_sensory_map() whenever it
        # is handed a tick index different from the one the cached vector belongs to, so
        # every inner step of one arena tick (steps_per_tick > 1) sees ONE realisation.
        # `_noise_tick` starts at a sentinel no caller can pass, so the first call of a
        # trial always draws.
        self._q_noise: np.ndarray = np.array([], dtype=float)
        self._noise_tick: object = _NO_TICK

        self.g_adapt = float(g_adapt)
        self.tau_adapt = float(tau_adapt)
        self.g_threshold = float(g_threshold)
        self.use_thresholding = bool(use_thresholding)
        self.scaling_mode = str(scaling_mode)
        # Readout order parameters, refreshed every compute_dynamics() call.
        self.last_magnitude = 0.0
        self.last_concentration = 0.0
        self.last_l2 = 0.0
        if self.g_adapt > 0.0 and self.tau_adapt <= 0.0:
            raise ValueError("tau_adapt must be positive when g_adapt > 0")
        
        self.adapt_ring = np.zeros(self.num_neurons, dtype=float)
        self.last_target_ids: list[str] = []
        self.last_target_base_qualities = np.array([], dtype=float)
        self.last_modulated_target_qualities = np.array([], dtype=float)
        self.last_noisy_target_qualities = np.array([], dtype=float)
        self._step_count: int = 0
        

    @staticmethod
    def _normalize_sensory_time_mode(mode: str | None) -> str:
        """Normalize how the modulation clock advances between updates."""
        normalized = str(mode or "world_time").strip().lower()
        if normalized in {"world", "world_time", "simulation", "simulation_time"}:
            return "world_time"
        if normalized in {"integration", "integration_time", "legacy"}:
            return "integration_time"
        raise ValueError(
            "sensory_time_mode must be 'world_time' or 'integration_time'"
        )


    def _normalize_target_quality_modulations(
        self,
        target_quality_modulations: Mapping[str, Mapping[str, float]] | None,
    ) -> dict[str, dict[str, float]]:
        """Normalize per-target sinusoidal modulation parameters."""
        if not target_quality_modulations:
            return {}

        normalized: dict[str, dict[str, float]] = {}
        for target_id, params in target_quality_modulations.items():
            if not isinstance(params, Mapping):
                raise ValueError(
                    f"target_quality_modulations['{target_id}'] must be a mapping"
                )
            normalized[str(target_id)] = {
                "epsilon": float(params.get("epsilon", 0.0)),
                "omega": float(params.get("omega", 0.0)),
                "psi": float(params.get("psi", 0.0)),
            }
        return normalized

    def _apply_target_quality_modulation(
        self,
        target_ids: Iterable[str] | None,
        target_qualities: np.ndarray,
    ) -> np.ndarray:
        """Return target qualities after applying sinusoidal per-target modulation."""
        if not self.target_quality_modulations or target_ids is None:
            return target_qualities

        target_id_list = [str(target_id) for target_id in target_ids]
        if len(target_id_list) != target_qualities.shape[0]:
            raise ValueError(
                "target_ids and target_qualities must have the same length: "
                f"{len(target_id_list)} vs {target_qualities.shape[0]}"
            )

        modulated = target_qualities.copy()
        for idx, target_id in enumerate(target_id_list):
            params = self.target_quality_modulations.get(target_id)
            if params is None:
                continue
            modulation = 1.0 + params["epsilon"] * np.sin(
                params["omega"] * self.sensory_time + params["psi"]
            )
            modulated[idx] *= modulation
        return modulated


    def compute_interaction_kernel(self) -> np.ndarray:
        """Compute cosine-based interaction kernel."""
        theta_col = self.theta[:, np.newaxis] # shape (num_neurons, 1)
        theta_row = self.theta[np.newaxis, :] # shape (1, num_neurons)
        delta = np.abs(_delta_angle(theta_col, theta_row)) # pairwise delta_ij matrix: shape (num_neurons, num_neurons)
        return (1.0 / self.num_neurons ) * np.cos(np.pi * (delta / np.pi) ** self.v)

    def _advance_sensory_time(self) -> None:
        """Advance the modulation clock after one mean-field update."""
        if self.sensory_time_mode == "integration_time":
            self.sensory_time += self.integration_time
            return
        self.sensory_time += self.sensory_dt

    def _sensory_quality_noise(self, num_targets: int, tick: int | None) -> np.ndarray:
        """Return this tick's sensory quality noise, one deviate per target.

        Drawn from `self.rng`, a sequential stream seeded from the arena `random_seed`
        by the caller. One draw per target per ARENA tick: the vector is cached against
        the tick it was drawn for, so all `steps_per_tick` inner steps of a tick - and
        all the Euler sub-steps inside each of those - share one realisation. Being a
        sequential stream, the realisation depends on how many ticks and how many
        targets have been consumed before it; changing `steps_per_tick` does not shift
        it, but changing the number of perceived targets does.

        `tick=None` means "no arena clock" and redraws on every call.
        """
        if self.sigma_s <= 0.0 or num_targets <= 0:
            self._q_noise = np.zeros(max(num_targets, 0), dtype=float)
            self._noise_tick = _NO_TICK if tick is None else tick
            return self._q_noise
        stale = (
            tick is None
            or tick != self._noise_tick
            or self._q_noise.shape[0] != num_targets
        )
        if stale:
            self._q_noise = self.rng.standard_normal(num_targets) * self.sigma_s
            self._noise_tick = _NO_TICK if tick is None else tick
        return self._q_noise

    def compute_sensory_map(
        self,
        num_targets: int,
        num_guards: int,
        target_ids: Iterable[str] | None,
        target_angles: Iterable[float],
        target_qualities: Iterable[float],
        guard_angles: Iterable[float] | None = None,
        guard_qualities: Iterable[float] | None = None,
        guard_decay_rate: float | None = None,
        guard_distances: Iterable[float] | None = None,
        tick: int | None = None,
    ) -> np.ndarray:
        """
        Compute sensory input b using von Mises bumps for targets and optional guard inhibition.

        `tick` is the ARENA tick index and is what the sigma_s sensory noise is keyed to:
        the draw is reused for every call carrying the same tick, so a model running
        `steps_per_tick > 1` integrates one realisation rather than a fresh one per inner
        step. Pass None (the default) to redraw on every call, which is what a bare
        unit-test harness with no arena clock wants.
        """
        b = np.zeros(self.num_neurons, dtype=float)
        self.last_target_ids = []
        self.last_target_base_qualities = np.array([], dtype=float)
        self.last_modulated_target_qualities = np.array([], dtype=float)
        self.last_noisy_target_qualities = np.array([], dtype=float)
        
        if num_targets > 0 and target_angles is not None and target_qualities is not None:
            target_id_list = [] if target_ids is None else [str(target_id) for target_id in target_ids]
            target_angles = np.asarray(target_angles, dtype=float).reshape(1, -1)
            target_qualities = np.asarray(target_qualities, dtype=float).reshape(-1)
            modulated_target_qualities = self._apply_target_quality_modulation(
                target_ids=target_id_list if target_ids is not None else None,
                target_qualities=target_qualities,
            )
            self.last_target_ids = target_id_list
            self.last_target_base_qualities = target_qualities.copy()
            self.last_modulated_target_qualities = modulated_target_qualities.copy()
            # Sensory noise on the PERCEIVED QUALITY, downstream of the sinusoidal
            # modulation, so sigma_s is expressed in quality units and reaches the ring
            # through the same von Mises kernel the clean quality does.
            noisy_target_qualities = (
                modulated_target_qualities
                + self._sensory_quality_noise(modulated_target_qualities.shape[0], tick)
            )
            self.last_noisy_target_qualities = noisy_target_qualities.copy()
            delta_targets = _delta_angle(self.theta[:, None], target_angles)
            vm_targets = np.exp(self.kappa * (np.cos(delta_targets) - 1.0))
            b = vm_targets @ noisy_target_qualities
            logger.debug(
                    "Target angles: %s",
                    np.array2string(
                        np.asarray(target_angles, dtype=float).reshape(-1),
                        precision=6,
                        separator=", ",
                        max_line_width=1000,
                    ),
                )
            if logger.isEnabledFor(logging.DEBUG):
                # Clean and noisy side by side: the clean vector is the MEAN of this
                # tick's sensory draw, the noisy one is what actually reaches the ring.
                fmt = lambda values: np.array2string(
                    np.asarray(values, dtype=float).reshape(-1),
                    precision=6,
                    separator=", ",
                    max_line_width=1000,
                )
                logger.debug(
                    "Target qualities (sigma_s=%.4g) ids=%s | mean (pre-noise)=%s | "
                    "noisy (on ring)=%s | noise=%s",
                    self.sigma_s,
                    target_id_list,
                    fmt(modulated_target_qualities),
                    fmt(noisy_target_qualities),
                    fmt(noisy_target_qualities - modulated_target_qualities),
                )

        if num_guards > 0 and guard_angles is not None and guard_qualities is not None and guard_distances is not None:
            guard_angles = np.asarray(guard_angles, dtype=float).reshape(1, -1)
            guard_qualities = np.asarray(guard_qualities, dtype=float).reshape(-1)
            guard_distances = np.asarray(guard_distances, dtype=float).reshape(-1)
            assert guard_qualities.shape == guard_distances.shape, \
            f"guard_qualities and guard_distances must have same length: {guard_qualities.shape} vs {guard_distances.shape}"
            delta_guards = _delta_angle(self.theta[:, None], guard_angles)
            vm_guards = np.exp(self.kappa * (np.cos(delta_guards) - 1.0))
            decay = 0.0 if guard_decay_rate is None else guard_decay_rate
            scaled = guard_qualities * np.exp(-decay * guard_distances)
            b += vm_guards @ scaled
            logger.debug(
                    "Guard angles: %s",
                    np.array2string(
                        np.asarray(guard_angles, dtype=float).reshape(-1),
                        precision=6,
                        separator=", ",
                        max_line_width=1000,
                    ),
                )

        b /= math.sqrt(self.num_neurons)
        self.b = b
        return self.b

    def reset(
        self,
        z: np.ndarray | None = None,
        a: np.ndarray | None = None,
        external_input: np.ndarray | None = None,
    ):
        """Reset internal state."""
        if z is not None:
            z = np.asarray(z, dtype=float).reshape(-1)
            if z.shape[0] != self.num_neurons:
                raise ValueError("Reset state dimension must match num_neurons")
            self.neural_ring = z
        else:
            self.neural_ring = np.zeros(self.num_neurons, dtype=float)


        if external_input is not None:
            external_input = np.asarray(external_input, dtype=float).reshape(-1)
            if external_input.shape[0] != self.num_neurons:
                raise ValueError("Reset input dimension must match num_neurons")
            self.b = external_input
        else:
            self.b = np.zeros_like(self.neural_ring)

        """Update external input vector b."""
        external_input = np.asarray(external_input, dtype=float).reshape(-1)
        if external_input.shape[0] != self.num_neurons:
            raise ValueError("external_input dimension must match num_neurons")
        self.b = external_input

        if a is not None:
            self.adapt_ring = np.asarray(a, dtype=float).reshape(-1)
        else:
            self.adapt_ring = np.zeros(self.num_neurons, dtype=float)
        self.sensory_time = 0.0
        self.last_target_ids = []
        self.last_target_base_qualities = np.array([], dtype=float)
        self.last_modulated_target_qualities = np.array([], dtype=float)
        self.last_noisy_target_qualities = np.array([], dtype=float)
        # Nothing to redraw: the sensory noise is per-tick, so it is enough to forget
        # which tick the cached vector belonged to. The generator is deliberately NOT
        # reseeded here - a trial continues its own stream.
        self._q_noise = np.array([], dtype=float)
        self._noise_tick = _NO_TICK

    @staticmethod
    def euler_integrate_sfa(y0, t_eval, u, b, M, beta, n, sigma, g_adapt, tau_adapt, randn_like_func):
        """
        Euler integration for stacked state y = [z; a] where:
        z_dot = -z + tanh(u M z + b - beta - g_adapt*a) - tanh(-beta) + noise
        a_dot = (-a + z)/tau_adapt
        """
        dt = t_eval[1] - t_eval[0]
        y = np.zeros((len(t_eval), len(y0)))
        y[0] = y0

        N = n  # number of neurons
        """for i in range(1, len(t_eval)):
            z_prev = y[i-1, :N]
            a_prev = y[i-1, N:]

            # Euler-Maruyama: noise increment scales as sqrt(dt), applied OUTSIDE the
            # drift so it is not multiplied by dt again (see Task 0.1). The adaptation
            # equation `a` is deterministic and takes no noise term.
            noise = randn_like_func(z_prev, sigma, 1.0 / np.sqrt(N)) * np.sqrt(dt)

            drive = u * (M @ z_prev) + b - beta - (g_adapt * a_prev)
            z_dot = -z_prev + np.tanh(drive) - np.tanh(-beta)

            a_dot = (-a_prev + z_prev) / tau_adapt

            y[i, :N] = z_prev + dt * z_dot + noise
            y[i, N:] = a_prev + dt * a_dot

        return y"""
        for i in range(1, len(t_eval)):
            z_prev = y[i-1, :N]
            a_prev = y[i-1, N:]

            noise = randn_like_func(z_prev, sigma * np.sqrt(dt), 1.0 / np.sqrt(N))

            drive = u * (M @ z_prev) + b - beta - (g_adapt * a_prev)
            z_dot = -z_prev + np.tanh(drive) - np.tanh(-beta) + noise

            a_dot = (-a_prev + z_prev) / tau_adapt

            y[i, :N] = z_prev + dt * z_dot
            y[i, N:] = a_prev + dt * a_dot

        return y
    
    @staticmethod
    def euler_integrate(y0, t_eval, u, b, M, beta, n, sigma, randn_like_func):
        dt = t_eval[1] - t_eval[0]
        y = np.zeros((len(t_eval), len(y0)))
        y[0] = y0
        """for i in range(1, len(t_eval)):
            # Euler-Maruyama: noise increment ~ sigma*sqrt(dt), applied OUTSIDE the drift
            # so it is not multiplied by dt a second time (see Task 0.1).
            noise = randn_like_func(y[i-1], sigma, 1.0 / np.sqrt(n)) * np.sqrt(dt)
            dydt = -y[i-1] + np.tanh(u * M @ y[i-1] + b - beta) - np.tanh(-beta)
            y[i] = y[i-1] + dt * dydt + noise
        return y"""
        for i in range(1, len(t_eval)):
            noise = randn_like_func(y[i-1], sigma * np.sqrt(dt), 1.0 / np.sqrt(n))
            dydt = -y[i-1] + np.tanh(u * M @ y[i-1] + b - beta) - np.tanh(-beta) + noise
            y[i] = y[i-1] + dt * dydt
        return y
    
    def randn_like(self, y, sigma, inv_sqrt_n):
        """Return the Euler-Maruyama noise increment for one sub-step.

        Draws from `self.noise_rng`, NOT from the global `np.random`. The global module
        RNG was the source here until the arena-seeding fix: it made the ring
        attractor's internal noise impossible to reproduce from a config, since nothing
        in the simulator ever seeded it.
        """
        return self.noise_rng.normal(0.0, sigma * inv_sqrt_n, size=y.shape)
    

    def compute_dynamics(self, total_time: float | None = None, dt: float | None = None):
        total_time = self.integration_time if total_time is None else float(total_time)
        dt = self.dt if dt is None else float(dt)
        if total_time <= 0.0:
            raise ValueError("total_time must be positive")
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        # Always take at least one Euler step and keep the configured dt.
        n_steps = max(1, int(np.ceil(total_time / dt)))
        t_eval = np.arange(n_steps + 1, dtype=float) * dt

        use_adaptation = self.g_adapt > 0.0 and self.tau_adapt > 0.0
        if use_adaptation:
            # Stack initial condition y0 = [z0; a0]
            y0 = np.concatenate([self.neural_ring.copy(), self.adapt_ring.copy()])
            result = MeanFieldSystem.euler_integrate_sfa(
                y0, t_eval,
                self.u, self.b, self.M, self.beta,
                self.num_neurons, self.sigma,
                self.g_adapt, self.tau_adapt,
                self.randn_like,
            )
            z_traj = result[:, :self.num_neurons]
            a_traj = result[:, self.num_neurons:]
        else:
            y0 = self.neural_ring.copy()
            z_traj = MeanFieldSystem.euler_integrate(
                y0, t_eval,
                self.u, self.b, self.M, self.beta,
                self.num_neurons, self.sigma,
                self.randn_like,
            )
            a_traj = None

        times = t_eval

        # Bump heading trajectory. `use_thresholding` still selects how the heading is
        # read (raw circular mean vs thresholded circular readout), preserving existing
        # bump semantics. The three order parameters below are computed regardless and
        # exposed for the configurable scaling_mode readout (Task 0.4): `concentration`
        # is bounded in [0, 1] and is the default forward-speed driver, while
        # `magnitude` reproduces the legacy (unbounded) behaviour on request.
        if not self.use_thresholding:
            bump_positions = np.array([compute_center_of_mass(z_t, self.theta) for z_t in z_traj])
        else:
            readout = np.array([circular_readout(z_t, self.theta, threshold=self.g_threshold) for z_t in z_traj])
            bump_positions = readout[:, 0]

        _, final_magnitude, final_concentration = circular_readout(
            z_traj[-1], self.theta, threshold=self.g_threshold
        )
        self.last_magnitude = float(final_magnitude)
        self.last_concentration = float(final_concentration)
        self.last_l2 = float(np.linalg.norm(z_traj[-1]))

        # Backward-compatible 3rd return value: readout magnitude when thresholding
        # (legacy use_thresholding=True), L2 norm otherwise (legacy False).
        final_norm = self.last_magnitude if self.use_thresholding else self.last_l2

        # Update internal states
        self.neural_ring = z_traj[-1]
        if a_traj is not None:
            self.adapt_ring = a_traj[-1]
        else:
            self.adapt_ring = np.zeros_like(self.adapt_ring)

        return times, bump_positions, final_norm

    # Integrate timesteps to simulate the neural field dynamics
    """def compute_dynamics(self, total_time=50, dt=0.1):
        t_eval = np.arange(0, total_time, dt)
        y0 = self.neural_ring.copy()
        result = MeanFieldSystem.euler_integrate(y0, t_eval, self.u, self.b, self.M, self.beta, self.num_neurons, self.sigma, MeanFieldSystem.randn_like)
        times = t_eval
        # Compute CoM of bump activity at each time
        bump_positions = np.array([compute_center_of_mass(z_t, self.theta) for z_t in result])
        final_norm = np.linalg.norm(result[-1])
        self.neural_ring = result[-1]  # Update neural field state
        return times, bump_positions, final_norm"""


    def step(
        self,
        target_ids: Iterable[str] | None = None,
        target_angles: Iterable[float] | None = None,
        target_qualities: Iterable[float] | None = None,
        guard_angles: Iterable[float] | None = None,
        guard_qualities: Iterable[float] | None = None,
        guard_decay_rate: float | None = None,
        guard_distances: Iterable[float] | None = None,
        tick: int | None = None,
    ):
        """
        Advance the system by one Euler step. Provide either an explicit external_input
        or target/guard descriptors to build b.

        `tick` is the arena tick index; see `compute_sensory_map`. Callers that run
        several steps per arena tick must pass the SAME tick for all of them, or the
        sigma_s sensory noise is redrawn mid-tick.
        """
        self._step_count += 1
        self.compute_sensory_map(
            num_targets=self.num_targets,
            num_guards=self.num_guards,
            target_ids=target_ids,
            target_angles=target_angles,
            target_qualities=target_qualities,
            guard_angles=guard_angles,
            guard_qualities=guard_qualities,
            guard_decay_rate=guard_decay_rate,
            guard_distances=guard_distances,
            tick=tick,
        )

        times, bump_positions, final_norm = self.compute_dynamics(
            total_time=self.integration_time,
            dt=self.dt,
        )
        z_new = self.neural_ring
        if np.any(np.isnan(z_new)) or np.any(np.isinf(z_new)):
            norm = float(np.linalg.norm(z_new))
            msg = (
                f"MeanFieldSystem diverged at tick {self._step_count}: "
                f"state norm={norm:.4f}, dt={self.dt}, beta={self.beta}, sigma={self.sigma}"
            )
            logger.debug(msg)
            raise RuntimeError(msg)
        self._advance_sensory_time()

        return self.neural_ring, bump_positions, final_norm

    def run(self, steps: int, **step_kwargs):
        """Run multiple integration steps; returns trajectory of z."""
        history = []
        for _ in range(steps):
            self.step(**step_kwargs)
            history.append(self.neural_ring.copy())
        return np.asarray(history)

    def get_state(self):
        """Return current neural ring state."""
        return self.neural_ring.copy()

    def get_sensory_map(self):
        """Return the latest processed sensory map b."""
        return self.b.copy()

    def get_modulated_target_qualities(self):
        """Return the latest time-varying target qualities, BEFORE sigma_s noise."""
        return self.last_modulated_target_qualities.copy()

    def get_noisy_target_qualities(self):
        """Return the latest target qualities as actually scattered onto the ring.

        Equals `get_modulated_target_qualities()` when sigma_s is 0.
        """
        return self.last_noisy_target_qualities.copy()
