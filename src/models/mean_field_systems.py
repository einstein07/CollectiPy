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

logger = logging.getLogger("sim.mean_field")
logger.setLevel(logging.DEBUG)

class MeanFieldSystem:
    """
    Phenomenological spiking ring attractor (mean-field version).

    Core dynamics:
        z_dot = -z + tanh((u0 - s) * M @ z + b - beta) - tanh(beta) + noise
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
        dt: float = 0.1,
        integration_time: float = 50.0,
        sensory_time_mode: str = "world_time",
        sensory_dt: float | None = None,
        rng: np.random.Generator | None = None,
        # SFA parameters
        g_adapt: float = 0.0, # set > 0 to enable SFA
        tau_adapt: float = 0.0, # adaptation time constant
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
            dt: Integration time step.
            initial_state: Optional initial state vector z.
            external_input: Optional initial external input vector b.
            rng: Optional numpy random generator for reproducibility.
            g_adapt: Adaptation strength (set > 0 to enable spike-frequency adaptation).
            tau_adapt: Adaptation time constant (used when g_adapt > 0).
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
        self.rng = rng or np.random.default_rng()

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

        self.g_adapt = float(g_adapt)
        self.tau_adapt = float(tau_adapt)
        if self.g_adapt > 0.0 and self.tau_adapt <= 0.0:
            raise ValueError("tau_adapt must be positive when g_adapt > 0")
        
        self.adapt_ring = np.zeros(self.num_neurons, dtype=float)
        self.last_target_ids: list[str] = []
        self.last_target_base_qualities = np.array([], dtype=float)
        self.last_modulated_target_qualities = np.array([], dtype=float)
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
    ) -> np.ndarray:
        """
        Compute sensory input b using von Mises bumps for targets and optional guard inhibition.
        """
        b = np.zeros(self.num_neurons, dtype=float)
        self.last_target_ids = []
        self.last_target_base_qualities = np.array([], dtype=float)
        self.last_modulated_target_qualities = np.array([], dtype=float)
        
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
            delta_targets = _delta_angle(self.theta[:, None], target_angles)
            vm_targets = np.exp(self.kappa * (np.cos(delta_targets) - 1.0))
            b = vm_targets @ modulated_target_qualities
            logger.debug(
                    "Target angles: %s",
                    np.array2string(
                        np.asarray(target_angles, dtype=float).reshape(-1),
                        precision=6,
                        separator=", ",
                        max_line_width=1000,
                    ),
                )
            logger.debug(
                    "Target qualities: %s",
                    np.array2string(
                        np.asarray(modulated_target_qualities, dtype=float).reshape(-1),
                        precision=6,
                        separator=", ",
                        max_line_width=1000,
                    ),
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
        for i in range(1, len(t_eval)):
            noise = randn_like_func(y[i-1], sigma * np.sqrt(dt), 1.0 / np.sqrt(n))
            dydt = -y[i-1] + np.tanh(u * M @ y[i-1] + b - beta) - np.tanh(-beta) + noise
            y[i] = y[i-1] + dt * dydt
        return y
    
    @staticmethod
    @njit(parallel=True, cache=True)
    def randn_like(y, sigma, inv_sqrt_n):
        out = np.empty_like(y)
        for i in prange(y.size):
            out[i] = np.random.normal(0.0, sigma) * inv_sqrt_n
        return out
    

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
                MeanFieldSystem.randn_like,
            )
            z_traj = result[:, :self.num_neurons]
            a_traj = result[:, self.num_neurons:]
        else:
            y0 = self.neural_ring.copy()
            z_traj = MeanFieldSystem.euler_integrate(
                y0, t_eval,
                self.u, self.b, self.M, self.beta,
                self.num_neurons, self.sigma,
                MeanFieldSystem.randn_like,
            )
            a_traj = None

        times = t_eval

        bump_positions = np.array([compute_center_of_mass(z_t, self.theta) for z_t in z_traj])
        final_norm = np.linalg.norm(z_traj[-1])

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
    ):
        """
        Advance the system by one Euler step. Provide either an explicit external_input
        or target/guard descriptors to build b.
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
        """Return the latest time-varying target qualities used for b."""
        return self.last_modulated_target_qualities.copy()
