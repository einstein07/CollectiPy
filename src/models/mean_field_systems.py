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

import math
from typing import Iterable
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

class MeanFieldSystem:
    """
    Phenomenological spiking ring attractor (mean-field version).

    Core dynamics:
        z_dot = -z + tanh((u0 - s) * M @ z + b - beta) - tanh(beta) + noise
        tau * s_dot = -s + k * ||z||^4
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
        sigma: float = 0.01,
        dt: float = 0.1,
        rng: np.random.Generator | None = None,
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


    def compute_interaction_kernel(self) -> np.ndarray:
        """Compute cosine-based interaction kernel."""
        theta_col = self.theta[:, np.newaxis] # shape (num_neurons, 1)
        theta_row = self.theta[np.newaxis, :] # shape (1, num_neurons)
        delta = np.abs(_delta_angle(theta_col, theta_row)) # pairwise delta_ij matrix: shape (num_neurons, num_neurons)
        return (1.0 / self.num_neurons) * np.cos(np.pi * (delta / np.pi) ** self.v)

    def compute_sensory_map(
        self,
        num_targets: int,
        num_guards: int,
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
        
        if num_targets > 0 and target_angles is not None and target_qualities is not None:
            target_angles = np.asarray(target_angles, dtype=float).reshape(1, -1)
            target_qualities = np.asarray(target_qualities, dtype=float).reshape(-1)
            delta_targets = _delta_angle(self.theta[:, None], target_angles)
            vm_targets = np.exp(self.kappa * (np.cos(delta_targets) - 1.0))
            b = vm_targets @ target_qualities

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

        b /= math.sqrt(self.num_neurons)
        self.b = b
        return self.b

    def reset(
        self,
        z: np.ndarray | None = None,
        s: float = 0.0,
        external_input: np.ndarray | None = None,
    ):
        """Reset internal state."""
        if z is not None:
            z = np.asarray(z, dtype=float).reshape(-1)
            if z.shape[0] != self.num_neurons:
                raise ValueError("Reset state dimension must match num_neurons")
            self.z = z
        else:
            self.z = np.zeros(self.num_neurons, dtype=float)

        self.s = float(s)

        if external_input is not None:
            external_input = np.asarray(external_input, dtype=float).reshape(-1)
            if external_input.shape[0] != self.num_neurons:
                raise ValueError("Reset input dimension must match num_neurons")
            self.b = external_input
        else:
            self.b = np.zeros_like(self.z)

        """Update external input vector b."""
        external_input = np.asarray(external_input, dtype=float).reshape(-1)
        if external_input.shape[0] != self.num_neurons:
            raise ValueError("external_input dimension must match num_neurons")
        self.b = external_input

    @njit(fastmath=True, parallel=True)
    def euler_integrate(y0, t_eval, u, b, M, beta, n, sigma, randn_like_func):
        dt = t_eval[1] - t_eval[0]
        y = np.zeros((len(t_eval), len(y0)))
        y[0] = y0
        for i in range(1, len(t_eval)):
            noise = randn_like_func(y[i-1], sigma * np.sqrt(dt), 1.0 / np.sqrt(n))
            dydt = -y[i-1] + np.tanh(u * M @ y[i-1] + b - beta) - np.tanh(-beta) + noise
            y[i] = y[i-1] + dt * dydt
        return y

    @njit(fastmath=True, parallel=True)
    def randn_like(y, sigma, inv_sqrt_n):
        out = np.empty_like(y)
        for i in prange(y.size):
            out[i] = np.random.normal(0.0, sigma) * inv_sqrt_n
        return out
    
    # Integrate timesteps to simulate the neural field dynamics
    def compute_dynamics(self, total_time=200, dt=0.1):
        t_eval = np.arange(0, total_time, dt)
        y0 = self.neural_field.copy()
        result = MeanFieldSystem.euler_integrate(y0, t_eval, self.u, self.b, self.M, self.beta, self.num_neurons, self.sigma, MeanFieldSystem.randn_like)
        times = t_eval
        # Compute CoM of bump activity at each time
        bump_positions = np.array([compute_center_of_mass(z_t, self.theta) for z_t in result])
        final_norm = np.linalg.norm(result[-1])
        self.neural_field = result[-1]  # Update neural field state
        return times, bump_positions, final_norm


    def step(
        self,
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
        self.compute_sensory_map(
            num_targets=self.num_targets,
            num_guards=self.num_guards,
            target_angles=target_angles,
            target_qualities=target_qualities,
            guard_angles=guard_angles,
            guard_qualities=guard_qualities,
            guard_decay_rate=guard_decay_rate,
            guard_distances=guard_distances,
        )

        times, bump_positions, final_norm = self.compute_dynamics(total_time=50)

        return self.neural_field, bump_positions, final_norm

    def run(self, steps: int, **step_kwargs):
        """Run multiple integration steps; returns trajectory of z."""
        history = []
        for _ in range(steps):
            self.step(**step_kwargs)
            history.append(self.z.copy())
        return np.asarray(history)

    def get_state(self):
        """Return current (z, s)."""
        return self.neural_field.copy()
