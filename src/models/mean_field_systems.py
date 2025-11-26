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

import numpy as np


def _delta_angle(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    """Return smallest signed difference between angles (radians)."""
    return (a1 - a2 + np.pi) % (2 * np.pi) - np.pi


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
        u0: float,
        beta: float = 1.0,
        v: float = 0.5,
        kappa: float = 20.0,
        tau: float = 33.0,
        k: float = 1.0,
        sigma: float = 0.0,
        dt: float = 0.1,
        initial_state: np.ndarray | None = None,
        initial_s: float = 0.0,
        external_input: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ):
        """
        Initialize the mean-field system.

        Args:
            num_neurons: Number of units in the ring.
            u0: Baseline coupling strength.
            beta: Implicit self-excitation offset.
            v: Shape parameter for the interaction kernel.
            kappa: Concentration for sensory von Mises inputs.
            tau: Time constant for inhibitory neuromodulation s.
            k: Gain applied to ||z||^4 in s dynamics.
            sigma: Noise standard deviation (added to z dynamics, scaled by sqrt(n)).
            dt: Integration time step.
            initial_state: Optional initial state vector z.
            initial_s: Optional initial inhibitory value s.
            external_input: Optional initial external input vector b.
            rng: Optional numpy random generator for reproducibility.
        """
        if num_neurons <= 0:
            raise ValueError("num_neurons must be positive")
        self.num_neurons = int(num_neurons)
        self.u0 = float(u0)
        self.beta = float(beta)
        self.v = float(v)
        self.kappa = float(kappa)
        self.tau = float(tau)
        self.k = float(k)
        self.sigma = float(sigma)
        self.dt = float(dt)
        self.rng = rng or np.random.default_rng()

        self.theta = np.linspace(-np.pi, np.pi, self.num_neurons, endpoint=False)
        self.M = self.compute_interaction_kernel()

        self.z = (
            np.zeros(self.num_neurons, dtype=float)
            if initial_state is None
            else np.asarray(initial_state, dtype=float).reshape(-1)
        )
        if self.z.shape[0] != self.num_neurons:
            raise ValueError("initial_state dimension must match num_neurons")

        self.s = float(initial_s)
        self.b = (
            np.zeros(self.num_neurons, dtype=float)
            if external_input is None
            else np.asarray(external_input, dtype=float).reshape(-1)
        )
        if self.b.shape[0] != self.num_neurons:
            raise ValueError("external_input dimension must match num_neurons")

    def compute_interaction_kernel(self) -> np.ndarray:
        """Compute cosine-based interaction kernel."""
        theta_col = self.theta[:, np.newaxis]
        theta_row = self.theta[np.newaxis, :]
        delta = np.abs(_delta_angle(theta_col, theta_row))
        return (1.0 / self.num_neurons) * np.cos(np.pi * (delta / np.pi) ** self.v)

    def compute_sensory_map(
        self,
        target_angles: Iterable[float],
        target_qualities: Iterable[float],
        guard_angles: Iterable[float] | None = None,
        guard_qualities: Iterable[float] | None = None,
        guard_decay_rate: float | None = None,
        guard_distance: float | None = None,
    ) -> np.ndarray:
        """
        Compute sensory input b using von Mises bumps for targets and optional guard inhibition.
        """
        target_angles = np.asarray(target_angles, dtype=float).reshape(1, -1)
        target_qualities = np.asarray(target_qualities, dtype=float).reshape(-1)
        delta_targets = _delta_angle(self.theta[:, None], target_angles)
        vm_targets = np.exp(self.kappa * (np.cos(delta_targets) - 1.0))
        b = vm_targets @ target_qualities

        if (
            guard_angles is not None
            and guard_qualities is not None
            and guard_decay_rate is not None
            and guard_distance is not None
        ):
            guard_angles = np.asarray(guard_angles, dtype=float).reshape(1, -1)
            guard_qualities = np.asarray(guard_qualities, dtype=float).reshape(-1)
            delta_guards = _delta_angle(self.theta[:, None], guard_angles)
            vm_guards = np.exp(self.kappa * (np.cos(delta_guards) - 1.0))
            scaled = guard_qualities * math.exp(-guard_decay_rate * guard_distance)
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

    def update_external_input(self, external_input: np.ndarray):
        """Update external input vector b."""
        external_input = np.asarray(external_input, dtype=float).reshape(-1)
        if external_input.shape[0] != self.num_neurons:
            raise ValueError("external_input dimension must match num_neurons")
        self.b = external_input

    def step(
        self,
        external_input: np.ndarray | None = None,
        target_angles: Iterable[float] | None = None,
        target_qualities: Iterable[float] | None = None,
        guard_angles: Iterable[float] | None = None,
        guard_qualities: Iterable[float] | None = None,
        guard_decay_rate: float | None = None,
        guard_distance: float | None = None,
    ):
        """
        Advance the system by one Euler step. Provide either an explicit external_input
        or target/guard descriptors to build b.
        """
        if external_input is not None:
            self.update_external_input(external_input)
        elif target_angles is not None and target_qualities is not None:
            self.compute_sensory_map(
                target_angles=target_angles,
                target_qualities=target_qualities,
                guard_angles=guard_angles,
                guard_qualities=guard_qualities,
                guard_decay_rate=guard_decay_rate,
                guard_distance=guard_distance,
            )

        noise = np.zeros_like(self.z)
        if self.sigma > 0:
            noise = self.rng.normal(0.0, self.sigma, size=self.z.shape) / math.sqrt(self.num_neurons)

        drive = (self.u0 - self.s) * (self.M @ self.z) + self.b - self.beta
        dz = -self.z + np.tanh(drive) - np.tanh(self.beta) + noise
        self.z = self.z + self.dt * dz

        z_norm = np.linalg.norm(self.z)
        ds = (-self.s + self.k * (z_norm ** 4)) / self.tau
        self.s = self.s + self.dt * ds
        return self.z, self.s

    def run(self, steps: int, **step_kwargs):
        """Run multiple integration steps; returns trajectory of z."""
        history = []
        for _ in range(steps):
            self.step(**step_kwargs)
            history.append(self.z.copy())
        return np.asarray(history)

    def bump_center_of_mass(self) -> float | None:
        """Return current bump center angle or None if no activity."""
        if np.allclose(self.z, 0.0):
            return None
        sin_sum = np.sum(self.z * np.sin(self.theta))
        cos_sum = np.sum(self.z * np.cos(self.theta))
        if sin_sum == 0 and cos_sum == 0:
            return None
        return math.atan2(sin_sum, cos_sum)

    def get_state(self):
        """Return current (z, s)."""
        return self.z.copy(), float(self.s)
