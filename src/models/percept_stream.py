# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
#  You may use, modify, and redistribute this file according to the terms of the
#  license. Attribution is required if this code is used in other works.
# ------------------------------------------------------------------------------

"""Sensory percept stream shared by the ring attractor and the embodied DDM.

Two modes sit behind one interface (`sensory_stream.mode`):

**legacy** (default, and the reproducibility path for every result generated to date)
    Each model generates its own sensory noise exactly as it always has: the ring
    attractor's `sigma_s` (per-tick noise on the target qualities) and the DDM's
    `eta_rate` (white, drawn per sub-step). The two models therefore see *different*
    realisations, and their noise levels are comparable only through an assumed readout
    model. `LegacyPerceptStream.sample()` is a pass-through that consumes no random
    numbers, so wiring it in cannot shift any existing RNG sequence.

**shared**
    ONE realisation of the percept, generated upstream of both models and reconstructed
    identically from the trial seed in any process:

        q_hat_i(t) = q_i(t) + beta_i + eps_i(t)
        beta_i     ~ N(0, s_beta^2)     drawn ONCE per trial per target (frozen bias)
        eps_i(t)   ~ N(0, eta^2 / dt)   drawn per tick per target       (white noise)

    Both channels are needed and they are not interchangeable: `beta` never averages
    out, `eps` averages out as 1/sqrt(t). Conflating a frozen bias with white noise is
    the category error this module exists to remove, so the two channels stay explicit
    and separately parameterised. Note the models' own `sigma_s` keys are NOT the same
    channel as each other: the DDM's is a frozen per-slot bias (a `beta` analogue),
    while the ring attractor's is per-tick quality noise (an `eps` analogue).

    Because the models receive the same numbers, a behavioural difference between them
    is attributable to the decision dynamics rather than to two independent noise draws.

**The precondition.** A shared stream is only well defined if the percept does not
depend on the agent's own state. `dist_mode` and `attention_mode` must therefore both be
`none`: otherwise `q_hat` depends on range or heading, which depend on the trajectory,
which differs between the models — and there would be nothing to share. Targets must
also be static. The angular positions `phi_i` and the ranges `d_i` still differ between
the models, because the agents move differently; that is correct and intended, since
geometry is part of the embodiment being compared. Only the *quality percepts* are
shared.

**RNG design.** Every draw is derived from its coordinates `(trial_seed, kind,
target_id, tick)` rather than from position in a sequential stream. This makes the
values order-independent (either model may sample in any order, any number of times),
position-independent (a model that terminates early does not shift what the other sees)
and — the property that matters for the current experiment layout, where the two models
never share a process — reproducible *across processes*: two separate SLURM jobs
reconstruct the identical stream from the same seed.

**What is still not matched.** `shared` mode makes the models' sensory *input*
identical. It does not make them noise-matched: the ring attractor's internal `sigma` is
neural noise with no DDM counterpart, so there is nothing to share it with. It remains a
free parameter, to be fixed by behavioural anchoring in a separate change. Exactly one
parameter is outstanding, and this module does not close it.

**`legacy` is kept indefinitely.** It is the reproducibility path for every result
generated to date, and it is the control arm for showing that the shared-stream protocol
did not itself change the answer.

**Both modes are reproducible from the arena seed.** They get there by different
routes, and the distinction matters when reading a result. `legacy` reproduces because
every generator a model owns is derived from the agent RNG the simulator seeds per run
(`TargetModel._make_rng`); the two models still draw *different* numbers, because they
draw them separately. `shared` reproduces because the percept is a pure function of
`(trial_seed, target_id, tick)`, which is a stronger property: it holds across
processes, across models, and independently of how often anything is sampled.

Historical note, because it bounds which old results can be compared: until the
arena-seeding fix the ring attractor's `sigma_s` bias came from an unseeded `Generator`
and its internal `sigma` noise from the global `np.random`, so ring-attractor runs
produced a different trajectory on every execution of the same config. Anything
generated before that fix is not reproducible, in either mode. Separately, `sigma_s`
itself was redefined: it used to be a frozen per-neuron bias added to `b`, and is now
per-tick noise on the target qualities, so its numeric value does not carry across that
change either.
"""

from __future__ import annotations

import hashlib
import logging
import math
from abc import ABC, abstractmethod
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

logger = logging.getLogger("sim.percept_stream")

# Relative tolerance when comparing tick rates. Rates come from config integers or from
# 1/dt round-trips, so anything looser than this is a genuine mismatch.
_RATE_RTOL = 1e-9


class PerceptStreamConfigError(ValueError):
    """A `sensory_stream` configuration that cannot produce a well-defined stream."""


class PerceptStream(ABC):
    """Interface both decision models read their per-target qualities through."""

    #: `legacy` or `shared`; stamped into every per-trial record.
    mode: str = "legacy"
    #: True when `sample()` returns its input untouched, so callers may skip the
    #: round-trip through the returned mapping and keep the original floats bit-exact.
    passthrough: bool = False

    @abstractmethod
    def sample(
        self,
        tick: int,
        target_ids: Sequence[str],
        clean_qualities: Mapping[str, float],
    ) -> dict[str, float]:
        """Return the perceived quality per target for this tick."""

    def assert_tick_rate(self, tick_rate: float) -> None:
        """Raise if a model samples at a rate other than the stream's own."""

    def describe(self) -> dict:
        """Return the resolved settings, for logging and per-trial stamping."""
        return {"sensory_stream_mode": self.mode}


class LegacyPerceptStream(PerceptStream):
    """Model-owned noise, exactly as today.

    `sample()` returns the clean qualities unchanged and each model applies its own
    `sigma_s` / `eta_rate` downstream. It must not consume any random numbers: the
    legacy RNG sequences belong to the models, and drawing here would shift them and
    stop existing results reproducing.
    """

    mode = "legacy"
    passthrough = True

    def sample(
        self,
        tick: int,
        target_ids: Sequence[str],
        clean_qualities: Mapping[str, float],
    ) -> dict[str, float]:
        """Return the clean qualities unchanged, consuming no randomness."""
        return {str(tid): float(clean_qualities[tid]) for tid in target_ids}


class SharedPerceptStream(PerceptStream):
    """One realisation, identical for every model reading it.

    Each draw is keyed by `(trial_seed, kind, target_id, tick)` and hashed to a sub-seed,
    so the value depends on its coordinates and never on consumption order. `beta` is
    keyed at tick 0 by construction, which is what makes it frozen within a trial rather
    than something to be bookkept.

    `eps` is defined as a RATE: the draw is `eta / sqrt(dt)`, so the integrated
    white-noise contribution over a fixed span of simulated time does not depend on the
    tick rate.
    """

    mode = "shared"
    passthrough = False

    def __init__(
        self,
        seed: int,
        dt: float,
        frozen_sd: float = 0.0,
        white_rate: float = 0.0,
    ):
        """Initialize the instance."""
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0.0:
            raise PerceptStreamConfigError(
                f"sensory_stream: dt must be a positive, finite number of seconds "
                f"(got {dt!r}); it is derived from the arena 'ticks_per_second'."
            )
        if float(frozen_sd) < 0.0:
            raise PerceptStreamConfigError(
                "sensory_stream.frozen_sd (s_beta) must be non-negative"
            )
        if float(white_rate) < 0.0:
            raise PerceptStreamConfigError(
                "sensory_stream.white_rate (eta) must be non-negative"
            )
        self._trial_seed = int(seed)
        self.dt = dt
        self.tick_rate = 1.0 / dt
        self.frozen_sd = float(frozen_sd)
        self.white_rate = float(white_rate)
        # eps ~ N(0, eta^2 / dt): the per-draw standard deviation is eta / sqrt(dt).
        self._eps_sd = self.white_rate / math.sqrt(dt)
        self._beta_cache: dict[str, float] = {}

    # ------------------------------------------------------------------
    @property
    def seed(self) -> int:
        """Return the trial seed the whole stream is reconstructed from."""
        return self._trial_seed

    def _draw(self, kind: str, target_id: str, tick: int) -> float:
        """Return the standard normal deviate belonging to these coordinates."""
        key = f"{self._trial_seed}|{kind}|{target_id}|{tick}".encode()
        sub = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")
        return float(np.random.default_rng(sub).standard_normal())

    def frozen_bias(self, target_id: str) -> float:
        """Return `beta_i`, the per-trial per-target sensory bias."""
        tid = str(target_id)
        cached = self._beta_cache.get(tid)
        if cached is None:
            # tick is pinned to 0 so the bias is constant within the trial by
            # construction; the cache is a speed-up, never the source of truth.
            cached = self.frozen_sd * self._draw("beta", tid, 0)
            self._beta_cache[tid] = cached
        return cached

    def white_noise(self, target_id: str, tick: int) -> float:
        """Return `eps_i(t)`, the per-tick white sensory noise."""
        return self._eps_sd * self._draw("eps", str(target_id), int(tick))

    # ------------------------------------------------------------------
    def sample(
        self,
        tick: int,
        target_ids: Sequence[str],
        clean_qualities: Mapping[str, float],
    ) -> dict[str, float]:
        """Return `q_hat_i = q_i + beta_i + eps_i(t)` for each target."""
        tick = int(tick)
        ids = [str(tid) for tid in target_ids]
        if len(set(ids)) != len(ids):
            raise PerceptStreamConfigError(
                "sensory_stream mode 'shared' keys every draw by target id, so the "
                f"percept ids must be unique; got {ids!r}."
            )
        out: dict[str, float] = {}
        for tid in ids:
            q = float(clean_qualities[tid])
            out[tid] = q + self.frozen_bias(tid) + self.white_noise(tid, tick)
        return out

    def assert_tick_rate(self, tick_rate: float) -> None:
        """Raise if a model samples at a rate other than the stream's own."""
        rate = float(tick_rate)
        if not math.isclose(rate, self.tick_rate, rel_tol=_RATE_RTOL):
            raise PerceptStreamConfigError(
                f"sensory_stream mode 'shared' was constructed for "
                f"{self.tick_rate:g} ticks/second (dt = {self.dt:g} s) but is being "
                f"sampled at {rate:g} ticks/second. 'white_rate' (eta) is a RATE, so "
                "dt would be ambiguous and the two models would not receive the same "
                "realisation."
            )

    def describe(self) -> dict:
        """Return the resolved settings, for logging and per-trial stamping."""
        return {
            "sensory_stream_mode": self.mode,
            "sensory_stream_seed": self._trial_seed,
            "sensory_stream_frozen_sd": self.frozen_sd,
            "sensory_stream_white_rate": self.white_rate,
            "sensory_stream_dt": self.dt,
        }


class UnseededSharedPerceptStream(PerceptStream):
    """Placeholder for `shared` mode before a trial seed is known.

    The models build their stream in `reset()`, which the simulator calls once per run
    with the arena seed already in place. A model constructed outside that path (a bare
    unit-test harness, say) can still be created, but sampling from it must fail loudly
    rather than silently invent a seed — an invented seed is exactly the failure this
    feature exists to prevent.
    """

    mode = "shared"
    passthrough = False

    def __init__(self, reason: str):
        """Initialize the instance."""
        self._reason = reason

    def sample(
        self,
        tick: int,
        target_ids: Sequence[str],
        clean_qualities: Mapping[str, float],
    ) -> dict[str, float]:
        """Always raise: there is no seed to reconstruct the stream from."""
        raise PerceptStreamConfigError(self._reason)

    def describe(self) -> dict:
        """Return the resolved settings, for logging and per-trial stamping."""
        return {"sensory_stream_mode": self.mode, "sensory_stream_seed": None}


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
_VALID_MODES = ("legacy", "shared")
_KNOWN_KEYS = {"mode", "frozen_sd", "white_rate", "seed"}


class SensoryStreamSpec:
    """The resolved, precondition-checked `sensory_stream` block.

    Preconditions that depend only on configuration are checked when the spec is built
    (i.e. at model construction). The trial seed is not known then — the simulator
    supplies it per run — so building the stream itself is deferred to `build()`.
    """

    def __init__(
        self,
        mode: str,
        frozen_sd: float = 0.0,
        white_rate: float = 0.0,
        seed: Optional[int] = None,
        tick_rate: float = 1.0,
    ):
        """Initialize the instance."""
        self.mode = mode
        self.frozen_sd = float(frozen_sd)
        self.white_rate = float(white_rate)
        self.seed = None if seed is None else int(seed)
        self.tick_rate = float(tick_rate)
        self.dt = 1.0 / self.tick_rate if self.tick_rate > 0.0 else float("nan")

    @property
    def is_shared(self) -> bool:
        """True when the shared realisation is active."""
        return self.mode == "shared"

    def build(self, trial_seed: Optional[int], owner: str = "") -> PerceptStream:
        """Return the stream for one trial, given the seed the simulator resolved."""
        if self.mode == "legacy":
            return LegacyPerceptStream()
        seed = self.seed if self.seed is not None else trial_seed
        if seed is None:
            return UnseededSharedPerceptStream(
                f"{owner}sensory_stream mode 'shared' has no seed: "
                "'sensory_stream.seed' is null and no arena random_seed reached the "
                "agent. Set 'seed' explicitly, or run through the simulator, which "
                "supplies the per-run arena seed."
            )
        return SharedPerceptStream(
            seed=int(seed),
            dt=self.dt,
            frozen_sd=self.frozen_sd,
            white_rate=self.white_rate,
        )

    def describe(self) -> dict:
        """Return the resolved settings, for logging."""
        return {
            "mode": self.mode,
            "frozen_sd": self.frozen_sd,
            "white_rate": self.white_rate,
            "seed": self.seed,
            "dt": self.dt,
        }


def resolve_sensory_stream_spec(
    config: Optional[Mapping],
    *,
    owner: str = "",
    tick_rate: float = 1.0,
    arena_tick_rate: Optional[float] = None,
    dist_mode: str = "none",
    attention_mode: str = "none",
    sigma_s: float = 0.0,
    eta_rate: Optional[Iterable[float] | float] = None,
) -> SensoryStreamSpec:
    """Parse a `sensory_stream` block and enforce every `shared`-mode precondition.

    Each violation names the offending key and says why it is incompatible: a silent
    violation produces a comparison that looks controlled and is not.
    """
    prefix = f"{owner}: " if owner else ""
    cfg = dict(config or {})
    unknown = set(cfg) - _KNOWN_KEYS
    # Tolerate the `_comment*` keys the JSON configs use for inline documentation.
    unknown = {k for k in unknown if not str(k).startswith("_")}
    if unknown:
        raise PerceptStreamConfigError(
            f"{prefix}unknown sensory_stream key(s) {sorted(unknown)}; "
            f"expected any of {sorted(_KNOWN_KEYS)}."
        )

    mode = str(cfg.get("mode", "legacy")).strip().lower()
    if mode not in _VALID_MODES:
        raise PerceptStreamConfigError(
            f"{prefix}sensory_stream.mode must be one of {list(_VALID_MODES)}; "
            f"got '{mode}'."
        )

    if mode == "legacy":
        # Nothing else in the block is read, and nothing about the model changes.
        return SensoryStreamSpec(mode="legacy", tick_rate=tick_rate)

    # --- Section 5 preconditions, all of them fatal in `shared` mode. -------------
    if str(dist_mode).strip().lower() != "none":
        raise PerceptStreamConfigError(
            f"{prefix}sensory_stream.mode 'shared' requires dist_mode 'none' "
            f"(got '{dist_mode}'). Distance modulation makes the percept a function of "
            "the agent's own range to the target, so the two models — which move "
            "differently — could not receive the same realisation."
        )
    if str(attention_mode).strip().lower() != "none":
        raise PerceptStreamConfigError(
            f"{prefix}sensory_stream.mode 'shared' requires attention_mode 'none' "
            f"(got '{attention_mode}'). An attentional gate makes the percept a "
            "function of the agent's own heading, so the two models — which move "
            "differently — could not receive the same realisation."
        )
    if arena_tick_rate is not None and not math.isclose(
        float(arena_tick_rate), float(tick_rate), rel_tol=_RATE_RTOL
    ):
        raise PerceptStreamConfigError(
            f"{prefix}sensory_stream.mode 'shared' requires the arena "
            f"'ticks_per_second' ({float(arena_tick_rate):g}) to equal the agent's "
            f"({float(tick_rate):g}). 'white_rate' (eta) is a noise RATE, so the "
            "stream's dt would be ambiguous."
        )
    if float(sigma_s) != 0.0:
        raise PerceptStreamConfigError(
            f"{prefix}sensory_stream.mode 'shared' requires sigma_s = 0 "
            f"(got {float(sigma_s)!r}). The model-owned sensory noise moves upstream "
            "into the shared stream, which sets it with sensory_stream.frozen_sd "
            "(s_beta, frozen) and white_rate (eta, per tick); leaving sigma_s non-zero "
            "would add a second, unshared noise source on top."
        )
    if eta_rate is not None:
        eta_values = (
            [float(eta_rate)]
            if isinstance(eta_rate, (int, float))
            else [float(v) for v in eta_rate]
        )
        if any(v != 0.0 for v in eta_values):
            raise PerceptStreamConfigError(
                f"{prefix}sensory_stream.mode 'shared' requires eta_rate = 0 "
                f"(got {eta_values!r}). The white sensory noise moves upstream into the "
                "shared stream and is set by sensory_stream.white_rate (eta); leaving "
                "eta_rate non-zero would add a second, unshared white channel."
            )

    if float(tick_rate) <= 0.0:
        raise PerceptStreamConfigError(
            f"{prefix}sensory_stream.mode 'shared' needs a positive tick rate to "
            f"resolve dt; got {float(tick_rate)!r}."
        )

    seed = cfg.get("seed", None)
    spec = SensoryStreamSpec(
        mode="shared",
        frozen_sd=float(cfg.get("frozen_sd", 0.0)),
        white_rate=float(cfg.get("white_rate", 0.0)),
        seed=None if seed is None else int(seed),
        tick_rate=float(tick_rate),
    )
    if spec.frozen_sd < 0.0 or spec.white_rate < 0.0:
        raise PerceptStreamConfigError(
            f"{prefix}sensory_stream.frozen_sd and sensory_stream.white_rate must be "
            "non-negative standard deviations."
        )
    return spec
