# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Seed derivation (CAMPAIGN_SPEC.md Section 6).

    sensory_seed  = H(campaign_seed, "sensory",  dQ, dtheta, replicate)
    internal_seed = H(campaign_seed, "internal", dQ, dtheta, criterion, replicate)

The criterion is ABSENT from the sensory seed: every point on a speed/accuracy curve
then sees the same sensory realisations, which is what makes the sweep paired.
Model identity is likewise absent, so the scheme stays correct when the ring-attractor
arm is added later.

H is the project's existing derivation style: the percept stream keys every draw as
`blake2b(digest_size=8)` over a '|'-joined coordinate string
(src/models/percept_stream.py). The same construction is used here, reduced mod 2^31
because arena `random_seed` values are 31-bit ints throughout the existing configs.

Factor levels enter as CANONICAL TOKENS (e.g. "q01", "a60", "ce0.03"), never as raw
floats, so the derivation cannot drift with float formatting.
"""

from __future__ import annotations

import hashlib


def derive_seed(campaign_seed: int, *parts) -> int:
    """Return the 31-bit seed for the given coordinates."""
    key = "|".join([str(int(campaign_seed))] + [str(p) for p in parts]).encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2 ** 31)


def sensory_seed(campaign_seed: int, q_token: str, a_token: str, replicate: int) -> int:
    """Seed of the shared percept stream. The criterion is deliberately absent."""
    return derive_seed(campaign_seed, "sensory", q_token, a_token, int(replicate))


def internal_seed(
    campaign_seed: int, q_token: str, a_token: str, crit_token: str, replicate: int
) -> int:
    """Seed of everything model-internal (via the arena seed). Criterion included."""
    return derive_seed(
        campaign_seed, "internal", q_token, a_token, crit_token, int(replicate)
    )
