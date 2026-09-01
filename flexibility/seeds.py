# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Seed derivation — TWO seeds per run, following `campaign/seeds.py`.

    sensory_seed  = H(campaign_seed, "sensory",  delta_token, replicate)
    internal_seed = H(campaign_seed, "internal", delta_token, arm, replicate)

**The arm is ABSENT from the sensory seed and PRESENT in the internal one.** That
asymmetry is the whole design:

  * every arm sees the SAME sensory realisation at the same (delta, replicate), which
    is what licenses the paired statistics -- McNemar on reversal, Wilcoxon
    signed-rank on latency, matched per cell;
  * model-internal randomness is deliberately NOT shared. Matching it across a ring
    attractor and a DDM would be meaningless anyway, since they consume randomness
    differently and at different rates; forcing a common root would imply a coupling
    that does not exist.

Both seeds are written EXPLICITLY into every generated config -- `sensory_stream.seed`
and `arena.random_seed` -- rather than leaving the sensory one null for the simulator
to resolve from the arena seed. Null works, but it makes the matching invisible: the
config records `"seed": null` and a reader has to know the resolution rule to see that
anything is matched at all. Writing it out makes each config self-documenting and the
pairing auditable straight from the results directory. This mirrors the convention in
`ra_ddm_frontier_slices` and `ra_ddm_frontier_ddm*`, where the sensory seed is
identical across model variants while the arena seed differs.

H is the project's existing derivation style: the percept stream keys every draw as
`blake2b(digest_size=8)` over a '|'-joined coordinate string
(src/models/percept_stream.py). The same construction is used here, reduced mod 2^31
because arena `random_seed` values are 31-bit ints throughout the existing configs.

Factor levels enter as CANONICAL TOKENS (e.g. "d1.0000pct"), never as raw floats, so
the derivation cannot drift with float formatting.
"""

from __future__ import annotations

import hashlib

from flexibility import factors


def derive_seed(campaign_seed: int, *parts) -> int:
    """Return the 31-bit seed for the given coordinates."""
    key = "|".join([str(int(campaign_seed))] + [str(p) for p in parts]).encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2 ** 31)


def sensory_seed(delta_token: str, replicate: int,
                 campaign_seed: int = factors.CAMPAIGN_SEED) -> int:
    """Seed of the shared percept stream. The ARM is deliberately absent."""
    return derive_seed(campaign_seed, "sensory", delta_token, int(replicate))


def internal_seed(arm: str, delta_token: str, replicate: int,
                  campaign_seed: int = factors.CAMPAIGN_SEED) -> int:
    """Seed of everything model-internal, via the arena. The ARM is included."""
    return derive_seed(campaign_seed, "internal", delta_token, arm, int(replicate))
