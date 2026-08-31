# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Seed derivation (Section 3).

    trial_seed = H(campaign_seed, "trial", delta_token, replicate)

THE ARM IS ABSENT FROM THE KEY, and that is the whole point. The historical
generator used md5(f"{u}_{diff}_{run}"), which put the gain in the key, so every arm
drew a different noise realisation and the four arms were UNPAIRED. Dropping the arm
makes all four replay the same realisation at the same (delta, replicate), which is
what licenses the paired statistics of Section 7 -- McNemar on reversal, Wilcoxon
signed-rank on latency, matched per (delta, replicate).

One seed, not two: the arena `random_seed` is the single trial seed, and the
simulator hands it to the agent, which resolves the shared percept stream from it
(`sensory_stream.seed: null`). Model-internal randomness descends from the same
number. Splitting sensory from internal seeds -- as the frontier campaign does, to
keep a speed/accuracy curve paired across its criterion axis -- would buy nothing
here, because the criterion is fixed and the arms differ in the decision rule rather
than in a swept model parameter.

H is the project's existing derivation style: the percept stream keys every draw as
`blake2b(digest_size=8)` over a '|'-joined coordinate string
(src/models/percept_stream.py). The same construction is used here, reduced mod 2^31
because arena `random_seed` values are 31-bit ints throughout the existing configs.
"""

from __future__ import annotations

import hashlib

from flexibility import factors


def derive_seed(campaign_seed: int, *parts) -> int:
    """Return the 31-bit seed for the given coordinates."""
    key = "|".join([str(int(campaign_seed))] + [str(p) for p in parts]).encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2 ** 31)


def trial_seed(delta_token: str, replicate: int,
               campaign_seed: int = factors.CAMPAIGN_SEED) -> int:
    """The arena seed for one (delta, replicate) — identical across all four arms."""
    return derive_seed(campaign_seed, "trial", delta_token, int(replicate))
