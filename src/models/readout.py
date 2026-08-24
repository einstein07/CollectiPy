# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
#  You may use, modify, and redistribute this file according to the terms of the
#  license. Attribution is required if this code is used in other works.
# ------------------------------------------------------------------------------

"""Shared circular readout for the ring attractor and the embodied accumulator.

Both the mean-field ring attractor (over its neurons) and the embodied DDM/LCA
(over its per-target accumulators) reduce a set of non-negative weights placed at
angles to a single heading plus two order parameters. Keeping this in one place is
what makes the trajectory comparison between the two models airtight: identical code
path, identical semantics.
"""

from __future__ import annotations

import math

import numpy as np


def circular_readout(weights, angles, threshold: float = 0.0):
    """Thresholded circular mean.

    Args:
        weights: activity/accumulator values placed at ``angles`` (any shape, flattened).
        angles: angle (radians) associated with each weight.
        threshold: values at or below this are zeroed before the circular mean.

    Returns:
        (heading, magnitude, concentration) where
          heading       = atan2(Sum w sin, Sum w cos)   in (-pi, pi]
          magnitude     = hypot(Sum w sin, Sum w cos)   >= 0, scales with total weight
          concentration = magnitude / (Sum w + eps)     in [0, 1], angular coherence
    """
    w = np.where(weights > threshold, weights, 0.0)
    sin_sum = float(np.sum(w * np.sin(angles)))
    cos_sum = float(np.sum(w * np.cos(angles)))
    heading = math.atan2(sin_sum, cos_sum)
    magnitude = math.hypot(sin_sum, cos_sum)
    concentration = magnitude / (float(np.sum(w)) + 1e-12)
    return heading, magnitude, concentration
