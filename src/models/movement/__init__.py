# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Fabio Oddi
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
#  You may use, modify, and redistribute this file according to the terms of the
#  license. Attribution is required if this code is used in other works.
# ------------------------------------------------------------------------------

"""
Movement model package.

Importing submodules registers the built-in movement models.
"""

# Register built-in models on import.
from . import random_walk  # noqa: F401
from . import random_way_point  # noqa: F401
from . import spin_model  # noqa: F401
from . import mean_field_model  # noqa: F401
