# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Flexibility campaign package.

Guards the interpreter version at import. The simulator this campaign drives uses
PEP 604 annotations (`X | Y`), which are a syntax error before Python 3.10 -- and the
failure surfaces as a bare `TypeError: unsupported operand type(s) for |` several
frames deep inside `src/plugin_base.py`, reached through an import chain that has
nothing to do with what the caller asked for. Failing here instead says what is
actually wrong and how to fix it.
"""

import sys

if sys.version_info < (3, 10):
    raise RuntimeError(
        f"the flexibility campaign needs Python >= 3.10, but this is "
        f"{sys.version.split()[0]} at {sys.executable}.\n"
        "The simulator uses PEP 604 ('X | Y') annotations, so 3.9 fails at import.\n"
        "  Use the project venv:  .venv/bin/python -m flexibility.<module>\n"
        "  On bwUniCluster:       module load devel/python/3.10.12_gnu_12.2\n"
        "  Or create the venv:    python3.10 -m venv .venv"
    )
