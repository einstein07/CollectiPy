"""Root pytest configuration — adds src/ to sys.path for all tests.

This ensures `from models.xxx import ...` works with pytest 6.x, which does not
support the `pythonpath` option in pyproject.toml (added in pytest 7.0).
"""
import sys
from pathlib import Path

# Insert src/ at the front of sys.path so project modules are importable
_src = str(Path(__file__).parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
