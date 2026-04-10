"""Pytest configuration for CollectiPy test suite."""
import pytest


def pytest_addoption(parser):
    """Add --regen flag to regenerate reference snapshots."""
    parser.addoption(
        "--regen",
        action="store_true",
        default=False,
        help="Regenerate .npy reference snapshots in tests/fixtures/",
    )
