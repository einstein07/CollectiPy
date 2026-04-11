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
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.slow (skipped by default).",
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --run-slow is passed."""
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="slow test — use --run-slow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
