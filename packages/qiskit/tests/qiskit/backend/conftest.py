import pytest


def pytest_addoption(parser):  # type: ignore
    parser.addoption(
        "--api",
        action="store_true",
        default=False,
        help="Run tests that use API",
    )


def pytest_collection_modifyitems(config, items):  # type: ignore
    if not config.getoption("--api"):
        skipper = pytest.mark.skip(reason="Only run when --api is given")
        for item in items:
            if "api" in item.keywords:
                item.add_marker(skipper)
