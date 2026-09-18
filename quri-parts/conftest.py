import sys

import pytest

collect_ignore: list[str] = []
if sys.version_info < (3, 10):
    collect_ignore.extend(
        [
            "packages/tket",
            "packages/qsub",
            "packages/qret",
        ]
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--with-gridsynth",
        action="store_true",
        default=False,
        help="Run tests marked `gridsynth`, which require the gridsynth command "
        "provided by pyqret.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "gridsynth: test requires the gridsynth command provided by pyqret. "
        "Run only when the --with-gridsynth option is given.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    # gridsynth-marked tests are skipped unless --with-gridsynth is passed.
    if config.getoption("--with-gridsynth"):
        return
    skip_gridsynth = pytest.mark.skip(reason="need --with-gridsynth option to run")
    for item in items:
        if "gridsynth" in item.keywords:
            item.add_marker(skip_gridsynth)
