import sys

collect_ignore: list[str] = []
if sys.version_info < (3, 10):
    collect_ignore.extend(
        [
            "packages/tket",
            "packages/qsub",
        ]
    )
