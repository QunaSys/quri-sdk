import warnings

warnings.warn(
    "quri_algo.qsub is deprecated. Use quri_algo.circuit_lib instead.",
    DeprecationWarning,
    stacklevel=2,
)

from quri_algo.circuit_lib.qpe import (  # noqa: F401, E402
    QPE,
    LineH,
    LineHSub,
    QFTdag,
    QFTdagSub,
    QPEListUk,
    QPEListUkSub,
    QPESub,
)
