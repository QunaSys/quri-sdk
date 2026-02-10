import warnings

warnings.warn(
    "quri_algo.qsub is deprecated. Use quri_algo.circuit_lib instead.",
    DeprecationWarning,
    stacklevel=2,
)

from quri_algo.circuit_lib.time_evolution.interface import (  # noqa: F401, E402
    TimeEvolutionOpFactory,
)
