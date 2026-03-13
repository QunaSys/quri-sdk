# NOTE: This module has a deprecated counterpart at
# quri_algo/qsub/time_evolution/interface.py which re-exports from here.
from typing import Any, Protocol

from quri_parts.qsub.op import OpFactory

from quri_algo.problem import QubitHamiltonian


class TimeEvolutionOpFactory(OpFactory[QubitHamiltonian, float, Any], Protocol):
    ...
