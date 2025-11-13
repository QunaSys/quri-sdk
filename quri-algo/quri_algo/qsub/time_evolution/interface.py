from typing import Any, Protocol

from quri_parts.qsub.op import OpFactory

from quri_algo.problem import QubitHamiltonian


class TimeEvolutionOpFactory(OpFactory[QubitHamiltonian, float, Any], Protocol):
    ...
