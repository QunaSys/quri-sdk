from typing import Any, Protocol

from quri_algo.problem import QubitHamiltonian

from quri_parts.qsub.op import OpFactory


class TimeEvolutionOpFactory(OpFactory[QubitHamiltonian, float, Any], Protocol):
    ...
