# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from enum import Enum
from typing import (
    Any,
    Callable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    TypeAlias,
    TypeVar,
)

from quri_parts.algo.optimizer import OptimizerState
from quri_parts.backend.units import TimeValue
from quri_parts.circuit import NonParametricQuantumCircuit

from quri_algo.algo.utils.timer import timer

LoweringLevel = Enum(
    "LoweringLevel",
    [
        ("LogicalCircuit", 0),
        ("ArchLogicalCircuit", 1),
        ("ArchInstruction", 2),
        ("DeviceInstruction", 3),
    ],
)


class AnalyzeResult(Protocol):
    """Analyze result protocol for consistency with the VM interface."""

    lowering_level: LoweringLevel
    qubit_count: int
    gate_count: int
    depth: int
    latency: TimeValue | None = None
    fidelity: float | None = None


Analyzer: TypeAlias = Callable[
    [NonParametricQuantumCircuit], AnalyzeResult
]  # Provides an interface for QURI VM


class AlgorithmResult(ABC):
    """Algorithm result."""

    def __init__(self, algorithm: "Algorithm", elapsed_time: Optional[float] = None):
        self.algorithm = algorithm
        self.elapsed_time = elapsed_time

    @property
    def name(self) -> str:
        """The name of the algorithm."""
        return self.algorithm.name


class VariationalAlgorithmResultMixin(ABC):
    def __init__(self, optimizer_history: Sequence[OptimizerState]) -> None:
        self.optimizer_history = optimizer_history

    @property
    def optimizer_result(self) -> OptimizerState:
        return self.optimizer_history[-1]

    @property
    def cost_function_value(self) -> float:
        return self.optimizer_result.cost

    @property
    def cost_function_history(self) -> Sequence[float]:
        return [optimizer_state.cost for optimizer_state in self.optimizer_history]


T = TypeVar("T")


class Analysis(ABC):
    """Analysis of the algorithm.

    Properties:
        lowering_level: The level of analysis, e.g., logical, arch, device.
        circuit_gate_count: gate count of each circuit,
        circuit_depth: circuit depth of each circuit,
        circuit_latency: latency of each circuit,
        circuit_execution_count: execution count of each circuit,
        circuit_fidelities: fidelity of each circuit,
        circuit_physical_qubit_count: required number of physical qubits for each circuit.,
    """

    def __init__(
        self,
        lowering_level: LoweringLevel,
        circuit_gate_count: Mapping[T, int],
        circuit_depth: Mapping[T, int],
        circuit_latency: Mapping[T, TimeValue | None],
        circuit_execution_count: Mapping[T, int],
        circuit_fidelities: Mapping[T, float | None],
        circuit_qubit_count: Mapping[T, int],
    ) -> None:
        self.lowering_level = lowering_level
        self.circuit_gate_count = circuit_gate_count
        self.circuit_depth = circuit_depth
        self.circuit_latency = circuit_latency
        self.circuit_execution_count = circuit_execution_count
        self.circuit_fidelities = circuit_fidelities
        self.circuit_qubit_count = circuit_qubit_count

    @property
    @abstractmethod
    def total_latency(self) -> TimeValue:
        """Total latency of the circuit is algorithm dependent."""
        pass

    @property
    @abstractmethod
    def max_physical_qubit_count(self) -> int:
        """Maximum physical qubit count is algorithm dependent."""
        pass


class Algorithm(ABC):
    """Base class for all algorithms."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the algorithm."""
        pass

    @timer
    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> AlgorithmResult:
        """Run the algorithm itself."""
        pass

    def __str__(self) -> str:
        """Basic information about the algorithm should be returned."""
        return self.name


class CircuitAnalysisMixin(Protocol):
    @abstractmethod
    def analyze(self, *args: Any, **kwargs: Any) -> Analysis:
        """The quantum resource analysis of the algorithm."""
        pass


class QuantumAlgorithm(Algorithm, CircuitAnalysisMixin, ABC):
    ...
