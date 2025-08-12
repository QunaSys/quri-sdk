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
from typing import Any, Optional, Sequence

from quri_parts.algo.optimizer import OptimizerState
from quri_parts.backend.units import TimeUnit, TimeValue
from quri_parts.circuit import ImmutableQuantumCircuit, NonParametricQuantumCircuit

from quri_algo.algo.interface import (
    Algorithm,
    AlgorithmResult,
    Analysis,
    AnalyzeResult,
    LoweringLevel,
    QuantumAlgorithm,
    VariationalAlgorithmResultMixin,
)
from quri_algo.algo.utils.mappings import CircuitMapping
from quri_algo.algo.utils.timer import timer
from quri_algo.algo.utils.variational_solvers import VariationalSolver
from quri_algo.core.cost_functions.base_classes import CostFunction


class CompilationResult(AlgorithmResult, VariationalAlgorithmResultMixin):
    def __init__(
        self,
        algorithm: Algorithm,
        optimized_circuit: NonParametricQuantumCircuit,
        optimizer_history: Sequence[OptimizerState],
        elapsed_time: Optional[float] = None,
    ):
        AlgorithmResult.__init__(self, algorithm, elapsed_time)
        VariationalAlgorithmResultMixin.__init__(self, optimizer_history)
        self.optimized_circuit = optimized_circuit


class CompilationAnalysis(Analysis):
    def __init__(
        self,
        lowering_level: LoweringLevel,
        circuit_gate_count: CircuitMapping[int],
        circuit_depth: CircuitMapping[int],
        circuit_latency: CircuitMapping[TimeValue | None],
        circuit_execution_count: CircuitMapping[int],
        circuit_fidelities: CircuitMapping[float | None],
        circuit_qubit_count: CircuitMapping[int],
        concurrency: int = 1,
    ):
        super().__init__(
            lowering_level,
            circuit_gate_count,
            circuit_depth,
            circuit_latency,
            circuit_execution_count,
            circuit_fidelities,
            circuit_qubit_count,
        )

        self.concurrency = concurrency

    @classmethod
    def from_circuit_list(
        cls,
        circuit_analysis: AnalyzeResult,
        total_circuit_executions: int,
        circuit_list: Sequence[ImmutableQuantumCircuit],
        concurrency: int = 1,
    ) -> "CompilationAnalysis":
        return cls(
            lowering_level=circuit_analysis.lowering_level,
            circuit_depth=CircuitMapping(
                [(c, circuit_analysis.depth) for c in circuit_list]
            ),
            circuit_gate_count=CircuitMapping(
                [(c, circuit_analysis.gate_count) for c in circuit_list]
            ),
            circuit_latency=CircuitMapping(
                [(c, circuit_analysis.latency) for c in circuit_list]
            ),
            circuit_execution_count=CircuitMapping(
                [(c, total_circuit_executions) for c in circuit_list]
            ),
            circuit_fidelities=CircuitMapping(
                [(c, circuit_analysis.fidelity) for c in circuit_list]
            ),
            circuit_qubit_count=CircuitMapping(
                [(c, circuit_analysis.qubit_count) for c in circuit_list]
            ),
            # The analysis is the same for all circuits
            concurrency=concurrency,
        )

    @property
    def total_latency(self) -> TimeValue:
        """Total latency of the circuit."""
        latency_in_ns = 0.0
        for c, l in self.circuit_latency.items():
            if l is None:
                continue
            n = (self.circuit_execution_count[c] - 1) // self.concurrency + 1
            latency_in_ns += l.in_ns() * n
        return TimeValue(latency_in_ns, TimeUnit.NANOSECOND)

    @property
    def max_physical_qubit_count(self) -> int:
        """Total number of physical qubits of the circuit."""
        qubit_counts = [c * self.concurrency for c in self.circuit_qubit_count.values()]
        return max(qubit_counts)


class QuantumCompiler(QuantumAlgorithm, ABC):
    @property
    @abstractmethod
    def cost_function(self) -> CostFunction:
        ...

    @property
    @abstractmethod
    def solver(self) -> VariationalSolver:
        ...

    @timer
    @abstractmethod
    def run(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> CompilationResult:
        pass
