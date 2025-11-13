# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import abstractmethod
from typing import Any, Callable, Protocol, TypeAlias, runtime_checkable

from quri_parts.core.state import CircuitQuantumState

from quri_algo.circuit.time_evolution.interface import TimeEvolutionCircuitFactory
from quri_algo.problem.operators.hamiltonian import QubitHamiltonian
from quri_algo.state.interface import TimeEvolutionStateFactory

HamiltonianMapping = Callable[[float], QubitHamiltonian]
TECircuitFactoryConstructor: TypeAlias = Callable[
    [QubitHamiltonian], TimeEvolutionCircuitFactory
]


@runtime_checkable
class AdiabaticTimeEvolutionStateFactoryBase(TimeEvolutionStateFactory, Protocol):
    """Base class for adiabatic state preparation."""

    hamiltonian_mapping: HamiltonianMapping
    TECircuitFactory: TECircuitFactoryConstructor

    @abstractmethod
    def __init__(
        self,
        hamiltonian_mapping: HamiltonianMapping,
        TECircuitFactory: TECircuitFactoryConstructor,
    ):
        ...

    @abstractmethod
    def __call__(
        self,
        evolution_time: float,
        discretization: int,
        initial_state: CircuitQuantumState,
        *args: Any,
        **kwargs: Any
    ) -> CircuitQuantumState:
        ...
