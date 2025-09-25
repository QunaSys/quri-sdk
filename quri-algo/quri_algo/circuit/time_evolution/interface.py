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
from typing import Protocol, runtime_checkable

from quri_parts.circuit import NonParametricQuantumCircuit

from quri_algo.circuit.interface import CircuitFactory
from quri_algo.circuit.utils.transpile import apply_transpiler


@runtime_checkable
class TimeEvolutionCircuitFactory(CircuitFactory, Protocol):
    """Encode a Hamiltonian to a time evolution circuit."""

    @apply_transpiler
    @abstractmethod
    def __call__(self, evolution_time: float) -> NonParametricQuantumCircuit:
        ...


@runtime_checkable
class ControlledTimeEvolutionCircuitFactory(CircuitFactory, Protocol):
    """Encode a Hamiltonian to a controlled-time evolution circuit."""

    @apply_transpiler
    @abstractmethod
    def __call__(self, evolution_time: float) -> NonParametricQuantumCircuit:
        ...
