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
from typing import Any, Optional, Protocol, runtime_checkable

from quri_parts.circuit import NonParametricQuantumCircuit
from quri_parts.circuit.transpile import CircuitTranspiler

from .utils.transpile import apply_transpiler


@runtime_checkable
class CircuitFactory(Protocol):
    transpiler: Optional[CircuitTranspiler]
    qubit_count: int

    @apply_transpiler  # type: ignore
    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> NonParametricQuantumCircuit:
        ...
