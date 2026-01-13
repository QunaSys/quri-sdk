# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Protocol, Union

import qulacs as ql
from numpy import complex128
from numpy.typing import NDArray

from quri_parts.circuit import ImmutableQuantumCircuit
from quri_parts.qulacs.circuit.compiled_circuit import _QulacsCircuit
from quri_parts.qulacs.utils import cast_to_list

QulacsCircuitT = Union[ImmutableQuantumCircuit, _QulacsCircuit]


class QulacsBackend(Protocol):
    def init_state(
        self, circuit: QulacsCircuitT, init_state: NDArray[complex128]
    ) -> ql.QuantumState:
        ...

    def should_use_multinomial(self, n_shots: int, qubit_count: int) -> bool:
        ...

    def validate_state_vector(
        self, vector: NDArray[complex128], qubit_count: int
    ) -> None:
        ...


@dataclass(frozen=True)
class DefaultQulacsBackend:
    def init_state(
        self, circuit: QulacsCircuitT, init_state: NDArray[complex128]
    ) -> ql.QuantumState:
        if len(init_state) != 2**circuit.qubit_count:
            raise ValueError("Inconsistent qubit length between circuit and state")

        qulacs_state = ql.QuantumState(circuit.qubit_count)
        qulacs_state.load(cast_to_list(init_state))
        return qulacs_state

    def should_use_multinomial(self, n_shots: int, qubit_count: int) -> bool:
        return n_shots > int(2 ** max(int(qubit_count), 10))

    def validate_state_vector(
        self, vector: NDArray[complex128], qubit_count: int
    ) -> None:
        return None


DEFAULT_BACKEND = DefaultQulacsBackend()
