# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC
from typing import TYPE_CHECKING, Optional, Union, cast

import numpy as np
from typing_extensions import TypeAlias

from quri_parts.circuit.circuit import GateSequence, ImmutableQuantumCircuit

from ..utils.array import readonly_array
from .state import CircuitQuantumStateMixin, QuantumState

if TYPE_CHECKING:
    import numpy.typing as npt

#: A type alias representing a numerical state vector,
#: equivalent to np.ndarray of complex floats.
StateVectorType: TypeAlias = "npt.NDArray[np.complex128]"


class QuantumStateVectorMixin(ABC):
    def __init__(
        self,
        n_qubits: int,
        vector: Optional[Union[StateVectorType, "npt.ArrayLike"]] = None,
    ) -> None:
        self._dim = 2**n_qubits
        self._vector: StateVectorType
        if vector is None:
            self._vector = cast(StateVectorType, np.zeros(self._dim))
            self._vector[0] = 1.0
        else:
            vector = np.asarray(vector, dtype=np.complex128)
            if len(vector) != self._dim:
                raise ValueError(f"The dimension of vector must be {self._dim}.")
            self._vector = vector

    @property
    def vector(self) -> StateVectorType:
        return readonly_array(self._vector)


class QuantumStateVector(
    QuantumStateVectorMixin, CircuitQuantumStateMixin, QuantumState
):
    """QuantumStateVector represents a state defined by a state vector with an
    optional circuit to be applied."""

    def __init__(
        self,
        n_qubits: int,
        vector: Optional[Union[StateVectorType, "npt.ArrayLike"]] = None,
        circuit: Optional[ImmutableQuantumCircuit] = None,
    ):
        self._n_qubits = n_qubits
        QuantumStateVectorMixin.__init__(self, n_qubits, vector)
        CircuitQuantumStateMixin.__init__(self, n_qubits, circuit)

    def __repr__(self) -> str:
        return (
            f"QuantumStateVector(n_qubits={self.qubit_count}, vector={self.vector}, "
            f"circuit={self.circuit})"
        )

    @property
    def qubit_count(self) -> int:
        return self._n_qubits

    def with_gates_applied(self, gates: GateSequence) -> "QuantumStateVector":
        """Returns a new state with the gates applied.

        The original state is not changed.
        """
        circuit = self._circuit + gates
        return QuantumStateVector(self._n_qubits, self.vector, circuit)
