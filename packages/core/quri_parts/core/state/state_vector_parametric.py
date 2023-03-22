# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional, Union

from quri_parts.circuit import GateSequence, UnboundParametricQuantumCircuitProtocol

from .state import QuantumState
from .state_parametric import ParametricCircuitQuantumStateMixin
from .state_vector import QuantumStateVector, QuantumStateVectorMixin, StateVectorType

if TYPE_CHECKING:
    import numpy.typing as npt


class ParametricQuantumStateVector(
    QuantumStateVectorMixin, ParametricCircuitQuantumStateMixin, QuantumState
):
    """ParametricQuantumStateVector represents a state defined by a state
    vector with a parametric circuit applied.

    This class holds an unbound parametric circuit, thus circuit
    parameters are not bound to concrete values. Use
    :meth:`~bind_parameters` when you need to bind concrete parameter
    values.
    """

    def __init__(
        self,
        n_qubits: int,
        circuit: UnboundParametricQuantumCircuitProtocol,
        vector: Optional[Union[StateVectorType, "npt.ArrayLike"]] = None,
    ) -> None:
        self._n_qubits = n_qubits
        QuantumStateVectorMixin.__init__(self, n_qubits, vector)
        ParametricCircuitQuantumStateMixin.__init__(self, n_qubits, circuit)

    def __repr__(self) -> str:
        return (
            f"ParametricQuantumStateVector(n_qubits={self.qubit_count}, "
            f"circuit={self.parametric_circuit}, vector={self.vector})"
        )

    @property
    def qubit_count(self) -> int:
        return self._n_qubits

    def with_primitive_circuit(self) -> "ParametricQuantumStateVector":
        """Returns a new ParametricQuantumStateVector whose circuit is replaced
        with the corresponding primitive circuit.

        The original state is not changed. For details about the
        primitive circuit, please refer to `.primitive_circuit()` in
        :class:`UnboundParametricQuantumCircuitProtocol`.
        """
        return ParametricQuantumStateVector(
            self._n_qubits, self._circuit.primitive_circuit(), self.vector
        )

    def with_gates_applied(self, gates: GateSequence) -> "ParametricQuantumStateVector":
        """Returns a new state with the gates applied.

        The original state is not changed.
        """
        circuit = self.parametric_circuit.get_mutable_copy()
        circuit.extend(gates)
        return ParametricQuantumStateVector(self._n_qubits, circuit, self.vector)

    def bind_parameters(self, params: Sequence[float]) -> QuantumStateVector:
        """Returns a new state with the circuit parameters assigned concrete
        values.

        This method does not modify self but returns a newly created
        state.
        """
        circuit = self.parametric_circuit.bind_parameters(params)
        return QuantumStateVector(self._n_qubits, self.vector, circuit)
