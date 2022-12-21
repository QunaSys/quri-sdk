# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Mapping

from quri_parts.circuit import NonParametricQuantumCircuit, QuantumCircuit

from .transpiler import CircuitTranspilerProtocol


class QubitRemappingTranspiler(CircuitTranspilerProtocol):
    """Remap qubits in the circuit with the specified mapping.

    The mapping ``qubit_mapping`` should be specified with "from" qubit
    indices as keys and "to" qubit indices as values. For example, if
    you want to convert a circuit using qubits 0, 1, 2, 3 by mapping
    them as 0 → 4, 1 → 2, 2 → 5, 3 → 0, then the ``qubit_mapping``
    should be ``{0: 4, 1: 2, 2: 5, 3: 0}``. The ``qubit_count`` of the
    converted circuit is determined by the largest destination qubit
    index. In the above example, the largest index is 5, so the
    converted circuit is for 6 qubits.
    """

    def __init__(self, qubit_mapping: Mapping[int, int]):
        if len(qubit_mapping) != len(set(qubit_mapping.values())):
            raise ValueError(
                f"qubit_mapping has duplicated indices in values: {qubit_mapping}"
            )
        self._qubit_mapping = qubit_mapping
        self._max_index = max(qubit_mapping.values())

    def __call__(
        self, circuit: NonParametricQuantumCircuit
    ) -> NonParametricQuantumCircuit:
        transpiled = QuantumCircuit(self._max_index + 1)
        qm = self._qubit_mapping
        try:
            for gate in circuit.gates:
                ci = tuple(qm[index] for index in gate.control_indices)
                ti = tuple(qm[index] for index in gate.target_indices)
                g = gate._replace(control_indices=ci, target_indices=ti)
                transpiled.add_gate(g)
        except KeyError as e:
            raise ValueError(f"Mapping for qubit {e.args} was not specified")

        return transpiled
