# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from quri_parts.circuit import (
    NonParametricQuantumCircuit,
    QuantumCircuit,
    gate_names,
    gates,
)

from .transpiler import CircuitTranspilerProtocol


class IdentityInsertionTranspiler(CircuitTranspilerProtocol):
    """If there are qubits to which any gate has not been applied, Identity
    gates are added for those qubits.

    The application of this transpiler ensures that every qubit has at
    least one gate acting on it.
    """

    def __call__(
        self, circuit: NonParametricQuantumCircuit
    ) -> NonParametricQuantumCircuit:
        non_applied = set(range(circuit.qubit_count))
        for gate in circuit.gates:
            for q in tuple(gate.control_indices) + tuple(gate.target_indices):
                non_applied.discard(q)
            if not non_applied:
                return circuit

        cc = QuantumCircuit(circuit.qubit_count)
        cc.extend(circuit.gates)
        for q in non_applied:
            cc.add_gate(gates.Identity(q))
        return cc


class IdentityEliminationTranspiler(CircuitTranspilerProtocol):
    """Generate a new circuit by removing all Identity gates from the given
    circuit."""

    def __call__(
        self, circuit: NonParametricQuantumCircuit
    ) -> NonParametricQuantumCircuit:
        ret = QuantumCircuit(circuit.qubit_count)
        for gate in circuit.gates:
            if gate.name != gate_names.Identity:
                ret.add_gate(gate)
        return ret
