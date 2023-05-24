# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

from qiskit import transpile
from qiskit.providers import Backend

from quri_parts.circuit import NonParametricQuantumCircuit
from quri_parts.circuit.transpile import CircuitTranspilerProtocol
from quri_parts.qiskit.circuit import circuit_from_qiskit, convert_circuit


class QiskitTranspiler(CircuitTranspilerProtocol):
    """A CircuitTranspiler that uses Qiskit's transpiler to convert circuits to
    backend-compatible circuits, convert gate sets, perform circuit
    optimization, etc.

    This transpiler converts NonParametricQuantumCircuit to NonParametricQuantumCircuit
    just like other transpilers in QURI Parts though the conversion of the circuit to
    Qiskit and vice versa is performed internally.

    Args:
        backend: Qiskit's Backend instance.
        basis_gates: Specify the gate set after decomposition as a list of gate name
            strings. The gate name notation follows Qiskit.
        optimization_level: Specifies the optimization level of the circuit.
    """

    def __init__(
        self,
        backend: Optional[Backend] = None,
        basis_gates: Optional[list[str]] = None,
        optimization_level: Optional[int] = None,
    ):
        self._backend = backend
        self._basis_gates = basis_gates
        self._optimization_level = optimization_level

    def __call__(
        self, circuit: NonParametricQuantumCircuit
    ) -> NonParametricQuantumCircuit:
        qiskit_circ = convert_circuit(circuit)
        optimized_qiskit_circ = transpile(
            qiskit_circ,
            backend=self._backend,
            basis_gates=self._basis_gates,
            optimization_level=self._optimization_level,
        )
        return circuit_from_qiskit(optimized_qiskit_circ)


__all__ = [
    "QiskitTranspiler",
]
