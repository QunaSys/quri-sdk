# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Mapping, Sequence
from typing import Optional, Union

from qiskit import transpile
from qiskit.providers import Backend

from quri_parts.circuit import ImmutableQuantumCircuit, gate_names
from quri_parts.circuit.gate_names import GateNameType
from quri_parts.circuit.transpile import CircuitTranspilerProtocol
from quri_parts.qiskit.circuit import circuit_from_qiskit, convert_circuit
from quri_parts.qiskit.circuit.gate_names import ECR, QiskitGateNameType

_qp_qiskit_gate_name_map: Mapping[Union[GateNameType, QiskitGateNameType], str] = {
    gate_names.Identity: "id",
    gate_names.X: "x",
    gate_names.Y: "y",
    gate_names.Z: "z",
    gate_names.H: "h",
    gate_names.S: "s",
    gate_names.Sdag: "sdg",
    gate_names.SqrtX: "sx",
    gate_names.SqrtXdag: "sxdg",
    gate_names.T: "t",
    gate_names.Tdag: "tdg",
    gate_names.RX: "rx",
    gate_names.RY: "ry",
    gate_names.RZ: "rz",
    gate_names.U3: "u",
    gate_names.CNOT: "cx",
    gate_names.CZ: "cz",
    ECR: "ecr",
    gate_names.SWAP: "swap",
    gate_names.TOFFOLI: "ccx",
    gate_names.Measurement: "measure",
}


class QiskitTranspiler(CircuitTranspilerProtocol):
    """A CircuitTranspiler that uses Qiskit's transpiler to convert circuits to
    backend-compatible circuits, convert gate sets, perform circuit
    optimization, etc.

    This transpiler converts ImmutableQuantumCircuit to ImmutableQuantumCircuit
    just like other transpilers in QURI Parts though the conversion of the circuit to
    Qiskit and vice versa is performed internally.

    Args:
        backend: Qiskit's Backend instance. If specified, the gate set for the device
            is used for the output and the basis_gates option is ignored.
        basis_gates: Specify the gate set after decomposition as a list of gate name
            strings. If omitted, all gates compatible with Qiskit may exist in the
            output.
        optimization_level: Specifies the optimization level of the circuit.
    """

    def __init__(
        self,
        backend: Optional[Backend] = None,
        basis_gates: Optional[Sequence[Union[GateNameType, QiskitGateNameType]]] = None,
        optimization_level: Optional[int] = None,
    ):
        self._basis_gates: Optional[list[str]] = None
        if basis_gates is not None:
            self._basis_gates = [_qp_qiskit_gate_name_map[name] for name in basis_gates]

        self._backend = backend
        self._optimization_level = optimization_level

    def __call__(self, circuit: ImmutableQuantumCircuit) -> ImmutableQuantumCircuit:
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
