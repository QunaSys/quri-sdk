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
from typing import Optional

from pytket import OpType, passes
from pytket.backends import Backend

from quri_parts.circuit import ImmutableQuantumCircuit, gate_names
from quri_parts.circuit.gate_names import GateNameType
from quri_parts.circuit.transpile import CircuitTranspilerProtocol
from quri_parts.tket.circuit import circuit_from_tket, convert_circuit

_qp_tket_gate_name_map: Mapping[GateNameType, OpType] = {
    gate_names.Identity: OpType.noop,
    gate_names.X: OpType.X,
    gate_names.Y: OpType.Y,
    gate_names.Z: OpType.Z,
    gate_names.H: OpType.H,
    gate_names.S: OpType.S,
    gate_names.Sdag: OpType.Sdg,
    gate_names.SqrtX: OpType.SX,
    gate_names.SqrtXdag: OpType.SXdg,
    gate_names.T: OpType.T,
    gate_names.Tdag: OpType.Tdg,
    gate_names.RX: OpType.Rx,
    gate_names.RY: OpType.Ry,
    gate_names.RZ: OpType.Rz,
    gate_names.U1: OpType.U1,
    gate_names.U2: OpType.U2,
    gate_names.U3: OpType.U3,
    gate_names.CNOT: OpType.CX,
    gate_names.CZ: OpType.CZ,
    gate_names.SWAP: OpType.SWAP,
    gate_names.TOFFOLI: OpType.CCX,
}


class TketTranspiler(CircuitTranspilerProtocol):
    """A CircuitTranspiler that uses Tket's transpiler to convert circuits to
    backend-compatible circuits, convert gate sets, perform circuit
    optimization, etc.

    This transpiler converts ImmutableQuantumCircuit to ImmutableQuantumCircuit
    just like other transpilers in QURI Parts though the conversion of the circuit to
    Tket and vice versa is performed internally.

    If the backend is specified, the circuit is transformed and optimized to a form
    executable in the backend at the optimization level specified in optimization_level.
    Since the corresponding gate set in the backend takes precedence, basis_gates
    argument is ignored.

    If the backend is not specified, conversion to the gate set specified in basis_gates
    and optimization in optimization_level are performed.

    Note that this transpiler may perform optimization assuming that the input state of
    all qubits is |0>.

    Args:
        backend: Tket's Backend instance. If specified, the gate set for the device
            is used for the output and the basis_gates option is ignored.
        basis_gates: Specify the gate set after decomposition as a list of gate name
            strings.
        optimization_level: Specifies the optimization level of the circuit from 0 to 3.

    Refs:
        https://cqcl.github.io/pytket/manual/manual_compiler.html
    """

    def __init__(
        self,
        backend: Optional[Backend] = None,
        basis_gates: Optional[Sequence[GateNameType]] = None,
        optimization_level: int = 3,
    ):
        if not (0 <= optimization_level <= 3):
            raise ValueError("optimization_level must be 0 to 3.")

        self._basis_gates: Optional[set[OpType]] = None
        if basis_gates is not None:
            self._basis_gates = {_qp_tket_gate_name_map[name] for name in basis_gates}

        self._backend = backend
        self._optimization_level = optimization_level

    def __call__(self, circuit: ImmutableQuantumCircuit) -> ImmutableQuantumCircuit:
        tket_circ = convert_circuit(circuit)

        if self._backend is not None:
            self._backend.default_compilation_pass(
                optimisation_level=self._optimization_level
            ).apply(tket_circ)
            return circuit_from_tket(tket_circ)

        pass_list = []
        if self._optimization_level == 1:
            pass_list.append(passes.SynthesiseTket())
        elif self._optimization_level >= 2:
            pass_list.append(passes.FullPeepholeOptimise())

        if self._basis_gates is not None:
            if hasattr(passes, "auto_rebase_pass"):
                pass_list.append(passes.auto_rebase_pass(self._basis_gates))
            else:
                pass_list.append(passes.AutoRebase(self._basis_gates))

        passes.SequencePass(pass_list).apply(tket_circ)
        return circuit_from_tket(tket_circ)


__all__ = [
    "TketTranspiler",
]
