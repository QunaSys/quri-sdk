# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Literal, Mapping, Sequence

from stim import Circuit as StimCircuit
from typing_extensions import TypeAlias, TypeGuard

from quri_parts.circuit import (
    ImmutableQuantumCircuit,
    QuantumGate,
    gate_names,
    is_clifford,
)
from quri_parts.circuit.gate_names import (
    CLIFFORD_GATE_NAMES,
    CNOT,
    CZ,
    SWAP,
    H,
    Identity,
    S,
    Sdag,
    SqrtX,
    SqrtXdag,
    SqrtY,
    SqrtYdag,
    X,
    Y,
    Z,
)
from quri_parts.circuit.transpile import (
    CliffordApproximationTranspiler,
    PauliDecomposeTranspiler,
)

_StimGateNameType: TypeAlias = Literal[
    "Identity",
    "X",
    "Y",
    "Z",
    "H",
    "S",
    "Sdag",
    "SqrtX",
    "SqrtXdag",
    "SqrtY",
    "SqrtYdag",
    "CNOT",
    "CZ",
    "SWAP",
]

STIM_GATE_NAMES: set[_StimGateNameType] = {
    Identity,
    X,
    Y,
    Z,
    H,
    S,
    Sdag,
    SqrtX,
    SqrtXdag,
    SqrtY,
    SqrtYdag,
    CNOT,
    CZ,
    SWAP,
}

_stim_gate_str: Mapping[_StimGateNameType, str] = {
    gate_names.Identity: "I",
    gate_names.X: "X",
    gate_names.Y: "Y",
    gate_names.Z: "Z",
    gate_names.H: "H",
    gate_names.S: "S",
    gate_names.Sdag: "S_DAG",
    gate_names.SqrtX: "SQRT_X",
    gate_names.SqrtXdag: "SQRT_X_DAG",
    gate_names.SqrtY: "SQRT_Y",
    gate_names.SqrtYdag: "SQRT_Y_DAG",
    gate_names.CNOT: "CNOT",
    gate_names.CZ: "CZ",
    gate_names.SWAP: "SWAP",
}


def _is_stim_supported_gate_name(gate_name: str) -> TypeGuard[_StimGateNameType]:
    return gate_name in STIM_GATE_NAMES


def convert_gate(gate: QuantumGate) -> Sequence[tuple[str, Sequence[int]]]:
    """Converts a :class:`~QuantumGate` to the list of the gate's name and the
    qubit indices applied."""
    if gate.name in CLIFFORD_GATE_NAMES:
        if gate.name == gate_names.Pauli:
            transpiled_gates = PauliDecomposeTranspiler().decompose(gate)
        else:
            transpiled_gates = [gate]
    elif is_clifford(gate):
        transpiled_gates = CliffordApproximationTranspiler().decompose(gate)
    else:
        raise ValueError(f"{gate.name} is not a Clifford gate.")

    ret = []
    for gate in transpiled_gates:
        if _is_stim_supported_gate_name(gate.name):
            stim_gate_str = _stim_gate_str[gate.name]
            targets = [*gate.control_indices, *gate.target_indices]
            ret.append((stim_gate_str, targets))
        else:
            assert False, "Unreachable"
    return ret


def convert_circuit(circuit: ImmutableQuantumCircuit) -> StimCircuit:
    """Converts a :class:`~ImmutableQuantumCircuit` to
    :class:`stim.Circuit`."""
    gate_str = ""
    for gate in circuit.gates:
        clifford_gates = convert_gate(gate)
        for cliff_gate in clifford_gates:
            s_indices = [str(index) for index in cliff_gate[1]]
            gate_str += f"{cliff_gate[0]} {' '.join(s_indices)} \n"
    return StimCircuit(gate_str)


__all__ = ["convert_circuit", "convert_gate"]
