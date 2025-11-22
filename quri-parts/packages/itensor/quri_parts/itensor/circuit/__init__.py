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
from typing import Callable, Optional

import juliacall
from juliacall import Main as jl

from quri_parts.circuit import ImmutableQuantumCircuit, gate_names
from quri_parts.circuit.gate_names import (
    SingleQubitGateNameType,
    ThreeQubitGateNameType,
    TwoQubitGateNameType,
    is_gate_name,
    is_single_qubit_gate_name,
    is_three_qubit_gate_name,
    is_two_qubit_gate_name,
)
from quri_parts.circuit.transpile import (
    CircuitTranspiler,
    PauliDecomposeTranspiler,
    PauliRotationDecomposeTranspiler,
    SequentialTranspiler,
)

_single_qubit_gate_itensor: Mapping[SingleQubitGateNameType, str] = {
    gate_names.Identity: "I",
    gate_names.X: "X",
    gate_names.Y: "Y",
    gate_names.Z: "Z",
    gate_names.H: "H",
    gate_names.S: "S",
    gate_names.Sdag: "Sdag",
    gate_names.SqrtX: "√X",
    gate_names.SqrtXdag: "√Xdag",
    gate_names.SqrtY: "√Y",
    gate_names.SqrtYdag: "√Ydag",
    gate_names.T: "T",
    gate_names.Tdag: "Tdag",
}

_single_qubit_rotation_gate_itensor: Mapping[SingleQubitGateNameType, str] = {
    gate_names.RX: "Rx",
    gate_names.RY: "Ry",
    gate_names.RZ: "Rz",
    gate_names.U1: "U1",
    gate_names.U2: "U2",
    gate_names.U3: "U3",
}

_two_qubit_gate_itensor: Mapping[TwoQubitGateNameType, str] = {
    gate_names.CNOT: "CNOT",
    gate_names.CZ: "CZ",
    gate_names.SWAP: "SWAP",
}

_three_qubit_gate_itensor: Mapping[ThreeQubitGateNameType, str] = {
    gate_names.TOFFOLI: "Toffoli",
}

#: CircuitTranspiler to convert a circit configuration suitable for ITensor.
ITensorSetTranspiler: Callable[[], CircuitTranspiler] = lambda: SequentialTranspiler(
    [PauliDecomposeTranspiler(), PauliRotationDecomposeTranspiler()]
)


def convert_circuit(
    circuit: ImmutableQuantumCircuit,
    qubit_sites: juliacall.VectorValue,
    transpiler: Optional[CircuitTranspiler] = ITensorSetTranspiler(),
) -> juliacall.VectorValue:
    """Convert an :class:`~ImmutableQuantumCircuit` to an ITensor ops.

    qubit_sites: collection of N "Qubit" sites. please follow
    `the Itensor doc <https://itensor.github.io/ITensors.jl/
    stable/IncludedSiteTypes.html#%22Qubit%22-SiteType>`_
    """
    if transpiler is not None:
        circuit = transpiler(circuit)

    gate_list: juliacall.VectorValue = jl.gate_list()
    for gate in circuit.gates:
        if not is_gate_name(gate.name):
            raise ValueError(f"Unknown gate name: {gate.name}")

        if is_single_qubit_gate_name(gate.name):
            if gate.name in _single_qubit_gate_itensor:
                gate_list = jl.add_single_qubit_gate(
                    gate_list,
                    _single_qubit_gate_itensor[gate.name],
                    gate.target_indices[0] + 1,
                )
            elif gate.name in _single_qubit_rotation_gate_itensor:
                if len(gate.params) == 1:
                    gate_list = jl.add_single_qubit_rotation_gate(
                        gate_list,
                        _single_qubit_rotation_gate_itensor[gate.name],
                        gate.target_indices[0] + 1,
                        gate.params[0],
                    )
                elif len(gate.params) == 2:
                    gate_list = jl.add_single_qubit_rotation_gate(
                        gate_list,
                        _single_qubit_rotation_gate_itensor[gate.name],
                        gate.target_indices[0] + 1,
                        gate.params[0],
                        gate.params[1],
                    )
                elif len(gate.params) == 3:
                    gate_list = jl.add_single_qubit_rotation_gate(
                        gate_list,
                        _single_qubit_rotation_gate_itensor[gate.name],
                        gate.target_indices[0] + 1,
                        gate.params[0],
                        gate.params[1],
                        gate.params[2],
                    )
                else:
                    raise ValueError("Invalid number of parameters.")
            else:
                raise ValueError(f"{gate.name} gate is not supported.")
        elif is_two_qubit_gate_name(gate.name):
            if gate.name == "SWAP":
                gate_list = jl.add_two_qubit_gate(
                    gate_list,
                    _two_qubit_gate_itensor[gate.name],
                    gate.target_indices[0] + 1,
                    gate.target_indices[1] + 1,
                )
            else:
                gate_list = jl.add_two_qubit_gate(
                    gate_list,
                    _two_qubit_gate_itensor[gate.name],
                    gate.control_indices[0] + 1,
                    gate.target_indices[0] + 1,
                )
        elif is_three_qubit_gate_name(gate.name):
            gate_list = jl.add_three_qubit_gate(
                gate_list,
                _three_qubit_gate_itensor[gate.name],
                gate.control_indices[0] + 1,
                gate.control_indices[1] + 1,
                gate.target_indices[0] + 1,
            )
        else:
            raise ValueError(f"{gate.name} gate is not supported.")
    circuit = jl.ops(gate_list, qubit_sites)
    return circuit
