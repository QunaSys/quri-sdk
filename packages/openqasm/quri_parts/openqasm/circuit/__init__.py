# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
from typing import TYPE_CHECKING, Mapping

from quri_parts.circuit import gate_names
from quri_parts.circuit.gate_names import (
    is_gate_name,
    is_multi_qubit_gate_name,
    is_parametric_gate_name,
    is_single_qubit_gate_name,
    is_three_qubit_gate_name,
    is_two_qubit_gate_name,
)

if TYPE_CHECKING:
    from quri_parts.circuit import NonParametricQuantumCircuit, QuantumGate
    from quri_parts.circuit.gate_names import (
        ParametricGateNameType,
        SingleQubitGateNameType,
        ThreeQubitGateNameType,
        TwoQubitGateNameType,
    )

_HEADER = """OPENQASM 3;
include "stdgates.inc";"""
_QUBIT_VAR_NAME = "q"

# For information on "stdgates.inc", see https://arxiv.org/abs/2104.14722v2

_single_qubit_gate_stdgates_symbol: Mapping["SingleQubitGateNameType", str] = {
    gate_names.Identity: "id",
    gate_names.X: "x",
    gate_names.Y: "y",
    gate_names.Z: "z",
    gate_names.H: "h",
    gate_names.S: "s",
    gate_names.Sdag: "sdag",
    gate_names.T: "t",
    gate_names.Tdag: "tdag",
}

_single_qubit_rotation_gate_stdgates_symbol: Mapping["SingleQubitGateNameType", str] = {
    gate_names.RX: "rx",
    gate_names.RY: "ry",
    gate_names.RZ: "rz",
}

_two_qubit_gate_stdgates_symbol: Mapping["TwoQubitGateNameType", str] = {
    gate_names.CNOT: "cx",
    gate_names.CZ: "cz",
    gate_names.SWAP: "swap",
}

_three_qubit_gate_stdgates_symbol: Mapping["ThreeQubitGateNameType", str] = {
    gate_names.TOFFOLI: "ccx",
}

_parametric_gate_stdgates_symbol: Mapping["ParametricGateNameType", str] = {
    gate_names.ParametricRX: "rx",
    gate_names.ParametricRY: "ry",
    gate_names.ParametricRZ: "rz",
}

_U_gate_stdgates_symbol: Mapping["SingleQubitGateNameType", str] = {
    gate_names.U1: "u1",
    gate_names.U2: "u2",
    gate_names.U3: "u3",
}

_not_implemented_gates: set["SingleQubitGateNameType"] = {
    gate_names.SqrtX,
    gate_names.SqrtXdag,
    gate_names.SqrtY,
    gate_names.SqrtYdag,
}


def convert_to_qasm(
    circuit: "NonParametricQuantumCircuit", text_io: io.TextIOBase
) -> None:
    """Converts a circuit to OpenQASM and writes it to IO stream.

    Args:
        circuit: Circuit to be converted
        text_io: Stream where output will be written
    """
    text_io.write(_HEADER + "\n")
    text_io.write(f"qubit[{int(circuit.qubit_count)}] {_QUBIT_VAR_NAME};\n")
    for gate in circuit.gates:
        text_io.write("\n")
        text_io.write(convert_gate_to_qasm_line(gate))


def convert_to_qasm_str(circuit: "NonParametricQuantumCircuit") -> str:
    str_io = io.StringIO()
    convert_to_qasm(circuit, str_io)
    return str_io.getvalue()


def _ref_q_str(index: int) -> str:
    return f"{_QUBIT_VAR_NAME}[{index}]"


def convert_gate_to_qasm_line(gate: "QuantumGate") -> str:
    """Converts a gate to OpenQASM format.

    Args:
        gate: gate to be converted

    Returns:
        OpenQASM string
    """
    if not is_gate_name(gate.name):
        raise ValueError(f"Unknown gate name: {gate.name}")

    if is_single_qubit_gate_name(gate.name):
        if gate.name in _single_qubit_gate_stdgates_symbol:
            gate_str = _single_qubit_gate_stdgates_symbol[gate.name]
            t_q_str = _ref_q_str(gate.target_indices[0])
            return f"{gate_str} {t_q_str};"

        elif gate.name in _single_qubit_rotation_gate_stdgates_symbol:
            gate_str = _single_qubit_rotation_gate_stdgates_symbol[gate.name]
            t_q_str = _ref_q_str(gate.target_indices[0])
            return f"{gate_str}({gate.params[0]}) {t_q_str};"

        elif gate.name in _U_gate_stdgates_symbol:
            gate_str = _U_gate_stdgates_symbol[gate.name]
            params = f"{', '.join(str(p) for p in gate.params)}"
            t_q_str = _ref_q_str(gate.target_indices[0])
            return f"{gate_str}({params}) {t_q_str};"

        elif gate.name in _not_implemented_gates:
            raise NotImplementedError()

    elif is_two_qubit_gate_name(gate.name):
        if gate.name in _two_qubit_gate_stdgates_symbol:
            gate_str = _two_qubit_gate_stdgates_symbol[gate.name]
            c_q_str, t_q_str = [
                _ref_q_str(i)
                for i in tuple(gate.control_indices) + tuple(gate.target_indices)
            ]
            return f"{gate_str} {c_q_str}, {t_q_str};"

    elif is_three_qubit_gate_name(gate.name):
        if gate.name in _three_qubit_gate_stdgates_symbol:
            gate_str = _three_qubit_gate_stdgates_symbol[gate.name]
            q_str1, q_str2, q_str3 = [
                _ref_q_str(i)
                for i in tuple(gate.control_indices) + tuple(gate.target_indices)
            ]
            return f"{gate_str} {q_str1}, {q_str2}, {q_str3};"

    elif is_parametric_gate_name(gate.name):
        raise ValueError("Parametric gates are not supported.")
    elif is_multi_qubit_gate_name(gate.name):
        raise ValueError("multi qubit gates are not supported.")

    assert False, "Unreachable"
