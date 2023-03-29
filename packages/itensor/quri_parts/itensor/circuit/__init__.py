from collections.abc import Mapping

import juliacall
from juliacall import Main as jl

from quri_parts.circuit import NonParametricQuantumCircuit, gate_names
from quri_parts.circuit.gate_names import (
    SingleQubitGateNameType,
    ThreeQubitGateNameType,
    TwoQubitGateNameType,
    is_gate_name,
    is_single_qubit_gate_name,
    is_three_qubit_gate_name,
    is_two_qubit_gate_name,
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


def convert_circuit(
    circuit: NonParametricQuantumCircuit, qubit_sites: juliacall.VectorValue
) -> juliacall.VectorValue:
    """Convert an :class:`~NonParametricQuantumCircuit` to an ITensor ops.

    qubit_sites: collection of N "Qubit" sites. please follow
    `the Itensor doc <https://itensor.github.io/ITensors.jl/
    stable/IncludedSiteTypes.html#%22Qubit%22-SiteType>`_
    """
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
                raise ValueError(f"Unknown single qubit gate name: {gate.name}")
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
            raise ValueError(f"Unknown gate name: {gate.name}")
    circuit = jl.ops(gate_list, qubit_sites)
    return circuit
