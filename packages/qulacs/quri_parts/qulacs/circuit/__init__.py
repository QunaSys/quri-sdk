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
from typing import Callable, Union, cast

import qulacs
from numpy.typing import ArrayLike
from typing_extensions import assert_never

from quri_parts.circuit import (
    LinearMappedUnboundParametricQuantumCircuitBase,
    NonParametricQuantumCircuit,
    QuantumGate,
    UnboundParametricQuantumCircuitBase,
    UnboundParametricQuantumCircuitProtocol,
    gate_names,
)
from quri_parts.circuit.gate_names import (
    MultiQubitGateNameType,
    SingleQubitGateNameType,
    ThreeQubitGateNameType,
    TwoQubitGateNameType,
    is_gate_name,
    is_multi_qubit_gate_name,
    is_parametric_gate_name,
    is_single_qubit_gate_name,
    is_three_qubit_gate_name,
    is_two_qubit_gate_name,
    is_unitary_matrix_gate_name,
)

from .. import cast_to_list
from .compiled_circuit import compile_circuit, compile_parametric_circuit

_single_qubit_gate_qulacs: Mapping[
    SingleQubitGateNameType, Callable[[int], qulacs.QuantumGateBase]
] = {
    gate_names.Identity: qulacs.gate.Identity,
    gate_names.X: qulacs.gate.X,
    gate_names.Y: qulacs.gate.Y,
    gate_names.Z: qulacs.gate.Z,
    gate_names.H: qulacs.gate.H,
    gate_names.S: qulacs.gate.S,
    gate_names.Sdag: qulacs.gate.Sdag,
    gate_names.SqrtX: qulacs.gate.sqrtX,
    gate_names.SqrtXdag: qulacs.gate.sqrtXdag,
    gate_names.SqrtY: qulacs.gate.sqrtY,
    gate_names.SqrtYdag: qulacs.gate.sqrtYdag,
    gate_names.T: qulacs.gate.T,
    gate_names.Tdag: qulacs.gate.Tdag,
}


def _u1_gate_qulacs(gate: QuantumGate) -> qulacs.QuantumGateBase:
    return cast(
        qulacs.QuantumGateBase,
        qulacs.gate.U1(*gate.target_indices, *gate.params),
    )


def _u2_gate_qulacs(gate: QuantumGate) -> qulacs.QuantumGateBase:
    return cast(
        qulacs.QuantumGateBase,
        qulacs.gate.U2(*gate.target_indices, *gate.params),
    )


def _u3_gate_qulacs(gate: QuantumGate) -> qulacs.QuantumGateBase:
    return cast(
        qulacs.QuantumGateBase,
        qulacs.gate.U3(*gate.target_indices, *gate.params),
    )


_single_qubit_reverse_rotation_gate_qulacs: Mapping[
    SingleQubitGateNameType, Callable[[int, float], qulacs.QuantumGateBase]
] = {
    gate_names.RX: qulacs.gate.RX,
    gate_names.RY: qulacs.gate.RY,
    gate_names.RZ: qulacs.gate.RZ,
}

_two_qubit_gate_qulacs: Mapping[
    TwoQubitGateNameType, Callable[[int, int], qulacs.QuantumGateBase]
] = {
    gate_names.CNOT: qulacs.gate.CNOT,
    gate_names.CZ: qulacs.gate.CZ,
    gate_names.SWAP: qulacs.gate.SWAP,
}

_three_qubit_gate_qulacs: Mapping[
    ThreeQubitGateNameType, Callable[[int, int, int], qulacs.QuantumGateBase]
] = {
    gate_names.TOFFOLI: qulacs.gate.TOFFOLI,
}

_multi_pauli_gate_qulacs: Mapping[
    MultiQubitGateNameType,
    Callable[[list[int], list[int]], qulacs.QuantumGateBase],
] = {
    gate_names.Pauli: qulacs.gate.Pauli,
}

_multi_pauli_rotation_gate_qulacs: Mapping[
    MultiQubitGateNameType,
    Callable[[list[int], list[int], float], qulacs.QuantumGateBase],
] = {
    gate_names.PauliRotation: qulacs.gate.PauliRotation,
}

_parametric_gate_qulacs = {
    gate_names.ParametricRX: qulacs.gate.ParametricRX,
    gate_names.ParametricRY: qulacs.gate.ParametricRY,
    gate_names.ParametricRZ: qulacs.gate.ParametricRZ,
    gate_names.ParametricPauliRotation: qulacs.gate.ParametricPauliRotation,
}


def _dense_matrix_gate_qulacs(
    t: Union[int, Sequence[int]], unitary_matrix: ArrayLike
) -> qulacs.QuantumGateBase:
    # We need to disable type check due to an error in qulacs type annotation
    # https://github.com/qulacs/qulacs/issues/537
    return cast(
        qulacs.QuantumGateBase,
        qulacs.gate.DenseMatrix(t, unitary_matrix),  # type: ignore
    )


def convert_gate(
    gate: QuantumGate,
) -> qulacs.QuantumGateBase:
    if not is_gate_name(gate.name):
        raise ValueError(f"Unknown gate name: {gate.name}")

    if is_single_qubit_gate_name(gate.name):
        if gate.name in _single_qubit_gate_qulacs:
            return _single_qubit_gate_qulacs[gate.name](
                *gate.target_indices, *gate.params
            )
        elif gate.name == gate_names.U1:
            return _u1_gate_qulacs(gate)
        elif gate.name == gate_names.U2:
            return _u2_gate_qulacs(gate)
        elif gate.name == gate_names.U3:
            return _u3_gate_qulacs(gate)
        elif gate.name in _single_qubit_reverse_rotation_gate_qulacs:
            neg_params = (-p for p in gate.params)
            return _single_qubit_reverse_rotation_gate_qulacs[gate.name](
                *gate.target_indices, *neg_params
            )
        else:
            assert False, "Unreachable"
    elif is_two_qubit_gate_name(gate.name):
        return _two_qubit_gate_qulacs[gate.name](
            *gate.control_indices, *gate.target_indices
        )
    elif is_three_qubit_gate_name(gate.name):
        return _three_qubit_gate_qulacs[gate.name](
            *gate.control_indices, *gate.target_indices
        )
    elif is_multi_qubit_gate_name(gate.name):
        # These cast are workaround for too strict type annotation of Qulacs
        target_indices = cast_to_list(gate.target_indices)
        pauli_ids = cast_to_list(gate.pauli_ids)
        if gate.name in _multi_pauli_gate_qulacs:
            return _multi_pauli_gate_qulacs[gate.name](target_indices, pauli_ids)
        elif gate.name in _multi_pauli_rotation_gate_qulacs:
            neg_params = (-p for p in gate.params)
            return _multi_pauli_rotation_gate_qulacs[gate.name](
                target_indices, pauli_ids, *neg_params
            )
        else:
            assert False, "Unreachable"
    elif is_unitary_matrix_gate_name(gate.name):
        return _dense_matrix_gate_qulacs(gate.target_indices, gate.unitary_matrix)
    elif is_parametric_gate_name(gate.name):
        raise ValueError("Parametric gates are not supported")
    else:
        # It seems that currently assert_never does not work here
        # assert_never(gate.name)
        assert False, "Unreachable"


def convert_circuit(circuit: NonParametricQuantumCircuit) -> qulacs.QuantumCircuit:
    qulacs_circuit = qulacs.QuantumCircuit(circuit.qubit_count)
    for gate in circuit.gates:
        qulacs_circuit.add_gate(convert_gate(gate))
    return qulacs_circuit


def convert_parametric_circuit(
    circuit: UnboundParametricQuantumCircuitProtocol,
) -> tuple[
    qulacs.ParametricQuantumCircuit, Callable[[Sequence[float]], Sequence[float]]
]:
    param_circuit: UnboundParametricQuantumCircuitBase
    param_mapper: Callable[[Sequence[float]], Sequence[float]]
    if isinstance(circuit, LinearMappedUnboundParametricQuantumCircuitBase):
        param_mapping = circuit.param_mapping
        param_circuit = circuit.primitive_circuit()
        orig_param_mapper = param_mapping.seq_mapper

        def param_mapper(s: Sequence[float]) -> Sequence[float]:
            return tuple(-p for p in orig_param_mapper(s))

    elif isinstance(circuit, UnboundParametricQuantumCircuitBase):
        param_circuit = circuit

        def param_mapper(s: Sequence[float]) -> Sequence[float]:
            return tuple(-p for p in s)

    else:
        raise ValueError(f"Unsupported parametric circuit type: {type(circuit)}")

    qulacs_circuit = qulacs.ParametricQuantumCircuit(circuit.qubit_count)
    for gate, _ in param_circuit._gates:
        if is_parametric_gate_name(gate.name):
            if gate.name == gate_names.ParametricRX:
                qulacs_circuit.add_parametric_RX_gate(gate.target_indices[0], 0)
            elif gate.name == gate_names.ParametricRY:
                qulacs_circuit.add_parametric_RY_gate(gate.target_indices[0], 0)
            elif gate.name == gate_names.ParametricRZ:
                qulacs_circuit.add_parametric_RZ_gate(gate.target_indices[0], 0)
            elif gate.name == gate_names.ParametricPauliRotation:
                # These cast are workaround for too strict type annotation of Qulacs
                target_indices = cast_to_list(gate.target_indices)
                pauli_ids = cast_to_list(gate.pauli_ids)
                qulacs_circuit.add_parametric_multi_Pauli_rotation_gate(
                    target_indices, pauli_ids, 0
                )
            else:
                assert_never(gate.name)
        else:
            assert isinstance(
                gate, QuantumGate
            ), f"gate should be a QuantumGate. actual={gate}"
            qulacs_circuit.add_gate(convert_gate(gate))

    return qulacs_circuit, param_mapper


__all__ = [
    "convert_gate",
    "convert_circuit",
    "convert_parametric_circuit",
    "compile_circuit",
    "compile_parametric_circuit",
]
