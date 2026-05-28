# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Collection, Mapping
from typing import Any, Iterable

import quri_parts.circuit.transpile as qt
from quri_parts.circuit import NonParametricQuantumCircuit, QuantumCircuit, gate_names
from quri_parts.qsub.eval.quriparts import (
    convert_op,
    primitive_op_gate_mapping,
    primitive_param_op_gate_mapping,
)
from quri_parts.qsub.lib import std
from quri_parts.qsub.op import BaseIdent
from quri_parts.qsub.qubit import Qubit

from .transpiler import Operations, SeparateTranspiler


class _Identity(Mapping[Qubit, Qubit]):
    def __getitem__(self, key: Qubit) -> Qubit:
        return key

    def __iter__(self) -> Any:
        raise NotImplementedError()

    def __len__(self) -> Any:
        raise NotImplementedError()


def convert_to_qp(ops: Operations) -> NonParametricQuantumCircuit:
    qc = 1 + max(max(qubit.uid for qubit in qs) for _, qs, _ in ops)
    circ = QuantumCircuit(qc)
    for op, qubits, regs in ops:
        circ.add_gate(convert_op(op, qubits, regs, _Identity()))
    return circ


_op_gate_map_qp = {
    gate_names.Identity: std.Identity,
    gate_names.X: std.X,
    gate_names.Y: std.Y,
    gate_names.Z: std.Z,
    gate_names.H: std.H,
    gate_names.S: std.S,
    gate_names.Sdag: std.Sdag,
    gate_names.SqrtX: std.SqrtX,
    gate_names.SqrtXdag: std.SqrtXdag,
    gate_names.SqrtY: std.SqrtY,
    gate_names.SqrtYdag: std.SqrtYdag,
    gate_names.T: std.T,
    gate_names.TOFFOLI: std.Toffoli,
    gate_names.Tdag: std.Tdag,
    gate_names.CNOT: std.CNOT,
    gate_names.CZ: std.CZ,
    gate_names.SWAP: std.SWAP,
}

_param_op_gate_map_qp = {
    gate_names.RX: std.RX,
    gate_names.RY: std.RY,
    gate_names.RZ: std.RZ,
    gate_names.U1: std.Phase,
}

_mc_op_get_map_qp = {
    gate_names.MCX: std.MCX,
    gate_names.MCY: std.MCY,
    gate_names.MCZ: std.MCZ,
    gate_names.MCS: std.MCS,
    gate_names.MCSdag: std.MCSdag,
    gate_names.MCT: std.MCT,
    gate_names.MCTdag: std.MCTdag,
    gate_names.MCSqrtX: std.MCSqrtX,
    gate_names.MCSqrtXdag: std.MCSqrtXdag,
    gate_names.MCSqrtY: std.MCSqrtY,
    gate_names.MCSqrtYdag: std.MCSqrtYdag,
    gate_names.MCH: std.MCH,
}

_mc_param_op_get_map_qp = {
    gate_names.MCRX: std.MCRX,
    gate_names.MCRY: std.MCRY,
    gate_names.MCRZ: std.MCRZ,
    gate_names.MCU1: std.MCPhase,
}


def convert_from_qp(circuit: NonParametricQuantumCircuit) -> Operations:
    ops = []
    for gate in circuit.gates:
        qubits = tuple(
            Qubit(i) for i in tuple(gate.control_indices) + tuple(gate.target_indices)
        )
        if gate.name in _op_gate_map_qp:
            ops.append((_op_gate_map_qp[gate.name], qubits, ()))
        elif gate.name in _param_op_gate_map_qp:
            assert len(gate.control_indices) == 0
            op = _param_op_gate_map_qp[gate.name](gate.params[0])
            ops.append((op, qubits, ()))
        elif gate.name in _mc_op_get_map_qp:
            assert len(gate.target_indices) == 1
            op = _mc_op_get_map_qp[gate.name](len(gate.control_indices))
            ops.append((op, qubits, ()))
        elif gate.name in _mc_param_op_get_map_qp:
            assert len(gate.target_indices) == 1
            op = _mc_param_op_get_map_qp[gate.name](
                len(gate.control_indices), gate.params[0]
            )
            ops.append((op, qubits, ()))
        else:
            raise ValueError(f"Conversion of {gate.name} from qp is not supported.")
    return ops


class SeparateQURIPartsTranspiler(SeparateTranspiler):
    def __init__(self, qp_trans: Iterable[qt.CircuitTranspiler]) -> None:
        self._transpilers = qp_trans

    @property
    def target_ops(self) -> Collection[BaseIdent]:
        return primitive_op_gate_mapping.keys() | primitive_param_op_gate_mapping.keys()

    def transpile_chunk(self, ops: Operations) -> Operations:
        circ = convert_to_qp(ops)
        for trans in self._transpilers:
            circ = trans(circ)
        return convert_from_qp(circ)
