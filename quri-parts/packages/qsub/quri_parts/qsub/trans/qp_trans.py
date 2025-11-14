# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Collection, Sequence
from typing import cast

import quri_parts.circuit.transpile as qt
from quri_parts.circuit import NonParametricQuantumCircuit, QuantumCircuit, gate_names
from quri_parts.qsub.eval.quriparts import (
    primitive_op_gate_mapping,
    primitive_param_op_gate_mapping,
)
from quri_parts.qsub.lib import std
from quri_parts.qsub.op import BaseIdent
from quri_parts.qsub.qubit import Qubit

from .transpiler import Operations, SeparateTranspiler


def convert_to_qp(ops: Operations) -> NonParametricQuantumCircuit:
    qc = 1 + max(max(qubit.uid for qubit in qs) for _, qs, _ in ops)
    circ = QuantumCircuit(qc)
    for op, qubits, regs in ops:
        if op.base_id in primitive_op_gate_mapping:
            circ.add_gate(
                primitive_op_gate_mapping[op.base_id](*[q.uid for q in qubits])
            )
        elif op.base_id in primitive_param_op_gate_mapping:
            circ.add_gate(
                primitive_param_op_gate_mapping[op.base_id](
                    qubits[0].uid, cast(float, op.id.params[0])
                )
            )
        else:
            raise ValueError(f"Conversion of {op.base_id} to qp is no supported.")
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
}


def convert_from_qp(circuit: NonParametricQuantumCircuit) -> Operations:
    ops = []
    for gate in circuit.gates:
        if gate.name in _op_gate_map_qp:
            qubits = tuple(gate.control_indices) + tuple(gate.target_indices)
            ops.append(
                (_op_gate_map_qp[gate.name], tuple(Qubit(i) for i in qubits), ())
            )
        elif gate.name in _param_op_gate_map_qp:
            op = _param_op_gate_map_qp[gate.name](gate.params[0])
            ops.append((op, tuple(Qubit(i) for i in gate.target_indices), ()))
        else:
            raise ValueError(f"Conversion of {gate.name} from qp is not supported.")
    return ops


class SeparateQURIPartsTranspiler(SeparateTranspiler):
    def __init__(self, qp_trans: Sequence[qt.CircuitTranspiler]) -> None:
        self._transpilers = qp_trans

    @property
    def target_ops(self) -> Collection[BaseIdent]:
        return primitive_op_gate_mapping.keys() | primitive_param_op_gate_mapping.keys()

    def transpile_chunk(self, ops: Operations) -> Operations:
        circ = convert_to_qp(ops)
        for trans in self._transpilers:
            circ = trans(circ)
        return convert_from_qp(circ)
