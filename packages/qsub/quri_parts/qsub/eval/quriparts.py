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
from typing import Callable, Optional, cast

from quri_parts.circuit import QuantumCircuit, QuantumGate, gates

from ..allocate import QubitAllocator
from ..codegen import CodeGenerator
from ..evaluate import EvaluatorHooks
from ..lib import std
from ..link import Linker
from ..machineinst import MachineOp, MachineSub, Primitive, SubId
from ..op import BaseIdent
from ..qubit import Qubit
from ..register import Register

quri_parts_codegen = CodeGenerator(
    [
        std.CNOT,
        std.CZ,
        std.H,
        std.Identity,
        std.S,
        std.Sdag,
        std.SqrtX,
        std.SqrtXdag,
        std.SqrtY,
        std.SqrtYdag,
        std.SWAP,
        std.T,
        std.Tdag,
        std.Toffoli,
        std.X,
        std.Y,
        std.Z,
        std.RX,
        std.RY,
        std.RZ,
    ],
)
quri_parts_linker = Linker({})


primitive_op_gate_mapping: Mapping[
    BaseIdent,
    Callable[[int], QuantumGate]
    | Callable[[int, int], QuantumGate]
    | Callable[[int, int, int], QuantumGate],
] = {
    std.CNOT.base_id: gates.CNOT,
    std.CZ.base_id: gates.CZ,
    std.H.base_id: gates.H,
    std.Identity.base_id: gates.Identity,
    std.S.base_id: gates.S,
    std.Sdag.base_id: gates.Sdag,
    std.SqrtX.base_id: gates.SqrtX,
    std.SqrtXdag.base_id: gates.SqrtXdag,
    std.SqrtY.base_id: gates.SqrtY,
    std.SqrtYdag.base_id: gates.SqrtYdag,
    std.SWAP.base_id: gates.SWAP,
    std.T.base_id: gates.T,
    std.Tdag.base_id: gates.Tdag,
    std.Toffoli.base_id: gates.TOFFOLI,
    std.X.base_id: gates.X,
    std.Y.base_id: gates.Y,
    std.Z.base_id: gates.Z,
}

primitive_param_op_gate_mapping: Mapping[
    BaseIdent,
    Callable[[int, float], QuantumGate],
] = {
    std.RX.base_id: gates.RX,
    std.RY.base_id: gates.RY,
    std.RZ.base_id: gates.RZ,
    std.Phase.base_id: gates.RZ,
}


def _convert_op(
    mop: MachineOp,
    qubits: Sequence[Qubit],
    regs: Sequence[Register],
    qubit_map: Mapping[Qubit, Qubit],
) -> QuantumGate:
    if mop.op.base_id in primitive_op_gate_mapping:
        return primitive_op_gate_mapping[mop.op.base_id](
            *[qubit_map[q].uid for q in qubits]
        )
    elif mop.op.base_id in primitive_param_op_gate_mapping:
        return primitive_param_op_gate_mapping[mop.op.id.base](
            qubit_map[qubits[0]].uid, cast(float, mop.op.id.params[0])
        )
    else:
        raise ValueError(f"Op mapping to QuantumGate is not supported for {mop.op.id}.")


class QURIPartsEvaluatorHooks(EvaluatorHooks[QuantumCircuit]):
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._gates: list[QuantumGate] = []
        self._qubit_map_stack: list[dict[Qubit, Qubit]] = []
        self._qubit_map: Optional[dict[Qubit, Qubit]] = None
        self._allocator: Optional[QubitAllocator] = None

    def result(self) -> QuantumCircuit:
        qubit_index = 0
        for gate in self._gates:
            qubit_index = max(
                qubit_index,
                *gate.control_indices,
                *gate.target_indices,
            )
        circ = QuantumCircuit(qubit_index + 1)
        circ.extend(self._gates)
        return circ

    def _update_qubit_map(self) -> None:
        self._qubit_map = {}

        for qubit_map in reversed(self._qubit_map_stack):
            rep = self._qubit_map.copy()
            for k, v in self._qubit_map.items():
                if v in qubit_map:
                    rep[k] = qubit_map[v]
            self._qubit_map = qubit_map | rep

    def enter_sub(
        self,
        sub: MachineSub,
        qubits: Sequence[Qubit],
        regs: Sequence[Register],
        call_stack: list[SubId],
    ) -> bool:
        if self._allocator is None:
            self._allocator = QubitAllocator()
            self._qubit_map_stack.append(dict(self._allocator.allocate_map(sub.qubits)))
        else:
            self._qubit_map_stack.append(dict(zip(sub.qubits, qubits)))
        self._qubit_map_stack.append(dict(self._allocator.allocate_map(sub.aux_qubits)))
        self._update_qubit_map()
        return True

    def exit_sub(
        self, sub: MachineSub, enter_sub: bool, call_stack: list[SubId]
    ) -> None:
        if self._allocator is None:
            raise ValueError("Uninitialized allocator")
        self._allocator.free_last(len(sub.aux_qubits))
        self._qubit_map_stack.pop()
        self._qubit_map_stack.pop()
        self._update_qubit_map()

    def primitive(
        self,
        mop: Primitive,
        qubits: Sequence[Qubit],
        regs: Sequence[Register],
        call_stack: list[SubId],
    ) -> None:
        if self._qubit_map is None:
            raise ValueError("Uninitialized qubit mapping")
        self._gates.append(_convert_op(mop, qubits, regs, self._qubit_map))
