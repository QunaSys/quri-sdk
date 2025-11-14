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
from typing import Callable, Dict, Optional

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
        std.MCH,
        std.MCRX,
        std.MCRY,
        std.MCRZ,
        std.MCS,
        std.MCSdag,
        std.MCSqrtX,
        std.MCSqrtXdag,
        std.MCSqrtY,
        std.MCSqrtYdag,
        std.MCT,
        std.MCTdag,
        std.MCPhase,
        std.MCX,
        std.MCY,
        std.MCZ,
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
    std.Phase.base_id: gates.U1,
}

mc_op_gate_mapping: Dict[BaseIdent, Callable[[int, list[int]], QuantumGate]] = {
    std.MCX.base_id: gates.MCX,
    std.MCY.base_id: gates.MCY,
    std.MCZ.base_id: gates.MCZ,
    std.MCS.base_id: gates.MCS,
    std.MCSdag.base_id: gates.MCSdag,
    std.MCT.base_id: gates.MCT,
    std.MCTdag.base_id: gates.MCTdag,
    std.MCSqrtX.base_id: gates.MCSqrtX,
    std.MCSqrtXdag.base_id: gates.MCSqrtXdag,
    std.MCSqrtY.base_id: gates.MCSqrtY,
    std.MCSqrtYdag.base_id: gates.MCSqrtYdag,
    std.MCH.base_id: gates.MCH,
}

mc_param_op_gate_mapping: Dict[
    BaseIdent, Callable[[int, float, list[int]], QuantumGate]
] = {
    std.MCRX.base_id: gates.MCRX,
    std.MCRY.base_id: gates.MCRY,
    std.MCRZ.base_id: gates.MCRZ,
    std.MCPhase.base_id: gates.MCU1,
}

mc_primitive_op_gate_mapping: Mapping[
    BaseIdent, Callable[[int, list[int]], QuantumGate]
] = {
    # TODO:
    std.X.base_id: gates.MCX,
    std.Y.base_id: gates.MCY,
    std.Z.base_id: gates.MCZ,
    std.H.base_id: gates.MCH,
    std.S.base_id: gates.MCS,
    std.Sdag.base_id: gates.MCSdag,
    std.T.base_id: gates.MCT,
    std.Tdag.base_id: gates.MCTdag,
    std.SqrtX.base_id: gates.MCSqrtX,
    std.SqrtXdag.base_id: gates.MCSqrtXdag,
    std.SqrtY.base_id: gates.MCSqrtY,
    std.SqrtYdag.base_id: gates.MCSqrtYdag,
}
mc_primitive_param_op_gate_mapping: Mapping[
    BaseIdent,
    Callable[[int, float, list[int]], QuantumGate],
] = {
    std.RX.base_id: gates.MCRX,
    std.RY.base_id: gates.MCRY,
    std.RZ.base_id: gates.MCRZ,
    std.Phase.base_id: gates.MCU1,
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
        param_float = mop.op.id.params[0]
        assert isinstance(param_float, float)
        return primitive_param_op_gate_mapping[mop.op.id.base](
            qubit_map[qubits[0]].uid, param_float
        )
    elif mop.op.base_id in mc_op_gate_mapping:
        full_qubits = [qubit_map[q].uid for q in qubits]
        return mc_op_gate_mapping[mop.op.base_id](full_qubits[-1], full_qubits[:-1])
    elif mop.op.base_id in mc_param_op_gate_mapping:
        full_qubits = [qubit_map[q].uid for q in qubits]
        angle = mop.op.id.params[1]
        assert isinstance(angle, float)
        return mc_param_op_gate_mapping[mop.op.base_id](
            full_qubits[-1], angle, full_qubits[:-1]
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
        self._cache: dict[
            tuple[SubId, tuple[int, ...], Sequence[Register], int], list[QuantumGate]
        ] = {}
        self._gate_stack: list[list[QuantumGate]] = []
        self._arg_stack: list[
            tuple[SubId, tuple[int, ...], Sequence[Register], int]
        ] = []

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

        assert self._qubit_map is not None
        mapped_qs = tuple(self._qubit_map[q].uid for q in qubits)
        k = (sub.sub_id, mapped_qs, regs, self._allocator.total())
        if k in self._cache:
            gates = self._cache[k]
            self._gates.extend(gates)
            self._gate_stack[-1].extend(gates)
            return False

        self._gate_stack.append([])
        self._arg_stack.append(k)

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

        if enter_sub:
            gates = self._gate_stack.pop()
            self._cache[self._arg_stack.pop()] = gates
            if self._gate_stack:
                self._gate_stack[-1].extend(gates)

    def primitive(
        self,
        mop: Primitive,
        qubits: Sequence[Qubit],
        regs: Sequence[Register],
        call_stack: list[SubId],
    ) -> None:
        if self._qubit_map is None:
            raise ValueError("Uninitialized qubit mapping")

        g = _convert_op(mop, qubits, regs, self._qubit_map)
        self._gates.append(g)

        self._gate_stack[-1].append(g)
