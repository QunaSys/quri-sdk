# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterable
from typing import Optional

from .machineinst import MachineInst, MachineSub, Primitive, SubCall
from .op import AbstractOp, Ident, Op
from .sub import Sub


class CodeGenerator:
    def __init__(self, ops: Iterable[AbstractOp]):
        self._primitives = set([op.base_id for op in ops])

    def is_primitive(self, op: Op) -> bool:
        return op.base_id in self._primitives

    def lower(self, sub: Sub, source_op_id: Optional[Ident] = None) -> MachineSub:
        """Lower ``sub`` to a :class:`MachineSub`.

        Args:
            sub: The sub to lower.
            source_op_id: Identifier of the op ``sub`` implements, recorded on
                the result so evaluators can name it. Leave ``None`` for a sub
                that does not implement an op, such as an entry sub.
        """
        instrs: list[MachineInst] = []
        for op, qubits, regs in sub.operations:
            if self.is_primitive(op):
                instrs.append((Primitive(op), qubits, regs))
            else:
                instrs.append((SubCall(op), qubits, regs))
        return MachineSub(
            sub.qubits,
            sub.registers,
            sub.aux_qubits,
            sub.aux_registers,
            instrs,
            source_op_id,
        )
