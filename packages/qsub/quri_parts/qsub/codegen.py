from collections.abc import Iterable

from .machineinst import MachineInst, MachineSub, Primitive, SubCall
from .op import AbstractOp, Op
from .sub import Sub


class CodeGenerator:
    def __init__(self, ops: Iterable[AbstractOp]):
        self._primitives = set([op.base_id for op in ops])

    def is_primitive(self, op: Op) -> bool:
        return op.base_id in self._primitives

    def lower(self, sub: Sub) -> MachineSub:
        instrs: list[MachineInst] = []
        for op, qubits, regs in sub.operations:
            if self.is_primitive(op):
                instrs.append((Primitive(op), qubits, regs))
            else:
                instrs.append((SubCall(op), qubits, regs))
        return MachineSub(
            sub.qubits, sub.registers, sub.aux_qubits, sub.aux_registers, instrs
        )
