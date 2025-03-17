import copy
from collections.abc import Mapping
from typing import TypeAlias

from .machineinst import MachineInst, MachineSub, SubCall, is_subcall
from .op import Op

CallTable: TypeAlias = Mapping[Op, MachineSub]


class Linker:
    def __init__(self, calltable: CallTable):
        self._calltable = copy.deepcopy(calltable)
        link_table(self._calltable)

    @property
    def calltable(self) -> CallTable:
        return copy.deepcopy(self._calltable)

    def link(self, sub: MachineSub) -> MachineSub:
        return link(sub, self._calltable)


# Link sub using calltable. Calltable also is linked and updated.
def link(sub: MachineSub, calltable: CallTable) -> MachineSub:
    link_table(calltable)
    instrs: list[MachineInst] = []
    for mop, qubits, regs in sub.instructions:
        if is_subcall(mop):
            if mop.op not in calltable:
                raise ValueError(f"Op {mop.op} is not found in calltable.")
            instrs.append((SubCall(mop.op, calltable[mop.op]), qubits, regs))
        else:
            instrs.append((mop, qubits, regs))
    return MachineSub(
        sub.qubits, sub.registers, sub.aux_qubits, sub.aux_registers, tuple(instrs)
    )


# Link subs in calltable each other. Rewrite calltable.
def link_table(calltable: CallTable) -> None:
    for sub in calltable.values():
        for mop, qubits, regs in sub.instructions:
            if is_subcall(mop):
                if mop.op not in calltable:
                    raise ValueError(f"Op {mop.op} is not found in calltable.")
                mop.sub = calltable[mop.op]
