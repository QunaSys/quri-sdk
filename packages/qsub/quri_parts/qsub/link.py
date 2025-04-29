# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
