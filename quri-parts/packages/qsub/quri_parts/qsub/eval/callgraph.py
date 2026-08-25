# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Sequence
from typing import Optional

from ..evaluate import EvaluatorHooks
from ..machineinst import MachineSub, Primitive, SubId, is_subcall
from ..op import Ident
from ..qubit import Qubit
from ..register import Register


class CallGraphEvaluatorHooks(EvaluatorHooks[dict[Optional[Ident], list[Ident]]]):
    """Report the call graph of the evaluated sub as an adjacency mapping.

    Each key is a caller sub and its value lists the distinct subs it
    calls directly, in first-call order; subs calling no other sub map
    to an empty list. Nodes are op ``Ident``\\ s, the same identity
    :class:`SubwiseProfileEntry` reports as ``source_op``, so the two
    reports join on them; use :meth:`Ident.to_str` for a display name.
    The key is ``None`` for a root sub whose source op is unknown (the
    entry sub of ``compile_sub``, which receives a ``Sub`` rather than
    an ``Op``). Parametric subs are distinct nodes per parameter
    (``MCX<5>`` and ``MCX<6>`` are separate nodes).
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        # Insertion order is the order of the reported nodes.
        self._callees: dict[SubId, list[Ident]] = {}
        self._nodes: dict[SubId, Optional[Ident]] = {}
        self._order: list[SubId] = []
        self._result: dict[Optional[Ident], list[Ident]] = {}

    def result(self) -> dict[Optional[Ident], list[Ident]]:
        return self._result

    def enter_sub(
        self,
        sub: MachineSub,
        qubits: Sequence[Qubit],
        regs: Sequence[Register],
        call_stack: list[SubId],
    ) -> bool:
        sid = sub.sub_id
        if sid in self._callees:
            return False

        self._order.append(sid)
        # A sub is named by the op calling it; only the root has no call
        # site, and falls back to the op it was resolved from.
        self._nodes.setdefault(sid, sub.source_op)
        callees: list[Ident] = []
        for mop, _, _ in sub.instructions:
            if is_subcall(mop) and mop.sub is not None:
                self._nodes[mop.sub.sub_id] = mop.op.id
                if mop.op.id not in callees:
                    callees.append(mop.op.id)
        self._callees[sid] = callees

        return True

    def primitive(
        self,
        mop: Primitive,
        qubits: Sequence[Qubit],
        regs: Sequence[Register],
        call_stack: list[SubId],
    ) -> None:
        pass

    def exit_sub(
        self, sub: MachineSub, enter_sub: bool, call_stack: list[SubId]
    ) -> None:
        if len(call_stack) == 1:
            self._result = {self._nodes[sid]: self._callees[sid] for sid in self._order}
