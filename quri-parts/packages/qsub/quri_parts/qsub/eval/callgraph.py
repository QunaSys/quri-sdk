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


class CallGraphEvaluatorHooks(EvaluatorHooks[dict[Optional[Ident], dict[Ident, int]]]):
    """Report the call graph of a sub.

    The result maps each sub to the subs it calls directly. Each
    callee maps to the number of calls in one invocation of the
    caller. Callees appear in the order of the first call. A sub that
    calls no sub maps to an empty dict.

    Each sub appears as the ``Ident`` of the op it implements, the
    same value :class:`SubwiseProfileEntry` reports as ``source_op``.
    Parametric subs are distinct per parameter: ``MCX<5>`` and
    ``MCX<6>`` are two subs. Only the entry sub of ``compile_sub`` has
    no op; its key is ``None``.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        # Nodes are reported in the order subs are first entered, so both
        # dicts rely on insertion order.
        self._caller_to_callees: dict[SubId, dict[Ident, int]] = {}
        self._nodes: dict[SubId, Optional[Ident]] = {}

    def result(self) -> dict[Optional[Ident], dict[Ident, int]]:
        return {
            self._nodes[sid]: callees
            for sid, callees in self._caller_to_callees.items()
        }

    def enter_sub(
        self,
        sub: MachineSub,
        qubits: Sequence[Qubit],
        regs: Sequence[Register],
        call_stack: list[SubId],
    ) -> bool:
        sid = sub.sub_id
        if sid in self._caller_to_callees:
            return False

        # A sub is named by the op calling it; only the root has no call
        # site, and falls back to the op it was resolved from.
        self._nodes.setdefault(sid, sub.source_op_id)
        callees: dict[Ident, int] = {}
        for mop, _, _ in sub.instructions:
            if is_subcall(mop) and mop.sub is not None:
                self._nodes[mop.sub.sub_id] = mop.op.id
                callees[mop.op.id] = callees.get(mop.op.id, 0) + 1
        self._caller_to_callees[sid] = callees

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
        pass
