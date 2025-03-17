from collections import defaultdict
from collections.abc import Sequence

from ..evaluate import EvaluatorHooks
from ..machineinst import MachineSub, Primitive, SubId
from ..qubit import Qubit
from ..register import Register


class AuxQubitCountEvaluatorHooks(EvaluatorHooks[int]):
    def __init__(self) -> None:
        self._max_aux_count = 0

        self._cache: dict[SubId, int] = defaultdict(int)

    def result(self) -> int:
        return self._max_aux_count

    def _merge_cache(self, caller_id: SubId, callee_id: SubId) -> None:
        self._cache[caller_id] = max(self._cache[caller_id], self._cache[callee_id])

    def enter_sub(
        self,
        sub: MachineSub,
        qubits: Sequence[Qubit],
        regs: Sequence[Register],
        call_stack: list[SubId],
    ) -> bool:
        return sub.sub_id not in self._cache

    def exit_sub(
        self, sub: MachineSub, enter_sub: bool, call_stack: list[SubId]
    ) -> None:
        if enter_sub:
            self._cache[sub.sub_id] += len(sub.aux_qubits)
        if len(call_stack) > 1:
            self._merge_cache(call_stack[-2], sub.sub_id)
        else:
            self._max_aux_count = self._cache[sub.sub_id]

    def primitive(
        self,
        mop: Primitive,
        qubits: Sequence[Qubit],
        regs: Sequence[Register],
        call_stack: list[SubId],
    ) -> None:
        ...
