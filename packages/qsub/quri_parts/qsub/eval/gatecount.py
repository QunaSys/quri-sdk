from collections import defaultdict
from collections.abc import Iterable, Sequence

from ..evaluate import EvaluatorHooks
from ..lib import std
from ..machineinst import MachineSub, Primitive, SubId
from ..op import AbstractOp, BaseIdent
from ..qubit import Qubit
from ..register import Register


class GateCountEvaluatorHooks(EvaluatorHooks[dict[BaseIdent, int]]):
    def __init__(self, ops: Iterable[AbstractOp] = ()) -> None:
        self._target_ops = set([op.base_id for op in ops])
        self._counts: dict[BaseIdent, int] = defaultdict(int)

        self._cache: dict[SubId, dict[BaseIdent, int]] = {}

    def result(self) -> dict[BaseIdent, int]:
        return self._counts

    def _merge_cache(self, caller_id: SubId, callee_id: SubId) -> None:
        for k, v in self._cache[callee_id].items():
            self._cache[caller_id][k] += v

    def enter_sub(
        self,
        sub: MachineSub,
        qubits: Sequence[Qubit],
        regs: Sequence[Register],
        call_stack: list[SubId],
    ) -> bool:
        if sub.sub_id in self._cache:
            return False
        else:
            self._cache[sub.sub_id] = defaultdict(int)
            return True

    def exit_sub(
        self, sub: MachineSub, enter_sub: bool, call_stack: list[SubId]
    ) -> None:
        if len(call_stack) > 1:
            self._merge_cache(call_stack[-2], sub.sub_id)
        else:
            self._counts = self._cache[sub.sub_id]

    def primitive(
        self,
        mop: Primitive,
        qubits: Sequence[Qubit],
        regs: Sequence[Register],
        call_stack: list[SubId],
    ) -> None:
        if self._target_ops and mop.op.base_id not in self._target_ops:
            return None
        sub_id = call_stack[-1]
        self._cache[sub_id][mop.op.base_id] += 1


class TGateCountEvaluatorHooks(GateCountEvaluatorHooks):
    def __init__(self) -> None:
        super().__init__([std.T])
