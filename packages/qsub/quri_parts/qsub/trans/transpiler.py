from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable, Sequence
from dataclasses import replace
from typing import Callable, Protocol, TypeAlias

from quri_parts.qsub.op import BaseIdent, Op
from quri_parts.qsub.qubit import Qubit
from quri_parts.qsub.register import Register
from quri_parts.qsub.sub import Sub

SubTranspiler: TypeAlias = Callable[[Sub], Sub]
Operations: TypeAlias = Sequence[tuple[Op, Sequence[Qubit], Sequence[Register]]]


class SubTranspilerProtocol(Protocol):
    def __call__(self, sub: Sub) -> Sub:
        ...


class SequentialTranspiler(SubTranspilerProtocol):
    def __init__(self, transpilers: Iterable[SubTranspiler]):
        self._transpilers = transpilers

    def __call__(self, sub: Sub) -> Sub:
        for transpiler in self._transpilers:
            sub = transpiler(sub)
        return sub


class SeparateTranspiler(SubTranspilerProtocol, ABC):
    @property
    @abstractmethod
    def target_ops(self) -> Collection[BaseIdent]:
        ...

    @abstractmethod
    def transpile_chunk(self, ops: Operations) -> Operations:
        ...

    def __call__(self, sub: Sub) -> Sub:
        ops = sub.operations
        rops = []
        i = 0
        while i < len(ops):
            while i < len(ops) and ops[i][0].base_id not in self.target_ops:
                rops.append(ops[i])
                i += 1

            chunk = []
            while i < len(ops) and ops[i][0].base_id in self.target_ops:
                chunk.append(ops[i])
                i += 1
            if chunk:
                rops.extend(self.transpile_chunk(chunk))

        return replace(sub, operations=rops)
