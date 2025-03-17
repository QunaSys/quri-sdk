from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from .machineinst import (
    MachineSub,
    MachineSubRecursionError,
    Primitive,
    SubId,
    is_primitive,
    is_subcall,
)
from .qubit import Qubit
from .register import Register

T = TypeVar("T")


class EvaluatorHooks(Generic[T], ABC):
    @abstractmethod
    def result(self) -> T:
        ...

    @abstractmethod
    def enter_sub(
        self,
        sub: MachineSub,
        qubits: Sequence[Qubit],
        regs: Sequence[Register],
        call_stack: list[SubId],
    ) -> bool:
        ...

    @abstractmethod
    def exit_sub(
        self, sub: MachineSub, enter_sub: bool, call_stack: list[SubId]
    ) -> None:
        ...

    @abstractmethod
    def primitive(
        self,
        mop: Primitive,
        qubits: Sequence[Qubit],
        regs: Sequence[Register],
        call_stack: list[SubId],
    ) -> None:
        ...


@dataclass
class Evaluator(Generic[T]):
    hooks: EvaluatorHooks[T]

    def run(self, sub: MachineSub) -> T:
        self._call_stack: list[SubId] = []
        self._call_sub(sub, sub.qubits, sub.registers)
        return self.hooks.result()

    def _call_sub(
        self, sub: MachineSub, qubits: Sequence[Qubit], regs: Sequence[Register]
    ) -> None:
        sub_id = sub.sub_id
        if sub_id in self._call_stack:
            raise MachineSubRecursionError(
                f"Sub contains itself in the call stack: {sub}"
            )
        self._call_stack.append(sub_id)

        enter_sub = self.hooks.enter_sub(sub, qubits, regs, self._call_stack)
        if enter_sub:
            for mop, qs, rs in sub.instructions:
                if is_primitive(mop):
                    self.hooks.primitive(mop, qs, rs, self._call_stack)
                elif is_subcall(mop):
                    if mop.sub is None:
                        raise ValueError(f"Unlinked SubCall: {mop}")
                    self._call_sub(mop.sub, qs, rs)
                else:
                    raise ValueError(f"Unsupported MachineOp: {mop}")

        self.hooks.exit_sub(sub, enter_sub, self._call_stack)
        self._call_stack.pop()
