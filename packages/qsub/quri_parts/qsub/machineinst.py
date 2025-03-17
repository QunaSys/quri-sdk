from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, TypeAlias

from typing_extensions import TypeGuard

from .op import Op
from .qubit import Qubit
from .register import Register

SubId: TypeAlias = int


@dataclass
class MachineOp:
    op: Op


@dataclass
class Primitive(MachineOp):
    ...


@dataclass
class SubCall(MachineOp):
    sub: Optional["MachineSub"] = None


def is_primitive(op: MachineOp) -> TypeGuard[Primitive]:
    return isinstance(op, Primitive)


def is_subcall(op: MachineOp) -> TypeGuard[SubCall]:
    return isinstance(op, SubCall)


MachineInst: TypeAlias = tuple[MachineOp, Sequence[Qubit], Sequence[Register]]


@dataclass
class MachineSub:
    qubits: Sequence[Qubit]
    registers: Sequence[Register]
    aux_qubits: Sequence[Qubit]
    aux_registers: Sequence[Register]
    instructions: Sequence[MachineInst]

    @property
    def sub_id(self) -> SubId:
        return id(self)


class MachineSubRecursionError(Exception):
    ...
