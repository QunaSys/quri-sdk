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
