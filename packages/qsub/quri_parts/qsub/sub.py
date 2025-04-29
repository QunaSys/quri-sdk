# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import pi
from typing import TypeAlias

from .op import Op, ParametricMixin, Params
from .qubit import Qubit
from .register import Register


def _op_app_str(op: Op, qubits: Sequence[Qubit], regs: Sequence[Register]) -> str:
    args = [str(x) for x in (*qubits, *regs)]
    return f"{op.id}({', '.join(args)})"


@dataclass
class Sub:
    qubits: Sequence[Qubit]
    registers: Sequence[Register]
    aux_qubits: Sequence[Qubit]
    aux_registers: Sequence[Register]
    operations: Sequence[tuple[Op, Sequence[Qubit], Sequence[Register]]]
    phase: float = 0

    def __str__(self) -> str:
        op_str_list = [
            _op_app_str(op, qubits, regs) for op, qubits, regs in self.operations
        ]
        if len(op_str_list) > 1:
            op_str = "\n " + ",\n ".join(op_str_list) + "\n"
        else:
            op_str = ", ".join(op_str_list)
        return f"Sub[{op_str}]"


class SubBuilder:
    def __init__(self, arg_qubits_count: int, arg_reg_count: int = 0) -> None:
        if arg_qubits_count < 0:
            raise ValueError("arg_qubits_count must be greater than or equal to 0.")
        if arg_reg_count < 0:
            raise ValueError("arg_reg_count must be greater than or equal to 0.")
        self._operations: list[tuple[Op, Sequence[Qubit], Sequence[Register]]] = []
        self._qubits = tuple(Qubit(i) for i in range(arg_qubits_count))
        self._aux_id = arg_qubits_count
        self._aux_qubits: list[Qubit] = []
        self._registers = tuple(Register(i) for i in range(arg_reg_count))
        self._aux_reg_id = arg_reg_count
        self._aux_regs: list[Register] = []
        self._phase: float = 0

    def add_op(
        self, op: Op, qubits: Sequence[Qubit], regs: Sequence[Register] = ()
    ) -> None:
        uq = set(qubits) - set(self.qubits) - set(self.aux_qubits)
        if uq:
            raise ValueError(f"undefined qubits: {uq}")
        ur = set(regs) - set(self.registers) - set(self.aux_registers)
        if ur:
            raise ValueError(f"undefined registers: {ur}")

        self._operations.append((op, tuple(qubits), tuple(regs)))

    @property
    def qubits(self) -> Sequence[Qubit]:
        return tuple(self._qubits)

    @property
    def aux_qubits(self) -> Sequence[Qubit]:
        return tuple(self._aux_qubits)

    @property
    def registers(self) -> Sequence[Register]:
        return tuple(self._registers)

    @property
    def aux_registers(self) -> Sequence[Register]:
        return tuple(self._aux_regs)

    def add_aux_qubit(self) -> Qubit:
        qubit = Qubit(self._aux_id)
        self._aux_id += 1
        self._aux_qubits.append(qubit)
        return qubit

    def add_aux_qubits(self, count: int) -> Sequence[Qubit]:
        return tuple(self.add_aux_qubit() for _ in range(count))

    def add_aux_register(self) -> Register:
        reg = Register(self._aux_reg_id)
        self._aux_reg_id += 1
        self._aux_regs.append(reg)
        return reg

    def add_aux_registers(self, count: int) -> Sequence[Register]:
        return tuple(self.add_aux_register() for _ in range(count))

    def add_phase(self, phase: float) -> float:
        self._phase += phase
        return self._phase

    def build(self) -> Sub:
        return Sub(
            tuple(self._qubits),
            tuple(self._registers),
            tuple(self._aux_qubits),
            tuple(self._aux_regs),
            tuple(self._operations),
            self._phase % (2 * pi),
        )


SubFactory: TypeAlias = Callable[Params, Sub]


class SubDef:
    qubit_count: int
    reg_count: int = 0

    def sub(self, builder: SubBuilder) -> None:
        raise NotImplementedError


def sub(sub_def: type[SubDef]) -> Sub:
    builder = SubBuilder(sub_def.qubit_count, sub_def.reg_count)
    sub_def().sub(builder)
    return builder.build()


class ParamSubDef(ParametricMixin[Params]):
    def sub(
        self, builder: SubBuilder, *params: Params.args, **_: Params.kwargs
    ) -> None:
        raise NotImplementedError


def param_sub(sub_def: type[ParamSubDef[Params]]) -> SubFactory[Params]:
    d = sub_def()

    def s(*params: Params.args, **_: Params.kwargs) -> Sub:
        builder = SubBuilder(
            d.qubit_count_fn(*params),
            d.reg_count_fn(*params),
        )
        d.sub(builder, *params)
        return builder.build()

    return s
