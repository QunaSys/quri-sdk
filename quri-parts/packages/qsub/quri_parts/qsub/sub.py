# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import pi
from typing import TypeAlias

from .op import Op, ParametricMixin, Params
from .qubit import Qubit
from .register import (
    QRegSpec,
    QuantumRegister,
    Register,
    check_register_appear_once,
    get_default_qreg_sequence,
)


@dataclass
class Sub:
    qubits: Sequence[Qubit]
    registers: Sequence[Register]
    aux_qubits: Sequence[Qubit]
    aux_registers: Sequence[Register]
    operations: Sequence[tuple[Op, Sequence[Qubit], Sequence[Register]]]
    phase: float = 0
    qregs: dict[str, QuantumRegister] | None = None
    aux_qregs: dict[str, QuantumRegister] | None = None

    @functools.cached_property
    def qreg_specs(self) -> Sequence[QRegSpec]:
        assert self.qregs is not None and self.aux_qregs is not None
        return tuple(
            QRegSpec(reg_name, qr.size)
            for reg_name, qr in (self.qregs | self.aux_qregs).items()
        )

    @functools.cached_property
    def _sub_qubit_to_local_qubit_str_mapping(self) -> dict[Qubit, str]:
        qubit_iter = iter((*self.qubits, *self.aux_qubits))
        qubit_to_local_qubit_map: dict[Qubit, str] = {}
        for qreg in self.qreg_specs:
            for local_idx in range(qreg.qubit_count):
                qubit_to_local_qubit_map[next(qubit_iter)] = f"{qreg.name}.q{local_idx}"
        return qubit_to_local_qubit_map

    def _get_op_qubit_str(self, op: Op, qubits: Sequence[Qubit]) -> str:
        local_qubit_strs = (
            self._sub_qubit_to_local_qubit_str_mapping[q] for q in qubits
        )
        text_comp = []
        for qreg_spec in op.qregs:
            qbit_str = ", ".join(
                [next(local_qubit_strs) for _ in range(qreg_spec.qubit_count)]
            )
            if qreg_spec.qubit_count > 1:
                qbit_str = f"({qbit_str})"
            text_comp.append(f"{qreg_spec.name}: {qbit_str}")
        return ", ".join(text_comp)

    def _get_cl_reg_str(self, registers: Sequence[Register]) -> str:
        if len(registers) == 0:
            return ""
        elif len(registers) == 1:
            return f"cl: {str(registers[0])}"
        else:
            return f"cl: ({', '.join(map(str, registers))})"

    def _get_operation_str(
        self, op: Op, qubits: Sequence[Qubit], registers: Sequence[Register]
    ) -> str:
        qubit_str = self._get_op_qubit_str(op, qubits)
        cl_str = self._get_cl_reg_str(registers)
        texts = ", ".join([text for text in (qubit_str, cl_str) if len(text) > 0])
        return f"{op.id}({texts})"

    def __str__(self) -> str:
        op_str_list = [
            self._get_operation_str(o, qs, rs) for o, qs, rs in self.operations
        ]
        if len(op_str_list) > 1:
            op_str = "\n " + ",\n ".join(op_str_list) + "\n"
        else:
            op_str = ", ".join(op_str_list)

        reg_strs = []
        if self.qregs is not None:
            reg_strs.extend([f"{v.name}: {v.size}" for v in self.qregs.values()])
        if self.aux_qregs is not None:
            reg_strs.extend([f"{v.name}: {v.size}" for v in self.aux_qregs.values()])
        if len(self.registers) > 0:
            reg_strs.extend([str(r) for r in self.registers])
        if len(self.aux_registers) > 0:
            reg_strs.extend([str(r) for r in self.aux_registers])
        reg_str = ", ".join(reg_strs)
        return f"Sub({reg_str})[{op_str}]"


class SubBuilder:
    def __init__(
        self,
        arg_qubits_count: int,
        arg_reg_count: int = 0,
        arg_qregs: Sequence[QRegSpec] | None = None,
    ) -> None:
        if arg_qubits_count < 0:
            raise ValueError("arg_qubits_count must be greater than or equal to 0.")
        if arg_reg_count < 0:
            raise ValueError("arg_reg_count must be greater than or equal to 0.")
        if arg_qregs is None:
            arg_qregs = get_default_qreg_sequence(arg_qubits_count)
        check_register_appear_once(arg_qregs)
        self._operations: list[tuple[Op, Sequence[Qubit], Sequence[Register]]] = []
        self._qubits = tuple(Qubit(i) for i in range(arg_qubits_count))
        self._aux_id = arg_qubits_count
        self._aux_qubits: list[Qubit] = []
        self._registers = tuple(Register(i) for i in range(arg_reg_count))
        self._aux_reg_id = arg_reg_count
        self._aux_regs: list[Register] = []
        self._phase: float = 0

        self._arg_qregs = arg_qregs
        self._qregs = self._get_qregs()
        self._aux_qregs: dict[str, QuantumRegister] = {}

    @staticmethod
    def from_qregs(
        arg_qregs: Sequence[QRegSpec], arg_reg_count: int = 0
    ) -> "SubBuilder":
        qubit_count = (
            sum([qr.qubit_count for qr in arg_qregs]) if len(arg_qregs) != 0 else 0
        )
        return SubBuilder(qubit_count, arg_reg_count, arg_qregs)

    def _get_qregs(self) -> dict[str, QuantumRegister]:
        if self._arg_qregs is None:
            return None
        idx = 0
        regs: dict[str, QuantumRegister] = {}
        for q_reg_arg in self._arg_qregs:
            qubits = self._qubits[idx : idx + q_reg_arg.qubit_count]  # noqa: E203
            regs[q_reg_arg.name] = QuantumRegister(q_reg_arg.name, qubits)
            idx += q_reg_arg.qubit_count
        if len(self.qubits) != idx:
            raise ValueError(
                f"Qubit count mismatch. Builder request {self.qubits}, "
                f"but got {idx} qubits from arg_qreg."
            )
        return regs

    def _validate_qreg_name(self, new_qreg_name: str) -> None:
        for name in itertools.chain(self._qregs, self._aux_qregs):
            assert name != new_qreg_name, (
                f"New QReg name {new_qreg_name} conflicting with"
                f" existing QReg name {name}."
            )

    def add_op(
        self, op: Op, qubits: Sequence[Qubit], regs: Sequence[Register] = ()
    ) -> None:
        uq = set(qubits) - set(self.qubits) - set(self.aux_qubits)
        if uq:
            raise ValueError(f"undefined qubits: {uq}")
        ur = set(regs) - set(self.registers) - set(self.aux_registers)
        if ur:
            raise ValueError(f"undefined registers: {ur}")
        if len(qubits) != len(set(qubits)):
            raise ValueError(f"duplicated qubits: {qubits}")

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

    @property
    def qregs(self) -> dict[str, QuantumRegister]:
        return self._qregs.copy()

    @property
    def aux_qregs(self) -> dict[str, QuantumRegister]:
        return self._aux_qregs.copy()

    def get_qregs(self, names: Sequence[str]) -> Sequence[QuantumRegister]:
        all_regs = self.qregs | self.aux_qregs
        return tuple(all_regs[name] for name in names)

    def _add_aux_qubit_internal(self) -> Qubit:
        qubit = Qubit(self._aux_id)
        self._aux_id += 1
        self._aux_qubits.append(qubit)
        return qubit

    def add_aux_qubit(self) -> Qubit:
        qubit = self._add_aux_qubit_internal()
        name = f"aux_{qubit.uid}"
        qreg = QuantumRegister(name, (qubit,))
        self._aux_qregs[name] = qreg
        return qubit

    def add_aux_qubits(self, count: int) -> Sequence[Qubit]:
        return tuple(self.add_aux_qubit() for _ in range(count))

    def add_aux_qreg(self, name: str, qubit_count: int) -> QuantumRegister:
        self._validate_qreg_name(name)
        ancs = tuple(self._add_aux_qubit_internal() for _ in range(qubit_count))
        qreg = QuantumRegister(name, ancs)
        self._aux_qregs[name] = qreg
        return qreg

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

    def connect(self, op: Op, **reg_map: Sequence[Qubit] | QuantumRegister) -> None:
        qubits: list[Qubit] = []
        for reg_spec in op.qregs:
            qs = reg_map[reg_spec.name]
            if isinstance(qs, QuantumRegister):
                qs = qs.qubits
            assert len(qs) == reg_spec.qubit_count, (
                "Qubit count mismatch. "
                f"Expected {reg_spec.qubit_count} for register {reg_spec.name}, "
                f"but got {len(qs)}."
            )
            qubits.extend(qs)
        self.add_op(op, tuple(qubits))

    def build(self) -> Sub:
        return Sub(
            tuple(self._qubits),
            tuple(self._registers),
            tuple(self._aux_qubits),
            tuple(self._aux_regs),
            tuple(self._operations),
            self._phase % (2 * pi),
            qregs=self._qregs,
            aux_qregs=self.aux_qregs,
        )


SubFactory: TypeAlias = Callable[Params, Sub]


class SubDef:
    qubit_count: int
    reg_count: int = 0
    qregs: Sequence[QRegSpec] | None = None

    def sub(self, builder: SubBuilder) -> None:
        raise NotImplementedError


def sub(sub_def: type[SubDef]) -> Sub:
    builder = SubBuilder(sub_def.qubit_count, sub_def.reg_count, sub_def.qregs)
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
            d.qubit_count_fn(*params), d.reg_count_fn(*params), d.qregs_fn(*params)
        )
        d.sub(builder, *params)
        return builder.build()

    return s
