# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Mapping
from typing import Optional, TypeAlias

from .allocate import QubitAllocator, RegisterAllocator
from .machineinst import MachineInst, MachineSub, MachineSubRecursionError, is_subcall
from .op import Op
from .qubit import Qubit
from .register import Register

CallTable: TypeAlias = Mapping[Op, MachineSub]


def map_qubits(sub: MachineSub, qubit_map: Mapping[Qubit, Qubit]) -> MachineSub:
    instrs: list[MachineInst] = []
    for inst, qubits, regs in sub.instructions:
        instrs.append((inst, tuple(qubit_map.get(q, q) for q in qubits), regs))
    return MachineSub(
        tuple(qubit_map.get(q, q) for q in sub.qubits),
        sub.registers,
        tuple(qubit_map.get(q, q) for q in sub.aux_qubits),
        sub.aux_registers,
        instrs,
    )


def map_registers(
    sub: MachineSub, register_map: Mapping[Register, Register]
) -> MachineSub:
    instrs: list[MachineInst] = []
    for inst, qubits, regs in sub.instructions:
        instrs.append((inst, qubits, tuple(register_map.get(r, r) for r in regs)))
    return MachineSub(
        sub.qubits,
        tuple(register_map.get(r, r) for r in sub.registers),
        sub.aux_qubits,
        tuple(register_map.get(r, r) for r in sub.aux_registers),
        instrs,
    )


def _expand(
    sub: MachineSub,
    recursive: bool = False,
    allocator: Optional[QubitAllocator] = None,
    allocator_regs: Optional[RegisterAllocator] = None,
    call_stack: tuple[int, ...] = (),
) -> MachineSub:
    if allocator is None or allocator_regs is None:
        allocator = QubitAllocator()
        entry_map = allocator.allocate_map(tuple(sub.qubits) + tuple(sub.aux_qubits))
        sub = map_qubits(sub, entry_map)

        allocator_regs = RegisterAllocator()
        entry_map_regs = allocator_regs.allocate_map(
            tuple(sub.registers) + tuple(sub.aux_registers)
        )
        sub = map_registers(sub, entry_map_regs)

    extend_aux_qubits = set(sub.aux_qubits)
    extend_aux_regs = set(sub.aux_registers)

    instrs: list[MachineInst] = []
    for mop, qubits, regs in sub.instructions:
        if is_subcall(mop):
            if mop.sub is None:
                raise ValueError(f"Unlinked SubCall: {mop}")

            sub_id = mop.sub.sub_id
            if sub_id in call_stack:
                raise MachineSubRecursionError(
                    f"Sub contains itself in the call stack: {mop.sub}"
                )

            arg_qubit_map = dict(zip(mop.sub.qubits, qubits))
            aux_qubit_map = dict(allocator.allocate_map(mop.sub.aux_qubits))

            arg_regs_map = dict(zip(mop.sub.registers, regs))
            aux_regs_map = dict(allocator_regs.allocate_map(mop.sub.aux_registers))

            remap_sub = map_qubits(mop.sub, arg_qubit_map | aux_qubit_map)
            remap_sub = map_registers(remap_sub, arg_regs_map | aux_regs_map)
            if recursive:
                remap_sub = _expand(
                    remap_sub,
                    recursive,
                    allocator,
                    allocator_regs,
                    call_stack + (sub_id,),
                )
            extend_aux_qubits |= set(remap_sub.aux_qubits)
            extend_aux_regs |= set(remap_sub.aux_registers)

            allocator.free_last(len(mop.sub.aux_qubits))
            allocator_regs.free_last(len(mop.sub.aux_registers))
            instrs.extend(remap_sub.instructions)
        else:
            instrs.append((mop, qubits, regs))

    return MachineSub(
        sub.qubits,
        sub.registers,
        tuple(extend_aux_qubits),
        tuple(extend_aux_regs),
        tuple(instrs),
    )


def expand(sub: MachineSub, recursive: bool = False) -> MachineSub:
    return _expand(sub, recursive)


def full_expand(sub: MachineSub) -> MachineSub:
    return expand(sub, True)
