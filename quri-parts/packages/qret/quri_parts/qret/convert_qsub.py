import itertools
from collections.abc import Iterable, Mapping, Sequence
from typing import Optional, cast

from pyqret.frontend import Argument, CircuitBuilder, CircuitGenerator, Context, Module
from pyqret.frontend import Qubit as QretQubit
from pyqret.frontend import Qubits as QretQubits
from pyqret.frontend import Register as QretRegister
from pyqret.frontend.gate import intrinsic

from quri_parts.qsub.allocate import QubitAllocator, RegisterAllocator
from quri_parts.qsub.compile import compile_sub
from quri_parts.qsub.evaluate import Evaluator, EvaluatorHooks
from quri_parts.qsub.lib import std
from quri_parts.qsub.machineinst import (
    MachineOp,
    MachineSub,
    Primitive,
    SubCall,
    SubId,
    is_primitive,
    is_subcall,
)
from quri_parts.qsub.op import AbstractOp, Op
from quri_parts.qsub.qubit import Qubit
from quri_parts.qsub.register import Register
from quri_parts.qsub.resolve import SubRepository, default_repository, resolve_sub
from quri_parts.qsub.sub import Sub
from quri_parts.qsub.trans import SubTranspiler

QRETInstrSet: Iterable[AbstractOp] = (
    std.M,
    std.Identity,
    std.X,
    std.Y,
    std.Z,
    std.H,
    std.S,
    std.Sdag,
    std.T,
    std.Tdag,
    std.RX,
    std.RY,
    std.RZ,
    std.CNOT,  # CX
    # CY
    std.CZ,
    std.Toffoli,  # CCX
    # CCY
    # CCZ
    std.MCX,
)
QRETInstrBaseIds = set(instr.base_id for instr in QRETInstrSet)

_meas_instr_map = {std.M: intrinsic.measure}
_unary_instr_map = {
    std.Identity: intrinsic.i,
    std.X: intrinsic.x,
    std.Y: intrinsic.y,
    std.Z: intrinsic.z,
    std.H: intrinsic.h,
    std.S: intrinsic.s,
    std.Sdag: intrinsic.sdag,
    std.T: intrinsic.t,
    std.Tdag: intrinsic.tdag,
}
_param_unary_instr_map = {
    std.RX.base_id: intrinsic.rx,
    std.RY.base_id: intrinsic.ry,
    std.RZ.base_id: intrinsic.rz,
}
_binary_instr_map = {
    std.CNOT: intrinsic.cx,
    std.CZ: intrinsic.cz,
}
_ternary_instr_map = {
    std.Toffoli: intrinsic.ccx,
}
_mc_instr_map = {
    std.MCX.base_id: intrinsic.mcx,
}


def generate_qubits(qubit_list: Sequence[QretQubit]) -> QretQubits:
    "Thus pyqret does not support unification, so touch the internal object"
    assert len(qubit_list) > 0
    ret = QretQubits()
    ret._impl = sum([q._impl for q in qubit_list[1:]], qubit_list[0]._impl)
    return ret


def _add_intrinsic(
    mop: MachineOp, qs: Sequence[QretQubit], rs: Sequence[QretRegister]
) -> None:
    op = mop.op
    base_id = op.base_id
    if op in _meas_instr_map:
        _meas_instr_map[op](qs[0], rs[0])
    elif op in _unary_instr_map:
        _unary_instr_map[op](qs[0])
    elif base_id in _param_unary_instr_map:
        param = cast(float, op.id.params[0])
        _param_unary_instr_map[base_id](qs[0], param)
    elif op in _binary_instr_map:
        _binary_instr_map[op](qs[1], qs[0])
    elif op in _ternary_instr_map:
        _ternary_instr_map[op](qs[2], qs[0], qs[1])
    elif base_id in _mc_instr_map:
        qubits = generate_qubits(qs[:-1])
        _mc_instr_map[base_id](qs[-1], qubits)
    else:
        raise ValueError(f"Unsupported op: {op}")


def _add_funcall(
    mop: SubCall,
    qs: Iterable[QretQubit],
    rs: Iterable[QretRegister],
    op_circuit_gen_map: Mapping[Op, CircuitGenerator],
    caller_arg: Argument,
    msubs: Mapping[Op, MachineSub],
    ancilla_counter: "itertools.count[int]",
) -> None:
    gen = op_circuit_gen_map[mop.op]
    circuit = gen.generate()

    callee_msub = msubs[mop.op]
    aux_qs: list[QretQubit] = []
    for _ in callee_msub.aux_qubits:
        name = f"_anc_q{next(ancilla_counter)}"
        caller_arg.add_clean_ancilla(name)
        aux_qs.append(cast(QretQubit, caller_arg[name]))

    aux_rs: list[QretRegister] = []
    for _ in callee_msub.aux_registers:
        name = f"_anc_r{next(ancilla_counter)}"
        caller_arg.add_input(name)
        aux_rs.append(cast(QretRegister, caller_arg[name]))

    circuit(*qs, *aux_qs, *rs, *aux_rs)


def _create_circuit_gen(
    op: Op,
    msub: MachineSub,
    op_circuit_gen_map: Mapping[Op, CircuitGenerator],
    builder: CircuitBuilder,
    msubs: Mapping[Op, MachineSub],
    ancilla_counter: "itertools.count[int]",
) -> CircuitGenerator:
    class _Gen(CircuitGenerator):  # type: ignore
        def name(self) -> str:
            return op.id.to_str()

        def arg(self) -> Argument:
            ret = Argument()
            for q in msub.qubits:
                ret.add_operate(f"q{q.uid}")
            for q in msub.aux_qubits:
                ret.add_clean_ancilla(f"q{q.uid}")
            # TODO distinguish input/output registers
            for r in msub.registers:
                ret.add_input(f"r{r.uid}")
            for r in msub.aux_registers:
                ret.add_input(f"r{r.uid}")
            return ret

        def logic(self, arg: Argument) -> None:
            qubit_map = {
                q: cast(QretQubit, arg[f"q{q.uid}"])
                for q in list(msub.qubits) + list(msub.aux_qubits)
            }
            register_map = {
                r: cast(QretRegister, arg[f"r{r.uid}"])
                for r in list(msub.registers) + list(msub.aux_registers)
            }
            for mop, qs, rs in msub.instructions:
                mapped_qs = tuple(qubit_map[q] for q in qs)
                mapped_rs = tuple(register_map[r] for r in rs)
                if is_primitive(mop):
                    _add_intrinsic(mop, mapped_qs, mapped_rs)
                elif is_subcall(mop):
                    _add_funcall(
                        mop,
                        mapped_qs,
                        mapped_rs,
                        op_circuit_gen_map,
                        arg,
                        msubs,
                        ancilla_counter,
                    )

    return _Gen(builder)


class QRetEvaluatorHooks(EvaluatorHooks[Module]):
    def __init__(self, module_name: str = "__qsub_sub__") -> None:
        self._module_name = module_name
        self.reset()

    def reset(self) -> None:
        self._primitives: list[
            tuple[Primitive, tuple[Qubit, ...], tuple[Register, ...]]
        ] = []
        self._qubit_map_stack: list[dict[Qubit, Qubit]] = []
        self._register_map_stack: list[dict[Register, Register]] = []
        self._qubit_map: Optional[dict[Qubit, Qubit]] = None
        self._register_map: Optional[dict[Register, Register]] = None
        self._qubit_allocator: Optional[QubitAllocator] = None
        self._register_allocator: Optional[RegisterAllocator] = None
        self._cache: dict[
            tuple[SubId, tuple[int, ...], tuple[int, ...], int, int],
            list[tuple[Primitive, tuple[Qubit, ...], tuple[Register, ...]]],
        ] = {}
        self._primitive_stack: list[
            list[tuple[Primitive, tuple[Qubit, ...], tuple[Register, ...]]]
        ] = []
        self._arg_stack: list[
            tuple[SubId, tuple[int, ...], tuple[int, ...], int, int]
        ] = []

    def _update_qubit_map(self) -> None:
        self._qubit_map = {}
        for qubit_map in reversed(self._qubit_map_stack):
            rep = self._qubit_map.copy()
            for k, v in self._qubit_map.items():
                if v in qubit_map:
                    rep[k] = qubit_map[v]
            self._qubit_map = qubit_map | rep

    def _update_register_map(self) -> None:
        self._register_map = {}
        for register_map in reversed(self._register_map_stack):
            rep = self._register_map.copy()
            for k, v in self._register_map.items():
                if v in register_map:
                    rep[k] = register_map[v]
            self._register_map = register_map | rep

    def enter_sub(
        self,
        sub: MachineSub,
        qubits: Sequence[Qubit],
        regs: Sequence[Register],
        call_stack: list[SubId],
    ) -> bool:
        del call_stack
        if self._qubit_allocator is None or self._register_allocator is None:
            self._qubit_allocator = QubitAllocator()
            self._register_allocator = RegisterAllocator()
            self._qubit_map_stack.append(
                dict(self._qubit_allocator.allocate_map(sub.qubits))
            )
            self._register_map_stack.append(
                dict(self._register_allocator.allocate_map(sub.registers))
            )
        else:
            self._qubit_map_stack.append(dict(zip(sub.qubits, qubits)))
            self._register_map_stack.append(dict(zip(sub.registers, regs)))

        self._qubit_map_stack.append(
            dict(self._qubit_allocator.allocate_map(sub.aux_qubits))
        )
        self._register_map_stack.append(
            dict(self._register_allocator.allocate_map(sub.aux_registers))
        )
        self._update_qubit_map()
        self._update_register_map()

        if (
            self._qubit_map is None
            or self._register_map is None
            or self._qubit_allocator is None
            or self._register_allocator is None
        ):
            raise ValueError("Uninitialized allocator or mapping")

        mapped_qs = tuple(self._qubit_map[q].uid for q in qubits)
        mapped_rs = tuple(self._register_map[r].uid for r in regs)
        k = (
            sub.sub_id,
            mapped_qs,
            mapped_rs,
            self._qubit_allocator.total(),
            self._register_allocator.total(),
        )
        if k in self._cache:
            primitives = self._cache[k]
            self._primitives.extend(primitives)
            self._primitive_stack[-1].extend(primitives)
            return False

        self._primitive_stack.append([])
        self._arg_stack.append(k)
        return True

    def exit_sub(
        self, sub: MachineSub, enter_sub: bool, call_stack: list[SubId]
    ) -> None:
        del call_stack
        if self._qubit_allocator is None or self._register_allocator is None:
            raise ValueError("Uninitialized allocator")

        self._qubit_allocator.free_last(len(sub.aux_qubits))
        self._register_allocator.free_last(len(sub.aux_registers))

        self._qubit_map_stack.pop()
        self._qubit_map_stack.pop()
        self._register_map_stack.pop()
        self._register_map_stack.pop()
        self._update_qubit_map()
        self._update_register_map()

        if enter_sub:
            primitives = self._primitive_stack.pop()
            self._cache[self._arg_stack.pop()] = primitives
            if self._primitive_stack:
                self._primitive_stack[-1].extend(primitives)

    def primitive(
        self,
        mop: Primitive,
        qubits: Sequence[Qubit],
        regs: Sequence[Register],
        call_stack: list[SubId],
    ) -> None:
        del call_stack
        if self._qubit_map is None or self._register_map is None:
            raise ValueError("Uninitialized mapping")

        mapped_qs = tuple(self._qubit_map[q] for q in qubits)
        mapped_rs = tuple(self._register_map[r] for r in regs)
        p = (mop, mapped_qs, mapped_rs)
        self._primitives.append(p)
        self._primitive_stack[-1].append(p)

    def result(self) -> Module:
        max_q = (
            max((q.uid for _, qs, _ in self._primitives for q in qs), default=-1) + 1
        )
        max_r = (
            max((r.uid for _, _, rs in self._primitives for r in rs), default=-1) + 1
        )
        qubit_count = max(
            max_q,
            self._qubit_allocator.total() if self._qubit_allocator is not None else 0,
        )
        register_count = max(
            max_r,
            (
                self._register_allocator.total()
                if self._register_allocator is not None
                else 0
            ),
        )

        context = Context()
        module = Module(self._module_name, context)
        builder = CircuitBuilder(module)
        module_name = self._module_name

        primitives = tuple(self._primitives)

        class _Gen(CircuitGenerator):  # type: ignore
            def name(self) -> str:
                return module_name

            def arg(self) -> Argument:
                ret = Argument()
                for i in range(qubit_count):
                    ret.add_operate(f"q{i}")
                for i in range(register_count):
                    ret.add_input(f"r{i}")
                return ret

            def logic(self, arg: Argument) -> None:
                qubit_map = {
                    i: cast(QretQubit, arg[f"q{i}"]) for i in range(qubit_count)
                }
                register_map = {
                    i: cast(QretRegister, arg[f"r{i}"]) for i in range(register_count)
                }
                for mop, qs, rs in primitives:
                    mapped_qs = tuple(qubit_map[q.uid] for q in qs)
                    mapped_rs = tuple(register_map[r.uid] for r in rs)
                    _add_intrinsic(mop, mapped_qs, mapped_rs)

        _Gen(builder).generate()
        return module


def create_module_from_qsub_sub(
    entry_sub: Sub,
    repository: SubRepository = default_repository(),
    primitives: Iterable[AbstractOp] = QRETInstrSet,
    sub_transpilers: Iterable[SubTranspiler] = (),
    module_name: str = "__qsub_sub__",
) -> Module:
    if not set(p.base_id for p in primitives).issubset(QRETInstrBaseIds):
        raise ValueError(
            f"primitives contain instructions incompatible with qret IR: {set(p.base_id for p in primitives) - QRETInstrBaseIds}"
        )

    msub = compile_sub(entry_sub, primitives, repository, sub_transpilers)
    return Evaluator(QRetEvaluatorHooks(module_name)).run(msub)


def create_module_from_qsub_op(
    entry_op: Op,
    repository: SubRepository = default_repository(),
    primitives: Iterable[AbstractOp] = QRETInstrSet,
) -> Module:
    if not set(p.base_id for p in primitives).issubset(QRETInstrBaseIds):
        raise ValueError(
            f"primitives contain instructions incompatible with qret IR: {set(p.base_id for p in primitives) - QRETInstrBaseIds}"
        )

    entry_sub = resolve_sub(entry_op, repository)
    if entry_sub is None:
        raise ValueError(f"Sub is not found in repository for op: {entry_op.id}")

    return create_module_from_qsub_sub(
        entry_sub,
        repository=repository,
        primitives=primitives,
        module_name=entry_op.id.to_str(),
    )
