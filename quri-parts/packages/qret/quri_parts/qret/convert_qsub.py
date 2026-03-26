import itertools
from collections.abc import Iterable, Mapping, Sequence
from typing import Optional, cast

from pyqret.frontend import Argument, CircuitBuilder, CircuitGenerator, Context, Module
from pyqret.frontend import Qubit as QretQubit
from pyqret.frontend import Qubits as QretQubits
from pyqret.frontend import Register as QretRegister
from pyqret.frontend.gate import intrinsic

from quri_parts.qsub.codegen import CodeGenerator
from quri_parts.qsub.allocate import QubitAllocator, RegisterAllocator
from quri_parts.qsub.compile import compile_sub
from quri_parts.qsub.evaluate import Evaluator, EvaluatorHooks
from quri_parts.qsub.lib import std
from quri_parts.qsub.link import link
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
from quri_parts.qsub.resolve import SubCollector, SubRepository, default_repository
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
) -> None:
    gen = op_circuit_gen_map[mop.op]
    circuit = gen.generate()
    circuit(*qs, *rs)


def _create_circuit_gen(
    op: Op,
    msub: MachineSub,
    op_circuit_gen_map: Mapping[Op, CircuitGenerator],
    builder: CircuitBuilder,
    msubs: Mapping[Op, MachineSub],
    ancilla_map: dict[MachineSub, int],
) -> CircuitGenerator:
    local_ancilla_count = ancilla_map[msub]
    class _Gen(CircuitGenerator):  # type: ignore
        def name(self) -> str:
            return op.id.to_str()

        def arg(self) -> Argument:
            ret = Argument()
            for q in msub.qubits:
                ret.add_operate(f"q{q.uid}")
            for i in range(local_ancilla_count):
                ret.add_clean_ancilla(f"a{i}")
            # TODO distinguish input/output registers
            for r in msub.registers:
                ret.add_input(f"r{r.uid}")
            for r in msub.aux_registers:
                ret.add_input(f"r{r.uid}")
            return ret

        def logic(self, arg: Argument) -> None:
            qubit_map = {
                q: cast(QretQubit, arg[f"a{q.uid}"])
                for q in msub.qubits
            }
            for i, q in enumerate(msub.aux_qubits):
                qubit_map[q] = cast(QretQubit, arg[f"a{i}"])
            register_map = {
                r: cast(QretRegister, arg[f"r{r.uid}"])
                for r in list(msub.registers) + list(msub.aux_registers)
            }
            ancillas = [cast(QretQubit, arg[f"a{i}"]) for i in range(len(msub.aux_registers), local_ancilla_count)]

            for mop, qs, rs in msub.instructions:
                mapped_qs = tuple(qubit_map[q] for q in qs)
                mapped_rs = tuple(register_map[r] for r in rs)
                if is_primitive(mop):
                    _add_intrinsic(mop, mapped_qs, mapped_rs)
                elif is_subcall(mop):
                    ancilla_count = ancilla_map[msubs[mop.op]]
                    _add_funcall(
                        mop,
                        list(mapped_qs) + ancillas[:ancilla_count],
                        mapped_rs,
                        op_circuit_gen_map,
                    )

    return _Gen(builder)


class AncillaCountEvaluatorHooks(EvaluatorHooks[dict[MachineSub, int]]):
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._ancilla_counts: list[int] = []
        self._functions: dict[MachineSub, int] = {}

    def enter_sub(
        self,
        sub: MachineSub,
        qubits: Sequence[Qubit],
        regs: Sequence[Register],
        call_stack: list[SubId],
    ) -> bool:
        self._ancilla_counts.append(0)
        return True

    def exit_sub(
        self, sub: MachineSub, enter_sub: bool, call_stack: list[SubId]
    ) -> None:
        count = self._ancilla_counts.pop() + len(sub.aux_qubits)
        self._functions[sub] = count
        if len(self._ancilla_counts) >= 2:
            self._ancilla_counts[-1] = max(self._ancilla_counts[-1], count)

    def primitive(
        self,
        mop: Primitive,
        qubits: Sequence[Qubit],
        regs: Sequence[Register],
        call_stack: list[SubId],
    ) -> None:
        pass

    def result(self) -> Module:
        return self._functions


def _generate_fn_name(op: Op, msub: MachineSub) -> str:
    ns, ident = op.base_id
    return ident


def create_module_from_qsub_op(
    entry_op: Op,
    module_name: str | None = None,
    repository: SubRepository = default_repository(),
    primitives: Iterable[AbstractOp] = QRETInstrSet,
) -> Module:
    if not set(p.base_id for p in primitives).issubset(QRETInstrBaseIds):
        raise ValueError(
            f"primitives contain instructions incompatible with qret IR: {set(p.base_id for p in primitives) - QRETInstrBaseIds}"
        )

    collector = SubCollector(repository)
    subs = collector.collect_subs(entry_op)
    codegen = CodeGenerator(primitives)
    msubs = {op: codegen.lower(sub) for op, sub in subs.items()}
    entry_msub = msubs[entry_op]
    link(entry_msub, msubs)

    evaluator = Evaluator(hooks = AncillaCountEvaluatorHooks())
    ancilla_map = evaluator.run(entry_msub)

    if module_name is None:
        module_name = f"__qsub__{entry_op.id.to_str()}"

    context = Context()
    module = Module(module_name, context)
    builder = CircuitBuilder(module)

    op_circuit_gen_map: dict[Op, CircuitGenerator] = {}
    for op, msub in msubs.items():
        op_circuit_gen_map[op] = _create_circuit_gen(
            op, msub, op_circuit_gen_map, builder, msubs, ancilla_map
        )
    # Explicitly generate the entry circuit so the module is populated.
    op_circuit_gen_map[entry_op].generate()
    return module
