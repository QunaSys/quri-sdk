from collections.abc import Iterable, Mapping, Sequence
from typing import cast

import pyqsvt.frontend.gate.intrinsic as intrinsic
from pyqsvt.frontend import Argument, CircuitBuilder, CircuitGenerator, Context, Module
from pyqsvt.frontend import Qubit as QsvtQubit
from pyqsvt.frontend import Qubits as QsvtQubits
from pyqsvt.frontend import Register as QsvtRegister

from quri_parts.qsub.codegen import CodeGenerator
from quri_parts.qsub.lib import std
from quri_parts.qsub.machineinst import MachineSub, SubCall, is_primitive, is_subcall
from quri_parts.qsub.op import AbstractOp, Op
from quri_parts.qsub.resolve import SubCollector, SubRepository, default_repository
from quri_parts.qsub.sub import Sub, SubBuilder

QSVTInstrSet = (
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
QSVTInstrBaseIds = set(instr.base_id for instr in QSVTInstrSet)

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


# Transpilation
def _swap_sub() -> Sub:
    b = SubBuilder(2)
    q0, q1 = b.qubits
    b.add_op(std.CNOT, (q0, q1))
    b.add_op(std.CNOT, (q1, q0))
    b.add_op(std.CNOT, (q0, q1))
    return b.build()


default_repository().register_sub(std.SWAP, _swap_sub())


def _phase_sub(phase: float) -> Sub:
    b = SubBuilder(1)
    b.add_op(std.RZ(phase), b.qubits)
    b.add_phase(phase / 2)
    return b.build()


default_repository().register_sub(std.Phase, _phase_sub)

# End transpilation


def _add_intrinsic(mop: SubCall, qs: Sequence[QsvtQubit], rs: Sequence[QsvtRegister]):
    op = mop.op
    base_id = op.base_id
    if op in _meas_instr_map:
        _meas_instr_map[op](qs[0], rs[0])
    elif op in _unary_instr_map:
        _unary_instr_map[op](qs[0])
    elif base_id in _param_unary_instr_map:
        _param_unary_instr_map[base_id](qs[0], *op.id.params)
    elif op in _binary_instr_map:
        _binary_instr_map[op](qs[1], qs[0])
    elif op in _ternary_instr_map:
        _ternary_instr_map[op](qs[2], qs[0], qs[1])
    if mop.op.base_id == std.MCX.base_id:
        intrinsic.mcx(qs[-1], sum(qs[:-1], QsvtQubits()))
    else:
        raise ValueError(f"Unsupported op: {op}")


def _add_funcall(
    mop: SubCall,
    qs: Iterable[QsvtQubit],
    rs: Iterable[QsvtRegister],
    op_circuit_gen_map: Mapping[Op, CircuitGenerator],
):
    gen = op_circuit_gen_map[mop.op]
    circuit = gen.generate()
    circuit(*qs, *rs)


def _create_circuit_gen(
    op: Op,
    msub: MachineSub,
    op_circuit_gen_map: Mapping[Op, CircuitGenerator],
    builder: CircuitBuilder,
) -> CircuitGenerator:
    class _Gen(CircuitGenerator):
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

        def logic(self, arg: Argument):
            qubit_map = {
                q: cast(QsvtQubit, arg[f"q{q.uid}"])
                for q in msub.qubits + msub.aux_qubits
            }
            register_map = {
                r: cast(QsvtRegister, arg[f"r{r.uid}"])
                for r in msub.registers + msub.aux_registers
            }
            for mop, qs, rs in msub.instructions:
                mapped_qs = tuple(qubit_map[q] for q in qs)
                mapped_rs = tuple(register_map[r] for r in rs)
                if is_primitive(mop):
                    _add_intrinsic(mop, mapped_qs, mapped_rs)
                elif is_subcall(mop):
                    _add_funcall(mop, mapped_qs, mapped_rs, op_circuit_gen_map)

    return _Gen(builder)


def create_module_from_qsub_op(
    entry_op: Op,
    repository: SubRepository = default_repository(),
    primitives: Iterable[AbstractOp] = QSVTInstrSet,
) -> Module:
    if not set(p.base_id for p in primitives).issubset(QSVTInstrBaseIds):
        raise ValueError(
            f"primitives contain instructions incompatible with qsvt IR: {set(p.base_id for p in primitives) - QSVTInstrBaseIds}"
        )

    if repository is None:
        repository = default_repository()

    collector = SubCollector(repository)
    subs = collector.collect_subs(entry_op)
    codegen = CodeGenerator(primitives)
    msubs = {op: codegen.lower(sub) for op, sub in subs.items()}

    context = Context()
    module = Module(f"__qsub__{entry_op.id.to_str()}", context)
    builder = CircuitBuilder(module)

    op_circuit_gen_map: dict[Op, CircuitGenerator] = {}
    for op, msub in msubs.items():
        op_circuit_gen_map[op] = _create_circuit_gen(
            op, msub, op_circuit_gen_map, builder
        )

    entry_gen = op_circuit_gen_map[entry_op]
    entry_circuit = entry_gen.generate()
    return module
