from cmath import pi

from quri_parts.circuit import QuantumCircuit
from quri_parts.circuit.transpile import CNOT2CZHTranspiler, RZ2NamedTranspiler
from quri_parts.qsub.compile import compile, compile_sub
from quri_parts.qsub.eval.quriparts import QURIPartsEvaluatorHooks
from quri_parts.qsub.evaluate import Evaluator
from quri_parts.qsub.lib.std import CNOT, CZ, RZ, H, S, T, X, Z
from quri_parts.qsub.machineinst import MachineSub
from quri_parts.qsub.opsub import OpSubDef, opsub
from quri_parts.qsub.sub import SubBuilder


def _eval_to_qp(msub: MachineSub) -> QuantumCircuit:
    return Evaluator(QURIPartsEvaluatorHooks()).run(msub)


class _SimpleDef(OpSubDef):
    name = "Simple"
    qubit_count = 2

    def sub(self, builder: SubBuilder) -> None:
        q0, q1 = builder.qubits
        builder.add_op(H, (q0,))
        builder.add_op(CNOT, (q0, q1))
        builder.add_op(H, (q1,))


Simple, SimpleSub = opsub(_SimpleDef)


class _WithRZDef(OpSubDef):
    name = "WithRZ"
    qubit_count = 1

    def sub(self, builder: SubBuilder) -> None:
        (q0,) = builder.qubits
        builder.add_op(RZ(pi / 4.0), (q0,))
        builder.add_op(H, (q0,))
        builder.add_op(RZ(pi / 2.0), (q0,))


WithRZ, WithRZSub = opsub(_WithRZDef)


class _NestedOuterDef(OpSubDef):
    name = "NestedOuter"
    qubit_count = 2

    def sub(self, builder: SubBuilder) -> None:
        q0, q1 = builder.qubits
        builder.add_op(WithRZ, (q0,))
        builder.add_op(CNOT, (q0, q1))
        builder.add_op(WithRZ, (q1,))


NestedOuter, NestedOuterSub = opsub(_NestedOuterDef)


# --- Tests for compile() with CircuitTranspiler ---


def test_compile_with_rz2named_transpiler() -> None:
    """compile() with RZ2NamedTranspiler converts RZ gates to named gates."""
    primitives = (H, S, T, CNOT)
    compiled = compile(WithRZ, primitives, sub_transpilers=[RZ2NamedTranspiler()])
    circ = _eval_to_qp(compiled)

    expect = QuantumCircuit(1)
    expect.add_T_gate(0)  # RZ(pi/4) -> T
    expect.add_H_gate(0)
    expect.add_S_gate(0)  # RZ(pi/2) -> S
    assert circ == expect


def test_compile_without_transpiler_keeps_rz() -> None:
    """compile() without transpiler keeps RZ gates as-is."""
    primitives = (H, RZ, CNOT)
    compiled = compile(WithRZ, primitives)
    circ = _eval_to_qp(compiled)

    expect = QuantumCircuit(1)
    expect.add_RZ_gate(0, pi / 4.0)
    expect.add_H_gate(0)
    expect.add_RZ_gate(0, pi / 2.0)
    assert circ == expect


def test_compile_with_cnot2czh_transpiler() -> None:
    """compile() with CNOT2CZHTranspiler decomposes CNOT to CZ+H."""
    primitives = (H, CZ)
    compiled = compile(Simple, primitives, sub_transpilers=[CNOT2CZHTranspiler()])
    circ = _eval_to_qp(compiled)

    expect = QuantumCircuit(2)
    expect.add_H_gate(0)
    # CNOT(0,1) -> H(1), CZ(0,1), H(1)
    expect.add_H_gate(1)
    expect.add_CZ_gate(0, 1)
    expect.add_H_gate(1)
    expect.add_H_gate(1)
    assert circ == expect


def test_compile_with_multiple_circuit_transpilers() -> None:
    """compile() with multiple CircuitTranspilers applies them in sequence."""

    class _MultiDef(OpSubDef):
        name = "Multi"
        qubit_count = 1

        def sub(self, builder: SubBuilder) -> None:
            (q0,) = builder.qubits
            builder.add_op(RZ(-pi), (q0,))
            builder.add_op(H, (q0,))
            builder.add_op(RZ(pi / 4.0), (q0,))

    Multi, MultiSub = opsub(_MultiDef)

    primitives = (H, S, T, Z)
    compiled = compile(Multi, primitives, sub_transpilers=[RZ2NamedTranspiler()])
    circ = _eval_to_qp(compiled)

    expect = QuantumCircuit(1)
    expect.add_Z_gate(0)  # RZ(-pi) -> Z
    expect.add_H_gate(0)
    expect.add_T_gate(0)  # RZ(pi/4) -> T
    assert circ == expect


def test_compile_nested_subs_with_transpiler() -> None:
    """compile() with CircuitTranspiler works on nested sub definitions."""
    primitives = (H, S, T, CNOT)
    compiled = compile(NestedOuter, primitives, sub_transpilers=[RZ2NamedTranspiler()])
    circ = _eval_to_qp(compiled)

    expect = QuantumCircuit(2)
    # WithRZ(q0): RZ(pi/4)->T, H, RZ(pi/2)->S
    expect.add_T_gate(0)
    expect.add_H_gate(0)
    expect.add_S_gate(0)
    # CNOT(q0, q1)
    expect.add_CNOT_gate(0, 1)
    # WithRZ(q1): RZ(pi/4)->T, H, RZ(pi/2)->S
    expect.add_T_gate(1)
    expect.add_H_gate(1)
    expect.add_S_gate(1)
    assert circ == expect


# --- Tests for compile_sub() with CircuitTranspiler ---


def test_compile_sub_with_rz2named_transpiler() -> None:
    """compile_sub() with RZ2NamedTranspiler converts RZ gates."""
    b = SubBuilder(1)
    (q0,) = b.qubits
    b.add_op(RZ(pi / 2.0), (q0,))
    b.add_op(H, (q0,))
    b.add_op(RZ(pi / 4.0), (q0,))
    entry_sub = b.build()

    primitives = (H, S, T)
    compiled = compile_sub(
        entry_sub, primitives, sub_transpilers=[RZ2NamedTranspiler()]
    )
    circ = _eval_to_qp(compiled)

    expect = QuantumCircuit(1)
    expect.add_S_gate(0)  # RZ(pi/2) -> S
    expect.add_H_gate(0)
    expect.add_T_gate(0)  # RZ(pi/4) -> T
    assert circ == expect


def test_compile_sub_with_cnot2czh_transpiler() -> None:
    """compile_sub() with CNOT2CZHTranspiler decomposes CNOT."""
    b = SubBuilder(2)
    q0, q1 = b.qubits
    b.add_op(X, (q0,))
    b.add_op(CNOT, (q0, q1))
    entry_sub = b.build()

    primitives = (X, H, CZ)
    compiled = compile_sub(
        entry_sub, primitives, sub_transpilers=[CNOT2CZHTranspiler()]
    )
    circ = _eval_to_qp(compiled)

    expect = QuantumCircuit(2)
    expect.add_X_gate(0)
    # CNOT(0,1) -> H(1), CZ(0,1), H(1)
    expect.add_H_gate(1)
    expect.add_CZ_gate(0, 1)
    expect.add_H_gate(1)
    assert circ == expect


def test_compile_sub_without_transpiler() -> None:
    """compile_sub() without transpiler preserves original gates."""
    b = SubBuilder(1)
    (q0,) = b.qubits
    b.add_op(RZ(pi / 4.0), (q0,))
    b.add_op(H, (q0,))
    entry_sub = b.build()

    primitives = (H, RZ)
    compiled = compile_sub(entry_sub, primitives)
    circ = _eval_to_qp(compiled)

    expect = QuantumCircuit(1)
    expect.add_RZ_gate(0, pi / 4.0)
    expect.add_H_gate(0)
    assert circ == expect


def test_compile_sub_with_subcall_and_transpiler() -> None:
    """compile_sub() with CircuitTranspiler and sub containing subcalls."""
    b = SubBuilder(2)
    q0, q1 = b.qubits
    b.add_op(WithRZ, (q0,))
    b.add_op(CNOT, (q0, q1))
    entry_sub = b.build()

    primitives = (H, S, T, CNOT)
    compiled = compile_sub(
        entry_sub, primitives, sub_transpilers=[RZ2NamedTranspiler()]
    )
    circ = _eval_to_qp(compiled)

    expect = QuantumCircuit(2)
    # WithRZ(q0): T, H, S
    expect.add_T_gate(0)
    expect.add_H_gate(0)
    expect.add_S_gate(0)
    # CNOT(q0, q1)
    expect.add_CNOT_gate(0, 1)
    assert circ == expect


def test_compile_and_compile_sub_agree_with_transpiler() -> None:
    """compile() and compile_sub() produce equivalent results."""
    transpilers = [RZ2NamedTranspiler()]
    primitives = (H, S, T, CNOT)

    compiled_from_op = compile(WithRZ, primitives, sub_transpilers=transpilers)
    circ_op = _eval_to_qp(compiled_from_op)

    compiled_from_sub = compile_sub(WithRZSub, primitives, sub_transpilers=transpilers)
    circ_sub = _eval_to_qp(compiled_from_sub)

    assert circ_op == circ_sub
