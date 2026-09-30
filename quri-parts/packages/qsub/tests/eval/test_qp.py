import pytest

from quri_parts.circuit import QuantumCircuit, gates
from quri_parts.qsub.compile import compile, compile_sub
from quri_parts.qsub.eval.quriparts import (
    QURIPartsEvaluatorHooks,
    quri_parts_codegen,
    quri_parts_linker,
)
from quri_parts.qsub.evaluate import Evaluator
from quri_parts.qsub.expand import full_expand
from quri_parts.qsub.lib.std import CNOT, CZ, RX, RY, RZ, H, S, T, Toffoli, X
from quri_parts.qsub.opsub import OpSubDef, opsub
from quri_parts.qsub.sub import SubBuilder


def test_execute_to_qp_circuit() -> None:
    builder = SubBuilder(3)
    q0, q1, q2 = builder.qubits
    a0, a1 = builder.add_aux_qubit(), builder.add_aux_qubit()
    builder.add_op(H, (a0,), ())
    builder.add_op(S, (q0,), ())
    builder.add_op(T, (q1,), ())
    builder.add_op(RZ(0.0), (a0,), ())
    builder.add_op(CZ, (q2, a1), ())
    builder.add_op(Toffoli, (a0, a1, q2), ())

    sub = builder.build()
    msub = quri_parts_codegen.lower(sub)
    msub = quri_parts_linker.link(msub)
    hook = QURIPartsEvaluatorHooks()
    Evaluator(hook).run(msub)
    target = hook.result()

    expect = QuantumCircuit(5)
    expect.add_H_gate(3)
    expect.add_S_gate(0)
    expect.add_T_gate(1)
    expect.add_RZ_gate(3, 0.0)
    expect.add_CZ_gate(2, 4)
    expect.add_TOFFOLI_gate(3, 4, 2)

    assert target == expect


def test_compare_to_expand() -> None:
    class FDef(OpSubDef):
        name = "F"
        qubit_count = 2

        def sub(self, builder: SubBuilder) -> None:
            a, b = builder.qubits
            builder.add_op(H, (b,))
            builder.add_op(CNOT, (b, a))
            builder.add_op(H, (a,))

    F, FSub = opsub(FDef)

    class GDef(OpSubDef):
        name = "G"
        qubit_count = 3

        def sub(self, builder: SubBuilder) -> None:
            a, b, c = builder.qubits
            builder.add_op(T, (c,))
            builder.add_op(F, (c, b))
            builder.add_op(F, (a, c))
            builder.add_op(T, (a,))

    G, GSub = opsub(GDef)

    class MainDef(OpSubDef):
        name = "Main"
        qubit_count = 5

        def sub(self, builder: SubBuilder) -> None:
            (a, b, c, d, e) = builder.qubits
            builder.add_op(S, (e,))
            builder.add_op(G, (e, c, a))
            builder.add_op(G, (d, b, c))
            builder.add_op(S, (a,))

    Main, MainSub = opsub(MainDef)

    primitives = (H, S, T, CNOT)
    compiled_msub = compile(Main, primitives)
    expanded_msub = full_expand(compiled_msub)

    expanded_evaluated_circuit = Evaluator(QURIPartsEvaluatorHooks()).run(expanded_msub)
    evaluated_circuit = Evaluator(QURIPartsEvaluatorHooks()).run(compiled_msub)

    assert expanded_evaluated_circuit == evaluated_circuit


def test_sub_cache_respects_qubit_argument_order() -> None:
    """The gate cache must not reuse results across calls to the same sub with
    different qubit argument orders.

    Regression: the cache key was derived from the call-site qubit sequence rather
    than the sub's own qubit-to-physical mapping, so F(q0, q1) and F(q1, q0)
    produced the same key and the second call incorrectly reused the first call's gates.
    """

    class FDef(OpSubDef):
        name = "F"
        qubit_count = 2

        def sub(self, builder: SubBuilder) -> None:
            q = builder.qubits[0]
            builder.add_op(X, (q,))

    F, _ = opsub(FDef)

    builder = SubBuilder(2)
    q0, q1 = builder.qubits
    builder.add_op(F, (q0, q1))  # X(0)
    builder.add_op(F, (q1, q0))  # X(1)

    compiled = compile_sub(builder.build(), (X,))
    circ = Evaluator(QURIPartsEvaluatorHooks()).run(compiled)

    expected = [gates.X(0), gates.X(1)]
    for i, g in enumerate(expected):
        assert circ.gates[i] == g, f"{i}-th gate doesn't match"


def test_reuse_evaluator() -> None:
    b = SubBuilder(1)
    b.add_op(X, b.qubits)

    sub = b.build()
    primitives = (X,)
    compiled = compile_sub(sub, primitives)

    qp_generator = Evaluator(QURIPartsEvaluatorHooks())
    qp_circuit_0 = qp_generator.run(compiled)
    qp_circuit_1 = qp_generator.run(compiled)

    assert qp_circuit_0.gates == qp_circuit_1.gates


def test_compile_sub_applies_default_gate_set_conversion() -> None:
    builder = SubBuilder(1)
    q = builder.qubits[0]
    builder.add_op(RX(0.1), (q,))
    builder.add_op(RY(0.2), (q,))
    builder.add_op(RZ(0.3), (q,))
    sub = builder.build()
    primitives = (H, S, RZ, CNOT)

    converted = Evaluator(QURIPartsEvaluatorHooks()).run(compile_sub(sub, primitives))
    assert {gate.name for gate in converted.gates} <= {"H", "S", "RZ", "CNOT"}

    with pytest.raises(ValueError, match="not found in calltable"):
        compile_sub(sub, primitives, sub_transpilers=())
