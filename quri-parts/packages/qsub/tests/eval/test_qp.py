from quri_parts.circuit import QuantumCircuit
from quri_parts.qsub.compile import compile, compile_sub
from quri_parts.qsub.eval.quriparts import (
    QURIPartsEvaluatorHooks,
    quri_parts_codegen,
    quri_parts_linker,
)
from quri_parts.qsub.evaluate import Evaluator
from quri_parts.qsub.expand import full_expand
from quri_parts.qsub.lib.std import CNOT, CZ, RZ, H, S, T, Toffoli, X
from quri_parts.qsub.opsub import OpSubDef, opsub
from quri_parts.qsub.qubit import Qubit
from quri_parts.qsub.sub import Sub, SubBuilder


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


def test_duplicate_qubit_error() -> None:
    """
    Each sub has the following internal qubits and aux_qubits:
    sub: qubits=(Qubit(0), Qubit(2))
         <- Intentional out-of-range index (2 >= len(qubits)).
            This doesn't happen as long as a Sub is created via SubBuilder.
    ASub: qubits=(Qubit(0), Qubit(1)), aux_qubits=(Qubit(2))

    QURIPartsEvaluator first allocates 2 qubits (Qubit(0), Qubit(1)) for the qp circuit and maps sub.qubits to them as:
    1) sub.Qubit(0) -> qp.Qubit(0), sub.Qubit(2) -> qp.Qubit(1)

    When entering ASub, it allocates 1 qubits (Qubit(2)) for the qp circuit and maps ASub.aux_qubits to it as:
    2) ASub.Qubit(2) -> qp.Qubit(2)

    ASub is called from sub as A(sub.Qubit(0), sub.Qubit(2)), so it corresponds to a mapping:
    3) ASub.Qubit(0) -> sub.Qubit(0), ASub.Qubit(1) -> sub.Qubit(2)

    Combining 1)-3) mappings, the ASub qubits should be mapped to qp qubits as:
    ASub.Qubit(0) -> sub.Qubit(0) -> qp.Qubit(0)
    ASub.Qubit(1) -> sub.Qubit(2) -> qp.Qubit(1)
    ASub.Qubit(2) -> qp.Qubit(2)

    However, the mapping logic at 645359febc1ebce9f176dba4037a5fa2615772b9 doesn't distinguish sub.Qubit(2) and qp.Qubit(2), so it maps ASub.Qubit(2) incorrectly as:
    ASub.Qubit(2) -> qp.Qubit(2) = sub.Qubit(2) -> qp.Qubit(1)
    """

    class ADef(OpSubDef):
        name = "A"
        qubit_count = 2

        def sub(self, builder: SubBuilder) -> None:
            builder.add_op(Toffoli, (builder.add_aux_qubit(), *builder.qubits))

    A, ASub = opsub(ADef)

    qs = (Qubit(0), Qubit(2))
    sub = Sub(qs, (), (), (), [(A, qs, ())])

    primitives = (Toffoli,)
    compiled = compile_sub(sub, primitives)

    qp_generator = Evaluator(QURIPartsEvaluatorHooks())
    qp_circuit = qp_generator.run(compiled)

    gates = qp_circuit.gates
    assert len(gates) == 1
    g = gates[0]
    indices = g.target_indices + g.control_indices
    assert len(set(indices)) == len(
        indices
    ), f"Duplicate qubits. {g.target_indices=}, {g.control_indices=}"
