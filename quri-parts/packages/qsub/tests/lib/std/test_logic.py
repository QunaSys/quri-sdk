from collections.abc import Sequence

from quri_parts.qsub.lib.std import (
    CNOT,
    CZ,
    Cbz,
    H,
    Label,
    M,
    Sdag,
    T,
    Tdag,
    Toffoli,
    scoped_and,
    scoped_and_clifford_t,
)
from quri_parts.qsub.op import Op
from quri_parts.qsub.qubit import Qubit
from quri_parts.qsub.register import Register
from quri_parts.qsub.sub import SubBuilder


def test_scoped_and() -> None:
    builder = SubBuilder(arg_qubits_count=3)
    q0, q1, q2 = builder.qubits
    with scoped_and(builder, q0, q1) as a:
        builder.add_op(CNOT, (a, q2))
    sub = builder.build()

    assert len(sub.aux_qubits) == 1
    a = sub.aux_qubits[0]
    assert sub.operations == (
        (Toffoli, (q0, q1, a), ()),
        (CNOT, (a, q2), ()),
        (Toffoli, (q0, q1, a), ()),
    )


def _clifford_t_and(
    q0: Qubit, q1: Qubit, a: Qubit
) -> Sequence[tuple[Op, Sequence[Qubit], Sequence[Register]]]:
    return (
        (H, (a,), ()),
        (T, (a,), ()),
        (CNOT, (q1, a), ()),
        (Tdag, (a,), ()),
        (CNOT, (q0, a), ()),
        (T, (a,), ()),
        (CNOT, (q1, a), ()),
        (Tdag, (a,), ()),
        (H, (a,), ()),
        (Sdag, (a,), ()),
    )


def _clifford_t_uncompute_and(
    q0: Qubit, q1: Qubit, a: Qubit, r: Register, l0: Register
) -> Sequence[tuple[Op, Sequence[Qubit], Sequence[Register]]]:
    return (
        (H, (a,), ()),
        (M, (a,), (r,)),
        (Cbz, (), (r, l0)),
        (CZ, (q0, q1), ()),
        (Label, (), (l0,)),
    )


def test_scoped_and_clifford_t() -> None:
    builder = SubBuilder(arg_qubits_count=3)
    q0, q1, q2 = builder.qubits
    with scoped_and_clifford_t(builder, q0, q1) as a:
        builder.add_op(CNOT, (a, q2))
    sub = builder.build()

    assert len(sub.aux_qubits) == 1
    a = sub.aux_qubits[0]
    assert len(sub.aux_registers) == 2
    r, l0 = sub.aux_registers
    assert sub.operations == (
        *_clifford_t_and(q0, q1, a),
        (CNOT, (a, q2), ()),
        *_clifford_t_uncompute_and(q0, q1, a, r, l0),
    )
