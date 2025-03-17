from collections.abc import Sequence

from quri_parts.qsub.lib.std import (
    CNOT,
    CZ,
    Cbz,
    Controlled,
    H,
    Label,
    M,
    Sdag,
    T,
    Tdag,
    Toffoli,
    X,
    Y,
    scoped_and,
    scoped_and_clifford_t,
)
from quri_parts.qsub.lib.std.logic import (
    MultiControlledCliffordTSub,
    MultiControlledSub,
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


class TestMultiControlled:
    def test_one_control(self) -> None:
        cy_sub = MultiControlledSub(Y, 1, 0b1)

        assert len(cy_sub.qubits) == 2
        assert len(cy_sub.aux_qubits) == 0

        i0, i1 = cy_sub.qubits
        assert cy_sub.operations == ((Controlled(Y), (i0, i1), ()),)

    def test_one_control_on_zero(self) -> None:
        cy_sub = MultiControlledSub(Y, 1, 0b0)

        assert len(cy_sub.qubits) == 2
        assert len(cy_sub.aux_qubits) == 0

        i0, i1 = cy_sub.qubits
        assert cy_sub.operations == (
            (X, (i0,), ()),
            (Controlled(Y), (i0, i1), ()),
            (X, (i0,), ()),
        )

    def test_two_controls(self) -> None:
        mcy_sub = MultiControlledSub(Y, 2, 0b11)

        assert len(mcy_sub.qubits) == 3
        assert len(mcy_sub.aux_qubits) == 1

        i0, i1, i2 = mcy_sub.qubits
        a0 = mcy_sub.aux_qubits[0]
        assert mcy_sub.operations == (
            (Toffoli, (i0, i1, a0), ()),
            (Controlled(Y), (a0, i2), ()),
            (Toffoli, (i0, i1, a0), ()),
        )

    def test_two_controls_on_zero(self) -> None:
        mcy_sub = MultiControlledSub(Y, 2, 0b10)

        assert len(mcy_sub.qubits) == 3
        assert len(mcy_sub.aux_qubits) == 1

        i0, i1, i2 = mcy_sub.qubits
        a0 = mcy_sub.aux_qubits[0]
        assert mcy_sub.operations == (
            (X, (i0,), ()),
            (Toffoli, (i0, i1, a0), ()),
            (Controlled(Y), (a0, i2), ()),
            (Toffoli, (i0, i1, a0), ()),
            (X, (i0,), ()),
        )

    def test_many_controls(self) -> None:
        mcy_sub = MultiControlledSub(Y, 4, 0b1111)

        assert len(mcy_sub.qubits) == 5
        assert len(mcy_sub.aux_qubits) == 3

        i0, i1, i2, i3, i4 = mcy_sub.qubits
        a0, a1, a2 = mcy_sub.aux_qubits
        assert mcy_sub.operations == (
            (Toffoli, (i0, i1, a0), ()),
            (Toffoli, (a0, i2, a1), ()),
            (Toffoli, (a1, i3, a2), ()),
            (Controlled(Y), (a2, i4), ()),
            (Toffoli, (a1, i3, a2), ()),
            (Toffoli, (a0, i2, a1), ()),
            (Toffoli, (i0, i1, a0), ()),
        )

    def test_many_controls_on_zero(self) -> None:
        mcy_sub = MultiControlledSub(Y, 4, 0b0101)

        assert len(mcy_sub.qubits) == 5
        assert len(mcy_sub.aux_qubits) == 3

        i0, i1, i2, i3, i4 = mcy_sub.qubits
        a0, a1, a2 = mcy_sub.aux_qubits
        assert mcy_sub.operations == (
            (X, (i1,), ()),
            (X, (i3,), ()),
            (Toffoli, (i0, i1, a0), ()),
            (Toffoli, (a0, i2, a1), ()),
            (Toffoli, (a1, i3, a2), ()),
            (Controlled(Y), (a2, i4), ()),
            (Toffoli, (a1, i3, a2), ()),
            (Toffoli, (a0, i2, a1), ()),
            (Toffoli, (i0, i1, a0), ()),
            (X, (i1,), ()),
            (X, (i3,), ()),
        )


class TestMultiControlledCliffordT:
    def test_one_control(self) -> None:
        cy_sub = MultiControlledCliffordTSub(Y, 1, 0b1)

        assert len(cy_sub.qubits) == 2
        assert len(cy_sub.aux_qubits) == 0

        i0, i1 = cy_sub.qubits
        assert cy_sub.operations == ((Controlled(Y), (i0, i1), ()),)

    def test_one_control_on_zero(self) -> None:
        cy_sub = MultiControlledCliffordTSub(Y, 1, 0b0)

        assert len(cy_sub.qubits) == 2
        assert len(cy_sub.aux_qubits) == 0

        i0, i1 = cy_sub.qubits
        assert cy_sub.operations == (
            (X, (i0,), ()),
            (Controlled(Y), (i0, i1), ()),
            (X, (i0,), ()),
        )

    def test_two_controls(self) -> None:
        mcy_sub = MultiControlledCliffordTSub(Y, 2, 0b11)

        assert len(mcy_sub.qubits) == 3
        assert len(mcy_sub.aux_qubits) == 1
        assert len(mcy_sub.aux_registers) == 2

        i0, i1, i2 = mcy_sub.qubits
        a0 = mcy_sub.aux_qubits[0]
        r, l0 = mcy_sub.aux_registers
        assert mcy_sub.operations == (
            *_clifford_t_and(i0, i1, a0),
            (Controlled(Y), (a0, i2), ()),
            *_clifford_t_uncompute_and(i0, i1, a0, r, l0),
        )

    def test_two_controls_on_zero(self) -> None:
        mcy_sub = MultiControlledCliffordTSub(Y, 2, 0b10)

        assert len(mcy_sub.qubits) == 3
        assert len(mcy_sub.aux_qubits) == 1
        assert len(mcy_sub.aux_registers) == 2

        i0, i1, i2 = mcy_sub.qubits
        a0 = mcy_sub.aux_qubits[0]
        r, l0 = mcy_sub.aux_registers
        assert mcy_sub.operations == (
            (X, (i0,), ()),
            *_clifford_t_and(i0, i1, a0),
            (Controlled(Y), (a0, i2), ()),
            *_clifford_t_uncompute_and(i0, i1, a0, r, l0),
            (X, (i0,), ()),
        )

    def test_many_controls(self) -> None:
        mcy_sub = MultiControlledCliffordTSub(Y, 4, 0b1111)

        assert len(mcy_sub.qubits) == 5
        assert len(mcy_sub.aux_qubits) == 3
        assert len(mcy_sub.aux_registers) == 6

        i0, i1, i2, i3, i4 = mcy_sub.qubits
        a0, a1, a2 = mcy_sub.aux_qubits
        r0, l0, r1, l1, r2, l2 = mcy_sub.aux_registers
        assert mcy_sub.operations == (
            *_clifford_t_and(i0, i1, a0),
            *_clifford_t_and(a0, i2, a1),
            *_clifford_t_and(a1, i3, a2),
            (Controlled(Y), (a2, i4), ()),
            *_clifford_t_uncompute_and(a1, i3, a2, r0, l0),
            *_clifford_t_uncompute_and(a0, i2, a1, r1, l1),
            *_clifford_t_uncompute_and(i0, i1, a0, r2, l2),
        )

    def test_many_controls_on_zero(self) -> None:
        mcy_sub = MultiControlledCliffordTSub(Y, 4, 0b0101)

        assert len(mcy_sub.qubits) == 5
        assert len(mcy_sub.aux_qubits) == 3
        assert len(mcy_sub.aux_registers) == 6

        i0, i1, i2, i3, i4 = mcy_sub.qubits
        a0, a1, a2 = mcy_sub.aux_qubits
        r0, l0, r1, l1, r2, l2 = mcy_sub.aux_registers
        assert mcy_sub.operations == (
            (X, (i1,), ()),
            (X, (i3,), ()),
            *_clifford_t_and(i0, i1, a0),
            *_clifford_t_and(a0, i2, a1),
            *_clifford_t_and(a1, i3, a2),
            (Controlled(Y), (a2, i4), ()),
            *_clifford_t_uncompute_and(a1, i3, a2, r0, l0),
            *_clifford_t_uncompute_and(a0, i2, a1, r1, l1),
            *_clifford_t_uncompute_and(i0, i1, a0, r2, l2),
            (X, (i1,), ()),
            (X, (i3,), ()),
        )
