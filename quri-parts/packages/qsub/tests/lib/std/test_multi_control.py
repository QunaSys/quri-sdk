from typing import Sequence
import math

from quri_parts.qsub.compile import compile
from quri_parts.qsub.lib.std import (
    CNOT,
    CZ,
    Cbz,
    S,
    RY,
    Controlled,
    H,
    Label,
    M,
    MCY,
    MultiControlled,
    Sdag,
    T,
    Tdag,
    Toffoli,
    X,
    Y,
)
from quri_parts.qsub.lib.std.multi_control import (  # type: ignore
    MultiControlledCliffordTSub,
    MultiControlledSub,
)
from quri_parts.qsub.lib.std.multi_control_gates import (  # type: ignore
    generate_multicontrolled_sub_resolver,
)
from quri_parts.qsub.op import Op
from quri_parts.qsub.opsub import OpSubDef, opsub
from quri_parts.qsub.primitive import AllBasicSet, SimulatorBasicSet
from quri_parts.qsub.qubit import Qubit
from quri_parts.qsub.register import Register
from quri_parts.qsub.resolve import default_repository
from quri_parts.qsub.sub import SubBuilder


def test_multi_controlled_op() -> None:
    mcy = MultiControlled(Y, 3, 5)
    assert mcy.id.params == (Y, 3, 5)
    assert mcy.qubit_count == 4
    assert mcy.reg_count == 0


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


def test_multi_control_with_resolver() -> None:
    mcy = MultiControlled(Y, 2, 0b11)


    new_repo = default_repository().copy()  # type: ignore
    new_repo.register_sub_resolver(
        MultiControlled, generate_multicontrolled_sub_resolver()
    )


    default_compiled_basic = compile(mcy, AllBasicSet)
    assert [(inst[0].op, list(inst[1])) for inst in default_compiled_basic.instructions] == [
        (Toffoli, [Qubit(0), Qubit(1), Qubit(3)]),
        (Controlled(Y), [Qubit(3), Qubit(2)]),
        (Toffoli, [Qubit(0), Qubit(1), Qubit(3)]),
    ]
    
    default_compiled_sim = compile(mcy, SimulatorBasicSet)
    assert [(inst[0].op, list(inst[1])) for inst in default_compiled_sim.instructions] == [
        (Toffoli, [Qubit(0), Qubit(1), Qubit(3)]),
        (Controlled(Y), [Qubit(3), Qubit(2)]),
        (Toffoli, [Qubit(0), Qubit(1), Qubit(3)]),
    ]

    compiled_basic = compile(mcy, AllBasicSet, new_repo)
    assert [(inst[0].op, list(inst[1])) for inst in compiled_basic.instructions] == [
        (MCY(2), [Qubit(0)])
    ]

    compiled_sim = compile(mcy, SimulatorBasicSet, new_repo)
    assert [(inst[0].op, list(inst[1])) for inst in compiled_sim.instructions] == [
        (MCY(2), [Qubit(0)])
    ]


def test_multi_control_with_resolver_complex() -> None:
    # Create a complex OpSubDef with three operations
    class ComplexOpSubDef(OpSubDef):
        name = "ComplexOp"
        qubit_count = 3

        def sub(self, builder: SubBuilder) -> None:
            q0, q1, q2 = builder.qubits
            builder.add_op(H, (q0,))
            builder.add_op(CNOT, (q0, q1))
            builder.add_op(Y, (q2,))

    # Create Op and Sub from the definition and register in repositories
    default_repo = default_repository().copy()  # type: ignore
    complex_op, complex_sub = opsub(ComplexOpSubDef, default_repo)
    print(f"{default_repo._mapping=}")

    # Initialize MultiControlled op with the complex_op and compile it with compile()
    mcy_with_complex_sub = MultiControlled(complex_op, 3, 0b111)

    default_compiled_basic = compile(mcy_with_complex_sub, AllBasicSet, default_repo)
    default_compiled_sim = compile(
        mcy_with_complex_sub, SimulatorBasicSet, default_repo
    )

    new_repo = default_repository().copy()  # type: ignore
    new_repo.register_sub(complex_op, complex_sub)
    new_repo.register_sub_resolver(
        MultiControlled, generate_multicontrolled_sub_resolver()
    )
    print(f"{new_repo._mapping=}")

    compiled_basic = compile(mcy_with_complex_sub, AllBasicSet, new_repo)
    compiled_sim = compile(mcy_with_complex_sub, SimulatorBasicSet, new_repo)

    # Verify the compiled results contain expected operations
    assert [(inst[0].op, list(inst[1])) for inst in default_compiled_basic.instructions] == [
        (Toffoli, [Qubit(0), Qubit(1), Qubit(6)]),
        (Toffoli, [Qubit(6), Qubit(2), Qubit(7)]),
        (Controlled(complex_op), [Qubit(7), Qubit(3), Qubit(4), Qubit(5)]),
        (Toffoli, [Qubit(6), Qubit(2), Qubit(7)]),
        (Toffoli, [Qubit(0), Qubit(1), Qubit(6)]),
    ]
    
    # This does not generate MC* gates, because MultiControlled op will be expanded
    # from the internal in default approach.
    assert [(inst[0].op, list(inst[1])) for inst in default_compiled_sim.instructions] == [
        (Toffoli, [Qubit(0), Qubit(1), Qubit(6)]),
        (Toffoli, [Qubit(6), Qubit(2), Qubit(7)]),
        (Controlled(complex_op), [Qubit(7), Qubit(3), Qubit(4), Qubit(5)]),
        (Toffoli, [Qubit(6), Qubit(2), Qubit(7)]),
        (Toffoli, [Qubit(0), Qubit(1), Qubit(6)]),
    ]

    assert [(inst[0].op, list(inst[1])) for inst in compiled_basic.instructions] == [
        (MultiControlled(H, 3, 0b111), [Qubit(0), Qubit(1), Qubit(2), Qubit(3)]),
        (MultiControlled(CNOT, 3, 0b111), [Qubit(0), Qubit(1), Qubit(2), Qubit(3), Qubit(4)]),
        (MultiControlled(Y, 3, 0b111), [Qubit(0), Qubit(1), Qubit(2), Qubit(5)]),
    ]
    assert [(inst[0].op, list(inst[1])) for inst in compiled_sim.instructions] == [
        (MultiControlled(H, 3, 0b111), [Qubit(0), Qubit(1), Qubit(2), Qubit(3)]),
        (MultiControlled(CNOT, 3, 0b111), [Qubit(0), Qubit(1), Qubit(2), Qubit(3), Qubit(4)]),
        (MultiControlled(Y, 3, 0b111), [Qubit(0), Qubit(1), Qubit(2), Qubit(5)]),
    ]
