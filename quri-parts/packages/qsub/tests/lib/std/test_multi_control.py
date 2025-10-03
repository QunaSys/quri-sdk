from typing import Sequence

from quri_parts.qsub.lib.std import (
    CNOT,
    CZ,
    RX,
    RY,
    RZ,
    Cbz,
    Controlled,
    H,
    Label,
    M,
    MultiControlled,
    Phase,
    S,
    Sdag,
    T,
    Tdag,
    Toffoli,
    X,
    Y,
    Z,
)
from quri_parts.qsub.lib.std.multi_control import (
    MultiControlledCliffordTSub,
    MultiControlledSub,
)
from quri_parts.qsub.lib.std.multi_control_gates import (
    MCH,
    MCRX,
    MCRY,
    MCRZ,
    MCS,
    MCX,
    MCY,
    MCZ,
    MCPhase,
    MultiControlledNamedMCGatesSub,
    generate_multicontrolled_sub_resolver,
)
from quri_parts.qsub.op import Op
from quri_parts.qsub.opsub import OpSubDef, opsub
from quri_parts.qsub.qubit import Qubit
from quri_parts.qsub.register import Register
from quri_parts.qsub.resolve import default_repository, resolve_sub
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

    new_repo = default_repository().copy()
    new_repo.register_sub_resolver(
        MultiControlled, generate_multicontrolled_sub_resolver()
    )

    default_sub = resolve_sub(mcy)
    assert default_sub is not None
    assert len(default_sub.qubits) == 3
    assert len(default_sub.aux_qubits) == 1
    q0, q1, q2 = default_sub.qubits
    a0 = default_sub.aux_qubits[0]
    assert default_sub.operations == (
        (Toffoli, (q0, q1, a0), ()),
        (Controlled(Y), (a0, q2), ()),
        (Toffoli, (q0, q1, a0), ()),
    )

    resolved_sub = resolve_sub(mcy, new_repo)
    assert resolved_sub is not None
    assert len(resolved_sub.qubits) == 3
    assert len(resolved_sub.aux_qubits) == 0
    assert resolved_sub.operations == ((MCY(2), resolved_sub.qubits, ()),)


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
    default_repo = default_repository().copy()
    complex_op, complex_sub = opsub(ComplexOpSubDef, default_repo)

    # Initialize MultiControlled op with the complex_op
    mcy_with_complex_sub = MultiControlled(complex_op, 3, 0b111)

    default_sub = resolve_sub(mcy_with_complex_sub, default_repo)
    assert default_sub is not None
    assert len(default_sub.qubits) == 6
    assert len(default_sub.aux_qubits) == 2
    q0, q1, q2, q3, q4, q5 = default_sub.qubits
    a0, a1 = default_sub.aux_qubits
    assert default_sub.operations == (
        (Toffoli, (q0, q1, a0), ()),
        (Toffoli, (a0, q2, a1), ()),
        (Controlled(complex_op), (a1, q3, q4, q5), ()),
        (Toffoli, (a0, q2, a1), ()),
        (Toffoli, (q0, q1, a0), ()),
    )

    new_repo = default_repository().copy()
    new_repo.register_sub(complex_op, complex_sub)
    new_repo.register_sub_resolver(
        MultiControlled, generate_multicontrolled_sub_resolver()
    )

    resolved_sub = resolve_sub(mcy_with_complex_sub, new_repo)
    assert resolved_sub is not None
    assert len(resolved_sub.qubits) == 9
    assert len(resolved_sub.aux_qubits) == 0

    # Check that it has the expected MultiControlled operations
    q0, q1, q2, q3, q4, q5, q6, q7, q8 = resolved_sub.qubits
    assert resolved_sub.operations == (
        (MultiControlled(H, 3, 0b111), (q0, q1, q2, q3), ()),
        (MultiControlled(CNOT, 3, 0b111), (q0, q1, q2, q3, q4), ()),
        (MultiControlled(Y, 3, 0b111), (q0, q1, q2, q5), ()),
    )


def test_resolve_multicontrolled_various_control_values() -> None:
    """Test resolve_sub with MultiControlled operations using various control
    values."""
    # Test cases with different control bit patterns
    test_cases = [
        # (control_bits, control_value, expected_resolved_control_value)
        (1, 0b0, 0b0),  # Single control on |0⟩
        (1, 0b1, 0b1),  # Single control on |1⟩
        (2, 0b00, 0b00),  # Two controls on |00⟩
        (2, 0b01, 0b01),  # Two controls on |01⟩
        (2, 0b10, 0b10),  # Two controls on |10⟩
        (2, 0b11, 0b11),  # Two controls on |11⟩
        (3, 0b101, 0b101),  # Three controls on |101⟩
        (3, 0b111, 0b111),  # Three controls on |111⟩
    ]

    for control_bits, control_value, expected_resolved_control_value in test_cases:
        mc_toffoli = MultiControlled(Toffoli, control_bits, control_value)
        sub = resolve_sub(mc_toffoli)
        assert sub is not None

        # Assert that the resolved control value matches expected
        assert control_value == expected_resolved_control_value


def test_resolve_multicontrolled_toffoli_control_values() -> None:
    """Test resolve_sub with MultiControlled Toffoli using different control
    values."""
    # Test cases with expected resolved control values
    # When MultiControlled(Toffoli, n, value) is resolved,
    # it becomes MultiControlled(X, n+2, transformed_value)
    # because Toffoli is a 3-qubit gate (2 controls + 1 target)
    test_cases = [
        # (control_bits, control_value, expected_control_value)
        (1, 0b0, 0b110),  # Single control on |0⟩ -> 4-qubit MCX with control |110⟩
        (1, 0b1, 0b111),  # Single control on |1⟩ -> 4-qubit MCX with control |111⟩
        (2, 0b00, 0b1100),  # Double control on |00⟩ -> 5-qubit MCX with control |1100⟩
        (2, 0b01, 0b1101),  # Double control on |01⟩ -> 5-qubit MCX with control |1101⟩
        (2, 0b10, 0b1110),  # Double control on |10⟩ -> 5-qubit MCX with control |1110⟩
        (2, 0b11, 0b1111),  # Double control on |11⟩ -> 5-qubit MCX with control |1111⟩
        (
            3,
            0b000,
            0b11000,
        ),  # Triple control on |000⟩ -> 6-qubit MCX with control |11000⟩
        (
            3,
            0b101,
            0b11101,
        ),  # Triple control on |101⟩ -> 6-qubit MCX with control |11101⟩
        (
            3,
            0b111,
            0b11111,
        ),  # Triple control on |111⟩ -> 6-qubit MCX with control |11111⟩
    ]

    # Use a repository with the multi-controlled resolver
    # to preserve MultiControlled operations
    repo = default_repository().copy()
    repo.register_sub_resolver(MultiControlled, generate_multicontrolled_sub_resolver())

    for control_bits, control_value, expected_control_value in test_cases:
        mc_toffoli = MultiControlled(Toffoli, control_bits, control_value)
        sub = resolve_sub(mc_toffoli, repo)
        assert sub is not None

        # Assert that sub.operations consists of exactly one MultiControlled operation
        assert len(sub.operations) == 1

        op, qubits, registers = sub.operations[0]
        assert hasattr(op, "id") and op.id.local_name == "MultiControlled"

        # Assert that the control value of the MultiControlled operation
        # matches expected
        # MultiControlled params are (target_op, control_bits, control_value)
        assert len(op.id.params) >= 3
        resolved_control_value = op.id.params[2]
        assert resolved_control_value == expected_control_value


class TestMultiControlledNamedMCGatesSub:
    """Test the MultiControlledNamedMCGatesSub function."""

    def test_mcx_gate(self) -> None:
        """Test mapping X to MCX."""
        sub = MultiControlledNamedMCGatesSub(X, 2, 0b11)
        assert sub is not None
        assert len(sub.qubits) == 3
        assert len(sub.operations) == 1
        assert sub.operations[0][0] == MCX(2)

    def test_mcx_gate_with_negation(self) -> None:
        """Test mapping X to MCX with control negation."""
        sub = MultiControlledNamedMCGatesSub(X, 2, 0b01)
        assert sub is not None
        assert len(sub.qubits) == 3
        assert len(sub.operations) == 3
        # Should have X before and after the MCX
        assert sub.operations[0][0] == X
        assert sub.operations[1][0] == MCX(2)
        assert sub.operations[2][0] == X

    def test_mcy_gate(self) -> None:
        """Test mapping Y to MCY."""
        sub = MultiControlledNamedMCGatesSub(Y, 3, 0b111)
        assert sub is not None
        assert len(sub.qubits) == 4
        assert len(sub.operations) == 1
        assert sub.operations[0][0] == MCY(3)

    def test_mcz_gate(self) -> None:
        """Test mapping Z to MCZ."""
        sub = MultiControlledNamedMCGatesSub(Z, 1, 0b1)
        assert sub is not None
        assert len(sub.qubits) == 2
        assert len(sub.operations) == 1
        assert sub.operations[0][0] == MCZ(1)

    def test_mcs_gate(self) -> None:
        """Test mapping S to MCS."""
        sub = MultiControlledNamedMCGatesSub(S, 2, 0b11)
        assert sub is not None
        assert len(sub.qubits) == 3
        assert len(sub.operations) == 1
        assert sub.operations[0][0] == MCS(2)

    def test_mch_gate(self) -> None:
        """Test mapping H to MCH."""
        sub = MultiControlledNamedMCGatesSub(H, 1, 0b1)
        assert sub is not None
        assert len(sub.qubits) == 2
        assert len(sub.operations) == 1
        assert sub.operations[0][0] == MCH(1)

    def test_mcrx_gate(self) -> None:
        """Test mapping RX to MCRX."""
        angle = 1.5
        sub = MultiControlledNamedMCGatesSub(RX(angle), 2, 0b11)
        assert sub is not None
        assert len(sub.qubits) == 3
        assert len(sub.operations) == 1
        assert sub.operations[0][0] == MCRX(2, angle)

    def test_mcry_gate(self) -> None:
        """Test mapping RY to MCRY."""
        angle = 0.5
        sub = MultiControlledNamedMCGatesSub(RY(angle), 1, 0b1)
        assert sub is not None
        assert len(sub.qubits) == 2
        assert len(sub.operations) == 1
        assert sub.operations[0][0] == MCRY(1, angle)

    def test_mcrz_gate(self) -> None:
        """Test mapping RZ to MCRZ."""
        angle = 2.3
        sub = MultiControlledNamedMCGatesSub(RZ(angle), 3, 0b111)
        assert sub is not None
        assert len(sub.qubits) == 4
        assert len(sub.operations) == 1
        assert sub.operations[0][0] == MCRZ(3, angle)

    def test_mcphase_gate(self) -> None:
        """Test mapping Phase to MCPhase."""
        angle = 1.2
        sub = MultiControlledNamedMCGatesSub(Phase(angle), 2, 0b11)
        assert sub is not None
        assert len(sub.qubits) == 3
        assert len(sub.operations) == 1
        assert sub.operations[0][0] == MCPhase(2, angle)

    def test_cnot_gate(self) -> None:
        """Test mapping CNOT gate."""
        sub = MultiControlledNamedMCGatesSub(CNOT, 1, 0b1)
        assert sub is not None
        assert len(sub.qubits) == 3
        assert len(sub.operations) == 1
        assert sub.operations[0][0] == Toffoli

    def test_cnot_gate_with_multiple_controls(self) -> None:
        """Test mapping CNOT gate with multiple controls."""
        sub = MultiControlledNamedMCGatesSub(CNOT, 2, 0b11)
        assert sub is not None
        assert len(sub.qubits) == 4
        assert len(sub.operations) == 1
        assert sub.operations[0][0] == MultiControlled(X, 3, 0b111)

    def test_cz_gate(self) -> None:
        """Test mapping CZ gate."""
        sub = MultiControlledNamedMCGatesSub(CZ, 1, 0b1)
        assert sub is not None
        assert len(sub.qubits) == 3
        assert len(sub.operations) == 1
        assert sub.operations[0][0] == MultiControlled(Z, 2, 0b11)

    def test_toffoli_gate(self) -> None:
        """Test mapping Toffoli gate."""
        sub = MultiControlledNamedMCGatesSub(Toffoli, 1, 0b1)
        assert sub is not None
        assert len(sub.qubits) == 4
        assert len(sub.operations) == 1
        assert sub.operations[0][0] == MultiControlled(X, 3, 0b111)

    def test_controlled_gate(self) -> None:
        """Test mapping Controlled gate."""
        controlled_y = Controlled(Y)
        sub = MultiControlledNamedMCGatesSub(controlled_y, 2, 0b11)
        assert sub is not None
        assert len(sub.qubits) == 4
        assert len(sub.operations) == 1
        assert sub.operations[0][0] == MultiControlled(Y, 3, 0b111)

    def test_nested_multicontrolled_gate(self) -> None:
        """Test mapping nested MultiControlled gate."""
        nested_mc = MultiControlled(X, 2, 0b11)
        sub = MultiControlledNamedMCGatesSub(nested_mc, 1, 0b1)
        assert sub is not None
        assert len(sub.qubits) == 4
        assert len(sub.operations) == 1
        assert sub.operations[0][0] == MultiControlled(X, 3, 0b111)
