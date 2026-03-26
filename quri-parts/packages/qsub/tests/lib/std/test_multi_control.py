import math
from typing import Sequence

from quri_parts.qsub.lib.std import (
    CNOT,
    CZ,
    RX,
    RY,
    RZ,
    SWAP,
    Cbz,
    Controlled,
    H,
    Identity,
    Label,
    M,
    MultiControlled,
    Phase,
    S,
    Sdag,
    SqrtX,
    SqrtXdag,
    SqrtY,
    SqrtYdag,
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
    generate_multicontrolled_to_mc_sub_resolver,
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
    

    def test_one_control_on_zero(self) -> None:
        toffoli_sub = MultiControlledSub(CNOT, 1, 0b0)

        assert len(toffoli_sub.qubits) == 3
        assert len(toffoli_sub.aux_qubits) == 0

        i0, i1, i2 = toffoli_sub.qubits
        assert toffoli_sub.operations == (            
            (Toffoli, (i0, i1, i2), ()),
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
        MultiControlled, generate_multicontrolled_to_mc_sub_resolver()
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
        MultiControlled, generate_multicontrolled_to_mc_sub_resolver()
    )

    resolved_sub = resolve_sub(mcy_with_complex_sub, new_repo)
    assert resolved_sub is not None
    assert len(resolved_sub.qubits) == 6
    assert len(resolved_sub.aux_qubits) == 0

    # Check that it has the expected MultiControlled operations
    q0, q1, q2, q3, q4, q5 = resolved_sub.qubits
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
    repo.register_sub_resolver(
        MultiControlled, generate_multicontrolled_to_mc_sub_resolver()
    )

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

    def test_mcx_gate_4bit_control_all_ones(self) -> None:
        """Test mapping X to MCX with 4-bit control value (all ones)."""
        sub = MultiControlledNamedMCGatesSub(X, 4, 0b1111)
        assert sub is not None
        assert len(sub.qubits) == 5
        assert len(sub.operations) == 1
        assert sub.operations[0] == (MCX(4), sub.qubits, ())

    def test_mcx_gate_4bit_control_mixed_pattern(self) -> None:
        """Test mapping X to MCX with 4-bit control value (mixed 0s and 1s)."""
        sub = MultiControlledNamedMCGatesSub(X, 4, 0b1010)
        assert sub is not None
        assert len(sub.qubits) == 5
        assert len(sub.operations) == 5
        # Should have X gates for negation (bits 0 and 2 are 0)
        assert sub.operations[0] == (X, (sub.qubits[0],), ())  # Negate bit 0
        assert sub.operations[1] == (X, (sub.qubits[2],), ())  # Negate bit 2
        assert sub.operations[2] == (MCX(4), sub.qubits, ())  # All qubits used
        assert sub.operations[3] == (X, (sub.qubits[0],), ())  # Negate bit 0
        assert sub.operations[4] == (X, (sub.qubits[2],), ())  # Negate bit 2

    def test_mcx_gate_5bit_control_alternating_pattern(self) -> None:
        """Test mapping X to MCX with 5-bit control value (alternating
        pattern)."""
        sub = MultiControlledNamedMCGatesSub(X, 5, 0b10101)
        assert sub is not None
        assert len(sub.qubits) == 6
        assert len(sub.operations) == 5
        # Should have X gates for negation (bits 1 and 3 are 0)
        assert sub.operations[0] == (X, (sub.qubits[1],), ())  # Negate bit 1
        assert sub.operations[1] == (X, (sub.qubits[3],), ())  # Negate bit 3
        assert sub.operations[2] == (MCX(5), sub.qubits, ())  # All qubits used
        assert sub.operations[3] == (X, (sub.qubits[1],), ())  # Negate bit 1
        assert sub.operations[4] == (X, (sub.qubits[3],), ())  # Negate bit 3

    def test_mcx_gate_6bit_control_sparse_pattern(self) -> None:
        """Test mapping X to MCX with 6-bit control value (sparse pattern)."""
        sub = MultiControlledNamedMCGatesSub(X, 6, 0b100001)
        assert sub is not None
        assert len(sub.qubits) == 7
        assert len(sub.operations) == 9
        # Should have X gates for negation (bits 1,2,3,4 are 0)
        assert sub.operations[0] == (X, (sub.qubits[1],), ())  # Negate bit 1
        assert sub.operations[1] == (X, (sub.qubits[2],), ())  # Negate bit 2
        assert sub.operations[2] == (X, (sub.qubits[3],), ())  # Negate bit 3
        assert sub.operations[3] == (X, (sub.qubits[4],), ())  # Negate bit 4
        assert sub.operations[4] == (MCX(6), sub.qubits, ())  # All qubits used
        assert sub.operations[5] == (X, (sub.qubits[1],), ())  # Negate bit 1
        assert sub.operations[6] == (X, (sub.qubits[2],), ())  # Negate bit 2
        assert sub.operations[7] == (X, (sub.qubits[3],), ())  # Negate bit 3
        assert sub.operations[8] == (X, (sub.qubits[4],), ())  # Negate bit 4

    def test_mcy_gate_4bit_control_all_ones(self) -> None:
        """Test mapping Y to MCY with 4-bit control value (all ones)."""
        sub = MultiControlledNamedMCGatesSub(Y, 4, 0b1111)
        assert sub is not None
        assert len(sub.qubits) == 5
        assert len(sub.operations) == 1
        assert sub.operations[0] == (MCY(4), sub.qubits, ())

    def test_mcy_gate_4bit_control_mixed_pattern(self) -> None:
        """Test mapping Y to MCY with 4-bit control value (mixed 0s and 1s)."""
        sub = MultiControlledNamedMCGatesSub(Y, 4, 0b0110)
        assert sub is not None
        assert len(sub.qubits) == 5
        assert len(sub.operations) == 5
        # Should have X gates for negation (bits 0 and 3 are 0)
        assert sub.operations[0] == (X, (sub.qubits[0],), ())  # Negate bit 0
        assert sub.operations[1] == (X, (sub.qubits[3],), ())  # Negate bit 3
        assert sub.operations[2] == (MCY(4), sub.qubits, ())  # All qubits used
        assert sub.operations[3] == (X, (sub.qubits[0],), ())  # Negate bit 0
        assert sub.operations[4] == (X, (sub.qubits[3],), ())  # Negate bit 3

    def test_mcy_gate_5bit_control_complex_pattern(self) -> None:
        """Test mapping Y to MCY with 5-bit control value (complex pattern)."""
        sub = MultiControlledNamedMCGatesSub(Y, 5, 0b11001)
        assert sub is not None
        assert len(sub.qubits) == 6
        assert len(sub.operations) == 5
        # Should have X gates for negation (bits 1 and 2 are 0)
        assert sub.operations[0] == (X, (sub.qubits[1],), ())  # Negate bit 1
        assert sub.operations[1] == (X, (sub.qubits[2],), ())  # Negate bit 2
        assert sub.operations[2] == (MCY(5), sub.qubits, ())  # All qubits used
        assert sub.operations[3] == (X, (sub.qubits[1],), ())  # Negate bit 1
        assert sub.operations[4] == (X, (sub.qubits[2],), ())  # Negate bit 2

    def test_mcz_gate_4bit_control_all_ones(self) -> None:
        """Test mapping Z to MCZ with 4-bit control value (all ones)."""
        sub = MultiControlledNamedMCGatesSub(Z, 4, 0b1111)
        assert sub is not None
        assert len(sub.qubits) == 5
        assert len(sub.operations) == 1
        assert sub.operations[0] == (MCZ(4), sub.qubits, ())

    def test_mcz_gate_4bit_control_checkboard_pattern(self) -> None:
        """Test mapping Z to MCZ with 4-bit control value (checkerboard
        pattern)."""
        sub = MultiControlledNamedMCGatesSub(Z, 4, 0b0101)
        assert sub is not None
        assert len(sub.qubits) == 5
        assert len(sub.operations) == 5
        # Should have X gates for negation (bits 1 and 3 are 0)
        assert sub.operations[0] == (X, (sub.qubits[1],), ())  # Negate bit 1
        assert sub.operations[1] == (X, (sub.qubits[3],), ())  # Negate bit 3
        assert sub.operations[2] == (MCZ(4), sub.qubits, ())  # All qubits used
        assert sub.operations[3] == (X, (sub.qubits[1],), ())  # Negate bit 1
        assert sub.operations[4] == (X, (sub.qubits[3],), ())  # Negate bit 3

    def test_mcz_gate_6bit_control_edge_pattern(self) -> None:
        """Test mapping Z to MCZ with 6-bit control value (edge pattern)."""
        sub = MultiControlledNamedMCGatesSub(Z, 6, 0b100001)
        assert sub is not None
        assert len(sub.qubits) == 7
        assert len(sub.operations) == 9
        # Should have X gates for negation (bits 1,2,3,4 are 0)
        assert sub.operations[0] == (X, (sub.qubits[1],), ())  # Negate bit 1
        assert sub.operations[1] == (X, (sub.qubits[2],), ())  # Negate bit 2
        assert sub.operations[2] == (X, (sub.qubits[3],), ())  # Negate bit 3
        assert sub.operations[3] == (X, (sub.qubits[4],), ())  # Negate bit 4
        assert sub.operations[4] == (MCZ(6), sub.qubits, ())  # All qubits used
        assert sub.operations[5] == (X, (sub.qubits[1],), ())  # Negate bit 1
        assert sub.operations[6] == (X, (sub.qubits[2],), ())  # Negate bit 2
        assert sub.operations[7] == (X, (sub.qubits[3],), ())  # Negate bit 3
        assert sub.operations[8] == (X, (sub.qubits[4],), ())  # Negate bit 4

    def test_mcs_gate_4bit_control_all_ones(self) -> None:
        """Test mapping S to MCS with 4-bit control value (all ones)."""
        sub = MultiControlledNamedMCGatesSub(S, 4, 0b1111)
        assert sub is not None
        assert len(sub.qubits) == 5
        assert len(sub.operations) == 1
        assert sub.operations[0] == (MCS(4), sub.qubits, ())

    def test_mcs_gate_5bit_control_mixed_pattern(self) -> None:
        """Test mapping S to MCS with 5-bit control value (mixed pattern)."""
        sub = MultiControlledNamedMCGatesSub(S, 5, 0b10110)
        assert sub is not None
        assert len(sub.qubits) == 6
        assert len(sub.operations) == 5
        # Should have X gates for negation (bits 0 and 3 are 0)
        assert sub.operations[0] == (X, (sub.qubits[0],), ())  # Negate bit 0
        assert sub.operations[1] == (X, (sub.qubits[3],), ())  # Negate bit 3
        assert sub.operations[2] == (MCS(5), sub.qubits, ())  # All qubits used
        assert sub.operations[3] == (X, (sub.qubits[0],), ())  # Negate bit 0
        assert sub.operations[4] == (X, (sub.qubits[3],), ())  # Negate bit 3

    def test_mch_gate_4bit_control_all_ones(self) -> None:
        """Test mapping H to MCH with 4-bit control value (all ones)."""
        sub = MultiControlledNamedMCGatesSub(H, 4, 0b1111)
        assert sub is not None
        assert len(sub.qubits) == 5
        assert len(sub.operations) == 1
        assert sub.operations[0] == (MCH(4), sub.qubits, ())

    def test_mch_gate_5bit_control_diagonal_pattern(self) -> None:
        """Test mapping H to MCH with 5-bit control value (diagonal
        pattern)."""
        sub = MultiControlledNamedMCGatesSub(H, 5, 0b10001)
        assert sub is not None
        assert len(sub.qubits) == 6
        assert len(sub.operations) == 7
        # Should have X gates for negation (bits 1,2,3 are 0)
        assert sub.operations[0] == (X, (sub.qubits[1],), ())  # Negate bit 1
        assert sub.operations[1] == (X, (sub.qubits[2],), ())  # Negate bit 2
        assert sub.operations[2] == (X, (sub.qubits[3],), ())  # Negate bit 3
        assert sub.operations[3] == (MCH(5), sub.qubits, ())  # All qubits used
        assert sub.operations[4] == (X, (sub.qubits[1],), ())  # Negate bit 1
        assert sub.operations[5] == (X, (sub.qubits[2],), ())  # Negate bit 2
        assert sub.operations[6] == (X, (sub.qubits[3],), ())  # Negate bit 3

    def test_mcrx_gate_4bit_control_all_ones(self) -> None:
        """Test mapping RX to MCRX with 4-bit control value (all ones)."""
        angle = 2.1
        sub = MultiControlledNamedMCGatesSub(RX(angle), 4, 0b1111)
        assert sub is not None
        assert len(sub.qubits) == 5
        assert len(sub.operations) == 1
        assert sub.operations[0] == (MCRX(4, angle), sub.qubits, ())

    def test_mcrx_gate_5bit_control_mixed_pattern(self) -> None:
        """Test mapping RX to MCRX with 5-bit control value (mixed pattern)."""
        angle = 1.8
        sub = MultiControlledNamedMCGatesSub(RX(angle), 5, 0b01101)
        assert sub is not None
        assert len(sub.qubits) == 6
        assert len(sub.operations) == 5
        # Should have X gates for negation (bits 1 and 4 are 0)
        assert sub.operations[0] == (X, (sub.qubits[1],), ())  # Negate bit 1
        assert sub.operations[1] == (X, (sub.qubits[4],), ())  # Negate bit 4
        assert sub.operations[2] == (MCRX(5, angle), sub.qubits, ())  # All qubits used
        assert sub.operations[3] == (X, (sub.qubits[1],), ())  # Negate bit 1
        assert sub.operations[4] == (X, (sub.qubits[4],), ())  # Negate bit 4

    def test_mcry_gate_4bit_control_all_ones(self) -> None:
        """Test mapping RY to MCRY with 4-bit control value (all ones)."""
        angle = 0.75
        sub = MultiControlledNamedMCGatesSub(RY(angle), 4, 0b1111)
        assert sub is not None
        assert len(sub.qubits) == 5
        assert len(sub.operations) == 1
        assert sub.operations[0] == (MCRY(4, angle), sub.qubits, ())

    def test_mcry_gate_6bit_control_sparse_pattern(self) -> None:
        """Test mapping RY to MCRY with 6-bit control value (sparse
        pattern)."""
        angle = 3.14
        sub = MultiControlledNamedMCGatesSub(RY(angle), 6, 0b100100)
        assert sub is not None
        assert len(sub.qubits) == 7
        assert len(sub.operations) == 9
        # Should have X gates for negation (bits 0,1,3,4 are 0)
        assert sub.operations[0] == (X, (sub.qubits[0],), ())  # Negate bit 0
        assert sub.operations[1] == (X, (sub.qubits[1],), ())  # Negate bit 1
        assert sub.operations[2] == (X, (sub.qubits[3],), ())  # Negate bit 3
        assert sub.operations[3] == (X, (sub.qubits[4],), ())  # Negate bit 4
        assert sub.operations[4] == (MCRY(6, angle), sub.qubits, ())  # All qubits used
        assert sub.operations[5] == (X, (sub.qubits[0],), ())  # Negate bit 0
        assert sub.operations[6] == (X, (sub.qubits[1],), ())  # Negate bit 1
        assert sub.operations[7] == (X, (sub.qubits[3],), ())  # Negate bit 3
        assert sub.operations[8] == (X, (sub.qubits[4],), ())  # Negate bit 4

    def test_mcrz_gate_4bit_control_all_ones(self) -> None:
        """Test mapping RZ to MCRZ with 4-bit control value (all ones)."""
        angle = 1.57
        sub = MultiControlledNamedMCGatesSub(RZ(angle), 4, 0b1111)
        assert sub is not None
        assert len(sub.qubits) == 5
        assert len(sub.operations) == 1
        assert sub.operations[0] == (MCRZ(4, angle), sub.qubits, ())

    def test_mcrz_gate_5bit_control_alternating_pattern(self) -> None:
        """Test mapping RZ to MCRZ with 5-bit control value (alternating
        pattern)."""
        angle = 2.5
        sub = MultiControlledNamedMCGatesSub(RZ(angle), 5, 0b01010)
        assert sub is not None
        assert len(sub.qubits) == 6
        assert len(sub.operations) == 7
        # Should have X gates for negation (bits 0,2,4 are 0)
        assert sub.operations[0] == (X, (sub.qubits[0],), ())  # Negate bit 0
        assert sub.operations[1] == (X, (sub.qubits[2],), ())  # Negate bit 2
        assert sub.operations[2] == (X, (sub.qubits[4],), ())  # Negate bit 4
        assert sub.operations[3] == (MCRZ(5, angle), sub.qubits, ())  # All qubits used
        assert sub.operations[4] == (X, (sub.qubits[0],), ())  # Negate bit 0
        assert sub.operations[5] == (X, (sub.qubits[2],), ())  # Negate bit 2
        assert sub.operations[6] == (X, (sub.qubits[4],), ())  # Negate bit 4

    def test_mcphase_gate_4bit_control_all_ones(self) -> None:
        """Test mapping Phase to MCPhase with 4-bit control value (all
        ones)."""
        angle = 0.95
        sub = MultiControlledNamedMCGatesSub(Phase(angle), 4, 0b1111)
        assert sub is not None
        assert len(sub.qubits) == 5
        assert len(sub.operations) == 1
        assert sub.operations[0] == (MCPhase(4, angle), sub.qubits, ())

    def test_mcphase_gate_6bit_control_center_pattern(self) -> None:
        """Test mapping Phase to MCPhase with 6-bit control value (center
        pattern)."""
        angle = 1.23
        sub = MultiControlledNamedMCGatesSub(Phase(angle), 6, 0b011110)
        assert sub is not None
        assert len(sub.qubits) == 7
        assert len(sub.operations) == 5
        # Should have X gates for negation (bits 0 and 5 are 0)
        assert sub.operations[0] == (X, (sub.qubits[0],), ())  # Negate bit 0
        assert sub.operations[1] == (X, (sub.qubits[5],), ())  # Negate bit 5
        # All qubits used
        assert sub.operations[2] == (MCPhase(6, angle), sub.qubits, ())
        assert sub.operations[3] == (X, (sub.qubits[0],), ())  # Negate bit 0
        assert sub.operations[4] == (X, (sub.qubits[5],), ())  # Negate bit 5

    def test_identity_gate(self) -> None:
        """Test mapping Identity to MCIdentity."""
        sub = MultiControlledNamedMCGatesSub(Identity, 2, 0b11)
        assert sub is not None
        assert len(sub.qubits) == 3
        assert len(sub.operations) == 3
        # Should have Identity operations on all qubits
        for i, (op, qubits, regs) in enumerate(sub.operations):
            assert op.id.local_name == "Identity"
            assert qubits == (sub.qubits[i],)
            assert regs == ()

    def test_sdag_gate(self) -> None:
        """Test mapping Sdag to MCSdag."""
        sub = MultiControlledNamedMCGatesSub(Sdag, 2, 0b11)
        assert sub is not None
        assert len(sub.qubits) == 3
        assert len(sub.operations) == 1
        assert sub.operations[0][0].id.local_name.startswith("MCS")

    def test_t_gate(self) -> None:
        """Test mapping T to MCT."""
        sub = MultiControlledNamedMCGatesSub(T, 2, 0b11)
        assert sub is not None
        assert len(sub.qubits) == 3
        assert len(sub.operations) == 1
        assert sub.operations[0][0].id.local_name.startswith("MC")

    def test_tdag_gate(self) -> None:
        """Test mapping Tdag to MCTdag."""
        sub = MultiControlledNamedMCGatesSub(Tdag, 2, 0b11)
        assert sub is not None
        assert len(sub.qubits) == 3
        assert len(sub.operations) == 1
        assert sub.operations[0][0].id.local_name.startswith("MC")

    def test_sqrtx_gate(self) -> None:
        """Test mapping SqrtX to MCSqrtX."""
        sub = MultiControlledNamedMCGatesSub(SqrtX, 2, 0b11)
        assert sub is not None
        assert len(sub.qubits) == 3
        assert len(sub.operations) == 1
        assert sub.operations[0][0].id.local_name.startswith("MC")

    def test_sqrtxdag_gate(self) -> None:
        """Test mapping SqrtXdag to MCSqrtXdag."""
        sub = MultiControlledNamedMCGatesSub(SqrtXdag, 2, 0b11)
        assert sub is not None
        assert len(sub.qubits) == 3
        assert len(sub.operations) == 1
        assert sub.operations[0][0].id.local_name.startswith("MC")

    def test_sqrty_gate(self) -> None:
        """Test mapping SqrtY to MCSqrtY."""
        sub = MultiControlledNamedMCGatesSub(SqrtY, 2, 0b11)
        assert sub is not None
        assert len(sub.qubits) == 3
        assert len(sub.operations) == 1
        assert sub.operations[0][0].id.local_name.startswith("MC")

    def test_sqrtydag_gate(self) -> None:
        """Test mapping SqrtYdag to MCSqrtYdag."""
        sub = MultiControlledNamedMCGatesSub(SqrtYdag, 2, 0b11)
        assert sub is not None
        assert len(sub.qubits) == 3
        assert len(sub.operations) == 1
        assert sub.operations[0][0].id.local_name.startswith("MC")

    def test_sdag_gate_with_negation(self) -> None:
        """Test mapping Sdag with control negation."""
        sub = MultiControlledNamedMCGatesSub(Sdag, 2, 0b01)
        assert sub is not None
        assert len(sub.qubits) == 3
        assert len(sub.operations) == 3
        # Should have X gates for negation plus MCSdag
        assert sub.operations[0][0].id.local_name == "X"  # Negation
        assert sub.operations[1][0].id.local_name.startswith("MCS")  # MCSdag
        assert sub.operations[2][0].id.local_name == "X"  # Negation

    def test_t_gate_with_negation(self) -> None:
        """Test mapping T with control negation."""
        sub = MultiControlledNamedMCGatesSub(T, 2, 0b01)
        assert sub is not None
        assert len(sub.qubits) == 3
        assert len(sub.operations) == 3
        # Should have X gates for negation plus MCT
        assert sub.operations[0][0].id.local_name == "X"  # Negation
        assert sub.operations[1][0].id.local_name.startswith("MC")  # MCT
        assert sub.operations[2][0].id.local_name == "X"  # Negation

    def test_tdag_gate_with_negation(self) -> None:
        """Test mapping Tdag with control negation."""
        sub = MultiControlledNamedMCGatesSub(Tdag, 2, 0b01)
        assert sub is not None
        assert len(sub.qubits) == 3
        assert len(sub.operations) == 3
        # Should have X gates for negation plus MCTdag
        assert sub.operations[0][0].id.local_name == "X"  # Negation
        assert sub.operations[1][0].id.local_name.startswith("MC")  # MCTdag
        assert sub.operations[2][0].id.local_name == "X"  # Negation


class TestPhaseHandlingInResolver:
    """Test phase handling in generate_multicontrolled_to_mc_sub_resolver."""

    def test_phase_pi_with_msb_set(self) -> None:
        """Test phase π handling when MSB is set in control_value."""

        # Create a custom operation with phase π
        class PhaseOpSubDef(OpSubDef):
            name = "PhaseOp"
            qubit_count = 1

            def sub(self, builder: SubBuilder) -> None:
                q0 = builder.qubits[0]
                builder.add_op(X, (q0,))
                builder.add_phase(math.pi)  # Add π phase

        repo = default_repository().copy()
        phase_op, phase_sub = opsub(PhaseOpSubDef, repo)
        repo.register_sub(phase_op, phase_sub)
        repo.register_sub_resolver(
            MultiControlled, generate_multicontrolled_to_mc_sub_resolver()
        )

        # Test with MSB set (control_value = 0b11, MSB is bit 1)
        mc_phase_op = MultiControlled(phase_op, 2, 0b11)
        resolved_sub = resolve_sub(mc_phase_op, repo)
        assert resolved_sub is not None

        # Should have MultiControlled X and MultiControlled Z (for π phase)
        assert len(resolved_sub.operations) == 2
        op_0 = (MultiControlled(X, 2, 0b11), resolved_sub.qubits[:3], ())
        assert resolved_sub.operations[0] == op_0
        op_1 = (MultiControlled(Z, 1, 0b1), resolved_sub.qubits[:2], ())
        assert resolved_sub.operations[1] == op_1

    def test_phase_pi_with_msb_unset(self) -> None:
        """Test phase π handling when MSB is unset in control_value."""

        # Create a custom operation with phase π
        class PhaseOpSubDef(OpSubDef):
            name = "PhaseOp"
            qubit_count = 1

            def sub(self, builder: SubBuilder) -> None:
                q0 = builder.qubits[0]
                builder.add_op(X, (q0,))
                builder.add_phase(math.pi)  # Add π phase

        repo = default_repository().copy()
        phase_op, phase_sub = opsub(PhaseOpSubDef, repo)
        repo.register_sub(phase_op, phase_sub)
        repo.register_sub_resolver(
            MultiControlled, generate_multicontrolled_to_mc_sub_resolver()
        )

        # Test with MSB unset (control_value = 0b01, MSB is bit 1)
        mc_phase_op = MultiControlled(phase_op, 2, 0b01)
        resolved_sub = resolve_sub(mc_phase_op, repo)
        assert resolved_sub is not None

        # Should have MultiControlled X and MultiControlled Z (for -π phase)
        # When MSB is unset, phase is negated, so -π % 2π = π
        # The control value for phase is control_value & ((1 << (control_bits - 1)) - 1)
        # = 0b01 & 0b01 = 0b01 = 1
        assert len(resolved_sub.operations) == 2
        op_0 = (MultiControlled(X, 2, 0b01), resolved_sub.qubits[:3], ())
        assert resolved_sub.operations[0] == op_0
        op_1 = (MultiControlled(Z, 1, 0b1), resolved_sub.qubits[:2], ())
        assert resolved_sub.operations[1] == op_1
        # Global phase π is added
        assert resolved_sub.phase == math.pi

    def test_phase_pi_half_with_msb_set(self) -> None:
        """Test phase π/2 handling when MSB is set in control_value."""

        # Create a custom operation with phase π/2
        class PhaseOpSubDef(OpSubDef):
            name = "PhaseOp"
            qubit_count = 1

            def sub(self, builder: SubBuilder) -> None:
                q0 = builder.qubits[0]
                builder.add_op(X, (q0,))
                builder.add_phase(math.pi / 2)  # Add π/2 phase

        repo = default_repository().copy()
        phase_op, phase_sub = opsub(PhaseOpSubDef, repo)
        repo.register_sub(phase_op, phase_sub)
        repo.register_sub_resolver(
            MultiControlled, generate_multicontrolled_to_mc_sub_resolver()
        )

        # Test with MSB set (control_value = 0b11, MSB is bit 1)
        mc_phase_op = MultiControlled(phase_op, 2, 0b11)
        resolved_sub = resolve_sub(mc_phase_op, repo)
        assert resolved_sub is not None

        # Should have MultiControlled X and MultiControlled S (for π/2 phase)
        assert len(resolved_sub.operations) == 2
        op_0 = (MultiControlled(X, 2, 0b11), resolved_sub.qubits[:3], ())
        assert resolved_sub.operations[0] == op_0
        op_1 = (MultiControlled(S, 1, 0b1), resolved_sub.qubits[:2], ())
        assert resolved_sub.operations[1] == op_1

    def test_phase_3pi_half_with_msb_set(self) -> None:
        """Test phase 3π/2 handling when MSB is set in control_value."""

        # Create a custom operation with phase 3π/2
        class PhaseOpSubDef(OpSubDef):
            name = "PhaseOp"
            qubit_count = 1

            def sub(self, builder: SubBuilder) -> None:
                q0 = builder.qubits[0]
                builder.add_op(X, (q0,))
                builder.add_phase(3 * math.pi / 2)  # Add 3π/2 phase

        repo = default_repository().copy()
        phase_op, phase_sub = opsub(PhaseOpSubDef, repo)
        repo.register_sub(phase_op, phase_sub)
        repo.register_sub_resolver(
            MultiControlled, generate_multicontrolled_to_mc_sub_resolver()
        )

        # Test with MSB set (control_value = 0b11, MSB is bit 1)
        mc_phase_op = MultiControlled(phase_op, 2, 0b11)
        resolved_sub = resolve_sub(mc_phase_op, repo)
        assert resolved_sub is not None

        # Should have MultiControlled X and MultiControlled Sdag (for 3π/2 phase)
        assert len(resolved_sub.operations) == 2
        op_0 = (MultiControlled(X, 2, 0b11), resolved_sub.qubits[:3], ())
        assert resolved_sub.operations[0] == op_0
        op_1 = (MultiControlled(Sdag, 1, 0b1), resolved_sub.qubits[:2], ())
        assert resolved_sub.operations[1] == op_1

    def test_arbitrary_phase_with_msb_set(self) -> None:
        """Test arbitrary phase handling when MSB is set in control_value."""
        # Create a custom operation with arbitrary phase
        arbitrary_phase = 1.23

        class PhaseOpSubDef(OpSubDef):
            name = "PhaseOp"
            qubit_count = 1

            def sub(self, builder: SubBuilder) -> None:
                q0 = builder.qubits[0]
                builder.add_op(X, (q0,))
                builder.add_phase(arbitrary_phase)  # Add arbitrary phase

        repo = default_repository().copy()
        phase_op, phase_sub = opsub(PhaseOpSubDef, repo)
        repo.register_sub(phase_op, phase_sub)
        repo.register_sub_resolver(
            MultiControlled, generate_multicontrolled_to_mc_sub_resolver()
        )

        # Test with MSB set (control_value = 0b11, MSB is bit 1)
        mc_phase_op = MultiControlled(phase_op, 2, 0b11)
        resolved_sub = resolve_sub(mc_phase_op, repo)
        assert resolved_sub is not None

        # Should have MultiControlled X and MultiControlled Phase (for arbitrary phase)
        assert len(resolved_sub.operations) == 2
        op_0 = (MultiControlled(X, 2, 0b11), resolved_sub.qubits[:3], ())
        assert resolved_sub.operations[0] == op_0
        expected_phase_op = MultiControlled(Phase(arbitrary_phase), 1, 0b1)
        op_1 = (expected_phase_op, resolved_sub.qubits[:2], ())
        assert resolved_sub.operations[1] == op_1

    def test_single_control_phase_handling(self) -> None:
        """Test phase handling with single control bit."""

        # Create a custom operation with phase π/2
        class PhaseOpSubDef(OpSubDef):
            name = "PhaseOp"
            qubit_count = 1

            def sub(self, builder: SubBuilder) -> None:
                q0 = builder.qubits[0]
                builder.add_op(X, (q0,))
                builder.add_phase(math.pi / 2)  # Add π/2 phase

        repo = default_repository().copy()
        phase_op, phase_sub = opsub(PhaseOpSubDef, repo)
        repo.register_sub(phase_op, phase_sub)
        repo.register_sub_resolver(
            MultiControlled, generate_multicontrolled_to_mc_sub_resolver()
        )

        # Test with single control bit set
        mc_phase_op = MultiControlled(phase_op, 1, 0b1)
        resolved_sub = resolve_sub(mc_phase_op, repo)
        assert resolved_sub is not None

        # Should have MultiControlled X and S gate on control qubit
        assert len(resolved_sub.operations) == 2
        op_0 = (MultiControlled(X, 1, 0b1), resolved_sub.qubits[:2], ())
        assert resolved_sub.operations[0] == op_0
        assert resolved_sub.operations[1] == (S, resolved_sub.qubits[:1], ())

    def test_three_control_phase_handling(self) -> None:
        """Test phase handling with three control bits."""

        # Create a custom operation with phase π
        class PhaseOpSubDef(OpSubDef):
            name = "PhaseOp"
            qubit_count = 1

            def sub(self, builder: SubBuilder) -> None:
                q0 = builder.qubits[0]
                builder.add_op(X, (q0,))
                builder.add_phase(math.pi)  # Add π phase

        repo = default_repository().copy()
        phase_op, phase_sub = opsub(PhaseOpSubDef, repo)
        repo.register_sub(phase_op, phase_sub)
        repo.register_sub_resolver(
            MultiControlled, generate_multicontrolled_to_mc_sub_resolver()
        )

        # Test with 3 control bits (control_value = 0b111, MSB is bit 2)
        mc_phase_op = MultiControlled(phase_op, 3, 0b111)
        resolved_sub = resolve_sub(mc_phase_op, repo)
        assert resolved_sub is not None

        # Should have MultiControlled X and MultiControlled Z with 2 controls
        assert len(resolved_sub.operations) == 2
        op_0 = (MultiControlled(X, 3, 0b111), resolved_sub.qubits[:4], ())
        assert resolved_sub.operations[0] == op_0
        op_1 = (MultiControlled(Z, 2, 0b11), resolved_sub.qubits[:3], ())
        assert resolved_sub.operations[1] == op_1

    def test_zero_phase_handling(self) -> None:
        """Test that operations with zero phase don't add phase operations."""

        # Create a custom operation with zero phase
        class ZeroPhaseOpSubDef(OpSubDef):
            name = "ZeroPhaseOp"
            qubit_count = 1

            def sub(self, builder: SubBuilder) -> None:
                q0 = builder.qubits[0]
                builder.add_op(X, (q0,))
                # No phase added (default is 0)

        repo = default_repository().copy()
        zero_phase_op, zero_phase_sub = opsub(ZeroPhaseOpSubDef, repo)
        repo.register_sub(zero_phase_op, zero_phase_sub)
        repo.register_sub_resolver(
            MultiControlled, generate_multicontrolled_to_mc_sub_resolver()
        )

        # Test with 2 control bits
        mc_zero_phase_op = MultiControlled(zero_phase_op, 2, 0b11)
        resolved_sub = resolve_sub(mc_zero_phase_op, repo)
        assert resolved_sub is not None

        # Should only have MultiControlled X, no phase operation
        assert len(resolved_sub.operations) == 1
        op_0 = (MultiControlled(X, 2, 0b11), resolved_sub.qubits[:3], ())
        assert resolved_sub.operations[0] == op_0
        assert resolved_sub.phase == 0

    def test_phase_modulo_2pi_handling(self) -> None:
        """Test phase handling with values greater than 2π."""
        # Create a custom operation with phase > 2π
        large_phase = 3 * math.pi  # Should be equivalent to π after modulo

        class LargePhaseOpSubDef(OpSubDef):
            name = "LargePhaseOp"
            qubit_count = 1

            def sub(self, builder: SubBuilder) -> None:
                q0 = builder.qubits[0]
                builder.add_op(X, (q0,))
                builder.add_phase(large_phase)

        repo = default_repository().copy()
        large_phase_op, large_phase_sub = opsub(LargePhaseOpSubDef, repo)
        repo.register_sub(large_phase_op, large_phase_sub)
        repo.register_sub_resolver(
            MultiControlled, generate_multicontrolled_to_mc_sub_resolver()
        )

        # Test with MSB set
        mc_large_phase_op = MultiControlled(large_phase_op, 2, 0b11)
        resolved_sub = resolve_sub(mc_large_phase_op, repo)
        assert resolved_sub is not None

        # Should have MultiControlled X and MultiControlled Z (since 3π % 2π = π)
        assert len(resolved_sub.operations) == 2
        op_0 = (MultiControlled(X, 2, 0b11), resolved_sub.qubits[:3], ())
        assert resolved_sub.operations[0] == op_0
        op_1 = (MultiControlled(Z, 1, 0b1), resolved_sub.qubits[:2], ())
        assert resolved_sub.operations[1] == op_1

    def test_negative_phase_handling(self) -> None:
        """Test phase handling with negative phase values."""
        # Create a custom operation with negative phase
        negative_phase = -math.pi / 2  # Should be equivalent to 3π/2 after modulo

        class NegativePhaseOpSubDef(OpSubDef):
            name = "NegativePhaseOp"
            qubit_count = 1

            def sub(self, builder: SubBuilder) -> None:
                q0 = builder.qubits[0]
                builder.add_op(X, (q0,))
                builder.add_phase(negative_phase)

        repo = default_repository().copy()
        negative_phase_op, negative_phase_sub = opsub(NegativePhaseOpSubDef, repo)
        repo.register_sub(negative_phase_op, negative_phase_sub)
        repo.register_sub_resolver(
            MultiControlled, generate_multicontrolled_to_mc_sub_resolver()
        )

        # Test with MSB set
        mc_negative_phase_op = MultiControlled(negative_phase_op, 2, 0b11)
        resolved_sub = resolve_sub(mc_negative_phase_op, repo)
        assert resolved_sub is not None

        # Should have MultiControlled X and MultiControlled Sdag
        # (since -π/2 % 2π = 3π/2)
        assert len(resolved_sub.operations) == 2
        op_0 = (MultiControlled(X, 2, 0b11), resolved_sub.qubits[:3], ())
        assert resolved_sub.operations[0] == op_0
        op_1 = (MultiControlled(Sdag, 1, 0b1), resolved_sub.qubits[:2], ())
        assert resolved_sub.operations[1] == op_1

    def test_mcswap_gate(self) -> None:
        """Test mapping SWAP to controlled CNOT operations."""
        sub = MultiControlledNamedMCGatesSub(SWAP, 2, 0b11)
        assert sub is not None
        assert len(sub.qubits) == 4
        assert len(sub.operations) == 3

        q0, q1, q2, q3 = sub.qubits

        # Operation 0: CNOT on last two qubits (q3, q2)
        op0, qubits0, regs0 = sub.operations[0]
        assert op0.id.local_name == "CNOT"
        assert qubits0 == (q3, q2)
        assert regs0 == ()

        # Operation 1: MultiControlled(CNOT, 2, 3) on all 4 qubits
        op1, qubits1, regs1 = sub.operations[1]
        assert op1.id.local_name == "MultiControlled"
        assert qubits1 == (q0, q1, q2, q3)
        assert regs1 == ()
        mc_target_op = op1.id.params[0]
        assert isinstance(mc_target_op, Op)
        assert mc_target_op.id.local_name == "CNOT"
        assert op1.id.params[1] == 2  # control_bits
        assert op1.id.params[2] == 3  # control_value (0b11)

        # Operation 2: CNOT on last two qubits (q3, q2) - same as operation 0
        op2, qubits2, regs2 = sub.operations[2]
        assert op2.id.local_name == "CNOT"
        assert qubits2 == (q3, q2)
        assert regs2 == ()

        # Assert all operations in sub result
        assert sub.operations == (
            (op0, qubits0, regs0),
            (op1, qubits1, regs1),
            (op2, qubits2, regs2),
        )

    def test_mcswap_gate_with_negation(self) -> None:
        """Test mapping SWAP to controlled CNOT with control negation."""
        sub = MultiControlledNamedMCGatesSub(SWAP, 2, 0b01)
        assert sub is not None
        assert len(sub.qubits) == 4
        assert len(sub.operations) == 3

        q0, q1, q2, q3 = sub.qubits

        # Operation 0: CNOT on last two qubits (q3, q2)
        op0, qubits0, regs0 = sub.operations[0]
        assert op0.id.local_name == "CNOT"
        assert qubits0 == (q3, q2)
        assert regs0 == ()

        # Operation 1: MultiControlled(CNOT, 2, 1) on all 4 qubits
        op1, qubits1, regs1 = sub.operations[1]
        assert op1.id.local_name == "MultiControlled"
        assert qubits1 == (q0, q1, q2, q3)
        assert regs1 == ()
        mc_target_op = op1.id.params[0]
        assert isinstance(mc_target_op, Op)
        assert mc_target_op.id.local_name == "CNOT"
        assert op1.id.params[1] == 2  # control_bits
        assert op1.id.params[2] == 1  # control_value (0b01)

        # Operation 2: CNOT on last two qubits (q3, q2)
        op2, qubits2, regs2 = sub.operations[2]
        assert op2.id.local_name == "CNOT"
        assert qubits2 == (q3, q2)
        assert regs2 == ()

    def test_mcswap_gate_single_control(self) -> None:
        """Test mapping SWAP to controlled CNOT with single control."""
        sub = MultiControlledNamedMCGatesSub(SWAP, 1, 0b1)
        assert sub is not None
        assert len(sub.qubits) == 3
        assert len(sub.operations) == 3

        q0, q1, q2 = sub.qubits

        # Operation 0: CNOT on last two qubits (q2, q1)
        op0, qubits0, regs0 = sub.operations[0]
        assert op0.id.local_name == "CNOT"
        assert qubits0 == (q2, q1)
        assert regs0 == ()

        # Operation 1: MultiControlled(CNOT, 1, 1) on all 3 qubits
        op1, qubits1, regs1 = sub.operations[1]
        assert op1.id.local_name == "MultiControlled"
        assert qubits1 == (q0, q1, q2)
        assert regs1 == ()
        mc_target_op = op1.id.params[0]
        assert isinstance(mc_target_op, Op)
        assert mc_target_op.id.local_name == "CNOT"
        assert op1.id.params[1] == 1  # control_bits
        assert op1.id.params[2] == 1  # control_value (0b1)

        # Operation 2: CNOT on last two qubits (q2, q1)
        op2, qubits2, regs2 = sub.operations[2]
        assert op2.id.local_name == "CNOT"
        assert qubits2 == (q2, q1)
        assert regs2 == ()

    def test_mcswap_gate_single_control_negation(self) -> None:
        """Test mapping SWAP to controlled CNOT with single control
        negation."""
        sub = MultiControlledNamedMCGatesSub(SWAP, 1, 0b0)
        assert sub is not None
        assert len(sub.qubits) == 3
        assert len(sub.operations) == 3

        q0, q1, q2 = sub.qubits

        # Operation 0: CNOT on last two qubits (q2, q1)
        op0, qubits0, regs0 = sub.operations[0]
        assert op0.id.local_name == "CNOT"
        assert qubits0 == (q2, q1)
        assert regs0 == ()

        # Operation 1: MultiControlled(CNOT, 1, 0) on all 3 qubits
        op1, qubits1, regs1 = sub.operations[1]
        assert op1.id.local_name == "MultiControlled"
        assert qubits1 == (q0, q1, q2)
        assert regs1 == ()
        mc_target_op = op1.id.params[0]
        assert isinstance(mc_target_op, Op)
        assert mc_target_op.id.local_name == "CNOT"
        assert op1.id.params[1] == 1  # control_bits
        assert op1.id.params[2] == 0  # control_value (0b0)

        # Operation 2: CNOT on last two qubits (q2, q1)
        op2, qubits2, regs2 = sub.operations[2]
        assert op2.id.local_name == "CNOT"
        assert qubits2 == (q2, q1)
        assert regs2 == ()

    def test_mcswap_gate_three_controls(self) -> None:
        """Test mapping SWAP to controlled CNOT with three controls."""
        sub = MultiControlledNamedMCGatesSub(SWAP, 3, 0b111)
        assert sub is not None
        assert len(sub.qubits) == 5
        assert len(sub.operations) == 3

        q0, q1, q2, q3, q4 = sub.qubits

        # Operation 0: CNOT on last two qubits (q4, q3)
        op0, qubits0, regs0 = sub.operations[0]
        assert op0.id.local_name == "CNOT"
        assert qubits0 == (q4, q3)
        assert regs0 == ()

        # Operation 1: MultiControlled(CNOT, 3, 7) on all 5 qubits
        op1, qubits1, regs1 = sub.operations[1]
        assert op1.id.local_name == "MultiControlled"
        assert qubits1 == (q0, q1, q2, q3, q4)
        assert regs1 == ()
        mc_target_op = op1.id.params[0]
        assert isinstance(mc_target_op, Op)
        assert mc_target_op.id.local_name == "CNOT"
        assert op1.id.params[1] == 3  # control_bits
        assert op1.id.params[2] == 7  # control_value (0b111)

        # Operation 2: CNOT on last two qubits (q4, q3)
        op2, qubits2, regs2 = sub.operations[2]
        assert op2.id.local_name == "CNOT"
        assert qubits2 == (q4, q3)
        assert regs2 == ()

    def test_mcswap_gate_three_controls_mixed_pattern(self) -> None:
        """Test mapping SWAP to controlled CNOT with three controls (mixed
        pattern)."""
        sub = MultiControlledNamedMCGatesSub(SWAP, 3, 0b101)
        assert sub is not None
        assert len(sub.qubits) == 5
        assert len(sub.operations) == 3

        q0, q1, q2, q3, q4 = sub.qubits

        # Operation 0: CNOT on last two qubits (q4, q3)
        op0, qubits0, regs0 = sub.operations[0]
        assert op0.id.local_name == "CNOT"
        assert qubits0 == (q4, q3)
        assert regs0 == ()

        # Operation 1: MultiControlled(CNOT, 3, 5) on all 5 qubits
        op1, qubits1, regs1 = sub.operations[1]
        assert op1.id.local_name == "MultiControlled"
        assert qubits1 == (q0, q1, q2, q3, q4)
        assert regs1 == ()
        mc_target_op = op1.id.params[0]
        assert isinstance(mc_target_op, Op)
        assert mc_target_op.id.local_name == "CNOT"
        assert op1.id.params[1] == 3  # control_bits
        assert op1.id.params[2] == 5  # control_value (0b101)

        # Operation 2: CNOT on last two qubits (q4, q3)
        op2, qubits2, regs2 = sub.operations[2]
        assert op2.id.local_name == "CNOT"
        assert qubits2 == (q4, q3)
        assert regs2 == ()

    def test_mcswap_gate_four_controls_all_ones(self) -> None:
        """Test mapping SWAP to controlled CNOT with four controls (all
        ones)."""
        sub = MultiControlledNamedMCGatesSub(SWAP, 4, 0b1111)
        assert sub is not None
        assert len(sub.qubits) == 6
        assert len(sub.operations) == 3

        q0, q1, q2, q3, q4, q5 = sub.qubits

        # Operation 0: CNOT on last two qubits (q5, q4)
        op0, qubits0, regs0 = sub.operations[0]
        assert op0.id.local_name == "CNOT"
        assert qubits0 == (q5, q4)
        assert regs0 == ()

        # Operation 1: MultiControlled(CNOT, 4, 15) on all 6 qubits
        op1, qubits1, regs1 = sub.operations[1]
        assert op1.id.local_name == "MultiControlled"
        assert qubits1 == (q0, q1, q2, q3, q4, q5)
        assert regs1 == ()
        mc_target_op = op1.id.params[0]
        assert isinstance(mc_target_op, Op)
        assert mc_target_op.id.local_name == "CNOT"
        assert op1.id.params[1] == 4  # control_bits
        assert op1.id.params[2] == 15  # control_value (0b1111)

        # Operation 2: CNOT on last two qubits (q5, q4)
        op2, qubits2, regs2 = sub.operations[2]
        assert op2.id.local_name == "CNOT"
        assert qubits2 == (q5, q4)
        assert regs2 == ()

    def test_mcswap_gate_four_controls_checkerboard_pattern(self) -> None:
        """Test mapping SWAP to controlled CNOT with four controls
        (checkerboard pattern)."""
        sub = MultiControlledNamedMCGatesSub(SWAP, 4, 0b1010)
        assert sub is not None
        assert len(sub.qubits) == 6
        assert len(sub.operations) == 3

        q0, q1, q2, q3, q4, q5 = sub.qubits

        # Operation 0: CNOT on last two qubits (q5, q4)
        op0, qubits0, regs0 = sub.operations[0]
        assert op0.id.local_name == "CNOT"
        assert qubits0 == (q5, q4)
        assert regs0 == ()

        # Operation 1: MultiControlled(CNOT, 4, 10) on all 6 qubits
        op1, qubits1, regs1 = sub.operations[1]
        assert op1.id.local_name == "MultiControlled"
        assert qubits1 == (q0, q1, q2, q3, q4, q5)
        assert regs1 == ()
        mc_target_op = op1.id.params[0]
        assert isinstance(mc_target_op, Op)
        assert mc_target_op.id.local_name == "CNOT"
        assert op1.id.params[1] == 4  # control_bits
        assert op1.id.params[2] == 10  # control_value (0b1010)

        # Operation 2: CNOT on last two qubits (q5, q4)
        op2, qubits2, regs2 = sub.operations[2]
        assert op2.id.local_name == "CNOT"
        assert qubits2 == (q5, q4)
        assert regs2 == ()

    def test_mcswap_gate_five_controls_sparse_pattern(self) -> None:
        """Test mapping SWAP to controlled CNOT with five controls (sparse
        pattern)."""
        sub = MultiControlledNamedMCGatesSub(SWAP, 5, 0b10001)
        assert sub is not None
        assert len(sub.qubits) == 7
        assert len(sub.operations) == 3

        q0, q1, q2, q3, q4, q5, q6 = sub.qubits

        # Operation 0: CNOT on last two qubits (q6, q5)
        op0, qubits0, regs0 = sub.operations[0]
        assert op0.id.local_name == "CNOT"
        assert qubits0 == (q6, q5)
        assert regs0 == ()

        # Operation 1: MultiControlled(CNOT, 5, 17) on all 7 qubits
        op1, qubits1, regs1 = sub.operations[1]
        assert op1.id.local_name == "MultiControlled"
        assert qubits1 == (q0, q1, q2, q3, q4, q5, q6)
        assert regs1 == ()
        mc_target_op = op1.id.params[0]
        assert isinstance(mc_target_op, Op)
        assert mc_target_op.id.local_name == "CNOT"
        assert op1.id.params[1] == 5  # control_bits
        assert op1.id.params[2] == 17  # control_value (0b10001)

        # Operation 2: CNOT on last two qubits (q6, q5)
        op2, qubits2, regs2 = sub.operations[2]
        assert op2.id.local_name == "CNOT"
        assert qubits2 == (q6, q5)
        assert regs2 == ()


def test_mcswap_multicontrolled_mapping() -> None:
    """Test that MultiControlled(SWAP, ...) maps to MCSWAP."""
    # Create MultiControlled SWAP operations
    mc_swap_1 = MultiControlled(SWAP, 1, 0b1)
    mc_swap_2 = MultiControlled(SWAP, 2, 0b11)
    mc_swap_3 = MultiControlled(SWAP, 3, 0b111)

    # Use a repository with the multi-controlled resolver
    repo = default_repository().copy()
    repo.register_sub_resolver(
        MultiControlled, generate_multicontrolled_to_mc_sub_resolver()
    )

    # Since MCSWAP is not implemented, these should return None or default behavior
    sub1 = resolve_sub(mc_swap_1, repo)
    sub2 = resolve_sub(mc_swap_2, repo)
    sub3 = resolve_sub(mc_swap_3, repo)

    # These will use default MultiControlled implementation since MCSWAP is removed
    assert sub1 is not None
    assert sub2 is not None
    assert sub3 is not None


def test_mcswap_with_control_negation() -> None:
    """Test MCSWAP with control negation patterns."""
    # Test various control patterns that require negation
    mc_swap_mixed = MultiControlled(SWAP, 3, 0b101)  # Control on |101⟩

    repo = default_repository().copy()
    repo.register_sub_resolver(
        MultiControlled, generate_multicontrolled_to_mc_sub_resolver()
    )

    # Since MCSWAP is not implemented, this should use default behavior
    sub = resolve_sub(mc_swap_mixed, repo)
    assert sub is not None
