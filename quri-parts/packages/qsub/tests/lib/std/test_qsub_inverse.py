import numpy as np
import pytest

from quri_parts.qsub.lib.std import (
    CNOT,
    CZ,
    RX,
    RY,
    RZ,
    SWAP,
    Controlled,
    H,
    Inverse,
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
from quri_parts.qsub.lib.std.inverse import inverse_sub_resolver
from quri_parts.qsub.namespace import NameSpace
from quri_parts.qsub.op import Ident, Op, ParameterValidationError
from quri_parts.qsub.resolve import SubRepository, default_repository
from quri_parts.qsub.sub import SubBuilder


class TestInverseOp:
    def test_inverse_op(self) -> None:
        ydag = Inverse(Y)
        assert ydag.id.params == (Y,)
        assert ydag.qubit_count == 1
        assert ydag.reg_count == 0
        assert ydag.unitary

    def test_raise_inverse_non_unitary(self) -> None:
        with pytest.raises(ParameterValidationError):
            Inverse(M)


_repo = default_repository()


class TestInverseSub:
    def test_inverse_sub(self) -> None:
        F = Op(Ident(NameSpace("test"), "F"), 2, 1)
        builder = SubBuilder(2, 1)
        q0, q1 = builder.qubits
        (r,) = builder.registers
        builder.add_op(CNOT, (q0, q1))
        builder.add_op(S, (q1,))
        builder.add_op(CZ, (q1, q0))
        builder.add_op(M, (q1,), (r,))
        f_sub = builder.build()
        repository = SubRepository()
        repository.register_sub(F, f_sub)

        fdag = Inverse(F)
        sub = inverse_sub_resolver(fdag, repository)
        assert sub
        assert len(sub.qubits) == 2
        assert len(sub.aux_qubits) == 0
        assert len(sub.registers) == 1
        assert len(sub.aux_registers) == 0
        q0, q1 = sub.qubits
        (r,) = sub.registers
        assert tuple(sub.operations) == (
            (M, (q1,), (r,)),
            (CZ, (q1, q0), ()),
            (Inverse(S), (q1,), ()),
            (CNOT, (q0, q1), ()),
        )
        assert sub.phase == 0

    def test_inverse_sub_with_aux_qubits(self) -> None:
        F = Op(Ident(NameSpace("test"), "F"), 1)
        builder = SubBuilder(1)
        (q,) = builder.qubits
        a = builder.add_aux_qubit()
        builder.add_op(H, (q,))
        builder.add_op(S, (a,))
        f_sub = builder.build()
        repository = SubRepository()
        repository.register_sub(F, f_sub)

        fdag = Inverse(F)
        sub = inverse_sub_resolver(fdag, repository)
        assert sub
        assert len(sub.qubits) == 1
        assert len(sub.aux_qubits) == 1
        assert len(sub.registers) == 0
        assert len(sub.aux_registers) == 0
        (q,) = sub.qubits
        (a,) = sub.aux_qubits
        assert tuple(sub.operations) == (
            (Inverse(S), (a,), ()),
            (H, (q,), ()),
        )
        assert sub.phase == 0

    def test_inverse_sub_with_aux_registers(self) -> None:
        F = Op(Ident(NameSpace("test"), "F"), 1)
        builder = SubBuilder(1)
        (q,) = builder.qubits
        r = builder.add_aux_register()
        builder.add_op(H, (q,))
        builder.add_op(S, (q,))
        builder.add_op(M, (q,), (r,))
        f_sub = builder.build()
        repository = SubRepository()
        repository.register_sub(F, f_sub)

        fdag = Inverse(F)
        sub = inverse_sub_resolver(fdag, repository)
        assert sub
        assert len(sub.qubits) == 1
        assert len(sub.aux_qubits) == 0
        assert len(sub.registers) == 0
        assert len(sub.aux_registers) == 1
        (q,) = sub.qubits
        (r,) = sub.aux_registers
        assert tuple(sub.operations) == (
            (M, (q,), (r,)),
            (Inverse(S), (q,), ()),
            (H, (q,), ()),
        )
        assert sub.phase == 0

    def test_inverse_global_phase(self) -> None:
        F = Op(Ident(NameSpace("test"), "F"), 1, 0)
        builder = SubBuilder(1)
        qs = builder.qubits
        builder.add_op(T, qs)
        builder.add_phase(0.12)
        f_sub = builder.build()
        repository = SubRepository()
        repository.register_sub(F, f_sub)

        fdag = Inverse(F)
        sub = inverse_sub_resolver(fdag, repository)
        assert sub
        assert len(sub.qubits) == 1
        assert len(sub.aux_qubits) == 0
        assert len(sub.registers) == 0
        assert len(sub.aux_registers) == 0
        qs = sub.qubits
        assert tuple(sub.operations) == ((Inverse(T), qs, ()),)
        assert sub.phase == -0.12 % (2 * np.pi)


def test_inverse_controlled() -> None:
    ctdag = Inverse(Controlled(T))
    resolver = _repo.find_resolver(ctdag)
    assert resolver
    sub = resolver(ctdag, _repo)
    assert sub
    assert len(sub.qubits) == 2
    assert len(sub.aux_qubits) == 0
    assert len(sub.registers) == 0
    assert len(sub.aux_registers) == 0
    assert tuple(sub.operations) == ((Controlled(Inverse(T)), sub.qubits, ()),)
    assert sub.phase == 0


def test_inverse_multicontrolled() -> None:
    mctdag = Inverse(MultiControlled(T, 3, 0b010))
    resolver = _repo.find_resolver(mctdag)
    assert resolver
    sub = resolver(mctdag, _repo)
    assert sub
    assert len(sub.qubits) == 4
    assert len(sub.aux_qubits) == 0
    assert len(sub.registers) == 0
    assert len(sub.aux_registers) == 0
    assert tuple(sub.operations) == (
        (MultiControlled(Inverse(T), 3, 0b010), sub.qubits, ()),
    )
    assert sub.phase == 0


def test_inverse_inverse() -> None:
    invinv = Inverse(Inverse(T))
    resolver = _repo.find_resolver(invinv)
    assert resolver is not None
    sub = resolver(invinv, _repo)
    assert sub is not None
    assert tuple(sub.operations) == ((T, sub.qubits, ()),)


def test_inverse_op() -> None:
    target_expect_list = [
        (X, X),
        (Y, Y),
        (Z, Z),
        (H, H),
        (RX(0.25), RX(-0.25)),
        (RY(0.25), RY(-0.25)),
        (RZ(0.25), RZ(-0.25)),
        (Phase(0.25), Phase(-0.25)),
        (SqrtX, SqrtXdag),
        (SqrtXdag, SqrtX),
        (SqrtY, SqrtYdag),
        (SqrtYdag, SqrtY),
        (S, Sdag),
        (Sdag, S),
        (T, Tdag),
        (Tdag, T),
        (CNOT, CNOT),
        (CZ, CZ),
        (SWAP, SWAP),
        (Toffoli, Toffoli),
    ]

    for target, expect in target_expect_list:
        target_dag = Inverse(target)
        resolver = _repo.find_resolver(target_dag)
        assert resolver
        sub = resolver(target_dag, _repo)
        assert sub
        assert len(sub.qubits) == target.qubit_count
        assert len(sub.aux_qubits) == 0
        assert len(sub.registers) == 0
        assert len(sub.aux_registers) == 0
        assert tuple(sub.operations) == ((expect, sub.qubits, ()),)
        assert sub.phase == 0
