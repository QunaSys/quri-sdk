import math

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
from quri_parts.qsub.lib.std.control import (
    controlled_sub_resolver,
    register_controlled_resolver,
)
from quri_parts.qsub.namespace import NameSpace
from quri_parts.qsub.op import Ident, Op, ParameterValidationError
from quri_parts.qsub.opsub import OpSubDef, opsub
from quri_parts.qsub.register import DEFAULT_QNAME, QRegSpec
from quri_parts.qsub.resolve import (
    SimpleSubRepository,
    SubRepository,
    default_repository,
    resolve_sub,
)
from quri_parts.qsub.sub import Sub, SubBuilder


class TestControlledOp:
    def test_controlled_op(self) -> None:
        cy = Controlled(Y)
        assert cy.id.params == (Y,)
        assert cy.qubit_count == 2
        assert cy.reg_count == 0
        assert cy.unitary

    def test_raise_controlled_non_unitary(self) -> None:
        with pytest.raises(ParameterValidationError):
            Controlled(M)


class TestControlledQRegs:
    def test_controlled_single_level_qregs(self) -> None:
        """Controlled(X) should have qregs: (ctrl, qs)"""
        cx = Controlled(X)
        assert len(cx.qregs) == 2
        assert cx.qregs[0] == QRegSpec("ctrl", 1)
        assert cx.qregs[1] == QRegSpec("qs", 1)

    def test_controlled_nested_two_level_qregs(self) -> None:
        """Controlled(Controlled(X)) should have qregs: (ctrl_1, ctrl, qs)"""
        ccx = Controlled(Controlled(X))
        assert len(ccx.qregs) == 3
        assert ccx.qregs[0] == QRegSpec("ctrl_1", 1)
        assert ccx.qregs[1] == QRegSpec("ctrl", 1)
        assert ccx.qregs[2] == QRegSpec("qs", 1)

    def test_controlled_nested_three_level_qregs(self) -> None:
        """Controlled(Controlled(Controlled(X))) should have qregs: (ctrl_2,
        ctrl_1, ctrl, qs)"""
        cccx = Controlled(Controlled(Controlled(X)))
        assert len(cccx.qregs) == 4
        assert cccx.qregs[0] == QRegSpec("ctrl_2", 1)
        assert cccx.qregs[1] == QRegSpec("ctrl_1", 1)
        assert cccx.qregs[2] == QRegSpec("ctrl", 1)
        assert cccx.qregs[3] == QRegSpec("qs", 1)

    def test_controlled_single_qubit_gate_qregs(self) -> None:
        """All single-qubit Cliffords should have controlled qregs: (ctrl,
        qs)"""
        for op in [
            X,
            Y,
            Z,
            H,
            S,
            Sdag,
            SqrtX,
            SqrtXdag,
            SqrtY,
            SqrtYdag,
            RX(0.5),
            RY(0.5),
            RZ(0.5),
            Phase(0.5),
            T,
            Tdag,
        ]:
            c_op = Controlled(op)
            assert c_op.qregs == (QRegSpec("ctrl", 1), QRegSpec(DEFAULT_QNAME, 1))

    def test_controlled_cnot_qregs(self) -> None:
        """Controlled(CNOT) should have qregs: (ctrl_1, ctrl, target)"""
        c_cnot = Controlled(CNOT)
        assert c_cnot.qregs == (
            QRegSpec("ctrl_1", 1),
            QRegSpec("ctrl", 1),
            QRegSpec("target", 1),
        )

    def test_controlled_cz_qregs(self) -> None:
        """Controlled(CZ) should have qregs: (ctrl_1, ctrl, target)"""
        c_cz = Controlled(CZ)
        assert c_cz.qregs == (
            QRegSpec("ctrl_1", 1),
            QRegSpec("ctrl", 1),
            QRegSpec("target", 1),
        )

    def test_controlled_swap_qregs(self) -> None:
        """Controlled(SWAP) should have qregs: (ctrl, qs)"""
        c_swap = Controlled(SWAP)
        assert c_swap.qregs == (
            QRegSpec("ctrl", 1),
            QRegSpec("qs", 2),
        )

    def test_controlled_toffoli_qregs(self) -> None:
        """Controlled(Toffoli) should have qregs: (ctrl_1, ctrl, target)"""
        c_toff = Controlled(Toffoli)
        assert c_toff.qregs == (
            QRegSpec("ctrl_1", 1),
            QRegSpec("ctrl", 2),
            QRegSpec("target", 1),
        )


_repo = default_repository()


class TestControlledSub:
    def test_no_sub(self) -> None:
        ccz = Controlled(CZ)
        sub = controlled_sub_resolver(ccz, SimpleSubRepository())
        assert sub is None

    def test_expand_sub(self) -> None:
        F = Op.from_qubit_count(Ident(NameSpace("test"), "F"), 2, 1)
        builder = SubBuilder(2, 1)
        q0, q1 = builder.qubits
        (r,) = builder.registers
        builder.add_op(CNOT, (q0, q1))
        builder.add_op(Y, (q1,))
        builder.add_op(CZ, (q1, q0))
        builder.add_op(M, (q1,), (r,))
        f_sub = builder.build()
        repository = SimpleSubRepository()
        repository.register_sub(F, f_sub)

        cf = Controlled(F)
        sub = controlled_sub_resolver(cf, repository)
        assert sub
        assert len(sub.qubits) == 3
        assert len(sub.aux_qubits) == 0
        assert len(sub.registers) == 1
        assert len(sub.aux_registers) == 0
        q0, q1, q2 = sub.qubits
        (r,) = sub.registers
        assert tuple(sub.operations) == (
            (Controlled(CNOT), (q0, q1, q2), ()),
            (Controlled(Y), (q0, q2), ()),
            (Controlled(CZ), (q0, q2, q1), ()),
            (M, (q2,), (r,)),
        )
        assert sub.phase == 0

    def test_expand_sub_with_aux_qubits(self) -> None:
        F = Op.from_qubit_count(Ident(NameSpace("test"), "F"), 1)
        builder = SubBuilder(1)
        (q,) = builder.qubits
        a = builder.add_aux_qubit()
        builder.add_op(H, (q,))
        builder.add_op(Y, (a,))
        f_sub = builder.build()
        repository = SimpleSubRepository()
        repository.register_sub(F, f_sub)

        cf = Controlled(F)
        sub = controlled_sub_resolver(cf, repository)
        assert sub
        assert len(sub.qubits) == 2
        assert len(sub.aux_qubits) == 1
        assert len(sub.registers) == 0
        assert len(sub.aux_registers) == 0
        q0, q1 = sub.qubits
        (a,) = sub.aux_qubits
        assert tuple(sub.operations) == (
            (Controlled(H), (q0, q1), ()),
            (Controlled(Y), (q0, a), ()),
        )
        assert sub.phase == 0

    def test_expand_sub_with_aux_register(self) -> None:
        F = Op.from_qubit_count(Ident(NameSpace("test"), "F"), 1)
        builder = SubBuilder(1)
        (q,) = builder.qubits
        builder.add_aux_qreg("aaa", 1)
        builder.add_op(H, (q,))
        builder.add_op(Y, builder.aux_qregs["aaa"][:])
        f_sub = builder.build()
        repository = SimpleSubRepository()
        repository.register_sub(F, f_sub)

        cf = Controlled(F)
        sub = controlled_sub_resolver(cf, repository)
        assert sub
        assert sub.aux_qregs is not None
        assert len(sub.qubits) == 2
        assert len(sub.aux_qubits) == 1
        assert len(sub.registers) == 0
        assert len(sub.aux_registers) == 0
        assert len(sub.aux_qregs) == 1
        assert next(iter(sub.aux_qregs)) == "aaa"
        q0, q1 = sub.qubits
        (a,) = sub.aux_qregs["aaa"][:]
        assert tuple(sub.operations) == (
            (Controlled(H), (q0, q1), ()),
            (Controlled(Y), (q0, a), ()),
        )
        assert sub.phase == 0

    def test_expand_sub_with_aux_registers(self) -> None:
        F = Op.from_qubit_count(Ident(NameSpace("test"), "F"), 1)
        builder = SubBuilder(1)
        (q,) = builder.qubits
        r = builder.add_aux_register()
        builder.add_op(H, (q,))
        builder.add_op(M, (q,), (r,))
        f_sub = builder.build()
        repository = SimpleSubRepository()
        repository.register_sub(F, f_sub)

        cf = Controlled(F)
        sub = controlled_sub_resolver(cf, repository)
        assert sub
        assert len(sub.qubits) == 2
        assert len(sub.aux_qubits) == 0
        assert len(sub.registers) == 0
        assert len(sub.aux_registers) == 1
        q0, q1 = sub.qubits
        (r,) = sub.aux_registers
        assert tuple(sub.operations) == (
            (Controlled(H), (q0, q1), ()),
            (M, (q1,), (r,)),
        )
        assert sub.phase == 0

    def test_expand_sub_with_phase(self) -> None:
        F = Op.from_qubit_count(Ident(NameSpace("test"), "F"), 1)
        builder = SubBuilder(1)
        (q,) = builder.qubits
        builder.add_op(H, (q,))
        builder.add_phase(math.pi / 4)
        f_sub = builder.build()
        repository = SimpleSubRepository()
        repository.register_sub(F, f_sub)

        cf = Controlled(F)
        sub = controlled_sub_resolver(cf, repository)
        assert sub
        assert len(sub.qubits) == 2
        assert len(sub.aux_qubits) == 0
        assert len(sub.registers) == 0
        assert len(sub.aux_registers) == 0
        q0, q1 = sub.qubits
        assert tuple(sub.operations) == (
            (Controlled(H), (q0, q1), ()),
            (Phase(math.pi / 4), (q0,), ()),
        )
        assert sub.phase == 0

    def test_expand_sub_with_pi_phase(self) -> None:
        F = Op.from_qubit_count(Ident(NameSpace("test"), "F"), 1)
        builder = SubBuilder(1)
        builder.add_op(H, builder.qubits)
        builder.add_phase(math.pi)
        f_sub = builder.build()
        repository = SimpleSubRepository()
        repository.register_sub(F, f_sub)

        cf = Controlled(F)
        sub = controlled_sub_resolver(cf, repository)
        assert sub
        assert len(sub.qubits) == 2
        assert len(sub.aux_qubits) == 0
        assert len(sub.registers) == 0
        assert len(sub.aux_registers) == 0
        q0, q1 = sub.qubits
        assert tuple(sub.operations) == (
            (Controlled(H), (q0, q1), ()),
            (Z, (q0,), ()),
        )
        assert sub.phase == 0

    def test_expand_sub_with_half_pi_phase(self) -> None:
        F = Op.from_qubit_count(Ident(NameSpace("test"), "F"), 1)
        builder = SubBuilder(1)
        builder.add_op(H, builder.qubits)
        builder.add_phase(math.pi / 2)
        f_sub = builder.build()
        repository = SimpleSubRepository()
        repository.register_sub(F, f_sub)

        cf = Controlled(F)
        sub = controlled_sub_resolver(cf, repository)
        assert sub
        assert len(sub.qubits) == 2
        assert len(sub.aux_qubits) == 0
        assert len(sub.registers) == 0
        assert len(sub.aux_registers) == 0
        q0, q1 = sub.qubits
        assert tuple(sub.operations) == (
            (Controlled(H), (q0, q1), ()),
            (S, (q0,), ()),
        )
        assert sub.phase == 0

    def test_expand_sub_with_minus_half_pi_phase(self) -> None:
        F = Op.from_qubit_count(Ident(NameSpace("test"), "F"), 1)
        builder = SubBuilder(1)
        builder.add_op(H, builder.qubits)
        builder.add_phase(-math.pi / 2)
        f_sub = builder.build()
        repository = SimpleSubRepository()
        repository.register_sub(F, f_sub)

        cf = Controlled(F)
        sub = controlled_sub_resolver(cf, repository)
        assert sub
        assert len(sub.qubits) == 2
        assert len(sub.aux_qubits) == 0
        assert len(sub.registers) == 0
        assert len(sub.aux_registers) == 0
        q0, q1 = sub.qubits
        assert tuple(sub.operations) == (
            (Controlled(H), (q0, q1), ()),
            (Sdag, (q0,), ()),
        )
        assert sub.phase == 0


def test_controlled_x() -> None:
    cx = Controlled(X)
    resolver = _repo.find_resolver(cx)
    assert resolver
    sub = resolver(cx, _repo)
    assert sub
    assert len(sub.qubits) == 2
    assert len(sub.aux_qubits) == 0
    assert len(sub.registers) == 0
    assert len(sub.aux_registers) == 0
    q0, q1 = sub.qubits
    assert tuple(sub.operations) == ((CNOT, (q0, q1), ()),)
    assert sub.phase == 0


def test_controlled_y() -> None:
    cy = Controlled(Y)
    resolver = _repo.find_resolver(cy)
    assert resolver
    sub = resolver(cy, _repo)
    assert sub
    assert len(sub.qubits) == 2
    assert len(sub.aux_qubits) == 0
    assert len(sub.registers) == 0
    assert len(sub.aux_registers) == 0
    q0, q1 = sub.qubits
    assert tuple(sub.operations) == (
        (Sdag, (q1,), ()),
        (CNOT, (q0, q1), ()),
        (S, (q1,), ()),
    )
    assert sub.phase == 0


def test_controlled_z() -> None:
    cz = Controlled(Z)
    resolver = _repo.find_resolver(cz)
    assert resolver
    sub = resolver(cz, _repo)
    assert sub
    assert len(sub.qubits) == 2
    assert len(sub.aux_qubits) == 0
    assert len(sub.registers) == 0
    assert len(sub.aux_registers) == 0
    q0, q1 = sub.qubits
    assert tuple(sub.operations) == ((CZ, (q0, q1), ()),)
    assert sub.phase == 0


def test_controlled_h() -> None:
    ch = Controlled(H)
    resolver = _repo.find_resolver(ch)
    assert resolver
    sub = resolver(ch, _repo)
    assert sub
    assert len(sub.qubits) == 2
    assert len(sub.aux_qubits) == 0
    assert len(sub.registers) == 0
    assert len(sub.aux_registers) == 0
    q0, q1 = sub.qubits
    assert tuple(sub.operations) == (
        (RY(-math.pi / 4), (q1,), ()),
        (CZ, (q0, q1), ()),
        (RY(math.pi / 4), (q1,), ()),
    )
    assert sub.phase == 0


def test_controlled_rx() -> None:
    crx = Controlled(RX(0.25))
    resolver = _repo.find_resolver(crx)
    assert resolver
    sub = resolver(crx, _repo)
    assert sub
    assert len(sub.qubits) == 2
    assert len(sub.aux_qubits) == 0
    assert len(sub.registers) == 0
    assert len(sub.aux_registers) == 0
    q0, q1 = sub.qubits
    assert tuple(sub.operations) == (
        (H, (q1,), ()),
        (CNOT, (q0, q1), ()),
        (RZ(-0.125), (q1,), ()),
        (CNOT, (q0, q1), ()),
        (RZ(0.125), (q1,), ()),
        (H, (q1,), ()),
    )
    assert sub.phase == 0


def test_controlled_ry() -> None:
    cry = Controlled(RY(0.25))
    resolver = _repo.find_resolver(cry)
    assert resolver
    sub = resolver(cry, _repo)
    assert sub
    assert len(sub.qubits) == 2
    assert len(sub.aux_qubits) == 0
    assert len(sub.registers) == 0
    assert len(sub.aux_registers) == 0
    q0, q1 = sub.qubits
    assert tuple(sub.operations) == (
        (CNOT, (q0, q1), ()),
        (RY(-0.125), (q1,), ()),
        (CNOT, (q0, q1), ()),
        (RY(0.125), (q1,), ()),
    )
    assert sub.phase == 0


def test_controlled_phase() -> None:
    cphase = Controlled(Phase(0.25))
    resolver = _repo.find_resolver(cphase)
    assert resolver
    sub = resolver(cphase, _repo)
    assert sub
    assert len(sub.qubits) == 2
    assert len(sub.aux_qubits) == 0
    assert len(sub.registers) == 0
    assert len(sub.aux_registers) == 0
    q0, q1 = sub.qubits
    assert tuple(sub.operations) == (
        (CNOT, (q0, q1), ()),
        (RZ(-0.125), (q1,), ()),
        (CNOT, (q0, q1), ()),
        (RZ(0.125), (q1,), ()),
        (Phase(0.125), (q0,), ()),
    )
    assert sub.phase == 0


def test_controlled_rz() -> None:
    crz = Controlled(RZ(0.25))
    resolver = _repo.find_resolver(crz)
    assert resolver
    sub = resolver(crz, _repo)
    assert sub
    assert len(sub.qubits) == 2
    assert len(sub.aux_qubits) == 0
    assert len(sub.registers) == 0
    assert len(sub.aux_registers) == 0
    q0, q1 = sub.qubits
    assert tuple(sub.operations) == (
        (CNOT, (q0, q1), ()),
        (RZ(-0.125), (q1,), ()),
        (CNOT, (q0, q1), ()),
        (RZ(0.125), (q1,), ()),
    )
    assert sub.phase == 0


def test_controlled_sqrtx() -> None:
    csqrtx = Controlled(SqrtX)
    resolver = _repo.find_resolver(csqrtx)
    assert resolver
    sub = resolver(csqrtx, _repo)
    assert sub
    assert len(sub.qubits) == 2
    assert len(sub.aux_qubits) == 0
    assert len(sub.registers) == 0
    assert len(sub.aux_registers) == 0
    q0, q1 = sub.qubits
    assert tuple(sub.operations) == (
        (H, (q1,), ()),
        (CNOT, (q0, q1), ()),
        (RZ(-math.pi / 4), (q1,), ()),
        (CNOT, (q0, q1), ()),
        (RZ(math.pi / 4), (q1,), ()),
        (H, (q1,), ()),
        (Phase(math.pi / 4), (q0,), ()),
    )
    assert sub.phase == 0


def test_controlled_sqrtxdag() -> None:
    csqrtxd = Controlled(SqrtXdag)
    resolver = _repo.find_resolver(csqrtxd)
    assert resolver
    sub = resolver(csqrtxd, _repo)
    assert sub
    assert len(sub.qubits) == 2
    assert len(sub.aux_qubits) == 0
    assert len(sub.registers) == 0
    assert len(sub.aux_registers) == 0
    q0, q1 = sub.qubits
    assert tuple(sub.operations) == (
        (H, (q1,), ()),
        (CNOT, (q0, q1), ()),
        (RZ(math.pi / 4), (q1,), ()),
        (CNOT, (q0, q1), ()),
        (RZ(-math.pi / 4), (q1,), ()),
        (H, (q1,), ()),
        (Phase(-math.pi / 4), (q0,), ()),
    )
    assert sub.phase == 0


def test_controlled_sqrty() -> None:
    csqrty = Controlled(SqrtY)
    resolver = _repo.find_resolver(csqrty)
    assert resolver
    sub = resolver(csqrty, _repo)
    assert sub
    assert len(sub.qubits) == 2
    assert len(sub.aux_qubits) == 0
    assert len(sub.registers) == 0
    assert len(sub.aux_registers) == 0
    q0, q1 = sub.qubits
    assert tuple(sub.operations) == (
        (CNOT, (q0, q1), ()),
        (RY(-math.pi / 4), (q1,), ()),
        (CNOT, (q0, q1), ()),
        (RY(math.pi / 4), (q1,), ()),
        (Phase(math.pi / 4), (q0,), ()),
    )
    assert sub.phase == 0


def test_controlled_sqrtydag() -> None:
    csqrtyd = Controlled(SqrtYdag)
    resolver = _repo.find_resolver(csqrtyd)
    assert resolver
    sub = resolver(csqrtyd, _repo)
    assert sub
    assert len(sub.qubits) == 2
    assert len(sub.aux_qubits) == 0
    assert len(sub.registers) == 0
    assert len(sub.aux_registers) == 0
    q0, q1 = sub.qubits
    assert tuple(sub.operations) == (
        (CNOT, (q0, q1), ()),
        (RY(math.pi / 4), (q1,), ()),
        (CNOT, (q0, q1), ()),
        (RY(-math.pi / 4), (q1,), ()),
        (Phase(-math.pi / 4), (q0,), ()),
    )
    assert sub.phase == 0


def test_controlled_s() -> None:
    cs = Controlled(S)
    resolver = _repo.find_resolver(cs)
    assert resolver
    sub = resolver(cs, _repo)
    assert sub
    assert len(sub.qubits) == 2
    assert len(sub.aux_qubits) == 0
    assert len(sub.registers) == 0
    assert len(sub.aux_registers) == 0
    q0, q1 = sub.qubits
    assert tuple(sub.operations) == (
        (CNOT, (q0, q1), ()),
        (Tdag, (q1,), ()),
        (CNOT, (q0, q1), ()),
        (T, (q0,), ()),
        (T, (q1,), ()),
    )
    assert sub.phase == 0


def test_controlled_sdag() -> None:
    csd = Controlled(Sdag)
    resolver = _repo.find_resolver(csd)
    assert resolver
    sub = resolver(csd, _repo)
    assert sub
    assert len(sub.qubits) == 2
    assert len(sub.aux_qubits) == 0
    assert len(sub.registers) == 0
    assert len(sub.aux_registers) == 0
    q0, q1 = sub.qubits
    assert tuple(sub.operations) == (
        (CNOT, (q0, q1), ()),
        (T, (q1,), ()),
        (CNOT, (q0, q1), ()),
        (Tdag, (q0,), ()),
        (Tdag, (q1,), ()),
    )
    assert sub.phase == 0


def test_controlled_t() -> None:
    ct = Controlled(T)
    resolver = _repo.find_resolver(ct)
    assert resolver
    sub = resolver(ct, _repo)
    assert sub
    assert len(sub.qubits) == 2
    assert len(sub.aux_qubits) == 0
    assert len(sub.registers) == 0
    assert len(sub.aux_registers) == 0
    q0, q1 = sub.qubits
    assert tuple(sub.operations) == (
        (CNOT, (q0, q1), ()),
        (Phase(-math.pi / 8), (q1,), ()),
        (CNOT, (q0, q1), ()),
        (Phase(math.pi / 8), (q0,), ()),
        (Phase(math.pi / 8), (q1,), ()),
    )
    assert sub.phase == 0


def test_controlled_tdag() -> None:
    ctd = Controlled(Tdag)
    resolver = _repo.find_resolver(ctd)
    assert resolver
    sub = resolver(ctd, _repo)
    assert sub
    assert len(sub.qubits) == 2
    assert len(sub.aux_qubits) == 0
    assert len(sub.registers) == 0
    assert len(sub.aux_registers) == 0
    q0, q1 = sub.qubits
    assert tuple(sub.operations) == (
        (CNOT, (q0, q1), ()),
        (Phase(math.pi / 8), (q1,), ()),
        (CNOT, (q0, q1), ()),
        (Phase(-math.pi / 8), (q0,), ()),
        (Phase(-math.pi / 8), (q1,), ()),
    )
    assert sub.phase == 0


def test_controlled_cnot() -> None:
    ccnot = Controlled(CNOT)
    resolver = _repo.find_resolver(ccnot)
    assert resolver
    sub = resolver(ccnot, _repo)
    assert sub
    assert len(sub.qubits) == 3
    assert len(sub.aux_qubits) == 0
    assert len(sub.registers) == 0
    assert len(sub.aux_registers) == 0
    q0, q1, q2 = sub.qubits
    assert tuple(sub.operations) == ((Toffoli, (q0, q1, q2), ()),)
    assert sub.phase == 0


def test_controlled_cz() -> None:
    ccz = Controlled(CZ)
    resolver = _repo.find_resolver(ccz)
    assert resolver
    sub = resolver(ccz, _repo)
    assert sub
    assert len(sub.qubits) == 3
    assert len(sub.aux_qubits) == 0
    assert len(sub.registers) == 0
    assert len(sub.aux_registers) == 0
    q0, q1, q2 = sub.qubits
    assert tuple(sub.operations) == ((MultiControlled(Z, 2, 0b11), (q0, q1, q2), ()),)
    assert sub.phase == 0


def test_controlled_swap() -> None:
    cswap = Controlled(SWAP)
    resolver = _repo.find_resolver(cswap)
    assert resolver
    sub = resolver(cswap, _repo)
    assert sub
    assert len(sub.qubits) == 3
    assert len(sub.aux_qubits) == 0
    assert len(sub.registers) == 0
    assert len(sub.aux_registers) == 0
    q0, q1, q2 = sub.qubits
    assert tuple(sub.operations) == (
        (CNOT, (q2, q1), ()),
        (Toffoli, (q0, q1, q2), ()),
        (CNOT, (q2, q1), ()),
    )
    assert sub.phase == 0


def test_controlled_toffoli() -> None:
    ctoffoli = Controlled(Toffoli)
    resolver = _repo.find_resolver(ctoffoli)
    assert resolver
    sub = resolver(ctoffoli, _repo)
    assert sub
    assert len(sub.qubits) == 4
    assert len(sub.aux_qubits) == 0
    assert len(sub.registers) == 0
    assert len(sub.aux_registers) == 0
    q0, q1, q2, q3 = sub.qubits
    assert tuple(sub.operations) == (
        (MultiControlled(X, 3, 0b111), (q0, q1, q2, q3), ()),
    )
    assert sub.phase == 0


def test_controlled_multicontrolled() -> None:
    cccy = MultiControlled(Y, 3, 0b010)
    ccccy = Controlled(cccy)
    resolver = _repo.find_resolver(ccccy)
    assert resolver
    sub = resolver(ccccy, _repo)
    assert sub
    assert len(sub.qubits) == 5
    assert len(sub.aux_qubits) == 0
    assert len(sub.registers) == 0
    assert len(sub.aux_registers) == 0
    q0, q1, q2, q3, q4 = sub.qubits
    assert tuple(sub.operations) == (
        (MultiControlled(Y, 4, 0b0101), (q0, q1, q2, q3, q4), ()),
    )
    assert sub.phase == 0


def test_inverse_optimized_controlled() -> None:
    class _HCXH(OpSubDef):
        name = "HCXH"
        qubit_count = 2
        qregs = (QRegSpec("qs", 2),)

        def sub(self, builder: SubBuilder) -> None:
            aux_q = builder.add_aux_qubit()
            qs = builder.qubits
            builder.add_op(H, (qs[0],))
            builder.add_op(H, (qs[1],))
            builder.add_op(CNOT, (qs[1], aux_q))
            builder.add_op(H, (qs[1],))
            builder.add_op(H, (qs[0],))

    HCXH, _ = opsub(_HCXH)

    def controlled_h_cx_h_resolver(op: Op, repository: SubRepository) -> Sub:
        target_op = op.id.params[0]
        assert isinstance(target_op, Op)
        builder = SubBuilder(op.qubit_count, op.reg_count)
        a = builder.add_aux_qubit()
        qs = builder.qubits
        builder.add_op(Controlled(H), (qs[0], qs[1]))
        builder.add_op(Controlled(H), (qs[0], qs[2]))
        builder.add_op(Controlled(S), (qs[0], qs[1]))
        builder.add_op(Controlled(S), (qs[0], qs[2]))
        builder.add_op(CNOT, (qs[2], a))
        builder.add_op(Controlled(H), (qs[0], qs[2]))
        builder.add_op(Controlled(H), (qs[0], qs[1]))
        return builder.build()

    test_sub_repository = SimpleSubRepository()
    register_controlled_resolver(test_sub_repository, controlled_h_cx_h_resolver, HCXH)

    inv_op = Controlled(Inverse(HCXH))
    ctrl_inv_sub = resolve_sub(inv_op, test_sub_repository)
    assert ctrl_inv_sub is not None
    qs = ctrl_inv_sub.qubits
    aux_qs = ctrl_inv_sub.aux_qubits
    expected_resolved_sub = (
        (Inverse(Controlled(H)), (qs[0], qs[1]), ()),
        (Inverse(Controlled(H)), (qs[0], qs[2]), ()),
        (CNOT, (qs[2], aux_qs[0]), ()),
        (Inverse(Controlled(S)), (qs[0], qs[2]), ()),
        (Inverse(Controlled(S)), (qs[0], qs[1]), ()),
        (Inverse(Controlled(H)), (qs[0], qs[2]), ()),
        (Inverse(Controlled(H)), (qs[0], qs[1]), ()),
    )
    assert tuple(ctrl_inv_sub.operations) == expected_resolved_sub


def test_inverse_optimized_controlled_qregs() -> None:
    class _HCXH(OpSubDef):
        name = "HCXH"
        qubit_count = 2
        qregs = (QRegSpec("my_reg", 2),)

        def sub(self, builder: SubBuilder) -> None:
            builder.add_op(H, (builder.qubits[0],))
            builder.add_op(H, (builder.qubits[1],))
            builder.add_op(CNOT, (builder.qubits[0], builder.qubits[1]))
            builder.add_op(H, (builder.qubits[0],))
            builder.add_op(H, (builder.qubits[1],))

    HCXH, _ = opsub(_HCXH)

    def controlled_h_h_resolver(op: Op, repository: SubRepository) -> Sub:
        builder = SubBuilder.from_qregs(op.qregs, op.reg_count)
        qs = builder.qubits
        builder.add_op(H, (qs[1],))
        builder.add_op(H, (qs[2],))
        builder.add_op(Controlled(CNOT), (qs[0], qs[1], qs[2]))
        builder.add_op(H, (qs[2],))
        builder.add_op(H, (qs[1],))
        return builder.build()

    test_sub_repository = SimpleSubRepository()
    register_controlled_resolver(test_sub_repository, controlled_h_h_resolver, HCXH)

    inv_op = Controlled(Inverse(HCXH))
    assert inv_op.qregs == (QRegSpec("ctrl", 1), QRegSpec("my_reg", 2))

    ctrl_inv_sub = resolve_sub(inv_op, test_sub_repository)
    assert ctrl_inv_sub is not None
    assert ctrl_inv_sub.qregs is not None
    assert "ctrl" in ctrl_inv_sub.qregs
    assert "my_reg" in ctrl_inv_sub.qregs
    assert ctrl_inv_sub.qregs["my_reg"].name == "my_reg"
