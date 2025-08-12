from cmath import pi

from quri_parts.circuit.transpile import RZ2NamedTranspiler
from quri_parts.qsub.lib.std import RZ, H, S, Sdag, T, Tdag, X, Z
from quri_parts.qsub.sub import SubBuilder, SubDef, sub
from quri_parts.qsub.trans.qp_trans import SeparateQURIPartsTranspiler


class _RZTestSub(SubDef):
    qubit_count = 2

    def sub(self, builder: SubBuilder) -> None:
        q0, q1 = builder.qubits
        builder.add_op(RZ(-pi), (q1,))
        builder.add_op(RZ(-pi * 3.0 / 4.0), (q0,))
        builder.add_op(RZ(-pi / 2.0), (q1,))
        builder.add_op(RZ(-pi / 4.0), (q0,))
        builder.add_op(H, (q0,))
        builder.add_op(X, (q1,))
        builder.add_op(RZ(pi / 4.0), (q0,))
        builder.add_op(RZ(pi / 2.0), (q1,))
        builder.add_op(RZ(pi * 3.0 / 4.0), (q0,))
        builder.add_op(RZ(pi), (q1,))


RZTestSub = sub(_RZTestSub)


def test_separate_rz2named() -> None:
    sub = RZTestSub
    qp_trans = [RZ2NamedTranspiler()]
    trans = SeparateQURIPartsTranspiler(qp_trans)
    tsub = trans(sub)

    assert len(tsub.qubits) == 2
    assert len(tsub.registers) == 0
    assert len(tsub.aux_qubits) == 0
    assert len(tsub.aux_registers) == 0

    q0, q1 = sub.qubits
    assert tsub.operations == [
        (Z, (q1,), ()),
        (Z, (q0,), ()),
        (T, (q0,), ()),
        (Sdag, (q1,), ()),
        (Tdag, (q0,), ()),
        (H, (q0,), ()),
        (X, (q1,), ()),
        (T, (q0,), ()),
        (S, (q1,), ()),
        (S, (q0,), ()),
        (T, (q0,), ()),
        (Z, (q1,), ()),
    ]
