from cmath import pi

import pytest

from quri_parts.circuit.transpile import RZ2NamedTranspiler
from quri_parts.qsub.lib.std import (
    MCH,
    MCRX,
    MCRY,
    MCRZ,
    MCS,
    MCT,
    MCX,
    MCY,
    MCZ,
    RZ,
    H,
    MCPhase,
    MCSdag,
    MCSqrtX,
    MCSqrtXdag,
    MCSqrtY,
    MCSqrtYdag,
    MCTdag,
    S,
    Sdag,
    T,
    Tdag,
    X,
    Z,
)
from quri_parts.qsub.op import OpFactory
from quri_parts.qsub.qubit import Qubit
from quri_parts.qsub.sub import SubBuilder, SubDef, sub
from quri_parts.qsub.trans.qp_trans import (
    SeparateQURIPartsTranspiler,
    convert_from_qp,
    convert_to_qp,
)
from quri_parts.qsub.trans.transpiler import Operations


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


def _roundtrip(ops: Operations) -> Operations:
    """Convert ops -> qp circuit -> ops and return the result."""
    circ = convert_to_qp(ops)
    return convert_from_qp(circ)


@pytest.mark.parametrize(
    "mc_op",
    [
        MCX,
        MCY,
        MCZ,
        MCS,
        MCSdag,
        MCT,
        MCTdag,
        MCSqrtX,
        MCSqrtXdag,
        MCSqrtY,
        MCSqrtYdag,
        MCH,
    ],
)
def test_mc_op_roundtrip(mc_op: OpFactory[int]) -> None:
    q = tuple(Qubit(i) for i in range(3))
    ops = [(mc_op(2), q, ())]
    result = _roundtrip(ops)
    assert result == ops


@pytest.mark.parametrize("mc_param_op", [MCRX, MCRY, MCRZ, MCPhase])
def test_mc_param_op_roundtrip(mc_param_op: OpFactory[int, float]) -> None:
    q = tuple(Qubit(i) for i in range(3))
    ops = [(mc_param_op(2, 0.5), q, ())]
    result = _roundtrip(ops)
    assert result == ops


@pytest.mark.parametrize("num_controls", [1, 2, 3])
def test_mc_op_roundtrip_varying_controls(num_controls: int) -> None:
    q = tuple(Qubit(i) for i in range(num_controls + 1))
    ops = [(MCX(num_controls), q, ())]
    result = _roundtrip(ops)
    assert result == ops


@pytest.mark.parametrize("num_controls", [1, 2, 3])
def test_mc_param_op_roundtrip_varying_controls(num_controls: int) -> None:
    q = tuple(Qubit(i) for i in range(num_controls + 1))
    ops = [(MCRZ(num_controls, 0.7), q, ())]
    result = _roundtrip(ops)
    assert result == ops


def test_mc_ops_mixed_with_primitives() -> None:
    q0, q1, q2 = Qubit(0), Qubit(1), Qubit(2)
    ops = [
        (H, (q0,), ()),
        (MCX(2), (q0, q1, q2), ()),
        (MCRZ(2, 0.5), (q0, q1, q2), ()),
        (X, (q2,), ()),
    ]
    result = _roundtrip(ops)
    assert result == ops
