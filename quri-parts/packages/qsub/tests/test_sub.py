import pytest

from quri_parts.qsub.namespace import DEFAULT
from quri_parts.qsub.op import OpDef, op
from quri_parts.qsub.qubit import Qubit
from quri_parts.qsub.register import QRegSpec
from quri_parts.qsub.sub import ParamSubDef, Sub, SubBuilder, SubDef, param_sub, sub


class _Toffoli(OpDef):
    name = "Toffoli"
    qubit_count = 3


Toffoli = op(_Toffoli)


class _X(OpDef):
    name = "X"
    qubit_count = 1


X = op(_X)


def test_sub_init() -> None:
    builder = SubBuilder(7)
    assert len(builder.qubits) == 7

    with pytest.raises(ValueError):
        SubBuilder(-1)


def test_sub_add_aux_qubit_and_register() -> None:
    builder = SubBuilder(5, 4)

    aux_qubits = tuple(builder.add_aux_qubit() for _ in range(3))
    aux_registers = tuple(builder.add_aux_register() for _ in range(2))

    sub = builder.build()
    assert sub.aux_qubits == aux_qubits
    assert sub.aux_registers == aux_registers
    assert len(sub.qubits) == 5
    assert len(sub.aux_qubits) == 3
    assert len(sub.registers) == 4
    assert len(sub.aux_registers) == 2


def test_sub_add_aux_qubits_and_registers() -> None:
    builder = SubBuilder(2, 3)

    aux_qubits = builder.add_aux_qubits(4)
    aux_registers = builder.add_aux_registers(5)

    sub = builder.build()
    assert sub.aux_qubits == aux_qubits
    assert sub.aux_registers == aux_registers
    assert len(sub.qubits) == 2
    assert len(sub.aux_qubits) == 4
    assert len(sub.registers) == 3
    assert len(sub.aux_registers) == 5


def test_sub_def() -> None:
    class _D(SubDef):
        qubit_count = 2

        def sub(self, builder: SubBuilder) -> None:
            q0, q1 = builder.qubits
            a = builder.add_aux_qubit()
            builder.add_op(Toffoli, (q0, q1, a))

    test_sub = sub(_D)

    assert isinstance(test_sub, Sub)
    assert len(test_sub.qubits) == 2
    assert len(test_sub.aux_qubits) == 1
    assert len(test_sub.operations) == 1


def test_sub_def_register() -> None:
    class _D(SubDef):
        qubit_count = 2
        reg_count = 1

        def sub(self, builder: SubBuilder) -> None:
            q0, q1 = builder.qubits
            a = builder.add_aux_qubit()
            builder.add_op(Toffoli, (q0, q1, a))

    test_sub = sub(_D)

    assert isinstance(test_sub, Sub)
    assert len(test_sub.qubits) == 2
    assert len(test_sub.aux_qubits) == 1
    assert len(test_sub.registers) == 1
    assert len(test_sub.operations) == 1


def test_parametric_sub_def() -> None:
    class _D(ParamSubDef[int]):
        qubit_count = 2
        reg_count = 1

        def sub(self, builder: SubBuilder, times: int) -> None:
            q0, q1 = builder.qubits
            a = builder.add_aux_qubit()
            for _ in range(times):
                builder.add_op(Toffoli, (q0, q1, a))

    psub = param_sub(_D)
    test_sub = psub(3)
    assert isinstance(test_sub, Sub)
    assert len(test_sub.qubits) == 2
    assert len(test_sub.aux_qubits) == 1
    assert len(test_sub.registers) == 1
    assert len(test_sub.operations) == 3


def test_parametric_sub_decorator_variable_qubits() -> None:
    class _D(ParamSubDef[int]):
        def qubit_count_fn(self, bits: int) -> int:
            return 2 * bits

        def reg_count_fn(self, bits: int) -> int:
            return 3 * bits

        def sub(self, builder: SubBuilder, bits: int) -> None:
            qubits = tuple(builder.qubits)
            for q in qubits:
                builder.add_op(X, (q,))

    psub = param_sub(_D)
    test_sub = psub(3)
    assert isinstance(test_sub, Sub)
    assert len(test_sub.qubits) == 6
    assert len(test_sub.registers) == 9


def test_sub_add_aux_qreg() -> None:
    builder = SubBuilder(2)

    qreg = builder.add_aux_qreg("my_aux", 2)

    assert qreg.name == "my_aux"
    assert len(qreg.qubits) == 2
    assert qreg.size == 2
    assert "my_aux" in builder.aux_qregs
    assert builder.aux_qregs["my_aux"] == qreg
    assert qreg.qubits[0] == Qubit(2)
    assert qreg.qubits[1] == Qubit(3)

    q_solo = builder.add_aux_qubit()
    assert q_solo == Qubit(4)


def test_sub_add_aux_qubits_register_names() -> None:
    builder = SubBuilder(2)

    qubits = builder.add_aux_qubits(3)

    # Each qubit should have a dedicated register named aux_{uid}
    assert qubits[0] == Qubit(2)
    assert qubits[1] == Qubit(3)
    assert qubits[2] == Qubit(4)

    assert "aux_2" in builder.aux_qregs
    assert "aux_3" in builder.aux_qregs
    assert "aux_4" in builder.aux_qregs

    assert builder.aux_qregs["aux_2"].qubits == (Qubit(2),)
    assert builder.aux_qregs["aux_3"].qubits == (Qubit(3),)
    assert builder.aux_qregs["aux_4"].qubits == (Qubit(4),)


def test_sub_from_qregs() -> None:
    specs = [QRegSpec("a", 2), QRegSpec("b", 1)]
    builder = SubBuilder.from_qregs(specs, arg_reg_count=2)

    assert len(builder.qubits) == 3
    assert len(builder.registers) == 2

    assert "a" in builder.qregs
    assert "b" in builder.qregs

    reg_a = builder.qregs["a"]
    reg_b = builder.qregs["b"]

    assert reg_a.name == "a"
    assert reg_a.size == 2
    assert reg_b.name == "b"
    assert reg_b.size == 1

    assert reg_a.qubits[0] == Qubit(0)
    assert reg_a.qubits[1] == Qubit(1)

    assert reg_b.qubits[0] == Qubit(2)


def test_sub_from_qregs_empty() -> None:
    builder = SubBuilder.from_qregs([], arg_reg_count=0)
    assert len(builder.qubits) == 0
    assert len(builder.registers) == 0


def test_connect_basic() -> None:
    class _D(OpDef):
        name = "MyOp"
        qubit_count = 3
        qregs = (QRegSpec("a", 1), QRegSpec("b", 2))

    D = op(_D)

    builder = SubBuilder.from_qregs([QRegSpec("reg1", 2), QRegSpec("reg2", 2)])
    reg1 = builder.qregs["reg1"]
    reg2 = builder.qregs["reg2"]

    builder.connect(D, a=(reg1[1],), b=reg2)

    built_sub = builder.build()
    assert len(built_sub.operations) == 1
    op_inst, qubits, _ = built_sub.operations[0]

    assert op_inst.id.base == (DEFAULT, "MyOp")
    assert qubits == (reg1[1], reg2[0], reg2[1])


def test_connect_with_slices() -> None:
    class _D(OpDef):
        name = "MyOp"
        qubit_count = 2
        qregs = (QRegSpec("a", 1), QRegSpec("b", 1))

    D = op(_D)

    builder = SubBuilder(2)
    qubits = builder.qubits
    builder.connect(D, a=(qubits[0],), b=(qubits[1],))

    built_sub = builder.build()
    assert len(built_sub.operations) == 1
    _, qs, _ = built_sub.operations[0]
    assert qs == (qubits[0], qubits[1])


def test_connect_mismatch() -> None:
    class _D(OpDef):
        name = "MyOp"
        qubit_count = 2
        qregs = (QRegSpec("a", 2),)  # Wants 2 qubits

    D = op(_D)

    builder = SubBuilder(2)
    q = builder.qubits[0]

    with pytest.raises(AssertionError, match="Qubit count mismatch.*Expected 2"):
        builder.connect(D, a=(q,))


def test_connect_missing_arg() -> None:
    class _D(OpDef):
        name = "MyOp"
        qubit_count = 1
        qregs = (QRegSpec("req", 1),)

    D = op(_D)
    builder = SubBuilder(1)

    with pytest.raises(KeyError):
        builder.connect(D, wrong_name=[builder.qubits[0]])
