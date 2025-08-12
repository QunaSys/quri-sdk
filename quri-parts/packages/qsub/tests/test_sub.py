import pytest

from quri_parts.qsub.op import OpDef, op
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
    assert len(test_sub.operations) == 6
