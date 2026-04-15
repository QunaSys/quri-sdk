from pyqret.frontend import Module, QuantumAttribute, QuantumType

import quri_parts.qsub.lib.std as std
from quri_parts.qret.convert_qsub import create_module_from_qsub_op
from quri_parts.qsub.opsub import UnitarySubDef, opsub
from quri_parts.qsub.sub import SubBuilder


class _InnerWithAux(UnitarySubDef):
    """An op whose Sub uses auxiliary qubits.

    qubit_count=2 but the Sub internally allocates 1 auxiliary qubit, so
    the MachineSub has qubits=(0,1) and aux_qubits=(2,).
    """

    name = "InnerWithAux"
    qubit_count = 2

    def sub(self, builder: SubBuilder) -> None:
        q0, q1 = builder.qubits
        aux = builder.add_aux_qubit()
        builder.add_op(std.H, (aux,))
        builder.add_op(std.CNOT, (q0, aux))
        builder.add_op(std.CNOT, (q1, aux))
        builder.add_op(std.CNOT, (q0, aux))
        builder.add_op(std.H, (aux,))


InnerWithAux, _ = opsub(_InnerWithAux)


class _OuterCallsInner(UnitarySubDef):
    """An op that calls InnerWithAux as a sub-operation."""

    name = "OuterCallsInner"
    qubit_count = 3

    def sub(self, builder: SubBuilder) -> None:
        q0, q1, q2 = builder.qubits
        builder.add_op(InnerWithAux, (q0, q1))
        builder.add_op(std.CNOT, (q2, q0))


OuterCallsInner, _ = opsub(_OuterCallsInner)


class _InnerWithAuxReg(UnitarySubDef):
    """An op whose Sub uses both auxiliary qubits and auxiliary registers."""

    name = "InnerWithAuxReg"
    qubit_count = 2

    def sub(self, builder: SubBuilder) -> None:
        q0, q1 = builder.qubits
        aux_q = builder.add_aux_qubit()
        _aux_r = builder.add_aux_register()  # noqa: F841
        builder.add_op(std.H, (aux_q,))
        builder.add_op(std.CNOT, (q0, aux_q))
        builder.add_op(std.CNOT, (q1, aux_q))
        builder.add_op(std.H, (aux_q,))


InnerWithAuxReg, _ = opsub(_InnerWithAuxReg)


class _OuterCallsInnerWithAuxReg(UnitarySubDef):
    name = "OuterCallsInnerAuxReg"
    qubit_count = 3

    def sub(self, builder: SubBuilder) -> None:
        q0, q1, q2 = builder.qubits
        builder.add_op(InnerWithAuxReg, (q0, q1))
        builder.add_op(std.CNOT, (q2, q0))


OuterCallsInnerWithAuxReg, _ = opsub(_OuterCallsInnerWithAuxReg)


class _MidWithAuxCallsInner(UnitarySubDef):
    """Nested case: this op has aux qubits and calls an op with aux qubits."""

    name = "MidWithAuxCallsInner"
    qubit_count = 2

    def sub(self, builder: SubBuilder) -> None:
        q0, q1 = builder.qubits
        mid_aux = builder.add_aux_qubit()
        builder.add_op(std.H, (mid_aux,))
        builder.add_op(InnerWithAux, (q0, q1))
        builder.add_op(std.CNOT, (mid_aux, q0))
        builder.add_op(std.H, (mid_aux,))


MidWithAuxCallsInner, _ = opsub(_MidWithAuxCallsInner)


class _OuterCallsMidNestedAux(UnitarySubDef):
    """Outer op that calls a nested aux-qubit sub-operation."""

    name = "OuterCallsMidNestedAux"
    qubit_count = 3

    def sub(self, builder: SubBuilder) -> None:
        q0, q1, q2 = builder.qubits
        builder.add_op(MidWithAuxCallsInner, (q0, q1))
        builder.add_op(std.CNOT, (q2, q0))


OuterCallsMidNestedAux, _ = opsub(_OuterCallsMidNestedAux)


def _get_circuit_name(circuits: list[str], op_name: str) -> str:
    return next(name for name in circuits if op_name in name)


def _assert_circuit_args(
    module: Module,
    circuit_name: str,
    expected: list[tuple[str, QuantumType, QuantumAttribute, int]],
) -> None:
    circuit = module.get_circuit(circuit_name)
    argument = circuit.argument

    expected_names = [name for name, _, _, _ in expected]
    assert argument.get_arg_names() == expected_names
    assert argument.get_num_args() == len(expected)

    for name, qtype, attr, size in expected:
        info = argument.view_arg_info(name)
        info_type = info.type() if callable(info.type) else info.type
        info_attr = info.attribute() if callable(info.attribute) else info.attribute
        info_size = info.size() if callable(info.size) else info.size
        assert info_type == qtype
        assert info_attr == attr
        assert info_size == size


def _get_opcode_strings(module: Module, circuit_name: str) -> list[str]:
    circuit = module.get_circuit(circuit_name)
    return [
        str(inst.get_opcode()).lower() for block in circuit.get_ir() for inst in block
    ]


def _count_opcode(opcodes: list[str], needle: str) -> int:
    return sum(1 for op in opcodes if needle in op)


class TestCreateModuleWithAuxQubits:
    def test_subcall_with_aux_qubits(self) -> None:
        """bug: argument count mismatch when a sub-operation has auxiliary qubits."""
        module = create_module_from_qsub_op(OuterCallsInner)
        circuits = module.get_circuit_list()
        assert len(circuits) == 2

        outer = _get_circuit_name(circuits, "OuterCallsInner")
        inner = _get_circuit_name(circuits, "InnerWithAux")

        _assert_circuit_args(
            module,
            outer,
            [
                ("q0", QuantumType.Qubit, QuantumAttribute.Operate, 1),
                ("q1", QuantumType.Qubit, QuantumAttribute.Operate, 1),
                ("q2", QuantumType.Qubit, QuantumAttribute.Operate, 1),
                ("a0", QuantumType.Qubit, QuantumAttribute.CleanAncilla, 1),
            ],
        )
        _assert_circuit_args(
            module,
            inner,
            [
                ("q0", QuantumType.Qubit, QuantumAttribute.Operate, 1),
                ("q1", QuantumType.Qubit, QuantumAttribute.Operate, 1),
                ("a0", QuantumType.Qubit, QuantumAttribute.CleanAncilla, 1),
            ],
        )

        outer_ops = _get_opcode_strings(module, outer)
        assert _count_opcode(outer_ops, "call") == 1
        assert _count_opcode(outer_ops, "cx") == 1

        inner_ops = _get_opcode_strings(module, inner)
        assert _count_opcode(inner_ops, "h") == 2
        assert _count_opcode(inner_ops, "cx") == 3

    def test_subcall_with_aux_qubits_and_registers(self) -> None:
        """Same bug with both auxiliary qubits and auxiliary registers."""
        module = create_module_from_qsub_op(OuterCallsInnerWithAuxReg)
        circuits = module.get_circuit_list()
        assert len(circuits) == 2

        outer = _get_circuit_name(circuits, "OuterCallsInnerAuxReg")
        inner = _get_circuit_name(circuits, "InnerWithAuxReg")

        _assert_circuit_args(
            module,
            outer,
            [
                ("q0", QuantumType.Qubit, QuantumAttribute.Operate, 1),
                ("q1", QuantumType.Qubit, QuantumAttribute.Operate, 1),
                ("q2", QuantumType.Qubit, QuantumAttribute.Operate, 1),
                ("a0", QuantumType.Qubit, QuantumAttribute.CleanAncilla, 1),
                ("ar0", QuantumType.Register, QuantumAttribute.Output, 1),
            ],
        )
        _assert_circuit_args(
            module,
            inner,
            [
                ("q0", QuantumType.Qubit, QuantumAttribute.Operate, 1),
                ("q1", QuantumType.Qubit, QuantumAttribute.Operate, 1),
                ("a0", QuantumType.Qubit, QuantumAttribute.CleanAncilla, 1),
                ("ar0", QuantumType.Register, QuantumAttribute.Output, 1),
            ],
        )

        outer_ops = _get_opcode_strings(module, outer)
        assert _count_opcode(outer_ops, "call") == 1
        assert _count_opcode(outer_ops, "cx") == 1

        inner_ops = _get_opcode_strings(module, inner)
        assert _count_opcode(inner_ops, "h") == 2
        assert _count_opcode(inner_ops, "cx") == 2

    def test_inner_with_aux_as_entry(self) -> None:
        """When the entry op itself has aux qubits but no SubCalls, it should
        work fine (no _add_funcall involved)."""
        module = create_module_from_qsub_op(InnerWithAux)
        circuits = module.get_circuit_list()
        assert len(circuits) == 1
        inner = _get_circuit_name(circuits, "InnerWithAux")

        _assert_circuit_args(
            module,
            inner,
            [
                ("q0", QuantumType.Qubit, QuantumAttribute.Operate, 1),
                ("q1", QuantumType.Qubit, QuantumAttribute.Operate, 1),
                ("a0", QuantumType.Qubit, QuantumAttribute.CleanAncilla, 1),
            ],
        )

        inner_ops = _get_opcode_strings(module, inner)
        assert _count_opcode(inner_ops, "h") == 2
        assert _count_opcode(inner_ops, "cx") == 3

    def test_nested_subcalls_with_aux_qubits(self) -> None:
        """Nested subcalls should propagate enough aux qubits to all
        callers."""
        module = create_module_from_qsub_op(OuterCallsMidNestedAux)
        circuits = module.get_circuit_list()
        assert len(circuits) == 3

        outer = _get_circuit_name(circuits, "OuterCallsMidNestedAux")
        mid = _get_circuit_name(circuits, "MidWithAuxCallsInner")
        inner = _get_circuit_name(circuits, "InnerWithAux")

        _assert_circuit_args(
            module,
            outer,
            [
                ("q0", QuantumType.Qubit, QuantumAttribute.Operate, 1),
                ("q1", QuantumType.Qubit, QuantumAttribute.Operate, 1),
                ("q2", QuantumType.Qubit, QuantumAttribute.Operate, 1),
                ("a0", QuantumType.Qubit, QuantumAttribute.CleanAncilla, 1),
                ("a1", QuantumType.Qubit, QuantumAttribute.CleanAncilla, 1),
            ],
        )
        _assert_circuit_args(
            module,
            mid,
            [
                ("q0", QuantumType.Qubit, QuantumAttribute.Operate, 1),
                ("q1", QuantumType.Qubit, QuantumAttribute.Operate, 1),
                ("a0", QuantumType.Qubit, QuantumAttribute.CleanAncilla, 1),
                ("a1", QuantumType.Qubit, QuantumAttribute.CleanAncilla, 1),
            ],
        )
        _assert_circuit_args(
            module,
            inner,
            [
                ("q0", QuantumType.Qubit, QuantumAttribute.Operate, 1),
                ("q1", QuantumType.Qubit, QuantumAttribute.Operate, 1),
                ("a0", QuantumType.Qubit, QuantumAttribute.CleanAncilla, 1),
            ],
        )

        outer_ops = _get_opcode_strings(module, outer)
        assert _count_opcode(outer_ops, "call") == 1
        assert _count_opcode(outer_ops, "cx") == 1

        mid_ops = _get_opcode_strings(module, mid)
        assert _count_opcode(mid_ops, "call") == 1
        assert _count_opcode(mid_ops, "h") == 2
        assert _count_opcode(mid_ops, "cx") == 1

        inner_ops = _get_opcode_strings(module, inner)
        assert _count_opcode(inner_ops, "h") == 2
        assert _count_opcode(inner_ops, "cx") == 3
