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
    """An op that calls InnerWithAux as a sub-operation.

    When create_module_from_qsub_op processes this, it:
    1. Creates a circuit for InnerWithAux with 3 arguments
       (2 operate + 1 clean_ancilla)
    2. In OuterCallsInner's logic, encounters a SubCall to InnerWithAux
       with only 2 qubits (the instruction's qubits)
    3. _add_funcall calls circuit(q0, q1) but circuit expects 3 args
    """

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


class TestCreateModuleWithAuxQubits:
    def test_subcall_with_aux_qubits(self) -> None:
        """Reproduces the QLSS notebook bug: argument count mismatch when a
        sub-operation has auxiliary qubits."""
        module = create_module_from_qsub_op(OuterCallsInner)
        circuits = module.get_circuit_list()
        assert len(circuits) >= 1
        assert any("OuterCallsInner" in name for name in circuits)

    def test_subcall_with_aux_qubits_and_registers(self) -> None:
        """Same bug with both auxiliary qubits and auxiliary registers."""
        module = create_module_from_qsub_op(OuterCallsInnerWithAuxReg)
        circuits = module.get_circuit_list()
        assert len(circuits) >= 1
        assert any("OuterCallsInnerAuxReg" in name for name in circuits)

    def test_inner_with_aux_as_entry(self) -> None:
        """When the entry op itself has aux qubits but no SubCalls, it should
        work fine (no _add_funcall involved)."""
        module = create_module_from_qsub_op(InnerWithAux)
        circuits = module.get_circuit_list()
        assert len(circuits) >= 1
        assert any("InnerWithAux" in name for name in circuits)
