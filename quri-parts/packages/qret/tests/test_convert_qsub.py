import pytest
from pyqret.frontend import Module

import quri_parts.qsub.lib.std as std
from quri_parts.qret.convert_qsub import create_module_from_qsub_op
from quri_parts.qsub.lib.qpe import QPE
from quri_parts.qsub.opsub import NonUnitarySubDef, UnitarySubDef, opsub
from quri_parts.qsub.primitive import FTQCBasicSet
from quri_parts.qsub.sub import SubBuilder


class _U(UnitarySubDef):
    name = "U"
    qubit_count = 3

    def sub(self, builder: SubBuilder) -> None:
        for i, q in enumerate(builder.qubits):
            builder.add_op(std.RX(0.1 * (i + 1)), (q,))


U, _ = opsub(_U)


class _UMCX(UnitarySubDef):
    name = "UMCX"
    qubit_count = 5

    def sub(self, builder: SubBuilder) -> None:
        q0, q1, q2, q3, q4 = builder.qubits
        builder.add_op(std.MCX(2), (q0, q1, q2))
        builder.add_op(std.MCX(3), (q0, q1, q2, q3))
        builder.add_op(std.MCX(4), (q0, q1, q2, q3, q4))


UMCX, _ = opsub(_UMCX)


class _Conditional(NonUnitarySubDef):
    name = "Conditional"
    qubit_count = 1
    reg_count = 1

    def sub(self, builder: SubBuilder) -> None:
        (q0,) = builder.qubits
        (r0,) = builder.registers
        builder.add_op(std.M, (q0,), (r0,))
        with std.conditional(builder, r0):
            builder.add_op(std.X, (q0,))


Conditional, _ = opsub(_Conditional)


class _Bell(NonUnitarySubDef):
    name = "Bell"
    qubit_count = 2
    reg_count = 2

    def sub(self, builder: SubBuilder) -> None:
        q0, q1 = builder.qubits
        r0, r1 = builder.registers
        builder.add_op(std.H, (q0,))
        builder.add_op(std.CNOT, (q0, q1))
        builder.add_op(std.M, (q0,), (r0,))
        builder.add_op(std.M, (q1,), (r1,))


Bell, _ = opsub(_Bell)


class _AndCliffordT(UnitarySubDef):
    name = "AndCliffordT"
    qubit_count = 3

    def sub(self, builder: SubBuilder) -> None:
        i0, i1, t = builder.qubits
        with std.scoped_and_clifford_t(builder, i0, i1) as a:
            builder.add_op(std.CNOT, (a, t))


AndCliffordT, _ = opsub(_AndCliffordT)


def _instruction_names(module: Module, circuit_name: str) -> list[str]:
    circuit = module.get_circuit(circuit_name)
    return [str(inst).split()[0] for block in circuit.get_ir() for inst in block]


class TestCreateModuleFromQsubOp:
    def test_create_module_from_qpe_op(self) -> None:
        qpe_u_op = QPE(4, U)
        module = create_module_from_qsub_op(qpe_u_op)

        circuits = module.get_circuit_list()
        assert len(circuits) >= 1
        assert any("QPE" in name for name in circuits)

    def test_circuit_ir_structure(self) -> None:
        qpe_u_op = QPE(4, U)
        module = create_module_from_qsub_op(qpe_u_op)

        circuits = module.get_circuit_list()
        for circuit_name in circuits:
            circuit = module.get_circuit(circuit_name)
            ir_blocks = list(circuit.get_ir())
            # Each circuit should have at least one basic block
            assert len(ir_blocks) > 0
            # First block should be entry
            assert ir_blocks[0].name() == "entry"

    def test_create_module_from_mcx_op(self) -> None:
        module = create_module_from_qsub_op(UMCX)

        circuits = module.get_circuit_list()
        assert len(circuits) == 1
        assert any("UMCX" in name for name in circuits)

    def test_entry_circuit_name_option(self) -> None:
        custom_entry_name = "entry_point"
        module = create_module_from_qsub_op(UMCX, entry_circuit_name=custom_entry_name)

        circuits = module.get_circuit_list()
        assert len(circuits) == 1
        assert custom_entry_name in circuits
        assert all("UMCX" not in name for name in circuits)

        module_without_wrapper = create_module_from_qsub_op(
            UMCX, entry_circuit_name=None
        )
        circuits_without_wrapper = module_without_wrapper.get_circuit_list()
        assert len(circuits_without_wrapper) == 1
        assert custom_entry_name not in circuits_without_wrapper
        assert any("UMCX" in name for name in circuits_without_wrapper)

    def test_create_module_from_conditional_op(self) -> None:
        module = create_module_from_qsub_op(Conditional)

        circuit = module.get_circuit(Conditional.id.to_str())
        ir_blocks = list(circuit.get_ir())
        assert len(ir_blocks) > 1
        assert "entry" in circuit.get_ir().gen_cfg()

    def test_create_module_from_measure_op(self) -> None:
        module = create_module_from_qsub_op(Bell)

        names = _instruction_names(module, Bell.id.to_str())
        assert names.count("Measurement") == 2

    def test_measure_with_gateset_primitives(self) -> None:
        module = create_module_from_qsub_op(Bell, primitives=FTQCBasicSet)

        circuits = module.get_circuit_list()
        assert circuits == [Bell.id.to_str()]
        names = _instruction_names(module, Bell.id.to_str())
        assert names.count("Measurement") == 2

    def test_conditional_with_gateset_primitives(self) -> None:
        gateset = (std.H, std.Sdag, std.T, std.Tdag, std.CNOT, std.CZ)
        module = create_module_from_qsub_op(AndCliffordT, primitives=gateset)

        circuit_name = AndCliffordT.id.to_str()
        assert module.get_circuit_list() == [circuit_name]
        names = _instruction_names(module, circuit_name)
        assert "Measurement" in names
        assert "Branch" in names

    @pytest.mark.parametrize("control_bits", [2, 3, 4])
    def test_create_module_from_single_mcx_op(self, control_bits: int) -> None:
        class _SingleMCX(UnitarySubDef):
            name = f"SingleMCX{control_bits}"
            qubit_count = control_bits + 1

            def sub(self, builder: SubBuilder) -> None:
                builder.add_op(std.MCX(control_bits), builder.qubits)

        single_mcx, _ = opsub(_SingleMCX)
        module = create_module_from_qsub_op(single_mcx)

        circuits = module.get_circuit_list()
        assert len(circuits) == 1
        assert any(f"SingleMCX{control_bits}" in name for name in circuits)
