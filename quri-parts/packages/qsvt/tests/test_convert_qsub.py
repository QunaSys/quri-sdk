import quri_parts.qsub.lib.std as std
import pytest
from quri_parts.qsub.lib.qpe import QPE
from quri_parts.qsub.opsub import UnitarySubDef, opsub
from quri_parts.qsub.sub import SubBuilder
from quri_parts.qsvt.convert_qsub import create_module_from_qsub_op


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


class TestCreateModuleFromQsubOp:
    def test_create_module_from_qpe_op(self) -> None:
        qpe_u_op = QPE(4, U)
        module = create_module_from_qsub_op(qpe_u_op)

        circuits = module.get_circuit_list()
        assert len(circuits) > 0

        # Verify expected circuit names are present
        circuit_names = set(circuits)
        assert any("QPE" in name for name in circuit_names)
        assert any("LineH" in name for name in circuit_names)
        assert any("Controlled" in name for name in circuit_names)
        assert any("QFTdag" in name for name in circuit_names)

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
        assert len(circuits) > 0
        assert any("UMCX" in name for name in circuits)

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
        assert len(circuits) > 0
        assert any(f"SingleMCX{control_bits}" in name for name in circuits)
