import quri_parts.qsub.lib.std as std
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
