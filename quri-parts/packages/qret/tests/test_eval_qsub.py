import quri_parts.qsub.lib.std as std
from quri_parts.qret.convert_qsub import create_module_from_qsub_op
from quri_parts.qsub.opsub import UnitarySubDef, opsub
from quri_parts.qsub.sub import SubBuilder


class _SimpleSub(UnitarySubDef):
    name = "SimpleSub"
    qubit_count = 2

    def sub(self, builder: SubBuilder) -> None:
        q0, q1 = builder.qubits
        builder.add_op(std.H, (q0,))
        builder.add_op(std.CNOT, (q0, q1))


SimpleSub, _ = opsub(_SimpleSub)


def test_create_module_from_qsub_op() -> None:
    module = create_module_from_qsub_op(SimpleSub)
    circuits = module.get_circuit_list()

    assert len(circuits) == 1
    assert any(name.endswith("SimpleSub") for name in circuits)


class _CustomEntry(UnitarySubDef):
    name = "custom_entry"
    qubit_count = 2

    def sub(self, builder: SubBuilder) -> None:
        q0, q1 = builder.qubits
        builder.add_op(std.H, (q0,))
        builder.add_op(std.CNOT, (q0, q1))


CustomEntry, _ = opsub(_CustomEntry)


def test_custom_entry_name_in_module() -> None:
    module = create_module_from_qsub_op(CustomEntry)
    circuits = module.get_circuit_list()

    assert len(circuits) == 1
    assert any(name.endswith("custom_entry") for name in circuits)
