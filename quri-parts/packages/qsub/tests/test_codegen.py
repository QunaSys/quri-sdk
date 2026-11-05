from quri_parts.qsub.codegen import CodeGenerator
from quri_parts.qsub.lib import std
from quri_parts.qsub.namespace import NameSpace
from quri_parts.qsub.op import Ident, Op
from quri_parts.qsub.sub import SubBuilder

NS = NameSpace("test")


class TestLower:
    def test_records_source_op_id(self) -> None:
        op = Op.from_qubit_count(Ident(NS, "op"), 1)
        builder = SubBuilder(1)
        builder.add_op(std.X, builder.qubits)

        msub = CodeGenerator([std.X]).lower(builder.build(), op.id)
        assert msub.source_op_id == op.id

    def test_source_op_id_defaults_to_none(self) -> None:
        builder = SubBuilder(1)
        builder.add_op(std.X, builder.qubits)

        msub = CodeGenerator([std.X]).lower(builder.build())
        assert msub.source_op_id is None
