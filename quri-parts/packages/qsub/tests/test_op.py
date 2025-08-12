from quri_parts.qsub.namespace import DEFAULT, NameSpace
from quri_parts.qsub.op import (
    Ident,
    NonUnitaryDef,
    Op,
    OpDef,
    ParamNonUnitaryDef,
    ParamOpDef,
    ParamUnitaryDef,
    UnitaryDef,
    op,
    param_op,
)


class TestOpDef:
    def test_op_minimal(self) -> None:
        class _D(OpDef):
            name = "FooBar"
            qubit_count = 3

        FooBar = op(_D)
        assert FooBar == Op(Ident(DEFAULT, "FooBar", ()), qubit_count=3, unitary=True)

    def test_unitary(self) -> None:
        class _D(UnitaryDef):
            name = "FooBar"
            qubit_count = 3

        FooBar = op(_D)
        assert FooBar == Op(Ident(DEFAULT, "FooBar", ()), qubit_count=3, unitary=True)

    def test_non_unitary(self) -> None:
        class _D(NonUnitaryDef):
            name = "FooBar"
            qubit_count = 3

        FooBar = op(_D)
        assert FooBar == Op(Ident(DEFAULT, "FooBar", ()), qubit_count=3, unitary=False)

    def test_op_full(self) -> None:
        namespace = NameSpace("foo")

        class _D(OpDef):
            ns = namespace
            name = "FooBar"
            params = (1, 2)
            qubit_count = 3
            reg_count = 2
            unitary = False

        FooBar = op(_D)
        assert FooBar == Op(
            Ident(namespace, "FooBar", (1, 2)),
            qubit_count=3,
            reg_count=2,
            unitary=False,
        )


class TestParametricOpDef:
    def test_parametric_op_minimal(self) -> None:
        class _D(ParamOpDef[None]):
            name = "FooBar"
            qubit_count = 3

        FooBar = param_op(_D)
        op = FooBar()
        assert FooBar.base_id == (DEFAULT, "FooBar")
        assert op.qubit_count == 3
        assert op.reg_count == 0
        assert op.unitary

    def test_parametric_unitary(self) -> None:
        class _D(ParamUnitaryDef[None]):
            name = "FooBar"
            qubit_count = 3

        FooBar = param_op(_D)
        op = FooBar()
        assert FooBar.base_id == (DEFAULT, "FooBar")
        assert op.qubit_count == 3
        assert op.reg_count == 0
        assert op.unitary

    def test_parametric_non_unitary(self) -> None:
        class _D(ParamNonUnitaryDef[None]):
            name = "FooBar"
            qubit_count = 3

        FooBar = param_op(_D)
        op = FooBar()
        assert FooBar.base_id == (DEFAULT, "FooBar")
        assert op.qubit_count == 3
        assert op.reg_count == 0
        assert not op.unitary

    def test_parametric_op_full(self) -> None:
        namespace = NameSpace("foo")

        class _D(ParamOpDef[int]):
            ns = namespace
            name = "FooBar"
            unitary = False

            def qubit_count_fn(self, bits: int) -> int:
                return 2 * bits

            def reg_count_fn(self, bits: int) -> int:
                return 3 * bits

        FooBar = param_op(_D)
        op = FooBar(2)
        assert FooBar.base_id == (namespace, "FooBar")
        assert op.qubit_count == 4
        assert op.reg_count == 6
        assert not op.unitary
