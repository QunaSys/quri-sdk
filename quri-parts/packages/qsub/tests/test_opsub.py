from quri_parts.qsub.namespace import DEFAULT, NameSpace
from quri_parts.qsub.op import Ident, Op
from quri_parts.qsub.opsub import (
    NonUnitarySubDef,
    OpSubDef,
    ParamNonUnitarySubDef,
    ParamOpSubDef,
    ParamUnitarySubDef,
    UnitarySubDef,
    opsub,
    param_opsub,
)
from quri_parts.qsub.resolve import default_repository
from quri_parts.qsub.sub import Sub, SubBuilder

X = Op(Ident(DEFAULT, "X", ()), qubit_count=1)


class TestOpSubDef:
    def test_opsub_minimal(self) -> None:
        class _D(OpSubDef):
            name = "Foo"
            qubit_count = 3

            def sub(self, builder: SubBuilder) -> None:
                a, b, c = builder.qubits
                builder.add_op(X, (a,))
                builder.add_op(X, (b,))
                builder.add_op(X, (c,))

        Foo, FooSub = opsub(_D, None)
        assert Foo == Op(Ident(DEFAULT, "Foo", ()), qubit_count=3, unitary=True)
        assert isinstance(FooSub, Sub)
        assert len(FooSub.qubits) == 3
        assert len(FooSub.operations) == 3

    def test_unitary_sub(self) -> None:
        class _D(UnitarySubDef):
            name = "Foo"
            qubit_count = 3

            def sub(self, builder: SubBuilder) -> None:
                a, b, c = builder.qubits
                builder.add_op(X, (a,))
                builder.add_op(X, (b,))
                builder.add_op(X, (c,))

        Foo, FooSub = opsub(_D, None)
        assert Foo == Op(Ident(DEFAULT, "Foo", ()), qubit_count=3, unitary=True)
        assert isinstance(FooSub, Sub)
        assert len(FooSub.qubits) == 3
        assert len(FooSub.operations) == 3

    def test_non_unitary_sub(self) -> None:
        class _D(NonUnitarySubDef):
            name = "Foo"
            qubit_count = 3

            def sub(self, builder: SubBuilder) -> None:
                a, b, c = builder.qubits
                builder.add_op(X, (a,))
                builder.add_op(X, (b,))
                builder.add_op(X, (c,))

        Foo, FooSub = opsub(_D, None)
        assert Foo == Op(Ident(DEFAULT, "Foo", ()), qubit_count=3, unitary=False)
        assert isinstance(FooSub, Sub)
        assert len(FooSub.qubits) == 3
        assert len(FooSub.operations) == 3

    def test_opsub_full(self) -> None:
        namespace = NameSpace("foo")

        class _D(OpSubDef):
            ns = namespace
            name = "Foo"
            params = (1, 2)
            qubit_count = 3
            reg_count = 2
            unitary = False

            def sub(self, builder: SubBuilder) -> None:
                a, b, c = builder.qubits
                builder.add_op(X, (a,))
                builder.add_op(X, (b,))
                builder.add_op(X, (c,))

        Foo, FooSub = opsub(_D, None)
        assert Foo == Op(
            Ident(namespace, "Foo", (1, 2)), qubit_count=3, reg_count=2, unitary=False
        )
        assert isinstance(FooSub, Sub)
        assert len(FooSub.qubits) == 3
        assert len(FooSub.operations) == 3

    def test_opsub_register(self) -> None:
        class _D(OpSubDef):
            name = "Foo"
            qubit_count = 3

            def sub(self, builder: SubBuilder) -> None:
                a, b, c = builder.qubits
                builder.add_op(X, (a,))
                builder.add_op(X, (b,))
                builder.add_op(X, (c,))

        Foo, FooSub = opsub(_D)
        repo = default_repository()
        resolver = repo.find_resolver(Foo)
        assert resolver is not None
        assert resolver(Foo, repo) == FooSub


class TestParamOpSubDef:
    def test_param_opsub_minimal(self) -> None:
        class _D(ParamOpSubDef[int]):
            name = "Foo"
            qubit_count = 3

            def sub(self, builder: SubBuilder, times: int) -> None:
                a, b, c = builder.qubits
                for _ in range(times):
                    builder.add_op(X, (a,))
                    builder.add_op(X, (b,))
                    builder.add_op(X, (c,))

        Foo, FooSub = param_opsub(_D, None)
        assert Foo.base_id == (DEFAULT, "Foo")

        assert Foo(2) == Op(Ident(DEFAULT, "Foo", (2,)), qubit_count=3)

        assert isinstance(FooSub(2), Sub)
        assert len(FooSub(2).qubits) == 3
        assert len(FooSub(2).operations) == 6

    def test_param_unitary_sub(self) -> None:
        class _D(ParamUnitarySubDef[int]):
            name = "Foo"
            qubit_count = 3

            def sub(self, builder: SubBuilder, times: int) -> None:
                a, b, c = builder.qubits
                for _ in range(times):
                    builder.add_op(X, (a,))
                    builder.add_op(X, (b,))
                    builder.add_op(X, (c,))

        Foo, FooSub = param_opsub(_D, None)
        assert Foo.base_id == (DEFAULT, "Foo")

        assert Foo(2) == Op(Ident(DEFAULT, "Foo", (2,)), qubit_count=3, unitary=True)

        assert isinstance(FooSub(2), Sub)
        assert len(FooSub(2).qubits) == 3
        assert len(FooSub(2).operations) == 6

    def test_param_non_unitary_sub(self) -> None:
        class _D(ParamNonUnitarySubDef[int]):
            name = "Foo"
            qubit_count = 3

            def sub(self, builder: SubBuilder, times: int) -> None:
                a, b, c = builder.qubits
                for _ in range(times):
                    builder.add_op(X, (a,))
                    builder.add_op(X, (b,))
                    builder.add_op(X, (c,))

        Foo, FooSub = param_opsub(_D, None)
        assert Foo.base_id == (DEFAULT, "Foo")

        assert Foo(2) == Op(Ident(DEFAULT, "Foo", (2,)), qubit_count=3, unitary=False)

        assert isinstance(FooSub(2), Sub)
        assert len(FooSub(2).qubits) == 3
        assert len(FooSub(2).operations) == 6

    def test_param_opsub_full(self) -> None:
        namespace = NameSpace("foo")

        class _D(ParamOpSubDef[int]):
            ns = namespace
            name = "Foo"
            unitary = False

            def qubit_count_fn(self, bits: int) -> int:
                return 2 * bits

            def reg_count_fn(self, bits: int) -> int:
                return 3 * bits

            def sub(self, builder: SubBuilder, bits: int) -> None:
                qubits = tuple(builder.qubits)
                for q in qubits:
                    builder.add_op(X, (q,))

        Foo, FooSub = param_opsub(_D, None)
        assert Foo.base_id == (namespace, "Foo")

        assert Foo(2) == Op(
            Ident(namespace, "Foo", (2,)), qubit_count=4, reg_count=6, unitary=False
        )

        sub = FooSub(2)
        assert isinstance(sub, Sub)
        assert len(sub.qubits) == 4
        assert len(sub.registers) == 6
        assert len(sub.operations) == 4

    def test_param_opsub_register(self) -> None:
        class _D(ParamOpSubDef[int]):
            name = "Foo"
            qubit_count = 3

            def sub(self, builder: SubBuilder, times: int) -> None:
                a, b, c = builder.qubits
                for _ in range(times):
                    builder.add_op(X, (a,))
                    builder.add_op(X, (b,))
                    builder.add_op(X, (c,))

        Foo, FooSub = param_opsub(_D)
        repo = default_repository()
        resolver = repo.find_resolver(Foo(2))
        assert resolver is not None
        assert resolver(Foo(2), repo) == FooSub(2)
