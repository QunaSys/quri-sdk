from typing import Optional, cast

from quri_parts.qsub.lib.std import MCX, H, MultiControlled, Pauli, X, Y
from quri_parts.qsub.namespace import NameSpace
from quri_parts.qsub.op import Ident, Op, OpFactory, SimpleParamOp
from quri_parts.qsub.qubit import Qubit
from quri_parts.qsub.resolve import (
    CompositeSubRepository,
    SimpleSubRepository,
    SimpleSubResolver,
    SubCollector,
    SubRepository,
)
from quri_parts.qsub.resolve.simulator_repo import simulator_repository
from quri_parts.qsub.sub import Sub, SubBuilder

NS = NameSpace("test")


class TestSubRepository:
    def test_find_resolver(self) -> None:
        cont: OpFactory[Ident] = SimpleParamOp((NS, "Controlled"), 1)
        mcont: OpFactory[Ident, int, int] = SimpleParamOp((NS, "MultiControlled"), 1)
        ch = cont(H.id)
        cy = cont(Y.id)
        mc21y = mcont(Y.id, 2, 1)
        mc21h = mcont(H.id, 2, 1)
        c_mc21y = cont(mc21y.id)
        mc21_cy = mcont(cy.id, 2, 1)

        repository = SimpleSubRepository()

        def dummy_sub(i: int) -> Sub:
            return Sub((Qubit(i),), (), (), (), ())

        resolvers: list[SimpleSubResolver[None]] = [
            SimpleSubResolver(dummy_sub(i)) for i in range(6)
        ]
        (
            c_resolver,
            mc_resolver,
            cy_resolver,
            mcy_resolver,
            cmc_resolver,
            mcc_resolver,
        ) = resolvers

        # Resolve with Controlled<>, MultiControlled<>
        repository.register_sub_resolver(cont.base_id, c_resolver)
        repository.register_sub_resolver(mcont.base_id, mc_resolver)
        assert repository.find_resolver(ch) == c_resolver
        assert repository.find_resolver(mc21h) == mc_resolver

        # Resolve with Controlled<Y>, MultiCOntrolled<Y, *, *>
        def cy_resolver_cond(op_id: Ident) -> bool:
            assert op_id.base == cont.base_id
            target_op_id = op_id.params[0]
            assert isinstance(target_op_id, Ident)
            if target_op_id.base == Y.base_id:
                return True
            return False

        def mcy_resolver_cond(op_id: Ident) -> bool:
            assert op_id.base == mcont.base_id
            target_op_id = op_id.params[0]
            assert isinstance(target_op_id, Ident)
            return target_op_id.base == Y.base_id

        repository.register_sub_resolver(cont.base_id, cy_resolver, cy_resolver_cond)
        repository.register_sub_resolver(mcont.base_id, mcy_resolver, mcy_resolver_cond)
        assert repository.find_resolver(cy) == cy_resolver
        assert repository.find_resolver(mc21y) == mcy_resolver

        # Resolve with Controlled<MultiControlled<>>, MultiControlled<Controlled<>>
        def cmc_resolver_cond(op_id: Ident) -> bool:
            assert op_id.base == cont.base_id
            target_op_id = op_id.params[0]
            assert isinstance(target_op_id, Ident)
            return target_op_id.base == mcont.base_id

        def mcc_resolver_cond(op_id: Ident) -> bool:
            assert op_id.base == mcont.base_id
            target_op_id = op_id.params[0]
            assert isinstance(target_op_id, Ident)
            return target_op_id.base == cont.base_id

        repository.register_sub_resolver(cont.base_id, cmc_resolver, cmc_resolver_cond)
        repository.register_sub_resolver(mcont.base_id, mcc_resolver, mcc_resolver_cond)
        assert repository.find_resolver(c_mc21y) == cmc_resolver
        assert repository.find_resolver(mc21_cy) == mcc_resolver

    def test_with_override(self) -> None:
        op1 = Op(Ident(NS, "op1"), 1)
        op2 = Op(Ident(NS, "op2"), 1)
        op3 = Op(Ident(NS, "op3"), 1)

        # Create parent repository with resolvers for op1 and op2
        parent_repo = SimpleSubRepository()

        def parent_op1_sub() -> Sub:
            builder = SubBuilder(1)
            (q,) = builder.qubits
            builder.add_op(X, (q,))
            return builder.build()

        def parent_op2_sub() -> Sub:
            builder = SubBuilder(1)
            (q,) = builder.qubits
            builder.add_op(Y, (q,))
            return builder.build()

        parent_repo.register_sub(op1, parent_op1_sub())
        parent_repo.register_sub(op2, parent_op2_sub())

        # Create child repository that overrides op1 and adds op3
        child_repo = SimpleSubRepository()

        def child_op1_sub() -> Sub:
            builder = SubBuilder(1)
            (q,) = builder.qubits
            builder.add_op(H, (q,))
            return builder.build()

        def child_op3_sub() -> Sub:
            builder = SubBuilder(1)
            (q,) = builder.qubits
            builder.add_op(X, (q,))
            builder.add_op(Y, (q,))
            return builder.build()

        child_repo.register_sub(op1, child_op1_sub())
        child_repo.register_sub(op3, child_op3_sub())

        # Create composite repository using with_override
        composite_repo = parent_repo.with_override(child_repo)

        # Test that child overrides parent for op1
        resolver1 = composite_repo.find_resolver(op1)
        assert resolver1 is not None
        sub1 = resolver1(op1, composite_repo)
        assert sub1 == child_op1_sub()  # Should use child's resolver

        # Test that parent resolver is used for op2 (not overridden)
        resolver2 = composite_repo.find_resolver(op2)
        assert resolver2 is not None
        sub2 = resolver2(op2, composite_repo)
        assert sub2 == parent_op2_sub()  # Should use parent's resolver

        # Test that child's new resolver works for op3
        resolver3 = composite_repo.find_resolver(op3)
        assert resolver3 is not None
        sub3 = resolver3(op3, composite_repo)
        assert sub3 == child_op3_sub()

    def test_composite_parent_and_child(self) -> None:
        """Test case where both parent and child repositories are CompositeSubRepository."""
        op0 = Op(Ident(NS, "op0"), 1)
        op1 = Op(Ident(NS, "op1"), 1)
        op2 = Op(Ident(NS, "op2"), 1)
        op3 = Op(Ident(NS, "op3"), 1)

        # Create base repository with op0
        override0_repo = SimpleSubRepository()

        def override0_op0_sub() -> Sub:
            builder = SubBuilder(1)
            (q,) = builder.qubits
            builder.add_op(X, (q,))
            return builder.build()

        override0_repo.register_sub(op0, override0_op0_sub())

        # Create first override repository with op1
        override1_repo = SimpleSubRepository()

        def override1_op1_sub() -> Sub:
            builder = SubBuilder(1)
            (q,) = builder.qubits
            builder.add_op(Y, (q,))
            return builder.build()

        override1_repo.register_sub(op1, override1_op1_sub())

        # First CompositeSubRepository (will be parent)
        parent_composite = override0_repo.with_override(override1_repo)
        assert isinstance(parent_composite, CompositeSubRepository)

        # Create second override repository with op2
        override2_repo = SimpleSubRepository()

        def override2_op2_sub() -> Sub:
            builder = SubBuilder(1)
            (q,) = builder.qubits
            builder.add_op(H, (q,))
            return builder.build()

        override2_repo.register_sub(op2, override2_op2_sub())

        # Second CompositeSubRepository (will be child)
        child_composite = parent_composite.with_override(override2_repo)
        assert isinstance(child_composite, CompositeSubRepository)

        # Create final CompositeSubRepository where both parent and child are Composite
        override3_repo = SimpleSubRepository()

        def override3_op3_sub() -> Sub:
            builder = SubBuilder(1)
            (q,) = builder.qubits
            builder.add_op(X, (q,))
            builder.add_op(Y, (q,))
            return builder.build()

        override3_repo.register_sub(op3, override3_op3_sub())

        # This creates a CompositeSubRepository where:
        # - parent is child_composite (which is itself a CompositeSubRepository)
        # - child is override3_repo (SimpleSubRepository)
        final_composite = child_composite.with_override(override3_repo)

        assert isinstance(final_composite, CompositeSubRepository)
        assert isinstance(final_composite.parent_repo, CompositeSubRepository)
        assert isinstance(final_composite.child_repo, SimpleSubRepository)

        # Verify resolution works correctly through the hierarchy
        # op3 should come from override3_repo
        resolver3 = final_composite.find_resolver(op3)
        assert resolver3 is not None
        assert resolver3(op3, final_composite) == override3_op3_sub()

        # op2 should come from override2_repo (through child_composite)
        resolver2 = final_composite.find_resolver(op2)
        assert resolver2 is not None
        assert resolver2(op2, final_composite) == override2_op2_sub()

        # op1 should come from override1_repo (through parent_composite)
        resolver1 = final_composite.find_resolver(op1)
        assert resolver1 is not None
        assert resolver1(op1, final_composite) == override1_op1_sub()

        # op0 should come from override0_repo
        resolver0 = final_composite.find_resolver(op0)
        assert resolver0 is not None
        assert resolver0(op0, final_composite) == override0_op0_sub()

    def test_both_parent_and_child_composite(self) -> None:
        """Test case where BOTH parent and child are CompositeSubRepository."""
        op0 = Op(Ident(NS, "op0"), 1)
        op1 = Op(Ident(NS, "op1"), 1)
        op2 = Op(Ident(NS, "op2"), 1)
        op3 = Op(Ident(NS, "op3"), 1)

        # Build parent composite: override0 + override1
        override0_repo = SimpleSubRepository()

        def override0_op0_sub() -> Sub:
            builder = SubBuilder(1)
            (q,) = builder.qubits
            builder.add_op(X, (q,))
            return builder.build()

        override0_repo.register_sub(op0, override0_op0_sub())

        override1_repo = SimpleSubRepository()

        def override1_op1_sub() -> Sub:
            builder = SubBuilder(1)
            (q,) = builder.qubits
            builder.add_op(Y, (q,))
            return builder.build()

        override1_repo.register_sub(op1, override1_op1_sub())

        parent_composite = override0_repo.with_override(override1_repo)
        assert isinstance(parent_composite, CompositeSubRepository)

        # Build child composite: override2 + override3
        override2_repo = SimpleSubRepository()

        def override2_op2_sub() -> Sub:
            builder = SubBuilder(1)
            (q,) = builder.qubits
            builder.add_op(H, (q,))
            return builder.build()

        override2_repo.register_sub(op2, override2_op2_sub())

        override3_repo = SimpleSubRepository()

        def override3_op3_sub() -> Sub:
            builder = SubBuilder(1)
            (q,) = builder.qubits
            builder.add_op(X, (q,))
            builder.add_op(Y, (q,))
            return builder.build()

        override3_repo.register_sub(op3, override3_op3_sub())

        child_composite = override2_repo.with_override(override3_repo)
        assert isinstance(child_composite, CompositeSubRepository)

        # Create final composite where BOTH parent and child are Composite
        final_composite = parent_composite.with_override(child_composite)

        assert isinstance(final_composite, CompositeSubRepository)
        assert isinstance(final_composite.parent_repo, CompositeSubRepository)
        assert isinstance(final_composite.child_repo, CompositeSubRepository)

        # Verify resolution works correctly
        # op3 should come from override3_repo (child's child)
        resolver3 = final_composite.find_resolver(op3)
        assert resolver3 is not None
        assert resolver3(op3, final_composite) == override3_op3_sub()

        # op2 should come from override2_repo (child's parent)
        resolver2 = final_composite.find_resolver(op2)
        assert resolver2 is not None
        assert resolver2(op2, final_composite) == override2_op2_sub()

        # op1 should come from override1_repo (parent's child)
        resolver1 = final_composite.find_resolver(op1)
        assert resolver1 is not None
        assert resolver1(op1, final_composite) == override1_op1_sub()

        # op0 should come from override0_repo (parent's parent)
        resolver0 = final_composite.find_resolver(op0)
        assert resolver0 is not None
        assert resolver0(op0, final_composite) == override0_op0_sub()

    def test_simulator_repository_resolves_multicontrolled(self) -> None:
        repo = simulator_repository()

        # Test simple MultiControlled operation
        mc_x = MultiControlled(X, 2, 0b11)
        resolver = repo.find_resolver(mc_x)
        assert resolver is not None
        sub = resolver(mc_x, repo)
        assert sub is not None
        assert sub.operations == ((MCX(2), sub.qubits, ()),)

        pauli = Pauli((1, 1))
        resolver = repo.find_resolver(pauli)
        assert resolver is not None
        sub = resolver(pauli, repo)
        assert sub is not None
        assert sub.operations == (
            (X, (sub.qubits[0],), ()),
            (X, (sub.qubits[1],), ()),
        )


class TestCollectSubs:
    def test_collect_subs(self) -> None:
        op1 = Op(Ident(NS, "op1"), 1)
        op2 = Op(Ident(NS, "op2"), 2)
        op3 = Op(Ident(NS, "op3"), 3)

        repository = SimpleSubRepository()

        # op3 uses op2 and X
        def op3_sub() -> Sub:
            builder = SubBuilder(3)
            q0, q1, q2 = builder.qubits
            builder.add_op(op2, (q0, q1))
            builder.add_op(X, (q2,))
            return builder.build()

        repository.register_sub(op3, op3_sub())

        # op2 uses X
        def op2_sub() -> Sub:
            builder = SubBuilder(2)
            q0, q1 = builder.qubits
            builder.add_op(X, (q0,))
            builder.add_op(X, (q1,))
            return builder.build()

        repository.register_sub(op2, op2_sub())

        # op1 uses X
        def op1_sub() -> Sub:
            builder = SubBuilder(1)
            (q,) = builder.qubits
            builder.add_op(X, (q,))
            return builder.build()

        repository.register_sub(op1, op1_sub())

        collector = SubCollector(repository)
        collected_subs = collector.collect_subs(op3)
        assert collected_subs == {op3: op3_sub(), op2: op2_sub()}

    def test_collect_subs_parametric(self) -> None:
        op = Op(Ident(NS, "op"), 1)

        indexed_op1: OpFactory[int] = SimpleParamOp((NS, "indexed_op1"), 1)

        indexed_op2: OpFactory[int] = SimpleParamOp((NS, "indexed_op2"), 1)

        repository = SimpleSubRepository()

        # op uses indexed_op1(2), indexed_op1(4)
        def op_sub() -> Sub:
            builder = SubBuilder(1)
            (q,) = builder.qubits
            builder.add_op(indexed_op1(2), (q,))
            builder.add_op(indexed_op1(4), (q,))
            return builder.build()

        repository.register_sub(op, op_sub())

        # indexed_op1 uses indexed_op2(2*index), indexed_op2(3*index)
        def indexed_op1_sub(index: int) -> Sub:
            builder = SubBuilder(1)
            (q,) = builder.qubits
            builder.add_op(indexed_op2(2 * index), (q,))
            if index < 3:
                builder.add_op(indexed_op2(3 * index), (q,))
            return builder.build()

        repository.register_sub(indexed_op1, indexed_op1_sub)

        # indexed_op2 uses indexed_op(index/2) if index is even, X if index is odd
        def indexed_op2_sub(index: int) -> Sub:
            builder = SubBuilder(1)
            (q,) = builder.qubits
            if index % 2 == 0:
                builder.add_op(indexed_op2(index // 2), (q,))
            else:
                builder.add_op(X, (q,))
            return builder.build()

        repository.register_sub(indexed_op2, indexed_op2_sub)

        collector = SubCollector(repository)
        collected_subs = collector.collect_subs(op)
        assert collected_subs == {
            op: op_sub(),
            # used in op
            indexed_op1(2): indexed_op1_sub(2),
            indexed_op1(4): indexed_op1_sub(4),
            # used in op1(2)
            indexed_op2(4): indexed_op2_sub(4),
            indexed_op2(6): indexed_op2_sub(6),
            # used in op2(4)
            indexed_op2(2): indexed_op2_sub(2),
            indexed_op2(1): indexed_op2_sub(1),
            #  used in op2(6)
            indexed_op2(3): indexed_op2_sub(3),
            # used in op1(4)
            indexed_op2(8): indexed_op2_sub(8),
        }

    def test_collect_subs_custom_resolver(self) -> None:
        op = Op(Ident(NS, "op"), 2)
        op1 = Op(Ident(NS, "op1"), 1)
        control: OpFactory[Op] = SimpleParamOp((NS, "control"), 2)

        repository = SimpleSubRepository()

        # op uses control_op1
        def op_sub() -> Sub:
            builder = SubBuilder(2)
            q0, q1 = builder.qubits
            builder.add_op(control(op1), (q0, q1))
            return builder.build()

        repository.register_sub(op, op_sub())

        # op1 contains two Y gates
        def op1_sub() -> Sub:
            builder = SubBuilder(1)
            (q,) = builder.qubits
            builder.add_op(Y, (q,))
            builder.add_op(Y, (q,))
            return builder.build()

        repository.register_sub(op1, op1_sub())

        # control custom resolver
        def control_sub_resolver(op: Op, repository: SubRepository) -> Optional[Sub]:
            target_op = cast(Op, op.id.params[0])
            op_sub_resolver = repository.find_resolver(target_op)
            if not op_sub_resolver:
                return None
            op_sub = op_sub_resolver(target_op, repository)
            if not op_sub:
                return None

            builder = SubBuilder(2)
            q0, q1 = builder.qubits
            for op, _, _ in op_sub.operations:
                builder.add_op(control(op), (q0, q1))
            return builder.build()

        repository.register_sub_resolver(control, control_sub_resolver)

        # control op1 (for expectation)
        def control_op1_sub() -> Sub:
            builder = SubBuilder(2)
            (q0, q1) = builder.qubits
            builder.add_op(control(Y), (q0, q1))
            builder.add_op(control(Y), (q0, q1))
            return builder.build()

        collector = SubCollector(repository)
        collected_subs = collector.collect_subs(op)
        assert collected_subs == {
            op: op_sub(),
            # used in op
            control(op1): control_op1_sub(),
        }
