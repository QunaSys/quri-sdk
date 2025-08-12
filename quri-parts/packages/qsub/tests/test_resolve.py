from typing import Optional, cast

from quri_parts.qsub.lib.std import H, X, Y
from quri_parts.qsub.namespace import NameSpace
from quri_parts.qsub.op import Ident, Op, OpFactory, SimpleParamOp
from quri_parts.qsub.qubit import Qubit
from quri_parts.qsub.resolve import SimpleSubResolver, SubCollector, SubRepository
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

        repository = SubRepository()

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


class TestCollectSubs:
    def test_collect_subs(self) -> None:
        op1 = Op(Ident(NS, "op1"), 1)
        op2 = Op(Ident(NS, "op2"), 2)
        op3 = Op(Ident(NS, "op3"), 3)

        repository = SubRepository()

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

        repository = SubRepository()

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

        repository = SubRepository()

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
