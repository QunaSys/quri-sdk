from collections.abc import Mapping, Sequence

from quri_parts.qsub.codegen import CodeGenerator
from quri_parts.qsub.compile import compile, compile_sub
from quri_parts.qsub.eval import CallGraphEvaluatorHooks
from quri_parts.qsub.evaluate import Evaluator
from quri_parts.qsub.lib import std
from quri_parts.qsub.link import Linker
from quri_parts.qsub.machineinst import MachineSub
from quri_parts.qsub.namespace import NameSpace
from quri_parts.qsub.op import AbstractOp, Ident, Op
from quri_parts.qsub.opsub import ParamUnitarySubDef, param_opsub
from quri_parts.qsub.register import QRegSpec
from quri_parts.qsub.resolve import SimpleSubRepository
from quri_parts.qsub.sub import Sub, SubBuilder


def _link(
    primitives: Sequence[AbstractOp], entry: tuple[Op, Sub], callees: Mapping[Op, Sub]
) -> MachineSub:
    """Lower and link hand-built subs, naming each after its op as compile
    does."""
    codegen = CodeGenerator(primitives)
    entry_op, entry_sub = entry
    lowered = {op: codegen.lower(sub, op.id) for op, sub in callees.items()}
    return Linker(lowered).link(codegen.lower(entry_sub, entry_op.id))


def test_nested_calls_count_repeats() -> None:
    """R -> G (x2) -> H (x2): repeated calls of the same sub contribute a
    single edge carrying the per-invocation call count, and primitives do not
    appear as nodes."""
    ns = NameSpace("test_nested")
    R = Op.from_qubit_count(Ident(ns, "R"), 2)
    G = Op.from_qubit_count(Ident(ns, "G"), 2)
    H = Op.from_qubit_count(Ident(ns, "H"), 2)

    hb = SubBuilder(2)
    hb.add_op(std.CNOT, hb.qubits)

    gb = SubBuilder(2)
    gb.add_op(std.CNOT, gb.qubits)
    gb.add_op(H, gb.qubits)
    gb.add_op(H, gb.qubits)

    rb = SubBuilder(2)
    rb.add_op(G, rb.qubits)
    rb.add_op(G, rb.qubits)

    linked = _link([std.CNOT], (R, rb.build()), {G: gb.build(), H: hb.build()})

    graph = Evaluator(CallGraphEvaluatorHooks()).run(linked)
    assert graph == {R.id: {G.id: 2}, G.id: {H.id: 2}, H.id: {}}


def test_shared_sub_across_parents() -> None:
    """R -> A -> C, R -> B -> C, R -> C directly: C stays a single node and
    callees keep instruction order."""
    ns = NameSpace("test_shared")
    R = Op.from_qubit_count(Ident(ns, "R"), 1)
    A = Op.from_qubit_count(Ident(ns, "A"), 1)
    B = Op.from_qubit_count(Ident(ns, "B"), 1)
    C = Op.from_qubit_count(Ident(ns, "C"), 1)

    cb = SubBuilder(1)
    cb.add_op(std.X, cb.qubits)

    ab = SubBuilder(1)
    ab.add_op(C, ab.qubits)

    bb = SubBuilder(1)
    bb.add_op(C, bb.qubits)

    rb = SubBuilder(1)
    rb.add_op(A, rb.qubits)
    rb.add_op(B, rb.qubits)
    rb.add_op(C, rb.qubits)

    linked = _link(
        [std.X], (R, rb.build()), {A: ab.build(), B: bb.build(), C: cb.build()}
    )

    graph = Evaluator(CallGraphEvaluatorHooks()).run(linked)
    assert graph == {
        R.id: {A.id: 1, B.id: 1, C.id: 1},
        A.id: {C.id: 1},
        B.id: {C.id: 1},
        C.id: {},
    }
    # Nodes are reported in first-entry order, which is depth-first: C is
    # reached through A before B is entered. Dict equality above ignores it.
    assert list(graph) == [R.id, A.id, C.id, B.id]


def test_root_is_named_via_compile() -> None:
    """compile() resolves the entry op, so the root node carries its source op
    with its parameter."""

    class _Twice(ParamUnitarySubDef[int]):
        name = "Twice"

        def qubit_count_fn(self, k: int) -> int:
            return 3

        def qregs_fn(self, k: int) -> Sequence[QRegSpec]:
            return (QRegSpec("q", 3),)

        def sub(self, b: SubBuilder, k: int) -> None:
            qs = b.qregs["q"].qubits
            for _ in range(k):
                b.add_op(std.Toffoli, (qs[0], qs[1], qs[2]))

    Twice, _ = param_opsub(_Twice)
    compiled = compile(Twice(2), (std.Toffoli,))

    graph = Evaluator(CallGraphEvaluatorHooks()).run(compiled)
    assert graph == {Twice(2).id: {}}


def test_entry_sub_of_compile_sub_is_anonymous() -> None:
    """compile_sub() receives a Sub rather than an Op, so its entry node is
    keyed by None while the subs it calls are still named."""
    ns = NameSpace("test_compile_sub")
    A = Op.from_qubit_count(Ident(ns, "A"), 1)

    ab = SubBuilder(1)
    ab.add_op(std.X, ab.qubits)
    repository = SimpleSubRepository()
    repository.register_sub(A, ab.build())

    eb = SubBuilder(1)
    eb.add_op(A, eb.qubits)
    compiled = compile_sub(eb.build(), (std.X,), repository)

    graph = Evaluator(CallGraphEvaluatorHooks()).run(compiled)
    assert graph == {None: {A.id: 1}, A.id: {}}


def test_parametric_sub_node_per_parameter() -> None:
    """A self-recursive parametric sub gets one node per parameter value."""

    class _Chain(ParamUnitarySubDef[int]):
        name = "Chain"

        def qubit_count_fn(self, k: int) -> int:
            return 1

        def qregs_fn(self, k: int) -> Sequence[QRegSpec]:
            return (QRegSpec("q", 1),)

        def sub(self, b: SubBuilder, k: int) -> None:
            (q,) = b.qregs["q"].qubits
            b.add_op(std.X, (q,))
            if k > 1:
                b.add_op(Chain(k - 1), (q,))

    Chain, _ = param_opsub(_Chain)
    compiled = compile(Chain(3), (std.X,))

    graph = Evaluator(CallGraphEvaluatorHooks()).run(compiled)
    assert graph == {
        Chain(3).id: {Chain(2).id: 1},
        Chain(2).id: {Chain(1).id: 1},
        Chain(1).id: {},
    }
