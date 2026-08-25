from collections.abc import Sequence

from quri_parts.qsub.codegen import CodeGenerator
from quri_parts.qsub.compile import compile
from quri_parts.qsub.eval import CallGraphEvaluatorHooks
from quri_parts.qsub.evaluate import Evaluator
from quri_parts.qsub.lib import std
from quri_parts.qsub.link import Linker
from quri_parts.qsub.namespace import NameSpace
from quri_parts.qsub.op import Ident, Op
from quri_parts.qsub.opsub import ParamUnitarySubDef, param_opsub
from quri_parts.qsub.register import QRegSpec
from quri_parts.qsub.sub import SubBuilder


def test_nested_calls_dedupe_repeats() -> None:
    """root -> G (x2) -> H (x2): repeated calls of the same sub contribute a
    single edge, and primitives do not appear as nodes."""
    ns = NameSpace("test_nested")
    G = Op.from_qubit_count(Ident(ns, "G"), 2)
    H = Op.from_qubit_count(Ident(ns, "H"), 2)

    hb = SubBuilder(2)
    hb.add_op(std.CNOT, hb.qubits)
    hsub = hb.build()

    gb = SubBuilder(2)
    gb.add_op(std.CNOT, gb.qubits)
    gb.add_op(H, gb.qubits)
    gb.add_op(H, gb.qubits)
    gsub = gb.build()

    rb = SubBuilder(2)
    rb.add_op(G, rb.qubits)
    rb.add_op(G, rb.qubits)

    codegen = CodeGenerator([std.CNOT])
    msub = codegen.lower(rb.build())
    linked = Linker({G: codegen.lower(gsub), H: codegen.lower(hsub)}).link(msub)

    graph = Evaluator(CallGraphEvaluatorHooks()).run(linked)
    assert graph == {None: [G.id], G.id: [H.id], H.id: []}


def test_shared_sub_across_parents() -> None:
    """root -> A -> C, root -> B -> C, root -> C directly: C stays a single
    node and callees keep instruction order."""
    ns = NameSpace("test_shared")
    A = Op.from_qubit_count(Ident(ns, "A"), 1)
    B = Op.from_qubit_count(Ident(ns, "B"), 1)
    C = Op.from_qubit_count(Ident(ns, "C"), 1)

    cb = SubBuilder(1)
    cb.add_op(std.X, cb.qubits)
    csub = cb.build()

    ab = SubBuilder(1)
    ab.add_op(C, ab.qubits)
    asub = ab.build()

    bb = SubBuilder(1)
    bb.add_op(C, bb.qubits)
    bsub = bb.build()

    rb = SubBuilder(1)
    rb.add_op(A, rb.qubits)
    rb.add_op(B, rb.qubits)
    rb.add_op(C, rb.qubits)

    codegen = CodeGenerator([std.X])
    msub = codegen.lower(rb.build())
    linked = Linker(
        {A: codegen.lower(asub), B: codegen.lower(bsub), C: codegen.lower(csub)}
    ).link(msub)

    graph = Evaluator(CallGraphEvaluatorHooks()).run(linked)
    assert graph == {
        None: [A.id, B.id, C.id],
        A.id: [C.id],
        B.id: [C.id],
        C.id: [],
    }


def test_root_is_named_via_compile() -> None:
    """compile() resolves the entry op, so the root node carries its source op
    with its parameter (unlike the raw lower/link path, where it is None)."""

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
    assert graph == {Twice(2).id: []}


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
        Chain(3).id: [Chain(2).id],
        Chain(2).id: [Chain(1).id],
        Chain(1).id: [],
    }


def test_subs_sharing_a_local_name_stay_distinct() -> None:
    """Subs sharing a local name across namespaces are distinct nodes: the
    namespace is part of the node identity."""
    ns1 = NameSpace("ns_one")
    ns2 = NameSpace("ns_two")
    F1 = Op.from_qubit_count(Ident(ns1, "F"), 1)
    F2 = Op.from_qubit_count(Ident(ns2, "F"), 1)

    f1b = SubBuilder(1)
    f1b.add_op(std.X, f1b.qubits)
    f1sub = f1b.build()

    f2b = SubBuilder(1)
    f2b.add_op(std.X, f2b.qubits)
    f2sub = f2b.build()

    rb = SubBuilder(1)
    rb.add_op(F1, rb.qubits)
    rb.add_op(F2, rb.qubits)

    codegen = CodeGenerator([std.X])
    msub = codegen.lower(rb.build())
    linked = Linker({F1: codegen.lower(f1sub), F2: codegen.lower(f2sub)}).link(msub)

    graph = Evaluator(CallGraphEvaluatorHooks()).run(linked)
    assert graph == {
        None: [F1.id, F2.id],
        F1.id: [],
        F2.id: [],
    }
