from collections.abc import Sequence

from quri_parts.qsub.codegen import CodeGenerator
from quri_parts.qsub.compile import compile
from quri_parts.qsub.eval import SubwiseProfileEntry, SubwiseProfileEvaluatorHooks
from quri_parts.qsub.evaluate import Evaluator
from quri_parts.qsub.lib import std
from quri_parts.qsub.link import Linker
from quri_parts.qsub.namespace import NameSpace
from quri_parts.qsub.op import Ident, Op
from quri_parts.qsub.opsub import ParamUnitarySubDef, param_opsub
from quri_parts.qsub.register import QRegSpec
from quri_parts.qsub.sub import SubBuilder


def _named(report: list[SubwiseProfileEntry]) -> dict[str, SubwiseProfileEntry]:
    return {e.name: e for e in report if e.name is not None}


def _root(report: list[SubwiseProfileEntry]) -> SubwiseProfileEntry:
    (root,) = [e for e in report if e.name is None]
    return root


def test_call_count_and_gate_scopes() -> None:
    """root -> G (x2) -> H (x2), each of G and H holding one CNOT.

    Exercises call-count multiplication through nesting and the
    self/descendants/total split, per-call and aggregated.
    """
    ns = NameSpace("test_scopes")
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

    report = Evaluator(SubwiseProfileEvaluatorHooks(gates=(std.CNOT,))).run(linked)
    named = _named(report)

    h = named["H"]
    assert h.call_count == 4  # 2 (per G) x 2 (G calls)
    assert h.gate_count_info["self_per_call"] == {"CNOT": 1}
    assert h.gate_count_info["descendants_per_call"] == {}
    assert h.gate_count_info["self"] == {"CNOT": 4}

    g = named["G"]
    assert g.call_count == 2
    assert g.gate_count_info["self_per_call"] == {"CNOT": 1}
    assert g.gate_count_info["descendants_per_call"] == {"CNOT": 2}
    assert g.gate_count_info["total_per_call"] == {"CNOT": 3}
    assert g.gate_count_info["total"] == {"CNOT": 6}  # 3 x call_count 2

    root = _root(report)
    assert root.call_count == 1
    assert root.gate_count_info["self_per_call"] == {}
    assert root.gate_count_info["descendants_per_call"] == {"CNOT": 6}


def test_shared_sub_across_parents() -> None:
    """root -> A -> C, root -> B -> C, root -> C directly: C keeps a single row
    with call counts summed over all parents and depths."""
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

    report = Evaluator(SubwiseProfileEvaluatorHooks(gates=(std.X,))).run(linked)
    named = _named(report)

    assert named["C"].call_count == 3
    assert named["A"].gate_count_info["descendants_per_call"] == {"X": 1}
    assert named["B"].gate_count_info["descendants_per_call"] == {"X": 1}
    assert _root(report).gate_count_info["descendants"] == {"X": 3}


def test_qubit_count_info_with_auxiliary() -> None:
    """auxiliary is the peak ancilla for one invocation, summed with a callee's
    own ancilla when they are live at the same time."""
    ns = NameSpace("test_aux")
    F = Op.from_qubit_count(Ident(ns, "F"), 1)

    fb = SubBuilder(1)
    (fq0,) = fb.qubits
    fa0 = fb.add_aux_qubit()
    fb.add_op(std.X, (fq0,))
    fb.add_op(std.X, (fa0,))
    fsub = fb.build()

    rb = SubBuilder(2)
    (rq0, _rq1) = rb.qubits
    rb.add_aux_qubit()
    rb.add_op(F, (rq0,))

    codegen = CodeGenerator([std.X])
    msub = codegen.lower(rb.build())
    linked = Linker({F: codegen.lower(fsub)}).link(msub)

    report = Evaluator(SubwiseProfileEvaluatorHooks()).run(linked)

    f = _named(report)["F"]
    assert f.qubit_count_info == {"data": 1, "auxiliary": 1, "total": 2}
    assert f.gate_count_info["self_per_call"] == {"X": 2}

    root = _root(report)
    # 2 data + 1 local aux + 1 aux live inside F = 4
    assert root.qubit_count_info == {"data": 2, "auxiliary": 2, "total": 4}


def test_auxiliary_peak_over_siblings() -> None:
    """Sequential callees share ancillas, so a parent's auxiliary is the max
    over its callees, not their sum."""
    ns = NameSpace("test_aux_peak")
    F1 = Op.from_qubit_count(Ident(ns, "F1"), 1)
    F2 = Op.from_qubit_count(Ident(ns, "F2"), 1)

    f1b = SubBuilder(1)
    f1b.add_op(std.X, (f1b.add_aux_qubit(),))
    f1sub = f1b.build()

    f2b = SubBuilder(1)
    f2b.add_op(std.X, (f2b.add_aux_qubit(),))
    f2b.add_op(std.X, (f2b.add_aux_qubit(),))
    f2sub = f2b.build()

    rb = SubBuilder(1)
    rb.add_op(F1, rb.qubits)
    rb.add_op(F2, rb.qubits)

    codegen = CodeGenerator([std.X])
    linked = Linker({F1: codegen.lower(f1sub), F2: codegen.lower(f2sub)}).link(
        codegen.lower(rb.build())
    )

    report = Evaluator(SubwiseProfileEvaluatorHooks()).run(linked)
    assert _root(report).qubit_count_info["auxiliary"] == 2  # max(1, 2)


def test_gate_filter() -> None:
    """Empty gates counts every primitive; a filter restricts the breakdown."""
    b = SubBuilder(3)
    q = b.qubits
    b.add_op(std.CNOT, (q[0], q[1]))
    b.add_op(std.Toffoli, (q[0], q[1], q[2]))
    msub = CodeGenerator([std.CNOT, std.Toffoli]).lower(b.build())

    (all_gates,) = Evaluator(SubwiseProfileEvaluatorHooks()).run(msub)
    assert all_gates.gate_count_info["self_per_call"] == {"CNOT": 1, "Toffoli": 1}

    (toffoli_only,) = Evaluator(SubwiseProfileEvaluatorHooks(gates=(std.Toffoli,))).run(
        msub
    )
    assert toffoli_only.gate_count_info["self_per_call"] == {"Toffoli": 1}


def test_gate_name_collision_across_namespaces() -> None:
    """Counted gates sharing a local name across namespaces stay distinct,
    keyed with their namespace instead of silently merging."""
    ns = NameSpace("custom")
    MyX = Op.from_qubit_count(Ident(ns, "X"), 1)

    b = SubBuilder(1)
    b.add_op(std.X, b.qubits)
    b.add_op(MyX, b.qubits)
    b.add_op(MyX, b.qubits)
    msub = CodeGenerator([std.X, MyX]).lower(b.build())

    (entry,) = Evaluator(SubwiseProfileEvaluatorHooks()).run(msub)
    std_ns = std.X.base_id[0]
    assert entry.gate_count_info["self_per_call"] == {
        f"{std_ns}.X": 1,
        "custom.X": 2,
    }


def test_parametric_sub_is_named_via_compile() -> None:
    """compile() attaches the entry op, so the root row is named with its
    parameter (unlike the raw lower/link path, where the root is anonymous)."""

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

    (entry,) = Evaluator(SubwiseProfileEvaluatorHooks(gates=(std.Toffoli,))).run(
        compiled
    )
    assert entry.name == "Twice(2)"
    assert entry.call_count == 1
    assert entry.gate_count_info["self_per_call"] == {"Toffoli": 2}


def test_parametric_sub_row_per_parameter() -> None:
    """A self-recursive parametric sub gets one row per parameter value."""

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

    report = Evaluator(SubwiseProfileEvaluatorHooks(gates=(std.X,))).run(compiled)
    named = _named(report)

    assert set(named) == {"Chain(3)", "Chain(2)", "Chain(1)"}
    assert all(e.call_count == 1 for e in named.values())
    assert named["Chain(3)"].gate_count_info["total_per_call"] == {"X": 3}
