from collections.abc import Sequence

from quri_parts.qsub.compile import compile, compile_sub
from quri_parts.qsub.eval import SubwiseProfileEntry, SubwiseProfileEvaluatorHooks
from quri_parts.qsub.evaluate import Evaluator
from quri_parts.qsub.lib import std
from quri_parts.qsub.opsub import ParamUnitarySubDef, param_opsub
from quri_parts.qsub.primitive import AllBasicSet
from quri_parts.qsub.register import QRegSpec
from quri_parts.qsub.sub import SubBuilder

_PRIMITIVES = (std.CNOT, std.Toffoli)


def _by_name(
    report: list[SubwiseProfileEntry],
) -> dict[str, SubwiseProfileEntry]:
    return {e.name: e for e in report if e.name is not None}


def test_repeated_same_sub() -> None:
    """A sub called twice must report call_count=2 and aggregate = 2x per-call."""
    MCX = _buggy_mcx()

    root_b = SubBuilder(3)
    rq = root_b.qubits
    root_b.add_op(MCX(2), rq)
    root_b.add_op(MCX(2), rq)
    root = root_b.build()

    compiled = compile_sub(root, _PRIMITIVES)
    report = Evaluator(SubwiseProfileEvaluatorHooks(gates=(std.Toffoli,))).run(compiled)

    leaf_entry = _by_name(report)["MCX(2)"]
    assert leaf_entry.call_count == 2
    assert leaf_entry.gate_count_info["self_per_call"] == {"Toffoli": 1}
    assert leaf_entry.gate_count_info["self"] == {"Toffoli": 2}
    assert leaf_entry.gate_count_info["descendants_per_call"] == {}
    assert leaf_entry.gate_count_info["total"] == {"Toffoli": 2}

    # The root of compile_sub has no source op, so it carries name=None.
    (root_entry,) = [e for e in report if e.name is None]
    assert root_entry.call_count == 1
    assert root_entry.gate_count_info["self_per_call"] == {}
    assert root_entry.gate_count_info["descendants_per_call"] == {"Toffoli": 2}
    assert root_entry.gate_count_info["total_per_call"] == {"Toffoli": 2}


def test_nested_parametric_mcx() -> None:
    """End-to-end against the subwise_mcx tutorial's buggy MCX numbers."""
    MCX = _buggy_mcx()
    compiled = compile(MCX(6), _PRIMITIVES)
    report = Evaluator(SubwiseProfileEvaluatorHooks(gates=(std.Toffoli,))).run(compiled)

    assert [e.name for e in report] == [
        "MCX(6)",
        "MCX(5)",
        "MCX(4)",
        "MCX(3)",
        "MCX(2)",
    ]

    expected_call_count = {6: 1, 5: 2, 4: 4, 3: 8, 2: 16}
    expected_total_per_call = {6: 31, 5: 15, 4: 7, 3: 3, 2: 1}
    expected_descendants_per_call = {6: 30, 5: 14, 4: 6, 3: 2}
    entries = _by_name(report)
    for n in range(2, 7):
        e = entries[f"MCX({n})"]
        assert e.call_count == expected_call_count[n]
        assert e.qubit_count_info == {"data": n + 1, "auxiliary": 0, "total": n + 1}
        assert e.gate_count_info["self_per_call"] == {"Toffoli": 1}
        assert e.gate_count_info["total_per_call"] == {
            "Toffoli": expected_total_per_call[n]
        }
        if n in expected_descendants_per_call:
            assert e.gate_count_info["descendants_per_call"] == {
                "Toffoli": expected_descendants_per_call[n]
            }
        else:
            assert e.gate_count_info["descendants_per_call"] == {}
        # aggregate == per_call x call_count
        assert e.gate_count_info["total"] == {
            "Toffoli": expected_total_per_call[n] * expected_call_count[n]
        }


def test_fixed_mcx_grows_linearly() -> None:
    """The corrected (single-recursion) MCX: every sub called once, descendants
    shrink by one per level."""
    MCX = _fixed_mcx()
    compiled = compile(MCX(6), _PRIMITIVES)
    report = Evaluator(SubwiseProfileEvaluatorHooks(gates=(std.Toffoli,))).run(compiled)

    entries = _by_name(report)
    for n in range(2, 7):
        e = entries[f"MCX({n})"]
        assert e.call_count == 1
        assert e.gate_count_info["total_per_call"] == {"Toffoli": n - 1}
        if n > 2:
            assert e.gate_count_info["descendants_per_call"] == {"Toffoli": n - 2}
        else:
            assert e.gate_count_info["descendants_per_call"] == {}


def test_gate_filter() -> None:
    """Empty gates counts every primitive; a filter restricts the breakdown."""
    b = SubBuilder(3)
    q = b.qubits
    b.add_op(std.CNOT, (q[0], q[1]))
    b.add_op(std.Toffoli, (q[0], q[1], q[2]))
    compiled = compile_sub(b.build(), AllBasicSet)

    (all_gates,) = Evaluator(SubwiseProfileEvaluatorHooks()).run(compiled)
    assert all_gates.gate_count_info["self_per_call"] == {"CNOT": 1, "Toffoli": 1}

    (toffoli_only,) = Evaluator(SubwiseProfileEvaluatorHooks(gates=(std.Toffoli,))).run(
        compiled
    )
    assert toffoli_only.gate_count_info["self_per_call"] == {"Toffoli": 1}


def _buggy_mcx():  # type: ignore[no-untyped-def]
    class _MCX(ParamUnitarySubDef[int]):
        name = "MCX"

        def qubit_count_fn(self, n: int) -> int:
            return n + 1

        def qregs_fn(self, n: int) -> Sequence[QRegSpec]:
            return (QRegSpec("ctrl", n), QRegSpec("tgt", 1))

        def sub(self, b: SubBuilder, n: int) -> None:
            ctrls = b.qregs["ctrl"].qubits
            (tgt,) = b.qregs["tgt"].qubits
            if n == 1:
                b.add_op(std.CNOT, (ctrls[0], tgt))
                return
            if n == 2:
                b.add_op(std.Toffoli, (ctrls[0], ctrls[1], tgt))
                return
            b.add_op(MCX(n - 1), (*ctrls[:-1], tgt))
            b.add_op(std.Toffoli, (ctrls[-2], ctrls[-1], tgt))
            b.add_op(MCX(n - 1), (*ctrls[:-1], tgt))

    MCX, _ = param_opsub(_MCX)
    return MCX


def _fixed_mcx():  # type: ignore[no-untyped-def]
    class _MCX(ParamUnitarySubDef[int]):
        name = "MCX"

        def qubit_count_fn(self, n: int) -> int:
            return n + 1

        def qregs_fn(self, n: int) -> Sequence[QRegSpec]:
            return (QRegSpec("ctrl", n), QRegSpec("tgt", 1))

        def sub(self, b: SubBuilder, n: int) -> None:
            ctrls = b.qregs["ctrl"].qubits
            (tgt,) = b.qregs["tgt"].qubits
            if n == 1:
                b.add_op(std.CNOT, (ctrls[0], tgt))
                return
            if n == 2:
                b.add_op(std.Toffoli, (ctrls[0], ctrls[1], tgt))
                return
            b.add_op(MCX(n - 1), (*ctrls[:-1], tgt))
            b.add_op(std.Toffoli, (ctrls[-2], ctrls[-1], tgt))

    MCX, _ = param_opsub(_MCX)
    return MCX
