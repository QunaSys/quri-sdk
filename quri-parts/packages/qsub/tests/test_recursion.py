import pytest

from quri_parts.qsub.codegen import CodeGenerator
from quri_parts.qsub.eval.gatecount import GateCountEvaluatorHooks
from quri_parts.qsub.eval.qubitcount import AuxQubitCountEvaluatorHooks
from quri_parts.qsub.evaluate import Evaluator
from quri_parts.qsub.expand import full_expand
from quri_parts.qsub.link import Linker
from quri_parts.qsub.machineinst import MachineSub, MachineSubRecursionError
from quri_parts.qsub.namespace import NameSpace
from quri_parts.qsub.op import Ident, Op
from quri_parts.qsub.sub import SubBuilder


def _recursive_msub_with_primitives() -> tuple[MachineSub, set[Op]]:
    NS = NameSpace("test")
    F = Op(Ident(NS, "F"), 1)
    G = Op(Ident(NS, "G"), 2)
    H = Op(Ident(NS, "H"), 1)

    fb = SubBuilder(2)
    (q0, q1) = fb.qubits
    fb.add_op(H, (q0,))
    fb.add_op(G, (q1,))
    fsub = fb.build()

    gb = SubBuilder(1)
    (gq0,) = gb.qubits
    ga0 = gb.add_aux_qubit()
    gb.add_op(H, (gq0,))
    gb.add_op(
        F,
        (
            gq0,
            ga0,
        ),
    )
    gsub = gb.build()

    primitives = {H}
    codegen = CodeGenerator(primitives)
    msub = codegen.lower(fsub)

    calltable = {
        F: codegen.lower(fsub),
        G: codegen.lower(gsub),
    }
    linked_msub = Linker(calltable).link(msub)

    return linked_msub, primitives


def test_execute_recursive_msub() -> None:
    linked_msub, primitives = _recursive_msub_with_primitives()

    with pytest.raises(MachineSubRecursionError):
        h_count_hook = GateCountEvaluatorHooks(primitives)
        Evaluator(h_count_hook).run(linked_msub)

    with pytest.raises(MachineSubRecursionError):
        aux_qubit_hook = AuxQubitCountEvaluatorHooks()
        Evaluator(aux_qubit_hook).run(linked_msub)


def test_expand_recursive_msub() -> None:
    linked_msub, primitives = _recursive_msub_with_primitives()

    with pytest.raises(MachineSubRecursionError):
        full_expand(linked_msub)
