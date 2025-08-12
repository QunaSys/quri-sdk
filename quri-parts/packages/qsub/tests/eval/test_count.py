from quri_parts.qsub.codegen import CodeGenerator
from quri_parts.qsub.compile import compile_sub
from quri_parts.qsub.eval.gatecount import TGateCountEvaluatorHooks
from quri_parts.qsub.eval.qubitcount import AuxQubitCountEvaluatorHooks
from quri_parts.qsub.evaluate import Evaluator
from quri_parts.qsub.lib.std import MultiControlled, S, T, X, Z
from quri_parts.qsub.link import Linker
from quri_parts.qsub.namespace import NameSpace
from quri_parts.qsub.op import Ident, Op
from quri_parts.qsub.primitive import AllBasicSet
from quri_parts.qsub.sub import SubBuilder


def test_t_count() -> None:
    zb = SubBuilder(1)
    (zq0,) = zb.qubits
    zb.add_op(S, (zq0,))
    zb.add_op(S, (zq0,))
    zsub = zb.build()

    sb = SubBuilder(1)
    (sq0,) = sb.qubits
    sb.add_op(T, (sq0,))
    sb.add_op(T, (sq0,))
    ssub = sb.build()

    builder = SubBuilder(1)
    (q0,) = builder.qubits
    builder.add_op(Z, (q0,))
    builder.add_op(S, (q0,))
    builder.add_op(T, (q0,))
    builder.add_op(S, (q0,))
    builder.add_op(Z, (q0,))
    sub = builder.build()

    codegen = CodeGenerator([T])
    msub = codegen.lower(sub)

    calltable = {
        Z: codegen.lower(zsub),
        S: codegen.lower(ssub),
    }
    linked_msub = Linker(calltable).link(msub)

    hook = TGateCountEvaluatorHooks()
    Evaluator(hook).run(linked_msub)

    assert hook.result()[T.base_id] == 13


def test_aux_qubits() -> None:
    NS = NameSpace("test")
    F = Op(Ident(NS, "F"), 1, 0)
    G = Op(Ident(NS, "G"), 3, 0)
    H = Op(Ident(NS, "H"), 2, 0)
    K = Op(Ident(NS, "K"), 2, 0)

    fb = SubBuilder(1)
    (fq0,) = fb.qubits
    fa0, fa1 = fb.add_aux_qubit(), fb.add_aux_qubit()  # 2 aux qubits
    fb.add_op(G, (fq0, fa0, fa1))
    fb.add_op(H, (fq0, fa0))  # +1 aux qubit
    fsub = fb.build()

    hb = SubBuilder(2)
    hq0, hq1 = hb.qubits
    ha0 = hb.add_aux_qubit()  # 1 aux qubit
    hb.add_op(G, (hq0, hq1, ha0))
    hb.add_op(K, (hq1, ha0))
    hsub = hb.build()

    builder = SubBuilder(2)
    q0, q1 = builder.qubits
    a0, a1 = builder.add_aux_qubit(), builder.add_aux_qubit()  # 2 aux qubits
    builder.add_op(F, (q0,))  # +3 aux qubits
    builder.add_op(H, (q1, a0))  # +1 aux qubit
    builder.add_op(H, (a1, q0))  # +1 aux qubit
    builder.add_op(K, (q1, a0))
    sub = builder.build()

    codegen = CodeGenerator([G, K])
    msub = codegen.lower(sub)

    calltable = {
        F: codegen.lower(fsub),
        H: codegen.lower(hsub),
    }
    linked_msub = Linker(calltable).link(msub)

    hook = AuxQubitCountEvaluatorHooks()
    Evaluator(hook).run(linked_msub)

    assert hook.result() == 5


def test_aux_same_sub_bug() -> None:
    b = SubBuilder(4)
    for _ in range(10):
        b.add_op(MultiControlled(X, 3, 0b0), b.qubits)
    sub = b.build()
    msub = compile_sub(sub, AllBasicSet)

    hook = AuxQubitCountEvaluatorHooks()
    Evaluator(hook).run(msub)

    assert hook.result() == 2
