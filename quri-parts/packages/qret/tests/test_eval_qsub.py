import quri_parts.qsub.lib.std as std
from quri_parts.qret.convert_qsub import QRetEvaluatorHooks, create_module_from_qsub_sub
from quri_parts.qsub.compile import compile_sub
from quri_parts.qsub.evaluate import Evaluator
from quri_parts.qsub.sub import SubBuilder


def test_create_module_from_qsub_sub() -> None:
    builder = SubBuilder(2)
    q0, q1 = builder.qubits
    aux = builder.add_aux_qubit()
    builder.add_op(std.H, (q0,))
    builder.add_op(std.CNOT, (q0, q1))
    builder.add_op(std.CNOT, (q0, aux))
    sub = builder.build()

    module = create_module_from_qsub_sub(sub)
    circuits = module.get_circuit_list()

    assert len(circuits) == 1
    assert "__qsub_sub__" in circuits


def test_qret_evaluator_hooks_on_machinesub() -> None:
    builder = SubBuilder(2, 1)
    q0, q1 = builder.qubits
    (r0,) = builder.registers
    builder.add_op(std.H, (q0,))
    builder.add_op(std.CNOT, (q0, q1))
    builder.add_op(std.M, (q1,), (r0,))
    sub = builder.build()

    msub = compile_sub(sub, primitives=(std.H, std.CNOT, std.M))
    module = Evaluator(QRetEvaluatorHooks("custom_entry")).run(msub)
    circuits = module.get_circuit_list()

    assert len(circuits) == 1
    assert "custom_entry" in circuits
