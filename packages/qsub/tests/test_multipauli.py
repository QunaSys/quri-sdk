import numpy as np

from quri_parts.circuit import gates
from quri_parts.qsub.compile import compile_sub
from quri_parts.qsub.eval.quriparts import QURIPartsEvaluatorHooks
from quri_parts.qsub.evaluate import Evaluator
from quri_parts.qsub.lib import std
from quri_parts.qsub.sub import SubBuilder


class TestMultiPauliOpSub:
    def test_resolve_pauli(self) -> None:
        builder = SubBuilder(3)
        qs = builder.qubits
        builder.add_op(std.Pauli((3, 1, 2)), (qs[1], qs[0], qs[2]))
        sub = builder.build()

        circ = Evaluator(QURIPartsEvaluatorHooks()).run(
            compile_sub(sub, primitives=(std.X, std.Y, std.Z))
        )
        assert list(circ.gates) == [
            gates.Z(1),
            gates.X(0),
            gates.Y(2),
        ]

    def test_resolve_pauli_single(self) -> None:
        builder = SubBuilder(1)
        qs = builder.qubits
        builder.add_op(std.Pauli((2,)), (qs[0],))
        sub = builder.build()

        circ = Evaluator(QURIPartsEvaluatorHooks()).run(
            compile_sub(sub, primitives=(std.X, std.Y, std.Z))
        )
        assert list(circ.gates) == [gates.Y(0)]

    def test_resolve_paulirotation(self) -> None:
        theta = np.pi / 4.0

        builder = SubBuilder(3)
        qs = builder.qubits
        builder.add_op(std.PauliRotation((2, 3, 1), theta), (qs[0], qs[2], qs[1]))
        sub = builder.build()

        circ = Evaluator(QURIPartsEvaluatorHooks()).run(
            compile_sub(
                sub, primitives=(std.H, std.SqrtX, std.SqrtXdag, std.RZ, std.CNOT)
            )
        )
        assert list(circ.gates) == [
            gates.SqrtX(0),
            gates.H(1),
            gates.CNOT(1, 0),
            gates.CNOT(2, 0),
            gates.RZ(0, theta),
            gates.CNOT(2, 0),
            gates.CNOT(1, 0),
            gates.SqrtXdag(0),
            gates.H(1),
        ]

    def test_resolve_paulirotation_single(self) -> None:
        theta = np.pi / 4.0

        builder = SubBuilder(1)
        qs = builder.qubits
        builder.add_op(std.PauliRotation((2,), theta), (qs[0],))
        sub = builder.build()

        circ = Evaluator(QURIPartsEvaluatorHooks()).run(
            compile_sub(
                sub, primitives=(std.H, std.SqrtX, std.SqrtXdag, std.RZ, std.CNOT)
            )
        )
        assert list(circ.gates) == [
            gates.SqrtX(0),
            gates.RZ(0, theta),
            gates.SqrtXdag(0),
        ]
