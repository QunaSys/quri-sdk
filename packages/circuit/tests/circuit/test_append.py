from typing import Any, Sequence

import pytest

from quri_parts.circuit import (
    ImmutableLinearMappedUnboundParametricQuantumCircuit,
    ImmutableParametricQuantumCircuit,
    LinearMappedUnboundParametricQuantumCircuit,
    ParametricQuantumCircuit,
    QuantumCircuit,
    QuantumGate,
    X,
    Z,
)


def gen_quantum_circuit() -> QuantumCircuit:
    a = QuantumCircuit(1)
    a.add_X_gate(0)
    a.add_Z_gate(0)
    return a


def gen_seq_of_gate() -> Sequence[QuantumGate]:
    return [X(0), Z(0)]


def gen_parametric_quantum_circuit() -> ParametricQuantumCircuit:
    a = ParametricQuantumCircuit(1)
    a.add_X_gate(0)
    a.add_Z_gate(0)
    return a


def gen_immutable_parametric_quantum_circuit() -> ImmutableParametricQuantumCircuit:
    a = gen_parametric_quantum_circuit()
    return a.freeze()


def gen_linear_mapped() -> LinearMappedUnboundParametricQuantumCircuit:
    a = LinearMappedUnboundParametricQuantumCircuit(1)
    a.add_X_gate(0)
    a.add_Z_gate(0)
    return a


def gen_immutable_linear_mapped() -> (
    ImmutableLinearMappedUnboundParametricQuantumCircuit
):
    a = gen_linear_mapped()
    return a.freeze()


@pytest.mark.parametrize(
    "lhs,rhs",
    [
        (gen_quantum_circuit, gen_seq_of_gate),
        (gen_parametric_quantum_circuit, gen_seq_of_gate),
        (gen_seq_of_gate, gen_parametric_quantum_circuit),
        (gen_immutable_parametric_quantum_circuit, gen_seq_of_gate),
        (gen_seq_of_gate, gen_immutable_parametric_quantum_circuit),
        (gen_parametric_quantum_circuit, gen_linear_mapped),
        (gen_parametric_quantum_circuit, gen_immutable_linear_mapped),
        (gen_immutable_parametric_quantum_circuit, gen_linear_mapped),
        (gen_immutable_parametric_quantum_circuit, gen_immutable_linear_mapped),
        (gen_linear_mapped, gen_parametric_quantum_circuit),
        (gen_linear_mapped, gen_immutable_parametric_quantum_circuit),
        (gen_immutable_linear_mapped, gen_parametric_quantum_circuit),
        (gen_immutable_linear_mapped, gen_immutable_parametric_quantum_circuit),
    ],
)
def test_add(lhs: Any, rhs: Any) -> None:
    a = lhs()
    b = rhs()
    c = a + b
    assert len(c.gates) == 4
    if isinstance(
        b,
        (
            LinearMappedUnboundParametricQuantumCircuit,
            ImmutableLinearMappedUnboundParametricQuantumCircuit,
        ),
    ):
        assert len(b.gates) == 2
    else:
        assert b == rhs()
    if isinstance(
        b,
        (
            LinearMappedUnboundParametricQuantumCircuit,
            ImmutableLinearMappedUnboundParametricQuantumCircuit,
        ),
    ):
        assert len(b.gates) == 2
    else:
        assert b == rhs()


@pytest.mark.parametrize(
    "lhs,rhs",
    [
        (gen_quantum_circuit, gen_seq_of_gate),
        (gen_parametric_quantum_circuit, gen_seq_of_gate),
        (gen_parametric_quantum_circuit, gen_linear_mapped),
        (gen_parametric_quantum_circuit, gen_immutable_linear_mapped),
    ],
)
def test_iadd(lhs: Any, rhs: Any) -> None:
    a = lhs()
    b = rhs()
    a += b
    assert len(a.gates) == 4
