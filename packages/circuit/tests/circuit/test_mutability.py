from typing import Any

import pytest

from quri_parts.circuit import ParametricQuantumCircuit, QuantumCircuit, X


@pytest.mark.parametrize("initializer", [QuantumCircuit, ParametricQuantumCircuit])
def test_immut_to_mut(initializer: Any) -> None:
    a = initializer(3)
    a.add_X_gate(0)
    a.add_CNOT_gate(1, 2)
    b = a.freeze()
    assert a == b
    assert id(a) != id(b)
    with pytest.raises(AttributeError):
        b.add_X_gate(0)
        b.add_gate(X(0))
    assert b.depth == 1
    c = b.get_mutable_copy()
    assert a == c
    assert b == c
    c.add_X_gate(0)
    assert c.depth == 2
    assert b.depth == 1
    assert a.depth == 1
    assert a != c
    assert b != c


@pytest.mark.parametrize("initializer", [QuantumCircuit, ParametricQuantumCircuit])
def test_mut_to_mut(initializer: Any) -> None:
    a = initializer(3)
    a.add_X_gate(0)
    assert a.depth == 1
    b = a.get_mutable_copy()
    assert a == b
    b.add_X_gate(0)
    assert b.depth == 2
    assert a.depth == 1
    assert a != b


def test_bind() -> None:
    a = ParametricQuantumCircuit(3)
    a.add_X_gate(0)
    b = QuantumCircuit(3)
    b.add_X_gate(0)
    c = a.bind_parameters([])
    assert b == c
