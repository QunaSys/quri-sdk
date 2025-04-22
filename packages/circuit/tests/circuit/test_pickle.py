import pickle

from quri_parts.circuit import (
    ParametricPauliRotation,
    ParametricQuantumCircuit,
    QuantumCircuit,
    UnitaryMatrix,
)


def test_pickle_quantum_circuit() -> None:
    a = QuantumCircuit(3)
    a.add_X_gate(0)
    assert a.depth == 1
    b = pickle.loads(pickle.dumps(a))
    assert id(a) != id(b)
    assert type(a) == type(b)  # noqa
    assert a == b
    b.add_Y_gate(0)
    assert a.depth == 1
    assert b.depth == 2
    assert a != b
    a.add_Y_gate(0)
    assert a.depth == 2
    assert a == b
    a.freeze()
    assert a == b
    b.freeze()
    c = pickle.loads(pickle.dumps(a))
    assert id(a) != id(c)
    assert a == c
    assert b == c


def test_pickle_parametric_quantum_circuit() -> None:
    a = ParametricQuantumCircuit(3)
    a.add_ParametricRX_gate(0)
    assert a.depth == 1
    b = pickle.loads(pickle.dumps(a))
    assert id(a) != id(b)
    assert type(a) == type(b)  # noqa
    assert a == b
    b.add_ParametricRY_gate(0)
    assert a.depth == 1
    assert b.depth == 2
    assert a != b
    a.add_ParametricRY_gate(0)
    assert a.depth == 2
    assert a == b
    a2 = a.freeze()
    assert a2 == b
    b2 = b.freeze()
    c = pickle.loads(pickle.dumps(a2))
    assert id(a2) != id(c)
    assert type(a2) is type(b2)
    assert type(a2) is type(c)
    assert a2 == c
    assert b2 == c


def test_pickle_quantum_gate() -> None:
    a = UnitaryMatrix([0], [[1, 0], [0, 1]])
    b = pickle.loads(pickle.dumps(a))
    assert id(a) != id(b)
    assert a == b


def test_pickle_parametric_quantum_gate() -> None:
    a = ParametricPauliRotation([0], [0])
    b = pickle.loads(pickle.dumps(a))
    assert id(a) != id(b)
    assert a == b
