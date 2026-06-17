from quri_parts.circuit import (
    CNOT,
    RZ,
    PauliRotation,
    QuantumCircuit,
    S,
    inverse_circuit,
)


def test_inverse_method() -> None:
    circuit = QuantumCircuit(
        3,
        cbit_count=1,
        gates=[
            S(0),
            RZ(1, 0.125),
            CNOT(0, 2),
            PauliRotation((0, 1), (1, 3), 0.75),
        ],
    )

    expected = inverse_circuit(circuit)

    inv_from_mutable = circuit.inverse()
    assert inv_from_mutable == expected
    assert inv_from_mutable.cbit_count == 0

    inv_from_immutable = circuit.freeze().inverse()
    assert inv_from_immutable == expected
