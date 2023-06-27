from numpy import allclose, array, pi

from quri_parts.circuit import QuantumCircuit
from quri_parts.core.state import (
    ComputationalBasisState,
    GeneralCircuitQuantumState,
    QuantumStateVector,
)
from quri_parts.qulacs.simulator import evaluate_state_to_vector, run_circuit


def test_evaluate_state_to_vector_general_circuit_state_to_vec() -> None:
    n = 2
    # quri-parts operations
    quantum_circuit = QuantumCircuit(n)
    quantum_circuit.add_CNOT_gate(0, 1)
    quantum_circuit.add_X_gate(0)
    quantum_circuit.add_RX_gate(0, pi / 7)
    quantum_circuit.add_RY_gate(0, pi / 8)
    quantum_circuit.add_H_gate(1)

    circuit_quantum_state = GeneralCircuitQuantumState(n, quantum_circuit)
    quantum_state_vector = evaluate_state_to_vector(circuit_quantum_state)

    expect_output_vector = array(
        [
            -0.134491 - 0.154323j,
            0.676132 - 0.0306967j,
            -0.134491 - 0.154323j,
            0.676132 - 0.0306967j,
        ]
    )

    assert len(quantum_state_vector.circuit.gates) == 0
    assert allclose(quantum_state_vector.vector, expect_output_vector)


def test_evaluate_state_to_vector_vec_to_vec() -> None:
    n = 2
    # quri-parts operations
    quantum_circuit = QuantumCircuit(n)
    quantum_circuit.add_CNOT_gate(0, 1)
    quantum_circuit.add_X_gate(0)
    quantum_circuit.add_RX_gate(0, pi / 7)
    quantum_circuit.add_RY_gate(0, pi / 8)
    quantum_circuit.add_H_gate(1)

    circuit_quantum_state = QuantumStateVector(n, array([1, 0, 0, 0]), quantum_circuit)
    quantum_state_vector = evaluate_state_to_vector(circuit_quantum_state)

    expect_output_vector = array(
        [
            -0.134491 - 0.154323j,
            0.676132 - 0.0306967j,
            -0.134491 - 0.154323j,
            0.676132 - 0.0306967j,
        ]
    )

    assert len(quantum_state_vector.circuit.gates) == 0
    assert allclose(quantum_state_vector.vector, expect_output_vector)


def test_evaluate_state_to_vector_comp_state_to_vec() -> None:
    circuit_quantum_state = ComputationalBasisState(4, bits=0b0101)
    quantum_state_vector = evaluate_state_to_vector(circuit_quantum_state)

    expect_output_vector = array([0] * 5 + [1] + [0] * 10)

    assert len(quantum_state_vector.circuit.gates) == 0
    assert allclose(quantum_state_vector.vector, expect_output_vector)


def test_run_circuit_simple() -> None:
    n = 2
    theta_x = pi / 7
    theta_y = pi / 8

    update_circuit = QuantumCircuit(n)
    update_circuit.add_CNOT_gate(0, 1)
    update_circuit.add_X_gate(0)
    update_circuit.add_RX_gate(0, theta_x)
    update_circuit.add_RY_gate(0, theta_y)
    update_circuit.add_H_gate(1)

    updated_quantum_state_vector = run_circuit(update_circuit, array([1, 0, 0, 0]))

    # expectation
    expect_output_vector = array(
        [
            -0.134491 - 0.154323j,
            0.676132 - 0.0306967j,
            -0.134491 - 0.154323j,
            0.676132 - 0.0306967j,
        ]
    )

    assert allclose(updated_quantum_state_vector, expect_output_vector)
