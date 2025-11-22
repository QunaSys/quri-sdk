from collections import Counter
from unittest import TestCase

from numpy import abs
from quri_parts.circuit import QuantumCircuit
from quri_parts.core.operator import PAULI_IDENTITY, Operator, pauli_label
from quri_parts.core.state import GeneralCircuitQuantumState

from quri_algo.algo.phase_estimation.qpe.qpe import QPEResult, TimeEvolutionQPE
from quri_algo.problem import QubitHamiltonian
from quri_algo.qsub.time_evolution.trotter_time_evo import TrotterTimeEvo


class QPETests(TestCase):
    def setUp(self) -> None:
        self.j = 1.0
        self.const = 0.2

        qp_op_heisenberg = Operator()
        qp_op_heisenberg.add_term(pauli_label("X0 X1"), self.j)
        qp_op_heisenberg.add_term(pauli_label("Y0 Y1"), self.j)
        qp_op_heisenberg.add_term(pauli_label("Z0 Z1"), self.j)

        qubit_count = 2
        self.h_heisenberg = QubitHamiltonian(qubit_count, qp_op_heisenberg)

        qp_op_heisenberg_const = qp_op_heisenberg + Operator(
            {PAULI_IDENTITY: self.const}
        )

        self.h_heisenberg_const = QubitHamiltonian(qubit_count, qp_op_heisenberg_const)

    def assert_is_close_enough(
        self,
        ancilla_qubits: int,
        expected_eigenvalue: float,
        result: QPEResult,
        scale: float,
    ) -> None:
        counts = Counter(result.ancilla_register_counts)
        most_common_eigenvalue = result.eigen_values[counts.most_common()[0][0]]
        precision = 2**-ancilla_qubits
        assert abs(most_common_eigenvalue - expected_eigenvalue) <= precision * scale

    def test_QPE_heis_00(self) -> None:
        qubit_count = 2
        ancilla_qubits = 4
        sampling_shots = 100000
        t = 1 / 16
        state = GeneralCircuitQuantumState(qubit_count)

        qpe = TimeEvolutionQPE(TrotterTimeEvo)
        result = qpe.run(state, sampling_shots, ancilla_qubits, self.h_heisenberg, t)
        expected_eigenvalue = self.j

        self.assert_is_close_enough(ancilla_qubits, expected_eigenvalue, result, abs(t))

    def test_QPE_heis_11(self) -> None:
        qubit_count = 2
        ancilla_qubits = 4
        sampling_shots = 100000
        t = 1 / 16

        state = GeneralCircuitQuantumState(qubit_count)
        circuit = QuantumCircuit(qubit_count)
        circuit.add_X_gate(0)
        circuit.add_X_gate(1)
        state = state.with_gates_applied(circuit)

        qpe = TimeEvolutionQPE(TrotterTimeEvo)
        result = qpe.run(state, sampling_shots, ancilla_qubits, self.h_heisenberg, t)
        expected_eigenvalue = self.j

        self.assert_is_close_enough(ancilla_qubits, expected_eigenvalue, result, abs(t))

    def test_QPE_heis_triplet(self) -> None:
        qubit_count = 2
        ancilla_qubits = 4
        sampling_shots = 100000
        t = 1 / 16

        state = GeneralCircuitQuantumState(qubit_count)
        circuit = QuantumCircuit(qubit_count)
        circuit.add_H_gate(0)
        circuit.add_CNOT_gate(0, 1)
        state = state.with_gates_applied(circuit)

        qpe = TimeEvolutionQPE(TrotterTimeEvo)
        result = qpe.run(state, sampling_shots, ancilla_qubits, self.h_heisenberg, t)
        expected_eigenvalue = self.j

        self.assert_is_close_enough(ancilla_qubits, expected_eigenvalue, result, abs(t))

    def test_QPE_heis_singlet(self) -> None:
        qubit_count = 2
        ancilla_qubits = 4
        sampling_shots = 100000

        t = -1 / 16  # if we expect a negative eigenvalue, we need this

        state = GeneralCircuitQuantumState(qubit_count)
        circuit = QuantumCircuit(qubit_count)
        circuit.add_H_gate(0)
        circuit.add_CNOT_gate(0, 1)
        circuit.add_Y_gate(0)
        state = state.with_gates_applied(circuit)

        qpe = TimeEvolutionQPE(TrotterTimeEvo)
        result = qpe.run(state, sampling_shots, ancilla_qubits, self.h_heisenberg, t)
        expected_eigenvalue = -3 * self.j

        self.assert_is_close_enough(ancilla_qubits, expected_eigenvalue, result, abs(t))

    def test_QPE_heis_00_const(self) -> None:
        qubit_count = 2
        ancilla_qubits = 4
        sampling_shots = 100000
        t = 1 / 16
        state = GeneralCircuitQuantumState(qubit_count)

        qpe = TimeEvolutionQPE(TrotterTimeEvo)
        result = qpe.run(
            state, sampling_shots, ancilla_qubits, self.h_heisenberg_const, t
        )
        expected_eigenvalue = self.j + self.const

        self.assert_is_close_enough(ancilla_qubits, expected_eigenvalue, result, abs(t))

    def test_QPE_heis_singlet_const(self) -> None:
        qubit_count = 2
        ancilla_qubits = 4
        sampling_shots = 100000

        t = (
            -1 / 16
        )  # if we expect a negative eigenvalue, we need this to prevent an int overflow in the ancilla register

        state = GeneralCircuitQuantumState(qubit_count)
        circuit = QuantumCircuit(qubit_count)
        circuit.add_H_gate(0)
        circuit.add_CNOT_gate(0, 1)
        circuit.add_Y_gate(0)
        state = state.with_gates_applied(circuit)

        qpe = TimeEvolutionQPE(TrotterTimeEvo)
        result = qpe.run(
            state, sampling_shots, ancilla_qubits, self.h_heisenberg_const, t
        )
        expected_eigenvalue = -3 * self.j + self.const

        self.assert_is_close_enough(ancilla_qubits, expected_eigenvalue, result, abs(t))
