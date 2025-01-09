# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Sequence
from typing import Union, cast

import numpy as np
import numpy.typing as npt
import qulacs

import quri_parts.circuit.gate_names as names
import quri_parts.circuit.gates as gates
from quri_parts.circuit import QuantumCircuit
from quri_parts.circuit.noise import (
    BitFlipNoise,
    BitPhaseFlipNoise,
    CircuitNoiseInstruction,
    DepolarizingNoise,
    DepthIntervalNoise,
    GateIntervalNoise,
    GateNoiseInstruction,
    KrausNoise,
    MeasurementNoise,
    NoiseModel,
    PauliNoise,
    PhaseFlipNoise,
)
from quri_parts.qulacs.circuit.noise import convert_circuit_with_noise_model


def gates_equal(g1: qulacs.QuantumGateBase, g2: qulacs.QuantumGateBase) -> bool:
    def gate_info(
        g: qulacs.QuantumGateBase,
    ) -> tuple[str, list[int], list[int]]:
        return (
            g.get_name(),
            g.get_target_index_list(),
            g.get_control_index_list(),
        )

    # We need to disable type check due to an error in qulacs type annotation
    # https://github.com/qulacs/qulacs/issues/537
    return (gate_info(g1) == gate_info(g2)) and np.array_equal(
        g1.get_matrix(), g2.get_matrix()
    )


def density_matrix_equal(c1: qulacs.QuantumCircuit, c2: qulacs.QuantumCircuit) -> bool:
    def get_matrix(c: qulacs.QuantumCircuit) -> npt.NDArray[np.complex128]:
        matrix = qulacs.DensityMatrix(c.get_qubit_count())
        matrix.set_zero_state()
        c.update_quantum_state(matrix)
        return cast(npt.NDArray[np.complex128], matrix.get_matrix())

    return np.array_equal(get_matrix(c1), get_matrix(c2))


def test_convert_simple_gate_mapping() -> None:
    noises = [
        BitFlipNoise(0.001, [0, 1], [names.H]),
        PhaseFlipNoise(0.002, [], [names.X]),
        BitPhaseFlipNoise(0.003, [2], []),
        DepolarizingNoise(0.004, [], []),
    ]
    model = NoiseModel(noises)
    qp_gates = [gates.X(0), gates.H(1), gates.CNOT(1, 2)]
    circuit = QuantumCircuit(3, gates=qp_gates)
    converted = convert_circuit_with_noise_model(circuit, model)
    expected_gates = [
        qulacs.gate.X(0),
        qulacs.gate.DephasingNoise(0, 0.002),
        qulacs.gate.DepolarizingNoise(0, 0.004),
        qulacs.gate.H(1),
        qulacs.gate.BitFlipNoise(1, 0.001),
        qulacs.gate.DepolarizingNoise(1, 0.004),
        qulacs.gate.CNOT(1, 2),
        qulacs.gate.IndependentXZNoise(2, 0.003),
        qulacs.gate.DepolarizingNoise(1, 0.004),
        qulacs.gate.DepolarizingNoise(2, 0.004),
    ]

    assert converted.get_gate_count() == len(expected_gates)
    for i, expected in enumerate(expected_gates):
        assert gates_equal(converted.get_gate(i), expected)

    expected = qulacs.QuantumCircuit(3)
    for gate in expected_gates:
        expected.add_gate(gate)
    assert density_matrix_equal(converted, expected)


def test_convert_gate_interval_noise() -> None:
    noises: Sequence[Union[CircuitNoiseInstruction, GateNoiseInstruction]] = [
        GateIntervalNoise(
            [BitFlipNoise(0.005), PhaseFlipNoise(0.004)], gate_interval=2
        ),
        BitPhaseFlipNoise(0.003, [0], [names.X]),
        DepolarizingNoise(0.002, [1], [names.H]),
    ]
    model = NoiseModel(noises)
    qp_gates = [
        gates.X(0),
        gates.X(1),
        gates.H(0),
        gates.CNOT(0, 1),
        gates.H(1),
        gates.X(1),
    ]
    circuit = QuantumCircuit(2, gates=qp_gates)
    converted = convert_circuit_with_noise_model(circuit, model)
    expected_gates = [
        qulacs.gate.X(0),
        qulacs.gate.IndependentXZNoise(0, 0.003),
        qulacs.gate.X(1),
        qulacs.gate.H(0),
        qulacs.gate.BitFlipNoise(0, 0.005),  # interval noise 0
        qulacs.gate.DephasingNoise(0, 0.004),  # interval noise 1
        qulacs.gate.CNOT(0, 1),
        qulacs.gate.BitFlipNoise(1, 0.005),  # interval noise 0
        qulacs.gate.DephasingNoise(1, 0.004),  # interval noise 1
        qulacs.gate.H(1),
        qulacs.gate.DepolarizingNoise(1, 0.002),
        qulacs.gate.X(1),
        qulacs.gate.BitFlipNoise(1, 0.005),  # interval noise 0
        qulacs.gate.DephasingNoise(1, 0.004),  # interval noise 1
    ]

    assert converted.get_gate_count() == len(expected_gates)
    for i, expected in enumerate(expected_gates):
        assert gates_equal(converted.get_gate(i), expected)

    expected = qulacs.QuantumCircuit(2)
    for gate in expected_gates:
        expected.add_gate(gate)
    assert density_matrix_equal(converted, expected)


def test_convert_depth_interval_noise() -> None:
    noises: Sequence[Union[CircuitNoiseInstruction, GateNoiseInstruction]] = [
        DepthIntervalNoise(
            [BitFlipNoise(0.005), PhaseFlipNoise(0.004)], depth_interval=2
        ),
        BitPhaseFlipNoise(0.003, [0], [names.X]),
        DepolarizingNoise(0.002, [1], [names.H]),
    ]
    model = NoiseModel(noises)
    qp_gates = [
        gates.X(0),
        gates.X(1),
        gates.H(0),
        gates.CNOT(0, 1),
        gates.H(1),
        gates.X(1),
    ]
    circuit = QuantumCircuit(2, gates=qp_gates)
    converted = convert_circuit_with_noise_model(circuit, model)
    expected_gates = [
        qulacs.gate.X(0),
        qulacs.gate.IndependentXZNoise(0, 0.003),
        qulacs.gate.X(1),
        qulacs.gate.H(0),
        qulacs.gate.BitFlipNoise(0, 0.005),  # interval noise 0
        qulacs.gate.DephasingNoise(0, 0.004),  # interval noise 1
        qulacs.gate.BitFlipNoise(1, 0.005),  # interval noise 0
        qulacs.gate.DephasingNoise(1, 0.004),  # interval noise 1
        qulacs.gate.CNOT(0, 1),
        qulacs.gate.H(1),
        qulacs.gate.DepolarizingNoise(1, 0.002),
        qulacs.gate.BitFlipNoise(1, 0.005),  # interval noise 0
        qulacs.gate.DephasingNoise(1, 0.004),  # interval noise 1
        qulacs.gate.X(1),
        qulacs.gate.BitFlipNoise(0, 0.005),  # interval noise 0
        qulacs.gate.DephasingNoise(0, 0.004),  # interval noise 1
    ]

    assert converted.get_gate_count() == len(expected_gates)
    for i, expected_gate in enumerate(expected_gates):
        assert gates_equal(converted.get_gate(i), expected_gate)

    expected = qulacs.QuantumCircuit(2)
    for gate in expected_gates:
        expected.add_gate(gate)
    assert density_matrix_equal(converted, expected)


def test_convert_measurement_noise() -> None:
    noises = [MeasurementNoise([BitFlipNoise(0.004)])]
    model = NoiseModel(noises)
    qp_gates = [
        gates.X(0),
        gates.X(1),
        gates.H(0),
        gates.CNOT(0, 1),
        gates.H(1),
        gates.X(1),
    ]
    circuit = QuantumCircuit(2, gates=qp_gates)
    converted = convert_circuit_with_noise_model(circuit, model)
    expected_gates = [
        qulacs.gate.X(0),
        qulacs.gate.X(1),
        qulacs.gate.H(0),
        qulacs.gate.CNOT(0, 1),
        qulacs.gate.H(1),
        qulacs.gate.X(1),
        qulacs.gate.BitFlipNoise(0, 0.004),
        qulacs.gate.BitFlipNoise(1, 0.004),
    ]

    assert converted.get_gate_count() == len(expected_gates)
    for i, expected_gate in enumerate(expected_gates):
        assert gates_equal(converted.get_gate(i), expected_gate)

    expected = qulacs.QuantumCircuit(2)
    for gate in expected_gates:
        expected.add_gate(gate)
    assert density_matrix_equal(converted, expected)


def test_convert_measurement_noise2() -> None:
    # 2-qubit circuit
    N = 2
    qc = QuantumCircuit(N)

    # 100% bitflip noise
    nm = NoiseModel(
        [MeasurementNoise([BitFlipNoise(1.0)], qubit_indices=list(range(N)))]
    )

    # 4 X-gate on qubit-0
    qc.add_X_gate(0)
    qc.add_X_gate(0)
    qc.add_X_gate(0)
    qc.add_X_gate(0)
    # 2 X-gate on qubit-1
    qc.add_X_gate(1)
    qc.add_X_gate(1)

    noise_qc = convert_circuit_with_noise_model(qc, nm)

    assert noise_qc.get_gate_count() == len(qc.gates) + N


def test_convert_kraus_cptp() -> None:
    p0, p1 = 0.005, 0.002
    kraus_ops = [
        [[np.sqrt(1 - p0 - p1), 0], [0, np.sqrt(1 - p0 - p1)]],
        [[np.sqrt(p0), 0], [0, 0]],
        [[0, np.sqrt(p0)], [0, 0]],
        [[0, 0], [np.sqrt(p1), 0]],
        [[0, 0], [0, np.sqrt(p1)]],
    ]
    noise = KrausNoise(kraus_ops, [0], [names.H])

    noise_model = NoiseModel([noise])
    qp_gates = [gates.H(0)]
    circuit = QuantumCircuit(1, gates=qp_gates)
    converted = convert_circuit_with_noise_model(circuit, noise_model)
    matrix = qulacs.DensityMatrix(1)
    matrix.set_zero_state()
    converted.update_quantum_state(matrix)

    rho = np.array([[0.5, 0.5], [0.5, 0.5]])  # density matrix after H application
    cptp = np.array([[0 + 0j, 0 + 0j], [0 + 0j, 0 + 0j]])
    for kraus in kraus_ops:
        k = np.array(kraus)
        kdg = k.conj().T
        cptp += k @ rho @ kdg  # K_i rho K_i^dagger

    # We need to disable type check due to an error in qulacs type annotation
    # https://github.com/qulacs/qulacs/issues/537
    assert np.allclose(cptp, matrix.get_matrix())


def test_convert_empty_circuit() -> None:
    circuit = QuantumCircuit(1)

    noise_model_empty = NoiseModel([])
    converted_empty = convert_circuit_with_noise_model(circuit, noise_model_empty)
    assert converted_empty.get_gate_count() == 0

    noise_model_readout = NoiseModel([MeasurementNoise([BitFlipNoise(1.0)])])
    converted_readout = convert_circuit_with_noise_model(circuit, noise_model_readout)
    assert converted_readout.get_gate_count() == 1


def test_convert_pauli_noise() -> None:
    circuit = QuantumCircuit(3)
    circuit.add_H_gate(2)
    circuit.add_X_gate(0)
    circuit.add_CNOT_gate(2, 1)
    circuit.add_Z_gate(2)

    noises = [
        PauliNoise(
            pauli_list=[[1, 2], [2, 3]],
            prob_list=[0.001, 0.002],
            qubit_indices=[1, 2],
            target_gates=[names.CNOT],
        )
    ]
    model = NoiseModel(noises)

    converted = convert_circuit_with_noise_model(circuit, model)
    expected_gates = [
        qulacs.gate.H(2),
        qulacs.gate.X(0),
        qulacs.gate.CNOT(2, 1),
        qulacs.gate.Probabilistic(
            [0.001, 0.002],
            [
                qulacs.gate.Pauli([1, 2], [1, 2]),
                qulacs.gate.Pauli([1, 2], [2, 3]),
            ],
        ),
        qulacs.gate.Z(2),
    ]

    assert converted.get_gate_count() == len(expected_gates)
    for i, expected_gate in enumerate(expected_gates):
        assert converted.get_gate(i).to_json() == expected_gate.to_json()
