# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Union

import numpy as np
import pytest

from quri_parts.circuit import ParametricQuantumCircuit, QuantumCircuit
from quri_parts.circuit.noise import BitFlipNoise, NoiseModel
from quri_parts.core.estimator import GeneralQuantumEstimator
from quri_parts.core.operator import Operator, PauliLabel, pauli_label
from quri_parts.core.state import (
    ComputationalBasisState,
    GeneralCircuitQuantumState,
    ParametricCircuitQuantumState,
    ParametricQuantumStateVector,
    QuantumStateVector,
    StateVectorType,
)
from quri_parts.qulacs import QulacsParametricStateT, QulacsStateT
from quri_parts.qulacs.circuit.compiled_circuit import (
    _QulacsCircuit,
    compile_circuit,
    compile_parametric_circuit,
)
from quri_parts.qulacs.estimator import (
    _Estimate,
    create_qulacs_density_matrix_concurrent_estimator,
    create_qulacs_density_matrix_concurrent_parametric_estimator,
    create_qulacs_density_matrix_estimator,
    create_qulacs_density_matrix_parametric_estimator,
    create_qulacs_general_density_matrix_estimator,
    create_qulacs_general_vector_estimator,
    create_qulacs_vector_concurrent_estimator,
    create_qulacs_vector_concurrent_parametric_estimator,
    create_qulacs_vector_estimator,
    create_qulacs_vector_parametric_estimator,
)


def create_vector(qubit_count: int, bits: int) -> StateVectorType:
    vector: StateVectorType = np.zeros(2**qubit_count, dtype=np.complex128)
    vector[bits] = 1.0
    return vector


def create_vector_state(qubit_count: int, bits: int) -> QuantumStateVector:
    return QuantumStateVector(qubit_count, create_vector(qubit_count, bits))


class TestVectorEstimator:
    def test_estimate_pauli_label(self) -> None:
        pauli = pauli_label("Z0 Z2 Z5")
        state = ComputationalBasisState(6, bits=0b110010)
        estimator = create_qulacs_vector_estimator()
        estimate = estimator(pauli, state)
        assert estimate.value == -1
        assert estimate.error == 0

    def test_estimate_operator(self) -> None:
        operator = Operator(
            {
                pauli_label("Z0 Z2 Z5"): 0.25,
                pauli_label("Z1 Z2 Z4"): 0.5j,
            }
        )
        state = ComputationalBasisState(6, bits=0b110010)
        estimator = create_qulacs_vector_estimator()
        estimate = estimator(operator, state)
        assert estimate.value == -0.25 + 0.5j
        assert estimate.error == 0

    def test_estimate_operator_with_precompiled_circuit(self) -> None:
        operator = Operator(
            {
                pauli_label("Z0 Z2 Z5"): 0.25,
                pauli_label("Z1 Z2 Z4"): 0.5j,
            }
        )
        estimator = create_qulacs_vector_estimator()
        circuit = QuantumCircuit(6)
        circuit.add_X_gate(1)
        circuit.add_X_gate(4)
        circuit.add_X_gate(5)

        compiled_circuit = compile_circuit(circuit)
        compiled_state = GeneralCircuitQuantumState(6, compiled_circuit)
        assert isinstance(compiled_state.circuit, _QulacsCircuit)
        estimate_with_compiled_state = estimator(operator, compiled_state)
        assert estimate_with_compiled_state.value == -0.25 + 0.5j
        assert estimate_with_compiled_state.error == 0

    def test_estimate_vector(self) -> None:
        pauli = pauli_label("Z0 Z2 Z5")
        state = create_vector_state(6, 0b110010)
        estimator = create_qulacs_vector_estimator()
        estimate = estimator(pauli, state)
        assert estimate.value == -1
        assert estimate.error == 0


class TestVectorConcurrentEstimator:
    def test_invalid_arguments(self) -> None:
        pauli = pauli_label("Z0 Z2 Z5")
        state = ComputationalBasisState(6, bits=0b110010)
        estimator = create_qulacs_vector_concurrent_estimator()

        with pytest.raises(ValueError):
            estimator([], [state])

        with pytest.raises(ValueError):
            estimator([pauli], [])

        with pytest.raises(ValueError):
            estimator([pauli] * 3, [state] * 2)

    def test_concurrent_estimate(self) -> None:
        operators: list[Union[PauliLabel, Operator]] = [
            pauli_label("Z0 Z2 Z5"),
            Operator(
                {
                    pauli_label("Z0 Z2 Z5"): 0.25,
                    pauli_label("Z1 Z2 Z4"): 0.5j,
                }
            ),
        ]
        states = [
            ComputationalBasisState(6, bits=0b110000),
            ComputationalBasisState(6, bits=0b110010),
        ]

        with ThreadPoolExecutor(max_workers=2) as executor:
            estimator = create_qulacs_vector_concurrent_estimator(
                executor, concurrency=2
            )
            result = estimator(operators, states)

        assert result == [
            _Estimate(value=-1, error=0),
            _Estimate(value=-0.25 + 0.5j, error=0),
        ]

    def test_concurent_estimate_with_prcompiled_circuit(self) -> None:
        operators: list[Union[PauliLabel, Operator]] = [
            pauli_label("Z0 Z2 Z5"),
            Operator(
                {
                    pauli_label("Z0 Z2 Z5"): 0.25,
                    pauli_label("Z1 Z2 Z4"): 0.5j,
                }
            ),
        ]
        circuit_1 = QuantumCircuit(6)
        circuit_1.add_X_gate(4)
        circuit_1.add_X_gate(5)

        circuit_2 = QuantumCircuit(6)
        circuit_2.add_X_gate(1)
        circuit_2.add_X_gate(4)
        circuit_2.add_X_gate(5)

        compiled_states = [
            GeneralCircuitQuantumState(n_qubits=6, circuit=compile_circuit(circuit_1)),
            GeneralCircuitQuantumState(n_qubits=6, circuit=compile_circuit(circuit_2)),
        ]
        assert isinstance(compiled_states[0].circuit, _QulacsCircuit)
        assert isinstance(compiled_states[1].circuit, _QulacsCircuit)

        with ThreadPoolExecutor(max_workers=2) as executor:
            estimator = create_qulacs_vector_concurrent_estimator(
                executor, concurrency=2
            )
            result_with_compiled_states = estimator(operators, compiled_states)

        assert result_with_compiled_states == [
            _Estimate(value=-1, error=0),
            _Estimate(value=-0.25 + 0.5j, error=0),
        ]

    def test_concurrent_estimate_vector(self) -> None:
        operators: list[Union[PauliLabel, Operator]] = [
            pauli_label("Z0 Z2 Z5"),
            Operator(
                {
                    pauli_label("Z0 Z2 Z5"): 0.25,
                    pauli_label("Z1 Z2 Z4"): 0.5j,
                }
            ),
        ]
        states = [
            create_vector_state(6, 0b110000),
            create_vector_state(6, 0b110010),
        ]

        with ThreadPoolExecutor(max_workers=2) as executor:
            estimator = create_qulacs_vector_concurrent_estimator(
                executor, concurrency=2
            )
            result = estimator(operators, states)

        assert result == [
            _Estimate(value=-1, error=0),
            _Estimate(value=-0.25 + 0.5j, error=0),
        ]

    def test_concurrent_estimate_single_state(self) -> None:
        operators: list[Union[PauliLabel, Operator]] = [
            pauli_label("Z0 Z2 Z5"),
            Operator(
                {
                    pauli_label("Z0 Z2 Z5"): 0.25,
                    pauli_label("Z1 Z2 Z4"): 0.5j,
                }
            ),
        ]
        states = [
            ComputationalBasisState(6, bits=0b110010),
        ]

        with ThreadPoolExecutor(max_workers=2) as executor:
            estimator = create_qulacs_vector_concurrent_estimator(
                executor, concurrency=2
            )
            result = estimator(operators, states)

        assert result == [
            _Estimate(value=-1, error=0),
            _Estimate(value=-0.25 + 0.5j, error=0),
        ]

    def test_concurrent_estimate_single_state_with_compiled_circuit(self) -> None:
        operators: list[Union[PauliLabel, Operator]] = [
            pauli_label("Z0 Z2 Z5"),
            Operator(
                {
                    pauli_label("Z0 Z2 Z5"): 0.25,
                    pauli_label("Z1 Z2 Z4"): 0.5j,
                }
            ),
        ]
        circuit = QuantumCircuit(6)
        circuit.add_X_gate(1)
        circuit.add_X_gate(4)
        circuit.add_X_gate(5)

        compiled_states = [GeneralCircuitQuantumState(6, compile_circuit(circuit))]

        assert isinstance(compiled_states[0].circuit, _QulacsCircuit)
        with ThreadPoolExecutor(max_workers=2) as executor:
            estimator = create_qulacs_vector_concurrent_estimator(
                executor, concurrency=2
            )
            result_with_compiled_circuit = estimator(operators, compiled_states)

        assert result_with_compiled_circuit == [
            _Estimate(value=-1, error=0),
            _Estimate(value=-0.25 + 0.5j, error=0),
        ]

    def test_concurrent_estimate_single_operator(self) -> None:
        operators: list[Union[PauliLabel, Operator]] = [
            Operator(
                {
                    pauli_label("Z0 Z2 Z5"): 0.25,
                    pauli_label("Z1 Z2 Z4"): 0.5j,
                }
            ),
        ]
        states = [
            ComputationalBasisState(6, bits=0b110010),
            ComputationalBasisState(6, bits=0b110011),
        ]

        with ThreadPoolExecutor(max_workers=2) as executor:
            estimator = create_qulacs_vector_concurrent_estimator(
                executor, concurrency=2
            )
            result = estimator(operators, states)

        assert result == [
            _Estimate(value=-0.25 + 0.5j, error=0),
            _Estimate(value=0.25 + 0.5j, error=0),
        ]


def parametric_circuit() -> ParametricQuantumCircuit:
    circuit = ParametricQuantumCircuit(6)

    circuit.add_RX_gate(0, -math.pi / 4)
    circuit.add_ParametricRX_gate(0)

    circuit.add_RY_gate(2, -math.pi / 4)
    circuit.add_ParametricRY_gate(2)

    circuit.add_H_gate(5)
    circuit.add_RZ_gate(5, -math.pi / 4)
    circuit.add_ParametricRZ_gate(5)

    circuit.add_ParametricPauliRotation_gate((0, 2, 5), (1, 2, 3))
    return circuit


def create_parametric_vector_state(
    qubit_count: int,
    circuit: ParametricQuantumCircuit,
    bits: int,
) -> ParametricQuantumStateVector:
    return ParametricQuantumStateVector(
        qubit_count, circuit, create_vector(qubit_count, bits)
    )


class TestVectorParametricEstimator:
    def test_estimate_pauli_label(self) -> None:
        pauli = pauli_label("Y0 X2 Y5")
        state = ParametricCircuitQuantumState(6, parametric_circuit())
        estimator = create_qulacs_vector_parametric_estimator()

        params = [0.0, 0.0, 0.0, 0.0]
        estimate = estimator(pauli, state, params)
        assert estimate.value == pytest.approx((1 / math.sqrt(2)) ** 3)
        assert estimate.error == 0

        params = [-math.pi / 4, 0, 0, 0]
        estimate = estimator(pauli, state, params)
        assert estimate.value == pytest.approx(0.5)
        assert estimate.error == 0

        params = [0, -math.pi / 4, 0, 0]
        estimate = estimator(pauli, state, params)
        assert estimate.value == pytest.approx(0.5)
        assert estimate.error == 0

        params = [0, 0, -math.pi / 4, 0]
        estimate = estimator(pauli, state, params)
        assert estimate.value == pytest.approx(0.5)
        assert estimate.error == 0

        params = [0, 0, 0, -math.pi / 4]
        estimate = estimator(pauli, state, params)
        assert estimate.value == pytest.approx(0)
        assert estimate.error == 0

    def test_estimate_operator(self) -> None:
        operator = Operator(
            {
                pauli_label("Y0 X2 Y5"): 0.25,
                pauli_label("Z1 X2 Z4"): 0.5j,
            }
        )
        state = ParametricCircuitQuantumState(6, parametric_circuit())
        estimator = create_qulacs_vector_parametric_estimator()

        params = [0.0, 0.0, 0.0, 0.0]
        estimate = estimator(operator, state, params)
        assert estimate.value == pytest.approx(
            0.25 * ((1 / math.sqrt(2)) ** 3) + 0.5j * (-1 / math.sqrt(2))
        )
        assert estimate.error == 0

        params = [-math.pi / 4, 0, 0, 0]
        estimate = estimator(operator, state, params)
        assert estimate.value == pytest.approx(0.25 * 0.5 + 0.5j * (-1 / math.sqrt(2)))
        assert estimate.error == 0

        params = [0, -math.pi / 4, 0, 0]
        estimate = estimator(operator, state, params)
        assert estimate.value == pytest.approx(0.25 * 0.5 + 0.5j * (-1))
        assert estimate.error == 0

        params = [0, 0, -math.pi / 4, 0]
        estimate = estimator(operator, state, params)
        assert estimate.value == pytest.approx(0.25 * 0.5 + 0.5j * (-1 / math.sqrt(2)))
        assert estimate.error == 0

        params = [0, 0, 0, -math.pi / 4]
        estimate = estimator(operator, state, params)
        assert estimate.value == pytest.approx(0 + 0.5j * (-0.5))
        assert estimate.error == 0

    def test_estimate_operator_with_compiled_circuit(self) -> None:
        operator = Operator(
            {
                pauli_label("Y0 X2 Y5"): 0.25,
                pauli_label("Z1 X2 Z4"): 0.5j,
            }
        )
        compiled_circuit = compile_parametric_circuit(parametric_circuit())
        compiled_state = ParametricCircuitQuantumState(6, compiled_circuit)
        estimator = create_qulacs_vector_parametric_estimator()

        params = [0, 0, 0, -math.pi / 4]
        estimate_with_compiled_state = estimator(operator, compiled_state, params)
        assert estimate_with_compiled_state.value == pytest.approx(0 + 0.5j * (-0.5))
        assert estimate_with_compiled_state.error == 0

    def test_estimate_vector(self) -> None:
        pauli = pauli_label("Y0 X2 Y5")
        state = create_parametric_vector_state(6, parametric_circuit(), 0b100000)
        estimator = create_qulacs_vector_parametric_estimator()

        params = [0.0, 0.0, 0.0, 0.0]
        estimate = estimator(pauli, state, params)
        assert estimate.value == pytest.approx(-((1 / math.sqrt(2)) ** 3))
        assert estimate.error == 0

        params = [-math.pi / 4, 0, 0, 0]
        estimate = estimator(pauli, state, params)
        assert estimate.value == pytest.approx(-0.5)
        assert estimate.error == 0

        params = [0, -math.pi / 4, 0, 0]
        estimate = estimator(pauli, state, params)
        assert estimate.value == pytest.approx(-0.5)
        assert estimate.error == 0

        params = [0, 0, -math.pi / 4, 0]
        estimate = estimator(pauli, state, params)
        assert estimate.value == pytest.approx(-0.5)
        assert estimate.error == 0

        params = [0, 0, 0, -math.pi / 4]
        estimate = estimator(pauli, state, params)
        assert estimate.value == pytest.approx(0)
        assert estimate.error == 0


class TestVectorConcurrentParametricEstimator:
    def test_concurrent_estimate(self) -> None:
        operator = Operator(
            {
                pauli_label("Y0 X2 Y5"): 0.25,
                pauli_label("Z1 X2 Z4"): 0.5j,
            }
        )
        state = ParametricCircuitQuantumState(6, parametric_circuit())
        params = [
            [0.0, 0.0, 0.0, 0.0],
            [-math.pi / 4, 0, 0, 0],
            [0, -math.pi / 4, 0, 0],
            [0, 0, -math.pi / 4, 0],
            [0, 0, 0, -math.pi / 4],
        ]

        with ThreadPoolExecutor(max_workers=2) as executor:
            estimator = create_qulacs_vector_concurrent_parametric_estimator(
                executor, concurrency=2
            )
            result = list(estimator(operator, state, params))

        assert len(result) == 5
        for r in result:
            assert r.error == 0

        assert result[0].value == pytest.approx(
            0.25 * ((1 / math.sqrt(2)) ** 3) + 0.5j * (-1 / math.sqrt(2))
        )
        assert result[1].value == pytest.approx(0.25 * 0.5 + 0.5j * (-1 / math.sqrt(2)))
        assert result[2].value == pytest.approx(0.25 * 0.5 + 0.5j * (-1))
        assert result[3].value == pytest.approx(0.25 * 0.5 + 0.5j * (-1 / math.sqrt(2)))
        assert result[4].value == pytest.approx(0 + 0.5j * (-0.5))

    def test_concurrent_estimate_with_compiled_circuit(self) -> None:
        operator = Operator(
            {
                pauli_label("Y0 X2 Y5"): 0.25,
                pauli_label("Z1 X2 Z4"): 0.5j,
            }
        )
        compiled_state = ParametricCircuitQuantumState(
            6, compile_parametric_circuit(parametric_circuit())
        )
        params = [
            [0.0, 0.0, 0.0, 0.0],
            [-math.pi / 4, 0, 0, 0],
            [0, -math.pi / 4, 0, 0],
            [0, 0, -math.pi / 4, 0],
            [0, 0, 0, -math.pi / 4],
        ]
        with ThreadPoolExecutor(max_workers=2) as executor:
            estimator = create_qulacs_vector_concurrent_parametric_estimator(
                executor, concurrency=2
            )
            result_with_compiled_state = list(
                estimator(operator, compiled_state, params)
            )

        assert len(result_with_compiled_state) == 5
        for r in result_with_compiled_state:
            assert r.error == 0

        assert result_with_compiled_state[0].value == pytest.approx(
            0.25 * ((1 / math.sqrt(2)) ** 3) + 0.5j * (-1 / math.sqrt(2))
        )
        assert result_with_compiled_state[1].value == pytest.approx(
            0.25 * 0.5 + 0.5j * (-1 / math.sqrt(2))
        )
        assert result_with_compiled_state[2].value == pytest.approx(
            0.25 * 0.5 + 0.5j * (-1)
        )
        assert result_with_compiled_state[3].value == pytest.approx(
            0.25 * 0.5 + 0.5j * (-1 / math.sqrt(2))
        )
        assert result_with_compiled_state[4].value == pytest.approx(0 + 0.5j * (-0.5))

    def test_concurrent_estimate_vector(self) -> None:
        operator = Operator(
            {
                pauli_label("Y0 X2 Y5"): 0.25,
                pauli_label("Z1 X2 Z4"): 0.5j,
            }
        )
        state = create_parametric_vector_state(6, parametric_circuit(), 0b100000)
        params = [
            [0.0, 0.0, 0.0, 0.0],
            [-math.pi / 4, 0, 0, 0],
            [0, -math.pi / 4, 0, 0],
            [0, 0, -math.pi / 4, 0],
            [0, 0, 0, -math.pi / 4],
        ]

        with ThreadPoolExecutor(max_workers=2) as executor:
            estimator = create_qulacs_vector_concurrent_parametric_estimator(
                executor, concurrency=2
            )
            result = list(estimator(operator, state, params))

        assert len(result) == 5
        for r in result:
            assert r.error == 0

        assert result[0].value == pytest.approx(
            0.25 * (-((1 / math.sqrt(2)) ** 3)) + 0.5j * (-1 / math.sqrt(2))
        )
        assert result[1].value == pytest.approx(
            0.25 * (-0.5) + 0.5j * (-1 / math.sqrt(2))
        )
        assert result[2].value == pytest.approx(0.25 * (-0.5) + 0.5j * (-1))
        assert result[3].value == pytest.approx(
            0.25 * (-0.5) + 0.5j * (-1 / math.sqrt(2))
        )
        assert result[4].value == pytest.approx(0 + 0.5j * (-0.5))


class TestGeneralVectorEstimator:
    @staticmethod
    def execute_test(
        general_estimator: GeneralQuantumEstimator[QulacsStateT, QulacsParametricStateT]
    ) -> None:
        pauli = pauli_label("Z0 Z2 Z5")
        operator = Operator(
            {
                pauli_label("Z0 Z2 Z5"): 0.25,
                pauli_label("Z1 Z2 Z4"): 0.5j,
            }
        )

        state_1 = ComputationalBasisState(6, bits=0b110000)
        state_2 = ComputationalBasisState(6, bits=0b110010)
        state_3 = create_vector_state(6, 0b110000)

        # test estimator
        estimator = create_qulacs_vector_estimator()
        assert general_estimator(pauli, state_1) == estimator(pauli, state_1)
        assert general_estimator(operator, state_2) == estimator(operator, state_2)
        assert general_estimator(pauli, state_3) == estimator(pauli, state_3)

        # test concurrent estimator
        concurrent_estimator = create_qulacs_vector_concurrent_estimator()
        estimates = concurrent_estimator([pauli], [state_1, state_2, state_3])
        assert general_estimator(pauli, [state_1, state_2, state_3]) == estimates

        estimates = concurrent_estimator([operator], [state_1, state_2, state_3])
        assert general_estimator([operator], [state_1, state_2, state_3]) == estimates

        estimates = concurrent_estimator([pauli, operator], [state_1])
        assert general_estimator([pauli, operator], state_1) == estimates

        estimates = concurrent_estimator([pauli, operator], [state_2])
        assert general_estimator([pauli, operator], [state_2]) == estimates

        estimates = concurrent_estimator([pauli, operator], [state_1, state_2])
        assert general_estimator([pauli, operator], [state_1, state_2]) == estimates

        # Set up parametric
        param_circuit = parametric_circuit()
        p_circuit_state = ParametricCircuitQuantumState(
            param_circuit.qubit_count, param_circuit
        )
        p_vector_state = ParametricQuantumStateVector(
            param_circuit.qubit_count,
            param_circuit,
            vector=create_vector(param_circuit.qubit_count, bits=1),
        )
        params = [np.random.random(param_circuit.parameter_count) for _ in range(4)]

        p_estimator = create_qulacs_vector_parametric_estimator()
        cp_estimator = create_qulacs_vector_concurrent_parametric_estimator()

        # test parametric estimator
        assert general_estimator(pauli, p_circuit_state, params[0]) == p_estimator(
            pauli, p_circuit_state, params[0].tolist()
        )
        assert general_estimator(pauli, p_vector_state, params[0]) == p_estimator(
            pauli, p_vector_state, params[0].tolist()
        )

        # test concurrent parametric estimator
        cp_estimator(pauli, p_circuit_state, [ps.tolist() for ps in params])
        general_estimator(pauli, p_circuit_state, params)
        assert general_estimator(pauli, p_circuit_state, params) == cp_estimator(
            pauli, p_circuit_state, [ps.tolist() for ps in params]
        )
        assert general_estimator(pauli, p_vector_state, params) == cp_estimator(
            pauli, p_vector_state, [ps.tolist() for ps in params]
        )

    def test_without_executor(self) -> None:
        general_estimator = create_qulacs_general_vector_estimator()
        self.execute_test(general_estimator)

    def test_with_executor(self) -> None:
        with ThreadPoolExecutor(max_workers=2) as executor:
            general_estimator = create_qulacs_general_vector_estimator(executor, 2)
            self.execute_test(general_estimator)


class TestDensityMatrixEstimatorWithNoiseModel:
    def test_estimate_with_empty_noise(self) -> None:
        model = NoiseModel()
        pauli = pauli_label("Z0 Z2 Z5")
        state = ComputationalBasisState(6, bits=0b110010)
        estimator = create_qulacs_density_matrix_estimator(model)
        estimate = estimator(pauli, state)
        assert estimate.value == -1
        assert estimate.error == 0

    def test_estimate_with_bitflip_noise(self) -> None:
        model = NoiseModel([BitFlipNoise(1.0)])
        pauli = pauli_label("Z0 Z2 Z5")
        state = ComputationalBasisState(6, bits=0b110010)
        estimator = create_qulacs_density_matrix_estimator(model)
        estimate = estimator(pauli, state)
        assert estimate.value == 1
        assert estimate.error == 0


class TestDensityMatrixParametricEstimatorWithNoiseModel:
    def test_estimate_with_empty_noise(self) -> None:
        model = NoiseModel()

        pauli = pauli_label("Y0 X2 Y5")
        state = ParametricCircuitQuantumState(6, parametric_circuit())
        estimator = create_qulacs_density_matrix_parametric_estimator(model)

        params = [0.0, 0.0, 0.0, 0.0]
        estimate = estimator(pauli, state, params)
        assert estimate.value == pytest.approx((1 / math.sqrt(2)) ** 3)
        assert estimate.error == 0

        params = [-math.pi / 4, 0, 0, 0]
        estimate = estimator(pauli, state, params)
        assert estimate.value == pytest.approx(0.5)
        assert estimate.error == 0

        params = [0, -math.pi / 4, 0, 0]
        estimate = estimator(pauli, state, params)
        assert estimate.value == pytest.approx(0.5)
        assert estimate.error == 0

        params = [0, 0, -math.pi / 4, 0]
        estimate = estimator(pauli, state, params)
        assert estimate.value == pytest.approx(0.5)
        assert estimate.error == 0

        params = [0, 0, 0, -math.pi / 4]
        estimate = estimator(pauli, state, params)
        assert estimate.value == pytest.approx(0)
        assert estimate.error == 0

    def test_estimate_with_bitflip_noise(self) -> None:
        model = NoiseModel([BitFlipNoise(1.0)])

        circuit = ParametricQuantumCircuit(6)
        circuit.add_ParametricRX_gate(0)
        circuit.add_ParametricRY_gate(2)
        circuit.add_ParametricRZ_gate(5)

        pauli = pauli_label("Z0 Z2 Z5")
        state = ParametricCircuitQuantumState(6, circuit)
        estimator = create_qulacs_density_matrix_parametric_estimator(model)

        params = [0.0, 0.0, 0.0]
        estimate = estimator(pauli, state, params)
        assert estimate.value == -1
        assert estimate.error == 0

        params = [math.pi, 0.0, math.pi]
        estimate = estimator(pauli, state, params)
        assert estimate.value == 1
        assert estimate.error == 0


class TestDensityMatrixConcurrentEstimatorWithNoiseModel:
    def test_estimate_with_empty_noise(self) -> None:
        model = NoiseModel()

        operators: list[Union[PauliLabel, Operator]] = [
            pauli_label("Z0 Z2 Z5"),
            Operator(
                {
                    pauli_label("Z0 Z2 Z5"): 0.25,
                    pauli_label("Z1 Z2 Z4"): 0.5j,
                }
            ),
        ]
        states = [
            ComputationalBasisState(6, bits=0b110000),
            ComputationalBasisState(6, bits=0b110010),
        ]

        with ThreadPoolExecutor(max_workers=2) as executor:
            estimator = create_qulacs_density_matrix_concurrent_estimator(
                model, executor, concurrency=2
            )
            result = estimator(operators, states)

        assert result == [
            _Estimate(value=-1, error=0),
            _Estimate(value=-0.25 + 0.5j, error=0),
        ]

    def test_estimate_with_bitflip_noise(self) -> None:
        model = NoiseModel([BitFlipNoise(1.0)])

        operators: list[Union[PauliLabel, Operator]] = [
            pauli_label("Z0 Z2 Z5"),
            Operator(
                {
                    pauli_label("Z0 Z2 Z5"): 0.25,
                    pauli_label("Z1 Z2 Z4"): 0.5j,
                }
            ),
        ]
        states = [
            ComputationalBasisState(6, bits=0b110000),
            ComputationalBasisState(6, bits=0b110010),
        ]

        with ThreadPoolExecutor(max_workers=2) as executor:
            estimator = create_qulacs_density_matrix_concurrent_estimator(
                model, executor, concurrency=2
            )
            result = estimator(operators, states)

        assert result == [
            _Estimate(value=1, error=0),
            _Estimate(value=0.25 + 0.5j, error=0),
        ]


class TestDensityMatrixConcurrentParametricEstimatorWithNoiseModel:
    def test_estimate_with_empty_noise(self) -> None:
        model = NoiseModel()

        operator = Operator(
            {
                pauli_label("Y0 X2 Y5"): 0.25,
                pauli_label("Z1 X2 Z4"): 0.5j,
            }
        )
        state = ParametricCircuitQuantumState(6, parametric_circuit())
        params = [
            [0.0, 0.0, 0.0, 0.0],
            [-math.pi / 4, 0, 0, 0],
            [0, -math.pi / 4, 0, 0],
            [0, 0, -math.pi / 4, 0],
            [0, 0, 0, -math.pi / 4],
        ]

        with ThreadPoolExecutor(max_workers=2) as executor:
            estimator = create_qulacs_density_matrix_concurrent_parametric_estimator(
                model, executor, concurrency=2
            )
            result = list(estimator(operator, state, params))

        assert len(result) == 5
        for r in result:
            assert r.error == 0
        assert result[0].value == pytest.approx(
            0.25 * ((1 / math.sqrt(2)) ** 3) + 0.5j * (-1 / math.sqrt(2))
        )
        assert result[1].value == pytest.approx(0.25 * 0.5 + 0.5j * (-1 / math.sqrt(2)))
        assert result[2].value == pytest.approx(0.25 * 0.5 + 0.5j * (-1))
        assert result[3].value == pytest.approx(0.25 * 0.5 + 0.5j * (-1 / math.sqrt(2)))
        assert result[4].value == pytest.approx(0 + 0.5j * (-0.5))

    def test_estimate_with_bitflip_noise(self) -> None:
        model = NoiseModel([BitFlipNoise(1.0)])

        circuit = ParametricQuantumCircuit(6)
        circuit.add_ParametricRX_gate(0)
        circuit.add_ParametricRY_gate(2)
        circuit.add_ParametricRZ_gate(5)

        operator = Operator(
            {
                pauli_label("Z0 Z2 Z5"): 1.0,
            }
        )
        state = ParametricCircuitQuantumState(6, circuit)
        params = [
            [0.0, 0.0, 0.0],
            [math.pi, 0.0, 0.0],
            [0.0, math.pi, 0.0],
            [math.pi, math.pi, 0.0],
            [math.pi, math.pi, math.pi],
        ]

        with ThreadPoolExecutor(max_workers=2) as executor:
            estimator = create_qulacs_density_matrix_concurrent_parametric_estimator(
                model, executor, concurrency=2
            )
            result = list(estimator(operator, state, params))

        assert len(result) == 5
        for r in result:
            assert r.error == 0
        assert result[0].value == -1
        assert result[1].value == 1
        assert result[2].value == 1
        assert result[3].value == -1
        assert result[4].value == -1


class TestGeneralDensityMatrixEstimator:
    @staticmethod
    def execute_test(
        general_estimator: GeneralQuantumEstimator[
            QulacsStateT, QulacsParametricStateT
        ],
        model: NoiseModel,
    ) -> None:
        pauli = pauli_label("Z0 Z2 Z5")
        operator = Operator(
            {
                pauli_label("Z0 Z2 Z5"): 0.25,
                pauli_label("Z1 Z2 Z4"): 0.5j,
            }
        )

        state_1 = ComputationalBasisState(6, bits=0b110000)
        state_2 = ComputationalBasisState(6, bits=0b110010)
        state_3 = create_vector_state(6, 0b110000)

        # test estimator
        estimator = create_qulacs_density_matrix_estimator(model)
        assert general_estimator(pauli, state_1) == estimator(pauli, state_1)
        assert general_estimator(operator, state_2) == estimator(operator, state_2)
        assert general_estimator(pauli, state_3) == estimator(pauli, state_3)

        # test concurrent estimator
        concurrent_estimator = create_qulacs_density_matrix_concurrent_estimator(model)
        estimates = concurrent_estimator([pauli], [state_1, state_2, state_3])
        assert general_estimator(pauli, [state_1, state_2, state_3]) == estimates

        estimates = concurrent_estimator([operator], [state_1, state_2, state_3])
        assert general_estimator([operator], [state_1, state_2, state_3]) == estimates

        estimates = concurrent_estimator([pauli, operator], [state_1])
        assert general_estimator([pauli, operator], state_1) == estimates

        estimates = concurrent_estimator([pauli, operator], [state_2])
        assert general_estimator([pauli, operator], [state_2]) == estimates

        estimates = concurrent_estimator([pauli, operator], [state_1, state_2])
        assert general_estimator([pauli, operator], [state_1, state_2]) == estimates

        # Set up parametric
        param_circuit = parametric_circuit()
        p_circuit_state = ParametricCircuitQuantumState(
            param_circuit.qubit_count, param_circuit
        )
        p_vector_state = ParametricQuantumStateVector(
            param_circuit.qubit_count,
            param_circuit,
            vector=create_vector(param_circuit.qubit_count, bits=1),
        )
        params = [np.random.random(param_circuit.parameter_count) for _ in range(4)]

        p_estimator = create_qulacs_density_matrix_parametric_estimator(model)
        cp_estimator = create_qulacs_density_matrix_concurrent_parametric_estimator(
            model
        )

        # test parametric estimator
        assert general_estimator(pauli, p_circuit_state, params[0]) == p_estimator(
            pauli, p_circuit_state, params[0].tolist()
        )
        assert general_estimator(pauli, p_vector_state, params[0]) == p_estimator(
            pauli, p_vector_state, params[0].tolist()
        )

        # test concurrent parametric estimator
        cp_estimator(pauli, p_circuit_state, [ps.tolist() for ps in params])
        general_estimator(pauli, p_circuit_state, params)
        assert general_estimator(pauli, p_circuit_state, params) == cp_estimator(
            pauli, p_circuit_state, [ps.tolist() for ps in params]
        )
        assert general_estimator(pauli, p_vector_state, params) == cp_estimator(
            pauli, p_vector_state, [ps.tolist() for ps in params]
        )

    def test_without_executor(self) -> None:
        model = NoiseModel([BitFlipNoise(1.0)])
        general_estimator = create_qulacs_general_density_matrix_estimator(model)
        self.execute_test(general_estimator, model)

    def test_with_executor(self) -> None:
        with ThreadPoolExecutor(max_workers=2) as executor:
            model = NoiseModel([BitFlipNoise(1.0)])
            general_estimator = create_qulacs_general_density_matrix_estimator(
                model, executor, 2
            )
            self.execute_test(general_estimator, model)
