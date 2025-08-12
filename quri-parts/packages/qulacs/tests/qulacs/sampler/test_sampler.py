# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import unittest.mock
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional, cast

import numpy as np
import numpy.testing as npt
import pytest

from quri_parts.circuit import (
    LinearMappedUnboundParametricQuantumCircuit,
    QuantumCircuit,
    UnboundParametricQuantumCircuit,
)
from quri_parts.circuit.noise import BitFlipNoise, NoiseModel
from quri_parts.core.sampling import (
    ConcurrentSampler,
    MeasurementCounts,
    Sampler,
    ideal_sample_from_state_vector,
)
from quri_parts.core.state import quantum_state
from quri_parts.qulacs.circuit.compiled_circuit import compile_circuit
from quri_parts.qulacs.sampler import (
    create_qulacs_density_matrix_concurrent_sampler,
    create_qulacs_density_matrix_general_sampler,
    create_qulacs_density_matrix_ideal_sampler,
    create_qulacs_density_matrix_sampler,
    create_qulacs_general_vector_ideal_sampler,
    create_qulacs_general_vector_sampler,
    create_qulacs_ideal_density_matrix_general_sampler,
    create_qulacs_noisesimulator_concurrent_sampler,
    create_qulacs_noisesimulator_general_sampler,
    create_qulacs_noisesimulator_sampler,
    create_qulacs_stochastic_state_vector_concurrent_sampler,
    create_qulacs_stochastic_state_vector_sampler,
    create_qulacs_vector_concurrent_sampler,
    create_qulacs_vector_ideal_sampler,
    create_qulacs_vector_sampler,
)
from quri_parts.qulacs.simulator import evaluate_state_to_vector

if TYPE_CHECKING:
    from concurrent.futures import Executor


def circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(4)
    circuit.add_X_gate(0)
    circuit.add_H_gate(1)
    circuit.add_Z_gate(2)
    circuit.add_Y_gate(3)
    return circuit


class TestQulacsVectorSampler:
    @pytest.mark.parametrize("qubits", [4, 12])
    @pytest.mark.parametrize("shots", [800, 1200, 2**12 + 100])
    def test_sampler(self, qubits: int, shots: int) -> None:
        circuit = QuantumCircuit(qubits)
        for i in range(qubits):
            circuit.add_H_gate(i)

        sampler = create_qulacs_vector_sampler()
        counts = sampler(circuit, shots)

        assert set(counts.keys()).issubset(range(2**qubits))
        assert all(c >= 0 for c in counts.values())
        assert sum(counts.values()) == shots

    @pytest.mark.parametrize("qubits", [4, 12])
    @pytest.mark.parametrize("shots", [800, 1200, 2**12 + 100])
    def test_ideal_sampler(self, qubits: int, shots: int) -> None:
        circuit = QuantumCircuit(qubits)
        # Apply all `H`
        for i in range(qubits):
            circuit.add_H_gate(i)

        sampler = create_qulacs_vector_ideal_sampler()
        counts = sampler(circuit, shots)

        assert set(counts.keys()).issubset(range(2**qubits))
        assert all(c >= 0 for c in counts.values())
        probs = np.array(list(counts.values())) / shots

        npt.assert_almost_equal(probs.sum(), 1.0)
        mean_val = np.mean(probs)
        # Uniform superposition exact value
        assert all(np.isclose(mean_val, x) for x in probs)

    @pytest.mark.parametrize("qubits", [4, 12])
    @pytest.mark.parametrize("shots", [800, 1200, 2**12 + 100])
    def test_sampler_with_compiled_circuit(self, qubits: int, shots: int) -> None:
        circuit = QuantumCircuit(qubits)
        for i in range(qubits):
            circuit.add_H_gate(i)
        compiled_circuit = compile_circuit(circuit)

        sampler = create_qulacs_vector_sampler()
        counts_with_compiled_circuit = sampler(compiled_circuit, shots)

        assert set(counts_with_compiled_circuit.keys()).issubset(range(2**qubits))
        assert all(c >= 0 for c in counts_with_compiled_circuit.values())
        assert sum(counts_with_compiled_circuit.values()) == shots


class TestQulacsVectorConcurrentSampler:
    def test_concurrent_sampler(self) -> None:
        circuit1 = circuit()
        circuit2 = circuit()
        circuit2.add_X_gate(3)

        with ThreadPoolExecutor(max_workers=2) as executor:
            sampler = create_qulacs_vector_concurrent_sampler(executor, 2)
            results = list(sampler([(circuit1, 1000), (circuit2, 2000)]))

        assert set(results[0]) == {0b1001, 0b1011}
        assert all(c >= 0 for c in results[0].values())
        assert sum(results[0].values()) == 1000

        assert set(results[1]) == {0b0001, 0b0011}
        assert all(c >= 0 for c in results[1].values())
        assert sum(results[1].values()) == 2000

    def test_concurrent_sampler_with_compiled_circuit(self) -> None:
        circuit1 = circuit()
        circuit2 = circuit()
        circuit2.add_X_gate(3)

        compiled_circuit_1 = compile_circuit(circuit1)
        compiled_circuit_2 = compile_circuit(circuit2)

        with ThreadPoolExecutor(max_workers=2) as executor:
            sampler = create_qulacs_vector_concurrent_sampler(executor, 2)
            results_with_compiled_circuit = list(
                sampler([(compiled_circuit_1, 1000), (compiled_circuit_2, 2000)])
            )

        assert set(results_with_compiled_circuit[0]) == {0b1001, 0b1011}
        assert all(c >= 0 for c in results_with_compiled_circuit[0].values())
        assert sum(results_with_compiled_circuit[0].values()) == 1000

        assert set(results_with_compiled_circuit[1]) == {0b0001, 0b0011}
        assert all(c >= 0 for c in results_with_compiled_circuit[1].values())
        assert sum(results_with_compiled_circuit[1].values()) == 2000


class TestQulacsVectorGeneralSampler(unittest.TestCase):
    def setUp(self) -> None:
        self.general_sampler = create_qulacs_general_vector_sampler()
        self.general_sampler.sampler = unittest.mock.Mock(
            side_effect=self.general_sampler.sampler
        )
        self.general_sampler.state_sampler = unittest.mock.Mock(
            side_effect=self.general_sampler.state_sampler
        )
        self.general_sampler.parametric_sampler = unittest.mock.Mock(
            side_effect=self.general_sampler.parametric_sampler
        )
        self.general_sampler.parametric_state_sampler = unittest.mock.Mock(
            side_effect=self.general_sampler.parametric_state_sampler
        )

        self.circuit = circuit()
        n_qubits = self.circuit.qubit_count
        for i in range(n_qubits):
            self.circuit.add_H_gate(i)

        self.state = quantum_state(n_qubits, circuit=self.circuit)

        self.param_circuit_1 = LinearMappedUnboundParametricQuantumCircuit(
            self.circuit.qubit_count
        )
        self.param_circuit_1.extend(self.circuit.gates)
        a, b = self.param_circuit_1.add_parameters("a", "b")
        self.param_circuit_1.add_ParametricRZ_gate(0, {a: 1, b: 2})
        self.param_state_1 = quantum_state(n_qubits, circuit=self.param_circuit_1)

        self.param_circuit_2 = UnboundParametricQuantumCircuit(self.circuit.qubit_count)
        self.param_circuit_2.extend(self.circuit.gates)
        self.param_circuit_2.add_ParametricRZ_gate(0)
        self.param_state_2 = quantum_state(
            n_qubits, circuit=self.param_circuit_2, vector=np.ones(2**n_qubits) / 4
        )

    def test_call_as_sampler(self) -> None:
        qubits = 4

        counts = self.general_sampler(self.circuit, 100)

        cast(unittest.mock.Mock, self.general_sampler.sampler).assert_called_once_with(
            self.circuit, 100
        )

        assert set(counts.keys()).issubset(range(2**qubits))
        assert all(c >= 0 for c in counts.values())
        assert sum(counts.values()) == 100

    def test_call_as_state_sampler(self) -> None:
        qubits = 4
        counts = self.general_sampler(self.state, 200)

        cast(
            unittest.mock.Mock, self.general_sampler.state_sampler
        ).assert_called_once_with(self.state, 200)

        assert set(counts.keys()).issubset(range(2**qubits))
        assert all(c >= 0 for c in counts.values())
        assert sum(counts.values()) == 200

    def test_call_as_param_sampler_linear_mapped_circuit(self) -> None:
        qubits = 4
        counts = self.general_sampler(self.param_circuit_1, 300, [10.0, 20.0])

        cast(
            unittest.mock.Mock, self.general_sampler.parametric_sampler
        ).assert_called_once_with(self.param_circuit_1, 300, [10.0, 20.0])

        assert set(counts.keys()).issubset(range(2**qubits))
        assert all(c >= 0 for c in counts.values())
        assert sum(counts.values()) == 300

    def test_call_as_param_sampler_unbound_circuit(self) -> None:
        qubits = 4
        counts = self.general_sampler(self.param_circuit_2, 400, [20.0])

        cast(
            unittest.mock.Mock, self.general_sampler.parametric_sampler
        ).assert_called_once_with(self.param_circuit_2, 400, [20.0])

        assert set(counts.keys()).issubset(range(2**qubits))
        assert all(c >= 0 for c in counts.values())
        assert sum(counts.values()) == 400

    def test_call_as_param_circuit_state_sampler(self) -> None:
        qubits = 4

        counts = self.general_sampler(self.param_state_1, 500, [20.0, 30.0])

        cast(
            unittest.mock.Mock, self.general_sampler.parametric_state_sampler
        ).assert_called_once_with(self.param_state_1, 500, [20.0, 30.0])

        assert set(counts.keys()).issubset(range(2**qubits))
        assert all(c >= 0 for c in counts.values())
        assert sum(counts.values()) == 500

    def test_call_as_param_state_vector_sampler(self) -> None:
        qubits = 4
        counts = self.general_sampler(self.param_state_2, 600, [20.0])

        cast(
            unittest.mock.Mock, self.general_sampler.parametric_state_sampler
        ).assert_called_once_with(self.param_state_2, 600, [20.0])

        assert set(counts.keys()).issubset(range(2**qubits))
        assert all(c >= 0 for c in counts.values())
        assert sum(counts.values()) == 600

    def test_call_as_mixed_concurrent_sampler(self) -> None:
        qubits = 4

        # Call without list
        counts = self.general_sampler(
            (self.circuit, 100),
            (self.state, 200),
            (self.param_circuit_1, 300, [10.0, 20.0]),
            (self.param_circuit_2, 400, [20.0]),
            (self.param_state_1, 500, [20.0, 30.0]),
            (self.param_state_2, 600, [20.0]),
        )

        for count, shots in zip(counts, [100, 200, 300, 400, 500, 600]):
            assert set(count.keys()).issubset(range(2**qubits))
            assert all(c >= 0 for c in count.values())
            assert sum(count.values()) == shots

        # Call with list
        counts = self.general_sampler(
            [
                (self.circuit, 100),
                (self.state, 200),
                (self.param_circuit_1, 300, [10.0, 20.0]),
                (self.param_circuit_2, 400, [20.0]),
                (self.param_state_1, 500, [20.0, 30.0]),
                (self.param_state_2, 600, [20.0]),
            ]
        )

        for count, shots in zip(counts, [100, 200, 300, 400, 500, 600]):
            assert set(count.keys()).issubset(range(2**qubits))
            assert all(c >= 0 for c in count.values())
            assert sum(count.values()) == shots


class TestQulacsVectorIdealGeneralSampler(unittest.TestCase):
    def setUp(self) -> None:
        self.general_sampler = create_qulacs_general_vector_ideal_sampler()
        self.general_sampler.sampler = unittest.mock.Mock(
            side_effect=self.general_sampler.sampler
        )
        self.general_sampler.state_sampler = unittest.mock.Mock(
            side_effect=self.general_sampler.state_sampler
        )
        self.general_sampler.parametric_sampler = unittest.mock.Mock(
            side_effect=self.general_sampler.parametric_sampler
        )
        self.general_sampler.parametric_state_sampler = unittest.mock.Mock(
            side_effect=self.general_sampler.parametric_state_sampler
        )

        self.circuit = circuit()
        n_qubits = self.circuit.qubit_count
        for i in range(n_qubits):
            self.circuit.add_H_gate(i)

        self.state = quantum_state(n_qubits, circuit=self.circuit)

        self.param_circuit_1 = LinearMappedUnboundParametricQuantumCircuit(
            self.circuit.qubit_count
        )
        self.param_circuit_1.extend(self.circuit.gates)
        a, b = self.param_circuit_1.add_parameters("a", "b")
        self.param_circuit_1.add_ParametricRZ_gate(0, {a: 1, b: 2})
        self.param_state_1 = quantum_state(n_qubits, circuit=self.param_circuit_1)

        self.param_circuit_2 = UnboundParametricQuantumCircuit(self.circuit.qubit_count)
        self.param_circuit_2.extend(self.circuit.gates)
        self.param_circuit_2.add_ParametricRZ_gate(0)
        self.param_state_2 = quantum_state(
            n_qubits, circuit=self.param_circuit_2, vector=np.ones(2**n_qubits) / 4
        )

        self.test_param_1 = [20.0, 30.0]
        self.test_param_2 = [20.0]

        (
            self.circuit_cnt,
            self.state_cnt,
            self.linear_mapped_circuit_cnt,
            self.unbound_circuit_cnt,
            self.linear_mapped_state_cnt,
            self.unbound_state_vec_cnt,
        ) = self._get_ideal_cnts()

    def _get_ideal_cnts(self) -> tuple[MeasurementCounts, ...]:
        n_qubit = 4
        circuit_state_vec = evaluate_state_to_vector(
            quantum_state(n_qubit, circuit=self.circuit)
        ).vector

        circuit_cnt = ideal_sample_from_state_vector(circuit_state_vec, 100)
        state_cnt = ideal_sample_from_state_vector(circuit_state_vec, 200)

        linear_mapped_circuit_vec = evaluate_state_to_vector(
            quantum_state(
                n_qubit,
                circuit=self.param_circuit_1.bind_parameters(self.test_param_1),
            )
        ).vector
        linear_mapped_circuit_cnt = ideal_sample_from_state_vector(
            linear_mapped_circuit_vec, 300
        )
        linear_mapped_state_cnt = ideal_sample_from_state_vector(
            linear_mapped_circuit_vec, 500
        )

        unbound_circuit_vec = evaluate_state_to_vector(
            quantum_state(
                n_qubit,
                circuit=self.param_circuit_2.bind_parameters(self.test_param_2),
            )
        ).vector
        unbound_circuit_cnt = ideal_sample_from_state_vector(unbound_circuit_vec, 400)

        param_state_vec = evaluate_state_to_vector(
            quantum_state(
                n_qubit,
                circuit=self.param_circuit_2.bind_parameters(self.test_param_2),
                vector=np.ones(2**n_qubit) / 4,
            )
        ).vector

        unbound_state_vec_cnt = ideal_sample_from_state_vector(param_state_vec, 600)

        return (
            circuit_cnt,
            state_cnt,
            linear_mapped_circuit_cnt,
            unbound_circuit_cnt,
            linear_mapped_state_cnt,
            unbound_state_vec_cnt,
        )

    def test_call_as_sampler(self) -> None:
        counts = self.general_sampler(self.circuit, 100)
        cast(unittest.mock.Mock, self.general_sampler.sampler).assert_called_once_with(
            self.circuit, 100
        )
        assert counts == self.circuit_cnt

    def test_call_as_state_sampler(self) -> None:
        counts = self.general_sampler(self.state, 200)
        cast(
            unittest.mock.Mock, self.general_sampler.state_sampler
        ).assert_called_once_with(self.state, 200)
        assert counts == self.state_cnt

    def test_call_as_param_sampler_linear_mapped_circuit(self) -> None:
        counts = self.general_sampler(self.param_circuit_1, 300, [20.0, 30.0])

        cast(
            unittest.mock.Mock, self.general_sampler.parametric_sampler
        ).assert_called_once_with(self.param_circuit_1, 300, [20.0, 30.0])
        assert counts == self.linear_mapped_circuit_cnt

    def test_call_as_param_sampler_unbound_circuit(self) -> None:
        counts = self.general_sampler(self.param_circuit_2, 400, [20.0])
        cast(
            unittest.mock.Mock, self.general_sampler.parametric_sampler
        ).assert_called_once_with(self.param_circuit_2, 400, [20.0])
        assert counts == self.unbound_circuit_cnt

    def test_call_as_param_circuit_state_sampler(self) -> None:
        counts = self.general_sampler(self.param_state_1, 500, [20.0, 30.0])
        cast(
            unittest.mock.Mock, self.general_sampler.parametric_state_sampler
        ).assert_called_once_with(self.param_state_1, 500, [20.0, 30.0])
        assert counts == self.linear_mapped_state_cnt

    def test_call_as_param_state_vector_sampler(self) -> None:
        counts = self.general_sampler(self.param_state_2, 600, [20.0])
        cast(
            unittest.mock.Mock, self.general_sampler.parametric_state_sampler
        ).assert_called_once_with(self.param_state_2, 600, [20.0])
        assert counts == self.unbound_state_vec_cnt

    def test_call_as_param_concurrent_sampler(self) -> None:
        counts = list(
            self.general_sampler(
                self.param_circuit_1,
                [(300, [10.0, 20.0]), (100, [0.0, 0.0])],
            )
        )
        assert len(counts) == 2
        assert counts[0] == self.linear_mapped_circuit_cnt
        assert counts[1] == self.circuit_cnt

    def test_call_as_param_concurrent_state_sampler(self) -> None:
        counts = list(
            self.general_sampler(
                self.param_state_1,
                [(300, [10.0, 20.0]), (100, [0.0, 0.0])],
            )
        )
        assert len(counts) == 2
        assert counts[0] == self.linear_mapped_circuit_cnt
        assert counts[1] == self.circuit_cnt

    def test_call_as_mixed_concurrent_sampler(self) -> None:
        # Call without list
        counts = self.general_sampler(
            (self.circuit, 100),
            (self.state, 200),
            (self.param_circuit_1, 300, [10.0, 20.0]),
            (self.param_circuit_2, 400, [20.0]),
            (self.param_state_1, 500, [20.0, 30.0]),
            (self.param_state_2, 600, [20.0]),
        )

        for count, expected_count in zip(
            counts,
            [
                self.circuit_cnt,
                self.state_cnt,
                self.linear_mapped_circuit_cnt,
                self.unbound_circuit_cnt,
                self.linear_mapped_state_cnt,
                self.unbound_state_vec_cnt,
            ],
        ):
            assert count == expected_count

        # Call with list
        counts = self.general_sampler(
            [
                (self.circuit, 100),
                (self.state, 200),
                (self.param_circuit_1, 300, [10.0, 20.0]),
                (self.param_circuit_2, 400, [20.0]),
                (self.param_state_1, 500, [20.0, 30.0]),
                (self.param_state_2, 600, [20.0]),
            ]
        )

        for count, expected_count in zip(
            counts,
            [
                self.circuit_cnt,
                self.state_cnt,
                self.linear_mapped_circuit_cnt,
                self.unbound_circuit_cnt,
                self.linear_mapped_state_cnt,
                self.unbound_state_vec_cnt,
            ],
        ):
            assert count == expected_count


class TestSamplerWithNoiseModel:
    # (2**4)**2/10 = 25.6, (2**7)**2)/10 = 1638.4
    @pytest.mark.parametrize("qubits", [4, 7])
    @pytest.mark.parametrize("shots", [800, 1200, 2000])
    @pytest.mark.parametrize(
        "sampler_creator",
        [
            create_qulacs_density_matrix_sampler,
            create_qulacs_density_matrix_general_sampler,
            create_qulacs_stochastic_state_vector_sampler,
            create_qulacs_noisesimulator_sampler,
            create_qulacs_noisesimulator_general_sampler,
        ],
    )
    def test_sampler_with_empty_noise(
        self,
        qubits: int,
        shots: int,
        sampler_creator: Callable[[NoiseModel], Sampler],
    ) -> None:
        circuit = QuantumCircuit(qubits)
        for i in range(qubits):
            circuit.add_H_gate(i)

        model = NoiseModel()
        sampler = sampler_creator(model)
        counts = sampler(circuit, shots)

        assert set(counts.keys()).issubset(range(2**qubits))
        assert all(c >= 0 for c in counts.values())
        npt.assert_almost_equal(sum(counts.values()), shots)

    @pytest.mark.parametrize("qubits", [4, 7])
    @pytest.mark.parametrize("shots", [800, 1200, 2000])
    @pytest.mark.parametrize(
        "sampler_creator",
        [
            create_qulacs_density_matrix_ideal_sampler,
        ],
    )
    def test_ideal_sampler_with_empty_noise(
        self,
        qubits: int,
        shots: int,
        sampler_creator: Callable[[NoiseModel], Sampler],
    ) -> None:
        circuit = QuantumCircuit(qubits)
        for i in range(qubits):
            circuit.add_H_gate(i)

        model = NoiseModel()
        sampler = sampler_creator(model)
        counts = sampler(circuit, shots)

        assert set(counts.keys()).issubset(range(2**qubits))
        assert all(c >= 0 for c in counts.values())
        probs = np.array(list(counts.values())) / shots

        npt.assert_almost_equal(probs.sum(), 1.0)
        mean_val = np.mean(probs)
        # Uniform superposition exact value
        assert all(np.isclose(mean_val, x) for x in probs)

    @pytest.mark.parametrize("qubits", [4, 7])
    @pytest.mark.parametrize("shots", [800, 1200, 2000])
    @pytest.mark.parametrize(
        "sampler_creator",
        [
            create_qulacs_density_matrix_sampler,
            create_qulacs_density_matrix_general_sampler,
            create_qulacs_stochastic_state_vector_sampler,
            create_qulacs_noisesimulator_sampler,
            create_qulacs_noisesimulator_general_sampler,
        ],
    )
    def test_sampler_with_bitflip_noise(
        self,
        qubits: int,
        shots: int,
        sampler_creator: Callable[[NoiseModel], Sampler],
    ) -> None:
        circuit = QuantumCircuit(qubits)
        circuit.add_H_gate(3)
        circuit.add_X_gate(2)
        circuit.add_X_gate(1)
        circuit.add_CNOT_gate(3, 0)

        model = NoiseModel([BitFlipNoise(1.0)])
        sampler = sampler_creator(model)
        counts = sampler(circuit, shots)

        assert set(counts.keys()) == {0b1001, 0b0000}
        # {0000, 0000} HXX_ -> {1110, 0110} BitFlip -> {0000, 1000} CNOT ->
        # {0000, 1001} BitFlip -> {1001, 0000}
        assert all(c >= 0 for c in counts.values())
        npt.assert_almost_equal(sum(counts.values()), shots)

    @pytest.mark.parametrize("qubits", [4, 7])
    @pytest.mark.parametrize("shots", [800, 1200, 2000])
    @pytest.mark.parametrize(
        "sampler_creator",
        [
            create_qulacs_density_matrix_ideal_sampler,
            create_qulacs_ideal_density_matrix_general_sampler,
        ],
    )
    def test_ideal_sampler_with_bitflip_noise(
        self,
        qubits: int,
        shots: int,
        sampler_creator: Callable[[NoiseModel], Sampler],
    ) -> None:
        circuit = QuantumCircuit(qubits)
        circuit.add_H_gate(3)
        circuit.add_X_gate(2)
        circuit.add_X_gate(1)
        circuit.add_CNOT_gate(3, 0)

        model = NoiseModel([BitFlipNoise(1.0)])
        sampler = sampler_creator(model)
        counts = sampler(circuit, shots)

        filtered_counts = {}
        for ind, value in counts.items():
            if value > 0:
                filtered_counts[ind] = value

        counts = filtered_counts

        assert set(counts.keys()) == {0b1001, 0b0000}
        # {0000, 0000} HXX_ -> {1110, 0110} BitFlip -> {0000, 1000} CNOT ->
        # {0000, 1001} BitFlip -> {1001, 0000}
        assert all(c >= 0 for c in counts.values())

        probs = np.array(list(counts.values())) / shots
        npt.assert_almost_equal(probs.sum(), 1.0)

        mean_val = np.mean(probs)
        # Uniform superposition exact value
        assert all(np.isclose(mean_val, x) for x in probs)


class TestConcurrentSamplerWithNoiseModel:
    @pytest.mark.parametrize(
        "sampler_creator",
        [
            create_qulacs_density_matrix_concurrent_sampler,
            create_qulacs_density_matrix_general_sampler,
            create_qulacs_stochastic_state_vector_concurrent_sampler,
            create_qulacs_noisesimulator_concurrent_sampler,
            create_qulacs_noisesimulator_general_sampler,
        ],
    )
    def test_sampler_with_empty_noise(
        self,
        sampler_creator: Callable[
            [NoiseModel, Optional["Executor"], int], ConcurrentSampler
        ],
    ) -> None:
        model = NoiseModel()

        circuit1 = circuit()
        circuit2 = circuit()
        circuit2.add_X_gate(3)

        with ThreadPoolExecutor(max_workers=2) as executor:
            sampler = sampler_creator(model, executor, 2)
            results = list(sampler([(circuit1, 1000), (circuit2, 2000)]))

        assert set(results[0]) == {0b1001, 0b1011}
        assert all(c >= 0 for c in results[0].values())
        assert sum(results[0].values()) == 1000

        assert set(results[1]) == {0b0001, 0b0011}
        assert all(c >= 0 for c in results[1].values())
        assert sum(results[1].values()) == 2000

    @pytest.mark.parametrize(
        "sampler_creator",
        [
            create_qulacs_density_matrix_concurrent_sampler,
            create_qulacs_density_matrix_general_sampler,
            create_qulacs_stochastic_state_vector_concurrent_sampler,
            create_qulacs_noisesimulator_concurrent_sampler,
            create_qulacs_noisesimulator_general_sampler,
        ],
    )
    def test_sampler_with_bitflip_noise(
        self,
        sampler_creator: Callable[
            [NoiseModel, Optional["Executor"], int], ConcurrentSampler
        ],
    ) -> None:
        model = NoiseModel([BitFlipNoise(1.0)])

        circuit1 = circuit()
        circuit2 = circuit()
        circuit2.add_CNOT_gate(1, 3)

        with ThreadPoolExecutor(max_workers=2) as executor:
            sampler = sampler_creator(model, executor, 2)
            results = list(sampler([(circuit1, 1000), (circuit2, 2000)]))

        assert set(results[0]) == {0b0110, 0b0100}
        assert all(c >= 0 for c in results[0].values())
        assert sum(results[0].values()) == 1000

        assert set(results[1]) == {0b0100, 0b1110}
        # {0b0110, 0b0100} CNOT -> {0b1110, 0b0100} BitFlip -> {0b0100, 0b1110}
        assert all(c >= 0 for c in results[1].values())
        assert sum(results[1].values()) == 2000
