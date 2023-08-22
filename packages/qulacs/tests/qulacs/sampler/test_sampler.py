# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional

import numpy as np
import numpy.testing as npt
import pytest

from quri_parts.circuit import QuantumCircuit
from quri_parts.circuit.noise import BitFlipNoise, NoiseModel
from quri_parts.core.sampling import ConcurrentSampler, Sampler
from quri_parts.qulacs.circuit.compiled_circuit import compile_circuit
from quri_parts.qulacs.sampler import (
    create_qulacs_density_matrix_concurrent_sampler,
    create_qulacs_density_matrix_ideal_sampler,
    create_qulacs_density_matrix_sampler,
    create_qulacs_noisesimulator_concurrent_sampler,
    create_qulacs_noisesimulator_sampler,
    create_qulacs_stochastic_state_vector_concurrent_sampler,
    create_qulacs_stochastic_state_vector_sampler,
    create_qulacs_vector_concurrent_sampler,
    create_qulacs_vector_ideal_sampler,
    create_qulacs_vector_sampler,
)

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


class TestSamplerWithNoiseModel:
    # (2**4)**2/10 = 25.6, (2**7)**2)/10 = 1638.4
    @pytest.mark.parametrize("qubits", [4, 7])
    @pytest.mark.parametrize("shots", [800, 1200, 2000])
    @pytest.mark.parametrize(
        "sampler_creator",
        [
            create_qulacs_density_matrix_sampler,
            create_qulacs_stochastic_state_vector_sampler,
            create_qulacs_noisesimulator_sampler,
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
            create_qulacs_stochastic_state_vector_sampler,
            create_qulacs_noisesimulator_sampler,
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
            create_qulacs_stochastic_state_vector_concurrent_sampler,
            create_qulacs_noisesimulator_concurrent_sampler,
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
            create_qulacs_stochastic_state_vector_concurrent_sampler,
            create_qulacs_noisesimulator_concurrent_sampler,
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
