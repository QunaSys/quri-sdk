# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from concurrent.futures import ThreadPoolExecutor

import pytest

from quri_parts.circuit import QuantumCircuit
from quri_parts.stim.sampler import (
    create_stim_clifford_concurrent_sampler,
    create_stim_clifford_sampler,
)


def circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(4)
    circuit.add_X_gate(0)
    circuit.add_H_gate(1)
    circuit.add_Z_gate(2)
    circuit.add_Y_gate(3)
    return circuit


def test_sampler_non_clifford() -> None:
    shots = 1000
    non_clifford_circuit = circuit()
    non_clifford_circuit.add_RX_gate(0, 0.1)

    sampler = create_stim_clifford_sampler()

    with pytest.raises(ValueError):
        sampler(non_clifford_circuit, shots)


@pytest.mark.parametrize("qubits", [4, 12])
@pytest.mark.parametrize("shots", [800, 1200, 2**12 + 100])
def test_sampler(qubits: int, shots: int) -> None:
    circuit = QuantumCircuit(qubits)
    for i in range(qubits):
        circuit.add_H_gate(i)

    sampler = create_stim_clifford_sampler()
    counts = sampler(circuit, shots)

    assert all(isinstance(bits, int) for bits in list(counts.keys()))
    assert set(counts.keys()).issubset(range(2**qubits))
    assert all(c >= 0 for c in counts.values())
    assert sum(counts.values()) == shots


def test_concurrent_sampler() -> None:
    circuit1 = circuit()
    circuit2 = circuit()
    circuit2.add_X_gate(3)

    with ThreadPoolExecutor(max_workers=2) as executor:
        sampler = create_stim_clifford_concurrent_sampler(executor, 2)
        results = list(sampler([(circuit1, 1000), (circuit2, 2000)]))

    assert set(results[0]) == {0b1001, 0b1011}
    assert all(c >= 0 for c in results[0].values())
    assert sum(results[0].values()) == 1000

    assert set(results[1]) == {0b0001, 0b0011}
    assert all(c >= 0 for c in results[1].values())
    assert sum(results[1].values()) == 2000
