# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import Any

import pytest

from quri_parts.circuit import QuantumCircuit
from quri_parts.itensor.sampler import (
    create_itensor_mps_concurrent_sampler,
    create_itensor_mps_sampler,
)


def circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(4)
    circuit.add_X_gate(0)
    circuit.add_H_gate(1)
    circuit.add_Z_gate(2)
    circuit.add_Y_gate(3)
    return circuit


class TestITensorMPSSampler:
    @pytest.mark.parametrize("qubits", [4, 12])
    @pytest.mark.parametrize("shots", [800, 1200, 2**12 + 100])
    @pytest.mark.parametrize(
        "sampler_kwargs",
        [
            {},
            {"maxdim": 100, "cutoff": 0.1},
            {"apply_dag": False, "move_sites_back": True},
        ],
    )
    def test_sampler(self, qubits: int, shots: int, sampler_kwargs: Any) -> None:
        circuit = QuantumCircuit(qubits)
        for i in range(qubits):
            circuit.add_H_gate(i)

        sampler = create_itensor_mps_sampler(**sampler_kwargs)
        counts = sampler(circuit, shots)

        assert set(counts.keys()).issubset(range(2**qubits))
        assert all(c >= 0 for c in counts.values())
        assert sum(counts.values()) == shots

        # sample on empty circuit
        circuit = QuantumCircuit(qubits)
        counts = sampler(circuit, shots)

        assert len(counts) == 1
        assert next(iter(counts)) == 0
        assert counts[0] == shots


class TestITensorMPSConcurrentSampler:
    @pytest.mark.skip(reason="This test is too slow.")
    @pytest.mark.parametrize(
        "sampler_kwargs",
        [
            {},
            {"maxdim": 100, "cutoff": 0.1},
            {"apply_dag": False, "move_sites_back": True},
        ],
    )
    def test_concurrent_sampler(self, sampler_kwargs: Any) -> None:
        circuit1 = circuit()
        circuit2 = circuit()
        circuit2.add_X_gate(3)
        circuit3 = QuantumCircuit(4)
        with ProcessPoolExecutor(
            max_workers=3, mp_context=get_context("spawn")
        ) as executor:
            sampler = create_itensor_mps_concurrent_sampler(
                executor, 3, **sampler_kwargs
            )
            results = list(
                sampler([(circuit1, 1000), (circuit2, 2000), (circuit3, 4000)])
            )
        assert set(results[0]) == {0b1001, 0b1011}
        assert all(c >= 0 for c in results[0].values())
        assert sum(results[0].values()) == 1000
        assert set(results[1]) == {0b0001, 0b0011}
        assert all(c >= 0 for c in results[1].values())
        assert sum(results[1].values()) == 2000
        assert set(results[2]) == {0b0000}
        assert all(c >= 0 for c in results[2].values())
        assert sum(results[2].values()) == 4000
