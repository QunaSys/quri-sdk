from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

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
    def test_sampler(self, qubits: int, shots: int) -> None:
        circuit = QuantumCircuit(qubits)
        for i in range(qubits):
            circuit.add_H_gate(i)

        sampler = create_itensor_mps_sampler()
        counts = sampler(circuit, shots)

        assert set(counts.keys()).issubset(range(2**qubits))
        assert all(c >= 0 for c in counts.values())
        assert sum(counts.values()) == shots


class TestITensorMPSConcurrentSampler:
    @pytest.mark.skip(reason="This test is too slow.")
    def test_concurrent_sampler(self) -> None:
        circuit1 = circuit()
        circuit2 = circuit()
        circuit2.add_X_gate(3)
        with ProcessPoolExecutor(
            max_workers=2, mp_context=get_context("spawn")
        ) as executor:
            sampler = create_itensor_mps_concurrent_sampler(executor, 2)
            results = list(sampler([(circuit1, 1000), (circuit2, 2000)]))
        assert set(results[0]) == {0b1001, 0b1011}
        assert all(c >= 0 for c in results[0].values())
        assert sum(results[0].values()) == 1000
        assert set(results[1]) == {0b0001, 0b0011}
        assert all(c >= 0 for c in results[1].values())
        assert sum(results[1].values()) == 2000
