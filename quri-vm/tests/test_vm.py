# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      https://mit-license.org/
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math import pi

# from quri_vm.backend.devices import (
from quri_parts.backend.devices import (
    clifford_t_device,
    nisq_iontrap_device,
    nisq_spcond_lattice,
    star_device,
)
from quri_parts.backend.units import TimeUnit, TimeValue
from quri_parts.circuit import (
    NonParametricQuantumCircuit,
    QuantumCircuit,
    gate_names,
    gates,
)
from quri_parts.circuit.topology import SquareLattice
from quri_parts.core.estimator import Estimate
from quri_parts.core.operator import pauli_label
from quri_parts.core.sampling import MeasurementCounts
from quri_parts.core.state import quantum_state

from quri_vm import VM, AnalyzeResult, VMBackend
from quri_vm.vm_backend import _DevicePropertyBackend


def _my_algorithm(
    vm: VM, shots: int
) -> tuple[MeasurementCounts, Estimate[complex], AnalyzeResult]:
    circuit = QuantumCircuit(2)
    circuit.add_H_gate(0)
    circuit.add_RX_gate(1, pi / 3)
    circuit.add_CNOT_gate(0, 1)

    op = pauli_label("Z0 Z1")
    state = quantum_state(2, circuit=circuit)

    samples = vm.sample(circuit, shots=shots)
    estimate = vm.estimate(op, state)
    analysis = vm.analyze(circuit)

    return samples, estimate, analysis


class TestVMInterface:
    def test_ideal_sample_estimate_analysis(self) -> None:
        abstract_vm = VM()

        circuit = QuantumCircuit(2)
        circuit.add_H_gate(0)
        circuit.add_RX_gate(1, pi / 3)
        circuit.add_CNOT_gate(0, 1)

        op = pauli_label("Z0 Z1")
        state = quantum_state(2, circuit=circuit)

        ideal_samples = abstract_vm.sample(circuit, 100)
        print(ideal_samples)
        assert sum(ideal_samples.values()) == 100

        ideal_estimate = abstract_vm.estimate(op, state)
        assert ideal_estimate.error == 0.0

        ideal_analysis = abstract_vm.analyze(circuit)
        assert ideal_analysis.qubit_count == 2
        assert ideal_analysis.gate_count == 3
        assert ideal_analysis.depth == 2

    def test_ideal_star_architecture(self) -> None:
        ideal_star_vm = VM.from_device_prop(
            star_device.generate_device_property(
                qubit_count=16,
                code_distance=7,
                qec_cycle=TimeValue(value=1.0, unit=TimeUnit.MICROSECOND),
            )
        )

        shots = 100
        samples, estimate, analysis = _my_algorithm(ideal_star_vm, shots=shots)
        assert sum(samples.values()) == shots
        assert estimate.error == 0.0
        assert analysis.latency is not None and 0.0 < analysis.latency.value
        assert analysis.fidelity is not None and analysis.fidelity == 1.0

    def test_noisy_star_architecture(self) -> None:
        noisy_star_vm = VM.from_device_prop(
            star_device.generate_device_property(
                qubit_count=16,
                code_distance=7,
                qec_cycle=TimeValue(value=1.0, unit=TimeUnit.MICROSECOND),
                physical_error_rate=1.0e-4,
            )
        )

        shots = 100
        samples, estimate, analysis = _my_algorithm(noisy_star_vm, shots=shots)
        assert sum(samples.values()) == shots
        assert estimate.error == 0.0
        assert analysis.latency is not None and 0.0 < analysis.latency.value
        assert analysis.fidelity is not None and 0.0 < analysis.fidelity < 1.0


def _circuit() -> NonParametricQuantumCircuit:
    circuit = QuantumCircuit(2)
    circuit.extend([gates.H(0), gates.X(1), gates.CNOT(0, 1), gates.RX(1, pi / 7.0)])
    return circuit


def _test_vm_with_backend(
    backend: VMBackend, circuit: NonParametricQuantumCircuit
) -> None:
    vm = VM(vm_backend=backend)
    circuit = vm.transpile(circuit)

    shots = 1000
    assert shots == sum(vm.sample(circuit, shots=shots).values())

    analysis = vm.analyze(circuit)
    tr_circuit = vm.transpile(circuit)
    assert analysis.qubit_count == tr_circuit.qubit_count
    assert analysis.gate_count == len(tr_circuit.gates)
    assert analysis.depth == tr_circuit.depth
    assert analysis.latency is not None and 0.0 < analysis.latency.value
    assert analysis.fidelity is not None and 0.0 < analysis.fidelity < 1.0


def _test_vm_with_device_property_backend(
    backend: _DevicePropertyBackend, circuit: NonParametricQuantumCircuit
) -> None:
    vm = VM(vm_backend=backend)
    circuit = vm.transpile(circuit)

    shots = 1000
    assert shots == sum(vm.sample(circuit, shots=shots).values())

    analysis = vm.analyze(circuit)
    if backend._device.analyze_transpiler is not None:
        tr_circuit = backend._device.analyze_transpiler(circuit)
    else:
        tr_circuit = vm.transpile(circuit)
    assert analysis.qubit_count == tr_circuit.qubit_count
    assert analysis.gate_count == len(tr_circuit.gates)
    assert analysis.depth == tr_circuit.depth
    assert analysis.latency is not None and 0.0 < analysis.latency.value
    assert analysis.fidelity is not None and 0.0 < analysis.fidelity < 1.0


class TestVMBackend:
    def test_backend_independent_vm(self) -> None:
        circuit = _circuit()

        vm = VM()
        shots = 1000
        assert shots == sum(vm.sample(circuit, shots=shots).values())

        analysis = vm.analyze(circuit)
        assert analysis.qubit_count == circuit.qubit_count
        assert analysis.gate_count == len(circuit.gates)
        assert analysis.depth == circuit.depth
        assert analysis.latency is None
        assert analysis.fidelity == 1.0

    def test_qulacs_star_backend_vm(self) -> None:
        circuit = _circuit()

        backend = _DevicePropertyBackend(
            star_device.generate_device_property(
                qubit_count=16,
                code_distance=7,
                qec_cycle=TimeValue(1.0e-6, TimeUnit.SECOND),
                physical_error_rate=1.0e-4,
            )
        )
        _test_vm_with_backend(backend, circuit)

    def test_qulacs_clifford_t_backend_vm(self) -> None:
        circuit = _circuit()

        backend = _DevicePropertyBackend(
            clifford_t_device.generate_device_property(
                qubit_count=16,
                code_distance=7,
                qec_cycle=TimeValue(1.0e-6, TimeUnit.SECOND),
                delta_sk=2.6e-5,
                mode_block="intermediate",
                physical_error_rate=1.0e-4,
            )
        )
        _test_vm_with_device_property_backend(backend, circuit)

    def test_qulacs_nisq_iontrap_backend_vm(self) -> None:
        circuit = _circuit()

        backend = _DevicePropertyBackend(
            nisq_iontrap_device.generate_device_property(
                qubit_count=16,
                native_gates={
                    gate_names.RX,
                    gate_names.RY,
                    gate_names.RZ,
                    gate_names.CZ,
                },
                gate_error_1q=1.0e-5,
                gate_error_2q=1.0e-3,
                gate_error_meas=1.0e-3,
                gate_time_1q=TimeValue(value=100.0, unit=TimeUnit.MICROSECOND),
                gate_time_2q=TimeValue(value=500.0, unit=TimeUnit.MICROSECOND),
                gate_time_meas=TimeValue(
                    value=1.0, unit=TimeUnit.MILLISECOND
                ),  # TODO check
                t1=TimeValue(value=10.0, unit=TimeUnit.SECOND),
                t2=TimeValue(value=1.0, unit=TimeUnit.SECOND),
            )
        )
        _test_vm_with_backend(backend, circuit)

    def test_qulacs_nisq_spcond_backend_vm(self) -> None:
        circuit = _circuit()

        backend = _DevicePropertyBackend(
            nisq_spcond_lattice.generate_device_property(
                lattice=SquareLattice(4, 4),
                native_gates={
                    gate_names.RX,
                    gate_names.RY,
                    gate_names.RZ,
                    gate_names.CZ,
                },
                gate_error_1q=1.0e-4,
                gate_error_2q=1.0e-2,
                gate_error_meas=1.0e-2,
                gate_time_1q=TimeValue(value=500.0, unit=TimeUnit.NANOSECOND),
                gate_time_2q=TimeValue(value=500.0, unit=TimeUnit.NANOSECOND),
                gate_time_meas=TimeValue(
                    value=500.0, unit=TimeUnit.NANOSECOND
                ),  # TODO check
                t1=TimeValue(value=200.0, unit=TimeUnit.MICROSECOND),
                t2=TimeValue(value=100.0, unit=TimeUnit.MICROSECOND),
            )
        )
        _test_vm_with_backend(backend, circuit)
