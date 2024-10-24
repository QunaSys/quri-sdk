import math

import numpy as np

from quri_parts.backend.cost_estimator import (
    estimate_circuit_fidelity,
    estimate_circuit_latency,
)
from quri_parts.backend.devices import (
    clifford_t_device,
    nisq_iontrap_device,
    nisq_spcond_lattice,
    star_device,
)
from quri_parts.backend.units import TimeUnit, TimeValue
from quri_parts.circuit import QuantumCircuit, gate_names, gates
from quri_parts.circuit.topology import SquareLattice


def _star_native_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(2)
    circuit.extend(
        [
            gates.H(0),
            gates.CNOT(0, 1),
            gates.RZ(1, np.pi / 7.0),
        ]
    )
    return circuit


def _non_star_native_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(2)
    circuit.extend(
        [
            gates.X(0),
            gates.CZ(0, 1),
            gates.RY(1, np.pi / 7.0),
        ]
    )
    return circuit


def test_star_device() -> None:
    device = star_device.generate_device_property(
        qubit_count=16,
        code_distance=7,
        qec_cycle=TimeValue(value=1.0, unit=TimeUnit.MICROSECOND),
        physical_error_rate=1.0e-4,
    )

    assert device.qubit_count == 16
    assert device.physical_qubit_count == 6272
    assert set(device.native_gates) == {
        gate_names.H,
        gate_names.S,
        gate_names.CNOT,
        gate_names.RZ,
    }
    assert device.background_error is not None
    error, time = device.background_error
    assert math.isclose(2.3302081608722602e-07, error)
    assert math.isclose(1.0e3, time.in_ns())

    circuit = _star_native_circuit()
    assert 0.0 < estimate_circuit_fidelity(circuit, device, background_error=True) < 1.0
    assert 0.0 < estimate_circuit_latency(circuit, device).value

    assert device.transpiler is not None
    assert {
        gate.name for gate in device.transpiler(_non_star_native_circuit()).gates
    } <= set(device.native_gates)

    assert device.noise_model is not None


def test_clifford_t_device() -> None:
    device = clifford_t_device.generate_device_property(
        qubit_count=16,
        code_distance=7,
        qec_cycle=TimeValue(value=1.0, unit=TimeUnit.MICROSECOND),
        delta_sk=2.6e-5,
        mode_block="intermediate",
        physical_error_rate=1.0e-4,
    )
    assert device.qubit_count == 16
    assert device.physical_qubit_count == 6762
    assert set(device.native_gates) == {
        gate_names.H,
        gate_names.S,
        gate_names.T,
        gate_names.CNOT,
    }
    assert device.background_error is not None
    error, time = device.background_error
    assert math.isclose(2.512255650177764e-07, error)
    assert math.isclose(1.0e3, time.in_ns())

    circuit = _star_native_circuit()
    assert 0.0 < estimate_circuit_fidelity(circuit, device, background_error=True) < 1.0
    assert 0.0 < estimate_circuit_latency(circuit, device).value


def test_nisq_iontrap_device() -> None:
    native_gates: set[gate_names.GateNameType] = {
        gate_names.RX,
        gate_names.RY,
        gate_names.RZ,
        gate_names.CZ,
    }

    device = nisq_iontrap_device.generate_device_property(
        qubit_count=16,
        native_gates=native_gates,
        gate_error_1q=1.0e-5,
        gate_error_2q=1.0e-3,
        gate_error_meas=1.0e-3,
        gate_time_1q=TimeValue(value=100.0, unit=TimeUnit.MICROSECOND),
        gate_time_2q=TimeValue(value=500.0, unit=TimeUnit.MICROSECOND),
        gate_time_meas=TimeValue(value=1.0, unit=TimeUnit.MILLISECOND),  # TODO check
        t1=TimeValue(value=10.0, unit=TimeUnit.SECOND),
        t2=TimeValue(value=1.0, unit=TimeUnit.SECOND),
    )
    assert device.qubit_count == 16
    assert set(device.native_gates) == native_gates

    assert device.transpiler is not None
    circuit = device.transpiler(_star_native_circuit())
    assert (
        0.0 < estimate_circuit_fidelity(circuit, device, background_error=False) < 1.0
    )
    assert 0.0 < estimate_circuit_latency(circuit, device).value


def test_nisq_spcond_device() -> None:
    native_gates: set[gate_names.GateNameType] = {
        gate_names.RX,
        gate_names.RY,
        gate_names.RZ,
        gate_names.CZ,
    }

    device = nisq_spcond_lattice.generate_device_property(
        lattice=SquareLattice(4, 4),
        native_gates=native_gates,
        gate_error_1q=1.0e-4,
        gate_error_2q=1.0e-2,
        gate_error_meas=1.0e-2,
        gate_time_1q=TimeValue(value=500.0, unit=TimeUnit.NANOSECOND),
        gate_time_2q=TimeValue(value=500.0, unit=TimeUnit.NANOSECOND),
        gate_time_meas=TimeValue(value=500.0, unit=TimeUnit.NANOSECOND),  # TODO check
        t1=TimeValue(value=200.0, unit=TimeUnit.MICROSECOND),
        t2=TimeValue(value=100.0, unit=TimeUnit.MICROSECOND),
    )
    assert device.qubit_count == 16
    assert set(device.native_gates) == native_gates

    assert device.transpiler is not None
    circuit = device.transpiler(_star_native_circuit())
    assert (
        0.0 < estimate_circuit_fidelity(circuit, device, background_error=False) < 1.0
    )
    assert 0.0 < estimate_circuit_latency(circuit, device).value
