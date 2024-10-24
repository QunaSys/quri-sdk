import math

import networkx as nx

from quri_parts.backend.cost_estimator import (
    estimate_circuit_fidelity,
    estimate_circuit_latency,
)
from quri_parts.backend.device import DeviceProperty, GateProperty, QubitProperty
from quri_parts.backend.units import TimeUnit, TimeValue
from quri_parts.circuit import QuantumCircuit, gate_names, gates
from quri_parts.circuit.gate_names import NonParametricGateNameType


def _ideal_device() -> DeviceProperty:
    qubit_count = 3
    qubits = list(range(qubit_count))
    qubit_properties = {q: QubitProperty() for q in qubits}
    native_gates: list[NonParametricGateNameType] = [
        gate_names.H,
        gate_names.S,
        gate_names.T,
        gate_names.CNOT,
        gate_names.TOFFOLI,
    ]
    gate_properties = [
        GateProperty(
            name,
            (),
            gate_error=0.0,
            gate_time=TimeValue(value=1.0, unit=TimeUnit.NANOSECOND),
        )
        for name in native_gates
    ]
    return DeviceProperty(
        qubit_count=qubit_count,
        qubits=qubits,
        qubit_graph=nx.complete_graph(qubit_count),
        qubit_properties=qubit_properties,
        native_gates=native_gates,
        gate_properties=gate_properties,
    )


def _imperfect_device() -> DeviceProperty:
    qubit_count = 3
    qubits = list(range(qubit_count))
    qubit_properties = {q: QubitProperty() for q in qubits}
    native_gates: list[NonParametricGateNameType] = [
        gate_names.H,
        gate_names.S,
        gate_names.T,
        gate_names.CNOT,
        gate_names.TOFFOLI,
    ]
    gate_properties = [
        GateProperty(
            gate_names.H,
            (),
            gate_error=1.0e-3,
            gate_time=TimeValue(value=10.0, unit=TimeUnit.MICROSECOND),
        ),
        GateProperty(
            gate_names.S,
            (),
            gate_error=2.0e-3,
            gate_time=TimeValue(value=20.0, unit=TimeUnit.MICROSECOND),
        ),
        GateProperty(
            gate_names.T,
            (),
            gate_error=3.0e-3,
            gate_time=TimeValue(value=30.0, unit=TimeUnit.MICROSECOND),
        ),
        GateProperty(
            gate_names.CNOT,
            (),
            gate_error=4.0e-3,
            gate_time=TimeValue(value=40.0, unit=TimeUnit.MICROSECOND),
        ),
        GateProperty(
            gate_names.TOFFOLI,
            (),
            gate_error=5.0e-3,
            gate_time=TimeValue(value=50.0, unit=TimeUnit.MICROSECOND),
        ),
    ]
    return DeviceProperty(
        qubit_count=qubit_count,
        qubits=qubits,
        qubit_graph=nx.complete_graph(qubit_count),
        qubit_properties=qubit_properties,
        native_gates=native_gates,
        gate_properties=gate_properties,
    )


def _circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(3)
    circuit.extend(
        [
            gates.H(0),
            gates.CNOT(0, 1),
            gates.S(0),
            gates.T(1),
            gates.H(2),
            gates.TOFFOLI(0, 1, 2),
        ]
    )
    return circuit


def test_estimate_cost_for_empty() -> None:
    circuit = QuantumCircuit(3)

    for device in [_ideal_device(), _imperfect_device()]:
        assert 0.0 == estimate_circuit_latency(circuit, device).value
        assert 1.0 == estimate_circuit_fidelity(circuit, device)


def test_estimate_cost_ideal() -> None:
    circuit = _circuit()
    device = _ideal_device()

    assert math.isclose(4.0, estimate_circuit_latency(circuit, device).value)
    assert math.isclose(1.0, estimate_circuit_fidelity(circuit, device))


def test_estimate_cost_real() -> None:
    circuit = _circuit()
    device = _imperfect_device()

    assert math.isclose(130.0e3, estimate_circuit_latency(circuit, device).value)
    assert math.isclose(0.9840996904986060, estimate_circuit_fidelity(circuit, device))


def test_estimate_cost_background() -> None:
    circuit = _circuit()
    device = _ideal_device()
    device.background_error = (1.0e-7, TimeValue(value=1.0, unit=TimeUnit.NANOSECOND))
    assert math.isclose(
        1.0,
        estimate_circuit_fidelity(circuit, device, background_error=False),
    )
    assert math.isclose(
        0.9999988000006607,
        estimate_circuit_fidelity(circuit, device, background_error=True),
    )
