# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Collection, Mapping
from typing import Callable, cast

from quri_parts.circuit import NonParametricQuantumCircuit, QuantumGate

from .device import DeviceProperty
from .units import TimeUnit, TimeValue


def _gate_weighted_depth(
    circuit: NonParametricQuantumCircuit,
    gate_weight: Callable[[QuantumGate], float],
) -> float:
    qubit_depth: dict[int, float] = {}

    for gate in circuit.gates:
        qubits = tuple(gate.control_indices) + tuple(gate.target_indices)
        depth = gate_weight(gate) + max(qubit_depth.get(q, 0.0) for q in qubits)
        for q in qubits:
            qubit_depth[q] = depth

    return max(qubit_depth.values()) if qubit_depth else 0.0


def _gate_kind_weighted_depth(
    circuit: NonParametricQuantumCircuit,
    gate_kind_weight: Mapping[str, float],
    default_weight: float = 0.0,
) -> float:
    return _gate_weighted_depth(
        circuit, lambda gate: gate_kind_weight.get(gate.name, default_weight)
    )


def _estimate_gate_latency(
    circuit: NonParametricQuantumCircuit,
    device: DeviceProperty,
    kinds: Collection[str] = [],
) -> TimeValue:
    latency = 0.0
    for gate in circuit.gates:
        lat = device.gate_property(gate).gate_time
        if lat is None:
            raise ValueError("Contains gate with unknown gate gate_time: {gate}")
        if kinds and gate.name not in kinds:
            continue
        latency += lat.in_ns()
    return TimeValue(value=latency, unit=TimeUnit.NANOSECOND)


def estimate_circuit_latency(
    circuit: NonParametricQuantumCircuit,
    device: DeviceProperty,
) -> TimeValue:
    """Estimates the execution time for processing the given quantum circuit on
    the given device.

    Args:
        circuit: NonParametricQuantumCircuit to be estimated.
        device: DeviceProperty of the device to execute the circuit.

    Returns:
        Estimated latency of the circuit in nano seconds when executing on the given
        device.
    """
    for gate in circuit.gates:
        if device.gate_property(gate).gate_time is None:
            raise ValueError(f"Contains gate with unknown gate_time: {gate}")

    return TimeValue(
        value=_gate_weighted_depth(
            circuit,
            lambda gate: cast(TimeValue, device.gate_property(gate).gate_time).in_ns(),
        ),
        unit=TimeUnit.NANOSECOND,
    )


def _estimate_gate_fidelity(
    circuit: NonParametricQuantumCircuit,
    device: DeviceProperty,
    kinds: Collection[str] = [],
) -> float:
    fidelity = 1.0
    for gate in circuit.gates:
        error = device.gate_property(gate).gate_error
        if error is None:
            raise ValueError(f"Contains gate with unknown gate_error: {gate}")
        if kinds and gate.name not in kinds:
            continue
        fidelity *= 1.0 - error
    return fidelity


def _estimate_background_fidelity(
    circuit: NonParametricQuantumCircuit, device: DeviceProperty
) -> float:
    if device.background_error is None:
        return 1.0
    circuit_latency = estimate_circuit_latency(circuit, device).value
    error, latency = device.background_error

    return cast(
        float,
        (1.0 - error) ** ((circuit_latency * circuit.qubit_count) / latency.in_ns()),
    )


def estimate_circuit_fidelity(
    circuit: NonParametricQuantumCircuit,
    device: DeviceProperty,
    background_error: bool = True,
) -> float:
    """Estimates the fidelity of processing the given quantum circuit on the
    given device.

    Args:
        circuit: NonParametricQuantumCircuit to be estimated.
        device: DeviceProperty of the device to execute the circuit.
        background_error: Specifies whether the background error is factored into the
            fidelity of the circuit. True by default.

    Returns:
        Estimated fidelity of the circuit when executing on the given device.
    """
    background_fidelity = (
        _estimate_background_fidelity(circuit, device) if background_error else 1.0
    )
    gate_fidelity = _estimate_gate_fidelity(circuit, device)
    return gate_fidelity * background_fidelity
