# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from collections.abc import Collection
from typing import Optional, cast

import networkx as nx

from quri_parts.backend.device import DeviceProperty, GateProperty, QubitProperty
from quri_parts.backend.units import TimeValue
from quri_parts.circuit import gate_names, noise
from quri_parts.circuit.gate_names import GateNameType, NonParametricGateNameType
from quri_parts.circuit.transpile import CircuitTranspiler, GateSetConversionTranspiler


def generate_device_property(
    qubit_count: int,
    native_gates_1q: Collection[str],
    native_gates_2q: Collection[str],
    gate_error_1q: float,
    gate_error_2q: float,
    gate_error_meas: float,
    gate_time_1q: TimeValue,
    gate_time_2q: TimeValue,
    gate_time_meas: TimeValue,
    t1: Optional[TimeValue] = None,
    t2: Optional[TimeValue] = None,
    transpiler: Optional[CircuitTranspiler] = None,
) -> DeviceProperty:
    """Generate DeviceProperty object for a typical NISQ trapped ion device.

    Assumes that the device's qubits are all-to-all connected and that a subset of
    the gates natively supported by QURI Parts can be used as the native gates.

    Args:
        qubit_count: Number of qubits.
        native_gates_1q: Single qubit native gates supported by the device.
        native_gates_2q: Two qubit native gates supported by the device.
        gate_error_1q: Error rate of single qubit gate operations.
        gate_error_2q: Error rate of two qubit gate operations.
        gate_error_meas: Error rate of readout operations.
        gate_time_1q: Latency of single qubit gate operations.
        gate_time_2q: Latency of two qubit gate operations.
        gate_time_meas: Latency of readout operations.
        t1: T1 coherence time.
        t2: T2 coherence time.
        transpiler: CircuitTranspiler to adapt the circuit to the device. If not
            specified, default transpiler is used.
    """
    if t1 is not None or t2 is not None:
        warnings.warn(
            "The t1 t2 error is not yet supported and is not reflected in the "
            "fidelity estimation or noise model."
        )

    gates_1q = set(native_gates_1q)
    gates_2q = set(native_gates_2q)
    native_gates = gates_1q | gates_2q
    meas = native_gates & {gate_names.Measurement}

    qubits = list(range(qubit_count))
    qubit_properties = {q: QubitProperty() for q in qubits}
    gate_properties = []
    gate_properties.extend(
        [
            GateProperty(name, (), gate_error=gate_error_1q, gate_time=gate_time_1q)
            for name in gates_1q
        ]
    )
    gate_properties.extend(
        [
            GateProperty(name, (), gate_error=gate_error_2q, gate_time=gate_time_2q)
            for name in gates_2q
        ]
    )
    gate_properties.extend(
        [
            GateProperty(name, (), gate_error=gate_error_meas, gate_time=gate_time_meas)
            for name in meas
        ]
    )

    transpiler = (
        transpiler
        if transpiler is not None
        else GateSetConversionTranspiler(cast(Collection[GateNameType], native_gates))
    )

    noise_model = noise.NoiseModel(
        [
            noise.DepolarizingNoise(
                error_prob=gate_error_1q,
                target_gates=list(cast(set[NonParametricGateNameType], gates_1q)),
            ),
            noise.DepolarizingNoise(
                error_prob=gate_error_2q,
                target_gates=list(cast(set[NonParametricGateNameType], gates_2q)),
            ),
            noise.MeasurementNoise([noise.BitFlipNoise(error_prob=gate_error_meas)]),
        ]
    )

    return DeviceProperty(
        qubit_count=qubit_count,
        qubits=qubits,
        qubit_graph=nx.complete_graph(qubit_count),
        qubit_properties=qubit_properties,
        native_gates=native_gates,
        gate_properties=gate_properties,
        physical_qubit_count=qubit_count,
        # TODO Calculate backgraound error from t1 and t2
        background_error=None,
        transpiler=transpiler,
        noise_model=noise_model,
    )
